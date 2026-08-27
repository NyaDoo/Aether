use std::time::Duration;

use crate::error::RedisResultExt;
use crate::redis::{
    cmd, run_lane_with_timeout, script, RedisCmd, RedisConnectionLane, RedisConnectionRouter,
    RedisKeyspace, RedisLaneDiagnostics,
};
use crate::{
    DataLayerError, RateLimitCheck, RateLimitInput, RateLimitScope, RuntimeSemaphoreError,
};

const RATE_LIMIT_CHECK_AND_CONSUME_SCRIPT: &str = r#"
local user_key = KEYS[1]
local key_key = KEYS[2]
local user_limit = tonumber(ARGV[1])
local key_limit = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local user_count = 0
if user_limit > 0 then
    user_count = tonumber(redis.call('GET', user_key) or '0')
    if user_count >= user_limit then
        return {0, 1, user_limit, 0}
    end
end

local key_count = 0
if key_limit > 0 then
    key_count = tonumber(redis.call('GET', key_key) or '0')
    if key_count >= key_limit then
        return {0, 2, key_limit, 0}
    end
end

local remaining = -1
if user_limit > 0 then
    user_count = redis.call('INCR', user_key)
    redis.call('EXPIRE', user_key, ttl)
    remaining = user_limit - user_count
end

if key_limit > 0 then
    key_count = redis.call('INCR', key_key)
    redis.call('EXPIRE', key_key, ttl)
    local key_remaining = key_limit - key_count
    if remaining == -1 or key_remaining < remaining then
        remaining = key_remaining
    end
end

return {1, 0, 0, remaining}
"#;

// Realtime dashboard buckets are updated in one Redis-side transaction.  The
// request/token fields intentionally share a hash so a snapshot never observes
// a half-applied delta across the two counters.
const REALTIME_BUCKET_ADD_SCRIPT: &str = r#"
local requests = tonumber(redis.call('HGET', KEYS[1], 'requests') or '0')
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens') or '0')
requests = math.max(0, requests + tonumber(ARGV[1]))
tokens = math.max(0, tokens + tonumber(ARGV[2]))
redis.call('HSET', KEYS[1], 'requests', requests, 'tokens', tokens)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return {requests, tokens}
"#;

const REALTIME_BUCKET_SUM_SCRIPT: &str = r#"
local requests = 0
local tokens = 0
for _, key in ipairs(KEYS) do
    requests = requests + tonumber(redis.call('HGET', key, 'requests') or '0')
    tokens = tokens + tonumber(redis.call('HGET', key, 'tokens') or '0')
end
return {requests, tokens}
"#;

// Exact-timestamp events use one sorted-set index and one hash payload store.
// The collection key is encoded into a Redis hash tag by
// `realtime_event_keys`, so both keys are guaranteed to land in the same
// cluster slot and each lifecycle operation can remain a single Lua script.
const REALTIME_EVENT_ADD_SCRIPT: &str = r#"
local index = KEYS[1]
local payloads = KEYS[2]
local event_id = ARGV[1]
local timestamp_ms = tonumber(ARGV[2])
local request_delta = tonumber(ARGV[3])
local token_delta = tonumber(ARGV[4])
local ttl_ms = tonumber(ARGV[5])

-- Keep the collection bounded even when traffic is continuous and a key's
-- TTL is refreshed by every new event.  The lower retention boundary is
-- exclusive: an event exactly at `timestamp - ttl` is evicted.
local cutoff = timestamp_ms - ttl_ms
local stale = redis.call('ZRANGEBYSCORE', index, '-inf', '(' .. tostring(cutoff))
for _, stale_id in ipairs(stale) do
    redis.call('ZREM', index, stale_id)
    redis.call('HDEL', payloads, stale_id)
end

-- Treat a pair that was only partially retained (for example after an
-- operator-side cleanup) as expired and repair it rather than permanently
-- suppressing a lifecycle retry.
local in_index = redis.call('ZSCORE', index, event_id)
if redis.call('HEXISTS', payloads, event_id) == 1 and in_index then
    return 0
end
redis.call('ZREM', index, event_id)
redis.call('HDEL', payloads, event_id)

redis.call('HSET', payloads, event_id,
    tostring(request_delta) .. '|' .. tostring(token_delta))
redis.call('ZADD', index, timestamp_ms, event_id)
redis.call('PEXPIRE', index, ttl_ms)
redis.call('PEXPIRE', payloads, ttl_ms)
return 1
"#;

const REALTIME_EVENT_REMOVE_SCRIPT: &str = r#"
local index = KEYS[1]
local payloads = KEYS[2]
local event_id = ARGV[1]
local removed = redis.call('HDEL', payloads, event_id)
-- Always remove the index member as well.  This also repairs a dangling
-- sorted-set member left by an interrupted/manual write.
redis.call('ZREM', index, event_id)
if redis.call('ZCARD', index) == 0 then
    redis.call('DEL', index)
    redis.call('DEL', payloads)
end
return removed
"#;

const REALTIME_EVENTS_SUM_SCRIPT: &str = r#"
local index = KEYS[1]
local payloads = KEYS[2]
local start_ms = ARGV[1]
local end_ms = ARGV[2]
local requests = 0
local tokens = 0

-- Redis score syntax `(start` makes the lower bound exclusive while the
-- plain end score keeps the upper bound inclusive: `(start_ms, end_ms]`.
local ids = redis.call('ZRANGEBYSCORE', index, '(' .. start_ms, end_ms)
for _, event_id in ipairs(ids) do
    local encoded = redis.call('HGET', payloads, event_id)
    if encoded then
        local request_delta, token_delta = string.match(encoded, '^([^|]+)|([^|]+)$')
        requests = requests + (tonumber(request_delta) or 0)
        tokens = tokens + (tonumber(token_delta) or 0)
    else
        -- Keep the two-part collection self-healing if an old/manual write
        -- left an index member without a payload.
        redis.call('ZREM', index, event_id)
    end
end
return {math.max(0, requests), math.max(0, tokens)}
"#;

// A token ledger reconciles incremental stream observations with the
// cumulative usage snapshot emitted by a terminal lifecycle event.  Both
// operations run in Redis Lua so multiple gateway instances observe one
// linearizable state.  `terminal_claimed=1` fences late stream frames after a
// terminal claim; a zero terminal total intentionally leaves the ledger open
// for a later enriched lifecycle event.
const REALTIME_TOKEN_STREAM_ADD_SCRIPT: &str = r#"
local ledger = KEYS[1]
local delta = tonumber(ARGV[1]) or 0
local ttl_ms = tonumber(ARGV[2])
if delta <= 0 then
    return 0
end
if redis.call('HGET', ledger, 'terminal_claimed') == '1' then
    return 0
end
local streamed = tonumber(redis.call('HGET', ledger, 'stream_tokens') or '0')
streamed = streamed + delta
redis.call('HSET', ledger, 'stream_tokens', streamed, 'terminal_claimed', '0')
redis.call('PEXPIRE', ledger, ttl_ms)
return delta
"#;

// Idempotent variant used by transport observers that carry a sequence/event
// identity. The marker field and cumulative stream total are updated in the
// same Lua invocation, avoiding a race with a concurrent terminal claim.
const REALTIME_TOKEN_STREAM_ADD_ONCE_SCRIPT: &str = r#"
local ledger = KEYS[1]
local event_id = ARGV[1]
local delta = tonumber(ARGV[2]) or 0
local ttl_ms = tonumber(ARGV[3])
if delta <= 0 then
    return 0
end
local marker = 'stream_event:' .. event_id
if redis.call('HEXISTS', ledger, marker) == 1 then
    return 0
end
if redis.call('HGET', ledger, 'terminal_claimed') == '1' then
    return 0
end
local streamed = tonumber(redis.call('HGET', ledger, 'stream_tokens') or '0')
streamed = streamed + delta
redis.call('HSET', ledger,
    'stream_tokens', streamed,
    'terminal_claimed', '0',
    marker, '1')
redis.call('PEXPIRE', ledger, ttl_ms)
return delta
"#;

const REALTIME_TOKEN_TERMINAL_CLAIM_SCRIPT: &str = r#"
local ledger = KEYS[1]
local total = tonumber(ARGV[1]) or 0
local ttl_ms = tonumber(ARGV[2])
if total <= 0 then
    return 0
end
if redis.call('HGET', ledger, 'terminal_claimed') == '1' then
    return 0
end
local streamed = tonumber(redis.call('HGET', ledger, 'stream_tokens') or '0')
local remainder = total - streamed
if remainder < 0 then
    remainder = 0
end
redis.call('HSET', ledger,
    'stream_tokens', streamed,
    'terminal_claimed', '1',
    'terminal_total', total)
redis.call('PEXPIRE', ledger, ttl_ms)
return remainder
"#;

// Two-phase terminal reconciliation.  The ledger is fenced immediately, but
// the exact event identity/remainder stay pending until the caller commits
// after the event-store write.  This closes the failure window between a
// successful Redis claim and a failed cross-slot event write.
const REALTIME_TOKEN_TERMINAL_PREPARE_SCRIPT: &str = r#"
local ledger = KEYS[1]
local event_id = ARGV[1]
local total = tonumber(ARGV[2]) or 0
local ttl_ms = tonumber(ARGV[3])
if total <= 0 then
    return 0
end
if redis.call('HGET', ledger, 'terminal_claimed') == '1' then
    local pending_event = redis.call('HGET', ledger, 'terminal_event_id')
    if pending_event == event_id and redis.call('HGET', ledger, 'terminal_committed') ~= '1' then
        local pending = tonumber(redis.call('HGET', ledger, 'terminal_remainder') or '0')
        redis.call('PEXPIRE', ledger, ttl_ms)
        return pending
    end
    return 0
end
local streamed = tonumber(redis.call('HGET', ledger, 'stream_tokens') or '0')
local remainder = total - streamed
if remainder < 0 then
    remainder = 0
end
redis.call('HSET', ledger,
    'stream_tokens', streamed,
    'terminal_claimed', '1',
    'terminal_total', total,
    'terminal_event_id', event_id,
    'terminal_remainder', remainder,
    'terminal_committed', (remainder == 0 and '1' or '0'))
redis.call('PEXPIRE', ledger, ttl_ms)
return remainder
"#;

const REALTIME_TOKEN_TERMINAL_COMMIT_SCRIPT: &str = r#"
local ledger = KEYS[1]
local event_id = ARGV[1]
if redis.call('HGET', ledger, 'terminal_event_id') ~= event_id then
    return 0
end
if redis.call('HGET', ledger, 'terminal_claimed') ~= '1' then
    return 0
end
redis.call('HSET', ledger, 'terminal_committed', '1', 'terminal_remainder', '0')
return 1
"#;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RedisRuntimeDiagnostics {
    pub connected_clients: Option<u64>,
    pub blocked_clients: Option<u64>,
    pub total_connections_received: Option<u64>,
    pub rejected_connections: Option<u64>,
    pub total_commands_processed: Option<u64>,
    pub instantaneous_ops_per_sec: Option<u64>,
    pub total_error_replies: Option<u64>,
    pub expired_keys: Option<u64>,
    pub evicted_keys: Option<u64>,
    pub keyspace_hits: Option<u64>,
    pub keyspace_misses: Option<u64>,
    pub used_memory_bytes: Option<u64>,
    pub maxmemory_bytes: Option<u64>,
    pub memory_fragmentation_ratio_basis_points: Option<u64>,
    pub lanes: Vec<RedisLaneDiagnostics>,
}

#[derive(Debug, Clone)]
pub(crate) struct RedisRuntimeRunner {
    connections: RedisConnectionRouter,
    keyspace: RedisKeyspace,
    command_timeout_ms: Option<u64>,
}

impl RedisRuntimeRunner {
    pub(crate) fn new(
        connections: RedisConnectionRouter,
        keyspace: RedisKeyspace,
        command_timeout_ms: Option<u64>,
    ) -> Self {
        Self {
            connections,
            keyspace,
            command_timeout_ms,
        }
    }

    pub(crate) async fn ping(&self) -> Result<(), DataLayerError> {
        let pong = self
            .query_string(RedisConnectionLane::Fast, "runtime redis ping", cmd("PING"))
            .await?;
        if pong.eq_ignore_ascii_case("PONG") {
            Ok(())
        } else {
            Err(DataLayerError::UnexpectedValue(format!(
                "unexpected runtime redis ping response {pong}"
            )))
        }
    }

    pub(crate) async fn diagnostics(&self) -> Result<RedisRuntimeDiagnostics, DataLayerError> {
        let info = self
            .query_string(
                RedisConnectionLane::Admin,
                "runtime redis diagnostics",
                cmd("INFO"),
            )
            .await?;
        Ok(parse_diagnostics(
            &info,
            self.connections.lane_diagnostics(),
        ))
    }

    pub(crate) async fn kv_set_plain(
        &self,
        key: &str,
        value: String,
    ) -> Result<(), DataLayerError> {
        let namespaced_key = self.keyspace.key(key);
        let mut command = cmd("SET");
        command.arg(namespaced_key).arg(value);
        self.query_string(RedisConnectionLane::Fast, "runtime kv set", command)
            .await?;
        Ok(())
    }

    pub(crate) async fn kv_set_with_ttl(
        &self,
        key: &str,
        value: String,
        ttl: Duration,
    ) -> Result<(), DataLayerError> {
        let namespaced_key = self.keyspace.key(key);
        let mut command = cmd("PSETEX");
        command
            .arg(namespaced_key)
            .arg(u64::try_from(ttl.as_millis().max(1)).unwrap_or(u64::MAX))
            .arg(value);
        self.query_string(RedisConnectionLane::Fast, "runtime kv set ttl", command)
            .await?;
        Ok(())
    }

    pub(crate) async fn kv_set_if_absent(
        &self,
        key: &str,
        value: String,
        ttl: Duration,
    ) -> Result<bool, DataLayerError> {
        let namespaced_key = self.keyspace.key(key);
        let mut command = cmd("SET");
        command
            .arg(namespaced_key)
            .arg(value)
            .arg("NX")
            .arg("PX")
            .arg(u64::try_from(ttl.as_millis().max(1)).unwrap_or(u64::MAX));
        let result = self
            .query::<Option<String>>(
                RedisConnectionLane::Fast,
                "runtime kv set if absent",
                command,
            )
            .await?;
        Ok(result.is_some())
    }

    pub(crate) async fn kv_get_many(
        &self,
        keys: &[String],
    ) -> Result<Vec<Option<String>>, DataLayerError> {
        let namespaced = keys
            .iter()
            .map(|key| self.keyspace.key(key))
            .collect::<Vec<_>>();
        let mut command = cmd("MGET");
        command.arg(&namespaced);
        self.query(RedisConnectionLane::Fast, "runtime kv mget", command)
            .await
    }

    pub(crate) async fn kv_delete_many(&self, keys: &[String]) -> Result<usize, DataLayerError> {
        let prefix = self.keyspace.key("");
        let namespaced = keys
            .iter()
            .map(|key| {
                if key_belongs_to_prefix(key, &prefix) {
                    key.clone()
                } else {
                    self.keyspace.key(key)
                }
            })
            .collect::<Vec<_>>();
        let mut command = cmd("DEL");
        command.arg(&namespaced);
        let deleted = self
            .query_i64(
                RedisConnectionLane::Admin,
                "runtime kv delete many",
                command,
            )
            .await?;
        Ok(usize::try_from(deleted).unwrap_or(0))
    }

    pub(crate) async fn kv_ttl_seconds(&self, key: &str) -> Result<Option<i64>, DataLayerError> {
        let namespaced_key = self.keyspace.key(key);
        let mut command = cmd("TTL");
        command.arg(&namespaced_key);
        let ttl = self
            .query_i64(RedisConnectionLane::Fast, "runtime kv ttl", command)
            .await?;
        Ok((ttl >= -1).then_some(ttl))
    }

    pub(crate) async fn realtime_bucket_add(
        &self,
        key: &str,
        request_delta: i64,
        token_delta: i64,
        ttl: Duration,
    ) -> Result<(i64, i64), DataLayerError> {
        let key = self.keyspace.key(key);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime bucket add",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_BUCKET_ADD_SCRIPT)
                    .key(key)
                    .arg(request_delta)
                    .arg(token_delta)
                    .arg(i64::try_from(ttl.as_secs().max(1)).unwrap_or(i64::MAX))
                    .invoke_async::<(i64, i64)>(&mut connection)
                    .await
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_bucket_read(
        &self,
        key: &str,
    ) -> Result<Option<(i64, i64)>, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("HMGET");
        command.arg(key).arg("requests").arg("tokens");
        let values = self
            .query::<Vec<Option<i64>>>(
                RedisConnectionLane::Fast,
                "runtime realtime bucket read",
                command,
            )
            .await?;
        let requests = values.first().copied().flatten();
        let tokens = values.get(1).copied().flatten();
        if requests.is_none() && tokens.is_none() {
            return Ok(None);
        }
        Ok(Some((
            requests.unwrap_or_default(),
            tokens.unwrap_or_default(),
        )))
    }

    pub(crate) async fn realtime_buckets_sum(
        &self,
        keys: &[String],
    ) -> Result<(i64, i64), DataLayerError> {
        if keys.is_empty() {
            return Ok((0, 0));
        }
        let namespaced = keys
            .iter()
            .map(|key| self.keyspace.key(key))
            .collect::<Vec<_>>();
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime bucket sum",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                let script = script(REALTIME_BUCKET_SUM_SCRIPT);
                let mut invocation = script.prepare_invoke();
                for key in &namespaced {
                    invocation.key(key);
                }
                invocation
                    .invoke_async::<(i64, i64)>(&mut connection)
                    .await
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_event_add(
        &self,
        collection_key: &str,
        event_id: &str,
        timestamp_ms: u64,
        request_delta: i64,
        token_delta: i64,
        ttl: Duration,
    ) -> Result<bool, DataLayerError> {
        let (index, payloads) = self.realtime_event_keys(collection_key);
        let timestamp_ms = i64::try_from(timestamp_ms).map_err(|_| {
            DataLayerError::InvalidInput(
                "realtime event timestamp must fit in a signed 64-bit score".to_string(),
            )
        })?;
        let ttl_ms = i64::try_from(ttl.as_millis().max(1)).unwrap_or(i64::MAX);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime event add",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_EVENT_ADD_SCRIPT)
                    .key(index)
                    .key(payloads)
                    .arg(event_id)
                    .arg(timestamp_ms)
                    .arg(request_delta)
                    .arg(token_delta)
                    .arg(ttl_ms)
                    .invoke_async::<i64>(&mut connection)
                    .await
                    .map(|added| added > 0)
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_event_remove(
        &self,
        collection_key: &str,
        event_id: &str,
    ) -> Result<bool, DataLayerError> {
        let (index, payloads) = self.realtime_event_keys(collection_key);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime event remove",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_EVENT_REMOVE_SCRIPT)
                    .key(index)
                    .key(payloads)
                    .arg(event_id)
                    .invoke_async::<i64>(&mut connection)
                    .await
                    .map(|removed| removed > 0)
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_events_sum(
        &self,
        collection_key: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Result<(i64, i64), DataLayerError> {
        let (index, payloads) = self.realtime_event_keys(collection_key);
        let start_ms = i64::try_from(start_ms).map_err(|_| {
            DataLayerError::InvalidInput(
                "realtime event lower timestamp must fit in a signed 64-bit score".to_string(),
            )
        })?;
        let end_ms = i64::try_from(end_ms).map_err(|_| {
            DataLayerError::InvalidInput(
                "realtime event upper timestamp must fit in a signed 64-bit score".to_string(),
            )
        })?;
        if start_ms >= end_ms {
            return Ok((0, 0));
        }
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime events sum",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_EVENTS_SUM_SCRIPT)
                    .key(index)
                    .key(payloads)
                    .arg(start_ms)
                    .arg(end_ms)
                    .invoke_async::<(i64, i64)>(&mut connection)
                    .await
                    .map(|(requests, tokens)| (requests.max(0), tokens.max(0)))
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_token_stream_add(
        &self,
        identity: &str,
        token_delta: u64,
        ttl: Duration,
    ) -> Result<u64, DataLayerError> {
        let key = self.realtime_token_ledger_key(identity);
        let ttl_ms = i64::try_from(ttl.as_millis().max(1)).unwrap_or(i64::MAX);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime token stream add",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_TOKEN_STREAM_ADD_SCRIPT)
                    .key(key)
                    .arg(i64::try_from(token_delta).unwrap_or(i64::MAX))
                    .arg(ttl_ms)
                    .invoke_async::<i64>(&mut connection)
                    .await
                    .map(|accepted| u64::try_from(accepted.max(0)).unwrap_or(0))
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_token_stream_add_once(
        &self,
        identity: &str,
        event_id: &str,
        token_delta: u64,
        ttl: Duration,
    ) -> Result<u64, DataLayerError> {
        let key = self.realtime_token_ledger_key(identity);
        let ttl_ms = i64::try_from(ttl.as_millis().max(1)).unwrap_or(i64::MAX);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime token stream add once",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_TOKEN_STREAM_ADD_ONCE_SCRIPT)
                    .key(key)
                    .arg(event_id)
                    .arg(i64::try_from(token_delta).unwrap_or(i64::MAX))
                    .arg(ttl_ms)
                    .invoke_async::<i64>(&mut connection)
                    .await
                    .map(|accepted| u64::try_from(accepted.max(0)).unwrap_or(0))
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_token_terminal_claim(
        &self,
        identity: &str,
        terminal_total: u64,
        ttl: Duration,
    ) -> Result<u64, DataLayerError> {
        let key = self.realtime_token_ledger_key(identity);
        let ttl_ms = i64::try_from(ttl.as_millis().max(1)).unwrap_or(i64::MAX);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime token terminal claim",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_TOKEN_TERMINAL_CLAIM_SCRIPT)
                    .key(key)
                    .arg(i64::try_from(terminal_total).unwrap_or(i64::MAX))
                    .arg(ttl_ms)
                    .invoke_async::<i64>(&mut connection)
                    .await
                    .map(|remainder| u64::try_from(remainder.max(0)).unwrap_or(0))
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_token_terminal_prepare(
        &self,
        identity: &str,
        event_id: &str,
        terminal_total: u64,
        ttl: Duration,
    ) -> Result<u64, DataLayerError> {
        let key = self.realtime_token_ledger_key(identity);
        let ttl_ms = i64::try_from(ttl.as_millis().max(1)).unwrap_or(i64::MAX);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime token terminal prepare",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_TOKEN_TERMINAL_PREPARE_SCRIPT)
                    .key(key)
                    .arg(event_id)
                    .arg(i64::try_from(terminal_total).unwrap_or(i64::MAX))
                    .arg(ttl_ms)
                    .invoke_async::<i64>(&mut connection)
                    .await
                    .map(|remainder| u64::try_from(remainder.max(0)).unwrap_or(0))
                    .map_redis_err()
            },
        )
        .await
    }

    pub(crate) async fn realtime_token_terminal_commit(
        &self,
        identity: &str,
        event_id: &str,
    ) -> Result<bool, DataLayerError> {
        let key = self.realtime_token_ledger_key(identity);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime realtime token terminal commit",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(REALTIME_TOKEN_TERMINAL_COMMIT_SCRIPT)
                    .key(key)
                    .arg(event_id)
                    .invoke_async::<i64>(&mut connection)
                    .await
                    .map(|committed| committed > 0)
                    .map_redis_err()
            },
        )
        .await
    }

    fn realtime_event_keys(&self, collection_key: &str) -> (String, String) {
        // Encode the caller key as hex so braces, colons, and arbitrary UTF-8
        // cannot alter the Redis hash-tag boundary.  Both names then carry the
        // exact same `{tag}` and are safe for Redis Cluster Lua execution.
        let tag = realtime_collection_tag(collection_key);
        (
            self.keyspace
                .key(&format!("realtime:events:{{{tag}}}:index")),
            self.keyspace
                .key(&format!("realtime:events:{{{tag}}}:payloads")),
        )
    }

    fn realtime_token_ledger_key(&self, identity: &str) -> String {
        // Encode the identity into a brace-safe hash tag.  This keeps the key
        // deterministic while preventing caller-controlled braces from
        // changing Redis Cluster slot selection.
        let tag = realtime_collection_tag(identity);
        self.keyspace
            .key(&format!("realtime:token-ledger:{{{tag}}}"))
    }

    pub(crate) async fn scan_keys(
        &self,
        pattern: &str,
        count: usize,
    ) -> Result<Vec<String>, DataLayerError> {
        let pattern = self.keyspace.key(pattern);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Admin,
            self.command_timeout_ms,
            "runtime scan keys",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Admin);
                let mut cursor = 0u64;
                let mut keys = Vec::new();
                loop {
                    let (next_cursor, mut batch) = cmd("SCAN")
                        .arg(cursor)
                        .arg("MATCH")
                        .arg(&pattern)
                        .arg("COUNT")
                        .arg(count.max(1))
                        .query_async::<(u64, Vec<String>)>(&mut connection)
                        .await
                        .map_redis_err()?;
                    keys.append(&mut batch);
                    if next_cursor == 0 {
                        break;
                    }
                    cursor = next_cursor;
                }
                keys.sort();
                Ok(keys)
            },
        )
        .await
    }

    pub(crate) async fn check_and_consume_rate_limit(
        &self,
        input: RateLimitInput<'_>,
    ) -> Result<RateLimitCheck, DataLayerError> {
        let user_key = self.keyspace.key(input.user_key);
        let key_key = self.keyspace.key(input.key_key);
        let raw = run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            self.command_timeout_ms,
            "runtime rate limit check",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(RATE_LIMIT_CHECK_AND_CONSUME_SCRIPT)
                    .key(user_key)
                    .key(key_key)
                    .arg(i64::from(input.user_limit))
                    .arg(i64::from(input.key_limit))
                    .arg(i64::try_from(input.ttl_seconds.max(1)).unwrap_or(i64::MAX))
                    .invoke_async::<Vec<i64>>(&mut connection)
                    .await
                    .map_redis_err()
            },
        )
        .await?;
        if raw.first().copied().unwrap_or_default() == 1 {
            return Ok(RateLimitCheck::Allowed {
                remaining: raw
                    .get(3)
                    .copied()
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or_default(),
            });
        }
        let scope = match raw.get(1).copied().unwrap_or_default() {
            2 => RateLimitScope::Key,
            _ => RateLimitScope::User,
        };
        let limit = raw
            .get(2)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(match scope {
                RateLimitScope::User => input.user_limit,
                RateLimitScope::Key => input.key_limit,
            });
        Ok(RateLimitCheck::Rejected { scope, limit })
    }

    pub(crate) async fn set_add(&self, key: &str, member: &str) -> Result<bool, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("SADD");
        command.arg(&key).arg(member);
        Ok(self
            .query_i64(RedisConnectionLane::Fast, "runtime set add", command)
            .await?
            > 0)
    }

    pub(crate) async fn set_remove(&self, key: &str, member: &str) -> Result<bool, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("SREM");
        command.arg(&key).arg(member);
        Ok(self
            .query_i64(RedisConnectionLane::Fast, "runtime set remove", command)
            .await?
            > 0)
    }

    pub(crate) async fn set_members(&self, key: &str) -> Result<Vec<String>, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("SMEMBERS");
        command.arg(&key);
        let mut values = self
            .query::<Vec<String>>(RedisConnectionLane::Admin, "runtime set members", command)
            .await?;
        values.sort();
        Ok(values)
    }

    pub(crate) async fn set_len(&self, key: &str) -> Result<usize, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("SCARD");
        command.arg(&key);
        let len = self
            .query_i64(RedisConnectionLane::Fast, "runtime set len", command)
            .await?;
        Ok(usize::try_from(len).unwrap_or(0))
    }

    pub(crate) async fn score_set(
        &self,
        key: &str,
        member: &str,
        score: f64,
    ) -> Result<(), DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("ZADD");
        command.arg(&key).arg(score).arg(member);
        self.query_i64(RedisConnectionLane::Fast, "runtime score set", command)
            .await?;
        Ok(())
    }

    pub(crate) async fn score_many(
        &self,
        key: &str,
        members: &[String],
    ) -> Result<Vec<Option<f64>>, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("ZMSCORE");
        command.arg(&key);
        for member in members {
            command.arg(member);
        }
        self.query(RedisConnectionLane::Fast, "runtime score many", command)
            .await
    }

    pub(crate) async fn score_range_by_min(
        &self,
        key: &str,
        min_score: f64,
    ) -> Result<Vec<String>, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("ZRANGEBYSCORE");
        command.arg(&key).arg(min_score).arg("+inf");
        self.query(RedisConnectionLane::Admin, "runtime score range", command)
            .await
    }

    pub(crate) async fn score_remove_by_score(
        &self,
        key: &str,
        max_score: f64,
    ) -> Result<usize, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("ZREMRANGEBYSCORE");
        command.arg(&key).arg("-inf").arg(max_score);
        let removed = self
            .query_i64(RedisConnectionLane::Admin, "runtime score trim", command)
            .await?;
        Ok(usize::try_from(removed).unwrap_or(0))
    }

    pub(crate) async fn score_remove(
        &self,
        key: &str,
        member: &str,
    ) -> Result<bool, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("ZREM");
        command.arg(&key).arg(member);
        Ok(self
            .query_i64(RedisConnectionLane::Fast, "runtime score remove", command)
            .await?
            > 0)
    }

    pub(crate) async fn score_remove_by_rank(
        &self,
        key: &str,
        start: i64,
        stop: i64,
    ) -> Result<usize, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("ZREMRANGEBYRANK");
        command.arg(&key).arg(start).arg(stop);
        let removed = self
            .query_i64(
                RedisConnectionLane::Admin,
                "runtime score rank trim",
                command,
            )
            .await?;
        Ok(usize::try_from(removed).unwrap_or(0))
    }

    pub(crate) async fn score_len(&self, key: &str) -> Result<usize, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("ZCARD");
        command.arg(&key);
        let len = self
            .query_i64(RedisConnectionLane::Fast, "runtime score len", command)
            .await?;
        Ok(usize::try_from(len).unwrap_or(0))
    }

    pub(crate) async fn key_expire(
        &self,
        key: &str,
        ttl: Duration,
    ) -> Result<bool, DataLayerError> {
        let key = self.keyspace.key(key);
        let mut command = cmd("PEXPIRE");
        command
            .arg(&key)
            .arg(u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX));
        Ok(self
            .query_i64(RedisConnectionLane::Fast, "runtime key expire", command)
            .await?
            > 0)
    }

    pub(crate) async fn semaphore_try_acquire(
        &self,
        gate: &'static str,
        limit: usize,
        key: &str,
        token: &str,
        lease_ttl_ms: u64,
        timeout_ms: Option<u64>,
    ) -> Result<(i64, i64), RuntimeSemaphoreError> {
        let now_ms = crate::unix_time_ms();
        let expires_at_ms = now_ms.saturating_add(lease_ttl_ms);
        let key = self.keyspace.key(key);
        let timeout_ms = timeout_ms.or(self.command_timeout_ms);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            timeout_ms,
            "runtime semaphore acquire",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(
                    "redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1]); \
                 local count = redis.call('ZCARD', KEYS[1]); \
                 if count >= tonumber(ARGV[3]) then \
                    redis.call('PEXPIRE', KEYS[1], ARGV[5]); \
                    return {0, count}; \
                 end; \
                 redis.call('ZADD', KEYS[1], ARGV[2], ARGV[4]); \
                 count = redis.call('ZCARD', KEYS[1]); \
                 redis.call('PEXPIRE', KEYS[1], ARGV[5]); \
                 return {1, count};",
                )
                .key(&key)
                .arg(now_ms as i64)
                .arg(expires_at_ms as i64)
                .arg(limit as i64)
                .arg(token)
                .arg(lease_ttl_ms as i64)
                .invoke_async::<(i64, i64)>(&mut connection)
                .await
                .map_redis_err()
            },
        )
        .await
        .map_err(|err| RuntimeSemaphoreError::Unavailable {
            gate,
            limit,
            message: format!("acquire failed: {err}"),
        })
    }

    pub(crate) async fn semaphore_renew(
        &self,
        gate: &'static str,
        limit: usize,
        key: &str,
        token: &str,
        lease_ttl_ms: u64,
        timeout_ms: Option<u64>,
    ) -> Result<i64, RuntimeSemaphoreError> {
        let now_ms = crate::unix_time_ms();
        let expires_at_ms = now_ms.saturating_add(lease_ttl_ms);
        let key = self.keyspace.key(key);
        let timeout_ms = timeout_ms.or(self.command_timeout_ms);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            timeout_ms,
            "runtime semaphore renew",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(
                    "redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1]); \
                 local score = redis.call('ZSCORE', KEYS[1], ARGV[2]); \
                 if not score then return 0; end; \
                 redis.call('ZADD', KEYS[1], 'XX', ARGV[3], ARGV[2]); \
                 redis.call('PEXPIRE', KEYS[1], ARGV[4]); \
                 return 1;",
                )
                .key(&key)
                .arg(now_ms as i64)
                .arg(token)
                .arg(expires_at_ms as i64)
                .arg(lease_ttl_ms as i64)
                .invoke_async::<i64>(&mut connection)
                .await
                .map_redis_err()
            },
        )
        .await
        .map_err(|err| RuntimeSemaphoreError::Unavailable {
            gate,
            limit,
            message: format!("renew failed: {err}"),
        })
    }

    pub(crate) async fn semaphore_release(
        &self,
        gate: &'static str,
        limit: usize,
        key: &str,
        token: &str,
        timeout_ms: Option<u64>,
    ) -> Result<(), RuntimeSemaphoreError> {
        let key = self.keyspace.key(key);
        let timeout_ms = timeout_ms.or(self.command_timeout_ms);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            timeout_ms,
            "runtime semaphore release",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(
                    "local removed = redis.call('ZREM', KEYS[1], ARGV[1]); \
                 if removed > 0 and redis.call('ZCARD', KEYS[1]) == 0 then \
                    redis.call('DEL', KEYS[1]); \
                 end; \
                 return removed;",
                )
                .key(&key)
                .arg(token)
                .invoke_async::<i64>(&mut connection)
                .await
                .map(|_| ())
                .map_redis_err()
            },
        )
        .await
        .map_err(|err| RuntimeSemaphoreError::Unavailable {
            gate,
            limit,
            message: format!("release failed: {err}"),
        })
    }

    pub(crate) async fn semaphore_live_count(
        &self,
        gate: &'static str,
        limit: usize,
        key: &str,
        timeout_ms: Option<u64>,
    ) -> Result<usize, RuntimeSemaphoreError> {
        let now_ms = crate::unix_time_ms();
        let key = self.keyspace.key(key);
        let timeout_ms = timeout_ms.or(self.command_timeout_ms);
        run_lane_with_timeout(
            &self.connections,
            RedisConnectionLane::Fast,
            timeout_ms,
            "runtime semaphore snapshot",
            async {
                let mut connection = self.connections.connection(RedisConnectionLane::Fast);
                script(
                    "redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1]); \
                 return redis.call('ZCARD', KEYS[1]);",
                )
                .key(&key)
                .arg(now_ms as i64)
                .invoke_async::<i64>(&mut connection)
                .await
                .map(|value| value.max(0) as usize)
                .map_redis_err()
            },
        )
        .await
        .map_err(|err| RuntimeSemaphoreError::Unavailable {
            gate,
            limit,
            message: format!("snapshot failed: {err}"),
        })
    }

    async fn query<T>(
        &self,
        lane: RedisConnectionLane,
        operation: &'static str,
        command: RedisCmd,
    ) -> Result<T, DataLayerError>
    where
        T: redis::FromRedisValue,
    {
        run_lane_with_timeout(
            &self.connections,
            lane,
            self.command_timeout_ms,
            operation,
            async {
                let mut connection = self.connections.connection(lane);
                command
                    .query_async::<T>(&mut connection)
                    .await
                    .map_redis_err()
            },
        )
        .await
    }

    async fn query_i64(
        &self,
        lane: RedisConnectionLane,
        operation: &'static str,
        command: RedisCmd,
    ) -> Result<i64, DataLayerError> {
        self.query(lane, operation, command).await
    }

    async fn query_string(
        &self,
        lane: RedisConnectionLane,
        operation: &'static str,
        command: RedisCmd,
    ) -> Result<String, DataLayerError> {
        self.query(lane, operation, command).await
    }
}

fn parse_diagnostics(info: &str, lanes: Vec<RedisLaneDiagnostics>) -> RedisRuntimeDiagnostics {
    RedisRuntimeDiagnostics {
        connected_clients: parse_info_u64(info, "connected_clients"),
        blocked_clients: parse_info_u64(info, "blocked_clients"),
        total_connections_received: parse_info_u64(info, "total_connections_received"),
        rejected_connections: parse_info_u64(info, "rejected_connections"),
        total_commands_processed: parse_info_u64(info, "total_commands_processed"),
        instantaneous_ops_per_sec: parse_info_u64(info, "instantaneous_ops_per_sec"),
        total_error_replies: parse_info_u64(info, "total_error_replies"),
        expired_keys: parse_info_u64(info, "expired_keys"),
        evicted_keys: parse_info_u64(info, "evicted_keys"),
        keyspace_hits: parse_info_u64(info, "keyspace_hits"),
        keyspace_misses: parse_info_u64(info, "keyspace_misses"),
        used_memory_bytes: parse_info_u64(info, "used_memory"),
        maxmemory_bytes: parse_info_u64(info, "maxmemory"),
        memory_fragmentation_ratio_basis_points: parse_info_f64_basis_points(
            info,
            "mem_fragmentation_ratio",
        ),
        lanes,
    }
}

fn parse_info_u64(info: &str, key: &str) -> Option<u64> {
    info.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key)
            .then(|| value.trim().parse::<u64>().ok())
            .flatten()
    })
}

fn parse_info_f64_basis_points(info: &str, key: &str) -> Option<u64> {
    info.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name != key {
            return None;
        }
        let parsed = value.trim().parse::<f64>().ok()?;
        (parsed.is_finite() && parsed >= 0.0).then(|| (parsed * 10_000.0).round() as u64)
    })
}

fn key_belongs_to_prefix(key: &str, prefix: &str) -> bool {
    prefix.is_empty()
        || key == prefix
        || key
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with(':'))
}

fn realtime_collection_tag(collection_key: &str) -> String {
    let mut tag = String::with_capacity(collection_key.len().saturating_mul(2));
    use std::fmt::Write;
    for byte in collection_key.as_bytes() {
        let _ = write!(tag, "{byte:02x}");
    }
    tag
}

#[cfg(test)]
mod tests {
    use super::{
        key_belongs_to_prefix, parse_diagnostics, realtime_collection_tag, RedisRuntimeDiagnostics,
        REALTIME_EVENTS_SUM_SCRIPT, REALTIME_EVENT_ADD_SCRIPT,
        REALTIME_TOKEN_STREAM_ADD_ONCE_SCRIPT, REALTIME_TOKEN_STREAM_ADD_SCRIPT,
        REALTIME_TOKEN_TERMINAL_CLAIM_SCRIPT, REALTIME_TOKEN_TERMINAL_COMMIT_SCRIPT,
        REALTIME_TOKEN_TERMINAL_PREPARE_SCRIPT,
    };

    #[test]
    fn parses_runtime_diagnostics_from_info() {
        let parsed = parse_diagnostics(
            "# Clients\r\nconnected_clients:5\r\nblocked_clients:2\r\n# Memory\r\nused_memory:1048576\r\nmaxmemory:8388608\r\nmem_fragmentation_ratio:1.25\r\n# Stats\r\ntotal_connections_received:42\r\nrejected_connections:0\r\ntotal_commands_processed:99\r\ninstantaneous_ops_per_sec:7\r\ntotal_error_replies:1\r\nexpired_keys:3\r\nevicted_keys:4\r\nkeyspace_hits:10\r\nkeyspace_misses:2\r\n",
            Vec::new(),
        );

        assert_eq!(
            parsed,
            RedisRuntimeDiagnostics {
                connected_clients: Some(5),
                blocked_clients: Some(2),
                total_connections_received: Some(42),
                rejected_connections: Some(0),
                total_commands_processed: Some(99),
                instantaneous_ops_per_sec: Some(7),
                total_error_replies: Some(1),
                expired_keys: Some(3),
                evicted_keys: Some(4),
                keyspace_hits: Some(10),
                keyspace_misses: Some(2),
                used_memory_bytes: Some(1_048_576),
                maxmemory_bytes: Some(8_388_608),
                memory_fragmentation_ratio_basis_points: Some(12_500),
                lanes: Vec::new(),
            }
        );
    }

    #[test]
    fn detects_namespaced_key_prefix_on_boundary() {
        assert!(key_belongs_to_prefix("aether:cache:item", "aether"));
        assert!(key_belongs_to_prefix("aether", "aether"));
        assert!(key_belongs_to_prefix("raw:key", ""));
        assert!(!key_belongs_to_prefix("aetherish:cache:item", "aether"));
    }

    #[test]
    fn realtime_collection_tag_is_stable_and_brace_safe() {
        let first = realtime_collection_tag("dashboard:{site}:events");
        let second = realtime_collection_tag("dashboard:{site}:events");
        assert_eq!(first, second);
        assert!(first.chars().all(|character| character.is_ascii_hexdigit()));
        assert!(!first.contains('{'));
        assert!(!first.contains('}'));
    }

    #[test]
    fn realtime_scripts_encode_open_closed_range_and_idempotency() {
        assert!(REALTIME_EVENT_ADD_SCRIPT.contains("HEXISTS"));
        assert!(REALTIME_EVENT_ADD_SCRIPT.contains("PEXPIRE"));
        assert!(REALTIME_EVENTS_SUM_SCRIPT.contains("'(' .. start_ms"));
        assert!(REALTIME_EVENTS_SUM_SCRIPT.contains("ZRANGEBYSCORE"));
    }

    #[test]
    fn realtime_token_scripts_fence_terminal_and_use_ttl() {
        assert!(REALTIME_TOKEN_STREAM_ADD_SCRIPT.contains("terminal_claimed"));
        assert!(REALTIME_TOKEN_STREAM_ADD_SCRIPT.contains("PEXPIRE"));
        assert!(REALTIME_TOKEN_STREAM_ADD_ONCE_SCRIPT.contains("HEXISTS"));
        assert!(REALTIME_TOKEN_STREAM_ADD_ONCE_SCRIPT.contains("stream_event:"));
        assert!(REALTIME_TOKEN_TERMINAL_CLAIM_SCRIPT.contains("stream_tokens"));
        assert!(REALTIME_TOKEN_TERMINAL_CLAIM_SCRIPT.contains("terminal_claimed"));
        assert!(REALTIME_TOKEN_TERMINAL_CLAIM_SCRIPT.contains("PEXPIRE"));
        assert!(REALTIME_TOKEN_TERMINAL_PREPARE_SCRIPT.contains("terminal_event_id"));
        assert!(REALTIME_TOKEN_TERMINAL_PREPARE_SCRIPT.contains("terminal_remainder"));
        assert!(REALTIME_TOKEN_TERMINAL_COMMIT_SCRIPT.contains("terminal_committed"));
    }
}
