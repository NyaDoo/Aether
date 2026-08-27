# Historical usage/billing recovery (read-only dry run)

This runbook is an evidence report and a dry-run query. It is intended for a
restored PostgreSQL clone or a read replica. It does **not** connect to the
production host, and it contains no data-changing statement. The only SQL
transaction statements are `BEGIN`, read-only session settings, and
`ROLLBACK`.

The companion query is [usage-recovery-dry-run.sql](./usage-recovery-dry-run.sql).

Before running it, verify the clone is on the adapter PostgreSQL migration
track with (at least) the usage split/body states, billing-v3 snapshot fields,
`upstream_is_stream`, counter-delta table, and the current
`usage_billing_facts`/outcome view (the relevant migration families are
`20260413020000`, `20260418000000`, `20260424000000`, `20260505130000`,
`20260512090000`, `20260518000000`, and `20260813000000`). This is a schema
precondition, not a request to run migrations. The logical/bootstrap track uses different types
and is not interchangeable with this SQL. If any required relation/column or
the view is absent, stop and adapt the report on a clone rather than guessing.

## What is authoritative

The database has two generations of usage columns. The split schema marks the
old body/header, wallet, and pricing columns as compatibility fallbacks:

| Evidence | Authority | What it can prove |
|---|---|---|
| `usage` | request identity, terminal status/code, timings, legacy token/cost mirrors | A request row exists and what the last accepted usage event contained |
| `usage_http_audits` + `usage_body_blobs` | request/provider/client headers and body captures | A body was captured, and (when the blob exists) bytes can be independently hashed/decompressed |
| `usage_settlement_snapshots` | settlement status, all wallet balance before/after components, provider monthly, finalized time, pricing/token snapshot | Whether settlement was attempted/finalized and the exact historical pricing inputs, if present; any wallet component/provider-monthly fields block a duplicate charge even if status is inconsistent |
| `entitlement_usage_ledgers` | daily entitlement debit | A quota amount was already consumed; unique key `(user_entitlement_id, request_id)` is the duplicate guard |
| `usage_counter_deltas` | aggregate counter transition outbox | A usage transition was emitted (not an independent provider invoice or wallet charge) |
| `audit_logs` | operational corroboration keyed by `request_id` | Only structured metadata/status that is actually present; free-text descriptions are not token evidence |
| `provider_usage_tracking` | provider/window aggregate | Window-level reconciliation only; it cannot attribute tokens/cost to one request |
| `wallet_transactions` | admin/payment/refund ledger | Do not infer a usage debit from `link_id=request_id` unless category/reason semantics are proven; the current settlement path does not insert these rows |

Schema references:

* `usage_body_blobs` and `usage_http_audits`: `crates/aether-data/adapters/postgres/migrations/20260413020000_squash_usage_schema_split.sql:10-53`.
* Pricing/token settlement columns: `crates/aether-data/runtime/schema/logical/006_usage.toml:946-1154`.
* `audit_logs`: `crates/aether-data/runtime/schema/logical/001_identity.toml:444-522`.
* `provider_usage_tracking`: `crates/aether-data/runtime/schema/logical/002_provider_catalog.toml:1451-1510`.
* `entitlement_usage_ledgers`: `crates/aether-data/runtime/schema/logical/005_wallet_billing.toml:988-1044`.

The latest `usage_billing_facts` view prefers
`usage_settlement_snapshots.billing_*` and falls back to the deprecated usage
mirrors (`crates/aether-data/adapters/postgres/migrations/20260813000000_add_request_outcome_metrics.sql:55-423`).
Use the view for canonical token/cost projections, but inspect the raw snapshot
status and settlement rows before making a recovery decision. The dry run also
flags a conflict when the raw snapshot status and the request-metadata status
disagree (including flat versus either nested copy, or the two nested copies
disagreeing); such a row is not even a manual snapshot-review candidate. If the
settlement columns were never materialized but the immutable `settlement_snapshot` JSON
was attached to `usage.request_metadata`, the query validates and exposes that
metadata as a fallback; malformed/non-numeric values are ignored.

The SQL intentionally does **not** parse `usage_settlement_snapshots.settlement_snapshot`
JSONB. Newer write paths normally materialize its scalar fields as well, but an
older or partially written row may contain only that JSONB. Such a row is a
conservative false negative in this report and requires a separate, offline
JSON/schema review; do not infer that no snapshot exists. Likewise, the query
uses the flat `billing_snapshot_status` metadata mirror in preference to the
nested `billing_snapshot.status`. If both are present but disagree, the row
must be treated as a metadata conflict during manual review even though this
read-only query cannot prove which value was intended. A legacy status such as
`resolved` is surfaced as raw metadata but is not accepted as `complete` by the
current settlement gate; route it to manual/legacy review, never automatic
charging.

The complete-snapshot predicate is deliberately stricter than the runtime's
settlement gate: it requires a finite non-negative historical cost and a
pricing source plus a rule ID/version (or an explicit free-tier marker). The
current snapshot writer stores the billing engine revision as
`billing_plan_snapshot.engine_version`; that field is **not** interchangeable
with `rule_version`. If the scalar `billing_rule_version` is absent, the row is
reported as a conservative false negative/manual review even when the snapshot
status says `complete`; do not fill the gap from today's catalog or guess a
rule version.

## How the stale cleanup treats accounting shells

The stale sweep is now fail-closed. It never promotes usage from a scheduler
candidate marker. For a non-video request older than the configured timeout,
the PostgreSQL, MySQL, and SQLite adapters select every usage row in
`pending`, `streaming`, `completed`, `failed`, or `cancelled` status when the
legacy billing mirror is still `pending`, `finalized_at` is null, and the
effective settlement snapshot is also pending/unfinalized. Each selected row
is guarded against a terminal snapshot race, then written as `failed` with
`billing_status='void'`, a finalization timestamp, zero cost mirrors, and a
void settlement snapshot. Open `pending`/`streaming` candidates are marked
failed in the same transaction; an already-successful candidate is left as
diagnostic history and does not make the usage row `completed`.

The selector explicitly excludes asynchronous video contracts using the
normalized `request_type`, `api_format`, `endpoint_kind`,
`endpoint_api_format`, and `provider_endpoint_kind` fields. Each value is
whitespace-normalized/case-folded and treated as video when it is exactly
`video` or its final colon-delimited kind is `video` (the same rule used by the
runtime billing-snapshot gate). This prevents provider-specific values such as
`openai:video`, `openai: video`, and `openai:\tvideo` from entering the
synchronous stale sweep. Terminal/finalized settlement snapshots are also
excluded. Verify the adapter that owns the production database before running
the report.

Cleanup now retains request/response headers, inline/compressed bodies, body
blobs, audit refs, and the usage row itself whenever either the legacy usage
mirror or a settlement snapshot is still `pending` and unfinalized. The
selectors, preview counts, and final mutations use the same guard; adapter
transactions re-check it before deleting evidence. This evidence-retention
rule is intentionally stricter than the stale settlement selector.

The previous implementation used a `success` candidate with
`finished_at`/`stream_completed` as a recovery promotion signal. That path has
been removed in all three adapters. A 2xx candidate, non-zero token mirror,
`stream_completed` marker, or first-byte timestamp is not authoritative usage
and cannot justify a completed or billable row. The report still exposes these
markers, including the historical `cleanup_promotion_fingerprint`, so old
false-success rows can be investigated without turning them into automatic
charges.

Body state is decisive. `inline` and `reference` are potentially complete;
`truncated`, `disabled`, `unavailable`, and `none` are not complete evidence.
The enum and storage labels are defined in
`crates/aether-data/contracts/src/repository/usage/types.rs:1499-1626`.
Resolved-row cleanup can delete old blobs and clear refs, after which
historical reconstruction is impossible from the database alone. Unresolved
pending/unfinalized rows are retained regardless of age.

## Recovery categories and exact boundaries

The SQL emits `recovery_category`, `proposed_action`, `cleanup_action`, and
`confidence`. Every category is **report-only**. No row is an approval to
replay, debit a wallet, or insert a ledger entry. `cleanup_action` describes
what the live stale sweep does; it is not a write performed by this read-only
query. Conditions are evaluated in the order shown; a human must prove
terminal provider usage, historical pricing, and idempotency in a separate
read-write change review before any historical charge repair is considered.

| Category | Required evidence | Action | Confidence |
|---|---|---|---|
| `already_settled_do_not_touch` | Effective billing status is `settled` | No replay; only repair a read model if separately approved | High |
| `intentional_void_do_not_charge` | Effective status `void` (normally failed or ordinary 499 cancellation) | Never reset to pending automatically | High |
| `settlement_attempted_insufficient_quota` | Effective status `insufficient_quota` | Do not charge again; route to quota/retry business process | High |
| `stale_cleanup_failed_void` | Non-video, older than the stale timeout, status `pending`/`streaming`/`completed`/`failed`/`cancelled`, billing `pending`, and no finalized settlement snapshot | The live sweep records `failed` + `void` under a terminal-snapshot race guard; candidate success is diagnostic only and never promotes usage | High for no automatic charge |
| `cancelled_drained_snapshot_manual_review` | `status='cancelled'`, `status_code=499`, billing `pending`, literal `request_metadata.billing_treat_as_completed=true`, stricter historical pricing fields, exactly matched candidate identity, a durable 2xx marker **and** reconstructable provider response, **no finalized/settlement/ledger/counter evidence and no status conflict** | Report only. Independently parse a provider terminal usage summary; preserve 499. No replay from this report | Manual only; SQL does not verify terminal proof |
| `async_video_manual_poll_snapshot_required` | A non-failed, non-cancelled request whose request/endpoint/provider kind equals `video` or ends `:video` | Report only; route to the separately approved video poller workflow; no stale-sweep replay | Manual workflow required |
| `false_success_snapshot_manual_review` | `status='completed'`, effective billing `pending`, stricter complete snapshot, exactly matched candidate identity, durable 2xx marker and complete response/client body, **not selected by the stale sweep**, **no finalized/settlement/ledger/counter evidence and no status conflict** | Report only. Re-read under lock and independently prove terminal usage, snapshot dimensions, and idempotency; never hand-debit a wallet | Manual only; SQL does not authorize settlement |
| `false_success_body_reconstruct_manual_review` | `status='completed'`, billing `pending`, exactly matched candidate identity, durable 2xx marker, complete response/client body, no complete historical pricing snapshot, **not selected by the stale sweep**, and no finalized/settlement/ledger/counter evidence or status conflict | Report only. Offline decompress/hash/map provider usage, pin a historical rule/price, then obtain separate approval; no write from this query | Manual only until terminal marker and historical price are proven |
| `false_success_no_terminal_evidence` | Cleanup fingerprint but no complete provider/client body; a scheduler candidate marker alone is not terminal proof | If stale, the live sweep fails and voids it; otherwise preserve pending and do not infer success or tokens from first-byte time | Low (diagnostic only) |
| `cancelled_partial_no_charge` | 499/cancelled without an explicit complete/drained terminal summary/snapshot, or body state partial/unavailable | Keep void/no charge. Never estimate tokens from prompt length or partial output | High |
| `failed_or_user_error_no_bill` | Failed status or HTTP status >=400 (unless explicit drained-cancel path above) | Keep void; body usage alone does not override failure billing policy | High |
| `aggregate_or_audit_only_manual_review` | Only counter/audit/provider-window evidence is available | Reconcile aggregates, but do not synthesize a request charge | Medium/low |
| `irrecoverable_or_unclassified_no_automatic_charge` | No complete terminal usage and no historical pricing snapshot | No automatic recovery | High for “do not charge” |

The distinction between ordinary and drained 499 is implemented in the event
contract, not inferred from latency: `Cancelled` normally maps to
`billing_status='void'`, while `billing_treat_as_completed=true` maps to
`'pending'` (`crates/aether-usage/runtime/src/record.rs:180-197`). A cancelled
stream with no terminal summary must have zero/unknown token fields; tests
explicitly prohibit estimates from request text or partial output
(`crates/aether-usage/runtime/src/write.rs`, tests around
`cancelled_stream_usage_does_not_estimate_tokens...`).

The report exposes `billing_treat_as_completed_marker` as a tri-state value
and `billing_treat_as_completed_proven` as its strict proof projection. Only a
JSON boolean literal `true` makes the proof column true. `false`, a non-boolean
value, or an absent field is not proof and leaves a legacy 499 in the
manual/no-charge path. Even `true` is never sufficient by itself: the SQL
also requires exactly matched candidate identity, a complete historical
pricing snapshot, a durable terminal candidate plus a reconstructable provider
response, no status
conflict, and no finalized/settlement/ledger/counter evidence before it emits
the manual-review category. It still never authorizes a charge.

## Running the dry run

Use a least-privileged read-only DSN and pin all parameters. The query emits
CSV and never selects payload bytes. `-q` and separate stderr are intentional:
psql status tags must not contaminate the evidence CSV, and a non-empty stderr or
non-zero exit means the run is invalid:

The SQL fails closed unless all three timestamps are finite,
`from_ts < to_ts <= as_of_ts`, and `stale_minutes` is finite and at least 120.
The live gateway clamps the stale cleanup timeout to a 120-minute floor, so a
smaller value would not describe the current runtime and is rejected. To study
an older deployment that really used a shorter timeout, adapt and version a
separate clone-only query instead of labeling those rows as current candidates.
Scope is by `usage.created_at`; for
an incident window ending at `as_of_ts`, begin at least one stale timeout before
the first suspected cleanup batch, otherwise a shell created just before the
window can be missed. The query reports `stale_cleanup_candidate` and
`cleanup_action` (either `failed_and_void` or
`not_selected_by_stale_sweep`) using the same five statuses and
non-video/effective-snapshot guards as the live sweep; it does not execute that
sweep. `as_of_ts` is an analysis watermark, not proof that rows were immutable
at that time.

```sh
psql "$REPORTING_DSN" -X -q -v ON_ERROR_STOP=1 \
  -v from_ts='2026-08-01T00:00:00Z' \
  -v to_ts='2026-09-01T00:00:00Z' \
  -v as_of_ts='2026-08-26T10:00:00Z' \
  -v stale_minutes=120 \
  -f docs/operations/usage-recovery-dry-run.sql \
  > recovery.csv 2> recovery.stderr
```

Record the database snapshot timestamp, schema/migration version, query file
SHA-256, row count, and a hash of the resulting CSV. A useful first summary is
performed locally, away from production credentials:

```sh
python3 - <<'PY' recovery.csv
import csv
import sys
from collections import Counter

with open(sys.argv[1], newline="", encoding="utf-8") as fh:
    counts = Counter(row["recovery_category"] for row in csv.DictReader(fh))
for category, count in counts.most_common():
    print(f"{count:>8} {category}")
PY
```

The query relies on the existing request/time and request-id indexes on `usage`,
`request_candidates`, `usage_body_blobs`, settlement snapshots, and counter
deltas. Do not
add an index on the live primary solely for this report; use a restored clone
or a read replica if the bounded scan exceeds the statement timeout.

The output deliberately includes both raw and canonical token/cost fields,
candidate terminal markers, the `stale_cleanup_candidate` and
`cleanup_action` decisions, body states/blob existence (including metadata-only
canonical refs), settlement evidence, raw-vs-metadata snapshot status (including
flat-vs-nested metadata disagreement), whether a JSONB snapshot/dimensions blob
is present, all counter-delta evidence, status conflicts, reference-corruption
flags, and the possible (non-authoritative) wallet links. A missing blob for a
non-null response/client reference is `capture_corrupt` in the evidence sense
and blocks that body from upgrading a recovery candidate; request/provider
capture corruption is separately reported and may still matter to an offline
reconciliation. The report separates
provider-response and client-response captures; a client-response body may be
transformed by the gateway and is therefore still manual-review evidence even
when it contains a usage object. The SQL never parses provider payloads, so
`has_durable_terminal_2xx_candidate` is only a routing/candidate hint, not
terminal proof; it can never override `stale_cleanup_candidate` or authorize a
completed promotion. Candidate evidence is accepted for a manual-review
category only when exactly one `request_candidates.id` matches the selected
`usage_routing_snapshots.candidate_id`, candidate index, provider, endpoint,
and provider key (with the usage mirror as the documented fallback). An
aggregate success from a different retry/provider is not attributable to the
usage row. Missing or conflicting identity remains report-only evidence and
cannot upgrade a recovery category.

Either `usage.finalized_at` or `usage_settlement_snapshots.finalized_at` is
existing settlement evidence and blocks manual charge-recovery categories,
even if the billing-status mirrors disagree. Re-read any surviving candidate
under a lock and retain the original finalized timestamp as an idempotency
boundary.

For provider-level corroboration, run this separate read-only aggregate query
over the same time window. A request may match more than one overlapping window,
so the result is for reconciliation only and must never be divided back into
per-request charges:

```sql
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;
WITH p AS (
  SELECT CAST(:'from_ts' AS timestamptz) AS from_ts,
         CAST(:'to_ts' AS timestamptz) AS to_ts
), request_totals AS (
  SELECT provider_id,
         COUNT(*) AS usage_requests,
         COALESCE(SUM(total_tokens), 0) AS usage_tokens,
         COALESCE(SUM(actual_total_cost_usd), 0) AS usage_actual_cost
  FROM public.usage_billing_facts
  CROSS JOIN p
  WHERE created_at >= p.from_ts AND created_at < p.to_ts
  GROUP BY provider_id
), window_totals AS (
  SELECT provider_id,
         COUNT(*) AS tracking_windows,
         COALESCE(SUM(total_requests), 0) AS tracked_requests,
         COALESCE(SUM(successful_requests), 0) AS tracked_successes,
         COALESCE(SUM(failed_requests), 0) AS tracked_failures,
         COALESCE(SUM(total_cost_usd), 0) AS tracked_cost
  FROM public.provider_usage_tracking
  CROSS JOIN p
  WHERE window_end > p.from_ts AND window_start < p.to_ts
  GROUP BY provider_id
)
SELECT COALESCE(r.provider_id, w.provider_id) AS provider_id,
       r.usage_requests, r.usage_tokens, r.usage_actual_cost,
       w.tracking_windows, w.tracked_requests, w.tracked_successes,
       w.tracked_failures, w.tracked_cost
FROM request_totals r
FULL OUTER JOIN window_totals w USING (provider_id)
ORDER BY COALESCE(r.provider_id, w.provider_id);
ROLLBACK;
```

`provider_usage_tracking` has no request ID or token columns and is therefore a
consistency check (and may be absent/stale), not a recovery source. The column
names above are those of the adapter-migration PostgreSQL table; the
`usage_billing_facts` names come from its companion view. The logical/bootstrap
schema uses different timestamp representations; verify the installed
migration set and view/table definitions on the clone before running any query.
If the view or table is unavailable, stop and do not substitute a guessed
column.

## Optional payload export (still read-only)

Only export payloads for request IDs already reviewed from `recovery.csv`.
Run this as a separate read-only query on the clone; keep the resulting file in
an access-controlled temporary directory. The export is intentionally explicit
about all four header/body fields and returns bounded inline JSON plus base64
compressed bytes for offline hashing/decompression. It is still highly
sensitive: request metadata, headers, and compressed bytes may contain prompt
content, PII, internal URLs, or secrets. Use a short-lived least-privilege role,
an allow-list relation/parameter (never string-concatenate IDs), an explicit
size cap, encryption at rest, redaction, and a deletion/retention ticket. Do not
export from a live primary or place DSNs/passwords in the output.

```sql
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL lock_timeout = '2s';
\pset format csv
\pset footer off
WITH export_params AS (
  SELECT (
    CASE
      WHEN CAST(:'max_payload_bytes' AS bigint) > 0
        THEN CAST(:'max_payload_bytes' AS text)
      ELSE 'invalid max_payload_bytes: require a positive integer'
    END
  )::bigint AS max_payload_bytes
)
SELECT
  u.request_id,
  u.provider_id,
  u.provider_api_key_id,
  u.model,
  u.target_model,
  u.api_format,
  u.endpoint_api_format,
  u.request_type,
  u.status,
  u.status_code,
  u.created_at,
  -- Metadata is retained for offline correlation only; redact before sharing.
  CASE WHEN u.request_metadata IS NOT NULL
             AND octet_length(u.request_metadata::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.request_metadata END AS request_metadata,
  -- Audit-owner headers are authoritative; legacy columns are compatibility
  -- fallbacks. All may contain credentials or personal data.
  CASE WHEN h.request_headers IS NOT NULL
             AND octet_length(h.request_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN h.request_headers END AS audit_request_headers,
  CASE WHEN h.provider_request_headers IS NOT NULL
             AND octet_length(h.provider_request_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN h.provider_request_headers END AS audit_provider_request_headers,
  CASE WHEN h.response_headers IS NOT NULL
             AND octet_length(h.response_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN h.response_headers END AS audit_response_headers,
  CASE WHEN h.client_response_headers IS NOT NULL
             AND octet_length(h.client_response_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN h.client_response_headers END AS audit_client_response_headers,
  CASE WHEN u.request_headers IS NOT NULL
             AND octet_length(u.request_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.request_headers END AS legacy_request_headers,
  CASE WHEN u.provider_request_headers IS NOT NULL
             AND octet_length(u.provider_request_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.provider_request_headers END AS legacy_provider_request_headers,
  CASE WHEN u.response_headers IS NOT NULL
             AND octet_length(u.response_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.response_headers END AS legacy_response_headers,
  CASE WHEN u.client_response_headers IS NOT NULL
             AND octet_length(u.client_response_headers::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.client_response_headers END AS legacy_client_response_headers,
  CASE WHEN u.request_body IS NOT NULL
             AND octet_length(u.request_body::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.request_body END AS legacy_request_body_json,
  CASE WHEN u.provider_request_body IS NOT NULL
             AND octet_length(u.provider_request_body::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.provider_request_body END AS legacy_provider_request_body_json,
  CASE WHEN u.response_body IS NOT NULL
             AND octet_length(u.response_body::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.response_body END AS legacy_response_body_json,
  CASE WHEN u.client_response_body IS NOT NULL
             AND octet_length(u.client_response_body::text)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN u.client_response_body END AS legacy_client_response_body_json,
  CASE WHEN u.request_body_compressed IS NOT NULL
             AND octet_length(u.request_body_compressed)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN encode(u.request_body_compressed, 'base64') END
       AS legacy_request_body_compressed_b64,
  CASE WHEN u.provider_request_body_compressed IS NOT NULL
             AND octet_length(u.provider_request_body_compressed)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN encode(u.provider_request_body_compressed, 'base64') END
       AS legacy_provider_request_body_compressed_b64,
  CASE WHEN u.response_body_compressed IS NOT NULL
             AND octet_length(u.response_body_compressed)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN encode(u.response_body_compressed, 'base64') END
       AS legacy_response_body_compressed_b64,
  CASE WHEN u.client_response_body_compressed IS NOT NULL
             AND octet_length(u.client_response_body_compressed)
                    <= CAST(:'max_payload_bytes' AS bigint)
       THEN encode(u.client_response_body_compressed, 'base64') END
       AS legacy_client_response_body_compressed_b64,
  COALESCE(h.request_body_ref,
      CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'request_body_ref'), '')
                    = format('usage://request/%s/request_body', u.request_id)
           THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'request_body_ref'), '')
      END)
      AS effective_request_body_ref,
  h.request_body_state,
  COALESCE(h.provider_request_body_ref,
      CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'provider_request_body_ref'), '')
                    = format('usage://request/%s/provider_request_body', u.request_id)
           THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'provider_request_body_ref'), '')
      END)
      AS effective_provider_request_body_ref,
  h.provider_request_body_state,
  COALESCE(h.response_body_ref,
      CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'response_body_ref'), '')
                    = format('usage://request/%s/response_body', u.request_id)
           THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'response_body_ref'), '')
      END)
      AS effective_response_body_ref,
  h.response_body_state,
  COALESCE(h.client_response_body_ref,
      CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'client_response_body_ref'), '')
                    = format('usage://request/%s/client_response_body', u.request_id)
           THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'client_response_body_ref'), '')
      END)
      AS effective_client_response_body_ref,
  h.client_response_body_state,
  encode(rb.payload_gzip, 'base64') AS response_payload_gzip_b64,
  encode(cb.payload_gzip, 'base64') AS client_payload_gzip_b64,
  encode(qb.payload_gzip, 'base64') AS request_payload_gzip_b64,
  encode(pb.payload_gzip, 'base64') AS provider_request_payload_gzip_b64
FROM public."usage" AS u
CROSS JOIN export_params AS ep
LEFT JOIN public.usage_http_audits AS h ON h.request_id = u.request_id
LEFT JOIN public.usage_body_blobs AS rb
  ON rb.request_id = u.request_id
 AND rb.body_ref = COALESCE(h.response_body_ref,
     CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'response_body_ref'), '')
                   = format('usage://request/%s/response_body', u.request_id)
          THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'response_body_ref'), '')
     END)
 AND rb.body_field = 'response_body'
 AND octet_length(rb.payload_gzip) <= CAST(:'max_payload_bytes' AS bigint)
LEFT JOIN public.usage_body_blobs AS cb
  ON cb.request_id = u.request_id
 AND cb.body_ref = COALESCE(h.client_response_body_ref,
     CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'client_response_body_ref'), '')
                   = format('usage://request/%s/client_response_body', u.request_id)
          THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'client_response_body_ref'), '')
     END)
 AND cb.body_field = 'client_response_body'
 AND octet_length(cb.payload_gzip) <= CAST(:'max_payload_bytes' AS bigint)
LEFT JOIN public.usage_body_blobs AS qb
  ON qb.request_id = u.request_id
 AND qb.body_ref = COALESCE(h.request_body_ref,
     CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'request_body_ref'), '')
                   = format('usage://request/%s/request_body', u.request_id)
          THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'request_body_ref'), '')
     END)
 AND qb.body_field = 'request_body'
 AND octet_length(qb.payload_gzip) <= CAST(:'max_payload_bytes' AS bigint)
LEFT JOIN public.usage_body_blobs AS pb
  ON pb.request_id = u.request_id
 AND pb.body_ref = COALESCE(h.provider_request_body_ref,
     CASE WHEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'provider_request_body_ref'), '')
                   = format('usage://request/%s/provider_request_body', u.request_id)
          THEN NULLIF(BTRIM(u.request_metadata::jsonb ->> 'provider_request_body_ref'), '')
     END)
 AND pb.body_field = 'provider_request_body'
 AND octet_length(pb.payload_gzip) <= CAST(:'max_payload_bytes' AS bigint)
-- Replace the placeholder with a parameterized/temporary allow-list, not by
-- interpolating untrusted text. For example, bind :request_id and use
-- `WHERE u.request_id = :'request_id'` for one reviewed request, or join a
-- temporary table created by an approved offline process.
WHERE u.request_id = :'request_id'
  AND ep.max_payload_bytes > 0
  AND (
    u.request_metadata IS NOT NULL
    OR h.request_headers IS NOT NULL
    OR h.provider_request_headers IS NOT NULL
    OR h.response_headers IS NOT NULL
    OR h.client_response_headers IS NOT NULL
    OR u.request_headers IS NOT NULL
    OR u.provider_request_headers IS NOT NULL
    OR u.response_headers IS NOT NULL
    OR u.client_response_headers IS NOT NULL
    OR rb.body_ref IS NOT NULL
    OR cb.body_ref IS NOT NULL
    OR qb.body_ref IS NOT NULL
    OR pb.body_ref IS NOT NULL
    OR u.request_body IS NOT NULL
    OR u.request_body_compressed IS NOT NULL
    OR u.provider_request_body IS NOT NULL
    OR u.provider_request_body_compressed IS NOT NULL
    OR u.response_body IS NOT NULL
    OR u.response_body_compressed IS NOT NULL
    OR u.client_response_body IS NOT NULL
    OR u.client_response_body_compressed IS NOT NULL
    OR EXISTS (
      SELECT 1 FROM public.usage_body_blobs AS any_blob
      WHERE any_blob.request_id = u.request_id
    )
  );
ROLLBACK;
```

For a single reviewed request, invoke the export with a bounded metadata,
header, and payload size,
for example `psql ... -X -q -v ON_ERROR_STOP=1 -v
request_id='req_approved_for_review' -v max_payload_bytes=10485760 -f export.sql
> payload.csv 2> payload.stderr`. Treat a missing metadata/header/payload value
caused by the cap as `not_exported`, not as evidence that the value/blob does
not exist. The same bound applies to legacy inline/compressed columns; a NULL
export may mean “over cap”.
Use a positive integer for `max_payload_bytes`. The export guard rejects zero
or negative values; missing or non-numeric values fail the SQL cast. In all
three cases treat the invocation as invalid (check the separated stderr/exit
status), never as evidence that a payload is absent.
The bound is a transfer safeguard, not a parser safety guarantee; the offline
decompressor must enforce its own memory/time limits.

Do not run the export against a live primary merely to inspect a suspected
row. PostgreSQL has no core gzip-inflate function suitable for this analysis;
decode and inflate the base64 values offline. The joins intentionally use the
stored `body_field` and request ID rather than trusting an audit ref; this covers
metadata-only canonical refs, but a missing blob or mismatched ref remains
capture-corrupt and is not terminal proof.

## Offline reconstruction algorithm

The offline tool/process should emit an append-only manifest, never an UPDATE:

```text
request_id, body_ref, body_sha256, body_state, decompressed_bytes,
provider_contract, terminal_marker, terminal_marker_source,
input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
total_tokens, usage_path, pricing_source, pricing_rule_id, pricing_version,
derived_actual_cost_usd, category, confidence, reason
```

1. Base64-decode and gzip-decompress each payload. Reject a bad gzip stream,
   an empty payload, and a body whose declared state is `truncated`,
   `disabled`, `unavailable`, or `none`. Hash the exact compressed and
   decompressed bytes before parsing.
2. Parse JSON or conservative SSE. Terminal markers follow the implementation:
   `response.completed` is OpenAI success; `response.failed`,
   `response.incomplete`, and `error` are failures; an SSE `[DONE]` is only
   accepted when the stream also has a valid terminal usage signal. For other
   providers require their explicit stop/finish event (for example Claude
   `message_stop` or a Gemini `finishReason`). An open stream, a candidate
   `stream_completed` marker, or a 2xx status alone is not terminal proof.
   Record the exact event/frame and parser version in the append-only manifest;
   this proof is outside the SQL and must be independently reviewed.
3. Apply the same provider usage precedence as
   `crates/aether-usage/runtime/src/usage_mapper.rs:7-346`: inspect the
   provider's explicit `usage`/`usageMetadata` (including nested response,
   message, item, and final chunk paths), map input/output/cache fields, and
   derive a missing total only from explicit input/output signals. Do **not**
   count words/characters or estimate from a partial response. Keep the raw
   JSON path that supplied every number.
4. For a 499/cancelled stream, accept tokens only when a terminal summary or
   explicit provider usage was captured. This is the code's tested behavior;
   partial output is not billable evidence.
5. Price only from a stored complete settlement/billing snapshot (including
   `billing_rule_id`, `billing_rule_version`, dimensions, and prices). The
   active catalog lookup in `crates/aether-data/adapters/postgres/src/billing.rs`
   is not versioned; recalculating an old request against today's catalog is
   not a safe historical repair. If only tokens are recoverable, stop at
   “usage facts recovered / billing manual review.” A manifest row never
   authorizes a charge; retain the raw evidence hash and obtain a separate
   approval for any later change.

## Hypothetical replay protocol (separate change approval required)

The query and this document do not perform or authorize replay. The following is
only a design checklist for a separately reviewed change ticket after an
operator has independently proved terminal provider usage, historical pricing,
and idempotency. No direct `UPDATE usage`, wallet debit, or hand-written ledger
insert is safe; if approved at all, use the existing event/upsert/settlement
path:

1. Re-run the dry-run predicates immediately before any proposed replay and
   acquire the request row lock in a short read-write transaction. Abort if the
   effective status changed, a settlement snapshot/finalized time appeared,
   either billing/snapshot status conflicts, or **any** `usage_counter_deltas`
   row (processed or unprocessed) now exists for the request. An entitlement
   ledger, provider-monthly delta, or wallet evidence is independently a hard
   stop. A read-only report cannot establish the lock-time predicates.
2. Reconstruct a terminal `UsageEvent` with the exact stored request/provider
   identity and explicit usage. For drained 499, keep `event_type=Cancelled`
   and `billing_treat_as_completed=true`; never turn it into `Completed`.
3. Enrich only with the historical complete pricing snapshot, then call the
   normal `write_event_record`/`settle_usage_if_needed` path. The worker first
   performs the guarded usage upsert and then settlement
   (`crates/aether-usage/runtime/src/worker.rs:675-685`). The upsert computes
   counter deltas from the before/after contribution. This is not a guarantee:
   a bug shell whose stored contribution is zero will produce a positive
   transition, and any pre-existing per-request delta must be reconciled first.
4. Settlement locks the usage row and short-circuits existing `settled`,
   `void`, and `insufficient_quota` statuses
   (`crates/aether-data/adapters/postgres/src/settlement.rs:443-459`). It then
   atomically updates wallet/entitlement state, writes
   `usage_settlement_snapshots`, and finalizes billing. A provider-monthly
   delta is enqueued only for a nonzero actual cost
   (`:704-715`).
5. After commit, verify the same request has one terminal usage row, a complete
   settlement snapshot/finalized timestamp, expected entitlement amount (if
   applicable), and no duplicate counter delta. Re-run the read-only query and
   retain before/after hashes. If any check fails, stop and reconcile manually.

The `usage_counter_deltas` outbox is corroboration, not an invoice: processed
rows can later be cleaned, and a zero delta does not prove that the provider
used zero tokens. Conversely, an existing positive provider-monthly delta or
entitlement ledger is enough to block an automatic re-charge.

## Relation to the supplied production log

The supplied one-hour log contains `pending_cleanup_completed` records with
`recovered=27` and `failed=0` in total. The attached file has **nine**
such records: lines 85, 190, 256, 429, 871, 1121, 1457, 2031, and 2328.
Those records were emitted by the pre-fix cleanup implementation: its
candidate-success promotion explains why the resulting rows can have
`status=completed/status_code=200` while still having no token, cost,
header/body, or finalized billing fields. The current implementation reports
`recovered=0` for this stale path and fails+voids the unfinalized shells after
the timeout. The log has no database payload or pricing evidence, so it cannot
classify any of the historical 27 rows as safely billable by itself; run the
query against a database snapshot and apply the categories above.

The reproducible log-only counts are:

```sh
wc -l /Users/nya/Downloads/aether-app-20260826095211.log   # 2583
rg -n 'pending_cleanup_completed' /Users/nya/Downloads/aether-app-20260826095211.log
rg -c 'event_name="http_request_completed"' /Users/nya/Downloads/aether-app-20260826095211.log  # 735
rg -c 'event_name="frontdoor_request_body_buffer_started"' /Users/nya/Downloads/aether-app-20260826095211.log  # 566
rg -c 'event_name="frontdoor_request_body_buffer_completed"' /Users/nya/Downloads/aether-app-20260826095211.log  # 566
rg -n '\| (ERROR|CRITICAL)' /Users/nya/Downloads/aether-app-20260826095211.log  # no matches
```

Among the 735 access completions, 701 are HTTP 200 and 34 are HTTP 400; 555
are AI calls (526 stream, 29 sync). The 67 warning lines are retry scheduling
(62 stream, 5 sync), not billing/settlement errors. Cleanup lines report only a
batch count and do not carry the recovered request IDs, token usage, pricing,
or body/header references. Thus the log proves the historical cleanup promotion
symptom and its frequency, but not the billable amount or the identity of
individual requests. Use `stale_cleanup_candidate` to find rows that the fixed
sweep will safely void; only rows outside that policy with independently
reconstructed terminal usage belong in a separately approved manual recovery
workflow.
