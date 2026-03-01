use std::path::Path;

use clap::Parser;
use serde::{Deserialize, Serialize};

/// Fields that existed in 0.1.x but were removed in 0.2.0.
const LEGACY_ONLY_KEYS: &[&str] = &[
    "hmac_key",
    "listen_port",
    "timestamp_tolerance",
    "connect_timeout_secs",
    "tls_handshake_timeout_secs",
    "enable_tls",
    "tls_cert",
    "tls_key",
];

/// Fields renamed from 0.1.x `delegate_*` to 0.2.0 `upstream_*`.
const DELEGATE_TO_UPSTREAM: &[(&str, &str)] = &[
    (
        "delegate_connect_timeout_secs",
        "upstream_connect_timeout_secs",
    ),
    (
        "delegate_pool_max_idle_per_host",
        "upstream_pool_max_idle_per_host",
    ),
    (
        "delegate_pool_idle_timeout_secs",
        "upstream_pool_idle_timeout_secs",
    ),
    ("delegate_tcp_keepalive_secs", "upstream_tcp_keepalive_secs"),
    ("delegate_tcp_nodelay", "upstream_tcp_nodelay"),
];

/// Aether tunnel proxy.
///
/// Deployed on overseas VPS to relay API traffic for Aether instances
/// behind the GFW. Connects to Aether via WebSocket tunnel, registers
/// with Aether, and relays upstream requests.
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
pub struct Config {
    /// Aether server URL (e.g. https://aether.example.com)
    #[arg(long, env = "AETHER_PROXY_AETHER_URL")]
    pub aether_url: String,

    /// Management Token for Aether admin API (ae_xxx)
    #[arg(long, env = "AETHER_PROXY_MANAGEMENT_TOKEN")]
    pub management_token: String,

    /// Public IP address of this node (auto-detected if omitted)
    #[arg(long, env = "AETHER_PROXY_PUBLIC_IP")]
    pub public_ip: Option<String>,

    /// Human-readable node name
    #[arg(long, env = "AETHER_PROXY_NODE_NAME", default_value = "proxy-01")]
    pub node_name: String,

    /// Region label (e.g. ap-northeast-1)
    #[arg(long, env = "AETHER_PROXY_NODE_REGION")]
    pub node_region: Option<String>,

    /// Heartbeat interval in seconds
    #[arg(long, env = "AETHER_PROXY_HEARTBEAT_INTERVAL", default_value_t = 30)]
    pub heartbeat_interval: u64,

    /// Allowed destination ports (default: 80,443,8080,8443)
    #[arg(
        long,
        env = "AETHER_PROXY_ALLOWED_PORTS",
        value_delimiter = ',',
        default_values_t = vec![80, 443, 8080, 8443]
    )]
    pub allowed_ports: Vec<u16>,

    /// Aether API request timeout in seconds
    #[arg(
        long,
        env = "AETHER_PROXY_AETHER_REQUEST_TIMEOUT",
        default_value_t = 10
    )]
    pub aether_request_timeout_secs: u64,

    /// Aether API connect timeout in seconds
    #[arg(
        long,
        env = "AETHER_PROXY_AETHER_CONNECT_TIMEOUT",
        default_value_t = 10
    )]
    pub aether_connect_timeout_secs: u64,

    /// Aether API max idle connections per host
    #[arg(
        long,
        env = "AETHER_PROXY_AETHER_POOL_MAX_IDLE_PER_HOST",
        default_value_t = 8
    )]
    pub aether_pool_max_idle_per_host: usize,

    /// Aether API idle timeout in seconds
    #[arg(
        long,
        env = "AETHER_PROXY_AETHER_POOL_IDLE_TIMEOUT",
        default_value_t = 90
    )]
    pub aether_pool_idle_timeout_secs: u64,

    /// Aether API TCP keepalive in seconds (0 disables)
    #[arg(long, env = "AETHER_PROXY_AETHER_TCP_KEEPALIVE", default_value_t = 60)]
    pub aether_tcp_keepalive_secs: u64,

    /// Aether API TCP_NODELAY
    #[arg(long, env = "AETHER_PROXY_AETHER_TCP_NODELAY", default_value_t = true)]
    pub aether_tcp_nodelay: bool,

    /// Enable HTTP/2 when talking to Aether API
    #[arg(long, env = "AETHER_PROXY_AETHER_HTTP2", default_value_t = true)]
    pub aether_http2: bool,

    /// Aether API retry attempts (including initial)
    #[arg(
        long,
        env = "AETHER_PROXY_AETHER_RETRY_MAX_ATTEMPTS",
        default_value_t = 3
    )]
    pub aether_retry_max_attempts: u32,

    /// Aether API retry base delay in milliseconds
    #[arg(
        long,
        env = "AETHER_PROXY_AETHER_RETRY_BASE_DELAY_MS",
        default_value_t = 200
    )]
    pub aether_retry_base_delay_ms: u64,

    /// Aether API retry max delay in milliseconds
    #[arg(
        long,
        env = "AETHER_PROXY_AETHER_RETRY_MAX_DELAY_MS",
        default_value_t = 2000
    )]
    pub aether_retry_max_delay_ms: u64,

    /// Maximum concurrent TCP connections (defaults to hardware estimate)
    #[arg(long, env = "AETHER_PROXY_MAX_CONCURRENT_CONNECTIONS")]
    pub max_concurrent_connections: Option<u64>,

    /// DNS cache TTL in seconds
    #[arg(long, env = "AETHER_PROXY_DNS_CACHE_TTL", default_value_t = 60)]
    pub dns_cache_ttl_secs: u64,

    /// DNS cache capacity (entries)
    #[arg(long, env = "AETHER_PROXY_DNS_CACHE_CAPACITY", default_value_t = 1024)]
    pub dns_cache_capacity: usize,

    /// Upstream HTTP client connect timeout in seconds
    #[arg(
        long,
        env = "AETHER_PROXY_UPSTREAM_CONNECT_TIMEOUT",
        default_value_t = 30
    )]
    pub upstream_connect_timeout_secs: u64,

    /// Upstream HTTP client max idle connections per host
    #[arg(
        long,
        env = "AETHER_PROXY_UPSTREAM_POOL_MAX_IDLE_PER_HOST",
        default_value_t = 64
    )]
    pub upstream_pool_max_idle_per_host: usize,

    /// Upstream HTTP client idle timeout in seconds
    #[arg(
        long,
        env = "AETHER_PROXY_UPSTREAM_POOL_IDLE_TIMEOUT",
        default_value_t = 300
    )]
    pub upstream_pool_idle_timeout_secs: u64,

    /// Upstream TCP keepalive in seconds (0 disables)
    #[arg(
        long,
        env = "AETHER_PROXY_UPSTREAM_TCP_KEEPALIVE",
        default_value_t = 60
    )]
    pub upstream_tcp_keepalive_secs: u64,

    /// Upstream TCP_NODELAY
    #[arg(
        long,
        env = "AETHER_PROXY_UPSTREAM_TCP_NODELAY",
        default_value_t = true
    )]
    pub upstream_tcp_nodelay: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "AETHER_PROXY_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Output logs as JSON
    #[arg(long, env = "AETHER_PROXY_LOG_JSON", default_value_t = false)]
    pub log_json: bool,

    /// Tunnel reconnect base delay in milliseconds (used by exponential backoff)
    #[arg(
        long,
        env = "AETHER_PROXY_TUNNEL_RECONNECT_BASE_MS",
        default_value_t = 500
    )]
    pub tunnel_reconnect_base_ms: u64,

    /// Tunnel reconnect max delay in milliseconds (cap for exponential backoff)
    #[arg(
        long,
        env = "AETHER_PROXY_TUNNEL_RECONNECT_MAX_MS",
        default_value_t = 30000
    )]
    pub tunnel_reconnect_max_ms: u64,

    /// WebSocket tunnel ping interval in seconds
    #[arg(long, env = "AETHER_PROXY_TUNNEL_PING_INTERVAL", default_value_t = 15)]
    pub tunnel_ping_interval_secs: u64,

    /// Maximum concurrent streams over tunnel (auto-detected from hardware if omitted)
    #[arg(long, env = "AETHER_PROXY_TUNNEL_MAX_STREAMS")]
    pub tunnel_max_streams: Option<u32>,

    /// WebSocket tunnel TCP connect timeout in seconds
    #[arg(
        long,
        env = "AETHER_PROXY_TUNNEL_CONNECT_TIMEOUT",
        default_value_t = 15
    )]
    pub tunnel_connect_timeout_secs: u64,

    /// WebSocket tunnel TCP keepalive in seconds (0 disables)
    #[arg(long, env = "AETHER_PROXY_TUNNEL_TCP_KEEPALIVE", default_value_t = 30)]
    pub tunnel_tcp_keepalive_secs: u64,

    /// WebSocket tunnel TCP_NODELAY
    #[arg(long, env = "AETHER_PROXY_TUNNEL_TCP_NODELAY", default_value_t = true)]
    pub tunnel_tcp_nodelay: bool,

    /// Tunnel connection staleness timeout in seconds (triggers reconnect if no data received)
    #[arg(long, env = "AETHER_PROXY_TUNNEL_STALE_TIMEOUT", default_value_t = 45)]
    pub tunnel_stale_timeout_secs: u64,

    /// Number of parallel WebSocket tunnel connections per server (connection pool)
    #[arg(long, env = "AETHER_PROXY_TUNNEL_CONNECTIONS", default_value_t = 3)]
    pub tunnel_connections: u32,
}

impl Config {
    /// Validate configuration values are within sane ranges.
    /// Called after parsing to catch misconfigurations early.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.heartbeat_interval == 0 {
            anyhow::bail!("heartbeat_interval must be > 0");
        }
        if self.heartbeat_interval > 3600 {
            anyhow::bail!("heartbeat_interval must be <= 3600");
        }
        if self.allowed_ports.is_empty() {
            anyhow::bail!("allowed_ports must not be empty");
        }
        for &port in &self.allowed_ports {
            if port == 0 {
                anyhow::bail!("allowed_ports: port 0 is not valid");
            }
        }
        if self.tunnel_connect_timeout_secs == 0 {
            anyhow::bail!("tunnel_connect_timeout_secs must be > 0");
        }
        if self.tunnel_ping_interval_secs == 0 {
            anyhow::bail!("tunnel_ping_interval_secs must be > 0");
        }
        if self.tunnel_stale_timeout_secs <= self.tunnel_ping_interval_secs {
            anyhow::bail!(
                "tunnel_stale_timeout_secs ({}) must be > tunnel_ping_interval_secs ({})",
                self.tunnel_stale_timeout_secs,
                self.tunnel_ping_interval_secs
            );
        }
        if self.tunnel_connections == 0 {
            anyhow::bail!("tunnel_connections must be > 0");
        }
        if self.aether_retry_max_attempts == 0 {
            anyhow::bail!("aether_retry_max_attempts must be >= 1");
        }
        if self.upstream_connect_timeout_secs == 0 {
            anyhow::bail!("upstream_connect_timeout_secs must be > 0");
        }
        Ok(())
    }
}

/// Per-server connection config (used in multi-server TOML `[[servers]]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerEntry {
    pub aether_url: String,
    pub management_token: String,
    /// Per-server node name override. Falls back to the global `node_name`.
    pub node_name: Option<String>,
}

// ---------------------------------------------------------------------------
// TOML config file support
// ---------------------------------------------------------------------------

/// Serializable config for TOML file persistence.
/// All fields are optional -- only populated values are written.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub management_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_ip: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heartbeat_interval: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_ports: Option<Vec<u16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_request_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_connect_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_pool_max_idle_per_host: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_pool_idle_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_tcp_keepalive_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_tcp_nodelay: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_http2: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_retry_max_attempts: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_retry_base_delay_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aether_retry_max_delay_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_concurrent_connections: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dns_cache_ttl_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dns_cache_capacity: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_connect_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_pool_max_idle_per_host: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_pool_idle_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_tcp_keepalive_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_tcp_nodelay: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_json: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_reconnect_base_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_reconnect_max_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_ping_interval_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_max_streams: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_connect_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_tcp_keepalive_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_tcp_nodelay: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_stale_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_connections: Option<u32>,

    /// Multi-server config: each entry connects to a separate Aether instance.
    /// When present, top-level aether_url/management_token are ignored for
    /// tunnel connections (but still injected as env for clap compatibility).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub servers: Vec<ServerEntry>,
}

impl ConfigFile {
    /// Load from a TOML file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    /// Save to a TOML file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Detect and migrate a 0.1.x config file to 0.2.0 format in-place.
    ///
    /// Returns `true` if migration was performed, `false` if already current.
    /// The original file is backed up as `<name>.v1.bak` before rewriting.
    pub fn migrate_legacy(path: &Path) -> anyhow::Result<bool> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Ok(false),
        };
        let mut table: toml::map::Map<String, toml::Value> = toml::from_str(&content)?;

        // Detect legacy format: presence of any 0.1.x-only key.
        let is_legacy = LEGACY_ONLY_KEYS.iter().any(|k| table.contains_key(*k))
            || DELEGATE_TO_UPSTREAM
                .iter()
                .any(|(old, _)| table.contains_key(*old));

        if !is_legacy {
            return Ok(false);
        }

        // 1. Rename delegate_* -> upstream_* (carry over user-customized values)
        for &(old, new) in DELEGATE_TO_UPSTREAM {
            if let Some(val) = table.remove(old) {
                table.entry(new.to_string()).or_insert(val);
            }
        }

        // 2. Build [[servers]] from top-level aether_url + management_token + node_name
        if !table.contains_key("servers") {
            let aether_url = table.get("aether_url").and_then(|v| v.as_str());
            let management_token = table.get("management_token").and_then(|v| v.as_str());
            if let (Some(url), Some(token)) = (aether_url, management_token) {
                let mut entry = toml::map::Map::new();
                entry.insert("aether_url".into(), toml::Value::String(url.to_string()));
                entry.insert(
                    "management_token".into(),
                    toml::Value::String(token.to_string()),
                );
                if let Some(name) = table.get("node_name").and_then(|v| v.as_str()) {
                    entry.insert("node_name".into(), toml::Value::String(name.to_string()));
                }
                table.insert(
                    "servers".into(),
                    toml::Value::Array(vec![toml::Value::Table(entry)]),
                );
            }
        }

        // 3. Remove top-level fields that are now in [[servers]] or obsolete
        table.remove("aether_url");
        table.remove("management_token");
        table.remove("node_name");
        for &key in LEGACY_ONLY_KEYS {
            table.remove(key);
        }

        // 4. Backup original file (abort migration if backup fails)
        let backup_path = path.with_extension("v1.bak");
        std::fs::copy(path, &backup_path).map_err(|e| {
            anyhow::anyhow!(
                "failed to backup config before migration: {} -> {}: {}",
                path.display(),
                backup_path.display(),
                e
            )
        })?;

        // 5. Write migrated config
        let new_content = toml::to_string_pretty(&table)?;
        std::fs::write(path, &new_content)?;

        eprintln!("  Config migrated from 0.1.x to 0.2.0 format.");
        eprintln!("  Backup saved: {}", backup_path.display());

        Ok(true)
    }

    /// Resolve the effective server list.
    ///
    /// If `[[servers]]` is present, use it. Otherwise fall back to the
    /// top-level `aether_url` + `management_token` as a single server.
    pub fn effective_servers(&self) -> Vec<ServerEntry> {
        if !self.servers.is_empty() {
            return self.servers.clone();
        }
        match (&self.aether_url, &self.management_token) {
            (Some(url), Some(token)) => vec![ServerEntry {
                aether_url: url.clone(),
                management_token: token.clone(),
                node_name: None,
            }],
            _ => vec![],
        }
    }

    /// Inject values as environment variables so clap picks them up.
    ///
    /// Only sets variables that are **not** already present in the
    /// environment, preserving the precedence: CLI > env > config file.
    pub fn inject_env(&self) {
        self.inject_env_inner(false);
    }

    /// Inject values as environment variables, **overriding** any existing
    /// values.  Used after setup to ensure the freshly-saved config takes
    /// effect before re-parsing.
    pub fn inject_env_override(&self) {
        self.inject_env_inner(true);
    }

    fn inject_env_inner(&self, force: bool) {
        macro_rules! set {
            ($env:expr, $val:expr) => {
                if let Some(ref v) = $val {
                    if force || std::env::var($env).is_err() {
                        std::env::set_var($env, v.to_string());
                    }
                }
            };
        }

        // When top-level fields are absent, fall back to the first [[servers]]
        // entry so that clap's required `aether_url` / `management_token` are
        // satisfied even with the new config format.
        let first_server = self.servers.first();
        let aether_url = self
            .aether_url
            .as_deref()
            .or(first_server.map(|s| s.aether_url.as_str()));
        let management_token = self
            .management_token
            .as_deref()
            .or(first_server.map(|s| s.management_token.as_str()));
        let node_name = self
            .node_name
            .as_deref()
            .or(first_server.and_then(|s| s.node_name.as_deref()));

        set!("AETHER_PROXY_AETHER_URL", aether_url);
        set!("AETHER_PROXY_MANAGEMENT_TOKEN", management_token);
        set!("AETHER_PROXY_PUBLIC_IP", self.public_ip);
        set!("AETHER_PROXY_NODE_NAME", node_name);
        set!("AETHER_PROXY_NODE_REGION", self.node_region);
        set!("AETHER_PROXY_HEARTBEAT_INTERVAL", self.heartbeat_interval);
        set!(
            "AETHER_PROXY_AETHER_REQUEST_TIMEOUT",
            self.aether_request_timeout_secs
        );
        set!(
            "AETHER_PROXY_AETHER_CONNECT_TIMEOUT",
            self.aether_connect_timeout_secs
        );
        set!(
            "AETHER_PROXY_AETHER_POOL_MAX_IDLE_PER_HOST",
            self.aether_pool_max_idle_per_host
        );
        set!(
            "AETHER_PROXY_AETHER_POOL_IDLE_TIMEOUT",
            self.aether_pool_idle_timeout_secs
        );
        set!(
            "AETHER_PROXY_AETHER_TCP_KEEPALIVE",
            self.aether_tcp_keepalive_secs
        );
        set!("AETHER_PROXY_AETHER_TCP_NODELAY", self.aether_tcp_nodelay);
        set!("AETHER_PROXY_AETHER_HTTP2", self.aether_http2);
        set!(
            "AETHER_PROXY_AETHER_RETRY_MAX_ATTEMPTS",
            self.aether_retry_max_attempts
        );
        set!(
            "AETHER_PROXY_AETHER_RETRY_BASE_DELAY_MS",
            self.aether_retry_base_delay_ms
        );
        set!(
            "AETHER_PROXY_AETHER_RETRY_MAX_DELAY_MS",
            self.aether_retry_max_delay_ms
        );
        set!(
            "AETHER_PROXY_MAX_CONCURRENT_CONNECTIONS",
            self.max_concurrent_connections
        );
        set!("AETHER_PROXY_DNS_CACHE_TTL", self.dns_cache_ttl_secs);
        set!("AETHER_PROXY_DNS_CACHE_CAPACITY", self.dns_cache_capacity);
        set!(
            "AETHER_PROXY_UPSTREAM_CONNECT_TIMEOUT",
            self.upstream_connect_timeout_secs
        );
        set!(
            "AETHER_PROXY_UPSTREAM_POOL_MAX_IDLE_PER_HOST",
            self.upstream_pool_max_idle_per_host
        );
        set!(
            "AETHER_PROXY_UPSTREAM_POOL_IDLE_TIMEOUT",
            self.upstream_pool_idle_timeout_secs
        );
        set!(
            "AETHER_PROXY_UPSTREAM_TCP_KEEPALIVE",
            self.upstream_tcp_keepalive_secs
        );
        set!(
            "AETHER_PROXY_UPSTREAM_TCP_NODELAY",
            self.upstream_tcp_nodelay
        );
        set!("AETHER_PROXY_LOG_LEVEL", self.log_level);
        set!("AETHER_PROXY_LOG_JSON", self.log_json);
        set!(
            "AETHER_PROXY_TUNNEL_RECONNECT_BASE_MS",
            self.tunnel_reconnect_base_ms
        );
        set!(
            "AETHER_PROXY_TUNNEL_RECONNECT_MAX_MS",
            self.tunnel_reconnect_max_ms
        );
        set!(
            "AETHER_PROXY_TUNNEL_PING_INTERVAL",
            self.tunnel_ping_interval_secs
        );
        set!("AETHER_PROXY_TUNNEL_MAX_STREAMS", self.tunnel_max_streams);
        set!(
            "AETHER_PROXY_TUNNEL_CONNECT_TIMEOUT",
            self.tunnel_connect_timeout_secs
        );
        set!(
            "AETHER_PROXY_TUNNEL_TCP_KEEPALIVE",
            self.tunnel_tcp_keepalive_secs
        );
        set!("AETHER_PROXY_TUNNEL_TCP_NODELAY", self.tunnel_tcp_nodelay);
        set!(
            "AETHER_PROXY_TUNNEL_STALE_TIMEOUT",
            self.tunnel_stale_timeout_secs
        );
        set!("AETHER_PROXY_TUNNEL_CONNECTIONS", self.tunnel_connections);

        // allowed_ports needs special handling (comma-separated)
        if let Some(ref ports) = self.allowed_ports {
            if force || std::env::var("AETHER_PROXY_ALLOWED_PORTS").is_err() {
                let s: String = ports
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                std::env::set_var("AETHER_PROXY_ALLOWED_PORTS", s);
            }
        }
    }
}
