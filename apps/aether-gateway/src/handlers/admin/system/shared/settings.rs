use crate::handlers::shared::{system_config_bool, system_config_string};
use crate::{AppState, GatewayError};
use axum::body::Bytes;
use axum::http;
use serde_json::json;
use std::fs;

#[derive(Debug, Clone, Copy)]
struct AdminApiFormatDefinition {
    value: &'static str,
    label: &'static str,
    default_path: &'static str,
    aliases: &'static [&'static str],
}

const ADMIN_API_FORMAT_DEFINITIONS: &[AdminApiFormatDefinition] = &[
    AdminApiFormatDefinition {
        value: "openai:chat",
        label: "OpenAI Chat",
        default_path: "/v1/chat/completions",
        aliases: &[
            "openai",
            "openai_compatible",
            "deepseek",
            "grok",
            "moonshot",
            "zhipu",
            "qwen",
            "baichuan",
            "minimax",
        ],
    },
    AdminApiFormatDefinition {
        value: "openai:cli",
        label: "OpenAI CLI",
        default_path: "/v1/responses",
        aliases: &["openai_cli", "responses"],
    },
    AdminApiFormatDefinition {
        value: "openai:compact",
        label: "OpenAI Compact",
        default_path: "/v1/responses/compact",
        aliases: &["openai_compact", "responses_compact"],
    },
    AdminApiFormatDefinition {
        value: "openai:video",
        label: "OpenAI Video",
        default_path: "/v1/videos",
        aliases: &["openai_video", "sora"],
    },
    AdminApiFormatDefinition {
        value: "claude:chat",
        label: "Claude Chat",
        default_path: "/v1/messages",
        aliases: &["claude", "anthropic", "claude_compatible"],
    },
    AdminApiFormatDefinition {
        value: "claude:cli",
        label: "Claude CLI",
        default_path: "/v1/messages",
        aliases: &["claude_cli", "claude-cli"],
    },
    AdminApiFormatDefinition {
        value: "gemini:chat",
        label: "Gemini Chat",
        default_path: "/v1beta/models/{model}:{action}",
        aliases: &["gemini", "google", "vertex"],
    },
    AdminApiFormatDefinition {
        value: "gemini:cli",
        label: "Gemini CLI",
        default_path: "/v1beta/models/{model}:{action}",
        aliases: &["gemini_cli", "gemini-cli"],
    },
    AdminApiFormatDefinition {
        value: "gemini:video",
        label: "Gemini Video",
        default_path: "/v1beta/models/{model}:predictLongRunning",
        aliases: &["gemini_video", "veo"],
    },
];

pub(crate) fn current_aether_version() -> String {
    let version_file =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/_version.py");
    if let Ok(contents) = fs::read_to_string(version_file) {
        for line in contents.lines() {
            let line = line.trim();
            if let Some(version) = line
                .strip_prefix("__version__ = version = '")
                .and_then(|value| value.strip_suffix('\''))
            {
                if !version.is_empty() {
                    return version.to_string();
                }
            }
        }
    }

    env!("CARGO_PKG_VERSION").to_string()
}

pub(crate) fn build_admin_system_check_update_payload() -> serde_json::Value {
    json!({
        "current_version": current_aether_version(),
        "latest_version": serde_json::Value::Null,
        "has_update": false,
        "release_url": serde_json::Value::Null,
        "release_notes": serde_json::Value::Null,
        "published_at": serde_json::Value::Null,
        "error": "检查更新需要 Rust 管理后端",
    })
}

pub(crate) async fn build_admin_system_stats_payload(
    state: &AppState,
) -> Result<serde_json::Value, GatewayError> {
    let providers = state
        .list_provider_catalog_providers(false)
        .await
        .unwrap_or_default();
    let total_providers = providers.len() as u64;
    let active_providers = providers
        .iter()
        .filter(|provider| provider.is_active)
        .count() as u64;
    let stats = state.read_admin_system_stats().await?;

    Ok(json!({
        "users": {
            "total": stats.total_users,
            "active": stats.active_users,
        },
        "providers": {
            "total": total_providers,
            "active": active_providers,
        },
        "api_keys": stats.total_api_keys,
        "requests": stats.total_requests,
    }))
}

pub(crate) async fn build_admin_system_settings_payload(
    state: &AppState,
) -> Result<serde_json::Value, GatewayError> {
    let default_provider_config = state
        .read_system_config_json_value("default_provider")
        .await?;
    let default_model_config = state.read_system_config_json_value("default_model").await?;
    let enable_usage_tracking_config = state
        .read_system_config_json_value("enable_usage_tracking")
        .await?;
    let password_policy_level_config = state
        .read_system_config_json_value("password_policy_level")
        .await?;

    let default_provider = match system_config_string(default_provider_config.as_ref()) {
        Some(value) => Some(value),
        None => state
            .list_provider_catalog_providers(false)
            .await
            .ok()
            .unwrap_or_default()
            .into_iter()
            .find(|provider| provider.is_active)
            .map(|provider| provider.name),
    };
    let default_model = system_config_string(default_model_config.as_ref());
    let enable_usage_tracking = system_config_bool(enable_usage_tracking_config.as_ref(), true);
    let password_policy_level = match system_config_string(password_policy_level_config.as_ref()) {
        Some(value) if matches!(value.as_str(), "weak" | "medium" | "strong") => value,
        _ => "weak".to_string(),
    };

    Ok(json!({
        "default_provider": default_provider,
        "default_model": default_model,
        "enable_usage_tracking": enable_usage_tracking,
        "password_policy_level": password_policy_level,
    }))
}

pub(crate) async fn apply_admin_system_settings_update(
    state: &AppState,
    request_body: &Bytes,
) -> Result<Result<serde_json::Value, (http::StatusCode, serde_json::Value)>, GatewayError> {
    let payload = match serde_json::from_slice::<serde_json::Value>(request_body) {
        Ok(serde_json::Value::Object(payload)) => payload,
        Ok(_) | Err(_) => {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    };

    if let Some(default_provider) = payload.get("default_provider") {
        if let Some(default_provider) = default_provider.as_str() {
            let default_provider = default_provider.trim();
            if default_provider.is_empty() {
                let _ = state
                    .upsert_system_config_json_value(
                        "default_provider",
                        &serde_json::Value::Null,
                        None,
                    )
                    .await?;
            } else {
                let provider_exists = state
                    .list_provider_catalog_providers(false)
                    .await
                    .ok()
                    .unwrap_or_default()
                    .into_iter()
                    .any(|provider| provider.is_active && provider.name == default_provider);
                if !provider_exists {
                    return Ok(Err((
                        http::StatusCode::BAD_REQUEST,
                        json!({ "detail": format!("提供商 '{default_provider}' 不存在或未启用") }),
                    )));
                }
                let _ = state
                    .upsert_system_config_json_value(
                        "default_provider",
                        &json!(default_provider),
                        Some("系统默认提供商，当用户未设置个人提供商时使用"),
                    )
                    .await?;
            }
        } else if !default_provider.is_null() {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    }

    if let Some(default_model) = payload.get("default_model") {
        if let Some(default_model) = default_model.as_str() {
            let value = default_model.trim();
            let config_value = if value.is_empty() {
                serde_json::Value::Null
            } else {
                json!(value)
            };
            let _ = state
                .upsert_system_config_json_value("default_model", &config_value, None)
                .await?;
        } else if !default_model.is_null() {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    }

    if let Some(enable_usage_tracking) = payload.get("enable_usage_tracking") {
        if let Some(enable_usage_tracking) = enable_usage_tracking.as_bool() {
            let _ = state
                .upsert_system_config_json_value(
                    "enable_usage_tracking",
                    &json!(enable_usage_tracking),
                    None,
                )
                .await?;
        } else if !enable_usage_tracking.is_null() {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    }

    if let Some(password_policy_level) = payload.get("password_policy_level") {
        if let Some(password_policy_level) = password_policy_level.as_str() {
            if !matches!(password_policy_level.trim(), "weak" | "medium" | "strong") {
                return Ok(Err((
                    http::StatusCode::BAD_REQUEST,
                    json!({ "detail": "请求数据验证失败" }),
                )));
            }
            let _ = state
                .upsert_system_config_json_value(
                    "password_policy_level",
                    &json!(password_policy_level.trim()),
                    None,
                )
                .await?;
        } else if !password_policy_level.is_null() {
            return Ok(Err((
                http::StatusCode::BAD_REQUEST,
                json!({ "detail": "请求数据验证失败" }),
            )));
        }
    }

    Ok(Ok(json!({ "message": "系统设置更新成功" })))
}

pub(crate) fn build_admin_api_formats_payload() -> serde_json::Value {
    json!({
        "formats": ADMIN_API_FORMAT_DEFINITIONS
            .iter()
            .map(|definition| json!({
                "value": definition.value,
                "label": definition.label,
                "default_path": definition.default_path,
                "aliases": definition.aliases,
            }))
            .collect::<Vec<_>>(),
    })
}
