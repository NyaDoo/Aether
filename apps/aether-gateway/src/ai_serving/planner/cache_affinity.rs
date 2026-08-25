use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::ai_serving::{extract_pool_sticky_session_token, ExecutionRuntimeAuthContext};
use crate::AppState;

const FEATURE_KEY: &str = "prompt_cache_affinity";

/// feature_settings 中 `prompt_cache_affinity` 的解析结果。
///
/// 配置形如：
/// ```json
/// {
///   "prompt_cache_affinity": {
///     "enabled": true,
///     "models": ["glm-5.3"],
///     "profiles": [
///       {"id": "A", "version": 1,
///        "match_first_user_sha256": "<64位hex>",
///        "cache_key": "databao:glm-5.3:A:1"}
///     ]
///   }
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub(crate) struct PromptCacheAffinitySettings {
    enabled: Option<bool>,
    models: Option<Vec<String>>,
    profiles: Vec<PromptCacheAffinityProfile>,
    auto: Option<AutoAffinitySettings>,
}

/// auto 模式：对首条 user 消息的稳定前缀实时哈希生成 key，
/// 租户换 prompt 后无需更新任何配置。
#[derive(Clone, Copy, Debug)]
struct AutoAffinitySettings {
    /// 首条 user 消息至少这么多字符才值得钉路由。
    min_chars: usize,
    /// 参与哈希的前缀长度；尾部动态内容不影响 key 稳定性。
    prefix_chars: usize,
}

const AUTO_MIN_CHARS_DEFAULT: usize = 4096;
const AUTO_PREFIX_CHARS_DEFAULT: usize = 16384;

#[derive(Clone, Debug)]
struct PromptCacheAffinityProfile {
    id: String,
    version: i64,
    match_first_user_sha256: String,
    cache_key: String,
}

impl PromptCacheAffinitySettings {
    fn merge_from_value(&mut self, value: Option<&Value>) {
        let Some(settings) = value
            .and_then(Value::as_object)
            .and_then(|features| features.get(FEATURE_KEY))
            .and_then(Value::as_object)
        else {
            return;
        };
        if let Some(enabled) = settings.get("enabled").and_then(Value::as_bool) {
            self.enabled = Some(enabled);
        }
        if let Some(models) = settings.get("models").and_then(Value::as_array) {
            self.models = Some(
                models
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|model| !model.is_empty())
                    .map(str::to_ascii_lowercase)
                    .collect(),
            );
        }
        if let Some(profiles) = settings.get("profiles").and_then(Value::as_array) {
            self.profiles = profiles
                .iter()
                .filter_map(parse_affinity_profile)
                .collect();
        }
        match settings.get("auto") {
            Some(Value::Object(auto)) => {
                if auto.get("enabled").and_then(Value::as_bool) == Some(false) {
                    self.auto = None;
                } else {
                    self.auto = Some(AutoAffinitySettings {
                        min_chars: read_usize(auto.get("min_chars"), AUTO_MIN_CHARS_DEFAULT),
                        prefix_chars: read_usize(auto.get("prefix_chars"), AUTO_PREFIX_CHARS_DEFAULT)
                            .max(1),
                    });
                }
            }
            Some(Value::Bool(true)) => {
                self.auto = Some(AutoAffinitySettings {
                    min_chars: AUTO_MIN_CHARS_DEFAULT,
                    prefix_chars: AUTO_PREFIX_CHARS_DEFAULT,
                });
            }
            Some(Value::Bool(false)) | Some(Value::Null) => {
                self.auto = None;
            }
            _ => {}
        }
    }

    fn effective_enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }

    /// 为缺 `prompt_cache_key` 的请求解析稳定亲和 key。
    ///
    /// 客户端已带非空 key、功能未启用、模型不在允许表、首条 user 消息
    /// 内容哈希未命中任何 profile 时都返回 `None`（完全透传）。
    pub(crate) fn resolve_affinity_cache_key(
        &self,
        requested_model: &str,
        body_json: &Value,
    ) -> Option<String> {
        if !self.effective_enabled() || (self.profiles.is_empty() && self.auto.is_none()) {
            return None;
        }
        if body_has_prompt_cache_key(body_json) {
            return None;
        }
        if let Some(models) = self.models.as_ref() {
            let requested = requested_model.trim().to_ascii_lowercase();
            if !models.iter().any(|model| model == &requested) {
                return None;
            }
        }
        let content = first_user_message_text(body_json)?;
        let digest = hex_sha256(content);
        if let Some(profile) = self
            .profiles
            .iter()
            .find(|profile| profile.match_first_user_sha256.eq_ignore_ascii_case(&digest))
        {
            debug!(
                profile_id = %profile.id,
                profile_version = profile.version,
                "prompt cache affinity profile matched"
            );
            return Some(profile.cache_key.clone());
        }
        let auto = self.auto?;
        if content.chars().count() < auto.min_chars {
            return None;
        }
        let prefix_digest = hex_sha256(utf8_char_prefix(content, auto.prefix_chars));
        debug!(
            key_fingerprint = %&prefix_digest[..16],
            "prompt cache affinity auto key derived"
        );
        Some(format!("aff:{}", &prefix_digest[..32]))
    }
}

fn parse_affinity_profile(value: &Value) -> Option<PromptCacheAffinityProfile> {
    let profile = value.as_object()?;
    let id = non_empty_trimmed(profile.get("id"))?;
    let match_first_user_sha256 = non_empty_trimmed(profile.get("match_first_user_sha256"))?;
    let cache_key = non_empty_trimmed(profile.get("cache_key"))?;
    if match_first_user_sha256.len() != 64
        || !match_first_user_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        warn!(profile_id = %id, "prompt cache affinity profile has invalid match_first_user_sha256");
        return None;
    }
    Some(PromptCacheAffinityProfile {
        id,
        version: profile.get("version").and_then(Value::as_i64).unwrap_or(1),
        match_first_user_sha256,
        cache_key,
    })
}

fn read_usize(value: Option<&Value>, default: usize) -> usize {
    value
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
}

/// 按字符数截取前缀，避免在多字节 UTF-8 序列中间切断。
fn utf8_char_prefix(content: &str, max_chars: usize) -> &str {
    match content.char_indices().nth(max_chars) {
        Some((byte_index, _)) => &content[..byte_index],
        None => content,
    }
}

fn non_empty_trimmed(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn body_has_prompt_cache_key(body_json: &Value) -> bool {
    match body_json.get("prompt_cache_key") {
        None | Some(Value::Null) => false,
        Some(Value::String(value)) => !value.trim().is_empty(),
        // 非字符串值交给既有契约校验处理，亲和注入不与之竞争。
        Some(_) => true,
    }
}

fn first_user_message_text(body_json: &Value) -> Option<&str> {
    body_json
        .get("messages")?
        .as_array()?
        .iter()
        .find(|message| {
            message
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|role| role.eq_ignore_ascii_case("user"))
        })?
        .get("content")?
        .as_str()
}

fn hex_sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// 读取 user 与 api key 两级 feature_settings 并合并（key 覆盖 user）。
/// 读取失败按功能关闭处理：缓存亲和是优化，不能因此拒绝请求。
pub(crate) async fn resolve_prompt_cache_affinity_settings(
    state: &AppState,
    auth_context: &ExecutionRuntimeAuthContext,
) -> PromptCacheAffinitySettings {
    let user_settings_fut = state.read_user_feature_settings(&auth_context.user_id);
    let key_settings_fut = state.read_auth_api_key_feature_settings(
        &auth_context.user_id,
        &auth_context.api_key_id,
        auth_context.api_key_is_standalone,
    );
    match tokio::try_join!(user_settings_fut, key_settings_fut) {
        Ok((user_settings, key_settings)) => {
            let mut settings = PromptCacheAffinitySettings::default();
            settings.merge_from_value(user_settings.as_ref());
            settings.merge_from_value(key_settings.as_ref());
            settings
        }
        Err(err) => {
            warn!(
                error = ?err,
                "gateway failed to read prompt cache affinity feature settings"
            );
            PromptCacheAffinitySettings::default()
        }
    }
}

/// 池粘性 token 解析：优先客户端显式信号，缺失时回退到亲和 key，
/// 让同 profile 的请求钉在同一上游账号上。
pub(crate) async fn resolve_chat_pool_sticky_session_token(
    state: &AppState,
    auth_context: &ExecutionRuntimeAuthContext,
    requested_model: &str,
    body_json: &Value,
) -> Option<String> {
    if let Some(token) = extract_pool_sticky_session_token(body_json) {
        return Some(token);
    }
    resolve_prompt_cache_affinity_settings(state, auth_context)
        .await
        .resolve_affinity_cache_key(requested_model, body_json)
}

/// 向上游请求体补写缺失的 `prompt_cache_key`。
/// 仅对 openai:chat 上游格式生效；其它格式没有该字段契约。
pub(crate) async fn inject_provider_prompt_cache_affinity_key(
    state: &AppState,
    auth_context: &ExecutionRuntimeAuthContext,
    requested_model: &str,
    provider_api_format: &str,
    provider_request_body: &mut Value,
) {
    if !provider_api_format.trim().eq_ignore_ascii_case("openai:chat") {
        return;
    }
    let settings = resolve_prompt_cache_affinity_settings(state, auth_context).await;
    let Some(cache_key) = settings.resolve_affinity_cache_key(requested_model, provider_request_body)
    else {
        return;
    };
    if let Some(object) = provider_request_body.as_object_mut() {
        object.insert("prompt_cache_key".to_string(), Value::String(cache_key));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{hex_sha256, PromptCacheAffinitySettings};

    fn settings_value(enabled: bool, prompt: &str) -> serde_json::Value {
        json!({
            "prompt_cache_affinity": {
                "enabled": enabled,
                "models": ["glm-5.3"],
                "profiles": [
                    {
                        "id": "A",
                        "version": 2,
                        "match_first_user_sha256": hex_sha256(prompt),
                        "cache_key": "databao:glm-5.3:A:2"
                    }
                ]
            }
        })
    }

    fn chat_body(prompt: &str) -> serde_json::Value {
        json!({
            "model": "glm-5.3",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": prompt}
            ]
        })
    }

    fn resolved(settings_json: &serde_json::Value, model: &str, body: &serde_json::Value) -> Option<String> {
        let mut settings = PromptCacheAffinitySettings::default();
        settings.merge_from_value(Some(settings_json));
        settings.resolve_affinity_cache_key(model, body)
    }

    #[test]
    fn matches_profile_by_first_user_message_hash() {
        assert_eq!(
            resolved(&settings_value(true, "big prompt"), "glm-5.3", &chat_body("big prompt")),
            Some("databao:glm-5.3:A:2".to_string())
        );
    }

    #[test]
    fn disabled_or_unconfigured_is_passthrough() {
        assert_eq!(
            resolved(&settings_value(false, "big prompt"), "glm-5.3", &chat_body("big prompt")),
            None
        );
        let empty = PromptCacheAffinitySettings::default();
        assert_eq!(empty.resolve_affinity_cache_key("glm-5.3", &chat_body("big prompt")), None);
    }

    #[test]
    fn unmatched_prompt_or_model_is_passthrough() {
        assert_eq!(
            resolved(&settings_value(true, "big prompt"), "glm-5.3", &chat_body("other prompt")),
            None
        );
        assert_eq!(
            resolved(&settings_value(true, "big prompt"), "gpt-4o", &chat_body("big prompt")),
            None
        );
    }

    #[test]
    fn never_overrides_client_prompt_cache_key() {
        let mut body = chat_body("big prompt");
        body["prompt_cache_key"] = json!("client-key");
        assert_eq!(resolved(&settings_value(true, "big prompt"), "glm-5.3", &body), None);
    }

    #[test]
    fn api_key_settings_override_user_settings() {
        let mut settings = PromptCacheAffinitySettings::default();
        settings.merge_from_value(Some(&settings_value(true, "big prompt")));
        settings.merge_from_value(Some(&json!({
            "prompt_cache_affinity": {"enabled": false}
        })));
        assert_eq!(
            settings.resolve_affinity_cache_key("glm-5.3", &chat_body("big prompt")),
            None
        );
    }

    #[test]
    fn non_string_first_user_content_is_passthrough() {
        let body = json!({
            "model": "glm-5.3",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "big prompt"}]}]
        });
        assert_eq!(resolved(&settings_value(true, "big prompt"), "glm-5.3", &body), None);
    }

    fn auto_settings_value(min_chars: usize) -> serde_json::Value {
        json!({
            "prompt_cache_affinity": {
                "enabled": true,
                "models": ["glm-5.3"],
                "auto": {"min_chars": min_chars}
            }
        })
    }

    #[test]
    fn auto_mode_derives_stable_key_for_big_prompt() {
        let prompt = "稳".repeat(5000);
        let first = resolved(&auto_settings_value(4096), "glm-5.3", &chat_body(&prompt));
        let second = resolved(&auto_settings_value(4096), "glm-5.3", &chat_body(&prompt));
        assert!(first.as_deref().is_some_and(|key| key.starts_with("aff:")));
        assert_eq!(first, second);
        let other = resolved(
            &auto_settings_value(4096),
            "glm-5.3",
            &chat_body(&"异".repeat(5000)),
        );
        assert!(other.is_some());
        assert_ne!(first, other);
    }

    #[test]
    fn auto_mode_ignores_dynamic_tail_beyond_prefix_chars() {
        let stable_prefix = "前".repeat(20000);
        let first = resolved(
            &auto_settings_value(4096),
            "glm-5.3",
            &chat_body(&format!("{stable_prefix}动态提问一")),
        );
        let second = resolved(
            &auto_settings_value(4096),
            "glm-5.3",
            &chat_body(&format!("{stable_prefix}完全不同的动态提问二")),
        );
        assert!(first.is_some());
        assert_eq!(first, second);
    }

    #[test]
    fn auto_mode_skips_small_prompts() {
        assert_eq!(
            resolved(&auto_settings_value(4096), "glm-5.3", &chat_body("短消息")),
            None
        );
    }

    #[test]
    fn profiles_take_precedence_over_auto() {
        let prompt = "长".repeat(5000);
        let mut settings_json = settings_value(true, &prompt);
        settings_json["prompt_cache_affinity"]["auto"] = json!({"min_chars": 1});
        assert_eq!(
            resolved(&settings_json, "glm-5.3", &chat_body(&prompt)),
            Some("databao:glm-5.3:A:2".to_string())
        );
    }

    #[test]
    fn invalid_profile_hash_is_rejected_at_parse() {
        let mut settings = PromptCacheAffinitySettings::default();
        settings.merge_from_value(Some(&json!({
            "prompt_cache_affinity": {
                "enabled": true,
                "profiles": [
                    {"id": "A", "match_first_user_sha256": "not-hex", "cache_key": "k"}
                ]
            }
        })));
        assert_eq!(
            settings.resolve_affinity_cache_key("glm-5.3", &chat_body("big prompt")),
            None
        );
    }
}
