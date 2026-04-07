pub(crate) fn module_available_from_env(env_key: &str, default_available: bool) -> bool {
    match std::env::var(env_key) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "true" | "1" | "yes"
        ),
        Err(_) => default_available,
    }
}

pub(crate) fn system_config_bool(value: Option<&serde_json::Value>, default: bool) -> bool {
    match value {
        Some(serde_json::Value::Bool(value)) => *value,
        Some(serde_json::Value::Number(value)) => {
            value.as_i64().map(|value| value != 0).unwrap_or(default)
        }
        Some(serde_json::Value::String(value)) => {
            match value.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => true,
                "false" | "0" | "no" | "off" => false,
                _ => default,
            }
        }
        _ => default,
    }
}

pub(crate) fn system_config_string(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(value)) => {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        }
        _ => None,
    }
}
