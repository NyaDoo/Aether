pub(crate) fn normalized_signature(api_format: &str) -> Option<&'static str> {
    match crate::ai_serving::normalize_api_format_alias(api_format).as_str() {
        "doubao:embedding" => Some("doubao:embedding"),
        "doubao:video" => Some("doubao:video"),
        "doubao:asset_library" => Some("doubao:asset_library"),
        _ => None,
    }
}

pub(crate) fn local_path(api_format: &str) -> Option<&'static str> {
    match crate::ai_serving::normalize_api_format_alias(api_format).as_str() {
        "doubao:embedding" => Some("/v1/embeddings"),
        "doubao:video" => Some("/v3/contents/generations/tasks"),
        "doubao:asset_library" => Some("/?Action={action}&Version=2024-01-01"),
        _ => None,
    }
}
