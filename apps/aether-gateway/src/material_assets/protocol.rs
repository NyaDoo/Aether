use http::Uri;
use serde_json::{Map, Value};

pub(crate) const ARK_ASSET_API_FORMAT: &str = "doubao:asset_library";
pub(crate) const ARK_ASSET_REQUIRED_CAPABILITY: &str = "ark_asset_library";
pub(crate) const ARK_ASSET_VERSION: &str = "2024-01-01";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArkAssetAction {
    CreateAssetGroup,
    ListAssetGroups,
    GetAssetGroup,
    UpdateAssetGroup,
    DeleteAssetGroup,
    CreateAsset,
    ListAssets,
    GetAsset,
    UpdateAsset,
    DeleteAsset,
    CreateVisualValidateSession,
    GetVisualValidateResult,
}

impl ArkAssetAction {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::CreateAssetGroup => "CreateAssetGroup",
            Self::ListAssetGroups => "ListAssetGroups",
            Self::GetAssetGroup => "GetAssetGroup",
            Self::UpdateAssetGroup => "UpdateAssetGroup",
            Self::DeleteAssetGroup => "DeleteAssetGroup",
            Self::CreateAsset => "CreateAsset",
            Self::ListAssets => "ListAssets",
            Self::GetAsset => "GetAsset",
            Self::UpdateAsset => "UpdateAsset",
            Self::DeleteAsset => "DeleteAsset",
            Self::CreateVisualValidateSession => "CreateVisualValidateSession",
            Self::GetVisualValidateResult => "GetVisualValidateResult",
        }
    }

    pub(crate) fn from_str(value: &str) -> Option<Self> {
        Some(match value.trim() {
            "CreateAssetGroup" => Self::CreateAssetGroup,
            "ListAssetGroups" => Self::ListAssetGroups,
            "GetAssetGroup" => Self::GetAssetGroup,
            "UpdateAssetGroup" => Self::UpdateAssetGroup,
            "DeleteAssetGroup" => Self::DeleteAssetGroup,
            "CreateAsset" => Self::CreateAsset,
            "ListAssets" => Self::ListAssets,
            "GetAsset" => Self::GetAsset,
            "UpdateAsset" => Self::UpdateAsset,
            "DeleteAsset" => Self::DeleteAsset,
            "CreateVisualValidateSession" => Self::CreateVisualValidateSession,
            "GetVisualValidateResult" => Self::GetVisualValidateResult,
            _ => return None,
        })
    }

    pub(crate) fn is_group(self) -> bool {
        matches!(
            self,
            Self::CreateAssetGroup
                | Self::ListAssetGroups
                | Self::GetAssetGroup
                | Self::UpdateAssetGroup
                | Self::DeleteAssetGroup
        )
    }

    pub(crate) fn is_asset(self) -> bool {
        matches!(
            self,
            Self::CreateAsset
                | Self::ListAssets
                | Self::GetAsset
                | Self::UpdateAsset
                | Self::DeleteAsset
        )
    }

    pub(crate) fn is_verification(self) -> bool {
        matches!(
            self,
            Self::CreateVisualValidateSession | Self::GetVisualValidateResult
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ArkAssetProtocolError {
    MissingAction,
    UnsupportedAction(String),
    InvalidVersion,
    InvalidBody(String),
}

impl std::fmt::Display for ArkAssetProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAction => f.write_str("missing Action query parameter"),
            Self::UnsupportedAction(action) => write!(f, "unsupported asset Action: {action}"),
            Self::InvalidVersion => write!(f, "Version must be {ARK_ASSET_VERSION}"),
            Self::InvalidBody(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ArkAssetProtocolError {}

/// Recognises both the native Action root and the explicit Aether alias.
pub(crate) fn is_asset_library_path(path: &str) -> bool {
    matches!(path, "/" | "/v3/asset-library" | "/v3/asset-library/")
        || path.starts_with("/v3/asset-library/")
}

pub(crate) fn action_from_request(
    uri: &Uri,
    body: &Value,
) -> Result<ArkAssetAction, ArkAssetProtocolError> {
    let query = uri.query().unwrap_or_default();
    let mut query_action = None;
    let mut query_version = None;
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        match key.as_ref() {
            "Action" | "action" => query_action = Some(value.into_owned()),
            "Version" | "version" => query_version = Some(value.into_owned()),
            _ => {}
        }
    }

    let body_action = body
        .get("Action")
        .or_else(|| body.get("action"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let body_version = body
        .get("Version")
        .or_else(|| body.get("version"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    for version in [query_version.as_deref(), body_version.as_deref()]
        .into_iter()
        .flatten()
    {
        if version != ARK_ASSET_VERSION {
            return Err(ArkAssetProtocolError::InvalidVersion);
        }
    }
    if uri.path() == "/" && query_version.as_deref() != Some(ARK_ASSET_VERSION) {
        return Err(ArkAssetProtocolError::InvalidVersion);
    }

    let path_action = uri
        .path()
        .strip_prefix("/v3/asset-library/")
        .map(str::trim)
        .filter(|action| !action.is_empty())
        .map(ToOwned::to_owned);
    let actions = [
        query_action.as_deref(),
        body_action.as_deref(),
        path_action.as_deref(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if let Some(first) = actions.first() {
        if actions.iter().any(|action| action != first) {
            return Err(ArkAssetProtocolError::InvalidBody(
                "conflicting Action values".to_string(),
            ));
        }
    }
    let action = actions
        .first()
        .copied()
        .ok_or(ArkAssetProtocolError::MissingAction)?;

    ArkAssetAction::from_str(action)
        .ok_or_else(|| ArkAssetProtocolError::UnsupportedAction(action.to_string()))
}

/// Removes transport-only envelope keys before forwarding a native request.
/// The Action and Version query parameters remain authoritative and are not
/// copied into the provider JSON body.
pub(crate) fn sanitize_action_body(body: &Value) -> Result<Value, ArkAssetProtocolError> {
    let Some(object) = body.as_object() else {
        return Err(ArkAssetProtocolError::InvalidBody(
            "asset Action body must be a JSON object".to_string(),
        ));
    };
    let mut sanitized = Map::new();
    for (key, value) in object {
        if matches!(key.as_str(), "Action" | "Version" | "action" | "version") {
            continue;
        }
        sanitized.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(sanitized))
}

/// Canonicalises provider payload fields shared by the Volcengine endpoint and
/// compatible relays. Single-resource operations use `Id`; legacy Aether
/// aliases are accepted at the public boundary but are never sent upstream.
pub(crate) fn canonicalize_provider_body(
    action: ArkAssetAction,
    body: &Value,
) -> Result<Value, ArkAssetProtocolError> {
    let Some(source) = body.as_object() else {
        return Err(ArkAssetProtocolError::InvalidBody(
            "asset Action body must be a JSON object".to_string(),
        ));
    };
    let mut canonical = source.clone();
    match action {
        ArkAssetAction::CreateAssetGroup => {
            canonicalize_create_group_type(&mut canonical)?;
        }
        ArkAssetAction::GetAssetGroup
        | ArkAssetAction::UpdateAssetGroup
        | ArkAssetAction::DeleteAssetGroup => {
            canonicalize_string_alias(&mut canonical, "Id", &["GroupId", "group_id", "id"]);
        }
        ArkAssetAction::GetAsset | ArkAssetAction::UpdateAsset | ArkAssetAction::DeleteAsset => {
            canonicalize_string_alias(&mut canonical, "Id", &["AssetId", "asset_id", "id"]);
        }
        ArkAssetAction::CreateVisualValidateSession => {
            canonicalize_string_alias(
                &mut canonical,
                "CallbackURL",
                &["callback_url", "ReturnUrl", "return_url"],
            );
        }
        _ => {}
    }
    canonical.retain(|name, value| {
        !value.is_null()
            && !name.eq_ignore_ascii_case("ProjectName")
            && !name.eq_ignore_ascii_case("project_name")
    });
    Ok(Value::Object(canonical))
}

fn canonicalize_create_group_type(
    object: &mut Map<String, Value>,
) -> Result<(), ArkAssetProtocolError> {
    let value = ["GroupType", "Type", "group_type"]
        .into_iter()
        .find_map(|candidate| {
            object.get(candidate).or_else(|| {
                object
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(candidate))
                    .map(|(_, value)| value)
            })
        });
    let group_type = match value {
        None | Some(Value::Null) => "AIGC".to_string(),
        Some(Value::String(value)) if value.trim().is_empty() => "AIGC".to_string(),
        Some(Value::String(value)) if value.trim() == "AIGC" => "AIGC".to_string(),
        Some(Value::String(_)) => {
            return Err(ArkAssetProtocolError::InvalidBody(
                "GroupType must be AIGC".to_string(),
            ));
        }
        Some(_) => {
            return Err(ArkAssetProtocolError::InvalidBody(
                "GroupType must be a string".to_string(),
            ));
        }
    };
    let aliases = object
        .keys()
        .filter(|name| {
            name.eq_ignore_ascii_case("GroupType")
                || name.eq_ignore_ascii_case("Type")
                || name.eq_ignore_ascii_case("group_type")
        })
        .cloned()
        .collect::<Vec<_>>();
    for alias in aliases {
        object.remove(&alias);
    }
    object.insert("GroupType".to_string(), Value::String(group_type));
    Ok(())
}

fn canonicalize_string_alias(
    object: &mut Map<String, Value>,
    canonical_name: &str,
    aliases: &[&str],
) {
    let value = object
        .get(canonical_name)
        .or_else(|| {
            object
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(canonical_name))
                .map(|(_, value)| value)
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            object.iter().find_map(|(name, value)| {
                if aliases.iter().any(|alias| name.eq_ignore_ascii_case(alias)) {
                    value
                        .as_str()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(ToOwned::to_owned)
                } else {
                    None
                }
            })
        });
    let alias_keys = object
        .keys()
        .filter(|name| {
            name.as_str() != canonical_name
                && (name.eq_ignore_ascii_case(canonical_name)
                    || aliases.iter().any(|alias| name.eq_ignore_ascii_case(alias)))
        })
        .cloned()
        .collect::<Vec<_>>();
    for alias in alias_keys {
        object.remove(&alias);
    }
    if let Some(value) = value {
        object.insert(canonical_name.to_string(), Value::String(value));
    }
}

pub(crate) fn extract_result(body: &Value) -> Option<&Value> {
    body.get("Result").or_else(|| body.get("result"))
}

pub(crate) fn response_status_from_body(body: &Value) -> Option<u16> {
    body.get("ResponseMetadata")
        .or_else(|| body.get("response_metadata"))
        .and_then(|metadata| metadata.get("Error"))
        .or_else(|| body.get("error"))
        .and_then(|error| error.get("HTTPCode").or_else(|| error.get("http_code")))
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok())
}

pub(crate) fn build_error_envelope(code: &str, message: &str) -> Value {
    serde_json::json!({
        "ResponseMetadata": {
            "Error": {
                "Code": code,
                "Message": message,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_ACTIONS: [ArkAssetAction; 12] = [
        ArkAssetAction::CreateAssetGroup,
        ArkAssetAction::ListAssetGroups,
        ArkAssetAction::GetAssetGroup,
        ArkAssetAction::UpdateAssetGroup,
        ArkAssetAction::DeleteAssetGroup,
        ArkAssetAction::CreateAsset,
        ArkAssetAction::ListAssets,
        ArkAssetAction::GetAsset,
        ArkAssetAction::UpdateAsset,
        ArkAssetAction::DeleteAsset,
        ArkAssetAction::CreateVisualValidateSession,
        ArkAssetAction::GetVisualValidateResult,
    ];

    #[test]
    fn parses_native_action_and_version() {
        let uri: Uri = "/?Action=CreateAsset&Version=2024-01-01".parse().unwrap();
        assert_eq!(
            action_from_request(&uri, &serde_json::json!({})),
            Ok(ArkAssetAction::CreateAsset)
        );
    }

    #[test]
    fn alias_path_and_body_action_are_supported() {
        let uri: Uri = "/v3/asset-library/CreateAssetGroup".parse().unwrap();
        assert_eq!(
            action_from_request(&uri, &serde_json::json!({})),
            Ok(ArkAssetAction::CreateAssetGroup)
        );
        let uri: Uri = "/v3/asset-library".parse().unwrap();
        assert_eq!(
            action_from_request(&uri, &serde_json::json!({"action":"ListAssets"})),
            Ok(ArkAssetAction::ListAssets)
        );
    }

    #[test]
    fn rejects_unknown_action_and_bad_version() {
        let uri: Uri = "/?Action=DeleteEverything&Version=2024-01-01"
            .parse()
            .unwrap();
        assert!(matches!(
            action_from_request(&uri, &serde_json::json!({})),
            Err(ArkAssetProtocolError::UnsupportedAction(_))
        ));
        let uri: Uri = "/v3/asset-library/DeleteEverything".parse().unwrap();
        assert!(matches!(
            action_from_request(&uri, &serde_json::json!({})),
            Err(ArkAssetProtocolError::UnsupportedAction(_))
        ));
        let uri: Uri = "/?Action=ListAssets&Version=2023-01-01".parse().unwrap();
        assert_eq!(
            action_from_request(&uri, &serde_json::json!({})),
            Err(ArkAssetProtocolError::InvalidVersion)
        );
        let uri: Uri = "/?Action=ListAssets".parse().unwrap();
        assert_eq!(
            action_from_request(&uri, &serde_json::json!({})),
            Err(ArkAssetProtocolError::InvalidVersion)
        );
        let uri: Uri = "/?Action=ListAssets&Version=2024-01-01".parse().unwrap();
        assert!(matches!(
            action_from_request(
                &uri,
                &serde_json::json!({"Action":"DeleteAsset","Version":"2024-01-01"})
            ),
            Err(ArkAssetProtocolError::InvalidBody(_))
        ));
    }

    #[test]
    fn strips_envelope_keys_without_mutating_payload() {
        let body = serde_json::json!({"Action":"ListAssets","GroupId":"g","Version":"2024-01-01"});
        assert_eq!(
            sanitize_action_body(&body).unwrap(),
            serde_json::json!({"GroupId":"g"})
        );
    }

    #[test]
    fn k23_request_fixtures_are_canonicalized_for_every_action() {
        for (action, input, expected) in [
            (
                ArkAssetAction::CreateAssetGroup,
                serde_json::json!({
                    "Name": "products",
                    "Description": "references",
                    "GroupType": "AIGC",
                    "ProjectName": "default"
                }),
                serde_json::json!({
                    "Name": "products",
                    "Description": "references",
                    "GroupType": "AIGC"
                }),
            ),
            (
                ArkAssetAction::ListAssetGroups,
                serde_json::json!({
                    "Filter": {"GroupType": "AIGC", "GroupIds": ["group-1"]},
                    "PageNumber": 1,
                    "PageSize": 10
                }),
                serde_json::json!({
                    "Filter": {"GroupType": "AIGC", "GroupIds": ["group-1"]},
                    "PageNumber": 1,
                    "PageSize": 10
                }),
            ),
            (
                ArkAssetAction::GetAssetGroup,
                serde_json::json!({"GroupId": "group-1", "ProjectName": "default"}),
                serde_json::json!({"Id": "group-1"}),
            ),
            (
                ArkAssetAction::UpdateAssetGroup,
                serde_json::json!({"GROUPID": "group-1", "Name": "new name"}),
                serde_json::json!({"Id": "group-1", "Name": "new name"}),
            ),
            (
                ArkAssetAction::DeleteAssetGroup,
                serde_json::json!({"GroupId": "group-1"}),
                serde_json::json!({"Id": "group-1"}),
            ),
            (
                ArkAssetAction::CreateAsset,
                serde_json::json!({
                    "GroupId": "group-1",
                    "URL": "https://example.com/image.jpg",
                    "AssetType": "Image",
                    "Name": null
                }),
                serde_json::json!({
                    "GroupId": "group-1",
                    "URL": "https://example.com/image.jpg",
                    "AssetType": "Image"
                }),
            ),
            (
                ArkAssetAction::ListAssets,
                serde_json::json!({
                    "Filter": {"GroupIds": ["group-1"], "Statuses": ["Active"]},
                    "PageNumber": 1,
                    "PageSize": 10
                }),
                serde_json::json!({
                    "Filter": {"GroupIds": ["group-1"], "Statuses": ["Active"]},
                    "PageNumber": 1,
                    "PageSize": 10
                }),
            ),
            (
                ArkAssetAction::GetAsset,
                serde_json::json!({"AssetId": "asset-1"}),
                serde_json::json!({"Id": "asset-1"}),
            ),
            (
                ArkAssetAction::DeleteAsset,
                serde_json::json!({"AssetId": "asset-1"}),
                serde_json::json!({"Id": "asset-1"}),
            ),
            (
                ArkAssetAction::UpdateAsset,
                serde_json::json!({"asset_id": "asset-1", "Name": "new name"}),
                serde_json::json!({"Id": "asset-1", "Name": "new name"}),
            ),
            (
                ArkAssetAction::CreateVisualValidateSession,
                serde_json::json!({"return_url": "https://example.com/callback"}),
                serde_json::json!({"CallbackURL": "https://example.com/callback"}),
            ),
            (
                ArkAssetAction::GetVisualValidateResult,
                serde_json::json!({"BytedToken": "token-1", "ProjectName": "default"}),
                serde_json::json!({"BytedToken": "token-1"}),
            ),
        ] {
            let output = canonicalize_provider_body(action, &input).unwrap();
            assert_eq!(output, expected, "{action:?}");
        }
    }

    #[test]
    fn visual_validation_uses_only_callback_url_upstream() {
        assert_eq!(
            canonicalize_provider_body(
                ArkAssetAction::CreateVisualValidateSession,
                &serde_json::json!({
                    "return_url": "https://example.com/legacy-callback"
                }),
            )
            .unwrap(),
            serde_json::json!({"CallbackURL": "https://example.com/legacy-callback"})
        );
    }

    #[test]
    fn create_group_defaults_group_type_in_canonical_upstream_body() {
        for input in [
            serde_json::json!({"Name": "products"}),
            serde_json::json!({"Name": "products", "GroupType": null}),
            serde_json::json!({"Name": "products", "GroupType": ""}),
            serde_json::json!({"Name": "products", "GroupType": "   "}),
        ] {
            assert_eq!(
                canonicalize_provider_body(ArkAssetAction::CreateAssetGroup, &input).unwrap(),
                serde_json::json!({"Name": "products", "GroupType": "AIGC"})
            );
        }
    }

    #[test]
    fn create_group_normalizes_group_type_aliases_for_upstream() {
        for input in [
            serde_json::json!({"Name": "products", "Type": "AIGC"}),
            serde_json::json!({"Name": "products", "group_type": "AIGC"}),
            serde_json::json!({"Name": "products", "gRoUpTyPe": "AIGC"}),
        ] {
            let canonical =
                canonicalize_provider_body(ArkAssetAction::CreateAssetGroup, &input).unwrap();
            assert_eq!(
                canonical,
                serde_json::json!({"Name": "products", "GroupType": "AIGC"})
            );
            let object = canonical.as_object().unwrap();
            assert_eq!(object.keys().filter(|key| key.contains("Type")).count(), 1);
        }
    }

    #[test]
    fn create_group_rejects_non_string_group_type_during_canonicalization() {
        assert_eq!(
            canonicalize_provider_body(
                ArkAssetAction::CreateAssetGroup,
                &serde_json::json!({"Name": "products", "GroupType": 1}),
            ),
            Err(ArkAssetProtocolError::InvalidBody(
                "GroupType must be a string".to_string()
            ))
        );
    }

    #[test]
    fn create_group_rejects_explicit_non_aigc_group_type_before_upstream() {
        assert_eq!(
            canonicalize_provider_body(
                ArkAssetAction::CreateAssetGroup,
                &serde_json::json!({"Name": "products", "Type": "LivenessFace"}),
            ),
            Err(ArkAssetProtocolError::InvalidBody(
                "GroupType must be AIGC".to_string()
            ))
        );
    }

    #[test]
    fn every_supported_action_round_trips_through_native_query_protocol() {
        for action in ALL_ACTIONS {
            let uri: Uri = format!("/?Action={}&Version={ARK_ASSET_VERSION}", action.as_str())
                .parse()
                .unwrap();
            assert_eq!(
                action_from_request(&uri, &serde_json::json!({})),
                Ok(action)
            );
            assert_eq!(ArkAssetAction::from_str(action.as_str()), Some(action));
        }
    }

    #[test]
    fn matching_query_and_body_actions_are_accepted_and_stripped() {
        let uri: Uri = "/?Action=GetAsset&Version=2024-01-01".parse().unwrap();
        assert_eq!(
            action_from_request(
                &uri,
                &serde_json::json!({"Action":"GetAsset","AssetId":"asset-local"})
            ),
            Ok(ArkAssetAction::GetAsset)
        );
        assert_eq!(
            sanitize_action_body(&serde_json::json!({"Action":"GetAsset","AssetId":"asset-local"}))
                .unwrap(),
            serde_json::json!({"AssetId":"asset-local"})
        );
    }

    #[test]
    fn action_body_must_be_an_object() {
        assert!(matches!(
            sanitize_action_body(&serde_json::json!(["not", "an", "object"])),
            Err(ArkAssetProtocolError::InvalidBody(_))
        ));
        assert!(matches!(
            sanitize_action_body(&Value::Null),
            Err(ArkAssetProtocolError::InvalidBody(_))
        ));
    }

    #[test]
    fn extracts_http_status_from_native_and_compat_error_envelopes() {
        assert_eq!(
            response_status_from_body(&serde_json::json!({
                "ResponseMetadata": {"Error": {"HTTPCode": 403}}
            })),
            Some(403)
        );
        assert_eq!(
            response_status_from_body(&serde_json::json!({
                "error": {"http_code": 429}
            })),
            Some(429)
        );
        assert_eq!(
            response_status_from_body(&serde_json::json!({
                "ResponseMetadata": {"RequestId": "request-1"},
                "Result": {}
            })),
            None
        );
    }

    #[test]
    fn path_recognition_does_not_match_similar_prefixes() {
        assert!(is_asset_library_path("/"));
        assert!(is_asset_library_path("/v3/asset-library/GetAsset"));
        assert!(!is_asset_library_path("/v3/asset-library-evil"));
        assert!(!is_asset_library_path("/v3/assets"));
    }
}
