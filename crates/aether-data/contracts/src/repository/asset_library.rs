use async_trait::async_trait;
use serde_json::Value;

const ID_MAX_LEN: usize = 64;
const UPSTREAM_ID_MAX_LEN: usize = 255;
const NAME_MAX_LEN: usize = 512;
const TYPE_MAX_LEN: usize = 64;
const STATUS_MAX_LEN: usize = 64;
const ACCOUNT_BINDING_MAX_LEN: usize = 128;
const PROJECT_MAX_LEN: usize = 255;
const HASH_MAX_LEN: usize = 128;
const ERROR_CODE_MAX_LEN: usize = 128;

#[derive(Debug, Clone, PartialEq)]
pub struct StoredAssetGroup {
    pub id: String,
    pub upstream_group_id: Option<String>,
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub provider_id: String,
    pub endpoint_id: String,
    pub key_id: String,
    pub account_binding: Option<String>,
    pub project: Option<String>,
    pub group_type: String,
    pub name: String,
    pub description: Option<String>,
    pub status: String,
    pub created_at_unix_secs: u64,
    pub updated_at_unix_secs: u64,
    pub deleted_at_unix_secs: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UpsertAssetGroupRecord {
    pub id: String,
    pub upstream_group_id: Option<String>,
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub provider_id: String,
    pub endpoint_id: String,
    pub key_id: String,
    pub account_binding: Option<String>,
    pub project: Option<String>,
    pub group_type: String,
    pub name: String,
    pub description: Option<String>,
    pub status: String,
    pub created_at_unix_secs: u64,
    pub updated_at_unix_secs: u64,
    pub deleted_at_unix_secs: Option<u64>,
}

impl UpsertAssetGroupRecord {
    pub fn validate(&self) -> Result<(), crate::DataLayerError> {
        validate_required_text("asset_groups.id", &self.id, ID_MAX_LEN)?;
        validate_optional_text(
            "asset_groups.upstream_group_id",
            self.upstream_group_id.as_deref(),
            UPSTREAM_ID_MAX_LEN,
        )?;
        validate_required_text("asset_groups.user_id", &self.user_id, ID_MAX_LEN)?;
        validate_optional_text(
            "asset_groups.api_key_id",
            self.api_key_id.as_deref(),
            ID_MAX_LEN,
        )?;
        validate_required_text("asset_groups.provider_id", &self.provider_id, ID_MAX_LEN)?;
        validate_required_text("asset_groups.endpoint_id", &self.endpoint_id, ID_MAX_LEN)?;
        validate_required_text("asset_groups.key_id", &self.key_id, ID_MAX_LEN)?;
        validate_optional_text(
            "asset_groups.account_binding",
            self.account_binding.as_deref(),
            ACCOUNT_BINDING_MAX_LEN,
        )?;
        validate_optional_text(
            "asset_groups.project",
            self.project.as_deref(),
            PROJECT_MAX_LEN,
        )?;
        validate_canonical_optional_binding(
            "asset_groups.account_binding",
            self.account_binding.as_deref(),
        )?;
        validate_canonical_optional_binding("asset_groups.project", self.project.as_deref())?;
        if self.account_binding.is_none() {
            return Err(crate::DataLayerError::InvalidInput(
                "asset_groups.account_binding is required".to_string(),
            ));
        }
        validate_required_text("asset_groups.group_type", &self.group_type, TYPE_MAX_LEN)?;
        validate_required_text("asset_groups.name", &self.name, NAME_MAX_LEN)?;
        validate_required_text("asset_groups.status", &self.status, STATUS_MAX_LEN)?;
        validate_timestamps(
            "asset_groups",
            self.created_at_unix_secs,
            self.updated_at_unix_secs,
            self.deleted_at_unix_secs,
        )
    }

    pub fn into_stored(self) -> StoredAssetGroup {
        StoredAssetGroup {
            id: self.id,
            upstream_group_id: self.upstream_group_id,
            user_id: self.user_id,
            api_key_id: self.api_key_id,
            provider_id: self.provider_id,
            endpoint_id: self.endpoint_id,
            key_id: self.key_id,
            account_binding: self.account_binding,
            project: self.project,
            group_type: self.group_type,
            name: self.name,
            description: self.description,
            status: self.status,
            created_at_unix_secs: self.created_at_unix_secs,
            updated_at_unix_secs: self.updated_at_unix_secs,
            deleted_at_unix_secs: self.deleted_at_unix_secs,
        }
    }

    pub fn has_same_immutable_identity(&self, stored: &StoredAssetGroup) -> bool {
        self.id == stored.id
            && self.upstream_group_id == stored.upstream_group_id
            && self.user_id == stored.user_id
            && self.provider_id == stored.provider_id
            && self.account_binding == stored.account_binding
            && self.project == stored.project
            && self.group_type == stored.group_type
            && self.deleted_at_unix_secs == stored.deleted_at_unix_secs
            && (stored.deleted_at_unix_secs.is_none() || self.status == stored.status)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AssetGroupListQuery {
    pub user_id: Option<String>,
    pub api_key_id: Option<String>,
    pub provider_id: Option<String>,
    pub group_type: Option<String>,
    pub status: Option<String>,
    pub search: Option<String>,
    pub include_deleted: bool,
    pub offset: usize,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredAssetGroupListPage {
    pub items: Vec<StoredAssetGroup>,
    pub total: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetProviderReference<'a> {
    ProviderId(&'a str),
    EndpointId(&'a str),
    KeyId(&'a str),
}

impl<'a> AssetProviderReference<'a> {
    pub fn id(self) -> &'a str {
        match self {
            Self::ProviderId(id) | Self::EndpointId(id) | Self::KeyId(id) => id,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AssetProviderReferenceCounts {
    pub asset_groups: u64,
    pub visual_validation_sessions: u64,
}

impl AssetProviderReferenceCounts {
    pub fn total(self) -> u64 {
        self.asset_groups
            .saturating_add(self.visual_validation_sessions)
    }

    pub fn is_referenced(self) -> bool {
        self.total() != 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredAsset {
    pub id: String,
    pub upstream_asset_id: Option<String>,
    pub group_id: String,
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub asset_type: String,
    pub name: String,
    pub status: String,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub moderation: Option<Value>,
    pub last_inference_at_unix_secs: Option<u64>,
    pub source_url_fingerprint: Option<String>,
    pub provider_url: Option<String>,
    pub provider_url_expires_at_unix_secs: Option<u64>,
    pub sanitized_metadata: Option<Value>,
    pub is_deleted: bool,
    pub deleted_at_unix_secs: Option<u64>,
    pub created_at_unix_secs: u64,
    pub updated_at_unix_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UpsertAssetRecord {
    pub id: String,
    pub upstream_asset_id: Option<String>,
    pub group_id: String,
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub asset_type: String,
    pub name: String,
    pub status: String,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub moderation: Option<Value>,
    pub last_inference_at_unix_secs: Option<u64>,
    pub source_url_fingerprint: Option<String>,
    pub provider_url: Option<String>,
    pub provider_url_expires_at_unix_secs: Option<u64>,
    pub sanitized_metadata: Option<Value>,
    pub is_deleted: bool,
    pub deleted_at_unix_secs: Option<u64>,
    pub created_at_unix_secs: u64,
    pub updated_at_unix_secs: u64,
}

impl UpsertAssetRecord {
    pub fn validate(&self) -> Result<(), crate::DataLayerError> {
        validate_required_text("assets.id", &self.id, ID_MAX_LEN)?;
        validate_optional_text(
            "assets.upstream_asset_id",
            self.upstream_asset_id.as_deref(),
            UPSTREAM_ID_MAX_LEN,
        )?;
        validate_required_text("assets.group_id", &self.group_id, ID_MAX_LEN)?;
        validate_required_text("assets.user_id", &self.user_id, ID_MAX_LEN)?;
        validate_optional_text("assets.api_key_id", self.api_key_id.as_deref(), ID_MAX_LEN)?;
        validate_required_text("assets.asset_type", &self.asset_type, TYPE_MAX_LEN)?;
        validate_required_text("assets.name", &self.name, NAME_MAX_LEN)?;
        validate_required_text("assets.status", &self.status, STATUS_MAX_LEN)?;
        validate_optional_text(
            "assets.error_code",
            self.error_code.as_deref(),
            ERROR_CODE_MAX_LEN,
        )?;
        validate_optional_text(
            "assets.source_url_fingerprint",
            self.source_url_fingerprint.as_deref(),
            HASH_MAX_LEN,
        )?;
        validate_optional_json_object("assets.moderation", self.moderation.as_ref())?;
        validate_optional_json_object(
            "assets.sanitized_metadata",
            self.sanitized_metadata.as_ref(),
        )?;
        if self.is_deleted != self.deleted_at_unix_secs.is_some() {
            return Err(crate::DataLayerError::InvalidInput(
                "assets.is_deleted and assets.deleted_at must agree".to_string(),
            ));
        }
        validate_timestamps(
            "assets",
            self.created_at_unix_secs,
            self.updated_at_unix_secs,
            self.deleted_at_unix_secs,
        )
    }

    pub fn into_stored(self) -> StoredAsset {
        StoredAsset {
            id: self.id,
            upstream_asset_id: self.upstream_asset_id,
            group_id: self.group_id,
            user_id: self.user_id,
            api_key_id: self.api_key_id,
            asset_type: self.asset_type,
            name: self.name,
            status: self.status,
            error_code: self.error_code,
            error_message: self.error_message,
            moderation: self.moderation,
            last_inference_at_unix_secs: self.last_inference_at_unix_secs,
            source_url_fingerprint: self.source_url_fingerprint,
            provider_url: self.provider_url,
            provider_url_expires_at_unix_secs: self.provider_url_expires_at_unix_secs,
            sanitized_metadata: self.sanitized_metadata,
            is_deleted: self.is_deleted,
            deleted_at_unix_secs: self.deleted_at_unix_secs,
            created_at_unix_secs: self.created_at_unix_secs,
            updated_at_unix_secs: self.updated_at_unix_secs,
        }
    }

    pub fn has_same_immutable_identity(&self, stored: &StoredAsset) -> bool {
        self.id == stored.id
            && self.upstream_asset_id == stored.upstream_asset_id
            && self.group_id == stored.group_id
            && self.user_id == stored.user_id
            && self.asset_type == stored.asset_type
            && self.is_deleted == stored.is_deleted
            && self.deleted_at_unix_secs == stored.deleted_at_unix_secs
            && (!stored.is_deleted || self.status == stored.status)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AssetListQuery {
    pub group_id: Option<String>,
    pub user_id: Option<String>,
    pub api_key_id: Option<String>,
    pub asset_type: Option<String>,
    pub status: Option<String>,
    pub search: Option<String>,
    pub include_deleted: bool,
    pub offset: usize,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredAssetListPage {
    pub items: Vec<StoredAsset>,
    pub total: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredArkVisualValidationSession {
    pub id: String,
    pub session_id: String,
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub provider_id: String,
    pub endpoint_id: String,
    pub key_id: String,
    pub account_binding: Option<String>,
    pub project: Option<String>,
    pub byted_token_hash: String,
    pub encrypted_byted_token: String,
    pub callback_state_hash: String,
    pub status: String,
    pub expires_at_unix_secs: u64,
    pub consumed_at_unix_secs: Option<u64>,
    pub group_id: Option<String>,
    pub sanitized_result: Option<Value>,
    pub created_at_unix_secs: u64,
    pub updated_at_unix_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UpsertArkVisualValidationSessionRecord {
    pub id: String,
    pub session_id: String,
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub provider_id: String,
    pub endpoint_id: String,
    pub key_id: String,
    pub account_binding: Option<String>,
    pub project: Option<String>,
    pub byted_token_hash: String,
    pub encrypted_byted_token: String,
    pub callback_state_hash: String,
    pub status: String,
    pub expires_at_unix_secs: u64,
    pub consumed_at_unix_secs: Option<u64>,
    pub group_id: Option<String>,
    pub sanitized_result: Option<Value>,
    pub created_at_unix_secs: u64,
    pub updated_at_unix_secs: u64,
}

impl UpsertArkVisualValidationSessionRecord {
    pub fn validate(&self) -> Result<(), crate::DataLayerError> {
        validate_required_text("ark_visual_validation_sessions.id", &self.id, ID_MAX_LEN)?;
        validate_required_text(
            "ark_visual_validation_sessions.session_id",
            &self.session_id,
            UPSTREAM_ID_MAX_LEN,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.user_id",
            &self.user_id,
            ID_MAX_LEN,
        )?;
        validate_optional_text(
            "ark_visual_validation_sessions.api_key_id",
            self.api_key_id.as_deref(),
            ID_MAX_LEN,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.provider_id",
            &self.provider_id,
            ID_MAX_LEN,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.endpoint_id",
            &self.endpoint_id,
            ID_MAX_LEN,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.key_id",
            &self.key_id,
            ID_MAX_LEN,
        )?;
        validate_optional_text(
            "ark_visual_validation_sessions.account_binding",
            self.account_binding.as_deref(),
            ACCOUNT_BINDING_MAX_LEN,
        )?;
        validate_optional_text(
            "ark_visual_validation_sessions.project",
            self.project.as_deref(),
            PROJECT_MAX_LEN,
        )?;
        validate_canonical_optional_binding(
            "ark_visual_validation_sessions.account_binding",
            self.account_binding.as_deref(),
        )?;
        validate_canonical_optional_binding(
            "ark_visual_validation_sessions.project",
            self.project.as_deref(),
        )?;
        if self.account_binding.is_none() {
            return Err(crate::DataLayerError::InvalidInput(
                "ark_visual_validation_sessions.account_binding is required".to_string(),
            ));
        }
        validate_required_text(
            "ark_visual_validation_sessions.byted_token_hash",
            &self.byted_token_hash,
            HASH_MAX_LEN,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.encrypted_byted_token",
            &self.encrypted_byted_token,
            usize::MAX,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.callback_state_hash",
            &self.callback_state_hash,
            HASH_MAX_LEN,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.status",
            &self.status,
            STATUS_MAX_LEN,
        )?;
        validate_optional_text(
            "ark_visual_validation_sessions.group_id",
            self.group_id.as_deref(),
            ID_MAX_LEN,
        )?;
        validate_optional_json_object(
            "ark_visual_validation_sessions.sanitized_result",
            self.sanitized_result.as_ref(),
        )?;
        if self.expires_at_unix_secs <= self.created_at_unix_secs {
            return Err(crate::DataLayerError::InvalidInput(
                "ark_visual_validation_sessions.expires_at must be after created_at".to_string(),
            ));
        }
        validate_timestamps(
            "ark_visual_validation_sessions",
            self.created_at_unix_secs,
            self.updated_at_unix_secs,
            self.consumed_at_unix_secs,
        )
    }

    pub fn into_stored(self) -> StoredArkVisualValidationSession {
        StoredArkVisualValidationSession {
            id: self.id,
            session_id: self.session_id,
            user_id: self.user_id,
            api_key_id: self.api_key_id,
            provider_id: self.provider_id,
            endpoint_id: self.endpoint_id,
            key_id: self.key_id,
            account_binding: self.account_binding,
            project: self.project,
            byted_token_hash: self.byted_token_hash,
            encrypted_byted_token: self.encrypted_byted_token,
            callback_state_hash: self.callback_state_hash,
            status: self.status,
            expires_at_unix_secs: self.expires_at_unix_secs,
            consumed_at_unix_secs: self.consumed_at_unix_secs,
            group_id: self.group_id,
            sanitized_result: self.sanitized_result,
            created_at_unix_secs: self.created_at_unix_secs,
            updated_at_unix_secs: self.updated_at_unix_secs,
        }
    }

    pub fn has_same_immutable_identity(&self, stored: &StoredArkVisualValidationSession) -> bool {
        self.id == stored.id
            && self.session_id == stored.session_id
            && self.user_id == stored.user_id
            && self.provider_id == stored.provider_id
            && self.account_binding == stored.account_binding
            && self.project == stored.project
            && self.byted_token_hash == stored.byted_token_hash
            && self.encrypted_byted_token == stored.encrypted_byted_token
            && self.callback_state_hash == stored.callback_state_hash
            && self.expires_at_unix_secs == stored.expires_at_unix_secs
            && (self.consumed_at_unix_secs.is_none()
                || stored.consumed_at_unix_secs.is_none()
                || self.consumed_at_unix_secs == stored.consumed_at_unix_secs)
            && (stored.group_id.is_none() || self.group_id == stored.group_id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConsumeArkVisualValidationSessionRecord {
    pub callback_state_hash: String,
    pub status: String,
    pub consumed_at_unix_secs: u64,
    pub sanitized_result: Option<Value>,
    pub updated_at_unix_secs: u64,
}

impl ConsumeArkVisualValidationSessionRecord {
    pub fn validate(&self) -> Result<(), crate::DataLayerError> {
        validate_required_text(
            "ark_visual_validation_sessions.callback_state_hash",
            &self.callback_state_hash,
            HASH_MAX_LEN,
        )?;
        validate_required_text(
            "ark_visual_validation_sessions.status",
            &self.status,
            STATUS_MAX_LEN,
        )?;
        validate_optional_json_object(
            "ark_visual_validation_sessions.sanitized_result",
            self.sanitized_result.as_ref(),
        )?;
        if self.consumed_at_unix_secs == 0 || self.updated_at_unix_secs < self.consumed_at_unix_secs
        {
            return Err(crate::DataLayerError::InvalidInput(
                "ark_visual_validation_sessions consume timestamps are invalid".to_string(),
            ));
        }
        Ok(())
    }
}

#[async_trait]
pub trait AssetLibraryReadRepository: Send + Sync {
    async fn count_provider_references(
        &self,
        reference: AssetProviderReference<'_>,
    ) -> Result<AssetProviderReferenceCounts, crate::DataLayerError>;

    async fn find_group_by_id(
        &self,
        group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, crate::DataLayerError>;

    async fn find_group_for_user(
        &self,
        group_id: &str,
        user_id: &str,
    ) -> Result<Option<StoredAssetGroup>, crate::DataLayerError>;

    async fn find_group_by_upstream(
        &self,
        provider_id: &str,
        endpoint_id: &str,
        key_id: &str,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, crate::DataLayerError>;

    async fn find_group_by_canonical_upstream(
        &self,
        provider_id: &str,
        account_binding: &str,
        project: Option<&str>,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, crate::DataLayerError>;

    async fn list_groups(
        &self,
        query: &AssetGroupListQuery,
    ) -> Result<StoredAssetGroupListPage, crate::DataLayerError>;

    async fn find_asset_by_id(
        &self,
        asset_id: &str,
    ) -> Result<Option<StoredAsset>, crate::DataLayerError>;

    async fn find_asset_for_user(
        &self,
        asset_id: &str,
        user_id: &str,
    ) -> Result<Option<StoredAsset>, crate::DataLayerError>;

    async fn find_asset_by_upstream(
        &self,
        group_id: &str,
        upstream_asset_id: &str,
    ) -> Result<Option<StoredAsset>, crate::DataLayerError>;

    async fn list_assets(
        &self,
        query: &AssetListQuery,
    ) -> Result<StoredAssetListPage, crate::DataLayerError>;

    async fn find_visual_validation_session_by_id(
        &self,
        id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, crate::DataLayerError>;

    async fn find_visual_validation_session_for_user(
        &self,
        id: &str,
        user_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, crate::DataLayerError>;

    async fn find_visual_validation_session_by_upstream(
        &self,
        provider_id: &str,
        key_id: &str,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, crate::DataLayerError>;

    async fn find_visual_validation_session_by_canonical_upstream(
        &self,
        provider_id: &str,
        account_binding: &str,
        project: Option<&str>,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, crate::DataLayerError>;

    async fn find_visual_validation_session_by_byted_token_hash(
        &self,
        byted_token_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, crate::DataLayerError>;

    async fn find_visual_validation_session_by_callback_state_hash(
        &self,
        callback_state_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, crate::DataLayerError>;
}

#[async_trait]
pub trait AssetLibraryWriteRepository: Send + Sync {
    async fn upsert_group(
        &self,
        record: UpsertAssetGroupRecord,
    ) -> Result<StoredAssetGroup, crate::DataLayerError>;

    async fn upsert_asset(
        &self,
        record: UpsertAssetRecord,
    ) -> Result<StoredAsset, crate::DataLayerError>;

    async fn soft_delete_group(
        &self,
        group_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, crate::DataLayerError>;

    async fn soft_delete_asset(
        &self,
        asset_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, crate::DataLayerError>;

    async fn upsert_visual_validation_session(
        &self,
        record: UpsertArkVisualValidationSessionRecord,
    ) -> Result<StoredArkVisualValidationSession, crate::DataLayerError>;

    async fn consume_visual_validation_session(
        &self,
        record: ConsumeArkVisualValidationSessionRecord,
    ) -> Result<Option<StoredArkVisualValidationSession>, crate::DataLayerError>;
}

pub trait AssetLibraryRepository: AssetLibraryReadRepository + AssetLibraryWriteRepository {}

impl<T> AssetLibraryRepository for T where
    T: AssetLibraryReadRepository + AssetLibraryWriteRepository
{
}

fn validate_required_text(
    field: &str,
    value: &str,
    max_len: usize,
) -> Result<(), crate::DataLayerError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(crate::DataLayerError::InvalidInput(format!(
            "{field} is empty"
        )));
    }
    if value.len() > max_len {
        return Err(crate::DataLayerError::InvalidInput(format!(
            "{field} exceeds {max_len} bytes"
        )));
    }
    Ok(())
}

fn validate_optional_text(
    field: &str,
    value: Option<&str>,
    max_len: usize,
) -> Result<(), crate::DataLayerError> {
    if let Some(value) = value {
        validate_required_text(field, value, max_len)?;
    }
    Ok(())
}

fn validate_canonical_optional_binding(
    field: &str,
    value: Option<&str>,
) -> Result<(), crate::DataLayerError> {
    if value.is_some_and(|value| value != value.trim()) {
        return Err(crate::DataLayerError::InvalidInput(format!(
            "{field} must not contain surrounding whitespace"
        )));
    }
    Ok(())
}

fn validate_optional_json_object(
    field: &str,
    value: Option<&Value>,
) -> Result<(), crate::DataLayerError> {
    if value.is_some_and(|value| !value.is_object()) {
        return Err(crate::DataLayerError::InvalidInput(format!(
            "{field} must be a JSON object"
        )));
    }
    Ok(())
}

fn validate_timestamps(
    table: &str,
    created_at_unix_secs: u64,
    updated_at_unix_secs: u64,
    terminal_at_unix_secs: Option<u64>,
) -> Result<(), crate::DataLayerError> {
    if created_at_unix_secs == 0 || updated_at_unix_secs < created_at_unix_secs {
        return Err(crate::DataLayerError::InvalidInput(format!(
            "{table} timestamps are invalid"
        )));
    }
    if terminal_at_unix_secs
        .is_some_and(|value| value < created_at_unix_secs || value > updated_at_unix_secs)
    {
        return Err(crate::DataLayerError::InvalidInput(format!(
            "{table} terminal timestamp is invalid"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AssetProviderReference, AssetProviderReferenceCounts,
        UpsertArkVisualValidationSessionRecord, UpsertAssetGroupRecord, UpsertAssetRecord,
    };

    fn valid_group() -> UpsertAssetGroupRecord {
        UpsertAssetGroupRecord {
            id: "group-1".to_string(),
            upstream_group_id: Some("upstream-group-1".to_string()),
            user_id: "user-1".to_string(),
            api_key_id: None,
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            account_binding: Some("a".repeat(128)),
            project: Some("p".repeat(255)),
            group_type: "face".to_string(),
            name: "group".to_string(),
            description: None,
            status: "active".to_string(),
            created_at_unix_secs: 1,
            updated_at_unix_secs: 1,
            deleted_at_unix_secs: None,
        }
    }

    #[test]
    fn canonical_identity_lengths_fit_mysql_utf8mb4_unique_keys() {
        let record = valid_group();
        assert!(record.validate().is_ok());

        let mut oversized_account = record.clone();
        oversized_account.account_binding = Some("a".repeat(129));
        assert!(oversized_account.validate().is_err());

        let mut oversized_project = record;
        oversized_project.project = Some("p".repeat(256));
        assert!(oversized_project.validate().is_err());
    }

    #[test]
    fn provider_reference_targets_and_counts_are_explicit() {
        assert_eq!(
            AssetProviderReference::ProviderId("provider-1").id(),
            "provider-1"
        );
        assert_eq!(
            AssetProviderReference::EndpointId("endpoint-1").id(),
            "endpoint-1"
        );
        assert_eq!(AssetProviderReference::KeyId("key-1").id(), "key-1");

        let counts = AssetProviderReferenceCounts {
            asset_groups: 2,
            visual_validation_sessions: 3,
        };
        assert_eq!(counts.total(), 5);
        assert!(counts.is_referenced());
        assert!(!AssetProviderReferenceCounts::default().is_referenced());
    }

    #[test]
    fn asset_delete_flag_and_timestamp_must_agree() {
        let record = UpsertAssetRecord {
            id: "asset-1".to_string(),
            upstream_asset_id: None,
            group_id: "group-1".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: None,
            asset_type: "image".to_string(),
            name: "portrait".to_string(),
            status: "active".to_string(),
            error_code: None,
            error_message: None,
            moderation: None,
            last_inference_at_unix_secs: None,
            source_url_fingerprint: None,
            provider_url: None,
            provider_url_expires_at_unix_secs: None,
            sanitized_metadata: None,
            is_deleted: true,
            deleted_at_unix_secs: None,
            created_at_unix_secs: 1,
            updated_at_unix_secs: 1,
        };

        assert!(record.validate().is_err());
    }

    #[test]
    fn visual_validation_result_must_be_sanitized_object() {
        let record = UpsertArkVisualValidationSessionRecord {
            id: "validation-1".to_string(),
            session_id: "session-1".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: None,
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            account_binding: None,
            project: None,
            byted_token_hash: "hash".to_string(),
            encrypted_byted_token: "encrypted".to_string(),
            callback_state_hash: "state-hash".to_string(),
            status: "pending".to_string(),
            expires_at_unix_secs: 10,
            consumed_at_unix_secs: None,
            group_id: None,
            sanitized_result: Some(serde_json::json!(["not", "an", "object"])),
            created_at_unix_secs: 1,
            updated_at_unix_secs: 1,
        };

        assert!(record.validate().is_err());
    }

    #[test]
    fn deletion_timestamp_cannot_follow_updated_at() {
        let mut record = valid_group();
        record.status = "deleted".to_string();
        record.updated_at_unix_secs = 2;
        record.deleted_at_unix_secs = Some(3);

        assert!(record.validate().is_err());
    }
}
