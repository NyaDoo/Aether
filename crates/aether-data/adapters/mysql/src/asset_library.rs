use async_trait::async_trait;
use sqlx::{mysql::MySqlRow, MySql, QueryBuilder, Row};

use aether_data_contracts::repository::asset_library::*;
use aether_data_contracts::DataLayerError;
use aether_data_query::{push_ci_contains_any, push_limit_offset, SqlDialect, WhereClause};

use crate::error::SqlResultExt;
use crate::MysqlPool;

const GROUP_COLUMNS: &str = r#"
SELECT id, upstream_group_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
       group_type, name, description, status,
       created_at_unix_secs, updated_at_unix_secs, deleted_at_unix_secs
FROM asset_groups
"#;

const ASSET_COLUMNS: &str = r#"
SELECT id, upstream_asset_id, group_id, user_id, api_key_id, asset_type, name, status,
       error_code, error_message, moderation, last_inference_at_unix_secs,
       source_url_fingerprint, provider_url, provider_url_expires_at_unix_secs,
       sanitized_metadata, is_deleted, deleted_at_unix_secs,
       created_at_unix_secs, updated_at_unix_secs
FROM assets
"#;

const SESSION_COLUMNS: &str = r#"
SELECT id, session_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
       byted_token_hash, encrypted_byted_token,
       callback_state_hash, status, expires_at_unix_secs, consumed_at_unix_secs,
       group_id, sanitized_result, created_at_unix_secs, updated_at_unix_secs
FROM ark_visual_validation_sessions
"#;

fn provider_reference_counts_sql(reference: AssetProviderReference<'_>) -> &'static str {
    match reference {
        AssetProviderReference::ProviderId(_) => {
            "SELECT (SELECT COUNT(*) FROM asset_groups WHERE provider_id = ?), (SELECT COUNT(*) FROM ark_visual_validation_sessions WHERE provider_id = ?)"
        }
        AssetProviderReference::EndpointId(_) => {
            "SELECT (SELECT COUNT(*) FROM asset_groups WHERE endpoint_id = ?), (SELECT COUNT(*) FROM ark_visual_validation_sessions WHERE endpoint_id = ?)"
        }
        AssetProviderReference::KeyId(_) => {
            "SELECT (SELECT COUNT(*) FROM asset_groups WHERE key_id = ?), (SELECT COUNT(*) FROM ark_visual_validation_sessions WHERE key_id = ?)"
        }
    }
}

#[derive(Debug, Clone)]
pub struct MysqlAssetLibraryRepository {
    pool: MysqlPool,
}

impl MysqlAssetLibraryRepository {
    pub fn new(pool: MysqlPool) -> Self {
        Self { pool }
    }

    async fn group_by_id(&self, id: &str) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        map_optional_group(
            sqlx::query(&format!("{GROUP_COLUMNS} WHERE id = ? LIMIT 1"))
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_sql_err()?,
        )
    }

    async fn asset_by_id(&self, id: &str) -> Result<Option<StoredAsset>, DataLayerError> {
        map_optional_asset(
            sqlx::query(&format!("{ASSET_COLUMNS} WHERE id = ? LIMIT 1"))
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_sql_err()?,
        )
    }

    async fn session_by_id(
        &self,
        id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!("{SESSION_COLUMNS} WHERE id = ? LIMIT 1"))
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_sql_err()?,
        )
    }
}

async fn lock_active_catalog_binding(
    transaction: &mut sqlx::Transaction<'_, MySql>,
    provider_id: &str,
    endpoint_id: &str,
    key_id: &str,
) -> Result<(), DataLayerError> {
    let provider_exists = sqlx::query_scalar::<_, String>(
        "SELECT id FROM providers WHERE id = ? AND is_active = TRUE FOR UPDATE",
    )
    .bind(provider_id)
    .fetch_optional(&mut **transaction)
    .await
    .map_sql_err()?
    .is_some();
    let endpoint_exists = if provider_exists {
        sqlx::query_scalar::<_, String>(
            "SELECT id FROM provider_endpoints WHERE id = ? AND provider_id = ? AND is_active = TRUE FOR UPDATE",
        )
        .bind(endpoint_id)
        .bind(provider_id)
        .fetch_optional(&mut **transaction)
        .await
        .map_sql_err()?
        .is_some()
    } else {
        false
    };
    let key_exists = if endpoint_exists {
        sqlx::query_scalar::<_, String>(
            "SELECT id FROM provider_api_keys WHERE id = ? AND provider_id = ? AND is_active = TRUE FOR UPDATE",
        )
        .bind(key_id)
        .bind(provider_id)
        .fetch_optional(&mut **transaction)
        .await
        .map_sql_err()?
        .is_some()
    } else {
        false
    };
    if !key_exists {
        return Err(DataLayerError::InvalidInput(
            "asset library provider, endpoint, or key binding is inactive or invalid".to_string(),
        ));
    }
    Ok(())
}

#[async_trait]
impl AssetLibraryReadRepository for MysqlAssetLibraryRepository {
    async fn count_provider_references(
        &self,
        reference: AssetProviderReference<'_>,
    ) -> Result<AssetProviderReferenceCounts, DataLayerError> {
        let (asset_groups, visual_validation_sessions) =
            sqlx::query_as::<_, (i64, i64)>(provider_reference_counts_sql(reference))
                .bind(reference.id())
                .bind(reference.id())
                .fetch_one(&self.pool)
                .await
                .map_sql_err()?;
        Ok(AssetProviderReferenceCounts {
            asset_groups: reference_count(asset_groups, "asset_groups")?,
            visual_validation_sessions: reference_count(
                visual_validation_sessions,
                "ark_visual_validation_sessions",
            )?,
        })
    }

    async fn find_group_by_id(
        &self,
        group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        self.group_by_id(group_id).await
    }

    async fn find_group_for_user(
        &self,
        group_id: &str,
        user_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        map_optional_group(
            sqlx::query(&format!(
                "{GROUP_COLUMNS} WHERE id = ? AND user_id = ? LIMIT 1"
            ))
            .bind(group_id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn find_group_by_upstream(
        &self,
        provider_id: &str,
        endpoint_id: &str,
        key_id: &str,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        map_optional_group(
            sqlx::query(&format!(
                "{GROUP_COLUMNS} WHERE provider_id = ? AND endpoint_id = ? AND key_id = ? AND upstream_group_id = ? LIMIT 1"
            ))
            .bind(provider_id)
            .bind(endpoint_id)
            .bind(key_id)
            .bind(upstream_group_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn find_group_by_canonical_upstream(
        &self,
        provider_id: &str,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        map_optional_group(
            sqlx::query(&format!(
                "{GROUP_COLUMNS} WHERE provider_id = ? AND upstream_group_id = ? LIMIT 1"
            ))
            .bind(provider_id)
            .bind(upstream_group_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn list_groups(
        &self,
        query: &AssetGroupListQuery,
    ) -> Result<StoredAssetGroupListPage, DataLayerError> {
        let total = group_count_query(query)
            .build_query_scalar::<i64>()
            .fetch_one(&self.pool)
            .await
            .map_sql_err()?;
        let rows = group_rows_query(query)
            .build()
            .fetch_all(&self.pool)
            .await
            .map_sql_err()?;
        Ok(StoredAssetGroupListPage {
            items: rows.iter().map(map_group).collect::<Result<Vec<_>, _>>()?,
            total: usize::try_from(total).unwrap_or_default(),
        })
    }

    async fn find_asset_by_id(
        &self,
        asset_id: &str,
    ) -> Result<Option<StoredAsset>, DataLayerError> {
        self.asset_by_id(asset_id).await
    }

    async fn find_asset_for_user(
        &self,
        asset_id: &str,
        user_id: &str,
    ) -> Result<Option<StoredAsset>, DataLayerError> {
        map_optional_asset(
            sqlx::query(&format!(
                "{ASSET_COLUMNS} WHERE id = ? AND user_id = ? LIMIT 1"
            ))
            .bind(asset_id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn find_asset_by_upstream(
        &self,
        group_id: &str,
        upstream_asset_id: &str,
    ) -> Result<Option<StoredAsset>, DataLayerError> {
        map_optional_asset(
            sqlx::query(&format!(
                "{ASSET_COLUMNS} WHERE group_id = ? AND upstream_asset_id = ? LIMIT 1"
            ))
            .bind(group_id)
            .bind(upstream_asset_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn list_assets(
        &self,
        query: &AssetListQuery,
    ) -> Result<StoredAssetListPage, DataLayerError> {
        let total = asset_count_query(query)
            .build_query_scalar::<i64>()
            .fetch_one(&self.pool)
            .await
            .map_sql_err()?;
        let rows = asset_rows_query(query)
            .build()
            .fetch_all(&self.pool)
            .await
            .map_sql_err()?;
        Ok(StoredAssetListPage {
            items: rows.iter().map(map_asset).collect::<Result<Vec<_>, _>>()?,
            total: usize::try_from(total).unwrap_or_default(),
        })
    }

    async fn find_visual_validation_session_by_id(
        &self,
        id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        self.session_by_id(id).await
    }

    async fn find_visual_validation_session_for_user(
        &self,
        id: &str,
        user_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE id = ? AND user_id = ? LIMIT 1"
            ))
            .bind(id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn find_visual_validation_session_by_upstream(
        &self,
        provider_id: &str,
        key_id: &str,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE provider_id = ? AND key_id = ? AND session_id = ? LIMIT 1"
            ))
            .bind(provider_id)
            .bind(key_id)
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn find_visual_validation_session_by_canonical_upstream(
        &self,
        provider_id: &str,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE provider_id = ? AND session_id = ? LIMIT 1"
            ))
            .bind(provider_id)
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn find_visual_validation_session_by_callback_state_hash(
        &self,
        callback_state_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE callback_state_hash = ? LIMIT 1"
            ))
            .bind(callback_state_hash)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }

    async fn find_visual_validation_session_by_byted_token_hash(
        &self,
        byted_token_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE byted_token_hash = ? LIMIT 1"
            ))
            .bind(byted_token_hash)
            .fetch_optional(&self.pool)
            .await
            .map_sql_err()?,
        )
    }
}

#[async_trait]
impl AssetLibraryWriteRepository for MysqlAssetLibraryRepository {
    async fn upsert_group(
        &self,
        record: UpsertAssetGroupRecord,
    ) -> Result<StoredAssetGroup, DataLayerError> {
        record.validate()?;
        let id = record.id.clone();
        let immutable_record = record.clone();
        let mut tx = self.pool.begin().await.map_sql_err()?;
        lock_active_catalog_binding(
            &mut tx,
            &record.provider_id,
            &record.endpoint_id,
            &record.key_id,
        )
        .await?;
        if let Some(existing) = map_optional_group(
            sqlx::query(&format!("{GROUP_COLUMNS} WHERE id = ? LIMIT 1 FOR UPDATE"))
                .bind(&id)
                .fetch_optional(&mut *tx)
                .await
                .map_sql_err()?,
        )? {
            if !record.has_same_immutable_identity(&existing) {
                return Err(DataLayerError::InvalidInput(
                    "asset group immutable identity cannot be changed".to_string(),
                ));
            }
        }
        sqlx::query(
            r#"
INSERT INTO asset_groups (
  id, upstream_group_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
  group_type, name, description, status,
  created_at_unix_secs, updated_at_unix_secs, deleted_at_unix_secs
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON DUPLICATE KEY UPDATE
  api_key_id = IF(id = VALUES(id), VALUES(api_key_id), api_key_id),
  name = IF(id = VALUES(id), VALUES(name), name),
  description = IF(id = VALUES(id), VALUES(description), description),
  status = IF(id = VALUES(id), VALUES(status), status),
  updated_at_unix_secs = IF(id = VALUES(id), VALUES(updated_at_unix_secs), updated_at_unix_secs)
"#,
        )
        .bind(record.id)
        .bind(record.upstream_group_id)
        .bind(record.user_id)
        .bind(record.api_key_id)
        .bind(record.provider_id)
        .bind(record.endpoint_id)
        .bind(record.key_id)
        .bind(record.group_type)
        .bind(record.name)
        .bind(record.description)
        .bind(record.status)
        .bind(to_i64(
            record.created_at_unix_secs,
            "asset group created_at",
        )?)
        .bind(to_i64(
            record.updated_at_unix_secs,
            "asset group updated_at",
        )?)
        .bind(to_optional_i64(
            record.deleted_at_unix_secs,
            "asset group deleted_at",
        )?)
        .execute(&mut *tx)
        .await
        .map_sql_err()?;
        let group = map_optional_group(
            sqlx::query(&format!("{GROUP_COLUMNS} WHERE id = ? LIMIT 1"))
                .bind(&id)
                .fetch_optional(&mut *tx)
                .await
                .map_sql_err()?,
        )?
        .ok_or_else(|| {
            DataLayerError::InvalidInput(
                "asset group upstream identity is already bound to another record".to_string(),
            )
        })?;
        if !immutable_record.has_same_immutable_identity(&group) {
            return Err(DataLayerError::InvalidInput(
                "asset group immutable identity cannot be changed".to_string(),
            ));
        }
        tx.commit().await.map_sql_err()?;
        Ok(group)
    }

    async fn upsert_asset(&self, record: UpsertAssetRecord) -> Result<StoredAsset, DataLayerError> {
        record.validate()?;
        let id = record.id.clone();
        let group_id = record.group_id.clone();
        let user_id = record.user_id.clone();
        let immutable_record = record.clone();
        let mut tx = self.pool.begin().await.map_sql_err()?;
        let parent_exists = sqlx::query_scalar::<_, String>(
            "SELECT id FROM asset_groups WHERE id = ? AND user_id = ? AND deleted_at_unix_secs IS NULL FOR UPDATE",
        )
        .bind(&group_id)
        .bind(&user_id)
        .fetch_optional(&mut *tx)
        .await
        .map_sql_err()?
        .is_some();
        if !parent_exists {
            return Err(DataLayerError::InvalidInput(format!(
                "assets.group_id {:?} does not exist or belongs to another user",
                group_id
            )));
        }
        if let Some(existing) = map_optional_asset(
            sqlx::query(&format!("{ASSET_COLUMNS} WHERE id = ? LIMIT 1 FOR UPDATE"))
                .bind(&id)
                .fetch_optional(&mut *tx)
                .await
                .map_sql_err()?,
        )? {
            if !record.has_same_immutable_identity(&existing) {
                return Err(DataLayerError::InvalidInput(
                    "asset immutable identity cannot be changed".to_string(),
                ));
            }
        }
        sqlx::query(
            r#"
INSERT INTO assets (
  id, upstream_asset_id, group_id, user_id, api_key_id, asset_type, name, status,
  error_code, error_message, moderation, last_inference_at_unix_secs,
  source_url_fingerprint, provider_url, provider_url_expires_at_unix_secs,
  sanitized_metadata, is_deleted, deleted_at_unix_secs,
  created_at_unix_secs, updated_at_unix_secs
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON DUPLICATE KEY UPDATE
  api_key_id = IF(id = VALUES(id), VALUES(api_key_id), api_key_id),
  name = IF(id = VALUES(id), VALUES(name), name),
  status = IF(id = VALUES(id), VALUES(status), status),
  error_code = IF(id = VALUES(id), VALUES(error_code), error_code),
  error_message = IF(id = VALUES(id), VALUES(error_message), error_message),
  moderation = IF(id = VALUES(id), VALUES(moderation), moderation),
  last_inference_at_unix_secs = IF(id = VALUES(id), VALUES(last_inference_at_unix_secs), last_inference_at_unix_secs),
  source_url_fingerprint = IF(id = VALUES(id), VALUES(source_url_fingerprint), source_url_fingerprint),
  provider_url = IF(id = VALUES(id), VALUES(provider_url), provider_url),
  provider_url_expires_at_unix_secs = IF(id = VALUES(id), VALUES(provider_url_expires_at_unix_secs), provider_url_expires_at_unix_secs),
  sanitized_metadata = IF(id = VALUES(id), VALUES(sanitized_metadata), sanitized_metadata),
  updated_at_unix_secs = IF(id = VALUES(id), VALUES(updated_at_unix_secs), updated_at_unix_secs)
"#,
        )
        .bind(record.id)
        .bind(record.upstream_asset_id)
        .bind(record.group_id)
        .bind(record.user_id)
        .bind(record.api_key_id)
        .bind(record.asset_type)
        .bind(record.name)
        .bind(record.status)
        .bind(record.error_code)
        .bind(record.error_message)
        .bind(json_text(record.moderation))
        .bind(to_optional_i64(
            record.last_inference_at_unix_secs,
            "asset last_inference_at",
        )?)
        .bind(record.source_url_fingerprint)
        .bind(record.provider_url)
        .bind(to_optional_i64(
            record.provider_url_expires_at_unix_secs,
            "asset provider_url_expires_at",
        )?)
        .bind(json_text(record.sanitized_metadata))
        .bind(record.is_deleted)
        .bind(to_optional_i64(
            record.deleted_at_unix_secs,
            "asset deleted_at",
        )?)
        .bind(to_i64(record.created_at_unix_secs, "asset created_at")?)
        .bind(to_i64(record.updated_at_unix_secs, "asset updated_at")?)
        .execute(&mut *tx)
        .await
        .map_sql_err()?;
        let asset = map_optional_asset(
            sqlx::query(&format!("{ASSET_COLUMNS} WHERE id = ? LIMIT 1"))
                .bind(&id)
                .fetch_optional(&mut *tx)
                .await
                .map_sql_err()?,
        )?
        .ok_or_else(|| {
            DataLayerError::InvalidInput(format!(
                "assets.group_id {:?} is invalid or the upstream asset identity is already bound",
                group_id
            ))
        })?;
        if !immutable_record.has_same_immutable_identity(&asset) {
            return Err(DataLayerError::InvalidInput(
                "asset immutable identity cannot be changed".to_string(),
            ));
        }
        tx.commit().await.map_sql_err()?;
        Ok(asset)
    }

    async fn soft_delete_group(
        &self,
        group_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, DataLayerError> {
        let deleted_at = nonzero_i64(deleted_at_unix_secs, "asset group deleted_at")?;
        let mut tx = self.pool.begin().await.map_sql_err()?;
        let group_exists =
            sqlx::query_scalar::<_, String>("SELECT id FROM asset_groups WHERE id = ? FOR UPDATE")
                .bind(group_id)
                .fetch_optional(&mut *tx)
                .await
                .map_sql_err()?
                .is_some();
        if !group_exists {
            tx.commit().await.map_sql_err()?;
            return Ok(false);
        }
        let deleted = sqlx::query(
            "UPDATE asset_groups SET status = 'deleted', deleted_at_unix_secs = ?, updated_at_unix_secs = ? WHERE id = ? AND deleted_at_unix_secs IS NULL",
        )
        .bind(deleted_at)
        .bind(deleted_at)
        .bind(group_id)
        .execute(&mut *tx)
        .await
        .map_sql_err()?
        .rows_affected()
            > 0;
        if deleted {
            sqlx::query(
                "UPDATE assets SET status = 'deleted', is_deleted = 1, deleted_at_unix_secs = ?, updated_at_unix_secs = ? WHERE group_id = ? AND is_deleted = 0",
            )
            .bind(deleted_at)
            .bind(deleted_at)
            .bind(group_id)
            .execute(&mut *tx)
            .await
            .map_sql_err()?;
        }
        tx.commit().await.map_sql_err()?;
        Ok(deleted)
    }

    async fn soft_delete_asset(
        &self,
        asset_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, DataLayerError> {
        let deleted_at = nonzero_i64(deleted_at_unix_secs, "asset deleted_at")?;
        Ok(sqlx::query(
            "UPDATE assets SET status = 'deleted', is_deleted = 1, deleted_at_unix_secs = ?, updated_at_unix_secs = ? WHERE id = ? AND is_deleted = 0",
        )
        .bind(deleted_at)
        .bind(deleted_at)
        .bind(asset_id)
        .execute(&self.pool)
        .await
        .map_sql_err()?
        .rows_affected()
            > 0)
    }

    async fn upsert_visual_validation_session(
        &self,
        record: UpsertArkVisualValidationSessionRecord,
    ) -> Result<StoredArkVisualValidationSession, DataLayerError> {
        record.validate()?;
        let id = record.id.clone();
        let group_id = record.group_id.clone();
        let user_id = record.user_id.clone();
        let provider_id = record.provider_id.clone();
        let immutable_record = record.clone();
        let mut tx = self.pool.begin().await.map_sql_err()?;
        lock_active_catalog_binding(
            &mut tx,
            &record.provider_id,
            &record.endpoint_id,
            &record.key_id,
        )
        .await?;
        if let Some(group_id) = group_id.as_deref() {
            let valid_group = sqlx::query_scalar::<_, String>(
                "SELECT id FROM asset_groups WHERE id = ? AND user_id = ? AND provider_id = ? AND deleted_at_unix_secs IS NULL FOR UPDATE",
            )
            .bind(group_id)
            .bind(&record.user_id)
            .bind(&record.provider_id)
            .fetch_optional(&mut *tx)
            .await
            .map_sql_err()?
            .is_some();
            if !valid_group {
                return Err(DataLayerError::InvalidInput(
                    "validation session group owner or provider binding is invalid".to_string(),
                ));
            }
        }
        if let Some(existing) = map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE id = ? LIMIT 1 FOR UPDATE"
            ))
            .bind(&id)
            .fetch_optional(&mut *tx)
            .await
            .map_sql_err()?,
        )? {
            if !record.has_same_immutable_identity(&existing) {
                return Err(DataLayerError::InvalidInput(
                    "validation session immutable identity cannot be changed".to_string(),
                ));
            }
            if existing.consumed_at_unix_secs.is_some() {
                tx.commit().await.map_sql_err()?;
                return Ok(existing);
            }
        }
        sqlx::query(
            r#"
INSERT INTO ark_visual_validation_sessions (
  id, session_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
  byted_token_hash, encrypted_byted_token,
  callback_state_hash, status, expires_at_unix_secs, consumed_at_unix_secs,
  group_id, sanitized_result, created_at_unix_secs, updated_at_unix_secs
) SELECT ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
   WHERE ? IS NULL OR EXISTS (
     SELECT 1 FROM asset_groups
      WHERE id = ? AND user_id = ? AND provider_id = ? AND deleted_at_unix_secs IS NULL
   )
ON DUPLICATE KEY UPDATE
  api_key_id = IF(id = VALUES(id) AND consumed_at_unix_secs IS NULL, VALUES(api_key_id), api_key_id),
  status = IF(id = VALUES(id) AND consumed_at_unix_secs IS NULL, VALUES(status), status),
  group_id = IF(id = VALUES(id) AND consumed_at_unix_secs IS NULL, COALESCE(group_id, VALUES(group_id)), group_id),
  sanitized_result = IF(id = VALUES(id) AND consumed_at_unix_secs IS NULL, VALUES(sanitized_result), sanitized_result),
  updated_at_unix_secs = IF(id = VALUES(id) AND consumed_at_unix_secs IS NULL, VALUES(updated_at_unix_secs), updated_at_unix_secs),
  consumed_at_unix_secs = IF(id = VALUES(id) AND consumed_at_unix_secs IS NULL, VALUES(consumed_at_unix_secs), consumed_at_unix_secs)
"#,
        )
        .bind(record.id)
        .bind(record.session_id)
        .bind(record.user_id)
        .bind(record.api_key_id)
        .bind(record.provider_id)
        .bind(record.endpoint_id)
        .bind(record.key_id)
        .bind(record.byted_token_hash)
        .bind(record.encrypted_byted_token)
        .bind(record.callback_state_hash)
        .bind(record.status)
        .bind(to_i64(
            record.expires_at_unix_secs,
            "validation expires_at",
        )?)
        .bind(to_optional_i64(
            record.consumed_at_unix_secs,
            "validation consumed_at",
        )?)
        .bind(record.group_id)
        .bind(json_text(record.sanitized_result))
        .bind(to_i64(
            record.created_at_unix_secs,
            "validation created_at",
        )?)
        .bind(to_i64(
            record.updated_at_unix_secs,
            "validation updated_at",
        )?)
        .bind(&group_id)
        .bind(&group_id)
        .bind(&user_id)
        .bind(&provider_id)
        .execute(&mut *tx)
        .await
        .map_sql_err()?;
        let session = map_optional_session(
            sqlx::query(&format!("{SESSION_COLUMNS} WHERE id = ? LIMIT 1"))
                .bind(&id)
                .fetch_optional(&mut *tx)
                .await
                .map_sql_err()?,
        )?
        .ok_or_else(|| {
            DataLayerError::InvalidInput(
                "validation session identity or group ownership is invalid".to_string(),
            )
        })?;
        if !immutable_record.has_same_immutable_identity(&session) {
            return Err(DataLayerError::InvalidInput(
                "validation session immutable or canonical upstream identity conflicts with an existing record"
                    .to_string(),
            ));
        }
        tx.commit().await.map_sql_err()?;
        Ok(session)
    }

    async fn consume_visual_validation_session(
        &self,
        record: ConsumeArkVisualValidationSessionRecord,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        record.validate()?;
        let callback_state_hash = record.callback_state_hash;
        let rows_affected = sqlx::query(
            r#"
UPDATE ark_visual_validation_sessions
SET status = ?,
    consumed_at_unix_secs = ?,
    sanitized_result = ?,
    updated_at_unix_secs = ?
WHERE callback_state_hash = ?
  AND consumed_at_unix_secs IS NULL
  AND expires_at_unix_secs > ?
"#,
        )
        .bind(record.status)
        .bind(to_i64(
            record.consumed_at_unix_secs,
            "validation consumed_at",
        )?)
        .bind(json_text(record.sanitized_result))
        .bind(to_i64(
            record.updated_at_unix_secs,
            "validation updated_at",
        )?)
        .bind(&callback_state_hash)
        .bind(to_i64(
            record.consumed_at_unix_secs,
            "validation consumed_at",
        )?)
        .execute(&self.pool)
        .await
        .map_sql_err()?
        .rows_affected();
        if rows_affected == 0 {
            return Ok(None);
        }
        self.find_visual_validation_session_by_callback_state_hash(&callback_state_hash)
            .await
    }
}

fn group_count_query(query: &AssetGroupListQuery) -> QueryBuilder<'_, MySql> {
    let mut builder = QueryBuilder::<MySql>::new("SELECT COUNT(*) FROM asset_groups");
    let mut where_clause = WhereClause::new();
    group_filters(&mut builder, &mut where_clause, query);
    builder
}

fn group_rows_query(query: &AssetGroupListQuery) -> QueryBuilder<'_, MySql> {
    let mut builder = QueryBuilder::<MySql>::new(GROUP_COLUMNS);
    let mut where_clause = WhereClause::new();
    group_filters(&mut builder, &mut where_clause, query);
    builder.push(" ORDER BY created_at_unix_secs DESC, updated_at_unix_secs DESC, id ASC");
    push_limit_offset(
        &mut builder,
        page_value(query.limit),
        page_value(query.offset),
    );
    builder
}

fn group_filters<'a>(
    builder: &mut QueryBuilder<'a, MySql>,
    where_clause: &mut WhereClause,
    query: &'a AssetGroupListQuery,
) {
    push_optional_eq(builder, where_clause, "user_id", query.user_id.as_deref());
    push_optional_eq(
        builder,
        where_clause,
        "api_key_id",
        query.api_key_id.as_deref(),
    );
    push_optional_eq(
        builder,
        where_clause,
        "provider_id",
        query.provider_id.as_deref(),
    );
    push_optional_eq(
        builder,
        where_clause,
        "group_type",
        query.group_type.as_deref(),
    );
    push_optional_eq(builder, where_clause, "status", query.status.as_deref());
    if !query.include_deleted {
        where_clause.push_next(builder);
        builder.push("deleted_at_unix_secs IS NULL");
    }
    if let Some(search) = trimmed(query.search.as_deref()) {
        push_ci_contains_any(
            builder,
            where_clause,
            SqlDialect::MySql,
            &[
                "id",
                "name",
                "COALESCE(upstream_group_id, '')",
                "COALESCE(description, '')",
            ],
            search,
        );
    }
}

fn asset_count_query(query: &AssetListQuery) -> QueryBuilder<'_, MySql> {
    let mut builder = QueryBuilder::<MySql>::new("SELECT COUNT(*) FROM assets");
    let mut where_clause = WhereClause::new();
    asset_filters(&mut builder, &mut where_clause, query);
    builder
}

fn asset_rows_query(query: &AssetListQuery) -> QueryBuilder<'_, MySql> {
    let mut builder = QueryBuilder::<MySql>::new(ASSET_COLUMNS);
    let mut where_clause = WhereClause::new();
    asset_filters(&mut builder, &mut where_clause, query);
    builder.push(" ORDER BY created_at_unix_secs DESC, updated_at_unix_secs DESC, id ASC");
    push_limit_offset(
        &mut builder,
        page_value(query.limit),
        page_value(query.offset),
    );
    builder
}

fn asset_filters<'a>(
    builder: &mut QueryBuilder<'a, MySql>,
    where_clause: &mut WhereClause,
    query: &'a AssetListQuery,
) {
    push_optional_eq(builder, where_clause, "group_id", query.group_id.as_deref());
    push_optional_eq(builder, where_clause, "user_id", query.user_id.as_deref());
    push_optional_eq(
        builder,
        where_clause,
        "api_key_id",
        query.api_key_id.as_deref(),
    );
    push_optional_eq(
        builder,
        where_clause,
        "asset_type",
        query.asset_type.as_deref(),
    );
    push_optional_eq(builder, where_clause, "status", query.status.as_deref());
    if !query.include_deleted {
        where_clause.push_next(builder);
        builder.push("is_deleted = 0");
    }
    if let Some(search) = trimmed(query.search.as_deref()) {
        push_ci_contains_any(
            builder,
            where_clause,
            SqlDialect::MySql,
            &["id", "name", "COALESCE(upstream_asset_id, '')"],
            search,
        );
    }
}

fn push_optional_eq<'a>(
    builder: &mut QueryBuilder<'a, MySql>,
    where_clause: &mut WhereClause,
    column: &str,
    value: Option<&'a str>,
) {
    if let Some(value) = value {
        where_clause.push_next(builder);
        builder.push(column).push(" = ").push_bind(value);
    }
}

fn map_optional_group(row: Option<MySqlRow>) -> Result<Option<StoredAssetGroup>, DataLayerError> {
    row.as_ref().map(map_group).transpose()
}

fn map_optional_asset(row: Option<MySqlRow>) -> Result<Option<StoredAsset>, DataLayerError> {
    row.as_ref().map(map_asset).transpose()
}

fn map_optional_session(
    row: Option<MySqlRow>,
) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
    row.as_ref().map(map_session).transpose()
}

fn map_group(row: &MySqlRow) -> Result<StoredAssetGroup, DataLayerError> {
    Ok(StoredAssetGroup {
        id: row.try_get("id").map_sql_err()?,
        upstream_group_id: row.try_get("upstream_group_id").map_sql_err()?,
        user_id: row.try_get("user_id").map_sql_err()?,
        api_key_id: row.try_get("api_key_id").map_sql_err()?,
        provider_id: row.try_get("provider_id").map_sql_err()?,
        endpoint_id: row.try_get("endpoint_id").map_sql_err()?,
        key_id: row.try_get("key_id").map_sql_err()?,
        group_type: row.try_get("group_type").map_sql_err()?,
        name: row.try_get("name").map_sql_err()?,
        description: row.try_get("description").map_sql_err()?,
        status: row.try_get("status").map_sql_err()?,
        created_at_unix_secs: read_u64(row, "created_at_unix_secs", "asset_groups")?,
        updated_at_unix_secs: read_u64(row, "updated_at_unix_secs", "asset_groups")?,
        deleted_at_unix_secs: read_optional_u64(row, "deleted_at_unix_secs", "asset_groups")?,
    })
}

fn map_asset(row: &MySqlRow) -> Result<StoredAsset, DataLayerError> {
    Ok(StoredAsset {
        id: row.try_get("id").map_sql_err()?,
        upstream_asset_id: row.try_get("upstream_asset_id").map_sql_err()?,
        group_id: row.try_get("group_id").map_sql_err()?,
        user_id: row.try_get("user_id").map_sql_err()?,
        api_key_id: row.try_get("api_key_id").map_sql_err()?,
        asset_type: row.try_get("asset_type").map_sql_err()?,
        name: row.try_get("name").map_sql_err()?,
        status: row.try_get("status").map_sql_err()?,
        error_code: row.try_get("error_code").map_sql_err()?,
        error_message: row.try_get("error_message").map_sql_err()?,
        moderation: parse_json(
            row.try_get("moderation").map_sql_err()?,
            "assets.moderation",
        )?,
        last_inference_at_unix_secs: read_optional_u64(
            row,
            "last_inference_at_unix_secs",
            "assets",
        )?,
        source_url_fingerprint: row.try_get("source_url_fingerprint").map_sql_err()?,
        provider_url: row.try_get("provider_url").map_sql_err()?,
        provider_url_expires_at_unix_secs: read_optional_u64(
            row,
            "provider_url_expires_at_unix_secs",
            "assets",
        )?,
        sanitized_metadata: parse_json(
            row.try_get("sanitized_metadata").map_sql_err()?,
            "assets.sanitized_metadata",
        )?,
        is_deleted: row.try_get("is_deleted").map_sql_err()?,
        deleted_at_unix_secs: read_optional_u64(row, "deleted_at_unix_secs", "assets")?,
        created_at_unix_secs: read_u64(row, "created_at_unix_secs", "assets")?,
        updated_at_unix_secs: read_u64(row, "updated_at_unix_secs", "assets")?,
    })
}

fn map_session(row: &MySqlRow) -> Result<StoredArkVisualValidationSession, DataLayerError> {
    Ok(StoredArkVisualValidationSession {
        id: row.try_get("id").map_sql_err()?,
        session_id: row.try_get("session_id").map_sql_err()?,
        user_id: row.try_get("user_id").map_sql_err()?,
        api_key_id: row.try_get("api_key_id").map_sql_err()?,
        provider_id: row.try_get("provider_id").map_sql_err()?,
        endpoint_id: row.try_get("endpoint_id").map_sql_err()?,
        key_id: row.try_get("key_id").map_sql_err()?,
        byted_token_hash: row.try_get("byted_token_hash").map_sql_err()?,
        encrypted_byted_token: row.try_get("encrypted_byted_token").map_sql_err()?,
        callback_state_hash: row.try_get("callback_state_hash").map_sql_err()?,
        status: row.try_get("status").map_sql_err()?,
        expires_at_unix_secs: read_u64(
            row,
            "expires_at_unix_secs",
            "ark_visual_validation_sessions",
        )?,
        consumed_at_unix_secs: read_optional_u64(
            row,
            "consumed_at_unix_secs",
            "ark_visual_validation_sessions",
        )?,
        group_id: row.try_get("group_id").map_sql_err()?,
        sanitized_result: parse_json(
            row.try_get("sanitized_result").map_sql_err()?,
            "ark_visual_validation_sessions.sanitized_result",
        )?,
        created_at_unix_secs: read_u64(
            row,
            "created_at_unix_secs",
            "ark_visual_validation_sessions",
        )?,
        updated_at_unix_secs: read_u64(
            row,
            "updated_at_unix_secs",
            "ark_visual_validation_sessions",
        )?,
    })
}

fn read_u64(row: &MySqlRow, field: &str, table: &str) -> Result<u64, DataLayerError> {
    let value: i64 = row.try_get(field).map_sql_err()?;
    u64::try_from(value).map_err(|_| {
        DataLayerError::UnexpectedValue(format!("{table}.{field} is negative: {value}"))
    })
}

fn read_optional_u64(
    row: &MySqlRow,
    field: &str,
    table: &str,
) -> Result<Option<u64>, DataLayerError> {
    let value: Option<i64> = row.try_get(field).map_sql_err()?;
    value
        .map(|value| {
            u64::try_from(value).map_err(|_| {
                DataLayerError::UnexpectedValue(format!("{table}.{field} is negative: {value}"))
            })
        })
        .transpose()
}

fn parse_json(
    raw: Option<String>,
    field: &str,
) -> Result<Option<serde_json::Value>, DataLayerError> {
    raw.map(|raw| {
        serde_json::from_str(&raw).map_err(|error| {
            DataLayerError::UnexpectedValue(format!("{field} contains invalid JSON: {error}"))
        })
    })
    .transpose()
}

fn json_text(value: Option<serde_json::Value>) -> Option<String> {
    value.map(|value| value.to_string())
}

fn to_i64(value: u64, field: &str) -> Result<i64, DataLayerError> {
    i64::try_from(value)
        .map_err(|_| DataLayerError::InvalidInput(format!("{field} exceeds i64: {value}")))
}

fn nonzero_i64(value: u64, field: &str) -> Result<i64, DataLayerError> {
    if value == 0 {
        return Err(DataLayerError::InvalidInput(format!("{field} is empty")));
    }
    to_i64(value, field)
}

fn to_optional_i64(value: Option<u64>, field: &str) -> Result<Option<i64>, DataLayerError> {
    value.map(|value| to_i64(value, field)).transpose()
}

fn page_value(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn trimmed(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn reference_count(value: i64, table: &str) -> Result<u64, DataLayerError> {
    u64::try_from(value).map_err(|_| {
        DataLayerError::UnexpectedValue(format!(
            "invalid {table} provider reference count: {value}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        asset_rows_query, group_rows_query, provider_reference_counts_sql,
        MysqlAssetLibraryRepository,
    };
    use aether_data_contracts::repository::asset_library::{
        AssetGroupListQuery, AssetListQuery, AssetProviderReference,
    };
    use sqlx::Execute;

    #[test]
    fn renders_mysql_list_queries() {
        let group_query = AssetGroupListQuery {
            user_id: Some("user-1".to_string()),
            limit: 20,
            ..AssetGroupListQuery::default()
        };
        let mut groups = group_rows_query(&group_query);
        assert!(groups.build().sql().contains("user_id = ?"));

        let asset_query = AssetListQuery {
            status: Some("Active".to_string()),
            limit: 20,
            ..AssetListQuery::default()
        };
        let mut assets = asset_rows_query(&asset_query);
        assert!(assets.build().sql().contains("is_deleted = 0"));
    }

    #[tokio::test]
    async fn builds_from_lazy_pool() {
        let pool = sqlx::mysql::MySqlPoolOptions::new().connect_lazy_with(
            "mysql://user:pass@localhost:3306/aether"
                .parse()
                .expect("mysql options"),
        );
        let _repository = MysqlAssetLibraryRepository::new(pool);
    }

    #[test]
    fn provider_reference_queries_include_audit_rows_for_every_target() {
        for (reference, column) in [
            (
                AssetProviderReference::ProviderId("provider-1"),
                "provider_id",
            ),
            (
                AssetProviderReference::EndpointId("endpoint-1"),
                "endpoint_id",
            ),
            (AssetProviderReference::KeyId("key-1"), "key_id"),
        ] {
            let sql = provider_reference_counts_sql(reference);
            assert!(sql.contains("asset_groups"));
            assert!(sql.contains("ark_visual_validation_sessions"));
            assert_eq!(sql.matches(&format!("{column} = ?")).count(), 2);
            assert!(!sql.contains("deleted_at"));
        }
    }

    #[test]
    fn asset_writes_share_the_parent_row_lock_protocol() {
        let source = include_str!("asset_library.rs");
        assert!(source.contains(
            "SELECT id FROM asset_groups WHERE id = ? AND user_id = ? AND deleted_at_unix_secs IS NULL FOR UPDATE"
        ));
        assert!(source.contains("SELECT id FROM asset_groups WHERE id = ? FOR UPDATE"));
        let mutable_group_identity = [
            "group_id = IF(id = VALUES(id), ",
            "VALUES(group_id), group_id)",
        ]
        .concat();
        let mutable_tombstone = [
            "is_deleted = IF(id = VALUES(id), ",
            "VALUES(is_deleted), is_deleted)",
        ]
        .concat();
        assert!(!source.contains(&mutable_group_identity));
        assert!(!source.contains(&mutable_tombstone));
    }

    #[test]
    fn provider_binding_migration_uses_canonical_upstream_identity_and_restrictive_provider_fks() {
        let schema = include_str!("../migrations/20260818000000_add_asset_library.sql");
        let provider_binding =
            include_str!("../migrations/20260818010000_bind_assets_to_provider.sql");
        assert!(provider_binding.contains(
            "ADD UNIQUE KEY uq_asset_groups_upstream (`provider_id`, `upstream_group_id`)"
        ));
        assert!(provider_binding
            .contains("ADD UNIQUE KEY uq_ark_validation_upstream (`provider_id`, `session_id`)"));
        assert_eq!(
            provider_binding
                .matches("DROP COLUMN account_binding")
                .count(),
            2
        );
        assert_eq!(provider_binding.matches("DROP COLUMN project").count(), 2);
        for constraint in [
            "fk_asset_groups_provider",
            "fk_asset_groups_endpoint",
            "fk_asset_groups_key",
            "fk_ark_validation_provider",
            "fk_ark_validation_endpoint",
            "fk_ark_validation_key",
        ] {
            let line = schema
                .lines()
                .find(|line| line.contains(constraint))
                .expect("material provider foreign key");
            assert!(line.contains("ON DELETE RESTRICT"), "{line}");
        }
    }
}
