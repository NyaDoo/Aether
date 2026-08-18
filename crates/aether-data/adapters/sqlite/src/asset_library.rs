use async_trait::async_trait;
use sqlx::{sqlite::SqliteRow, QueryBuilder, Row, Sqlite};

use aether_data_contracts::repository::asset_library::*;
use aether_data_contracts::DataLayerError;
use aether_data_query::{push_ci_contains_any, push_limit_offset, SqlDialect, WhereClause};

use crate::error::SqlResultExt;
use crate::SqlitePool;

const GROUP_COLUMNS: &str = r#"
SELECT id, upstream_group_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
       account_binding, project, group_type, name, description, status,
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
       account_binding, project, byted_token_hash, encrypted_byted_token,
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
pub struct SqliteAssetLibraryRepository {
    pool: SqlitePool,
}

impl SqliteAssetLibraryRepository {
    pub fn new(pool: SqlitePool) -> Self {
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
    transaction: &mut sqlx::Transaction<'_, Sqlite>,
    provider_id: &str,
    endpoint_id: &str,
    key_id: &str,
) -> Result<(), DataLayerError> {
    sqlx::query("UPDATE providers SET id = id WHERE id = ?")
        .bind(provider_id)
        .execute(&mut **transaction)
        .await
        .map_sql_err()?;
    let binding_exists = sqlx::query_scalar::<_, String>(
        r#"
SELECT p.id
  FROM providers p
  JOIN provider_endpoints e ON e.id = ? AND e.provider_id = p.id AND e.is_active = 1
  JOIN provider_api_keys k ON k.id = ? AND k.provider_id = p.id AND k.is_active = 1
 WHERE p.id = ? AND p.is_active = 1
"#,
    )
    .bind(endpoint_id)
    .bind(key_id)
    .bind(provider_id)
    .fetch_optional(&mut **transaction)
    .await
    .map_sql_err()?
    .is_some();
    if !binding_exists {
        return Err(DataLayerError::InvalidInput(
            "asset library provider, endpoint, or key binding is inactive or invalid".to_string(),
        ));
    }
    Ok(())
}

#[async_trait]
impl AssetLibraryReadRepository for SqliteAssetLibraryRepository {
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
        account_binding: &str,
        project: Option<&str>,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        map_optional_group(
            sqlx::query(&format!(
                "{GROUP_COLUMNS} WHERE provider_id = ? AND account_binding = ? AND COALESCE(project, '') = ? AND upstream_group_id = ? LIMIT 1"
            ))
            .bind(provider_id)
            .bind(account_binding)
            .bind(project.unwrap_or_default())
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
        account_binding: &str,
        project: Option<&str>,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE provider_id = ? AND account_binding = ? AND COALESCE(project, '') = ? AND session_id = ? LIMIT 1"
            ))
            .bind(provider_id)
            .bind(account_binding)
            .bind(project.unwrap_or_default())
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
impl AssetLibraryWriteRepository for SqliteAssetLibraryRepository {
    async fn upsert_group(
        &self,
        record: UpsertAssetGroupRecord,
    ) -> Result<StoredAssetGroup, DataLayerError> {
        record.validate()?;
        let id = record.id.clone();
        let immutable_record = record.clone();
        let project = storage_project(record.project.clone());
        let mut tx = self.pool.begin().await.map_sql_err()?;
        lock_active_catalog_binding(
            &mut tx,
            &record.provider_id,
            &record.endpoint_id,
            &record.key_id,
        )
        .await?;
        if let Some(existing) = map_optional_group(
            sqlx::query(&format!("{GROUP_COLUMNS} WHERE id = ? LIMIT 1"))
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
  account_binding, project, group_type, name, description, status,
  created_at_unix_secs, updated_at_unix_secs, deleted_at_unix_secs
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(id) DO UPDATE SET
  api_key_id = excluded.api_key_id,
  endpoint_id = excluded.endpoint_id,
  key_id = excluded.key_id,
  name = excluded.name,
  description = excluded.description,
  status = excluded.status,
  updated_at_unix_secs = excluded.updated_at_unix_secs
WHERE asset_groups.upstream_group_id IS excluded.upstream_group_id
  AND asset_groups.user_id = excluded.user_id
  AND asset_groups.provider_id = excluded.provider_id
  AND asset_groups.account_binding = excluded.account_binding
  AND asset_groups.project = excluded.project
  AND asset_groups.group_type = excluded.group_type
  AND asset_groups.deleted_at_unix_secs IS excluded.deleted_at_unix_secs
  AND (asset_groups.deleted_at_unix_secs IS NULL OR asset_groups.status = excluded.status)
"#,
        )
        .bind(record.id)
        .bind(record.upstream_group_id)
        .bind(record.user_id)
        .bind(record.api_key_id)
        .bind(record.provider_id)
        .bind(record.endpoint_id)
        .bind(record.key_id)
        .bind(record.account_binding)
        .bind(project)
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
            DataLayerError::UnexpectedValue("upserted asset group is missing".to_string())
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
        sqlx::query("UPDATE asset_groups SET id = id WHERE id = ?")
            .bind(&group_id)
            .execute(&mut *tx)
            .await
            .map_sql_err()?;
        let parent_exists = sqlx::query_scalar::<_, String>(
            "SELECT id FROM asset_groups WHERE id = ? AND user_id = ? AND deleted_at_unix_secs IS NULL",
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
            sqlx::query(&format!("{ASSET_COLUMNS} WHERE id = ? LIMIT 1"))
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
ON CONFLICT(id) DO UPDATE SET
  api_key_id = excluded.api_key_id,
  name = excluded.name,
  status = excluded.status,
  error_code = excluded.error_code,
  error_message = excluded.error_message,
  moderation = excluded.moderation,
  last_inference_at_unix_secs = excluded.last_inference_at_unix_secs,
  source_url_fingerprint = excluded.source_url_fingerprint,
  provider_url = excluded.provider_url,
  provider_url_expires_at_unix_secs = excluded.provider_url_expires_at_unix_secs,
  sanitized_metadata = excluded.sanitized_metadata,
  updated_at_unix_secs = excluded.updated_at_unix_secs
WHERE assets.upstream_asset_id IS excluded.upstream_asset_id
  AND assets.group_id = excluded.group_id
  AND assets.user_id = excluded.user_id
  AND assets.asset_type = excluded.asset_type
  AND assets.is_deleted = excluded.is_deleted
  AND assets.deleted_at_unix_secs IS excluded.deleted_at_unix_secs
  AND (assets.is_deleted = 0 OR assets.status = excluded.status)
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
                "assets.group_id {:?} does not exist or belongs to another user",
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
        sqlx::query("UPDATE asset_groups SET id = id WHERE id = ?")
            .bind(group_id)
            .execute(&mut *tx)
            .await
            .map_sql_err()?;
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
        let immutable_record = record.clone();
        let project = storage_project(record.project.clone());
        let mut tx = self.pool.begin().await.map_sql_err()?;
        lock_active_catalog_binding(
            &mut tx,
            &record.provider_id,
            &record.endpoint_id,
            &record.key_id,
        )
        .await?;
        if let Some(group_id) = group_id.as_deref() {
            sqlx::query("UPDATE asset_groups SET id = id WHERE id = ?")
                .bind(group_id)
                .execute(&mut *tx)
                .await
                .map_sql_err()?;
            let valid_group = sqlx::query_scalar::<_, String>(
                "SELECT id FROM asset_groups WHERE id = ? AND user_id = ? AND provider_id = ? AND account_binding = ? AND project = ? AND deleted_at_unix_secs IS NULL",
            )
            .bind(group_id)
            .bind(&record.user_id)
            .bind(&record.provider_id)
            .bind(&record.account_binding)
            .bind(&project)
            .fetch_optional(&mut *tx)
            .await
            .map_sql_err()?
            .is_some();
            if !valid_group {
                return Err(DataLayerError::InvalidInput(
                    "validation session group owner, account, or project binding is invalid"
                        .to_string(),
                ));
            }
        }
        if let Some(existing) = map_optional_session(
            sqlx::query(&format!("{SESSION_COLUMNS} WHERE id = ? LIMIT 1"))
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
  account_binding, project, byted_token_hash, encrypted_byted_token,
  callback_state_hash, status, expires_at_unix_secs, consumed_at_unix_secs,
  group_id, sanitized_result, created_at_unix_secs, updated_at_unix_secs
) SELECT ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
   WHERE ? IS NULL OR EXISTS (
     SELECT 1 FROM asset_groups
      WHERE id = ? AND user_id = ? AND deleted_at_unix_secs IS NULL
   )
ON CONFLICT(id) DO UPDATE SET
  api_key_id = excluded.api_key_id,
  endpoint_id = excluded.endpoint_id,
  key_id = excluded.key_id,
  status = excluded.status,
  consumed_at_unix_secs = excluded.consumed_at_unix_secs,
  group_id = COALESCE(ark_visual_validation_sessions.group_id, excluded.group_id),
  sanitized_result = excluded.sanitized_result,
  updated_at_unix_secs = excluded.updated_at_unix_secs
WHERE ark_visual_validation_sessions.consumed_at_unix_secs IS NULL
  AND ark_visual_validation_sessions.session_id = excluded.session_id
  AND ark_visual_validation_sessions.user_id = excluded.user_id
  AND ark_visual_validation_sessions.provider_id = excluded.provider_id
  AND ark_visual_validation_sessions.account_binding = excluded.account_binding
  AND ark_visual_validation_sessions.project = excluded.project
  AND ark_visual_validation_sessions.byted_token_hash = excluded.byted_token_hash
  AND ark_visual_validation_sessions.encrypted_byted_token = excluded.encrypted_byted_token
  AND ark_visual_validation_sessions.callback_state_hash = excluded.callback_state_hash
  AND ark_visual_validation_sessions.expires_at_unix_secs = excluded.expires_at_unix_secs
  AND (ark_visual_validation_sessions.group_id IS NULL OR ark_visual_validation_sessions.group_id IS excluded.group_id)
"#,
        )
        .bind(record.id)
        .bind(record.session_id)
        .bind(record.user_id)
        .bind(record.api_key_id)
        .bind(record.provider_id)
        .bind(record.endpoint_id)
        .bind(record.key_id)
        .bind(record.account_binding)
        .bind(project)
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
                "validation session group does not exist or belongs to another user".to_string(),
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
        let row = sqlx::query(
            r#"
UPDATE ark_visual_validation_sessions
SET status = ?,
    consumed_at_unix_secs = ?,
    sanitized_result = ?,
    updated_at_unix_secs = ?
WHERE callback_state_hash = ?
  AND consumed_at_unix_secs IS NULL
  AND expires_at_unix_secs > ?
RETURNING *
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
        .bind(record.callback_state_hash)
        .bind(to_i64(
            record.consumed_at_unix_secs,
            "validation consumed_at",
        )?)
        .fetch_optional(&self.pool)
        .await
        .map_sql_err()?;
        map_optional_session(row)
    }
}

fn group_count_query(query: &AssetGroupListQuery) -> QueryBuilder<'_, Sqlite> {
    let mut builder = QueryBuilder::<Sqlite>::new("SELECT COUNT(*) FROM asset_groups");
    let mut where_clause = WhereClause::new();
    group_filters(&mut builder, &mut where_clause, query);
    builder
}

fn group_rows_query(query: &AssetGroupListQuery) -> QueryBuilder<'_, Sqlite> {
    let mut builder = QueryBuilder::<Sqlite>::new(GROUP_COLUMNS);
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
    builder: &mut QueryBuilder<'a, Sqlite>,
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
            SqlDialect::Sqlite,
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

fn asset_count_query(query: &AssetListQuery) -> QueryBuilder<'_, Sqlite> {
    let mut builder = QueryBuilder::<Sqlite>::new("SELECT COUNT(*) FROM assets");
    let mut where_clause = WhereClause::new();
    asset_filters(&mut builder, &mut where_clause, query);
    builder
}

fn asset_rows_query(query: &AssetListQuery) -> QueryBuilder<'_, Sqlite> {
    let mut builder = QueryBuilder::<Sqlite>::new(ASSET_COLUMNS);
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
    builder: &mut QueryBuilder<'a, Sqlite>,
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
            SqlDialect::Sqlite,
            &["id", "name", "COALESCE(upstream_asset_id, '')"],
            search,
        );
    }
}

fn push_optional_eq<'a>(
    builder: &mut QueryBuilder<'a, Sqlite>,
    where_clause: &mut WhereClause,
    column: &str,
    value: Option<&'a str>,
) {
    if let Some(value) = value {
        where_clause.push_next(builder);
        builder.push(column).push(" = ").push_bind(value);
    }
}

fn map_optional_group(row: Option<SqliteRow>) -> Result<Option<StoredAssetGroup>, DataLayerError> {
    row.as_ref().map(map_group).transpose()
}

fn map_optional_asset(row: Option<SqliteRow>) -> Result<Option<StoredAsset>, DataLayerError> {
    row.as_ref().map(map_asset).transpose()
}

fn map_optional_session(
    row: Option<SqliteRow>,
) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
    row.as_ref().map(map_session).transpose()
}

fn map_group(row: &SqliteRow) -> Result<StoredAssetGroup, DataLayerError> {
    Ok(StoredAssetGroup {
        id: row.try_get("id").map_sql_err()?,
        upstream_group_id: row.try_get("upstream_group_id").map_sql_err()?,
        user_id: row.try_get("user_id").map_sql_err()?,
        api_key_id: row.try_get("api_key_id").map_sql_err()?,
        provider_id: row.try_get("provider_id").map_sql_err()?,
        endpoint_id: row.try_get("endpoint_id").map_sql_err()?,
        key_id: row.try_get("key_id").map_sql_err()?,
        account_binding: row.try_get("account_binding").map_sql_err()?,
        project: read_project(row.try_get("project").map_sql_err()?),
        group_type: row.try_get("group_type").map_sql_err()?,
        name: row.try_get("name").map_sql_err()?,
        description: row.try_get("description").map_sql_err()?,
        status: row.try_get("status").map_sql_err()?,
        created_at_unix_secs: read_u64(row, "created_at_unix_secs", "asset_groups")?,
        updated_at_unix_secs: read_u64(row, "updated_at_unix_secs", "asset_groups")?,
        deleted_at_unix_secs: read_optional_u64(row, "deleted_at_unix_secs", "asset_groups")?,
    })
}

fn map_asset(row: &SqliteRow) -> Result<StoredAsset, DataLayerError> {
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
        is_deleted: row.try_get::<i64, _>("is_deleted").map_sql_err()? != 0,
        deleted_at_unix_secs: read_optional_u64(row, "deleted_at_unix_secs", "assets")?,
        created_at_unix_secs: read_u64(row, "created_at_unix_secs", "assets")?,
        updated_at_unix_secs: read_u64(row, "updated_at_unix_secs", "assets")?,
    })
}

fn map_session(row: &SqliteRow) -> Result<StoredArkVisualValidationSession, DataLayerError> {
    Ok(StoredArkVisualValidationSession {
        id: row.try_get("id").map_sql_err()?,
        session_id: row.try_get("session_id").map_sql_err()?,
        user_id: row.try_get("user_id").map_sql_err()?,
        api_key_id: row.try_get("api_key_id").map_sql_err()?,
        provider_id: row.try_get("provider_id").map_sql_err()?,
        endpoint_id: row.try_get("endpoint_id").map_sql_err()?,
        key_id: row.try_get("key_id").map_sql_err()?,
        account_binding: row.try_get("account_binding").map_sql_err()?,
        project: read_project(row.try_get("project").map_sql_err()?),
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

fn read_u64(row: &SqliteRow, field: &str, table: &str) -> Result<u64, DataLayerError> {
    let value: i64 = row.try_get(field).map_sql_err()?;
    u64::try_from(value).map_err(|_| {
        DataLayerError::UnexpectedValue(format!("{table}.{field} is negative: {value}"))
    })
}

fn read_optional_u64(
    row: &SqliteRow,
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

fn storage_project(project: Option<String>) -> String {
    project.unwrap_or_default()
}

fn read_project(project: String) -> Option<String> {
    (!project.is_empty()).then_some(project)
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
    use serde_json::json;

    use super::SqliteAssetLibraryRepository;
    use crate::{run_migrations, SqlitePool};
    use aether_data_contracts::repository::asset_library::{
        AssetLibraryReadRepository, AssetLibraryWriteRepository, AssetListQuery,
        AssetProviderReference, AssetProviderReferenceCounts,
        ConsumeArkVisualValidationSessionRecord, UpsertArkVisualValidationSessionRecord,
        UpsertAssetGroupRecord, UpsertAssetRecord,
    };

    async fn seed_dependencies(pool: &SqlitePool) {
        sqlx::query(
            "INSERT INTO users (id, username, password_hash, role, is_active, created_at, updated_at) VALUES ('user-1', 'user-1', 'hash', 'user', 1, 1, 1)",
        )
        .execute(pool)
        .await
        .expect("user seed");
        sqlx::query(
            "INSERT INTO providers (id, name, provider_type, enabled, is_active, created_at, updated_at) VALUES ('provider-1', 'provider-1', 'custom', 1, 1, 1, 1)",
        )
        .execute(pool)
        .await
        .expect("provider seed");
        sqlx::query(
            "INSERT INTO provider_api_keys (id, provider_id, name, encrypted_key, is_active, created_at, updated_at) VALUES ('key-1', 'provider-1', 'key', 'encrypted', 1, 1, 1)",
        )
        .execute(pool)
        .await
        .expect("key seed");
        sqlx::query(
            "INSERT INTO provider_api_keys (id, provider_id, name, encrypted_key, is_active, created_at, updated_at) VALUES ('key-2', 'provider-1', 'key-2', 'encrypted', 1, 1, 1)",
        )
        .execute(pool)
        .await
        .expect("rotated key seed");
        sqlx::query(
            "INSERT INTO provider_endpoints (id, provider_id, name, base_url, enabled, is_active, created_at, updated_at) VALUES ('endpoint-1', 'provider-1', 'endpoint', 'https://example.com', 1, 1, 1, 1)",
        )
        .execute(pool)
        .await
        .expect("endpoint seed");
        sqlx::query(
            "INSERT INTO provider_endpoints (id, provider_id, name, base_url, enabled, is_active, created_at, updated_at) VALUES ('endpoint-2', 'provider-1', 'endpoint-2', 'https://example.com', 1, 1, 1, 1)",
        )
        .execute(pool)
        .await
        .expect("rotated endpoint seed");
    }

    fn group() -> UpsertAssetGroupRecord {
        UpsertAssetGroupRecord {
            id: "group-1".to_string(),
            upstream_group_id: Some("up-group-1".to_string()),
            user_id: "user-1".to_string(),
            api_key_id: None,
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            account_binding: Some("account-1".to_string()),
            project: Some("project-1".to_string()),
            group_type: "AIGC".to_string(),
            name: "Portraits".to_string(),
            description: Some("Face references".to_string()),
            status: "active".to_string(),
            created_at_unix_secs: 10,
            updated_at_unix_secs: 10,
            deleted_at_unix_secs: None,
        }
    }

    #[tokio::test]
    async fn round_trips_assets_and_consumes_session_once() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("sqlite pool");
        run_migrations(&pool).await.expect("migrations");
        seed_dependencies(&pool).await;
        let repository = SqliteAssetLibraryRepository::new(pool);
        repository.upsert_group(group()).await.expect("group");
        repository
            .upsert_asset(UpsertAssetRecord {
                id: "asset-1".to_string(),
                upstream_asset_id: Some("up-asset-1".to_string()),
                group_id: "group-1".to_string(),
                user_id: "user-1".to_string(),
                api_key_id: None,
                asset_type: "Image".to_string(),
                name: "Portrait".to_string(),
                status: "Active".to_string(),
                error_code: None,
                error_message: None,
                moderation: Some(json!({"result": "pass"})),
                last_inference_at_unix_secs: None,
                source_url_fingerprint: Some("fingerprint".to_string()),
                provider_url: Some("https://example.com/a.png".to_string()),
                provider_url_expires_at_unix_secs: Some(100),
                sanitized_metadata: Some(json!({"width": 1024})),
                is_deleted: false,
                deleted_at_unix_secs: None,
                created_at_unix_secs: 10,
                updated_at_unix_secs: 10,
            })
            .await
            .expect("asset");
        assert_eq!(
            repository
                .list_assets(&AssetListQuery {
                    user_id: Some("user-1".to_string()),
                    limit: 20,
                    ..AssetListQuery::default()
                })
                .await
                .expect("list")
                .total,
            1
        );
        repository
            .upsert_visual_validation_session(UpsertArkVisualValidationSessionRecord {
                id: "validation-1".to_string(),
                session_id: "up-session-1".to_string(),
                user_id: "user-1".to_string(),
                api_key_id: None,
                provider_id: "provider-1".to_string(),
                endpoint_id: "endpoint-1".to_string(),
                key_id: "key-1".to_string(),
                account_binding: Some("account-1".to_string()),
                project: Some("project-1".to_string()),
                byted_token_hash: "token-hash".to_string(),
                encrypted_byted_token: "encrypted-token".to_string(),
                callback_state_hash: "state-hash".to_string(),
                status: "pending".to_string(),
                expires_at_unix_secs: 100,
                consumed_at_unix_secs: None,
                group_id: Some("group-1".to_string()),
                sanitized_result: None,
                created_at_unix_secs: 10,
                updated_at_unix_secs: 10,
            })
            .await
            .expect("session");
        let consume = ConsumeArkVisualValidationSessionRecord {
            callback_state_hash: "state-hash".to_string(),
            status: "succeeded".to_string(),
            consumed_at_unix_secs: 50,
            sanitized_result: Some(json!({"verified": true})),
            updated_at_unix_secs: 50,
        };
        assert!(repository
            .consume_visual_validation_session(consume.clone())
            .await
            .expect("consume")
            .is_some());
        let mut retry = UpsertArkVisualValidationSessionRecord {
            id: "validation-1".to_string(),
            session_id: "up-session-1".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: None,
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            account_binding: Some("account-1".to_string()),
            project: Some("project-1".to_string()),
            byted_token_hash: "token-hash".to_string(),
            encrypted_byted_token: "encrypted-token".to_string(),
            callback_state_hash: "state-hash".to_string(),
            status: "pending".to_string(),
            expires_at_unix_secs: 100,
            consumed_at_unix_secs: None,
            group_id: Some("group-1".to_string()),
            sanitized_result: None,
            created_at_unix_secs: 10,
            updated_at_unix_secs: 60,
        };
        let stored = repository
            .upsert_visual_validation_session(retry.clone())
            .await
            .expect("retry consumed session");
        assert_eq!(stored.status, "succeeded");
        assert_eq!(stored.consumed_at_unix_secs, Some(50));
        assert_eq!(
            repository
                .find_visual_validation_session_by_byted_token_hash("token-hash")
                .await
                .expect("BytedToken hash lookup")
                .expect("validation session")
                .id,
            "validation-1"
        );
        retry.id = "validation-2".to_string();
        assert!(repository
            .upsert_visual_validation_session(retry)
            .await
            .is_err());
        assert!(repository
            .consume_visual_validation_session(consume)
            .await
            .expect("consume twice")
            .is_none());
        assert!(repository
            .soft_delete_group("group-1", 60)
            .await
            .expect("soft-delete group"));
        assert_eq!(
            repository
                .list_assets(&AssetListQuery {
                    user_id: Some("user-1".to_string()),
                    limit: 20,
                    ..AssetListQuery::default()
                })
                .await
                .expect("list after group deletion")
                .total,
            0
        );
        let deleted_asset = repository
            .find_asset_by_id("asset-1")
            .await
            .expect("find soft-deleted asset")
            .expect("asset remains for internal reconciliation");
        assert!(deleted_asset.is_deleted);
        assert_eq!(deleted_asset.deleted_at_unix_secs, Some(60));

        let mut resurrected = UpsertAssetRecord {
            id: "asset-1".to_string(),
            upstream_asset_id: Some("up-asset-1".to_string()),
            group_id: "group-1".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: None,
            asset_type: "Image".to_string(),
            name: "Resurrected".to_string(),
            status: "Active".to_string(),
            error_code: None,
            error_message: None,
            moderation: None,
            last_inference_at_unix_secs: None,
            source_url_fingerprint: None,
            provider_url: None,
            provider_url_expires_at_unix_secs: None,
            sanitized_metadata: None,
            is_deleted: false,
            deleted_at_unix_secs: None,
            created_at_unix_secs: 10,
            updated_at_unix_secs: 61,
        };
        assert!(repository.upsert_asset(resurrected.clone()).await.is_err());
        resurrected.group_id = "missing".to_string();
        assert!(repository.upsert_asset(resurrected).await.is_err());
    }

    #[tokio::test]
    async fn counts_provider_references_including_soft_deleted_groups() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("sqlite pool");
        run_migrations(&pool).await.expect("migrations");
        seed_dependencies(&pool).await;
        let repository = SqliteAssetLibraryRepository::new(pool);
        repository.upsert_group(group()).await.expect("group");
        repository
            .upsert_visual_validation_session(UpsertArkVisualValidationSessionRecord {
                id: "validation-1".to_string(),
                session_id: "up-session-1".to_string(),
                user_id: "user-1".to_string(),
                api_key_id: None,
                provider_id: "provider-1".to_string(),
                endpoint_id: "endpoint-1".to_string(),
                key_id: "key-1".to_string(),
                account_binding: Some("account-1".to_string()),
                project: Some("project-1".to_string()),
                byted_token_hash: "token-hash".to_string(),
                encrypted_byted_token: "encrypted-token".to_string(),
                callback_state_hash: "state-hash".to_string(),
                status: "pending".to_string(),
                expires_at_unix_secs: 100,
                consumed_at_unix_secs: None,
                group_id: Some("group-1".to_string()),
                sanitized_result: None,
                created_at_unix_secs: 10,
                updated_at_unix_secs: 10,
            })
            .await
            .expect("validation session");
        assert!(repository
            .soft_delete_group("group-1", 20)
            .await
            .expect("soft-delete group"));

        for reference in [
            AssetProviderReference::ProviderId("provider-1"),
            AssetProviderReference::EndpointId("endpoint-1"),
            AssetProviderReference::KeyId("key-1"),
        ] {
            assert_eq!(
                repository
                    .count_provider_references(reference)
                    .await
                    .expect("provider reference counts"),
                AssetProviderReferenceCounts {
                    asset_groups: 1,
                    visual_validation_sessions: 1,
                }
            );
        }
        assert_eq!(
            repository
                .count_provider_references(AssetProviderReference::ProviderId("missing"))
                .await
                .expect("missing provider reference counts"),
            AssetProviderReferenceCounts::default()
        );
    }

    #[tokio::test]
    async fn canonical_group_identity_and_provider_foreign_keys_are_stable() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("sqlite pool");
        run_migrations(&pool).await.expect("migrations");
        seed_dependencies(&pool).await;
        let repository = SqliteAssetLibraryRepository::new(pool.clone());
        repository.upsert_group(group()).await.expect("group");

        let mut duplicate = group();
        duplicate.id = "group-2".to_string();
        duplicate.endpoint_id = "endpoint-2".to_string();
        duplicate.key_id = "key-2".to_string();
        assert!(repository.upsert_group(duplicate).await.is_err());

        let group = repository
            .find_group_by_canonical_upstream(
                "provider-1",
                "account-1",
                Some("project-1"),
                "up-group-1",
            )
            .await
            .expect("canonical lookup")
            .expect("canonical group");
        assert_eq!(group.id, "group-1");
        assert!(
            sqlx::query("DELETE FROM provider_api_keys WHERE id = 'key-1'")
                .execute(&pool)
                .await
                .is_err()
        );
        assert!(
            sqlx::query("DELETE FROM provider_endpoints WHERE id = 'endpoint-1'")
                .execute(&pool)
                .await
                .is_err()
        );
        assert!(sqlx::query("DELETE FROM providers WHERE id = 'provider-1'")
            .execute(&pool)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn rejects_new_material_references_to_inactive_catalog_bindings() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("sqlite pool");
        run_migrations(&pool).await.expect("migrations");
        seed_dependencies(&pool).await;
        let repository = SqliteAssetLibraryRepository::new(pool.clone());

        sqlx::query(
            "INSERT INTO providers (id, name, provider_type, enabled, is_active, created_at, updated_at) VALUES ('provider-2', 'provider-2', 'custom', 1, 1, 1, 1)",
        )
        .execute(&pool)
        .await
        .expect("other provider seed");
        sqlx::query(
            "INSERT INTO provider_endpoints (id, provider_id, name, base_url, enabled, is_active, created_at, updated_at) VALUES ('endpoint-other', 'provider-2', 'other-endpoint', 'https://example.com', 1, 1, 1, 1)",
        )
        .execute(&pool)
        .await
        .expect("other endpoint seed");
        sqlx::query(
            "INSERT INTO provider_api_keys (id, provider_id, name, encrypted_key, is_active, created_at, updated_at) VALUES ('key-other', 'provider-2', 'other-key', 'encrypted', 1, 1, 1)",
        )
        .execute(&pool)
        .await
        .expect("other key seed");
        let mut wrong_endpoint = group();
        wrong_endpoint.endpoint_id = "endpoint-other".to_string();
        assert!(repository.upsert_group(wrong_endpoint).await.is_err());
        let mut wrong_key = group();
        wrong_key.key_id = "key-other".to_string();
        assert!(repository.upsert_group(wrong_key).await.is_err());

        sqlx::query("UPDATE providers SET is_active = 0 WHERE id = 'provider-1'")
            .execute(&pool)
            .await
            .expect("disable provider");
        assert!(repository.upsert_group(group()).await.is_err());

        sqlx::query("UPDATE providers SET is_active = 1 WHERE id = 'provider-1'")
            .execute(&pool)
            .await
            .expect("enable provider");
        sqlx::query("UPDATE provider_endpoints SET is_active = 0 WHERE id = 'endpoint-1'")
            .execute(&pool)
            .await
            .expect("disable endpoint");
        assert!(repository.upsert_group(group()).await.is_err());

        sqlx::query("UPDATE provider_endpoints SET is_active = 1 WHERE id = 'endpoint-1'")
            .execute(&pool)
            .await
            .expect("enable endpoint");
        sqlx::query("UPDATE provider_api_keys SET is_active = 0 WHERE id = 'key-1'")
            .execute(&pool)
            .await
            .expect("disable key");
        assert!(repository.upsert_group(group()).await.is_err());

        sqlx::query("UPDATE provider_api_keys SET is_active = 1 WHERE id = 'key-1'")
            .execute(&pool)
            .await
            .expect("enable key");
        repository
            .upsert_group(group())
            .await
            .expect("active group");

        sqlx::query("UPDATE providers SET is_active = 0 WHERE id = 'provider-1'")
            .execute(&pool)
            .await
            .expect("disable provider for session");
        let session = UpsertArkVisualValidationSessionRecord {
            id: "validation-inactive-provider".to_string(),
            session_id: "up-session-inactive-provider".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: None,
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            account_binding: Some("account-1".to_string()),
            project: Some("project-1".to_string()),
            byted_token_hash: "token-hash-inactive-provider".to_string(),
            encrypted_byted_token: "encrypted-token-inactive-provider".to_string(),
            callback_state_hash: "state-hash-inactive-provider".to_string(),
            status: "pending".to_string(),
            expires_at_unix_secs: 100,
            consumed_at_unix_secs: None,
            group_id: Some("group-1".to_string()),
            sanitized_result: None,
            created_at_unix_secs: 10,
            updated_at_unix_secs: 10,
        };
        assert!(repository
            .upsert_visual_validation_session(session)
            .await
            .is_err());
    }
}
