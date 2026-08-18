use async_trait::async_trait;
use futures_util::TryStreamExt;
use sqlx::{postgres::PgRow, PgPool, Postgres, QueryBuilder, Row};

use aether_data_contracts::repository::asset_library::*;
use aether_data_contracts::DataLayerError;
use aether_data_query::{push_ci_contains_any, push_limit_offset, SqlDialect, WhereClause};

use crate::error::SqlxResultExt;

const GROUP_COLUMNS: &str = r#"
SELECT id, upstream_group_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
       group_type, name, description, status,
       created_at_unix_secs, updated_at_unix_secs, deleted_at_unix_secs
FROM public.asset_groups
"#;

const ASSET_COLUMNS: &str = r#"
SELECT id, upstream_asset_id, group_id, user_id, api_key_id, asset_type, name, status,
       error_code, error_message, moderation, last_inference_at_unix_secs,
       source_url_fingerprint, provider_url, provider_url_expires_at_unix_secs,
       sanitized_metadata, is_deleted, deleted_at_unix_secs,
       created_at_unix_secs, updated_at_unix_secs
FROM public.assets
"#;

const SESSION_COLUMNS: &str = r#"
SELECT id, session_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
       byted_token_hash, encrypted_byted_token,
       callback_state_hash, status, expires_at_unix_secs, consumed_at_unix_secs,
       group_id, sanitized_result, created_at_unix_secs, updated_at_unix_secs
FROM public.ark_visual_validation_sessions
"#;

fn provider_reference_counts_sql(reference: AssetProviderReference<'_>) -> &'static str {
    match reference {
        AssetProviderReference::ProviderId(_) => {
            "SELECT (SELECT COUNT(*) FROM public.asset_groups WHERE provider_id = $1), (SELECT COUNT(*) FROM public.ark_visual_validation_sessions WHERE provider_id = $1)"
        }
        AssetProviderReference::EndpointId(_) => {
            "SELECT (SELECT COUNT(*) FROM public.asset_groups WHERE endpoint_id = $1), (SELECT COUNT(*) FROM public.ark_visual_validation_sessions WHERE endpoint_id = $1)"
        }
        AssetProviderReference::KeyId(_) => {
            "SELECT (SELECT COUNT(*) FROM public.asset_groups WHERE key_id = $1), (SELECT COUNT(*) FROM public.ark_visual_validation_sessions WHERE key_id = $1)"
        }
    }
}

#[derive(Debug, Clone)]
pub struct SqlxAssetLibraryRepository {
    pool: PgPool,
}

impl SqlxAssetLibraryRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    async fn group_by_id(&self, id: &str) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        map_optional_group(
            sqlx::query(&format!("{GROUP_COLUMNS} WHERE id = $1 LIMIT 1"))
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_postgres_err()?,
        )
    }

    async fn asset_by_id(&self, id: &str) -> Result<Option<StoredAsset>, DataLayerError> {
        map_optional_asset(
            sqlx::query(&format!("{ASSET_COLUMNS} WHERE id = $1 LIMIT 1"))
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_postgres_err()?,
        )
    }

    async fn session_by_id(
        &self,
        id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!("{SESSION_COLUMNS} WHERE id = $1 LIMIT 1"))
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_postgres_err()?,
        )
    }
}

async fn lock_active_catalog_binding(
    transaction: &mut sqlx::Transaction<'_, Postgres>,
    provider_id: &str,
    endpoint_id: &str,
    key_id: &str,
) -> Result<(), DataLayerError> {
    let provider_exists = sqlx::query_scalar::<_, String>(
        "SELECT id FROM public.providers WHERE id = $1 AND is_active = TRUE FOR UPDATE",
    )
    .bind(provider_id)
    .fetch_optional(&mut **transaction)
    .await
    .map_postgres_err()?
    .is_some();
    let endpoint_exists = if provider_exists {
        sqlx::query_scalar::<_, String>(
            "SELECT id FROM public.provider_endpoints WHERE id = $1 AND provider_id = $2 AND is_active = TRUE FOR UPDATE",
        )
        .bind(endpoint_id)
        .bind(provider_id)
        .fetch_optional(&mut **transaction)
        .await
        .map_postgres_err()?
        .is_some()
    } else {
        false
    };
    let key_exists = if endpoint_exists {
        sqlx::query_scalar::<_, String>(
            "SELECT id FROM public.provider_api_keys WHERE id = $1 AND provider_id = $2 AND is_active = TRUE FOR UPDATE",
        )
        .bind(key_id)
        .bind(provider_id)
        .fetch_optional(&mut **transaction)
        .await
        .map_postgres_err()?
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
impl AssetLibraryReadRepository for SqlxAssetLibraryRepository {
    async fn count_provider_references(
        &self,
        reference: AssetProviderReference<'_>,
    ) -> Result<AssetProviderReferenceCounts, DataLayerError> {
        let (asset_groups, visual_validation_sessions) =
            sqlx::query_as::<_, (i64, i64)>(provider_reference_counts_sql(reference))
                .bind(reference.id())
                .fetch_one(&self.pool)
                .await
                .map_postgres_err()?;
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
                "{GROUP_COLUMNS} WHERE id = $1 AND user_id = $2 LIMIT 1"
            ))
            .bind(group_id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
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
                "{GROUP_COLUMNS} WHERE provider_id = $1 AND endpoint_id = $2 AND key_id = $3 AND upstream_group_id = $4 LIMIT 1"
            ))
            .bind(provider_id)
            .bind(endpoint_id)
            .bind(key_id)
            .bind(upstream_group_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
        )
    }

    async fn find_group_by_canonical_upstream(
        &self,
        provider_id: &str,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        map_optional_group(
            sqlx::query(&format!(
                "{GROUP_COLUMNS} WHERE provider_id = $1 AND upstream_group_id = $2 LIMIT 1"
            ))
            .bind(provider_id)
            .bind(upstream_group_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
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
            .map_postgres_err()?;
        let mut builder = group_rows_query(query);
        let built = builder.build();
        let mut rows = built.fetch(&self.pool);
        let mut items = Vec::new();
        while let Some(row) = rows.try_next().await.map_postgres_err()? {
            items.push(map_group(&row)?);
        }
        Ok(StoredAssetGroupListPage {
            items,
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
                "{ASSET_COLUMNS} WHERE id = $1 AND user_id = $2 LIMIT 1"
            ))
            .bind(asset_id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
        )
    }

    async fn find_asset_by_upstream(
        &self,
        group_id: &str,
        upstream_asset_id: &str,
    ) -> Result<Option<StoredAsset>, DataLayerError> {
        map_optional_asset(
            sqlx::query(&format!(
                "{ASSET_COLUMNS} WHERE group_id = $1 AND upstream_asset_id = $2 LIMIT 1"
            ))
            .bind(group_id)
            .bind(upstream_asset_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
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
            .map_postgres_err()?;
        let mut builder = asset_rows_query(query);
        let built = builder.build();
        let mut rows = built.fetch(&self.pool);
        let mut items = Vec::new();
        while let Some(row) = rows.try_next().await.map_postgres_err()? {
            items.push(map_asset(&row)?);
        }
        Ok(StoredAssetListPage {
            items,
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
                "{SESSION_COLUMNS} WHERE id = $1 AND user_id = $2 LIMIT 1"
            ))
            .bind(id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
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
                "{SESSION_COLUMNS} WHERE provider_id = $1 AND key_id = $2 AND session_id = $3 LIMIT 1"
            ))
            .bind(provider_id)
            .bind(key_id)
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
        )
    }

    async fn find_visual_validation_session_by_canonical_upstream(
        &self,
        provider_id: &str,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE provider_id = $1 AND session_id = $2 LIMIT 1"
            ))
            .bind(provider_id)
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
        )
    }

    async fn find_visual_validation_session_by_callback_state_hash(
        &self,
        callback_state_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE callback_state_hash = $1 LIMIT 1"
            ))
            .bind(callback_state_hash)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
        )
    }

    async fn find_visual_validation_session_by_byted_token_hash(
        &self,
        byted_token_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        map_optional_session(
            sqlx::query(&format!(
                "{SESSION_COLUMNS} WHERE byted_token_hash = $1 LIMIT 1"
            ))
            .bind(byted_token_hash)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
        )
    }
}

#[async_trait]
impl AssetLibraryWriteRepository for SqlxAssetLibraryRepository {
    async fn upsert_group(
        &self,
        record: UpsertAssetGroupRecord,
    ) -> Result<StoredAssetGroup, DataLayerError> {
        record.validate()?;
        let mut tx = self.pool.begin().await.map_postgres_err()?;
        lock_active_catalog_binding(
            &mut tx,
            &record.provider_id,
            &record.endpoint_id,
            &record.key_id,
        )
        .await?;
        let row = sqlx::query(
            r#"
INSERT INTO public.asset_groups (
  id, upstream_group_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
  group_type, name, description, status,
  created_at_unix_secs, updated_at_unix_secs, deleted_at_unix_secs
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
ON CONFLICT (id) DO UPDATE SET
  api_key_id = EXCLUDED.api_key_id,
  name = EXCLUDED.name,
  description = EXCLUDED.description,
  status = EXCLUDED.status,
  updated_at_unix_secs = EXCLUDED.updated_at_unix_secs
WHERE public.asset_groups.upstream_group_id IS NOT DISTINCT FROM EXCLUDED.upstream_group_id
  AND public.asset_groups.user_id = EXCLUDED.user_id
  AND public.asset_groups.provider_id = EXCLUDED.provider_id
  AND public.asset_groups.endpoint_id = EXCLUDED.endpoint_id
  AND public.asset_groups.key_id = EXCLUDED.key_id
  AND public.asset_groups.group_type = EXCLUDED.group_type
  AND public.asset_groups.deleted_at_unix_secs IS NOT DISTINCT FROM EXCLUDED.deleted_at_unix_secs
  AND (public.asset_groups.deleted_at_unix_secs IS NULL OR public.asset_groups.status = EXCLUDED.status)
RETURNING *
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
        .fetch_one(&mut *tx)
        .await
        .map_err(map_group_upsert_error)?;
        let group = map_group(&row)?;
        tx.commit().await.map_postgres_err()?;
        Ok(group)
    }

    async fn upsert_asset(&self, record: UpsertAssetRecord) -> Result<StoredAsset, DataLayerError> {
        record.validate()?;
        let group_id = record.group_id.clone();
        let mut tx = self.pool.begin().await.map_postgres_err()?;
        let parent_exists = sqlx::query_scalar::<_, String>(
            "SELECT id FROM public.asset_groups WHERE id = $1 AND user_id = $2 AND deleted_at_unix_secs IS NULL FOR UPDATE",
        )
        .bind(&record.group_id)
        .bind(&record.user_id)
        .fetch_optional(&mut *tx)
        .await
        .map_postgres_err()?
        .is_some();
        if !parent_exists {
            return Err(DataLayerError::InvalidInput(format!(
                "assets.group_id {:?} does not exist or belongs to another user",
                group_id
            )));
        }
        let row = sqlx::query(
            r#"
INSERT INTO public.assets (
  id, upstream_asset_id, group_id, user_id, api_key_id, asset_type, name, status,
  error_code, error_message, moderation, last_inference_at_unix_secs,
  source_url_fingerprint, provider_url, provider_url_expires_at_unix_secs,
  sanitized_metadata, is_deleted, deleted_at_unix_secs,
  created_at_unix_secs, updated_at_unix_secs
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
ON CONFLICT (id) DO UPDATE SET
  api_key_id = EXCLUDED.api_key_id,
  name = EXCLUDED.name,
  status = EXCLUDED.status,
  error_code = EXCLUDED.error_code,
  error_message = EXCLUDED.error_message,
  moderation = EXCLUDED.moderation,
  last_inference_at_unix_secs = EXCLUDED.last_inference_at_unix_secs,
  source_url_fingerprint = EXCLUDED.source_url_fingerprint,
  provider_url = EXCLUDED.provider_url,
  provider_url_expires_at_unix_secs = EXCLUDED.provider_url_expires_at_unix_secs,
  sanitized_metadata = EXCLUDED.sanitized_metadata,
  updated_at_unix_secs = EXCLUDED.updated_at_unix_secs
WHERE public.assets.upstream_asset_id IS NOT DISTINCT FROM EXCLUDED.upstream_asset_id
  AND public.assets.group_id = EXCLUDED.group_id
  AND public.assets.user_id = EXCLUDED.user_id
  AND public.assets.asset_type = EXCLUDED.asset_type
  AND public.assets.is_deleted = EXCLUDED.is_deleted
  AND public.assets.deleted_at_unix_secs IS NOT DISTINCT FROM EXCLUDED.deleted_at_unix_secs
  AND (NOT public.assets.is_deleted OR public.assets.status = EXCLUDED.status)
RETURNING *
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
        .bind(record.moderation)
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
        .bind(record.sanitized_metadata)
        .bind(record.is_deleted)
        .bind(to_optional_i64(
            record.deleted_at_unix_secs,
            "asset deleted_at",
        )?)
        .bind(to_i64(record.created_at_unix_secs, "asset created_at")?)
        .bind(to_i64(record.updated_at_unix_secs, "asset updated_at")?)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| map_asset_upsert_error(error, &group_id))?;
        let asset = map_asset(&row)?;
        tx.commit().await.map_postgres_err()?;
        Ok(asset)
    }

    async fn soft_delete_group(
        &self,
        group_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, DataLayerError> {
        let deleted_at = nonzero_i64(deleted_at_unix_secs, "asset group deleted_at")?;
        let mut tx = self.pool.begin().await.map_postgres_err()?;
        let group_exists = sqlx::query_scalar::<_, String>(
            "SELECT id FROM public.asset_groups WHERE id = $1 FOR UPDATE",
        )
        .bind(group_id)
        .fetch_optional(&mut *tx)
        .await
        .map_postgres_err()?
        .is_some();
        if !group_exists {
            tx.commit().await.map_postgres_err()?;
            return Ok(false);
        }
        let result = sqlx::query(
            "UPDATE public.asset_groups SET status = 'deleted', deleted_at_unix_secs = $2, updated_at_unix_secs = $2 WHERE id = $1 AND deleted_at_unix_secs IS NULL",
        )
        .bind(group_id)
        .bind(deleted_at)
        .execute(&mut *tx)
        .await
        .map_postgres_err()?;
        let deleted = result.rows_affected() > 0;
        if deleted {
            sqlx::query(
                "UPDATE public.assets SET status = 'deleted', is_deleted = TRUE, deleted_at_unix_secs = $2, updated_at_unix_secs = $2 WHERE group_id = $1 AND is_deleted = FALSE",
            )
            .bind(group_id)
            .bind(deleted_at)
            .execute(&mut *tx)
            .await
            .map_postgres_err()?;
        }
        tx.commit().await.map_postgres_err()?;
        Ok(deleted)
    }

    async fn soft_delete_asset(
        &self,
        asset_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, DataLayerError> {
        let deleted_at = nonzero_i64(deleted_at_unix_secs, "asset deleted_at")?;
        let result = sqlx::query(
            "UPDATE public.assets SET status = 'deleted', is_deleted = TRUE, deleted_at_unix_secs = $2, updated_at_unix_secs = $2 WHERE id = $1 AND is_deleted = FALSE",
        )
        .bind(asset_id)
        .bind(deleted_at)
        .execute(&self.pool)
        .await
        .map_postgres_err()?;
        Ok(result.rows_affected() > 0)
    }

    async fn upsert_visual_validation_session(
        &self,
        record: UpsertArkVisualValidationSessionRecord,
    ) -> Result<StoredArkVisualValidationSession, DataLayerError> {
        record.validate()?;
        let id = record.id.clone();
        let immutable_record = record.clone();
        let mut tx = self.pool.begin().await.map_postgres_err()?;
        lock_active_catalog_binding(
            &mut tx,
            &record.provider_id,
            &record.endpoint_id,
            &record.key_id,
        )
        .await?;
        if let Some(group_id) = record.group_id.as_deref() {
            let valid_group = sqlx::query_scalar::<_, String>(
                r#"
SELECT id FROM public.asset_groups
 WHERE id = $1
   AND user_id = $2
   AND provider_id = $3
   AND deleted_at_unix_secs IS NULL
 FOR UPDATE
"#,
            )
            .bind(group_id)
            .bind(&record.user_id)
            .bind(&record.provider_id)
            .fetch_optional(&mut *tx)
            .await
            .map_postgres_err()?
            .is_some();
            if !valid_group {
                return Err(DataLayerError::InvalidInput(
                    "validation session group owner or provider binding is invalid".to_string(),
                ));
            }
        }
        let row = sqlx::query(
            r#"
INSERT INTO public.ark_visual_validation_sessions (
  id, session_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
  byted_token_hash, encrypted_byted_token,
  callback_state_hash, status, expires_at_unix_secs, consumed_at_unix_secs,
  group_id, sanitized_result, created_at_unix_secs, updated_at_unix_secs
) SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17
   WHERE $14 IS NULL OR EXISTS (
     SELECT 1 FROM public.asset_groups
      WHERE id = $14 AND user_id = $3 AND provider_id = $5 AND deleted_at_unix_secs IS NULL
   )
ON CONFLICT (id) DO UPDATE SET
  api_key_id = EXCLUDED.api_key_id,
  status = EXCLUDED.status,
  consumed_at_unix_secs = EXCLUDED.consumed_at_unix_secs,
  group_id = COALESCE(public.ark_visual_validation_sessions.group_id, EXCLUDED.group_id),
  sanitized_result = EXCLUDED.sanitized_result,
  updated_at_unix_secs = EXCLUDED.updated_at_unix_secs
WHERE public.ark_visual_validation_sessions.consumed_at_unix_secs IS NULL
  AND public.ark_visual_validation_sessions.session_id = EXCLUDED.session_id
  AND public.ark_visual_validation_sessions.user_id = EXCLUDED.user_id
  AND public.ark_visual_validation_sessions.provider_id = EXCLUDED.provider_id
  AND public.ark_visual_validation_sessions.endpoint_id = EXCLUDED.endpoint_id
  AND public.ark_visual_validation_sessions.key_id = EXCLUDED.key_id
  AND public.ark_visual_validation_sessions.byted_token_hash = EXCLUDED.byted_token_hash
  AND public.ark_visual_validation_sessions.encrypted_byted_token = EXCLUDED.encrypted_byted_token
  AND public.ark_visual_validation_sessions.callback_state_hash = EXCLUDED.callback_state_hash
  AND public.ark_visual_validation_sessions.expires_at_unix_secs = EXCLUDED.expires_at_unix_secs
  AND (EXCLUDED.consumed_at_unix_secs IS NULL OR public.ark_visual_validation_sessions.consumed_at_unix_secs IS NULL)
  AND (public.ark_visual_validation_sessions.group_id IS NULL OR public.ark_visual_validation_sessions.group_id IS NOT DISTINCT FROM EXCLUDED.group_id)
RETURNING *
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
        .bind(record.sanitized_result)
        .bind(to_i64(
            record.created_at_unix_secs,
            "validation created_at",
        )?)
        .bind(to_i64(
            record.updated_at_unix_secs,
            "validation updated_at",
        )?)
        .fetch_optional(&mut *tx)
        .await
        .map_postgres_err()?;
        if let Some(row) = row {
            let session = map_session(&row)?;
            tx.commit().await.map_postgres_err()?;
            return Ok(session);
        }
        let existing = map_optional_session(
            sqlx::query(&format!("{SESSION_COLUMNS} WHERE id = $1 LIMIT 1"))
                .bind(&id)
                .fetch_optional(&mut *tx)
                .await
                .map_postgres_err()?,
        )?;
        if existing.as_ref().is_some_and(|existing| {
            existing.consumed_at_unix_secs.is_some()
                && immutable_record.has_same_immutable_identity(existing)
        }) {
            tx.commit().await.map_postgres_err()?;
            return Ok(existing.expect("checked above"));
        }
        Err(DataLayerError::InvalidInput(
            "validation session immutable or canonical upstream identity conflicts with an existing record"
                .to_string(),
        ))
    }

    async fn consume_visual_validation_session(
        &self,
        record: ConsumeArkVisualValidationSessionRecord,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        record.validate()?;
        map_optional_session(
            sqlx::query(
                r#"
UPDATE public.ark_visual_validation_sessions
SET status = $2,
    consumed_at_unix_secs = $3,
    sanitized_result = $4,
    updated_at_unix_secs = $5
WHERE callback_state_hash = $1
  AND consumed_at_unix_secs IS NULL
  AND expires_at_unix_secs > $3
RETURNING *
"#,
            )
            .bind(record.callback_state_hash)
            .bind(record.status)
            .bind(to_i64(
                record.consumed_at_unix_secs,
                "validation consumed_at",
            )?)
            .bind(record.sanitized_result)
            .bind(to_i64(
                record.updated_at_unix_secs,
                "validation updated_at",
            )?)
            .fetch_optional(&self.pool)
            .await
            .map_postgres_err()?,
        )
    }
}

fn group_count_query(query: &AssetGroupListQuery) -> QueryBuilder<'_, Postgres> {
    let mut builder =
        QueryBuilder::<Postgres>::new("SELECT COUNT(*)::bigint FROM public.asset_groups");
    let mut where_clause = WhereClause::new();
    group_filters(&mut builder, &mut where_clause, query);
    builder
}

fn group_rows_query(query: &AssetGroupListQuery) -> QueryBuilder<'_, Postgres> {
    let mut builder = QueryBuilder::<Postgres>::new(GROUP_COLUMNS);
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
    builder: &mut QueryBuilder<'a, Postgres>,
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
            SqlDialect::Postgres,
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

fn asset_count_query(query: &AssetListQuery) -> QueryBuilder<'_, Postgres> {
    let mut builder = QueryBuilder::<Postgres>::new("SELECT COUNT(*)::bigint FROM public.assets");
    let mut where_clause = WhereClause::new();
    asset_filters(&mut builder, &mut where_clause, query);
    builder
}

fn asset_rows_query(query: &AssetListQuery) -> QueryBuilder<'_, Postgres> {
    let mut builder = QueryBuilder::<Postgres>::new(ASSET_COLUMNS);
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
    builder: &mut QueryBuilder<'a, Postgres>,
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
        builder.push("is_deleted = FALSE");
    }
    if let Some(search) = trimmed(query.search.as_deref()) {
        push_ci_contains_any(
            builder,
            where_clause,
            SqlDialect::Postgres,
            &["id", "name", "COALESCE(upstream_asset_id, '')"],
            search,
        );
    }
}

fn push_optional_eq<'a>(
    builder: &mut QueryBuilder<'a, Postgres>,
    where_clause: &mut WhereClause,
    column: &str,
    value: Option<&'a str>,
) {
    if let Some(value) = value {
        where_clause.push_next(builder);
        builder.push(column).push(" = ").push_bind(value);
    }
}

fn map_optional_group(row: Option<PgRow>) -> Result<Option<StoredAssetGroup>, DataLayerError> {
    row.as_ref().map(map_group).transpose()
}

fn map_optional_asset(row: Option<PgRow>) -> Result<Option<StoredAsset>, DataLayerError> {
    row.as_ref().map(map_asset).transpose()
}

fn map_optional_session(
    row: Option<PgRow>,
) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
    row.as_ref().map(map_session).transpose()
}

fn map_group(row: &PgRow) -> Result<StoredAssetGroup, DataLayerError> {
    Ok(StoredAssetGroup {
        id: row.try_get("id").map_postgres_err()?,
        upstream_group_id: row.try_get("upstream_group_id").map_postgres_err()?,
        user_id: row.try_get("user_id").map_postgres_err()?,
        api_key_id: row.try_get("api_key_id").map_postgres_err()?,
        provider_id: row.try_get("provider_id").map_postgres_err()?,
        endpoint_id: row.try_get("endpoint_id").map_postgres_err()?,
        key_id: row.try_get("key_id").map_postgres_err()?,
        group_type: row.try_get("group_type").map_postgres_err()?,
        name: row.try_get("name").map_postgres_err()?,
        description: row.try_get("description").map_postgres_err()?,
        status: row.try_get("status").map_postgres_err()?,
        created_at_unix_secs: read_u64(row, "created_at_unix_secs", "asset_groups")?,
        updated_at_unix_secs: read_u64(row, "updated_at_unix_secs", "asset_groups")?,
        deleted_at_unix_secs: read_optional_u64(row, "deleted_at_unix_secs", "asset_groups")?,
    })
}

fn map_asset(row: &PgRow) -> Result<StoredAsset, DataLayerError> {
    Ok(StoredAsset {
        id: row.try_get("id").map_postgres_err()?,
        upstream_asset_id: row.try_get("upstream_asset_id").map_postgres_err()?,
        group_id: row.try_get("group_id").map_postgres_err()?,
        user_id: row.try_get("user_id").map_postgres_err()?,
        api_key_id: row.try_get("api_key_id").map_postgres_err()?,
        asset_type: row.try_get("asset_type").map_postgres_err()?,
        name: row.try_get("name").map_postgres_err()?,
        status: row.try_get("status").map_postgres_err()?,
        error_code: row.try_get("error_code").map_postgres_err()?,
        error_message: row.try_get("error_message").map_postgres_err()?,
        moderation: row.try_get("moderation").map_postgres_err()?,
        last_inference_at_unix_secs: read_optional_u64(
            row,
            "last_inference_at_unix_secs",
            "assets",
        )?,
        source_url_fingerprint: row.try_get("source_url_fingerprint").map_postgres_err()?,
        provider_url: row.try_get("provider_url").map_postgres_err()?,
        provider_url_expires_at_unix_secs: read_optional_u64(
            row,
            "provider_url_expires_at_unix_secs",
            "assets",
        )?,
        sanitized_metadata: row.try_get("sanitized_metadata").map_postgres_err()?,
        is_deleted: row.try_get("is_deleted").map_postgres_err()?,
        deleted_at_unix_secs: read_optional_u64(row, "deleted_at_unix_secs", "assets")?,
        created_at_unix_secs: read_u64(row, "created_at_unix_secs", "assets")?,
        updated_at_unix_secs: read_u64(row, "updated_at_unix_secs", "assets")?,
    })
}

fn map_session(row: &PgRow) -> Result<StoredArkVisualValidationSession, DataLayerError> {
    Ok(StoredArkVisualValidationSession {
        id: row.try_get("id").map_postgres_err()?,
        session_id: row.try_get("session_id").map_postgres_err()?,
        user_id: row.try_get("user_id").map_postgres_err()?,
        api_key_id: row.try_get("api_key_id").map_postgres_err()?,
        provider_id: row.try_get("provider_id").map_postgres_err()?,
        endpoint_id: row.try_get("endpoint_id").map_postgres_err()?,
        key_id: row.try_get("key_id").map_postgres_err()?,
        byted_token_hash: row.try_get("byted_token_hash").map_postgres_err()?,
        encrypted_byted_token: row.try_get("encrypted_byted_token").map_postgres_err()?,
        callback_state_hash: row.try_get("callback_state_hash").map_postgres_err()?,
        status: row.try_get("status").map_postgres_err()?,
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
        group_id: row.try_get("group_id").map_postgres_err()?,
        sanitized_result: row.try_get("sanitized_result").map_postgres_err()?,
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

fn read_u64(row: &PgRow, field: &str, table: &str) -> Result<u64, DataLayerError> {
    let value: i64 = row.try_get(field).map_postgres_err()?;
    u64::try_from(value).map_err(|_| {
        DataLayerError::UnexpectedValue(format!("{table}.{field} is negative: {value}"))
    })
}

fn read_optional_u64(row: &PgRow, field: &str, table: &str) -> Result<Option<u64>, DataLayerError> {
    let value: Option<i64> = row.try_get(field).map_postgres_err()?;
    value
        .map(|value| {
            u64::try_from(value).map_err(|_| {
                DataLayerError::UnexpectedValue(format!("{table}.{field} is negative: {value}"))
            })
        })
        .transpose()
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

fn map_asset_upsert_error(error: sqlx::Error, group_id: &str) -> DataLayerError {
    if matches!(error, sqlx::Error::RowNotFound) {
        DataLayerError::InvalidInput(format!(
            "assets.group_id {group_id:?} does not exist or belongs to another user"
        ))
    } else {
        DataLayerError::postgres(error)
    }
}

fn map_group_upsert_error(error: sqlx::Error) -> DataLayerError {
    if matches!(error, sqlx::Error::RowNotFound)
        || error
            .as_database_error()
            .is_some_and(|error| error.is_unique_violation())
    {
        DataLayerError::InvalidInput(
            "asset group immutable or canonical upstream identity conflicts with an existing record"
                .to_string(),
        )
    } else {
        DataLayerError::postgres(error)
    }
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
    use super::{asset_rows_query, group_rows_query, provider_reference_counts_sql};
    use aether_data_contracts::repository::asset_library::{
        AssetGroupListQuery, AssetListQuery, AssetProviderReference,
    };
    use sqlx::Execute;

    #[test]
    fn renders_owner_scoped_group_query() {
        let query = AssetGroupListQuery {
            user_id: Some("user-1".to_string()),
            search: Some("portrait".to_string()),
            limit: 20,
            offset: 5,
            ..AssetGroupListQuery::default()
        };
        let mut builder = group_rows_query(&query);
        let sql = builder.build().sql().to_string();
        assert!(sql.contains("user_id = $1"));
        assert!(sql.contains("deleted_at_unix_secs IS NULL"));
        assert!(sql.contains("LIMIT $"));
    }

    #[test]
    fn renders_asset_status_filter() {
        let query = AssetListQuery {
            status: Some("active".to_string()),
            limit: 10,
            ..AssetListQuery::default()
        };
        let mut builder = asset_rows_query(&query);
        let sql = builder.build().sql().to_string();
        assert!(sql.contains("status = $1"));
        assert!(sql.contains("is_deleted = FALSE"));
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
            assert!(sql.contains("public.asset_groups"));
            assert!(sql.contains("public.ark_visual_validation_sessions"));
            assert_eq!(sql.matches(&format!("{column} = $1")).count(), 2);
            assert!(!sql.contains("deleted_at"));
        }
    }

    #[test]
    fn asset_writes_share_the_parent_row_lock_protocol() {
        let source = include_str!("asset_library.rs");
        assert!(source.contains(
            "SELECT id FROM public.asset_groups WHERE id = $1 AND user_id = $2 AND deleted_at_unix_secs IS NULL FOR UPDATE"
        ));
        assert!(source.contains("SELECT id FROM public.asset_groups WHERE id = $1 FOR UPDATE"));
        let mutable_group_identity = ["group_id = ", "EXCLUDED.group_id,"].concat();
        let mutable_tombstone = ["is_deleted = ", "EXCLUDED.is_deleted,"].concat();
        assert!(!source.contains(&mutable_group_identity));
        assert!(!source.contains(&mutable_tombstone));
    }

    #[test]
    fn provider_binding_migration_uses_canonical_upstream_identity_and_restrictive_provider_fks() {
        let schema = include_str!("../migrations/20260818000000_add_asset_library.sql");
        let provider_binding =
            include_str!("../migrations/20260818010000_bind_assets_to_provider.sql");
        assert!(provider_binding.contains(
            "ADD CONSTRAINT uq_asset_groups_upstream UNIQUE (provider_id, upstream_group_id)"
        ));
        assert!(provider_binding.contains(
            "ADD CONSTRAINT uq_ark_validation_upstream UNIQUE (provider_id, session_id)"
        ));
        assert!(provider_binding.contains("DROP COLUMN IF EXISTS account_binding"));
        assert!(provider_binding.contains("DROP COLUMN IF EXISTS project"));
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
