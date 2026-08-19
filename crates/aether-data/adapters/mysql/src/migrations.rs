use sqlx::{
    migrate::{AppliedMigration, Migrate, MigrateError, Migrator},
    MySqlConnection, MySqlPool,
};
use tracing::warn;

use aether_data_contracts::PendingMigrationInfo;

pub static MIGRATOR: Migrator = sqlx::migrate!("./migrations");
const LEGACY_BASELINE_MIGRATION_VERSION: i64 = 20260403000000;

pub async fn run_migrations(pool: &MySqlPool) -> Result<(), MigrateError> {
    let mut conn = pool.acquire().await?;
    if MIGRATOR.locking {
        conn.lock().await?;
    }
    let result = run_migrations_locked(&mut conn).await;
    if MIGRATOR.locking {
        match conn.unlock().await {
            Ok(()) => {}
            Err(unlock_error) if result.is_ok() => return Err(unlock_error),
            Err(unlock_error) => warn!(
                error = %unlock_error,
                "MySQL migration lock release failed after migration error"
            ),
        }
    }
    result
}

async fn run_migrations_locked(conn: &mut MySqlConnection) -> Result<(), MigrateError> {
    conn.ensure_migrations_table().await?;
    if let Some(version) = conn.dirty_version().await? {
        return Err(MigrateError::Dirty(version));
    }
    let applied_migrations = conn.list_applied_migrations().await?;
    validate_applied_migrations(&applied_migrations)?;
    let applied_versions = applied_migrations
        .iter()
        .map(|migration| migration.version)
        .collect::<std::collections::HashSet<_>>();
    for migration in MIGRATOR
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
        .filter(|migration| !applied_versions.contains(&migration.version))
    {
        conn.apply(migration).await?;
    }
    Ok(())
}

pub async fn pending_migrations(
    pool: &MySqlPool,
) -> Result<Vec<PendingMigrationInfo>, MigrateError> {
    let mut conn = pool.acquire().await?;
    let applied_migrations = match conn.list_applied_migrations().await {
        Ok(applied_migrations) => applied_migrations,
        Err(err) if is_missing_sqlx_migrations_table_error(&err) => {
            return Ok(pending_migrations_from_applied(&[]));
        }
        Err(err) => return Err(err),
    };
    if let Some(version) = conn.dirty_version().await? {
        return Err(MigrateError::Dirty(version));
    }
    validate_applied_migrations(&applied_migrations)?;
    Ok(pending_migrations_from_applied(&applied_migrations))
}

pub async fn prepare_database_for_startup(
    pool: &MySqlPool,
) -> Result<Vec<PendingMigrationInfo>, MigrateError> {
    pending_migrations(pool).await
}

fn is_missing_sqlx_migrations_table_error(err: &MigrateError) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("_sqlx_migrations")
        && (message.contains("no such table")
            || message.contains("doesn't exist")
            || message.contains("does not exist")
            || message.contains("unknown table"))
}

fn pending_migrations_from_applied(
    applied_migrations: &[sqlx::migrate::AppliedMigration],
) -> Vec<PendingMigrationInfo> {
    let applied_versions = applied_migrations
        .iter()
        .map(|migration| migration.version)
        .collect::<std::collections::HashSet<_>>();
    MIGRATOR
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
        .filter(|migration| !applied_versions.contains(&migration.version))
        .map(|migration| PendingMigrationInfo {
            version: migration.version,
            description: migration.description.to_string(),
        })
        .collect()
}

fn validate_applied_migrations(
    applied_migrations: &[AppliedMigration],
) -> Result<(), MigrateError> {
    if MIGRATOR.ignore_missing {
        return Ok(());
    }
    let known_versions = MIGRATOR
        .iter()
        .map(|migration| migration.version)
        .collect::<std::collections::HashSet<_>>();
    if let Some(migration) = applied_migrations
        .iter()
        .find(|migration| !known_versions.contains(&migration.version))
    {
        return Err(MigrateError::VersionMissing(migration.version));
    }
    for migration in MIGRATOR
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
    {
        if let Some(applied_migration) = applied_migrations
            .iter()
            .find(|applied_migration| applied_migration.version == migration.version)
        {
            if migration.checksum != applied_migration.checksum {
                if migration.version == LEGACY_BASELINE_MIGRATION_VERSION {
                    warn!(
                        version = migration.version,
                        description = %migration.description,
                        "MySQL legacy baseline checksum mismatch (ignored for asset-library recovery)"
                    );
                } else {
                    return Err(MigrateError::VersionMismatch(migration.version));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{
        pending_migrations, prepare_database_for_startup, validate_applied_migrations,
        LEGACY_BASELINE_MIGRATION_VERSION, MIGRATOR,
    };
    use sqlx::migrate::{AppliedMigration, MigrateError};

    const ASSET_LIBRARY_MIGRATION_VERSION: i64 = 20260818000000;
    const REQUEST_OUTCOME_STATISTICS_RESET_MIGRATION_VERSION: i64 = 20260819000000;

    #[test]
    fn embeds_mysql_migration_sources() {
        let versions = MIGRATOR
            .iter()
            .map(|migration| migration.version)
            .collect::<Vec<_>>();
        assert!(!versions.is_empty());
        assert!(versions.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn embeds_asset_library_forward_migration() {
        let migration = MIGRATOR
            .iter()
            .find(|migration| migration.version == ASSET_LIBRARY_MIGRATION_VERSION)
            .expect("asset library migration should be embedded");
        let sql = migration.sql.as_ref();

        for table in ["asset_groups", "assets", "ark_visual_validation_sessions"] {
            assert!(
                sql.contains(&format!("CREATE TABLE IF NOT EXISTS {table}")),
                "asset library migration is missing {table}"
            );
        }
    }

    #[test]
    fn request_outcome_statistics_reset_preserves_historical_totals() {
        let migration = MIGRATOR
            .iter()
            .find(|migration| {
                migration.version == REQUEST_OUTCOME_STATISTICS_RESET_MIGRATION_VERSION
            })
            .expect("request outcome statistics reset migration should be embedded");
        let sql = migration.sql.as_ref();

        for table in [
            "stats_hourly",
            "stats_daily",
            "stats_summary",
            "stats_user_summary",
            "provider_api_keys",
            "usage_counter_deltas",
        ] {
            assert!(
                sql.contains(&format!("UPDATE {table}")),
                "missing reset for {table}"
            );
        }
        for outcome_field in [
            "success_requests = 0",
            "error_requests = 0",
            "sla_eligible_requests = 0",
            "user_error_requests = 0",
            "success_count = 0",
            "error_count = 0",
            "sla_eligible_count = 0",
            "user_error_count = 0",
        ] {
            assert!(
                sql.contains(outcome_field),
                "missing reset for {outcome_field}"
            );
        }
        for preserved_field in [
            "request_count = 0",
            "total_requests = 0",
            "total_tokens = 0",
            "total_cost_usd = 0",
            "total_response_time_ms = 0",
        ] {
            assert!(
                !sql.contains(preserved_field),
                "must preserve {preserved_field}"
            );
        }
        assert!(!sql.contains("DELETE FROM"));
        assert!(sql.contains("WHERE kind = 'provider_api_key'"));
    }

    #[test]
    fn embeds_user_aksk_credential_type_constraint() {
        let migration = MIGRATOR
            .iter()
            .find(|migration| migration.version == 20260818020000)
            .expect("user AK/SK migration should be embedded");
        let sql = migration.sql.as_ref();

        assert!(sql.contains("ADD COLUMN credential_type VARCHAR(32) NOT NULL DEFAULT 'api_key'"));
        assert!(sql.contains("ADD COLUMN access_key_id VARCHAR(128) NULL"));
        assert!(sql.contains("CONSTRAINT api_keys_credential_type_check"));
        assert!(sql.contains("CHECK (credential_type IN ('api_key', 'volc_aksk'))"));
    }

    #[test]
    fn embeds_cross_driver_schema_parity_migration() {
        let migration = MIGRATOR
            .iter()
            .find(|migration| migration.version == 20260725010000)
            .expect("cross-driver schema parity migration should be embedded");
        let sql = migration.sql.as_ref();

        for required_fragment in [
            "CREATE TABLE IF NOT EXISTS usage_body_blobs",
            "CREATE TABLE IF NOT EXISTS usage_http_audits",
            "CREATE TABLE IF NOT EXISTS stats_summary",
            "CREATE TABLE IF NOT EXISTS user_model_usage_counts",
            "CREATE TABLE IF NOT EXISTS api_key_provider_mappings",
            "CREATE TABLE IF NOT EXISTS provider_usage_tracking",
            "ADD COLUMN `settlement_snapshot_schema_version`",
            "ADD COLUMN `billing_effective_input_tokens`",
            "ADD COLUMN `converted_request_body`",
            "ADD COLUMN `p99_first_byte_time_ms`",
            "idx_usage_stale_pending_created_request",
        ] {
            assert!(
                sql.contains(required_fragment),
                "parity migration is missing {required_fragment}"
            );
        }
    }

    #[test]
    fn embeds_advanced_stats_parity_migration() {
        let migration = MIGRATOR
            .iter()
            .find(|migration| migration.version == 20260725020000)
            .expect("advanced stats parity migration should be embedded");
        let sql = migration.sql.as_ref();

        for required_fragment in [
            "CREATE TABLE stats_user_summary",
            "CREATE TABLE stats_user_daily_api_format",
            "CREATE TABLE stats_user_daily_model_provider",
            "CREATE TABLE stats_daily_model_provider",
            "CREATE TABLE stats_daily_cost_savings",
            "CREATE TABLE stats_user_daily_cost_savings_model_provider",
            "ADD COLUMN completed_total_input_context",
            "ADD COLUMN settled_total_cost",
            "ADD COLUMN response_time_samples",
            "UPDATE stats_hourly SET is_complete = 0",
            "UPDATE stats_daily SET is_complete = 0",
        ] {
            assert!(
                sql.contains(required_fragment),
                "advanced stats migration is missing {required_fragment}"
            );
        }
    }

    #[test]
    fn rejects_applied_migration_versions_unknown_to_this_binary() {
        let version = MIGRATOR
            .iter()
            .map(|migration| migration.version)
            .max()
            .expect("mysql migrations should not be empty")
            + 1;
        let error = validate_applied_migrations(&[AppliedMigration {
            version,
            checksum: Cow::Borrowed(&[]),
        }])
        .expect_err("unknown applied migration should block startup");

        assert!(matches!(error, MigrateError::VersionMissing(found) if found == version));
    }

    #[test]
    fn accepts_checksum_drift_for_legacy_baseline_only() {
        let migration = MIGRATOR
            .iter()
            .find(|migration| migration.version == LEGACY_BASELINE_MIGRATION_VERSION)
            .expect("legacy MySQL baseline should be embedded");
        validate_applied_migrations(&[AppliedMigration {
            version: migration.version,
            checksum: Cow::Borrowed(&[0]),
        }])
        .expect("legacy baseline checksum drift should remain recoverable");
    }

    #[test]
    fn rejects_checksum_drift_for_non_baseline_migrations() {
        let migration = MIGRATOR
            .iter()
            .find(|migration| migration.version != LEGACY_BASELINE_MIGRATION_VERSION)
            .expect("non-baseline MySQL migration should be embedded");
        let error = validate_applied_migrations(&[AppliedMigration {
            version: migration.version,
            checksum: Cow::Borrowed(&[0]),
        }])
        .expect_err("non-baseline checksum drift should remain strict");

        assert!(
            matches!(error, MigrateError::VersionMismatch(found) if found == migration.version)
        );
    }

    #[tokio::test]
    async fn pending_and_startup_preparation_reject_dirty_mysql_migration_state_when_url_is_set() {
        let Some(database_url) = std::env::var("AETHER_TEST_MYSQL_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            eprintln!("skipping mysql dirty migration test because AETHER_TEST_MYSQL_URL is unset");
            return;
        };

        let pool = sqlx::mysql::MySqlPoolOptions::new()
            .max_connections(1)
            .connect(&database_url)
            .await
            .expect("mysql test pool should connect");
        let dirty_version = MIGRATOR
            .iter()
            .next()
            .expect("mysql migrations should not be empty")
            .version;

        let mut conn = pool
            .acquire()
            .await
            .expect("mysql connection should acquire");
        sqlx::query(
            r#"
CREATE TEMPORARY TABLE _sqlx_migrations (
    version BIGINT PRIMARY KEY,
    description TEXT NOT NULL,
    installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL,
    checksum BLOB NOT NULL,
    execution_time BIGINT NOT NULL
)
"#,
        )
        .execute(&mut *conn)
        .await
        .expect("temporary mysql migrations table should create");
        sqlx::query(
            r#"
INSERT INTO _sqlx_migrations (
    version,
    description,
    success,
    checksum,
    execution_time
) VALUES (?, 'dirty test migration', FALSE, ?, 0)
"#,
        )
        .bind(dirty_version)
        .bind(Vec::<u8>::new())
        .execute(&mut *conn)
        .await
        .expect("dirty mysql migration should insert");
        drop(conn);

        let pending_error = pending_migrations(&pool)
            .await
            .expect_err("dirty mysql migration should fail pending inspection");
        assert!(
            matches!(&pending_error, MigrateError::Dirty(version) if *version == dirty_version),
            "unexpected pending migration error: {pending_error}"
        );

        let preparation_error = prepare_database_for_startup(&pool)
            .await
            .expect_err("dirty mysql migration should fail startup preparation");
        assert!(
            matches!(&preparation_error, MigrateError::Dirty(version) if *version == dirty_version),
            "unexpected startup preparation error: {preparation_error}"
        );
    }
}
