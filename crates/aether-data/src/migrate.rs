use sqlx::PgPool;

/// Run all pending migrations embedded at compile time from `migrations/`.
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::migrate::MigrateError> {
    sqlx::migrate!("./migrations").run(pool).await
}
