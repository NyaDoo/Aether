-- Baseline migration: marks the existing schema (created by alembic) as ready.
-- All tables already exist from the Python era; this file is the starting point
-- for all future Rust-managed migrations.
SELECT 1;
