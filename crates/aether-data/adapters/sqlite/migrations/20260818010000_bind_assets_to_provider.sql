PRAGMA defer_foreign_keys = ON;

CREATE TABLE asset_groups_provider_bound (
    id TEXT PRIMARY KEY NOT NULL,
    upstream_group_id TEXT,
    user_id TEXT NOT NULL,
    api_key_id TEXT,
    provider_id TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    key_id TEXT NOT NULL,
    group_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    created_at_unix_secs INTEGER NOT NULL,
    updated_at_unix_secs INTEGER NOT NULL,
    deleted_at_unix_secs INTEGER,
    UNIQUE (provider_id, upstream_group_id),
    CONSTRAINT fk_asset_groups_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_asset_groups_api_key FOREIGN KEY (api_key_id) REFERENCES api_keys (id) ON DELETE SET NULL,
    CONSTRAINT fk_asset_groups_provider FOREIGN KEY (provider_id) REFERENCES providers (id) ON DELETE RESTRICT,
    CONSTRAINT fk_asset_groups_endpoint FOREIGN KEY (endpoint_id) REFERENCES provider_endpoints (id) ON DELETE RESTRICT,
    CONSTRAINT fk_asset_groups_key FOREIGN KEY (key_id) REFERENCES provider_api_keys (id) ON DELETE RESTRICT
);

INSERT INTO asset_groups_provider_bound (
    id, upstream_group_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
    group_type, name, description, status, created_at_unix_secs,
    updated_at_unix_secs, deleted_at_unix_secs
)
SELECT id, upstream_group_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
       group_type, name, description, status, created_at_unix_secs,
       updated_at_unix_secs, deleted_at_unix_secs
FROM asset_groups;

CREATE TABLE assets_provider_bound (
    id TEXT PRIMARY KEY NOT NULL,
    upstream_asset_id TEXT,
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    api_key_id TEXT,
    asset_type TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    moderation TEXT,
    last_inference_at_unix_secs INTEGER,
    source_url_fingerprint TEXT,
    provider_url TEXT,
    provider_url_expires_at_unix_secs INTEGER,
    sanitized_metadata TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at_unix_secs INTEGER,
    created_at_unix_secs INTEGER NOT NULL,
    updated_at_unix_secs INTEGER NOT NULL,
    UNIQUE (group_id, upstream_asset_id),
    CONSTRAINT fk_assets_group FOREIGN KEY (group_id) REFERENCES asset_groups_provider_bound (id) ON DELETE CASCADE,
    CONSTRAINT fk_assets_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_assets_api_key FOREIGN KEY (api_key_id) REFERENCES api_keys (id) ON DELETE SET NULL
);

INSERT INTO assets_provider_bound SELECT * FROM assets;

CREATE TABLE ark_visual_validation_sessions_provider_bound (
    id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    api_key_id TEXT,
    provider_id TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    key_id TEXT NOT NULL,
    byted_token_hash TEXT NOT NULL,
    encrypted_byted_token TEXT NOT NULL,
    callback_state_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    expires_at_unix_secs INTEGER NOT NULL,
    consumed_at_unix_secs INTEGER,
    group_id TEXT,
    sanitized_result TEXT,
    created_at_unix_secs INTEGER NOT NULL,
    updated_at_unix_secs INTEGER NOT NULL,
    UNIQUE (provider_id, session_id),
    UNIQUE (callback_state_hash),
    UNIQUE (byted_token_hash),
    CONSTRAINT fk_ark_validation_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_ark_validation_api_key FOREIGN KEY (api_key_id) REFERENCES api_keys (id) ON DELETE SET NULL,
    CONSTRAINT fk_ark_validation_provider FOREIGN KEY (provider_id) REFERENCES providers (id) ON DELETE RESTRICT,
    CONSTRAINT fk_ark_validation_endpoint FOREIGN KEY (endpoint_id) REFERENCES provider_endpoints (id) ON DELETE RESTRICT,
    CONSTRAINT fk_ark_validation_key FOREIGN KEY (key_id) REFERENCES provider_api_keys (id) ON DELETE RESTRICT,
    CONSTRAINT fk_ark_validation_group FOREIGN KEY (group_id) REFERENCES asset_groups_provider_bound (id) ON DELETE SET NULL
);

INSERT INTO ark_visual_validation_sessions_provider_bound (
    id, session_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
    byted_token_hash, encrypted_byted_token, callback_state_hash, status,
    expires_at_unix_secs, consumed_at_unix_secs, group_id, sanitized_result,
    created_at_unix_secs, updated_at_unix_secs
)
SELECT id, session_id, user_id, api_key_id, provider_id, endpoint_id, key_id,
       byted_token_hash, encrypted_byted_token, callback_state_hash, status,
       expires_at_unix_secs, consumed_at_unix_secs, group_id, sanitized_result,
       created_at_unix_secs, updated_at_unix_secs
FROM ark_visual_validation_sessions;

DROP TABLE assets;
DROP TABLE ark_visual_validation_sessions;
DROP TABLE asset_groups;

ALTER TABLE asset_groups_provider_bound RENAME TO asset_groups;
ALTER TABLE assets_provider_bound RENAME TO assets;
ALTER TABLE ark_visual_validation_sessions_provider_bound RENAME TO ark_visual_validation_sessions;

CREATE INDEX idx_asset_groups_user_deleted_created
    ON asset_groups (user_id, deleted_at_unix_secs, created_at_unix_secs);
CREATE INDEX idx_asset_groups_user_type_status
    ON asset_groups (user_id, group_type, status);
CREATE INDEX idx_assets_group_deleted_created
    ON assets (group_id, is_deleted, created_at_unix_secs);
CREATE INDEX idx_assets_user_type_status
    ON assets (user_id, asset_type, status);
CREATE INDEX idx_assets_user_deleted_created
    ON assets (user_id, is_deleted, created_at_unix_secs);
CREATE INDEX idx_ark_validation_user_status_expiry
    ON ark_visual_validation_sessions (user_id, status, expires_at_unix_secs);
CREATE INDEX idx_ark_validation_group
    ON ark_visual_validation_sessions (group_id);
