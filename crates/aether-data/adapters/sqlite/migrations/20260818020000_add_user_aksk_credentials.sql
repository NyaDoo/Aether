ALTER TABLE api_keys
    ADD COLUMN credential_type TEXT NOT NULL DEFAULT 'api_key'
    CHECK (credential_type IN ('api_key', 'volc_aksk'));

ALTER TABLE api_keys
    ADD COLUMN access_key_id TEXT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS api_keys_access_key_id_key
    ON api_keys (access_key_id)
    WHERE access_key_id IS NOT NULL;
