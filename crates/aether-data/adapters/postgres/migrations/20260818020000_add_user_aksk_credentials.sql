ALTER TABLE api_keys
    ALTER COLUMN key_hash TYPE VARCHAR(255);

ALTER TABLE api_keys
    ADD COLUMN IF NOT EXISTS credential_type VARCHAR(32) NOT NULL DEFAULT 'api_key';

ALTER TABLE api_keys
    ADD COLUMN IF NOT EXISTS access_key_id VARCHAR(128);

CREATE UNIQUE INDEX IF NOT EXISTS api_keys_access_key_id_key
    ON api_keys (access_key_id)
    WHERE access_key_id IS NOT NULL;

ALTER TABLE api_keys
    DROP CONSTRAINT IF EXISTS api_keys_credential_type_check;

ALTER TABLE api_keys
    ADD CONSTRAINT api_keys_credential_type_check
    CHECK (credential_type IN ('api_key', 'volc_aksk'));
