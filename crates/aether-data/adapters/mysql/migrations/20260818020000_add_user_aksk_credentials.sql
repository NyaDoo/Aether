ALTER TABLE api_keys
    ADD COLUMN credential_type VARCHAR(32) NOT NULL DEFAULT 'api_key';

ALTER TABLE api_keys
    ADD COLUMN access_key_id VARCHAR(128) NULL;

ALTER TABLE api_keys
    ADD CONSTRAINT api_keys_credential_type_check
    CHECK (credential_type IN ('api_key', 'volc_aksk'));

CREATE UNIQUE INDEX api_keys_access_key_id_key ON api_keys (access_key_id);
