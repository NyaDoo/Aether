-- Ark private asset library tables.
CREATE TABLE IF NOT EXISTS public.asset_groups (
    id character varying(64) NOT NULL,
    upstream_group_id character varying(255),
    user_id character varying(64) NOT NULL,
    api_key_id character varying(64),
    provider_id character varying(64) NOT NULL,
    endpoint_id character varying(64) NOT NULL,
    key_id character varying(64) NOT NULL,
    group_type character varying(64) NOT NULL,
    name character varying(512) NOT NULL,
    description text,
    status character varying(64) NOT NULL,
    created_at_unix_secs bigint NOT NULL,
    updated_at_unix_secs bigint NOT NULL,
    deleted_at_unix_secs bigint,
    PRIMARY KEY (id),
    CONSTRAINT uq_asset_groups_upstream UNIQUE (provider_id, upstream_group_id),
    CONSTRAINT fk_asset_groups_user FOREIGN KEY (user_id) REFERENCES public.users (id) ON DELETE CASCADE,
    CONSTRAINT fk_asset_groups_api_key FOREIGN KEY (api_key_id) REFERENCES public.api_keys (id) ON DELETE SET NULL,
    CONSTRAINT fk_asset_groups_provider FOREIGN KEY (provider_id) REFERENCES public.providers (id) ON DELETE RESTRICT,
    CONSTRAINT fk_asset_groups_endpoint FOREIGN KEY (endpoint_id) REFERENCES public.provider_endpoints (id) ON DELETE RESTRICT,
    CONSTRAINT fk_asset_groups_key FOREIGN KEY (key_id) REFERENCES public.provider_api_keys (id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_asset_groups_user_deleted_created ON public.asset_groups (user_id, deleted_at_unix_secs, created_at_unix_secs);
CREATE INDEX IF NOT EXISTS idx_asset_groups_user_type_status ON public.asset_groups (user_id, group_type, status);

CREATE TABLE IF NOT EXISTS public.assets (
    id character varying(64) NOT NULL,
    upstream_asset_id character varying(255),
    group_id character varying(64) NOT NULL,
    user_id character varying(64) NOT NULL,
    api_key_id character varying(64),
    asset_type character varying(64) NOT NULL,
    name character varying(512) NOT NULL,
    status character varying(64) NOT NULL,
    error_code character varying(128),
    error_message text,
    moderation jsonb,
    last_inference_at_unix_secs bigint,
    source_url_fingerprint character varying(128),
    provider_url text,
    provider_url_expires_at_unix_secs bigint,
    sanitized_metadata jsonb,
    is_deleted boolean NOT NULL DEFAULT false,
    deleted_at_unix_secs bigint,
    created_at_unix_secs bigint NOT NULL,
    updated_at_unix_secs bigint NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_assets_group_upstream UNIQUE (group_id, upstream_asset_id),
    CONSTRAINT fk_assets_group FOREIGN KEY (group_id) REFERENCES public.asset_groups (id) ON DELETE CASCADE,
    CONSTRAINT fk_assets_user FOREIGN KEY (user_id) REFERENCES public.users (id) ON DELETE CASCADE,
    CONSTRAINT fk_assets_api_key FOREIGN KEY (api_key_id) REFERENCES public.api_keys (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_group_deleted_created ON public.assets (group_id, is_deleted, created_at_unix_secs);
CREATE INDEX IF NOT EXISTS idx_assets_user_type_status ON public.assets (user_id, asset_type, status);
CREATE INDEX IF NOT EXISTS idx_assets_user_deleted_created ON public.assets (user_id, is_deleted, created_at_unix_secs);

CREATE TABLE IF NOT EXISTS public.ark_visual_validation_sessions (
    id character varying(64) NOT NULL,
    session_id character varying(255) NOT NULL,
    user_id character varying(64) NOT NULL,
    api_key_id character varying(64),
    provider_id character varying(64) NOT NULL,
    endpoint_id character varying(64) NOT NULL,
    key_id character varying(64) NOT NULL,
    byted_token_hash character varying(128) NOT NULL,
    encrypted_byted_token text NOT NULL,
    callback_state_hash character varying(128) NOT NULL,
    status character varying(64) NOT NULL,
    expires_at_unix_secs bigint NOT NULL,
    consumed_at_unix_secs bigint,
    group_id character varying(64),
    sanitized_result jsonb,
    created_at_unix_secs bigint NOT NULL,
    updated_at_unix_secs bigint NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_ark_validation_upstream UNIQUE (provider_id, session_id),
    CONSTRAINT uq_ark_validation_callback_state UNIQUE (callback_state_hash),
    CONSTRAINT uq_ark_validation_byted_token_hash UNIQUE (byted_token_hash),
    CONSTRAINT fk_ark_validation_user FOREIGN KEY (user_id) REFERENCES public.users (id) ON DELETE CASCADE,
    CONSTRAINT fk_ark_validation_api_key FOREIGN KEY (api_key_id) REFERENCES public.api_keys (id) ON DELETE SET NULL,
    CONSTRAINT fk_ark_validation_provider FOREIGN KEY (provider_id) REFERENCES public.providers (id) ON DELETE RESTRICT,
    CONSTRAINT fk_ark_validation_endpoint FOREIGN KEY (endpoint_id) REFERENCES public.provider_endpoints (id) ON DELETE RESTRICT,
    CONSTRAINT fk_ark_validation_key FOREIGN KEY (key_id) REFERENCES public.provider_api_keys (id) ON DELETE RESTRICT,
    CONSTRAINT fk_ark_validation_group FOREIGN KEY (group_id) REFERENCES public.asset_groups (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_ark_validation_user_status_expiry ON public.ark_visual_validation_sessions (user_id, status, expires_at_unix_secs);
CREATE INDEX IF NOT EXISTS idx_ark_validation_group ON public.ark_visual_validation_sessions (group_id);
