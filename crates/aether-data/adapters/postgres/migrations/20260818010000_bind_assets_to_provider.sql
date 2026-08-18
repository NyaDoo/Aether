ALTER TABLE public.asset_groups
    DROP CONSTRAINT IF EXISTS uq_asset_groups_upstream;

ALTER TABLE public.asset_groups
    DROP COLUMN IF EXISTS account_binding,
    DROP COLUMN IF EXISTS project;

ALTER TABLE public.asset_groups
    ADD CONSTRAINT uq_asset_groups_upstream UNIQUE (provider_id, upstream_group_id);

ALTER TABLE public.ark_visual_validation_sessions
    DROP CONSTRAINT IF EXISTS uq_ark_validation_upstream;

ALTER TABLE public.ark_visual_validation_sessions
    DROP COLUMN IF EXISTS account_binding,
    DROP COLUMN IF EXISTS project;

ALTER TABLE public.ark_visual_validation_sessions
    ADD CONSTRAINT uq_ark_validation_upstream UNIQUE (provider_id, session_id);
