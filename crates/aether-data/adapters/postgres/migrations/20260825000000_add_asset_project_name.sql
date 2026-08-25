ALTER TABLE public.asset_groups
    ADD COLUMN IF NOT EXISTS project_name character varying(128) NOT NULL DEFAULT 'default';

ALTER TABLE public.ark_visual_validation_sessions
    ADD COLUMN IF NOT EXISTS project_name character varying(128) NOT NULL DEFAULT 'default';
