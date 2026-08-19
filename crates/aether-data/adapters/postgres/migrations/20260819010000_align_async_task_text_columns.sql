-- The legacy Postgres baseline predates the portable logical schema widths for
-- request candidates and video tasks. Provider error codes can exceed the old
-- 50-character columns, so align every drifted text column in the same forward
-- migration instead of moving the failure to the next provider-supplied field.
ALTER TABLE public.request_candidates
    ALTER COLUMN id TYPE VARCHAR(64),
    ALTER COLUMN request_id TYPE VARCHAR(128),
    ALTER COLUMN user_id TYPE VARCHAR(64),
    ALTER COLUMN api_key_id TYPE VARCHAR(64),
    ALTER COLUMN username TYPE VARCHAR(255),
    ALTER COLUMN api_key_name TYPE VARCHAR(255),
    ALTER COLUMN provider_id TYPE VARCHAR(64),
    ALTER COLUMN endpoint_id TYPE VARCHAR(64),
    ALTER COLUMN key_id TYPE VARCHAR(64),
    ALTER COLUMN status TYPE VARCHAR(32),
    ALTER COLUMN error_type TYPE VARCHAR(128);

ALTER TABLE public.video_tasks
    ALTER COLUMN id TYPE VARCHAR(64),
    ALTER COLUMN short_id TYPE VARCHAR(32),
    ALTER COLUMN request_id TYPE VARCHAR(128),
    ALTER COLUMN user_id TYPE VARCHAR(64),
    ALTER COLUMN api_key_id TYPE VARCHAR(64),
    ALTER COLUMN username TYPE VARCHAR(255),
    ALTER COLUMN api_key_name TYPE VARCHAR(255),
    ALTER COLUMN external_task_id TYPE VARCHAR(255),
    ALTER COLUMN provider_id TYPE VARCHAR(64),
    ALTER COLUMN endpoint_id TYPE VARCHAR(64),
    ALTER COLUMN key_id TYPE VARCHAR(64),
    ALTER COLUMN client_api_format TYPE VARCHAR(128),
    ALTER COLUMN provider_api_format TYPE VARCHAR(128),
    ALTER COLUMN model TYPE VARCHAR(255),
    ALTER COLUMN resolution TYPE VARCHAR(64),
    ALTER COLUMN aspect_ratio TYPE VARCHAR(32),
    ALTER COLUMN size TYPE VARCHAR(64),
    ALTER COLUMN status TYPE VARCHAR(32),
    ALTER COLUMN progress_message TYPE TEXT,
    ALTER COLUMN error_code TYPE VARCHAR(128),
    ALTER COLUMN video_url TYPE TEXT,
    ALTER COLUMN thumbnail_url TYPE TEXT,
    ALTER COLUMN remixed_from_task_id TYPE VARCHAR(64);
