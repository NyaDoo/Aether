ALTER TABLE asset_groups
    ADD COLUMN project_name TEXT NOT NULL DEFAULT 'default';

ALTER TABLE ark_visual_validation_sessions
    ADD COLUMN project_name TEXT NOT NULL DEFAULT 'default';
