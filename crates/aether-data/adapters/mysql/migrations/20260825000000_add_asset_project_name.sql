ALTER TABLE asset_groups
    ADD COLUMN project_name varchar(128) NOT NULL DEFAULT 'default' AFTER key_id;

ALTER TABLE ark_visual_validation_sessions
    ADD COLUMN project_name varchar(128) NOT NULL DEFAULT 'default' AFTER key_id;
