ALTER TABLE asset_groups
    DROP INDEX uq_asset_groups_upstream,
    DROP COLUMN account_binding,
    DROP COLUMN project,
    ADD UNIQUE KEY uq_asset_groups_upstream (`provider_id`, `upstream_group_id`);

ALTER TABLE ark_visual_validation_sessions
    DROP INDEX uq_ark_validation_upstream,
    DROP COLUMN account_binding,
    DROP COLUMN project,
    ADD UNIQUE KEY uq_ark_validation_upstream (`provider_id`, `session_id`);
