-- Aether historical usage/billing recovery forensic query (PostgreSQL).
--
-- READ-ONLY SAFETY:
--   * Run this against a restored clone/read replica first.
--   * This file never UPDATEs/DELETEs/INSERTs.  BEGIN/ROLLBACK only bracket
--     the read-only snapshot and session settings.
--   * Do not paste a production password/DSN into this file or into a report.
--
-- Invocation (psql 14+; values are required so the run is reproducible):
--   psql "$REPORTING_DSN" -X -q -v ON_ERROR_STOP=1 \
--     -v from_ts='2026-08-01T00:00:00Z' \
--     -v to_ts='2026-09-01T00:00:00Z' \
--     -v as_of_ts='2026-08-26T10:00:00Z' \
--     -v stale_minutes=120 \
--     -f docs/operations/usage-recovery-dry-run.sql \
--     > recovery.csv 2> recovery.stderr
-- Keep stdout and stderr separate.  `-q` suppresses psql command/status tags;
-- a non-empty stderr or non-zero exit is a failed run, not an empty report.
--
-- The result is one row per usage request.  Payload bytes are deliberately
-- not selected; use the export query in the companion markdown only for an
-- offline, access-controlled decompression pass.
-- Every category/action below is report-only.  This query never proves a
-- provider terminal event and never authorizes replay, settlement, or debit.
-- The live stale sweep is deliberately conservative: for non-video rows in
-- pending/streaming/completed/failed/cancelled status whose billing remains
-- pending and unfinalized after the timeout, it records failed+void.  A
-- request-candidate success marker is diagnostic only and never promotes usage
-- to completed.

BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;
SET LOCAL statement_timeout = '120s';
SET LOCAL lock_timeout = '2s';
\pset format csv
\pset footer off

WITH
params AS (
    SELECT
        CAST(:'from_ts' AS timestamptz) AS from_ts,
        CAST(:'to_ts' AS timestamptz) AS to_ts,
        CAST(:'as_of_ts' AS timestamptz) AS as_of_ts,
        CAST(:'stale_minutes' AS double precision) AS stale_minutes,
        (
        CASE
            WHEN isfinite(CAST(:'from_ts' AS timestamptz))
             AND isfinite(CAST(:'to_ts' AS timestamptz))
             AND isfinite(CAST(:'as_of_ts' AS timestamptz))
             AND CAST(:'from_ts' AS timestamptz) < CAST(:'to_ts' AS timestamptz)
             AND CAST(:'to_ts' AS timestamptz) <= CAST(:'as_of_ts' AS timestamptz)
             -- The live gateway clamps stale cleanup to a 120-minute floor.
             -- A smaller forensic value would classify rows the current
             -- runtime cannot select and is therefore rejected here.
             AND CAST(:'stale_minutes' AS double precision) >= 120
             AND CAST(:'stale_minutes' AS double precision)
                    = CAST(:'stale_minutes' AS double precision)
            AND CAST(:'stale_minutes' AS double precision) < 'Infinity'::double precision
            THEN '1'
            ELSE 'invalid recovery parameters: require finite timestamps with from_ts < to_ts <= as_of_ts and finite stale_minutes >= 120'
        END
        )::integer AS parameter_validation_guard
),
scoped AS MATERIALIZED (
    SELECT
        u.request_id,
        u.id AS usage_id,
        u.user_id,
        u.api_key_id,
        u.provider_id,
        u.provider_endpoint_id,
        u.provider_api_key_id,
        u.provider_name,
        u.model,
        u.target_model,
        u.request_type,
        u.api_format,
        u.api_family,
        u.endpoint_kind,
        u.endpoint_api_format,
        u.provider_api_family,
        u.provider_endpoint_kind,
        COALESCE(u.is_stream, FALSE) AS is_stream,
        COALESCE(u.upstream_is_stream, u.is_stream, FALSE) AS upstream_is_stream,
        u.status,
        u.status_code,
        u.outcome_class,
        u.sla_eligible,
        u.billing_status AS usage_billing_status,
        s.billing_status AS settlement_billing_status,
        COALESCE(s.billing_status, u.billing_status) AS effective_billing_status,
        COALESCE(
            s.billing_snapshot_status,
            NULLIF(BTRIM(u.request_metadata::jsonb ->> 'billing_snapshot_status'), ''),
            NULLIF(BTRIM(u.request_metadata::jsonb #>> '{billing_snapshot,status}'), ''),
            NULLIF(BTRIM(u.request_metadata::jsonb #>> '{settlement_snapshot,status}'), '')
        ) AS effective_snapshot_status,
        COALESCE(
            NULLIF(BTRIM(u.request_metadata::jsonb ->> 'billing_snapshot_status'), ''),
            NULLIF(BTRIM(u.request_metadata::jsonb #>> '{billing_snapshot,status}'), ''),
            NULLIF(BTRIM(u.request_metadata::jsonb #>> '{settlement_snapshot,status}'), '')
        ) AS metadata_snapshot_status,
        NULLIF(BTRIM(u.request_metadata::jsonb ->> 'billing_snapshot_status'), '')
            AS metadata_flat_snapshot_status,
        COALESCE(
            NULLIF(BTRIM(u.request_metadata::jsonb #>> '{billing_snapshot,status}'), ''),
            NULLIF(BTRIM(u.request_metadata::jsonb #>> '{settlement_snapshot,status}'), '')
        ) AS metadata_nested_snapshot_status,
        NULLIF(BTRIM(u.request_metadata::jsonb #>> '{billing_snapshot,status}'), '')
            AS metadata_billing_snapshot_status,
        NULLIF(BTRIM(u.request_metadata::jsonb #>> '{settlement_snapshot,status}'), '')
            AS metadata_settlement_snapshot_status,
        COALESCE(
            CASE
                WHEN jsonb_typeof(
                    u.request_metadata::jsonb #> '{settlement_snapshot,actual_total_cost}'
                ) = 'number'
                THEN (
                    u.request_metadata::jsonb #>> '{settlement_snapshot,actual_total_cost}'
                )::numeric
            END,
            CASE
                WHEN jsonb_typeof(
                    u.request_metadata::jsonb #> '{billing_snapshot,actual_total_cost}'
                ) = 'number'
                THEN (
                    u.request_metadata::jsonb #>> '{billing_snapshot,actual_total_cost}'
                )::numeric
            END
        ) AS metadata_snapshot_actual_total_cost_usd,
        COALESCE(
            u.request_metadata::jsonb #>>
                '{settlement_snapshot,billing_plan_snapshot,rule_id}',
            u.request_metadata::jsonb #>> '{billing_snapshot,rule_id}'
        ) AS metadata_snapshot_rule_id,
        COALESCE(
            u.request_metadata::jsonb #>>
                '{settlement_snapshot,billing_plan_snapshot,rule_version}',
            u.request_metadata::jsonb #>> '{billing_snapshot,rule_version}'
        ) AS metadata_snapshot_rule_version,
        COALESCE(
            u.request_metadata::jsonb #>>
                '{settlement_snapshot,pricing_snapshot,pricing_source}',
            u.request_metadata::jsonb #>> '{billing_snapshot,pricing_source}'
        ) AS metadata_snapshot_pricing_source,
        CASE
            WHEN jsonb_typeof(
                u.request_metadata::jsonb #> '{settlement_snapshot,pricing_snapshot,is_free_tier}'
            ) = 'boolean'
            THEN (
                u.request_metadata::jsonb #>>
                    '{settlement_snapshot,pricing_snapshot,is_free_tier}'
            )::boolean
            WHEN jsonb_typeof(u.request_metadata::jsonb -> 'is_free_tier') = 'boolean'
            THEN (u.request_metadata::jsonb ->> 'is_free_tier')::boolean
            WHEN jsonb_typeof(
                u.request_metadata::jsonb #> '{billing_snapshot,is_free_tier}'
            ) = 'boolean'
            THEN (
                u.request_metadata::jsonb #>> '{billing_snapshot,is_free_tier}'
            )::boolean
        END AS metadata_snapshot_is_free_tier,
        s.billing_snapshot_status AS settlement_snapshot_status,
        (s.settlement_snapshot IS NOT NULL) AS settlement_snapshot_json_present,
        (s.billing_dimensions IS NOT NULL) AS settlement_billing_dimensions_present,
        s.billing_actual_total_cost_usd,
        s.billing_total_cost_usd,
        s.billing_input_tokens,
        s.billing_effective_input_tokens,
        s.billing_output_tokens,
        s.billing_cache_creation_tokens,
        s.billing_cache_read_tokens,
        s.billing_rule_id,
        s.billing_rule_version,
        s.billing_pricing_source,
        s.is_free_tier,
        s.wallet_id AS settlement_wallet_id,
        s.wallet_balance_before AS settlement_wallet_balance_before,
        s.wallet_balance_after AS settlement_wallet_balance_after,
        s.wallet_recharge_balance_before AS settlement_wallet_recharge_balance_before,
        s.wallet_recharge_balance_after AS settlement_wallet_recharge_balance_after,
        s.wallet_gift_balance_before AS settlement_wallet_gift_balance_before,
        s.wallet_gift_balance_after AS settlement_wallet_gift_balance_after,
        s.provider_monthly_used_usd AS settlement_provider_monthly_used_usd,
        s.finalized_at AS settlement_finalized_at,
        u.input_tokens AS usage_input_tokens,
        u.output_tokens AS usage_output_tokens,
        u.total_tokens AS usage_total_tokens,
        u.input_output_total_tokens,
        u.cache_creation_input_tokens,
        u.cache_creation_input_tokens_5m,
        u.cache_creation_input_tokens_1h,
        u.cache_read_input_tokens,
        u.input_cost_usd,
        u.output_cost_usd,
        u.total_cost_usd AS usage_total_cost_usd,
        u.actual_total_cost_usd AS usage_actual_total_cost_usd,
        u.wallet_id AS usage_wallet_id,
        u.wallet_balance_before AS usage_wallet_balance_before,
        u.wallet_balance_after AS usage_wallet_balance_after,
        u.wallet_recharge_balance_before AS usage_wallet_recharge_balance_before,
        u.wallet_recharge_balance_after AS usage_wallet_recharge_balance_after,
        u.wallet_gift_balance_before AS usage_wallet_gift_balance_before,
        u.wallet_gift_balance_after AS usage_wallet_gift_balance_after,
        u.first_byte_time_ms,
        u.response_time_ms,
        u.finalized_at AS usage_finalized_at,
        u.created_at,
        u.updated_at_unix_secs,
        u.request_metadata,
        CASE
            WHEN jsonb_typeof(u.request_metadata::jsonb -> 'billing_treat_as_completed')
                    = 'boolean'
            THEN (u.request_metadata::jsonb ->> 'billing_treat_as_completed')::boolean
        END AS billing_treat_as_completed_marker,
        u.candidate_id AS usage_candidate_id,
        u.candidate_index AS usage_candidate_index,
        COALESCE(r.candidate_id, u.candidate_id) AS selected_candidate_id,
        COALESCE(r.candidate_index, u.candidate_index) AS selected_candidate_index,
        COALESCE(r.selected_provider_id, u.provider_id) AS selected_provider_id,
        COALESCE(r.selected_endpoint_id, u.provider_endpoint_id) AS selected_endpoint_id,
        COALESCE(r.selected_provider_api_key_id, u.provider_api_key_id)
            AS selected_provider_api_key_id,
        (
            (r.candidate_id IS NOT NULL AND u.candidate_id IS NOT NULL
             AND r.candidate_id <> u.candidate_id)
            OR (r.candidate_index IS NOT NULL AND u.candidate_index IS NOT NULL
                AND r.candidate_index <> u.candidate_index)
            OR (r.selected_provider_id IS NOT NULL AND u.provider_id IS NOT NULL
                AND r.selected_provider_id <> u.provider_id)
            OR (r.selected_endpoint_id IS NOT NULL AND u.provider_endpoint_id IS NOT NULL
                AND r.selected_endpoint_id <> u.provider_endpoint_id)
            OR (
                r.selected_provider_api_key_id IS NOT NULL
                AND u.provider_api_key_id IS NOT NULL
                AND r.selected_provider_api_key_id <> u.provider_api_key_id
            )
        ) AS candidate_identity_conflict,
        -- Deprecated inline/compressed columns are retained as fallbacks.
        NULLIF(BTRIM(u.request_metadata::jsonb ->> 'request_body_ref'), '')
            AS metadata_request_body_ref,
        NULLIF(BTRIM(u.request_metadata::jsonb ->> 'provider_request_body_ref'), '')
            AS metadata_provider_request_body_ref,
        NULLIF(BTRIM(u.request_metadata::jsonb ->> 'response_body_ref'), '')
            AS metadata_response_body_ref,
        NULLIF(BTRIM(u.request_metadata::jsonb ->> 'client_response_body_ref'), '')
            AS metadata_client_response_body_ref,
        (u.request_headers IS NOT NULL) AS legacy_request_headers_present,
        (u.provider_request_headers IS NOT NULL) AS legacy_provider_request_headers_present,
        (u.response_headers IS NOT NULL) AS legacy_response_headers_present,
        (u.client_response_headers IS NOT NULL) AS legacy_client_response_headers_present,
        (u.request_body IS NOT NULL OR u.request_body_compressed IS NOT NULL)
            AS legacy_request_body_present,
        (u.provider_request_body IS NOT NULL OR u.provider_request_body_compressed IS NOT NULL)
            AS legacy_provider_request_body_present,
        (u.response_body IS NOT NULL OR u.response_body_compressed IS NOT NULL)
            AS legacy_response_body_present,
        (u.client_response_body IS NOT NULL OR u.client_response_body_compressed IS NOT NULL)
            AS legacy_client_response_body_present,
        CASE
            WHEN lower(regexp_replace(COALESCE(u.request_type, ''), '[[:space:]]', '', 'g')) = 'video'
              OR lower(regexp_replace(COALESCE(u.api_format, ''), '[[:space:]]', '', 'g')) = 'video'
              OR lower(regexp_replace(COALESCE(u.endpoint_kind, ''), '[[:space:]]', '', 'g')) = 'video'
              OR lower(regexp_replace(COALESCE(u.endpoint_api_format, ''), '[[:space:]]', '', 'g')) = 'video'
              OR lower(regexp_replace(COALESCE(u.provider_endpoint_kind, ''), '[[:space:]]', '', 'g')) = 'video'
              OR lower(regexp_replace(COALESCE(u.request_type, ''), '[[:space:]]', '', 'g')) LIKE '%:video'
              OR lower(regexp_replace(COALESCE(u.api_format, ''), '[[:space:]]', '', 'g')) LIKE '%:video'
              OR lower(regexp_replace(COALESCE(u.endpoint_kind, ''), '[[:space:]]', '', 'g')) LIKE '%:video'
              OR lower(regexp_replace(COALESCE(u.endpoint_api_format, ''), '[[:space:]]', '', 'g')) LIKE '%:video'
              OR lower(regexp_replace(COALESCE(u.provider_endpoint_kind, ''), '[[:space:]]', '', 'g')) LIKE '%:video'
            THEN TRUE ELSE FALSE
        END AS is_video_contract
    FROM public."usage" AS u
    LEFT JOIN public.usage_settlement_snapshots AS s
      ON s.request_id = u.request_id
    LEFT JOIN public.usage_routing_snapshots AS r
      ON r.request_id = u.request_id
    CROSS JOIN params AS p
    WHERE u.created_at >= p.from_ts
      AND u.created_at < p.to_ts
      AND p.parameter_validation_guard = 1
),
candidate_evidence AS (
    SELECT
        c.*,
        (
            x.selected_candidate_id IS NOT NULL
            AND NOT x.candidate_identity_conflict
            AND c.id = x.selected_candidate_id
            AND (
                x.selected_candidate_index IS NULL
                OR c.candidate_index = x.selected_candidate_index
            )
            AND (
                x.selected_provider_id IS NULL
                OR c.provider_id = x.selected_provider_id
            )
            AND (
                x.selected_endpoint_id IS NULL
                OR c.endpoint_id = x.selected_endpoint_id
            )
            AND (
                x.selected_provider_api_key_id IS NULL
                OR c.key_id = x.selected_provider_api_key_id
            )
        ) AS candidate_identity_match
    FROM public.request_candidates AS c
    JOIN scoped AS x ON x.request_id = c.request_id
),
candidate_rollup AS (
    SELECT
        c.request_id,
        COUNT(*)::bigint AS candidate_rows,
        COUNT(*) FILTER (WHERE c.status = 'success')::bigint AS success_candidate_rows,
        COUNT(*) FILTER (WHERE c.status = 'streaming')::bigint AS streaming_candidate_rows,
        COUNT(*) FILTER (WHERE c.status = 'pending')::bigint AS pending_candidate_rows,
        COUNT(*) FILTER (WHERE c.status IN ('failed', 'cancelled'))::bigint
            AS failed_or_cancelled_candidate_rows,
        COUNT(*) FILTER (WHERE c.candidate_identity_match)::bigint
            AS identity_matched_candidate_rows,
        BOOL_OR(
            c.candidate_identity_match
            AND c.status = 'success'
            AND c.finished_at IS NOT NULL
            AND COALESCE((c.extra_data::jsonb ->> 'stream_completed'), 'false') = 'true'
        ) AS has_durable_terminal_candidate,
        BOOL_OR(c.candidate_identity_match AND c.status = 'streaming')
            AS has_open_stream_candidate,
        BOOL_OR(c.candidate_identity_match AND c.status = 'pending')
            AS has_open_pending_candidate,
        MAX(c.finished_at) FILTER (
            WHERE c.candidate_identity_match
              AND c.status = 'success'
              AND c.finished_at IS NOT NULL
              AND COALESCE((c.extra_data::jsonb ->> 'stream_completed'), 'false') = 'true'
        ) AS durable_candidate_finished_at,
        BOOL_OR(
            c.candidate_identity_match
            AND c.status = 'success'
            AND c.status_code BETWEEN 200 AND 299
        ) AS candidate_success_2xx,
        BOOL_OR(
            c.candidate_identity_match
            AND c.status = 'success'
            AND c.status_code BETWEEN 200 AND 299
            AND c.finished_at IS NOT NULL
            AND COALESCE((c.extra_data::jsonb ->> 'stream_completed'), 'false') = 'true'
        ) AS has_durable_terminal_2xx_candidate,
        -- This marker is scheduler evidence only.  Provider payload parsing is
        -- intentionally out of scope, so it is never terminal proof.
        BOOL_OR(
            c.candidate_identity_match
            AND COALESCE((c.extra_data::jsonb ->> 'stream_completed'), 'false') = 'true'
        ) AS any_stream_completed_marker
    FROM candidate_evidence AS c
    GROUP BY c.request_id
),
http_rollup AS (
    SELECT
        x.request_id,
        h.body_capture_mode,
        h.request_body_state,
        h.provider_request_body_state,
        h.response_body_state,
        h.client_response_body_state,
        (h.request_headers IS NOT NULL) AS has_request_headers,
        (h.provider_request_headers IS NOT NULL) AS has_provider_request_headers,
        (h.response_headers IS NOT NULL) AS has_response_headers,
        (h.client_response_headers IS NOT NULL) AS has_client_response_headers,
        (h.request_body_ref IS NOT NULL) AS request_body_ref_present,
        (h.provider_request_body_ref IS NOT NULL) AS provider_request_body_ref_present,
        (h.response_body_ref IS NOT NULL) AS response_body_ref_present,
        (h.client_response_body_ref IS NOT NULL) AS client_response_body_ref_present,
        (
            h.request_body_ref IS NOT NULL
            OR (
                h.request_body_ref IS NULL
                AND x.metadata_request_body_ref = format(
                    'usage://request/%s/request_body', x.request_id
                )
            )
        ) AS effective_request_body_ref_present,
        (
            h.provider_request_body_ref IS NOT NULL
            OR (
                h.provider_request_body_ref IS NULL
                AND x.metadata_provider_request_body_ref = format(
                    'usage://request/%s/provider_request_body', x.request_id
                )
            )
        ) AS effective_provider_request_body_ref_present,
        (
            h.response_body_ref IS NOT NULL
            OR (
                h.response_body_ref IS NULL
                AND x.metadata_response_body_ref = format(
                    'usage://request/%s/response_body', x.request_id
                )
            )
        ) AS effective_response_body_ref_present,
        (
            h.client_response_body_ref IS NOT NULL
            OR (
                h.client_response_body_ref IS NULL
                AND x.metadata_client_response_body_ref = format(
                    'usage://request/%s/client_response_body', x.request_id
                )
            )
        ) AS effective_client_response_body_ref_present,
        (h.request_body_ref IS NOT NULL AND EXISTS (
            SELECT 1 FROM public.usage_body_blobs AS b
            WHERE b.body_ref = h.request_body_ref
              AND b.request_id = x.request_id
              AND b.body_field = 'request_body'
        )) AS request_blob_present,
        (h.provider_request_body_ref IS NOT NULL AND EXISTS (
            SELECT 1 FROM public.usage_body_blobs AS b
            WHERE b.body_ref = h.provider_request_body_ref
              AND b.request_id = x.request_id
              AND b.body_field = 'provider_request_body'
        )) AS provider_request_blob_present,
        (h.response_body_ref IS NOT NULL AND EXISTS (
            SELECT 1 FROM public.usage_body_blobs AS b
            WHERE b.body_ref = h.response_body_ref
              AND b.request_id = x.request_id
              AND b.body_field = 'response_body'
        )) AS response_blob_present,
        (h.client_response_body_ref IS NOT NULL AND EXISTS (
            SELECT 1 FROM public.usage_body_blobs AS b
            WHERE b.body_ref = h.client_response_body_ref
              AND b.request_id = x.request_id
              AND b.body_field = 'client_response_body'
        )) AS client_response_blob_present,
        (
            h.request_body_ref IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = h.request_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'request_body'
            )
        ) AS request_ref_capture_corrupt,
        (
            h.provider_request_body_ref IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = h.provider_request_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'provider_request_body'
            )
        ) AS provider_request_ref_capture_corrupt,
        (
            h.response_body_ref IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = h.response_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'response_body'
            )
        ) AS response_ref_capture_corrupt,
        (
            h.client_response_body_ref IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = h.client_response_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'client_response_body'
            )
        ) AS client_response_ref_capture_corrupt,
        (
            h.response_body_ref IS NULL
            AND x.metadata_response_body_ref = format(
                'usage://request/%s/response_body', x.request_id
            )
            AND EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_response_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'response_body'
            )
        ) AS metadata_response_blob_present,
        (
            h.client_response_body_ref IS NULL
            AND x.metadata_client_response_body_ref = format(
                'usage://request/%s/client_response_body', x.request_id
            )
            AND EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_client_response_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'client_response_body'
            )
        ) AS metadata_client_response_blob_present,
        (
            h.response_body_ref IS NULL
            AND x.metadata_response_body_ref = format(
                'usage://request/%s/response_body', x.request_id
            )
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_response_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'response_body'
            )
        ) AS metadata_response_ref_capture_corrupt,
        (
            h.client_response_body_ref IS NULL
            AND x.metadata_client_response_body_ref = format(
                'usage://request/%s/client_response_body', x.request_id
            )
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_client_response_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'client_response_body'
            )
        ) AS metadata_client_response_ref_capture_corrupt,
        (
            h.request_body_ref IS NULL
            AND x.metadata_request_body_ref = format(
                'usage://request/%s/request_body', x.request_id
            )
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_request_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'request_body'
            )
        ) AS metadata_request_ref_capture_corrupt,
        (
            h.provider_request_body_ref IS NULL
            AND x.metadata_provider_request_body_ref = format(
                'usage://request/%s/provider_request_body', x.request_id
            )
            AND NOT EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_provider_request_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'provider_request_body'
            )
        ) AS metadata_provider_request_ref_capture_corrupt,
        (
            h.request_body_ref IS NULL
            AND x.metadata_request_body_ref = format(
                'usage://request/%s/request_body', x.request_id
            )
            AND EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_request_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'request_body'
            )
        ) AS metadata_request_blob_present,
        (
            h.provider_request_body_ref IS NULL
            AND x.metadata_provider_request_body_ref = format(
                'usage://request/%s/provider_request_body', x.request_id
            )
            AND EXISTS (
                SELECT 1 FROM public.usage_body_blobs AS b
                WHERE b.body_ref = x.metadata_provider_request_body_ref
                  AND b.request_id = x.request_id
                  AND b.body_field = 'provider_request_body'
            )
        ) AS metadata_provider_request_blob_present,
        (SELECT octet_length(b.payload_gzip)
           FROM public.usage_body_blobs AS b
          WHERE b.body_ref = h.response_body_ref
            AND b.request_id = x.request_id
            AND b.body_field = 'response_body'
          LIMIT 1) AS response_blob_bytes,
        (SELECT octet_length(b.payload_gzip)
           FROM public.usage_body_blobs AS b
          WHERE b.body_ref = h.client_response_body_ref
            AND b.request_id = x.request_id
            AND b.body_field = 'client_response_body'
          LIMIT 1) AS client_response_blob_bytes
    FROM scoped AS x
    LEFT JOIN public.usage_http_audits AS h ON h.request_id = x.request_id
),
counter_rollup AS (
    SELECT
        d.request_id,
        COUNT(*)::bigint AS counter_delta_rows,
        COUNT(*) FILTER (WHERE d.processed_at IS NULL)::bigint AS unprocessed_counter_delta_rows,
        COUNT(*) FILTER (
            WHERE d.kind = 'provider_api_key' AND d.total_tokens_delta <> 0
        )::bigint AS provider_token_delta_rows,
        COUNT(*) FILTER (
            WHERE d.kind = 'provider_api_key' AND d.total_cost_usd_delta <> 0
        )::bigint AS provider_cost_delta_rows,
        COUNT(*) FILTER (
            WHERE d.kind = 'provider_monthly' AND d.total_cost_usd_delta <> 0
        )::bigint AS provider_monthly_cost_delta_rows,
        COALESCE(SUM(d.total_tokens_delta) FILTER (WHERE d.kind = 'provider_api_key'), 0)::bigint
            AS provider_token_delta_net,
        COALESCE(SUM(d.total_cost_usd_delta) FILTER (WHERE d.kind = 'provider_api_key'), 0.0)
            AS provider_cost_delta_net,
        COALESCE(SUM(d.total_cost_usd_delta) FILTER (WHERE d.kind = 'provider_monthly'), 0.0)
            AS provider_monthly_cost_delta_net
    FROM public.usage_counter_deltas AS d
    JOIN scoped AS x ON x.request_id = d.request_id
    GROUP BY d.request_id
),
entitlement_rollup AS (
    SELECT
        e.request_id,
        COUNT(*)::bigint AS entitlement_ledger_rows,
        COALESCE(SUM(e.amount_usd), 0.0) AS entitlement_amount_usd,
        MIN(e.created_at) AS first_entitlement_ledger_at,
        MAX(e.created_at) AS last_entitlement_ledger_at
    FROM public.entitlement_usage_ledgers AS e
    JOIN scoped AS x ON x.request_id = e.request_id
    GROUP BY e.request_id
),
audit_rollup AS (
    SELECT
        a.request_id,
        COUNT(*)::bigint AS audit_rows,
        BOOL_OR(a.status_code BETWEEN 200 AND 299) AS audit_has_2xx,
        BOOL_OR(
            COALESCE((a.event_metadata::jsonb ->> 'stream_completed'), 'false') = 'true'
            OR COALESCE((a.event_metadata::jsonb ->> 'observed_finish'), 'false') = 'true'
        ) AS audit_has_terminal_marker,
        MAX(a.created_at) AS last_audit_at
    FROM public.audit_logs AS a
    JOIN scoped AS x ON x.request_id = a.request_id
    GROUP BY a.request_id
),
wallet_link_rollup AS (
    -- Informational only.  The current settlement implementation does not
    -- insert wallet_transactions; arbitrary admin/refund links are not charge
    -- evidence unless their category/reason semantics are separately proven.
    SELECT
        w.link_id AS request_id,
        COUNT(*)::bigint AS possible_wallet_transaction_rows,
        COALESCE(SUM(w.amount), 0.0) AS possible_wallet_transaction_amount,
        STRING_AGG(DISTINCT COALESCE(w.link_type, ''), ',') AS possible_wallet_link_types,
        STRING_AGG(DISTINCT COALESCE(w.reason_code, ''), ',') AS possible_wallet_reason_codes
    FROM public.wallet_transactions AS w
    JOIN scoped AS x ON x.request_id = w.link_id
    GROUP BY w.link_id
),
facts AS (
    SELECT
        x.*,
        f.input_tokens AS canonical_input_tokens,
        f.effective_input_tokens AS canonical_effective_input_tokens,
        f.output_tokens AS canonical_output_tokens,
        f.cache_creation_input_tokens AS canonical_cache_creation_tokens,
        f.cache_read_input_tokens AS canonical_cache_read_tokens,
        f.total_tokens AS canonical_total_tokens,
        f.total_cost_usd AS canonical_total_cost_usd,
        f.actual_total_cost_usd AS canonical_actual_total_cost_usd,
        f.billing_status AS facts_billing_status,
        f.finalized_at AS facts_finalized_at
    FROM scoped AS x
    LEFT JOIN public.usage_billing_facts AS f ON f.request_id = x.request_id
),
evidence AS (
    SELECT
        f.*,
        p.as_of_ts AS analysis_as_of_ts,
        (f.billing_treat_as_completed_marker IS TRUE)
            AS billing_treat_as_completed_proven,
        COALESCE(c.candidate_rows, 0) AS candidate_rows,
        COALESCE(c.success_candidate_rows, 0) AS success_candidate_rows,
        COALESCE(c.streaming_candidate_rows, 0) AS streaming_candidate_rows,
        COALESCE(c.pending_candidate_rows, 0) AS pending_candidate_rows,
        COALESCE(c.failed_or_cancelled_candidate_rows, 0)
            AS failed_or_cancelled_candidate_rows,
        COALESCE(c.identity_matched_candidate_rows, 0)
            AS identity_matched_candidate_rows,
        (
            NOT f.candidate_identity_conflict
            AND COALESCE(c.identity_matched_candidate_rows, 0) = 1
        ) AS candidate_identity_proven,
        (
            COALESCE(c.has_durable_terminal_candidate, FALSE)
            AND NOT COALESCE(c.has_open_stream_candidate, FALSE)
            AND NOT COALESCE(c.has_open_pending_candidate, FALSE)
        ) AS has_durable_terminal_candidate,
        COALESCE(c.has_durable_terminal_2xx_candidate, FALSE)
            AS has_durable_terminal_2xx_candidate,
        FALSE AS terminal_proof_verified_by_query,
        COALESCE(c.has_open_stream_candidate, FALSE) AS has_open_stream_candidate,
        COALESCE(c.has_open_pending_candidate, FALSE) AS has_open_pending_candidate,
        c.durable_candidate_finished_at,
        COALESCE(c.candidate_success_2xx, FALSE) AS candidate_success_2xx,
        COALESCE(c.any_stream_completed_marker, FALSE) AS any_stream_completed_marker,
        COALESCE(h.body_capture_mode, 'no_http_audit') AS body_capture_mode,
        h.request_body_state,
        h.provider_request_body_state,
        h.response_body_state,
        h.client_response_body_state,
        COALESCE(h.has_request_headers, FALSE) AS has_request_headers,
        COALESCE(h.has_provider_request_headers, FALSE) AS has_provider_request_headers,
        COALESCE(h.has_response_headers, FALSE) AS has_response_headers,
        COALESCE(h.has_client_response_headers, FALSE) AS has_client_response_headers,
        COALESCE(h.request_body_ref_present, FALSE) AS request_body_ref_present,
        COALESCE(h.provider_request_body_ref_present, FALSE) AS provider_request_body_ref_present,
        COALESCE(h.response_body_ref_present, FALSE) AS response_body_ref_present,
        COALESCE(h.client_response_body_ref_present, FALSE) AS client_response_body_ref_present,
        COALESCE(h.effective_request_body_ref_present, FALSE)
            AS effective_request_body_ref_present,
        COALESCE(h.effective_provider_request_body_ref_present, FALSE)
            AS effective_provider_request_body_ref_present,
        COALESCE(h.effective_response_body_ref_present, FALSE)
            AS effective_response_body_ref_present,
        COALESCE(h.effective_client_response_body_ref_present, FALSE)
            AS effective_client_response_body_ref_present,
        COALESCE(h.request_blob_present, FALSE) AS request_blob_present,
        COALESCE(h.provider_request_blob_present, FALSE) AS provider_request_blob_present,
        COALESCE(h.response_blob_present, FALSE) AS response_blob_present,
        COALESCE(h.client_response_blob_present, FALSE) AS client_response_blob_present,
        COALESCE(h.request_ref_capture_corrupt, FALSE) AS request_ref_capture_corrupt,
        COALESCE(h.provider_request_ref_capture_corrupt, FALSE)
            AS provider_request_ref_capture_corrupt,
        COALESCE(h.response_ref_capture_corrupt, FALSE) AS response_ref_capture_corrupt,
        COALESCE(h.client_response_ref_capture_corrupt, FALSE)
            AS client_response_ref_capture_corrupt,
        COALESCE(h.metadata_response_blob_present, FALSE) AS metadata_response_blob_present,
        COALESCE(h.metadata_client_response_blob_present, FALSE)
            AS metadata_client_response_blob_present,
        COALESCE(h.metadata_request_blob_present, FALSE)
            AS metadata_request_blob_present,
        COALESCE(h.metadata_provider_request_blob_present, FALSE)
            AS metadata_provider_request_blob_present,
        COALESCE(h.metadata_response_ref_capture_corrupt, FALSE)
            AS metadata_response_ref_capture_corrupt,
        COALESCE(h.metadata_client_response_ref_capture_corrupt, FALSE)
            AS metadata_client_response_ref_capture_corrupt,
        COALESCE(h.metadata_request_ref_capture_corrupt, FALSE)
            AS metadata_request_ref_capture_corrupt,
        COALESCE(h.metadata_provider_request_ref_capture_corrupt, FALSE)
            AS metadata_provider_request_ref_capture_corrupt,
        h.response_blob_bytes,
        h.client_response_blob_bytes,
        COALESCE(d.counter_delta_rows, 0) AS counter_delta_rows,
        COALESCE(d.unprocessed_counter_delta_rows, 0) AS unprocessed_counter_delta_rows,
        COALESCE(d.provider_token_delta_rows, 0) AS provider_token_delta_rows,
        COALESCE(d.provider_cost_delta_rows, 0) AS provider_cost_delta_rows,
        COALESCE(d.provider_monthly_cost_delta_rows, 0)
            AS provider_monthly_cost_delta_rows,
        COALESCE(d.provider_token_delta_net, 0) AS provider_token_delta_net,
        COALESCE(d.provider_cost_delta_net, 0.0) AS provider_cost_delta_net,
        COALESCE(d.provider_monthly_cost_delta_net, 0.0)
            AS provider_monthly_cost_delta_net,
        COALESCE(e.entitlement_ledger_rows, 0) AS entitlement_ledger_rows,
        COALESCE(e.entitlement_amount_usd, 0.0) AS entitlement_amount_usd,
        e.first_entitlement_ledger_at,
        e.last_entitlement_ledger_at,
        COALESCE(a.audit_rows, 0) AS audit_rows,
        COALESCE(a.audit_has_2xx, FALSE) AS audit_has_2xx,
        COALESCE(a.audit_has_terminal_marker, FALSE) AS audit_has_terminal_marker,
        a.last_audit_at,
        COALESCE(w.possible_wallet_transaction_rows, 0) AS possible_wallet_transaction_rows,
        COALESCE(w.possible_wallet_transaction_amount, 0.0)
            AS possible_wallet_transaction_amount,
        w.possible_wallet_link_types,
        w.possible_wallet_reason_codes,
        (
            NOT COALESCE(h.response_ref_capture_corrupt, FALSE)
            AND NOT COALESCE(h.metadata_response_ref_capture_corrupt, FALSE)
            AND (
            (lower(COALESCE(h.response_body_state, '')) = 'reference'
             AND (COALESCE(h.response_blob_present, FALSE)
                  OR COALESCE(h.metadata_response_blob_present, FALSE)))
            OR (lower(COALESCE(h.response_body_state, '')) = 'inline'
                AND f.legacy_response_body_present)
            OR (h.response_body_state IS NULL
                AND (
                    f.legacy_response_body_present
                    OR (
                        COALESCE(h.effective_response_body_ref_present, FALSE)
                        AND (
                            COALESCE(h.response_blob_present, FALSE)
                            OR COALESCE(h.metadata_response_blob_present, FALSE)
                        )
                    )
                ))
            )
        ) AS response_capture_reconstructable,
        (
            NOT COALESCE(h.client_response_ref_capture_corrupt, FALSE)
            AND NOT COALESCE(h.metadata_client_response_ref_capture_corrupt, FALSE)
            AND (
            (lower(COALESCE(h.client_response_body_state, '')) = 'reference'
             AND (COALESCE(h.client_response_blob_present, FALSE)
                  OR COALESCE(h.metadata_client_response_blob_present, FALSE)))
            OR (lower(COALESCE(h.client_response_body_state, '')) = 'inline'
                AND f.legacy_client_response_body_present)
            OR (h.client_response_body_state IS NULL
                AND (
                    f.legacy_client_response_body_present
                    OR (
                        COALESCE(h.effective_client_response_body_ref_present, FALSE)
                        AND (
                            COALESCE(h.client_response_blob_present, FALSE)
                            OR COALESCE(h.metadata_client_response_blob_present, FALSE)
                        )
                    )
                ))
            )
        ) AS client_response_capture_reconstructable,
        (
            lower(COALESCE(f.effective_snapshot_status, '')) = 'complete'
            AND NOT (
                f.settlement_snapshot_status IS NOT NULL
                AND f.metadata_snapshot_status IS NOT NULL
                AND lower(f.settlement_snapshot_status)
                    <> lower(f.metadata_snapshot_status)
            )
            AND (
                COALESCE(
                    f.billing_actual_total_cost_usd,
                    f.metadata_snapshot_actual_total_cost_usd
                ) IS NOT NULL
            )
            AND COALESCE(
                f.billing_actual_total_cost_usd,
                f.metadata_snapshot_actual_total_cost_usd
            ) >= 0
            AND lower(COALESCE(
                f.billing_actual_total_cost_usd,
                f.metadata_snapshot_actual_total_cost_usd
            )::text)
                NOT IN ('nan', 'infinity', '-infinity')
            AND (
                (
                    COALESCE(f.billing_rule_id, f.metadata_snapshot_rule_id) IS NOT NULL
                    AND COALESCE(
                        f.billing_rule_version,
                        f.metadata_snapshot_rule_version
                    ) IS NOT NULL
                    AND COALESCE(
                        f.billing_pricing_source,
                        f.metadata_snapshot_pricing_source
                    ) IS NOT NULL
                )
                OR (
                    (f.is_free_tier IS TRUE OR f.metadata_snapshot_is_free_tier IS TRUE)
                    AND COALESCE(
                        f.billing_pricing_source,
                        f.metadata_snapshot_pricing_source
                    ) IS NOT NULL
                )
            )
        ) AS complete_pricing_snapshot,
        (
            (
                f.settlement_snapshot_status IS NOT NULL
                AND f.metadata_snapshot_status IS NOT NULL
                AND lower(f.settlement_snapshot_status)
                    <> lower(f.metadata_snapshot_status)
            )
            OR (
                f.metadata_flat_snapshot_status IS NOT NULL
                AND f.metadata_nested_snapshot_status IS NOT NULL
                AND lower(f.metadata_flat_snapshot_status)
                    <> lower(f.metadata_nested_snapshot_status)
            )
            OR (
                f.metadata_billing_snapshot_status IS NOT NULL
                AND f.metadata_settlement_snapshot_status IS NOT NULL
                AND lower(f.metadata_billing_snapshot_status)
                    <> lower(f.metadata_settlement_snapshot_status)
            )
        ) AS snapshot_status_conflict,
        (
            f.settlement_billing_status IS NOT NULL
            AND f.usage_billing_status IS NOT NULL
            AND lower(f.settlement_billing_status)
                <> lower(f.usage_billing_status)
        ) AS billing_status_conflict,
        (
            f.effective_billing_status IN ('settled', 'void', 'insufficient_quota')
            OR f.usage_finalized_at IS NOT NULL
            OR f.settlement_finalized_at IS NOT NULL
            OR f.settlement_wallet_id IS NOT NULL
            OR f.settlement_wallet_balance_before IS NOT NULL
            OR f.settlement_wallet_balance_after IS NOT NULL
            OR f.settlement_wallet_recharge_balance_before IS NOT NULL
            OR f.settlement_wallet_recharge_balance_after IS NOT NULL
            OR f.settlement_wallet_gift_balance_before IS NOT NULL
            OR f.settlement_wallet_gift_balance_after IS NOT NULL
            OR f.settlement_provider_monthly_used_usd IS NOT NULL
            OR f.usage_wallet_id IS NOT NULL
            OR f.usage_wallet_balance_before IS NOT NULL
            OR f.usage_wallet_balance_after IS NOT NULL
            OR f.usage_wallet_recharge_balance_before IS NOT NULL
            OR f.usage_wallet_recharge_balance_after IS NOT NULL
            OR f.usage_wallet_gift_balance_before IS NOT NULL
            OR f.usage_wallet_gift_balance_after IS NOT NULL
            OR COALESCE(e.entitlement_ledger_rows, 0) > 0
            OR COALESCE(d.counter_delta_rows, 0) > 0
        ) AS existing_settlement_evidence,
        (
            f.status = 'completed'
            AND f.effective_billing_status = 'pending'
            AND f.status_code = 200
            AND f.first_byte_time_ms IS NOT NULL
            AND (f.response_time_ms IS NULL OR f.response_time_ms = 0)
            AND COALESCE(f.canonical_total_tokens, 0) = 0
            AND COALESCE(f.canonical_actual_total_cost_usd, 0) = 0
            AND NOT (
                COALESCE(h.has_request_headers, FALSE)
                OR f.legacy_request_headers_present
            )
            AND NOT (
                COALESCE(h.has_provider_request_headers, FALSE)
                OR f.legacy_provider_request_headers_present
            )
            AND NOT (
                COALESCE(h.has_response_headers, FALSE)
                OR f.legacy_response_headers_present
            )
            AND NOT (
                COALESCE(h.has_client_response_headers, FALSE)
                OR f.legacy_client_response_headers_present
            )
            AND NOT COALESCE(h.effective_response_body_ref_present, FALSE)
            AND NOT COALESCE(h.effective_client_response_body_ref_present, FALSE)
            AND NOT f.legacy_response_body_present
                AND f.created_at < p.as_of_ts - (p.stale_minutes * INTERVAL '1 minute')
        ) AS cleanup_promotion_fingerprint,
        (
            f.status IN ('pending', 'streaming', 'completed', 'failed', 'cancelled')
            AND f.usage_billing_status = 'pending'
            AND f.usage_finalized_at IS NULL
            AND f.effective_billing_status = 'pending'
            AND f.settlement_finalized_at IS NULL
            AND NOT f.is_video_contract
            AND f.created_at < p.as_of_ts - (p.stale_minutes * INTERVAL '1 minute')
        ) AS stale_cleanup_candidate
    FROM facts AS f
    LEFT JOIN candidate_rollup AS c ON c.request_id = f.request_id
    LEFT JOIN http_rollup AS h ON h.request_id = f.request_id
    LEFT JOIN counter_rollup AS d ON d.request_id = f.request_id
    LEFT JOIN entitlement_rollup AS e ON e.request_id = f.request_id
    LEFT JOIN audit_rollup AS a ON a.request_id = f.request_id
    LEFT JOIN wallet_link_rollup AS w ON w.request_id = f.request_id
    CROSS JOIN params AS p
),
categorized AS (
    SELECT
        e.*,
        CASE
            WHEN e.effective_billing_status = 'settled'
                THEN 'already_settled_do_not_touch'
            WHEN e.effective_billing_status = 'void'
                THEN 'intentional_void_do_not_charge'
            WHEN e.effective_billing_status = 'insufficient_quota'
                THEN 'settlement_attempted_insufficient_quota'
            WHEN e.stale_cleanup_candidate
                THEN 'stale_cleanup_failed_void'
            WHEN e.status = 'cancelled'
                 AND e.status_code = 499
                 AND e.effective_billing_status = 'pending'
                 -- Only a persisted literal true proves the drained-cancel
                 -- intent.  It remains insufficient without the independent
                 -- terminal/pricing/identity/idempotency gates below.
                 AND e.billing_treat_as_completed_proven
                 AND e.complete_pricing_snapshot
                 AND NOT e.existing_settlement_evidence
                 AND NOT e.snapshot_status_conflict
                 AND NOT e.billing_status_conflict
                 AND e.candidate_identity_proven
                 AND e.has_durable_terminal_candidate
                 -- A scheduler/candidate 2xx marker is not provider usage.
                 -- Keep this row eligible for manual review only when the
                 -- upstream response can be reconstructed and independently
                 -- parsed for a terminal usage summary.
                 AND e.has_durable_terminal_2xx_candidate
                 AND e.response_capture_reconstructable
                THEN 'cancelled_drained_snapshot_manual_review'
            WHEN e.status = 'cancelled'
                 AND e.status_code = 499
                THEN 'cancelled_partial_no_charge'
            WHEN e.status = 'failed'
                 OR COALESCE(e.status_code, 0) >= 400
                THEN 'failed_or_user_error_no_bill'
            WHEN e.is_video_contract
                THEN 'async_video_manual_poll_snapshot_required'
            WHEN e.status = 'completed'
                 AND e.effective_billing_status = 'pending'
                 AND e.complete_pricing_snapshot
                 AND NOT e.existing_settlement_evidence
                 AND NOT e.snapshot_status_conflict
                 AND NOT e.billing_status_conflict
                 AND e.candidate_identity_proven
                 AND e.has_durable_terminal_candidate
                 AND e.has_durable_terminal_2xx_candidate
                 AND (
                     e.response_capture_reconstructable
                     OR e.client_response_capture_reconstructable
                 )
                THEN 'false_success_snapshot_manual_review'
            WHEN e.status = 'completed'
                 AND e.effective_billing_status = 'pending'
                 AND e.has_durable_terminal_2xx_candidate
                 AND e.has_durable_terminal_candidate
                 AND e.candidate_identity_proven
                 AND (
                     e.response_capture_reconstructable
                     OR e.client_response_capture_reconstructable
                 )
                 AND NOT e.existing_settlement_evidence
                 AND NOT e.snapshot_status_conflict
                 AND NOT e.billing_status_conflict
                THEN 'false_success_body_reconstruct_manual_review'
            WHEN e.cleanup_promotion_fingerprint
                 AND NOT e.response_capture_reconstructable
                 AND NOT e.client_response_capture_reconstructable
                THEN 'false_success_no_terminal_evidence'
            WHEN e.counter_delta_rows > 0
                 OR e.audit_rows > 0
                THEN 'aggregate_or_audit_only_manual_review'
            ELSE 'irrecoverable_or_unclassified_no_automatic_charge'
        END AS recovery_category
    FROM evidence AS e
),
classified AS (
    SELECT
        c.*,
        CASE
            WHEN c.stale_cleanup_candidate
                THEN 'failed_and_void'
            ELSE 'not_selected_by_stale_sweep'
        END AS cleanup_action,
        CASE
            WHEN c.effective_billing_status = 'settled'
                THEN 'no_action'
            WHEN c.effective_billing_status IN ('void', 'insufficient_quota')
                THEN 'do_not_replay_without_business_decision'
            WHEN c.recovery_category = 'stale_cleanup_failed_void'
                THEN 'stale_sweep_failed_and_void; candidate_success_is_diagnostic; no_promotion'
            WHEN c.recovery_category = 'false_success_snapshot_manual_review'
                THEN 'report_only_verify_terminal_payload_snapshot_and_idempotency; no replay'
            WHEN c.recovery_category = 'cancelled_drained_snapshot_manual_review'
                THEN 'report_only_verify_terminal_payload_and_original_treat_as_completed_event; preserve 499; no replay'
            WHEN c.recovery_category = 'false_success_body_reconstruct_manual_review'
                THEN 'report_only_offline_decompress_parse_terminal_usage_pin_historical_price; no replay'
            WHEN c.recovery_category = 'async_video_manual_poll_snapshot_required'
                THEN 'report_only_route_to_separately_approved_video_poller; no replay'
            ELSE 'no_safe_automatic_charge'
        END AS proposed_action,
        CASE
            WHEN c.recovery_category = 'stale_cleanup_failed_void'
                THEN 'high_no_charge_after_timeout'
            WHEN c.recovery_category IN (
                'false_success_snapshot_manual_review',
                'cancelled_drained_snapshot_manual_review'
            ) THEN 'manual_only_terminal_proof_and_lock_recheck_required'
            WHEN c.recovery_category = 'false_success_body_reconstruct_manual_review'
                THEN 'manual_only_terminal_marker_and_historical_price_required'
            WHEN c.recovery_category = 'async_video_manual_poll_snapshot_required'
                THEN 'manual_video_workflow_only'
            WHEN c.cleanup_promotion_fingerprint
                THEN 'low_without_body_or_snapshot'
            ELSE 'not_a_recovery_candidate'
        END AS confidence
    FROM categorized AS c
)
SELECT
    request_id,
    analysis_as_of_ts,
    created_at,
    status,
    status_code,
    outcome_class,
    effective_billing_status,
    usage_billing_status,
    settlement_billing_status,
    billing_status_conflict,
    effective_snapshot_status,
    metadata_snapshot_status,
    metadata_flat_snapshot_status,
    metadata_nested_snapshot_status,
    metadata_billing_snapshot_status,
    metadata_settlement_snapshot_status,
    metadata_snapshot_actual_total_cost_usd,
    metadata_snapshot_rule_id,
    metadata_snapshot_rule_version,
    metadata_snapshot_pricing_source,
    metadata_snapshot_is_free_tier,
    settlement_snapshot_status,
    settlement_snapshot_json_present,
    settlement_billing_dimensions_present,
    request_type,
    api_format,
    endpoint_api_format,
    provider_id,
    provider_endpoint_id,
    provider_api_key_id,
    usage_candidate_id,
    usage_candidate_index,
    selected_candidate_id,
    selected_candidate_index,
    selected_provider_id,
    selected_endpoint_id,
    selected_provider_api_key_id,
    candidate_identity_conflict,
    model,
    is_stream,
    upstream_is_stream,
    first_byte_time_ms,
    response_time_ms,
    billing_treat_as_completed_marker,
    billing_treat_as_completed_proven,
    usage_finalized_at,
    settlement_finalized_at,
    settlement_wallet_id,
    settlement_wallet_balance_before,
    settlement_wallet_balance_after,
    settlement_wallet_recharge_balance_before,
    settlement_wallet_recharge_balance_after,
    settlement_wallet_gift_balance_before,
    settlement_wallet_gift_balance_after,
    settlement_provider_monthly_used_usd,
    usage_wallet_id,
    usage_wallet_balance_before,
    usage_wallet_balance_after,
    usage_wallet_recharge_balance_before,
    usage_wallet_recharge_balance_after,
    usage_wallet_gift_balance_before,
    usage_wallet_gift_balance_after,
    canonical_input_tokens,
    canonical_effective_input_tokens,
    canonical_output_tokens,
    canonical_cache_creation_tokens,
    canonical_cache_read_tokens,
    canonical_total_tokens,
    canonical_total_cost_usd,
    canonical_actual_total_cost_usd,
    billing_rule_id,
    billing_rule_version,
    billing_pricing_source,
    is_free_tier,
    candidate_rows,
    success_candidate_rows,
    streaming_candidate_rows,
    pending_candidate_rows,
    identity_matched_candidate_rows,
    candidate_identity_proven,
    has_durable_terminal_candidate,
    has_durable_terminal_2xx_candidate,
    terminal_proof_verified_by_query,
    has_open_stream_candidate,
    has_open_pending_candidate,
    durable_candidate_finished_at,
    body_capture_mode,
    request_body_state,
    provider_request_body_state,
    response_body_state,
    client_response_body_state,
    has_request_headers,
    has_provider_request_headers,
    has_response_headers,
    has_client_response_headers,
    legacy_request_headers_present,
    legacy_provider_request_headers_present,
    legacy_response_headers_present,
    legacy_client_response_headers_present,
    request_body_ref_present,
    provider_request_body_ref_present,
    response_body_ref_present,
    client_response_body_ref_present,
    effective_request_body_ref_present,
    effective_provider_request_body_ref_present,
    effective_response_body_ref_present,
    effective_client_response_body_ref_present,
    request_blob_present,
    provider_request_blob_present,
    response_blob_present,
    client_response_blob_present,
    request_ref_capture_corrupt,
    provider_request_ref_capture_corrupt,
    response_ref_capture_corrupt,
    client_response_ref_capture_corrupt,
    metadata_response_blob_present,
    metadata_client_response_blob_present,
    metadata_request_blob_present,
    metadata_provider_request_blob_present,
    metadata_response_ref_capture_corrupt,
    metadata_client_response_ref_capture_corrupt,
    metadata_request_ref_capture_corrupt,
    metadata_provider_request_ref_capture_corrupt,
    response_blob_bytes,
    counter_delta_rows,
    unprocessed_counter_delta_rows,
    provider_token_delta_rows,
    provider_cost_delta_rows,
    provider_monthly_cost_delta_rows,
    provider_token_delta_net,
    provider_cost_delta_net,
    provider_monthly_cost_delta_net,
    entitlement_ledger_rows,
    entitlement_amount_usd,
    audit_rows,
    audit_has_terminal_marker,
    possible_wallet_transaction_rows,
    possible_wallet_link_types,
    possible_wallet_reason_codes,
    response_capture_reconstructable,
    client_response_capture_reconstructable,
    complete_pricing_snapshot,
    snapshot_status_conflict,
    existing_settlement_evidence,
    cleanup_promotion_fingerprint,
    stale_cleanup_candidate,
    cleanup_action,
    recovery_category,
    proposed_action,
    confidence
FROM classified
ORDER BY created_at ASC, request_id ASC;

ROLLBACK;
