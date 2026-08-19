-- Request outcome columns were introduced without backfilling historical usage. Reset only the
-- incompatible outcome read model; preserve request volume, tokens, costs, cache, billing, and
-- every raw usage row.
UPDATE public.stats_hourly
SET success_requests = 0,
    error_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_hourly_user
SET success_requests = 0,
    error_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_daily
SET success_requests = 0,
    error_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_daily_api_key
SET success_requests = 0,
    error_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_user_daily
SET success_requests = 0,
    error_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_summary
SET all_time_success_requests = 0,
    all_time_error_requests = 0,
    all_time_sla_eligible_requests = 0,
    all_time_user_error_requests = 0;

UPDATE public.stats_user_summary
SET all_time_success_requests = 0,
    all_time_error_requests = 0,
    all_time_sla_eligible_requests = 0,
    all_time_user_error_requests = 0;

UPDATE public.stats_user_daily_model
SET success_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_user_daily_provider
SET success_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_user_daily_api_format
SET success_requests = 0,
    sla_eligible_requests = 0,
    user_error_requests = 0;

UPDATE public.stats_daily_error
SET count = 0;

UPDATE public.provider_api_keys
SET success_count = 0,
    error_count = 0,
    sla_eligible_count = 0,
    user_error_count = 0;

-- Preserve request/token/cost deltas; only discard incompatible pre-reset outcome increments.
UPDATE public.usage_counter_deltas
SET success_count_delta = 0,
    error_count_delta = 0,
    sla_eligible_count_delta = 0,
    user_error_count_delta = 0
WHERE kind = 'provider_api_key';
