use aether_data_contracts::repository::candidates::{
    RequestCandidateStatus, StoredRequestCandidate, UpsertRequestCandidateRecord,
};
use aether_data_contracts::DataLayerError;
use serde_json::json;
use std::time::Instant;
use tracing::info;

use crate::data::GatewayDataState;

use super::{
    now_unix_secs, record_completed_cleanup_run, record_failed_cleanup_run, system_config_bool,
    system_config_u64, system_config_usize,
};

const DEFAULT_TERMINAL_RECONCILIATION_BATCH_SIZE: usize = 256;
const MAX_TERMINAL_RECONCILIATION_BATCH_SIZE: usize = 1_000;
const DEFAULT_TERMINAL_RECONCILIATION_BATCHES_PER_RUN: usize = 4;
const MAX_TERMINAL_RECONCILIATION_BATCHES_PER_RUN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RequestCandidateTerminalSweepCursor {
    created_at_unix_ms: u64,
    id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct RequestCandidateTerminalSweepSummary {
    pub(super) scanned: usize,
    pub(super) reconciled: usize,
    pub(super) identity_mismatches: usize,
    pub(super) usage_not_terminal: usize,
    pub(super) write_conflicts: usize,
}

fn terminal_candidate_status_from_usage(status: &str) -> Option<RequestCandidateStatus> {
    if status.eq_ignore_ascii_case("completed") {
        Some(RequestCandidateStatus::Success)
    } else if status.eq_ignore_ascii_case("failed") {
        Some(RequestCandidateStatus::Failed)
    } else if status.eq_ignore_ascii_case("cancelled") {
        Some(RequestCandidateStatus::Cancelled)
    } else {
        None
    }
}

fn terminal_candidate_record_from_usage(
    candidate: StoredRequestCandidate,
    usage: &aether_data_contracts::repository::usage::StoredRequestUsageAudit,
    status: RequestCandidateStatus,
) -> UpsertRequestCandidateRecord {
    UpsertRequestCandidateRecord {
        id: candidate.id,
        request_id: candidate.request_id,
        user_id: candidate.user_id,
        api_key_id: candidate.api_key_id,
        username: candidate.username,
        api_key_name: candidate.api_key_name,
        candidate_index: candidate.candidate_index,
        retry_index: candidate.retry_index,
        provider_id: candidate.provider_id,
        endpoint_id: candidate.endpoint_id,
        key_id: candidate.key_id,
        status,
        skip_reason: candidate.skip_reason,
        is_cached: Some(candidate.is_cached),
        status_code: usage.status_code.or(candidate.status_code),
        error_type: usage.error_category.clone().or(candidate.error_type),
        error_message: usage.error_message.clone().or(candidate.error_message),
        latency_ms: usage.response_time_ms.or(candidate.latency_ms),
        concurrent_requests: candidate.concurrent_requests,
        extra_data: candidate.extra_data,
        required_capabilities: candidate.required_capabilities,
        created_at_unix_ms: Some(candidate.created_at_unix_ms),
        started_at_unix_ms: candidate.started_at_unix_ms,
        finished_at_unix_ms: usage
            .finalized_at_unix_secs
            .map(|value| value.saturating_mul(1_000)),
    }
}

/// Repair the candidate half of a terminal handoff from durable usage facts.
///
/// This is intentionally one-way: candidate state never creates or changes a
/// usage row. Missing candidate identity, an outcome outside the three durable
/// terminal states, or a conflicting candidate index all fail closed.
pub(super) async fn reconcile_active_request_candidates_from_terminal_usage_once(
    data: &GatewayDataState,
    cursor: &mut Option<RequestCandidateTerminalSweepCursor>,
) -> Result<RequestCandidateTerminalSweepSummary, DataLayerError> {
    if !data.has_request_candidate_reader()
        || !data.has_request_candidate_writer()
        || !data.has_usage_reader()
    {
        *cursor = None;
        return Ok(RequestCandidateTerminalSweepSummary::default());
    }

    let batch_size = system_config_usize(
        data,
        "request_candidate_terminal_reconciliation_batch_size",
        DEFAULT_TERMINAL_RECONCILIATION_BATCH_SIZE,
    )
    .await?
    .clamp(1, MAX_TERMINAL_RECONCILIATION_BATCH_SIZE);
    let max_batches = system_config_usize(
        data,
        "request_candidate_terminal_reconciliation_batches_per_run",
        DEFAULT_TERMINAL_RECONCILIATION_BATCHES_PER_RUN,
    )
    .await?
    .clamp(1, MAX_TERMINAL_RECONCILIATION_BATCHES_PER_RUN);

    let mut summary = RequestCandidateTerminalSweepSummary::default();
    let mut scan_cursor = cursor.clone();
    for _ in 0..max_batches {
        let candidates = data
            .list_active_request_candidates_after(
                scan_cursor.as_ref().map(|value| value.created_at_unix_ms),
                scan_cursor.as_ref().map(|value| value.id.as_str()),
                batch_size,
            )
            .await?;
        if candidates.is_empty() {
            scan_cursor = None;
            break;
        }

        let page_is_full = candidates.len() == batch_size;
        let page_cursor = candidates
            .last()
            .map(|candidate| RequestCandidateTerminalSweepCursor {
                created_at_unix_ms: candidate.created_at_unix_ms,
                id: candidate.id.clone(),
            });
        summary.scanned = summary.scanned.saturating_add(candidates.len());
        let mut terminal_updates = Vec::new();
        for candidate in candidates {
            if !matches!(
                candidate.status,
                RequestCandidateStatus::Pending | RequestCandidateStatus::Streaming
            ) {
                continue;
            }
            let Some(usage) = data
                .find_request_usage_by_request_id_shallow(candidate.request_id.as_str())
                .await?
            else {
                summary.usage_not_terminal = summary.usage_not_terminal.saturating_add(1);
                continue;
            };
            let Some(status) = terminal_candidate_status_from_usage(usage.status.as_str()) else {
                summary.usage_not_terminal = summary.usage_not_terminal.saturating_add(1);
                continue;
            };
            if usage.finalized_at_unix_secs.is_none() {
                summary.usage_not_terminal = summary.usage_not_terminal.saturating_add(1);
                continue;
            }
            // The typed fields are populated by repository reads with routing
            // snapshot presence semantics. Do not use the legacy metadata
            // fallback here: a present snapshot with a NULL candidate id is an
            // authoritative absence and must fail closed.
            let candidate_id_matches = usage.candidate_id.as_deref() == Some(candidate.id.as_str());
            let candidate_index_matches = usage
                .candidate_index
                .is_none_or(|index| index == u64::from(candidate.candidate_index));
            if !candidate_id_matches || !candidate_index_matches {
                summary.identity_mismatches = summary.identity_mismatches.saturating_add(1);
                continue;
            }
            terminal_updates.push(terminal_candidate_record_from_usage(
                candidate, &usage, status,
            ));
        }

        let attempted_updates = terminal_updates.len();
        let finalized = data
            .finalize_active_request_candidates_exact(terminal_updates)
            .await?;
        summary.reconciled = summary.reconciled.saturating_add(finalized);
        summary.write_conflicts = summary
            .write_conflicts
            .saturating_add(attempted_updates.saturating_sub(finalized));
        scan_cursor = page_cursor;
        *cursor = scan_cursor.clone();
        if !page_is_full {
            scan_cursor = None;
            break;
        }
    }
    *cursor = scan_cursor;
    Ok(summary)
}

pub(crate) async fn cleanup_request_candidates_once(
    data: &GatewayDataState,
) -> Result<usize, DataLayerError> {
    if !system_config_bool(data, "enable_auto_cleanup", true).await? {
        return Ok(0);
    }

    let detail_log_retention_days = system_config_u64(data, "detail_log_retention_days", 7).await?;
    let retention_days = system_config_u64(
        data,
        "request_candidates_retention_days",
        detail_log_retention_days,
    )
    .await?
    .max(3);
    let cleanup_batch_size = system_config_usize(data, "cleanup_batch_size", 1_000).await?;
    let delete_limit = system_config_usize(
        data,
        "request_candidates_cleanup_batch_size",
        cleanup_batch_size.max(1),
    )
    .await?
    .max(1);
    let cutoff_unix_secs = now_unix_secs().saturating_sub(retention_days.saturating_mul(86_400));

    let mut total_deleted = 0usize;
    loop {
        let deleted = data
            .delete_request_candidates_created_before(cutoff_unix_secs, delete_limit)
            .await?;
        total_deleted += deleted;
        if deleted < delete_limit {
            break;
        }
    }

    Ok(total_deleted)
}

pub(super) async fn run_request_candidate_cleanup_once(
    data: &GatewayDataState,
) -> Result<(), DataLayerError> {
    let started_at_unix_secs = now_unix_secs();
    let started_at = Instant::now();
    let deleted = match cleanup_request_candidates_once(data).await {
        Ok(deleted) => deleted,
        Err(err) => {
            record_failed_cleanup_run(
                data,
                "request_candidate_cleanup",
                "auto",
                started_at_unix_secs,
                started_at,
                &err,
            )
            .await;
            return Err(err);
        }
    };
    record_completed_cleanup_run(
        data,
        "request_candidate_cleanup",
        "auto",
        started_at_unix_secs,
        started_at,
        json!({ "request_candidates_deleted": deleted }),
        format!("候选记录自动清理完成，删除 {deleted} 行"),
    )
    .await;
    if deleted > 0 {
        info!(deleted, "gateway deleted expired request candidates");
    }
    Ok(())
}
