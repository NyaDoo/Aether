use std::time::Duration;

use aether_runtime_state::{MemoryRuntimeStateConfig, RuntimeState};
use axum::http::{HeaderMap, Method, Uri};
use chrono::Utc;

use crate::dashboard_realtime::{
    admission_marker_key, observe_usage_event_for_test, reserve_admission, reserve_admission_at,
    rollback_admission, settle_admission, snapshot_at, snapshot_at_millis,
    token_delta_from_usage_data,
};
use crate::usage::{UsageEvent, UsageEventData, UsageEventType};

fn current_second() -> u64 {
    Utc::now().timestamp().max(0) as u64
}

#[tokio::test]
async fn in_flight_admission_is_visible_and_failed_terminal_is_excluded() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let request_id = "dashboard-realtime-in-flight";

    let guard = reserve_admission(&runtime, request_id)
        .await
        .expect("admission reservation should succeed")
        .expect("first request should reserve a marker");

    // RPM is intentionally admitted-request based: the request is visible
    // while its upstream stream is still in flight, before a terminal usage
    // record exists.
    let in_flight = snapshot_at(&runtime, current_second())
        .await
        .expect("in-flight snapshot should succeed");
    assert_eq!(in_flight.rpm, 1);
    assert_eq!(in_flight.tpm, 0);

    assert!(guard
        .settle(true, 0)
        .await
        .expect("failed terminal should settle the reservation"));

    let after_failure = snapshot_at(&runtime, current_second())
        .await
        .expect("post-failure snapshot should succeed");
    assert_eq!(after_failure.rpm, 0);
    assert_eq!(after_failure.tpm, 0);

    // The marker is consumed atomically; a duplicate terminal report cannot
    // subtract again or alter the token total.
    assert!(!settle_admission(&runtime, request_id, true, 0)
        .await
        .expect("duplicate failed terminal should be harmless"));
}

#[tokio::test]
async fn successful_terminal_adds_token_delta_once_and_keeps_request_count() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let request_id = "dashboard-realtime-success";

    let guard = reserve_admission(&runtime, request_id)
        .await
        .expect("admission reservation should succeed")
        .expect("request should reserve a marker");
    // Exercise the terminal sink directly so dropping the guard cannot race
    // the explicit settlement below.
    guard.handoff();

    assert!(settle_admission(&runtime, request_id, false, 123)
        .await
        .expect("successful terminal should settle"));
    let first = snapshot_at(&runtime, current_second())
        .await
        .expect("success snapshot should succeed");
    assert_eq!(first.rpm, 1);
    assert_eq!(first.tpm, 123);

    // A retry carrying a different cumulative usage value must not double
    // count: the request marker makes terminal settlement idempotent.
    assert!(!settle_admission(&runtime, request_id, false, 246)
        .await
        .expect("duplicate successful terminal should be harmless"));
    let duplicate = snapshot_at(&runtime, current_second())
        .await
        .expect("duplicate snapshot should succeed");
    assert_eq!(duplicate.rpm, 1);
    assert_eq!(duplicate.tpm, 123);
}

#[tokio::test]
async fn streaming_usage_events_add_token_deltas_to_their_event_buckets() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let as_of = current_second();
    let data = |total_tokens| UsageEventData {
        provider_name: "provider".to_string(),
        model: "model".to_string(),
        total_tokens: Some(total_tokens),
        ..UsageEventData::default()
    };

    // Streaming observations are deltas by contract.  They may arrive in
    // separate seconds and are summed by the fixed-clock rolling snapshot.
    observe_usage_event_for_test(
        &runtime,
        &UsageEvent {
            event_type: UsageEventType::Streaming,
            request_id: "dashboard-realtime-stream-delta".to_string(),
            timestamp_ms: as_of.saturating_mul(1_000),
            data: data(100),
        },
    )
    .await
    .expect("first streaming observation should succeed");
    observe_usage_event_for_test(
        &runtime,
        &UsageEvent {
            event_type: UsageEventType::Streaming,
            request_id: "dashboard-realtime-stream-delta".to_string(),
            timestamp_ms: as_of.saturating_add(1).saturating_mul(1_000),
            data: data(50),
        },
    )
    .await
    .expect("second streaming observation should succeed");

    let snapshot = snapshot_at(&runtime, as_of.saturating_add(1))
        .await
        .expect("streaming snapshot should succeed");
    assert_eq!(snapshot.rpm, 0);
    assert_eq!(snapshot.tpm, 150);
}

#[tokio::test]
async fn streaming_replays_with_same_identity_remain_additive() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let as_of: u64 = 20_000;
    let event = |tokens| UsageEvent {
        event_type: UsageEventType::Streaming,
        request_id: "dashboard-realtime-stream-same-second".to_string(),
        timestamp_ms: as_of.saturating_mul(1_000),
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            total_tokens: Some(tokens),
            ..UsageEventData::default()
        },
    };

    // Streaming observations represent token increments, not a cumulative
    // terminal total.  Even identical request/type/timestamp fields must not
    // be deduplicated, because two chunks can legitimately share a clock
    // tick and carry equal deltas.
    observe_usage_event_for_test(&runtime, &event(7))
        .await
        .expect("first streaming delta should succeed");
    observe_usage_event_for_test(&runtime, &event(7))
        .await
        .expect("second streaming delta should succeed");

    let snapshot = snapshot_at(&runtime, as_of)
        .await
        .expect("same-second streaming snapshot should succeed");
    assert_eq!(snapshot.tpm, 14);
}

#[tokio::test]
async fn streaming_deltas_and_terminal_cumulative_usage_are_not_double_counted() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 42_050_000;
    let request_id = "dashboard-realtime-stream-terminal-reconcile";
    let candidate_id = Some("candidate-stream-terminal".to_string());

    // The live observer has already published the first 40 tokens. The
    // terminal usage snapshot is cumulative (100), so only the unseen 60
    // tokens should be added at settlement.
    observe_usage_event_for_test(
        &runtime,
        &UsageEvent {
            event_type: UsageEventType::Streaming,
            request_id: request_id.to_string(),
            timestamp_ms,
            data: UsageEventData {
                provider_name: "provider".to_string(),
                model: "model".to_string(),
                candidate_id: candidate_id.clone(),
                total_tokens: Some(40),
                ..UsageEventData::default()
            },
        },
    )
    .await
    .expect("streaming delta should succeed");
    observe_usage_event_for_test(
        &runtime,
        &UsageEvent {
            event_type: UsageEventType::Completed,
            request_id: request_id.to_string(),
            timestamp_ms: timestamp_ms + 1,
            data: UsageEventData {
                provider_name: "provider".to_string(),
                model: "model".to_string(),
                candidate_id,
                total_tokens: Some(100),
                ..UsageEventData::default()
            },
        },
    )
    .await
    .expect("terminal cumulative usage should succeed");

    let snapshot = snapshot_at_millis(&runtime, timestamp_ms + 1)
        .await
        .expect("reconciled snapshot should succeed");
    assert_eq!(snapshot.tpm, 100);
}

#[tokio::test]
async fn failed_usage_event_keeps_observed_tokens_but_removes_rpm_admission() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let request_id = "dashboard-realtime-failed-with-usage";
    reserve_admission(&runtime, request_id)
        .await
        .expect("admission reservation should succeed")
        .expect("request should reserve a marker")
        .handoff();

    let event = UsageEvent {
        event_type: UsageEventType::Failed,
        request_id: request_id.to_string(),
        timestamp_ms: current_second().saturating_mul(1_000),
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            total_tokens: Some(42),
            status_code: Some(502),
            ..UsageEventData::default()
        },
    };
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("failed usage observation should succeed");

    let snapshot = snapshot_at(&runtime, current_second())
        .await
        .expect("failed snapshot should succeed");
    // `failed_requests: excluded_from_rpm_only` applies to RPM.  Tokens already emitted by
    // an upstream stream remain part of TPM, even when the final lifecycle
    // state is failed.
    assert_eq!(snapshot.rpm, 0);
    assert_eq!(snapshot.tpm, 42);
}

#[tokio::test]
async fn terminal_usage_uses_forwarded_realtime_admission_identity() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let admission_id = "forwarded-realtime-admission";
    let timestamp_ms = 1_700_000_000_123;
    reserve_admission_at(&runtime, admission_id, timestamp_ms)
        .await
        .expect("admission reservation should succeed")
        .expect("request should reserve a marker")
        .handoff();

    // The usage lifecycle request ID can differ from the private admission
    // identity after owner forwarding.  Settlement must use the explicit
    // metadata field or the RPM event would remain stuck in the window.
    let event = UsageEvent {
        event_type: UsageEventType::Failed,
        request_id: "public-trace-after-forward".to_string(),
        timestamp_ms: timestamp_ms + 25,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            request_metadata: Some(serde_json::json!({
                "realtime_admission_id": admission_id,
            })),
            total_tokens: Some(8),
            ..UsageEventData::default()
        },
    };
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("forwarded terminal observation should succeed");

    let snapshot = snapshot_at_millis(&runtime, timestamp_ms + 25)
        .await
        .expect("forwarded snapshot should succeed");
    assert_eq!(snapshot.rpm, 0);
    assert_eq!(snapshot.tpm, 8);
}

#[tokio::test]
async fn duplicate_admission_is_ignored_and_explicit_rollback_is_idempotent() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let request_id = "dashboard-realtime-duplicate";

    let first = reserve_admission(&runtime, request_id)
        .await
        .expect("first reservation should succeed")
        .expect("first reservation should win");
    first.handoff();
    assert!(reserve_admission(&runtime, request_id)
        .await
        .expect("duplicate reservation should be handled")
        .is_none());

    assert!(rollback_admission(&runtime, request_id)
        .await
        .expect("rollback should consume the marker"));
    assert!(!rollback_admission(&runtime, request_id)
        .await
        .expect("duplicate rollback should be harmless"));
    assert_eq!(
        snapshot_at(&runtime, current_second())
            .await
            .expect("rollback snapshot should succeed")
            .rpm,
        0
    );
}

#[tokio::test]
async fn fixed_clock_snapshot_uses_a_half_open_trailing_window() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let as_of: u64 = 10_000;

    runtime
        .realtime_event_add(
            "dashboard:realtime:v1",
            "boundary",
            (as_of - 60) * 1_000,
            7,
            70,
            Duration::from_secs(120),
        )
        .await
        .expect("boundary event write should succeed");
    runtime
        .realtime_event_add(
            "dashboard:realtime:v1",
            "inside",
            (as_of - 59) * 1_000,
            1,
            11,
            Duration::from_secs(120),
        )
        .await
        .expect("inside-window event write should succeed");
    runtime
        .realtime_event_add(
            "dashboard:realtime:v1",
            "current",
            as_of * 1_000,
            2,
            22,
            Duration::from_secs(120),
        )
        .await
        .expect("current event write should succeed");

    let snapshot = snapshot_at(&runtime, as_of)
        .await
        .expect("fixed-clock snapshot should succeed");
    assert_eq!(snapshot.rpm, 3);
    assert_eq!(snapshot.tpm, 33);
}

#[test]
fn token_delta_prefers_canonical_total_and_only_falls_back_to_components() {
    let mut data = UsageEventData {
        provider_name: "provider".to_string(),
        model: "model".to_string(),
        input_tokens: Some(4),
        output_tokens: Some(6),
        // Cache counters describe portions of the input context.  They are
        // not additional usage on top of input_tokens, so a missing
        // canonical total must not count them a second time.
        cache_creation_input_tokens: Some(20),
        cache_creation_ephemeral_5m_input_tokens: Some(3),
        cache_creation_ephemeral_1h_input_tokens: Some(4),
        cache_read_input_tokens: Some(30),
        request_metadata: Some(serde_json::json!({"reasoning_tokens": 2})),
        ..UsageEventData::default()
    };
    assert_eq!(token_delta_from_usage_data(&data), 12);

    data.total_tokens = Some(10);
    assert_eq!(token_delta_from_usage_data(&data), 10);

    data.total_tokens = None;
    data.request_metadata = Some(serde_json::json!({
        "dimensions": {"reasoning_tokens": 7}
    }));
    assert_eq!(token_delta_from_usage_data(&data), 17);
}

#[test]
fn admission_marker_is_namespaced_and_trace_safe() {
    let marker = admission_marker_key("a-sensitive-trace-id");
    assert!(marker.starts_with("dashboard:realtime:v1:admission:"));
    assert!(!marker.contains("a-sensitive-trace-id"));
}

#[test]
fn realtime_dashboard_route_is_classified_as_public_support() {
    let decision = crate::control::classify_control_route(
        &Method::GET,
        &Uri::from_static("/api/dashboard/realtime?scope=site"),
        &HeaderMap::new(),
    )
    .expect("realtime dashboard route should classify");
    assert_eq!(decision.route_class.as_deref(), Some("public_support"));
    assert_eq!(decision.route_family.as_deref(), Some("dashboard"));
    assert_eq!(decision.route_kind.as_deref(), Some("realtime"));
    assert_eq!(
        decision.auth_endpoint_signature.as_deref(),
        Some("user:dashboard")
    );
    assert!(!decision.is_execution_runtime_candidate());
}

#[tokio::test]
async fn realtime_dashboard_route_requires_a_user_bearer_token() {
    let app = crate::build_router_with_state(crate::AppState::new().expect("gateway should build"));
    let request = axum::http::Request::builder()
        .method(Method::GET)
        .uri("/api/dashboard/realtime")
        .body(axum::body::Body::empty())
        .expect("request should build");
    let response = super::send_request(app, request).await;
    assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
    let payload = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body should read");
    let payload: serde_json::Value =
        serde_json::from_slice(&payload).expect("response should be json");
    assert_eq!(payload["detail"], "缺少用户凭证");
}

#[tokio::test]
async fn duplicate_non_stream_terminal_observation_is_token_idempotent() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = current_second().saturating_mul(1_000);
    let event = UsageEvent {
        event_type: UsageEventType::Completed,
        request_id: "dashboard-realtime-terminal-duplicate".to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            input_tokens: Some(80),
            output_tokens: Some(20),
            total_tokens: Some(100),
            ..UsageEventData::default()
        },
    };

    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("first terminal observation should succeed");
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("duplicate terminal observation should be harmless");

    let snapshot = snapshot_at(&runtime, current_second())
        .await
        .expect("duplicate terminal snapshot should succeed");
    assert_eq!(snapshot.tpm, 100);
}

#[tokio::test]
async fn completed_terminal_observation_consumes_admission_marker_once() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let request_id = "dashboard-realtime-completed-settlement";
    let timestamp_ms = current_second().saturating_mul(1_000);
    reserve_admission(&runtime, request_id)
        .await
        .expect("admission reservation should succeed")
        .expect("request should reserve a marker")
        .handoff();

    let event = UsageEvent {
        event_type: UsageEventType::Completed,
        request_id: request_id.to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            total_tokens: Some(9),
            ..UsageEventData::default()
        },
    };
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("completed observation should settle");
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("duplicate completed observation should be harmless");

    let snapshot = snapshot_at(&runtime, event.timestamp_ms / 1_000)
        .await
        .expect("completed snapshot should succeed");
    assert_eq!(snapshot.rpm, 1);
    assert_eq!(snapshot.tpm, 9);
    assert!(runtime
        .kv_take(&admission_marker_key(request_id))
        .await
        .expect("admission marker lookup should succeed")
        .is_none());
}

#[tokio::test]
async fn terminal_retry_with_changed_usage_is_still_token_idempotent() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 42_000_000;
    let event = UsageEvent {
        event_type: UsageEventType::Completed,
        request_id: "dashboard-realtime-terminal-corrected".to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            total_tokens: Some(100),
            ..UsageEventData::default()
        },
    };
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("first terminal observation should succeed");

    // A replay can carry a corrected/enriched total while retaining the same
    // lifecycle identity.  It must not turn one request into two TPM deltas;
    // a retry after a failed bucket write is handled separately by clearing
    // the marker before returning the write error.
    let mut corrected = event.clone();
    corrected.timestamp_ms = timestamp_ms.saturating_add(1_000);
    corrected.data.total_tokens = Some(120);
    observe_usage_event_for_test(&runtime, &corrected)
        .await
        .expect("corrected terminal replay should be harmless");

    let snapshot = snapshot_at(&runtime, timestamp_ms / 1_000)
        .await
        .expect("corrected replay snapshot should succeed");
    assert_eq!(snapshot.tpm, 100);
}

#[tokio::test]
async fn zero_token_terminal_observation_does_not_block_later_enrichment() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 43_000_000;
    let mut event = UsageEvent {
        event_type: UsageEventType::Completed,
        request_id: "dashboard-realtime-terminal-enrichment".to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            ..UsageEventData::default()
        },
    };
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("zero-token terminal observation should succeed");
    event.data.total_tokens = Some(37);
    event.timestamp_ms = timestamp_ms.saturating_add(1_000);
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("enriched terminal observation should succeed");

    let snapshot = snapshot_at(&runtime, timestamp_ms / 1_000 + 1)
        .await
        .expect("enriched terminal snapshot should succeed");
    assert_eq!(snapshot.tpm, 37);
}

#[tokio::test]
async fn zero_token_terminal_replay_can_later_record_enriched_tokens() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 42_100_000;
    let mut event = UsageEvent {
        event_type: UsageEventType::Completed,
        request_id: "dashboard-realtime-terminal-late-usage".to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            ..UsageEventData::default()
        },
    };

    // Some terminal paths are emitted before provider usage enrichment and
    // therefore carry no tokens.  A zero-token observation must not consume
    // the dedupe marker, otherwise the later enriched replay would be lost.
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("zero-token terminal observation should succeed");
    event.data.total_tokens = Some(120);
    observe_usage_event_for_test(&runtime, &event)
        .await
        .expect("enriched terminal replay should succeed");

    let snapshot = snapshot_at(&runtime, timestamp_ms / 1_000)
        .await
        .expect("late usage snapshot should succeed");
    assert_eq!(snapshot.tpm, 120);
}

#[tokio::test]
async fn distinct_candidate_attempts_with_one_trace_keep_both_token_deltas() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 42_200_000;
    let event = |candidate_id: &str, tokens| UsageEvent {
        event_type: UsageEventType::Completed,
        request_id: "dashboard-realtime-multi-attempt".to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            candidate_id: Some(candidate_id.to_string()),
            total_tokens: Some(tokens),
            ..UsageEventData::default()
        },
    };

    // Fallback/replanned provider attempts may share a client trace/request
    // id but have distinct candidate identities.  Each attempt consumed
    // provider tokens and therefore contributes its own TPM delta.
    observe_usage_event_for_test(&runtime, &event("candidate-a", 10))
        .await
        .expect("first attempt observation should succeed");
    observe_usage_event_for_test(&runtime, &event("candidate-b", 20))
        .await
        .expect("second attempt observation should succeed");

    let snapshot = snapshot_at(&runtime, timestamp_ms / 1_000)
        .await
        .expect("multi-attempt snapshot should succeed");
    assert_eq!(snapshot.tpm, 30);
}

#[tokio::test]
async fn same_candidate_quota_retry_attempts_keep_each_attempts_tokens() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 42_250_000;
    let request_id = "dashboard-realtime-same-candidate-retry";
    let admission_id = "dashboard-realtime-same-candidate-admission";
    let candidate_id = "candidate-reused-by-quota-retry";
    let event = |event_type, attempt_id: &str, tokens| UsageEvent {
        event_type,
        request_id: request_id.to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            candidate_id: Some(candidate_id.to_string()),
            request_metadata: Some(serde_json::json!({
                "realtime_admission_id": admission_id,
                "provider_request_order_id": attempt_id,
                // Attempt A is an intermediate transparent-retry failure; it
                // must retain its token usage without rolling back the one
                // logical RPM admission.
                "realtime_admission_defer_failure": attempt_id == "attempt-a",
            })),
            total_tokens: Some(tokens),
            ..UsageEventData::default()
        },
    };

    reserve_admission_at(&runtime, admission_id, timestamp_ms)
        .await
        .expect("logical admission should reserve")
        .expect("logical admission should be new")
        .handoff();

    observe_usage_event_for_test(&runtime, &event(UsageEventType::Streaming, "attempt-a", 4))
        .await
        .expect("attempt A stream delta should succeed");
    observe_usage_event_for_test(&runtime, &event(UsageEventType::Failed, "attempt-a", 10))
        .await
        .expect("attempt A terminal total should succeed");

    observe_usage_event_for_test(&runtime, &event(UsageEventType::Streaming, "attempt-b", 5))
        .await
        .expect("attempt B stream delta should succeed");
    observe_usage_event_for_test(&runtime, &event(UsageEventType::Completed, "attempt-b", 20))
        .await
        .expect("attempt B terminal total should succeed");
    // A lifecycle replay for attempt B is idempotent within that attempt.
    observe_usage_event_for_test(&runtime, &event(UsageEventType::Completed, "attempt-b", 20))
        .await
        .expect("attempt B terminal replay should succeed");

    let snapshot = snapshot_at_millis(&runtime, timestamp_ms)
        .await
        .expect("quota retry snapshot should succeed");
    assert_eq!(snapshot.rpm, 1);
    assert_eq!(snapshot.tpm, 30);
}

#[tokio::test]
async fn forwarded_admission_identity_prevents_terminal_token_collision() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 42_300_000;
    let event = |admission_id: &str, tokens| UsageEvent {
        event_type: UsageEventType::Completed,
        // A client/owner hop may preserve or reuse the same public trace ID;
        // the forwarded admission marker is the request-level identity.
        request_id: "dashboard-realtime-reused-trace".to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            request_metadata: Some(serde_json::json!({
                "realtime_admission_id": admission_id,
            })),
            total_tokens: Some(tokens),
            ..UsageEventData::default()
        },
    };

    observe_usage_event_for_test(&runtime, &event("admission-a", 10))
        .await
        .expect("first forwarded terminal observation should succeed");
    observe_usage_event_for_test(&runtime, &event("admission-b", 20))
        .await
        .expect("second forwarded terminal observation should succeed");

    let snapshot = snapshot_at_millis(&runtime, timestamp_ms)
        .await
        .expect("forwarded identity snapshot should succeed");
    assert_eq!(snapshot.tpm, 30);
}

#[tokio::test]
async fn terminal_phase_replay_does_not_double_count_tokens() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let timestamp_ms = 42_400_000;
    let event = |event_type, tokens| UsageEvent {
        event_type,
        request_id: "dashboard-realtime-phase-replay".to_string(),
        timestamp_ms,
        data: UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            request_metadata: Some(serde_json::json!({
                "realtime_admission_id": "phase-replay-admission",
            })),
            total_tokens: Some(tokens),
            ..UsageEventData::default()
        },
    };

    observe_usage_event_for_test(&runtime, &event(UsageEventType::Failed, 42))
        .await
        .expect("failed terminal observation should succeed");
    observe_usage_event_for_test(&runtime, &event(UsageEventType::Completed, 84))
        .await
        .expect("completed replay observation should succeed");

    let snapshot = snapshot_at_millis(&runtime, timestamp_ms)
        .await
        .expect("phase replay snapshot should succeed");
    assert_eq!(snapshot.tpm, 42);
}
