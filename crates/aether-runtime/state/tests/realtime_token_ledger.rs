//! Contract tests for stream/terminal token reconciliation.

use std::sync::Arc;
use std::time::Duration;

use aether_runtime_state::{DataLayerError, MemoryRuntimeStateConfig, RuntimeState};

#[tokio::test]
async fn stream_deltas_are_subtracted_from_the_first_terminal_claim() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);

    assert_eq!(
        runtime
            .realtime_token_stream_add("request-1", 40, ttl)
            .await
            .expect("stream delta should succeed"),
        40
    );
    assert_eq!(
        runtime
            .realtime_token_stream_add("request-1", 20, ttl)
            .await
            .expect("second stream delta should succeed"),
        20
    );

    // The terminal usage is cumulative, so only the unseen remainder is
    // emitted to the realtime event stream.
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-1", 100, ttl)
            .await
            .expect("terminal claim should succeed"),
        40
    );

    // Claims are one-shot and late stream frames are fenced after a claim.
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-1", 120, ttl)
            .await
            .expect("duplicate terminal claim should be a no-op"),
        0
    );
    assert_eq!(
        runtime
            .realtime_token_stream_add("request-1", 10, ttl)
            .await
            .expect("late stream delta should be accepted as a no-op"),
        0
    );
}

#[tokio::test]
async fn stream_add_once_deduplicates_event_ids_atomically() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);

    assert_eq!(
        runtime
            .realtime_token_stream_add_once("request-once", "frame-1", 40, ttl)
            .await
            .expect("first stream frame should succeed"),
        40
    );
    assert_eq!(
        runtime
            .realtime_token_stream_add_once("request-once", "frame-1", 40, ttl)
            .await
            .expect("replayed stream frame should be a no-op"),
        0
    );
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-once", 100, ttl)
            .await
            .expect("terminal claim should succeed"),
        60
    );
}

#[tokio::test]
async fn concurrent_duplicate_stream_frames_are_counted_once() {
    let runtime = Arc::new(RuntimeState::memory(MemoryRuntimeStateConfig::default()));
    let ttl = Duration::from_secs(60);
    const WRITERS: usize = 128;

    let mut tasks = Vec::with_capacity(WRITERS);
    for _ in 0..WRITERS {
        let runtime = Arc::clone(&runtime);
        tasks.push(tokio::spawn(async move {
            runtime
                .realtime_token_stream_add_once("request-duplicate-race", "frame-1", 7, ttl)
                .await
                .expect("concurrent stream frame should succeed")
        }));
    }
    let mut accepted = 0u64;
    for task in tasks {
        accepted = accepted.saturating_add(task.await.expect("writer should finish"));
    }
    assert_eq!(accepted, 7);
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-duplicate-race", 10, ttl)
            .await
            .expect("terminal claim should succeed"),
        3
    );
}

#[tokio::test]
async fn concurrent_terminal_claims_are_one_shot() {
    let runtime = Arc::new(RuntimeState::memory(MemoryRuntimeStateConfig::default()));
    let ttl = Duration::from_secs(60);
    let mut tasks = Vec::new();
    for _ in 0..32 {
        let runtime = Arc::clone(&runtime);
        tasks.push(tokio::spawn(async move {
            runtime
                .realtime_token_terminal_claim("request-terminal-race", 100, ttl)
                .await
                .expect("terminal claim should succeed")
        }));
    }
    let mut claimed = 0u64;
    for task in tasks {
        claimed = claimed.saturating_add(task.await.expect("claimer should finish"));
    }
    assert_eq!(claimed, 100);
}

#[tokio::test]
async fn zero_terminal_claim_keeps_ledger_open_for_enrichment() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);

    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-enriched", 0, ttl)
            .await
            .expect("zero terminal snapshot should be a no-op"),
        0
    );
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-enriched", 25, ttl)
            .await
            .expect("enriched terminal snapshot should claim"),
        25
    );
}

#[tokio::test]
async fn terminal_total_below_streamed_total_has_no_negative_remainder() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);

    runtime
        .realtime_token_stream_add("request-under", 100, ttl)
        .await
        .expect("stream delta should succeed");
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-under", 80, ttl)
            .await
            .expect("terminal claim should succeed"),
        0
    );
}

#[tokio::test]
async fn terminal_prepare_is_retryable_until_event_commit() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);
    let identity = "request-prepare-retry";
    let event_id = "terminal-event-1";

    assert_eq!(
        runtime
            .realtime_token_stream_add(identity, 40, ttl)
            .await
            .expect("stream delta should succeed"),
        40
    );
    assert_eq!(
        runtime
            .realtime_token_terminal_prepare(identity, event_id, 100, ttl)
            .await
            .expect("terminal prepare should succeed"),
        60
    );
    // A retry after the exact event write failed returns the same pending
    // remainder; late stream frames remain fenced while it is pending.
    assert_eq!(
        runtime
            .realtime_token_stream_add(identity, 10, ttl)
            .await
            .expect("late stream delta should be fenced"),
        0
    );
    assert_eq!(
        runtime
            .realtime_token_terminal_prepare(identity, event_id, 100, ttl)
            .await
            .expect("terminal prepare retry should succeed"),
        60
    );
    assert!(runtime
        .realtime_token_terminal_commit(identity, event_id)
        .await
        .expect("terminal commit should succeed"));
    assert!(runtime
        .realtime_token_terminal_commit(identity, event_id)
        .await
        .expect("duplicate terminal commit should remain idempotent"));
    assert_eq!(
        runtime
            .realtime_token_terminal_prepare(identity, event_id, 120, ttl)
            .await
            .expect("committed terminal retry should be a no-op"),
        0
    );
}

#[tokio::test]
async fn stream_adds_are_atomic_under_concurrency() {
    let runtime = Arc::new(RuntimeState::memory(MemoryRuntimeStateConfig::default()));
    let ttl = Duration::from_secs(60);
    const WRITERS: usize = 128;
    const DELTA: u64 = 7;

    let mut tasks = Vec::with_capacity(WRITERS);
    for _ in 0..WRITERS {
        let runtime = Arc::clone(&runtime);
        tasks.push(tokio::spawn(async move {
            runtime
                .realtime_token_stream_add("request-concurrent", DELTA, ttl)
                .await
                .expect("concurrent stream delta should succeed")
        }));
    }
    let mut accepted = 0u64;
    for task in tasks {
        accepted = accepted.saturating_add(task.await.expect("writer should finish"));
    }
    assert_eq!(accepted, WRITERS as u64 * DELTA);
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-concurrent", accepted + 1, ttl)
            .await
            .expect("terminal claim should succeed"),
        1
    );
}

#[tokio::test]
async fn ledger_entries_expire_and_inputs_are_validated() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());

    assert!(matches!(
        runtime
            .realtime_token_stream_add(" ", 1, Duration::from_secs(1))
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime
            .realtime_token_stream_add_once("request", " ", 1, Duration::from_secs(1))
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime
            .realtime_token_terminal_claim("request", 1, Duration::ZERO)
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));

    let short_ttl = Duration::from_millis(20);
    assert_eq!(
        runtime
            .realtime_token_stream_add("request-expiring", 5, short_ttl)
            .await
            .expect("short-lived stream delta should succeed"),
        5
    );
    tokio::time::sleep(Duration::from_millis(40)).await;

    // Expiration resets the reconciliation state; a terminal event arriving
    // after retention has elapsed is treated as a fresh cumulative snapshot.
    assert_eq!(
        runtime
            .realtime_token_terminal_claim("request-expiring", 5, Duration::from_secs(1))
            .await
            .expect("claim after expiry should succeed"),
        5
    );
}
