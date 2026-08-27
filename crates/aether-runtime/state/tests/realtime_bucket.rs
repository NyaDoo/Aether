//! Contract tests for the runtime primitive used by the dashboard realtime
//! counters.
//!
//! These tests intentionally exercise the public `RuntimeState` API rather
//! than the in-memory implementation details.  That keeps the atomicity and
//! compensation guarantees identical for callers that use a shared Redis
//! runtime state and for the process-local fallback.

use std::sync::Arc;
use std::time::Duration;

use aether_runtime_state::{DataLayerError, MemoryRuntimeStateConfig, RuntimeState};

#[tokio::test]
async fn memory_realtime_bucket_applies_concurrent_deltas_atomically() {
    let runtime = Arc::new(RuntimeState::memory(MemoryRuntimeStateConfig::default()));
    let key = "dashboard:realtime:atomic";
    const WRITERS: usize = 128;
    const TOKENS_PER_WRITE: i64 = 17;

    let mut tasks = Vec::with_capacity(WRITERS);
    for _ in 0..WRITERS {
        let runtime = Arc::clone(&runtime);
        tasks.push(tokio::spawn(async move {
            runtime
                .realtime_bucket_add(key, 1, TOKENS_PER_WRITE, Duration::from_secs(60))
                .await
                .expect("realtime delta should be accepted");
        }));
    }

    for task in tasks {
        task.await.expect("realtime writer should finish");
    }

    let snapshot = runtime
        .realtime_bucket_read(key)
        .await
        .expect("realtime read should succeed")
        .expect("bucket should exist");
    assert_eq!(snapshot.requests, WRITERS as i64);
    assert_eq!(snapshot.tokens, WRITERS as i64 * TOKENS_PER_WRITE);
}

#[tokio::test]
async fn memory_realtime_bucket_clamps_terminal_compensation_at_zero() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let key = "dashboard:realtime:compensation";

    assert_eq!(
        runtime
            .realtime_bucket_add(key, 3, 300, Duration::from_secs(60))
            .await
            .expect("admission delta should succeed"),
        aether_runtime_state::RealtimeBucket {
            requests: 3,
            tokens: 300,
        }
    );
    assert_eq!(
        runtime
            .realtime_bucket_add(key, -1, -100, Duration::from_secs(60))
            .await
            .expect("terminal compensation should succeed"),
        aether_runtime_state::RealtimeBucket {
            requests: 2,
            tokens: 200,
        }
    );

    // A failed/duplicate terminal must not make the exported counters
    // negative.  This is important when retries race with cancellation.
    let clamped = runtime
        .realtime_bucket_add(key, -10, -1_000, Duration::from_secs(60))
        .await
        .expect("over-compensation should be idempotent-safe");
    assert_eq!(clamped.requests, 0);
    assert_eq!(clamped.tokens, 0);

    let duplicate = runtime
        .realtime_bucket_add(key, -1, -1, Duration::from_secs(60))
        .await
        .expect("duplicate compensation should succeed");
    assert_eq!(duplicate.requests, 0);
    assert_eq!(duplicate.tokens, 0);
}

#[tokio::test]
async fn memory_realtime_bucket_expires_after_ttl() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let key = "dashboard:realtime:expiry";

    runtime
        .realtime_bucket_add(key, 1, 42, Duration::from_secs(1))
        .await
        .expect("realtime delta should succeed");
    assert!(runtime
        .realtime_bucket_read(key)
        .await
        .expect("realtime read should succeed")
        .is_some());

    tokio::time::sleep(Duration::from_millis(1_200)).await;
    assert!(runtime
        .realtime_bucket_read(key)
        .await
        .expect("realtime read after ttl should succeed")
        .is_none());
}

#[tokio::test]
async fn realtime_bucket_rejects_empty_keys_and_zero_ttl() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());

    assert!(matches!(
        runtime
            .realtime_bucket_add("  ", 1, 1, Duration::from_secs(60))
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime
            .realtime_bucket_add("dashboard:realtime:invalid", 1, 1, Duration::ZERO)
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime.realtime_bucket_read("").await,
        Err(DataLayerError::InvalidInput(_))
    ));
}
