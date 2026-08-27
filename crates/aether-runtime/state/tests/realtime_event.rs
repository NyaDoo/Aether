//! Contract tests for exact-timestamp realtime events.

use std::sync::Arc;
use std::time::Duration;

use aether_runtime_state::{DataLayerError, MemoryRuntimeStateConfig, RuntimeState};

#[tokio::test]
async fn memory_events_use_an_exact_open_closed_interval_and_are_idempotent() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);

    assert!(runtime
        .realtime_event_add("dashboard", "at-lower", 1_000, 1, 10, ttl)
        .await
        .expect("first event should be stored"));
    assert!(runtime
        .realtime_event_add("dashboard", "inside", 1_001, 2, 20, ttl)
        .await
        .expect("second event should be stored"));
    assert!(runtime
        .realtime_event_add("dashboard", "at-upper", 2_000, 4, 40, ttl)
        .await
        .expect("third event should be stored"));

    // `(start_ms, end_ms]`: the lower-bound event is excluded and the upper
    // bound is included.
    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 1_000, 2_000)
            .await
            .expect("sum should succeed"),
        aether_runtime_state::RealtimeBucket {
            requests: 6,
            tokens: 60,
        }
    );

    // A duplicate lifecycle notification cannot overwrite or add a second
    // copy of the event, even when its payload differs.
    assert!(!runtime
        .realtime_event_add("dashboard", "inside", 1_001, 99, 99, ttl)
        .await
        .expect("duplicate event should be accepted as a no-op"));
    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 1_000, 2_000)
            .await
            .expect("sum after duplicate should succeed")
            .tokens,
        60
    );
}

#[tokio::test]
async fn memory_event_remove_is_atomic_and_idempotent() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);
    runtime
        .realtime_event_add("dashboard", "request-1", 10_000, 1, 123, ttl)
        .await
        .expect("event should be stored");

    assert!(runtime
        .realtime_event_remove("dashboard", "request-1")
        .await
        .expect("first remove should succeed"));
    assert!(!runtime
        .realtime_event_remove("dashboard", "request-1")
        .await
        .expect("second remove should be a no-op"));
    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 9_999, 10_001)
            .await
            .expect("sum after remove should succeed"),
        aether_runtime_state::RealtimeBucket::default()
    );
}

#[tokio::test]
async fn memory_event_adds_are_atomic_under_concurrency() {
    let runtime = Arc::new(RuntimeState::memory(MemoryRuntimeStateConfig::default()));
    let ttl = Duration::from_secs(60);
    const EVENTS: usize = 128;

    let mut tasks = Vec::with_capacity(EVENTS);
    for index in 0..EVENTS {
        let runtime = Arc::clone(&runtime);
        tasks.push(tokio::spawn(async move {
            runtime
                .realtime_event_add(
                    "dashboard",
                    &format!("event-{index}"),
                    50_000 + index as u64,
                    1,
                    3,
                    ttl,
                )
                .await
                .expect("concurrent event should be stored")
        }));
    }
    let mut inserted = 0;
    for task in tasks {
        if task.await.expect("writer should finish") {
            inserted += 1;
        }
    }
    assert_eq!(inserted, EVENTS);
    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 49_999, 51_000)
            .await
            .expect("sum should succeed"),
        aether_runtime_state::RealtimeBucket {
            requests: EVENTS as i64,
            tokens: EVENTS as i64 * 3,
        }
    );
}

#[tokio::test]
async fn memory_event_ttl_and_invalid_ranges_are_handled() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    assert!(matches!(
        runtime
            .realtime_event_add("dashboard", "event", 1, 1, 1, Duration::ZERO)
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime
            .realtime_event_add(" ", "event", 1, 1, 1, Duration::from_secs(1))
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime
            .realtime_event_add("dashboard", " ", 1, 1, 1, Duration::from_secs(1))
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime
            .realtime_event_add(
                "dashboard",
                "out-of-range",
                u64::MAX,
                1,
                1,
                Duration::from_secs(1),
            )
            .await,
        Err(DataLayerError::InvalidInput(_))
    ));

    runtime
        .realtime_event_add(
            "dashboard",
            "expires",
            1_000,
            1,
            5,
            Duration::from_millis(20),
        )
        .await
        .expect("short-lived event should be stored");
    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 0, 2_000)
            .await
            .expect("sum before expiry should succeed")
            .tokens,
        5
    );
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 0, 2_000)
            .await
            .expect("sum after expiry should succeed"),
        aether_runtime_state::RealtimeBucket::default()
    );

    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 10, 10)
            .await
            .expect("empty range should succeed"),
        aether_runtime_state::RealtimeBucket::default()
    );
}

#[tokio::test]
async fn memory_event_signed_compensation_is_clamped_at_zero() {
    let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
    let ttl = Duration::from_secs(60);
    runtime
        .realtime_event_add("dashboard", "positive", 2_000, 1, 10, ttl)
        .await
        .expect("positive event");
    runtime
        .realtime_event_add("dashboard", "negative", 2_001, -4, -50, ttl)
        .await
        .expect("signed compensation event");
    assert_eq!(
        runtime
            .realtime_events_sum("dashboard", 1_999, 2_010)
            .await
            .expect("sum should succeed"),
        aether_runtime_state::RealtimeBucket::default()
    );
}
