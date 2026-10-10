//! `spec/ruby_llm/mcp/listener_spec.rb:196` on its own: it counts `tracing` WARN events, whose
//! per-callsite interest cache races with other tests registering callsites concurrently, so it
//! runs alone in this binary (like `spec_mcp_http_warnings.rs`).

mod listener_support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use listener_support::*;

// spec: mcp/listener_spec.rb:196 when a subscription ends > with a server that ends each subscription > subscribes again, waiting longer each time up to a minute
#[test]
fn subscribes_again_waiting_longer_each_time_up_to_a_minute() {
    // One thread, so the listener task's warnings reach this thread's collector.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let warnings = Arc::new(Mutex::new(0));
    let _guard =
        tracing::dispatcher::set_default(&tracing::Dispatch::new(Warnings(warnings.clone())));
    let delays = Arc::new(Mutex::new(Vec::new()));
    runtime.block_on(async {
        let client = Scripted::new(script(|attempt, notify| async move {
            notify.notify(acknowledgment(tools()));
            if attempt == 10 {
                forever().await
            } else {
                Ok(())
            }
        }));
        let listener = listener(&client, Duration::from_secs(2), quiet())
            .with_sleep(recorded_sleep(delays.clone()));
        listener.start(tools()).await.unwrap();
        eventually(|| client.attempts().len() == 10).await;
        listener.stop().await;
    });

    let limits = [1, 2, 4, 8, 16, 32, 60, 60, 60];
    let delays = delays.lock().unwrap().clone();
    assert_eq!(delays.len(), limits.len());
    for (delay, limit) in delays.iter().zip(limits) {
        let (low, high) = (limit as f64 / 2.0, limit as f64);
        assert!(
            (low..=high).contains(&delay.as_secs_f64()),
            "{delay:?} outside {low}..={high}"
        );
    }
    assert_eq!(*warnings.lock().unwrap(), 9);
}
