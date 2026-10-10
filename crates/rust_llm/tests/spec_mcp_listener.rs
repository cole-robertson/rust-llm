//! `spec/ruby_llm/mcp/listener_spec.rb`: the listener over a scripted client whose subscriptions
//! each play out a script with their number, counted from 1.

mod listener_support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use listener_support::*;
use rust_llm::Error;
use rust_llm::mcp::{Listener, McpError};
use serde_json::json;
use tokio::sync::{Notify, mpsc};

// spec: mcp/listener_spec.rb:51 returns once the server acknowledges, with what it agreed to send
#[tokio::test]
async fn returns_once_the_server_acknowledges() {
    let client = Scripted::new(acknowledging());
    let listener = listener(&client, Duration::from_secs(2), quiet());
    assert_eq!(listener.start(tools()).await.unwrap(), Some(tools()));
    assert_eq!(client.attempts(), [tools()]);
    listener.stop().await;
}

// spec: mcp/listener_spec.rb:56 keeps a subscription for the same changes and replaces it for others
#[tokio::test]
async fn keeps_a_subscription_for_the_same_changes_and_replaces_it_for_others() {
    let client = Scripted::new(acknowledging());
    let listener = listener(&client, Duration::from_secs(2), quiet());
    let files = json!({ "toolsListChanged": true, "resourceSubscriptions": ["file:///notes.md"] });
    for _ in 0..2 {
        listener.start(tools()).await.unwrap();
    }
    listener.start(files.clone()).await.unwrap();
    assert_eq!(client.attempts(), [tools(), files]);
    assert_eq!(client.cancelled(), [1]);
    listener.stop().await;
}

// spec: mcp/listener_spec.rb:66 does nothing when there is nothing to listen for
#[tokio::test]
async fn does_nothing_when_there_is_nothing_to_listen_for() {
    let client = Scripted::new(acknowledging());
    let listener = listener(&client, Duration::from_secs(2), quiet());
    assert_eq!(listener.start(json!({})).await.unwrap(), None);
    assert!(client.attempts().is_empty());
}

// spec: mcp/listener_spec.rb:90 when a callback is running > lets it finish before stopping
#[tokio::test]
async fn lets_a_running_callback_finish_before_stopping() {
    let client = Scripted::new(script(|_, notify| async move {
        notify.notify(acknowledgment(tools()));
        notify.notify(json!({ "method": "notifications/tools/list_changed" }));
        forever().await
    }));
    let steps = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(Notify::new());
    let (recorder, signal) = (steps.clone(), started.clone());
    let callback = rust_llm::mcp::listener_callback(move |notification| {
        let (recorder, signal) = (recorder.clone(), signal.clone());
        async move {
            if notification["method"] != "notifications/tools/list_changed" {
                return;
            }
            recorder.lock().unwrap().push("started");
            signal.notify_one();
            tokio::time::sleep(Duration::from_millis(200)).await;
            recorder.lock().unwrap().push("finished");
        }
    });
    let listener = listener(&client, Duration::from_secs(2), callback);
    let waiting = started.notified();
    listener.start(tools()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .unwrap();

    listener.stop().await;

    assert_eq!(*steps.lock().unwrap(), ["started", "finished"]);
    assert_eq!(client.cancelled(), [1]);
}

// spec: mcp/listener_spec.rb:106 when started while a chat follows tool progress > reports nothing to that chat
#[tokio::test]
async fn reports_nothing_to_a_chat_that_follows_tool_progress() {
    let client = Scripted::new(script(|_, notify| async move {
        notify.notify(acknowledgment(tools()));
        notify.notify(json!({ "method": "notifications/tools/list_changed" }));
        forever().await
    }));
    let (sender, mut listeners) = mpsc::unbounded_channel();
    let callback = rust_llm::mcp::listener_callback(move |_| {
        let sender = sender.clone();
        async move {
            let _ = sender.send(rust_llm::progress::listener().is_some());
        }
    });
    let listener = listener(&client, Duration::from_secs(2), callback);
    let chat: rust_llm::progress::Listener = Arc::new(|_| {});
    rust_llm::progress::listen(Some(chat), listener.start(tools()))
        .await
        .unwrap();
    assert_eq!(listeners.recv().await, Some(false));
    listener.stop().await;
}

// spec: mcp/listener_spec.rb:140 with a server that refuses the subscription > raises why
#[tokio::test]
async fn raises_why_a_server_refuses_the_subscription() {
    let client = Scripted::new(script(|_, _| async {
        Err(McpError::new("Unauthorized").into())
    }));
    let listener = listener(&client, Duration::from_secs(2), quiet());
    let Err(Error::Mcp(e)) = listener.start(tools()).await else {
        panic!("expected an MCP error");
    };
    assert_eq!(e.message, "Unauthorized");
    assert_eq!(client.attempts().len(), 1);
}

// spec: mcp/listener_spec.rb:150 with a server that never acknowledges > raises and cancels the subscription
#[tokio::test]
async fn raises_and_cancels_when_the_server_never_acknowledges() {
    let client = Scripted::new(script(|_, _| forever()));
    let listener = listener(&client, Duration::from_millis(200), quiet());
    let Err(Error::Mcp(e)) = listener.start(tools()).await else {
        panic!("expected an MCP error");
    };
    assert!(e.message.contains("did not acknowledge"), "{}", e.message);
    assert_eq!(client.cancelled(), [1]);
}

// spec: mcp/listener_spec.rb:168 when a subscription resumes > reports what it listens to, since changes in between are lost
#[tokio::test]
async fn reports_what_it_listens_to_when_a_subscription_resumes() {
    let client = Scripted::new(script(|attempt, notify| async move {
        notify.notify(acknowledgment(tools()));
        if attempt == 2 {
            forever().await
        } else {
            Ok(())
        }
    }));
    let (sender, mut resumes) = mpsc::unbounded_channel();
    let resumed = rust_llm::mcp::listener_callback(move |listened| {
        let sender = sender.clone();
        async move {
            let _ = sender.send(listened);
        }
    });
    let listener = Listener::new(
        client.clone(),
        "files",
        Duration::from_secs(2),
        Some(resumed),
        quiet(),
    )
    .with_sleep(recorded_sleep(Arc::default()));
    listener.start(tools()).await.unwrap();
    let first = tokio::time::timeout(Duration::from_secs(5), resumes.recv())
        .await
        .unwrap();
    assert_eq!(first, Some(tools()));
    assert_eq!(client.attempts().len(), 2);
    assert!(resumes.try_recv().is_err());
    listener.stop().await;
}

// spec: mcp/listener_spec.rb:209 when a subscription ends > with a server that agrees to send nothing > stops
#[tokio::test]
async fn stops_when_the_server_agrees_to_send_nothing() {
    let client = Scripted::new(script(|_, notify| async move {
        notify.notify(acknowledgment(json!({})));
        Ok(())
    }));
    let listener = listener(&client, Duration::from_secs(2), quiet())
        .with_sleep(recorded_sleep(Arc::default()));
    assert_eq!(listener.start(tools()).await.unwrap(), Some(json!({})));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(client.attempts().len(), 1);
}

// spec: mcp/listener_spec.rb:225 when a subscription ends > with a server that does not know the method > stops
#[tokio::test]
async fn stops_when_the_server_does_not_know_the_method() {
    let client = Scripted::new(script(|_, notify| async move {
        notify.notify(acknowledgment(tools()));
        Err(McpError {
            message: "Method not found".into(),
            code: Some(-32_601),
            ..Default::default()
        }
        .into())
    }));
    let listener = listener(&client, Duration::from_secs(2), quiet())
        .with_sleep(recorded_sleep(Arc::default()));
    listener.start(tools()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(client.attempts().len(), 1);
}
