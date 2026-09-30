//! `batch_helpers_spec.rb` `.find` with a configured `batch_store` (`config.batch_store`), and the
//! `persist`/`sync` calls `batch.rb` makes on it.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::batch::{BatchAttributes, BatchStore};
use rust_llm::{Batch, Chat, Config, Result};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A store that hands back a batch for `msgbatch_123` and records every call it receives.
#[derive(Default)]
struct RecordingStore(Mutex<Vec<String>>);

#[async_trait]
impl BatchStore for RecordingStore {
    async fn fetch(
        &self,
        id: &str,
        provider: Option<&str>,
        config: Arc<Config>,
    ) -> Result<Option<Batch>> {
        self.0
            .lock()
            .unwrap()
            .push(format!("fetch {id} {provider:?}"));
        if id != "msgbatch_123" {
            return Ok(None);
        }
        let attributes = BatchAttributes {
            id: id.into(),
            raw_status: Some("in_progress".into()),
            ..Default::default()
        };
        Batch::from_attributes(config, "anthropic", attributes).map(Some)
    }

    async fn persist(&self, batch: &Batch) -> Result<()> {
        self.0
            .lock()
            .unwrap()
            .push(format!("persist {}", batch.id()));
        Ok(())
    }

    async fn sync(&self, batch: &Batch) -> Result<()> {
        self.0.lock().unwrap().push(format!(
            "sync {} {}",
            batch.id(),
            batch.raw_status().unwrap_or_default()
        ));
        Ok(())
    }
}

fn config(server: &MockServer, store: Arc<RecordingStore>) -> Arc<Config> {
    let mut c = Config::default();
    c.set("anthropic_api_base", server.uri());
    c.set("anthropic_api_key", "test");
    c.max_retries = 0;
    c.batch_store = Some(store);
    Arc::new(c)
}

// spec: batch_helpers_spec.rb:213 .find > returns a batch the store already has
/// Ruby asserts the very object the store returned; the port asserts it is that batch and that the
/// provider was never asked (the store answers without a provider, as in the spec).
#[tokio::test]
async fn find_returns_a_batch_the_store_already_has() {
    let server = MockServer::start().await;
    let store = Arc::new(RecordingStore::default());

    let batch = Batch::find_with_config(config(&server, store.clone()), "msgbatch_123", None)
        .await
        .unwrap();

    assert_eq!(batch.id(), "msgbatch_123");
    assert_eq!(batch.raw_status(), Some("in_progress"));
    assert_eq!(*store.0.lock().unwrap(), vec!["fetch msgbatch_123 None"]);
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
}

/// `submit_chats` persists the new batch; `refresh` syncs the state it fetched (`persist_state`).
#[tokio::test]
async fn submitting_persists_and_refreshing_syncs_the_store() {
    let server = MockServer::start().await;
    let batch = |status: &str| {
        json!({ "id": "msgbatch_9", "type": "message_batch", "processing_status": status,
                "request_counts": { "processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0 } })
    };
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch("in_progress")))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_9"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch("ended")))
        .mount(&server)
        .await;
    let store = Arc::new(RecordingStore::default());
    let mut chat = Chat::with_config(
        config(&server, store.clone()),
        Some("claude-haiku-4-5"),
        Some("anthropic"),
        false,
    )
    .unwrap();
    chat.ask_later("Hi").unwrap();

    let mut batch = rust_llm::batch(chat).await.unwrap();
    batch.refresh().await.unwrap();

    assert_eq!(
        *store.0.lock().unwrap(),
        vec!["persist msgbatch_9", "sync msgbatch_9 ended"]
    );
}
