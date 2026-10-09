//! Port of `spec/ruby_llm/active_record/acts_as_search_suggestions_spec.rb`: Google's search
//! suggestions are shown on a grounded answer and never written to a row. Replays RubyLLM's
//! recorded cassettes with their request bodies compared.

#[path = "../../rust_llm/tests/support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};

use rust_llm::ProviderTool;
use rust_llm::providers::ProtocolName;
use rust_llm_loco::{ChatRecord, migrations};
use sea_orm::{ConnectionTrait, Database, DatabaseConnection, DbBackend, Statement};
use sea_orm_migration::SchemaManager;
use serde_json::{Value, json};
use support::Cassette;

const QUESTION: &str =
    "Search the web: what is the latest stable Ruby version? Answer in one sentence.";
/// `suggestion_markers`.
const MARKERS: [&str; 4] = [
    "gradient-container",
    "app-vertex-grounding",
    "searchEntryPoint",
    "search_suggestions",
];

async fn db() -> DatabaseConnection {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

/// `stored_text(chat)`: every column of the chat's rows (chat, messages, usages, tool calls),
/// joined.
async fn stored_text(db: &DatabaseConnection, chat_id: i32) -> String {
    let mut text = String::new();
    for sql in [
        format!("SELECT * FROM chats WHERE id = {chat_id}"),
        format!("SELECT * FROM messages WHERE chat_id = {chat_id}"),
        format!("SELECT * FROM rust_llm_usages WHERE chat_id = {chat_id}"),
        format!(
            "SELECT * FROM rust_llm_tool_calls WHERE message_type = 'Message' AND message_id IN (SELECT id FROM messages WHERE chat_id = {chat_id})"
        ),
    ] {
        for row in db
            .query_all_raw(Statement::from_string(DbBackend::Sqlite, sql))
            .await
            .unwrap()
        {
            for column in row.column_names() {
                if let Ok(Some(v)) = row.try_get::<Option<String>>("", &column) {
                    text.push_str(&v);
                    text.push(' ');
                }
            }
        }
    }
    text
}

/// A RubyLLM cassette from `rust_llm`'s converted set, replayed with its request bodies compared.
/// (`Cassette::start` reads from the calling crate's `tests/cassettes`, so the recording is
/// loaded here and handed to `start_edited`.)
async fn replay(name: &str) -> Cassette {
    let path = format!(
        "{}/../rust_llm/tests/cassettes/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let recorded: Vec<support::Interaction> =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Cassette::serve(recorded).await
}

fn suggestions(message: &rust_llm::Message) -> String {
    message
        .server_tool_calls
        .iter()
        .filter_map(|c| c.search_suggestions.clone())
        .collect()
}

// spec: active_record/acts_as_search_suggestions_spec.rb:18 shows Gemini search suggestions on the live answer without storing them
#[tokio::test]
async fn shows_gemini_search_suggestions_on_the_live_answer_without_storing_them() {
    let db = db().await;
    let cassette = replay(
        "activerecord_actsas_shows_gemini_search_suggestions_on_the_live_answer_without_storing_them",
    )
    .await;
    let config = support::config_for(&cassette, "gemini");
    let record = ChatRecord::create_with(&db, "gemini-3.5-flash", Some("gemini"), true)
        .await
        .unwrap();
    let chat = record.to_llm_with(&db, config).await.unwrap();
    let mut chat = chat.with_provider_tools([ProviderTool::from("web_search")]);

    let response = record.ask(&db, &mut chat, QUESTION).await.unwrap();

    let shown = suggestions(&response);
    assert!(
        shown.contains("gradient-container") && shown.contains("app-vertex-grounding"),
        "{shown}"
    );
    let stored = stored_text(&db, record.id()).await;
    for marker in MARKERS {
        assert!(!stored.contains(marker), "{marker} stored");
    }
    let last = record.messages(&db).await.unwrap().pop().unwrap();
    let reloaded = rust_llm_loco::message_to_llm(&db, &last).await.unwrap();
    assert!(
        reloaded
            .server_tool_calls
            .iter()
            .all(|c| c.search_suggestions.is_none())
    );
    cassette.assert_all_matched().await;
}

// spec: active_record/acts_as_search_suggestions_spec.rb:28 replays a searched Gemini interaction from its record without the suggestions
#[tokio::test]
async fn replays_a_searched_gemini_interaction_from_its_record_without_the_suggestions() {
    let db = db().await;
    let cassette = replay(
        "activerecord_actsas_replays_a_searched_gemini_interaction_from_its_record_without_the_suggestions",
    )
    .await;
    let config = support::config_for(&cassette, "gemini");
    let mut record = ChatRecord::create_with(&db, "gemini-3.8-flash", Some("gemini"), true)
        .await
        .unwrap();
    record.protocol = Some(ProtocolName::Interactions);
    let chat = record.to_llm_with(&db, config.clone()).await.unwrap();
    let mut chat = chat.with_provider_tools([ProviderTool::from("web_search")]);

    let response = record.ask(&db, &mut chat, QUESTION).await.unwrap();

    assert!(suggestions(&response).contains("gradient-container"));
    assert_eq!(
        response.tokens().server_tool_use.map(Value::Object),
        Some(json!({ "web_search_requests": 2 }))
    );
    let stored = stored_text(&db, record.id()).await;
    for marker in MARKERS {
        assert!(!stored.contains(marker), "{marker} stored");
    }

    let mut restored = ChatRecord::find(&db, record.id()).await.unwrap();
    restored.assume_model_exists = true;
    restored.protocol = Some(ProtocolName::Interactions);
    let payloads: Arc<Mutex<Vec<Value>>> = Arc::default();
    let sink = payloads.clone();
    let chat = restored.to_llm_with(&db, config).await.unwrap();
    let mut followup_chat = chat
        .with_provider_tools([ProviderTool::from("web_search")])
        .before_request(move |payload| sink.lock().unwrap().push(payload.clone()));
    let followup = restored
        .ask(
            &db,
            &mut followup_chat,
            "In which year was that version released? Answer with the year only.",
        )
        .await
        .unwrap();

    // Ruby's `payloads.first` is the request sent; the port also runs its hooks on a trial
    // render that decides which attachments to read, so the sent request is the last payload.
    let first = payloads.lock().unwrap().last().cloned().unwrap();
    let replayed = first["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["type"] == "google_search_result")
        .cloned()
        .expect("the search result is replayed");
    assert!(replayed.get("call_id").is_some() && replayed.get("signature").is_some());
    assert_eq!(replayed["result"], json!([{}]));
    assert!(followup.content().contains("20"), "{}", followup.content());
    cassette.assert_all_matched().await;
}
