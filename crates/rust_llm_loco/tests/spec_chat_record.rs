//! The remaining `spec/ruby_llm/active_record/*` and `agent_rails_spec.rb` examples, on SQLite:
//! record cancellation, context and default-model handling, copying and out-of-band completions,
//! orphaned tool-result cleanup, reloaded messages (model, provider, finish predicates, thinking,
//! attachments), compaction and remote MCP approvals through `Agent.find`, persisted batches
//! (`rust_llm_batches`), destroying a chat, and agents in Rails mode. `// spec:` lines tie each
//! test to its Ruby example.

use std::sync::Arc;
use std::time::Duration;

use rust_llm::agent::InstructionDeclaration;
use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::message::{Operation, RawResponse};
use rust_llm::providers::ProtocolName;
use rust_llm::{
    Agent, Attachment, Config, Cost, FinishReason, Message, Resolution, Role, ToolCall, UsageEntry,
    UsageStatus,
};
use rust_llm_loco::entities::{
    chats, messages, rust_llm_attachments, rust_llm_tool_calls, rust_llm_usages,
};
use rust_llm_loco::{ChatRecord, message_to_llm, migrations};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, Database, DatabaseConnection, EntityTrait,
    PaginatorTrait, QueryFilter,
};
use sea_orm_migration::SchemaManager;
use serde_json::{Map, Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

/// `model_for(:openai, :temperature)`.
const MODEL: &str = "gpt-4.1-nano";

/// `include_context 'with configured RubyLLM'`.
async fn db() -> DatabaseConnection {
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
        c.set("anthropic_api_key", "test");
        c.set("gemini_api_key", "test");
        c.set("xai_api_key", "test");
    });
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

fn config_for(server: &MockServer) -> Arc<Config> {
    let mut c = (*rust_llm::config()).clone();
    for (provider, path) in [
        ("openai", "/v1"),
        ("anthropic", ""),
        ("xai", "/v1"),
        ("gemini", "/v1beta"),
    ] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{path}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    c.max_retries = 0;
    Arc::new(c)
}

fn context_for(server: &MockServer) -> rust_llm::Context {
    rust_llm::Context::new((*config_for(server)).clone())
}

fn tool_call(name: &str) -> ToolCall {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    ToolCall::new(format!("call_{name}_{n}"), name, Map::new())
}

fn calling(calls: &[&ToolCall]) -> Message {
    let mut m = Message::assistant("");
    let map: IndexMap<ToolCall> = calls.iter().map(|c| (c.id.clone(), (*c).clone())).collect();
    m.tool_calls = Some(map);
    m
}

async fn contents(record: &ChatRecord, db: &DatabaseConnection) -> Vec<String> {
    record
        .messages(db)
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.content.unwrap_or_default())
        .collect()
}

async fn system_rows(record: &ChatRecord, db: &DatabaseConnection) -> Vec<(String, bool)> {
    record
        .messages(db)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.role == "system")
        .map(|m| (m.content.unwrap_or_default(), m.cache_until_here))
        .collect()
}

fn texts(chat: &rust_llm::Chat) -> Vec<String> {
    chat.messages()
        .iter()
        .map(|m| m.content().to_string())
        .collect()
}

async fn usage_row(
    db: &DatabaseConnection,
    chat_id: i32,
    message_id: i32,
    provider: &str,
    model: &str,
    status: &str,
) {
    rust_llm_usages::ActiveModel {
        chat_type: Set("Chat".into()),
        chat_id: Set(chat_id as i64),
        message_type: Set(Some("Message".into())),
        message_id: Set(Some(message_id as i64)),
        operation: Set("chat".into()),
        provider: Set(provider.into()),
        model: Set(model.into()),
        status: Set(status.into()),
        created_at: Set(chrono::Utc::now().into()),
        updated_at: Set(chrono::Utc::now().into()),
        ..Default::default()
    }
    .insert(db)
    .await
    .unwrap();
}

// ---- chat_methods_spec: cancellation ---------------------------------------------------------

// spec: active_record/chat_methods_spec.rb:34 #cancel forwards the request to a chat that is already built
#[tokio::test]
async fn cancel_forwards_the_request_to_a_chat_that_is_already_built() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let chat = record.to_llm(&db).await.unwrap();

    record.cancel_chat(&db, &chat).await.unwrap();

    assert!(chat.is_cancelled());
    assert!(record.is_cancelled(&db).await.unwrap());
}

// spec: active_record/chat_methods_spec.rb:54 cancellation requests from another process polls the row at most once per interval
#[tokio::test]
async fn polls_the_row_at_most_once_per_interval() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();

    assert!(
        !record
            .consume_persisted_cancellation_request(&db)
            .await
            .unwrap()
    );
    chats::Entity::update_many()
        .col_expr(
            chats::Column::Cancelled,
            sea_orm::sea_query::Expr::value(true),
        )
        .filter(chats::Column::Id.eq(record.id()))
        .exec(&db)
        .await
        .unwrap();
    assert!(
        !record
            .consume_persisted_cancellation_request(&db)
            .await
            .unwrap(),
        "the interval has not elapsed, so the row is not read"
    );
    assert!(record.is_cancelled(&db).await.unwrap(), "left on the row");
}

// ---- chat_methods_spec: model assignment and context -----------------------------------------

// spec: active_record/chat_methods_spec.rb:133 model assignment falls back to the configured default model
#[tokio::test]
async fn falls_back_to_the_configured_default_model() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let chat = record.to_llm(&db).await.unwrap();

    let chat = record.with_default_model(&db, chat).await.unwrap();

    let default = rust_llm::config().default_model.clone();
    assert_eq!(record.model(&db).await.unwrap().model_id, default);
    assert_eq!(chat.model().id, default);
}

// spec: active_record/chat_methods_spec.rb:191 #with_context rebinds the record and any built chat
#[tokio::test]
async fn with_context_rebinds_the_record_and_any_built_chat() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let chat = record.to_llm(&db).await.unwrap();
    let context = rust_llm::context(|c| c.request_timeout = Duration::from_secs(42));

    let chat = record
        .with_context(Some(context.clone()), Some(chat))
        .unwrap()
        .unwrap();

    assert!(Arc::ptr_eq(
        record.context().unwrap().config(),
        context.config()
    ));
    assert_eq!(chat.config().request_timeout, Duration::from_secs(42));
    assert_eq!(
        record.to_llm(&db).await.unwrap().config().request_timeout,
        Duration::from_secs(42)
    );
}

// spec: active_record/chat_methods_spec.rb:815 #with_context before the chat is built records the context before the chat is built
#[tokio::test]
async fn with_context_before_the_chat_is_built() {
    let db = db().await;
    let mut record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let context = rust_llm::context(|c| c.request_timeout = Duration::from_secs(7));

    assert!(record.with_context(Some(context), None).unwrap().is_none());

    assert_eq!(
        record.to_llm(&db).await.unwrap().config().request_timeout,
        Duration::from_secs(7)
    );
}

// ---- chat_methods_spec: add_message / add_completion -----------------------------------------

// spec: active_record/chat_methods_spec.rb:331 #add_message copies an existing message record into the conversation
#[tokio::test]
async fn add_message_copies_an_existing_message_record() {
    let db = db().await;
    let source = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut source_chat = source.to_llm(&db).await.unwrap();
    let original = source
        .add_message(&db, &mut source_chat, Message::user("Keep this context"))
        .await
        .unwrap();
    let destination = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = destination.to_llm(&db).await.unwrap();

    let copied = destination
        .add_message_record(&db, &mut chat, &original)
        .await
        .unwrap();

    assert_ne!(copied.id, original.id);
    assert_eq!(copied.content, original.content);
    let reloaded = messages::Entity::find_by_id(original.id)
        .one(&db)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reloaded.chat_id, source.id(), "the original stays put");
    assert_eq!(texts(&chat), ["Keep this context"]);
    assert_eq!(contents(&destination, &db).await, ["Keep this context"]);
}

// spec: active_record/chat_methods_spec.rb:495 delegation to the underlying chat persists completions added out of band
#[tokio::test]
async fn persists_completions_added_out_of_band() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();

    let returned = record
        .add_completion(&db, &mut chat, Message::assistant("Batch response"))
        .await
        .unwrap();

    assert_eq!(returned.content(), "Batch response");
    assert_eq!(
        contents(&record, &db).await.last().unwrap(),
        "Batch response"
    );
    let usages = record.usages(&db).await.unwrap();
    assert_eq!(usages.len(), 1, "its usage is recorded with it");
}

// ---- chat_methods_spec: orphaned tool result cleanup -----------------------------------------

// spec: active_record/chat_methods_spec.rb:560 orphaned tool result cleanup destroys the whole round when a tool call is still unanswered
#[tokio::test]
async fn cleanup_destroys_a_round_with_an_unanswered_call() {
    let db = db().await;
    let answered = tool_call("answered");
    let unanswered = tool_call("unanswered");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .add_message(&db, &mut chat, calling(&[&answered, &unanswered]))
        .await
        .unwrap();
    record
        .add_message(&db, &mut chat, Message::tool_result(&answered.id, "done"))
        .await
        .unwrap();

    record
        .cleanup_orphaned_tool_results(&db, &mut chat)
        .await
        .unwrap();

    assert!(record.messages(&db).await.unwrap().is_empty());
}

// spec: active_record/chat_methods_spec.rb:577 orphaned tool result cleanup keeps a completed round
#[tokio::test]
async fn cleanup_keeps_a_completed_round() {
    let db = db().await;
    let call = tool_call("lookup");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .add_message(&db, &mut chat, calling(&[&call]))
        .await
        .unwrap();
    record
        .add_message(&db, &mut chat, Message::tool_result(&call.id, "done"))
        .await
        .unwrap();

    record
        .cleanup_orphaned_tool_results(&db, &mut chat)
        .await
        .unwrap();

    assert_eq!(record.messages(&db).await.unwrap().len(), 2);
}

// spec: active_record/chat_methods_spec.rb:588 orphaned tool result cleanup leaves a plain conversation alone
#[tokio::test]
async fn cleanup_leaves_a_plain_conversation_alone() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .add_message(&db, &mut chat, Message::user("hello"))
        .await
        .unwrap();

    record
        .cleanup_orphaned_tool_results(&db, &mut chat)
        .await
        .unwrap();

    assert_eq!(record.messages(&db).await.unwrap().len(), 1);
}

// ---- acts_as_spec -----------------------------------------------------------------------------

// spec: active_record/acts_as_spec.rb:231 tool-call persistence removes internal tool and usage rows when their chat is destroyed
#[tokio::test]
async fn destroying_a_chat_removes_its_tool_call_and_usage_rows() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let call = ToolCall::new("call_destroy", "noop", Map::new());
    let message = record
        .add_message(&db, &mut chat, calling(&[&call]))
        .await
        .unwrap();
    usage_row(&db, record.id(), message.id, "openai", MODEL, "succeeded").await;
    let tool_calls = || rust_llm_tool_calls::Entity::find().count(&db);
    let usages = || rust_llm_usages::Entity::find().count(&db);
    let (calls_before, usages_before) = (tool_calls().await.unwrap(), usages().await.unwrap());

    record.destroy(&db).await.unwrap();

    assert_eq!(tool_calls().await.unwrap(), calls_before - 1);
    assert_eq!(usages().await.unwrap(), usages_before - 1);
    assert_eq!(messages::Entity::find().count(&db).await.unwrap(), 0);
}

// ---- message_methods_spec: Rails-backed message records --------------------------------------

// spec: active_record/message_methods_spec.rb:49 Rails-backed message records reads the model of the successful attempt after reloading the message
#[tokio::test]
async fn reads_the_model_of_the_successful_attempt_after_reloading() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let row = record
        .add_message(&db, &mut chat, Message::assistant("Answer"))
        .await
        .unwrap();
    usage_row(&db, record.id(), row.id, "openai", "gpt-4.1", "succeeded").await;
    usage_row(&db, record.id(), row.id, "openai", MODEL, "failed").await;

    let reloaded = message_to_llm(&db, &row).await.unwrap();
    assert_eq!(reloaded.model.as_deref(), Some("gpt-4.1"));
    assert_eq!(reloaded.model_info().unwrap().id, "gpt-4.1");

    let next = record
        .add_message(&db, &mut chat, Message::user("Continue"))
        .await
        .unwrap();
    assert_eq!(message_to_llm(&db, &next).await.unwrap().model, None);
}

// spec: active_record/message_methods_spec.rb:74 Rails-backed message records uses the successful attempt provider when reloading an overlapping model id
#[tokio::test]
async fn uses_the_successful_attempt_provider_for_an_overlapping_model_id() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    // `gpt-4.1-nano` is served by openai and azure in the registry; the attempt that succeeded
    // ran on azure, the failed one on openai.
    let row = record
        .add_message(&db, &mut chat, Message::assistant("Answer"))
        .await
        .unwrap();
    usage_row(&db, record.id(), row.id, "azure", MODEL, "succeeded").await;
    usage_row(&db, record.id(), row.id, "openai", MODEL, "failed").await;

    let info = message_to_llm(&db, &row)
        .await
        .unwrap()
        .model_info()
        .unwrap();
    let expected = rust_llm::models().find(MODEL, Some("azure")).unwrap();
    assert_eq!(info, expected);
    assert_eq!(info.provider, "azure");
}

// spec: active_record/message_methods_spec.rb:91 Rails-backed message records recognizes tool calls when the provider reports a normal stop
#[tokio::test]
async fn recognizes_tool_calls_when_the_provider_reports_a_normal_stop() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let mut message = calling(&[&tool_call("lookup")]);
    message.finish_reason = Some(FinishReason::Stop);
    let row = record.add_message(&db, &mut chat, message).await.unwrap();

    let reloaded = message_to_llm(&db, &row).await.unwrap();
    assert!(reloaded.is_tool_call_stop());
    assert!(!reloaded.is_stopped());
}

// ---- attachments -------------------------------------------------------------------------------

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/../rust_llm/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

// spec: active_record/acts_as_attachment_spec.rb:243 attachment types handles videos
#[tokio::test]
async fn a_reloaded_video_attachment_is_a_video() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let mut message = Message::user("Video test");
    message.attachments = vec![Attachment::from_bytes(
        fixture("ruby.mp4"),
        "test.mp4",
        Some("video/mp4"),
    )];
    let row = record.add_message(&db, &mut chat, message).await.unwrap();

    let reloaded = message_to_llm(&db, &row).await.unwrap();
    assert_eq!(
        reloaded.attachments[0].kind(),
        rust_llm::attachment::AttachmentType::Video
    );
}

// spec: active_record/attachment_helpers_spec.rb:86 #persist_content keeps the media resolution across a reload
#[tokio::test]
async fn keeps_the_media_resolution_across_a_reload() {
    let db = db().await;
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    let mut message = Message::user("see attached");
    message.attachments = vec![
        Attachment::from_bytes(b"png".to_vec(), "page.png", None).with_resolution(Resolution::High),
    ];
    let row = record.add_message(&db, &mut chat, message).await.unwrap();

    let reloaded = message_to_llm(&db, &row).await.unwrap();
    assert_eq!(reloaded.attachments[0].resolution, Some(Resolution::High));
    let stored = rust_llm_attachments::Entity::find()
        .one(&db)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.metadata, Some(json!({ "resolution": "high" })));
}

// ---- chat_methods_thinking_replay_spec ---------------------------------------------------------

// spec: active_record/chat_methods_thinking_replay_spec.rb:20 replays all Anthropic thinking blocks after reloading the message
#[tokio::test]
async fn replays_all_anthropic_thinking_blocks_after_reloading() {
    let db = db().await;
    let blocks = json!([
        { "type": "thinking", "thinking": "First.", "signature": "sig-one" },
        { "type": "redacted_thinking", "data": "encrypted" },
        { "type": "thinking", "thinking": "Second.", "signature": "sig-two" }
    ]);
    let body = json!({ "model": "claude-haiku-4-5", "content": blocks, "usage": {} });
    let message =
        rust_llm::protocols::anthropic::parse_completion_body(&body, RawResponse::default())
            .unwrap();
    let record = ChatRecord::create(&db, "claude-haiku-4-5", Some("anthropic"))
        .await
        .unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .add_message(&db, &mut chat, Message::user("Hi"))
        .await
        .unwrap();
    record.add_message(&db, &mut chat, message).await.unwrap();
    record
        .add_message(&db, &mut chat, Message::user("And now?"))
        .await
        .unwrap();

    let payload = ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .to_llm(&db)
        .await
        .unwrap()
        .render()
        .unwrap();
    assert_eq!(payload["messages"][1]["content"], blocks);
}

// spec: active_record/chat_methods_thinking_replay_spec.rb:53 drops a persisted Gemini thought signature when the chat moves to Anthropic
#[tokio::test]
async fn drops_a_persisted_gemini_signature_when_the_chat_moves_to_anthropic() {
    let db = db().await;
    let parts = json!([{ "text": "221", "thoughtSignature": "gemini-signature" }]);
    let body = json!({ "modelVersion": "gemini-2.5-flash", "candidates": [{ "content": { "parts": parts } }] });
    let gemini = rust_llm::models()
        .find("gemini-2.5-flash", Some("gemini"))
        .unwrap();
    let mut message =
        rust_llm::protocols::gemini::parse_completion_body(&gemini, &body, RawResponse::default())
            .unwrap();
    message.usage_entries = vec![UsageEntry {
        id: UsageEntry::next_id(),
        operation: Operation::Chat,
        provider: "gemini".into(),
        model: "gemini-2.5-flash".into(),
        status: UsageStatus::Succeeded,
        tokens: Default::default(),
        cost: Cost::default(),
    }];
    let mut record = ChatRecord::create(&db, "gemini-2.5-flash", Some("gemini"))
        .await
        .unwrap();
    let mut chat = record.to_llm(&db).await.unwrap();
    record
        .add_message(&db, &mut chat, Message::user("13*17?"))
        .await
        .unwrap();
    // `persist_message_completion` with the usage entry `persist_usage_entry` recorded.
    chat.add_message(message);
    record.persist_collected(&db, &mut chat).await.unwrap();
    let mut chat = record
        .with_model(&db, chat, "claude-haiku-4-5", Some("anthropic"))
        .await
        .unwrap();
    record
        .add_message(&db, &mut chat, Message::user("As a table."))
        .await
        .unwrap();

    let payload = ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .to_llm(&db)
        .await
        .unwrap()
        .render()
        .unwrap();
    assert_eq!(
        payload["messages"][1],
        json!({ "role": "assistant", "content": [{ "type": "text", "text": "221" }] })
    );
    let rows = record.messages(&db).await.unwrap();
    assert_eq!(
        rows[1].thinking_signature.as_deref(),
        Some("gemini-signature")
    );
}

// ---- chat_methods_compact_spec ----------------------------------------------------------------

/// `klass.model model_for(:xai, :provider_tools), provider: :xai` with `chat_model Chat`.
struct Compacting {
    context: rust_llm::Context,
}

impl Agent for Compacting {
    fn model(&self) -> Option<&str> {
        Some("grok-4.3")
    }
    fn provider(&self) -> Option<&str> {
        Some("xai")
    }
    fn context(&self) -> Option<rust_llm::Context> {
        Some(self.context.clone())
    }
}

fn compaction_body(output: Value) -> Value {
    json!({ "object": "response.compaction", "output": output,
            "usage": { "input_tokens": 31, "output_tokens": 11 } })
}

async fn compact_server(bodies: Vec<Value>) -> MockServer {
    let server = MockServer::start().await;
    let n = bodies.len();
    for (i, body) in bodies.into_iter().enumerate() {
        let mock = Mock::given(matchers::method("POST"))
            .and(matchers::path("/v1/responses/compact"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .with_priority((i + 1) as u8);
        if i + 1 < n {
            mock.up_to_n_times(1).mount(&server).await
        } else {
            mock.mount(&server).await
        }
    }
    server
}

// spec: active_record/chat_methods_compact_spec.rb:26 persists compaction and its usage through Agent.find without deleting earlier messages
#[tokio::test]
async fn persists_compaction_and_its_usage_through_agent_find() {
    let db = db().await;
    let output =
        json!([{ "type": "compaction", "id": "cmp_1", "encrypted_content": "opaque context" }]);
    let body = compaction_body(output.clone());
    let server = compact_server(vec![body.clone()]).await;
    let agent = Compacting {
        context: context_for(&server),
    };
    let (record, mut chat) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    record
        .with_instructions(&db, &mut chat, "Remember Thimble.")
        .await
        .unwrap();
    record
        .ask_later(&db, &mut chat, "The project language is Ruby.")
        .await
        .unwrap();
    record
        .add_message(&db, &mut chat, Message::assistant("I will remember Ruby."))
        .await
        .unwrap();
    let ids: Vec<i32> = record
        .messages(&db)
        .await
        .unwrap()
        .iter()
        .map(|m| m.id)
        .collect();

    let result = record.compact(&db, &mut chat).await.unwrap();
    let (restored, restored_chat) = ChatRecord::find_for_agent(&db, record.id(), &agent)
        .await
        .unwrap();

    assert_eq!(result.role, Role::Assistant);
    let rows = restored.messages(&db).await.unwrap();
    assert!(ids.iter().all(|id| rows.iter().any(|r| r.id == *id)));
    assert_eq!(rows.len(), 4);
    assert_eq!(rows.last().unwrap().raw_content, Some(body));
    let tokens = restored.tokens(&db).await.unwrap();
    assert_eq!((tokens.input, tokens.output), (Some(31), Some(11)));
    assert_eq!(restored_chat.render().unwrap()["input"], output);

    let (mut restored, mut restored_chat) = (restored, restored_chat);
    restored
        .set_instructions(
            &db,
            &mut restored_chat,
            Some("Answer with only the language name."),
            false,
            true,
            false,
        )
        .await
        .unwrap();
    restored
        .ask_later(&db, &mut restored_chat, "Which language?")
        .await
        .unwrap();
    let payload = ChatRecord::find_for_agent(&db, record.id(), &agent)
        .await
        .unwrap()
        .1
        .render()
        .unwrap();
    assert_eq!(
        payload["instructions"],
        "Answer with only the language name."
    );
    let mut expected = output.as_array().unwrap().clone();
    expected.push(json!({ "role": "user", "content": "Which language?" }));
    assert_eq!(payload["input"], Value::Array(expected));
}

// spec: active_record/chat_methods_compact_spec.rb:50 keeps only the last compacted context on the wire after reloading multiple rounds
#[tokio::test]
async fn keeps_only_the_last_compacted_context_after_reloading() {
    let db = db().await;
    let first =
        json!([{ "type": "compaction", "id": "cmp_1", "encrypted_content": "opaque context" }]);
    let second =
        json!([{ "type": "compaction", "id": "cmp_2", "encrypted_content": "new context" }]);
    let server = compact_server(vec![
        compaction_body(first),
        compaction_body(second.clone()),
    ])
    .await;
    let agent = Compacting {
        context: context_for(&server),
    };
    let (record, mut chat) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    record
        .ask_later(&db, &mut chat, "The project language is Ruby.")
        .await
        .unwrap();
    record.compact(&db, &mut chat).await.unwrap();
    let (restored, mut restored_chat) = ChatRecord::find_for_agent(&db, record.id(), &agent)
        .await
        .unwrap();
    restored
        .ask_later(&db, &mut restored_chat, "The project codename is Thimble.")
        .await
        .unwrap();

    restored.compact(&db, &mut restored_chat).await.unwrap();

    let (found, found_chat) = ChatRecord::find_for_agent(&db, record.id(), &agent)
        .await
        .unwrap();
    assert_eq!(found_chat.render().unwrap()["input"], second);
    assert_eq!(record.messages(&db).await.unwrap().len(), 4);
    let tokens = found.tokens(&db).await.unwrap();
    assert_eq!((tokens.input, tokens.output), (Some(62), Some(22)));
}

// spec: active_record/chat_methods_compact_spec.rb:66 honors cancellation written by another process before compaction
#[tokio::test]
async fn honors_a_cancellation_written_by_another_process_before_compaction() {
    let db = db().await;
    let server = compact_server(vec![compaction_body(json!([]))]).await;
    let agent = Compacting {
        context: context_for(&server),
    };
    let (record, mut chat) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    record
        .ask_later(&db, &mut chat, "Keep this message.")
        .await
        .unwrap();
    let (restored, mut restored_chat) = ChatRecord::find_for_agent(&db, record.id(), &agent)
        .await
        .unwrap();
    ChatRecord::find(&db, record.id())
        .await
        .unwrap()
        .cancel(&db)
        .await
        .unwrap();

    let err = restored.compact(&db, &mut restored_chat).await.unwrap_err();

    assert!(
        matches!(err, rust_llm_loco::Error::Llm(rust_llm::Error::Cancelled)),
        "{err:?}"
    );
    assert_eq!(record.messages(&db).await.unwrap().len(), 1);
    assert!(server.received_requests().await.unwrap().is_empty());
}

// ---- chat_methods_server_approval_spec ---------------------------------------------------------

/// `provider_tools mcp: { name: 'docs', url: ..., require_approval: 'always' }` on
/// `model_for(:openai), provider: :openai, protocol: :responses`.
struct Docs {
    context: rust_llm::Context,
}

impl Agent for Docs {
    fn model(&self) -> Option<&str> {
        Some("gpt-5-nano")
    }
    fn provider(&self) -> Option<&str> {
        Some("openai")
    }
    fn protocol(&self) -> Option<ProtocolName> {
        Some(ProtocolName::Responses)
    }
    fn context(&self) -> Option<rust_llm::Context> {
        Some(self.context.clone())
    }
    fn provider_tools(&self) -> Vec<rust_llm::provider_tools::ProviderTool> {
        vec![rust_llm::provider_tools::ProviderTool::with_options(
            "mcp",
            json!({ "name": "docs", "url": "https://example.test/mcp", "require_approval": "always" }),
        )]
    }
}

fn raw_approval(call_id: &str) -> Value {
    json!({ "type": "mcp_approval_request", "id": call_id, "name": "search",
            "arguments": "{\"query\":\"Ruby\"}", "server_label": "docs" })
}

async fn parked_docs_chat(db: &DatabaseConnection, agent: &Docs, call_id: &str) -> ChatRecord {
    let (record, mut chat) = ChatRecord::create_for_agent(db, agent).await.unwrap();
    let body = json!({ "status": "completed", "output": [raw_approval(call_id)] });
    let message = rust_llm::protocols::responses::parse_completion_body(
        rust_llm::Provider::OpenAI,
        &body,
        RawResponse::default(),
    )
    .unwrap();
    record.add_message(db, &mut chat, message).await.unwrap();
    record
}

// spec: active_record/chat_methods_server_approval_spec.rb:32 preserves a remote #{approved ? 'approval' : 'denial'} through Agent.find and Rails result serialization
#[tokio::test]
async fn preserves_a_remote_decision_through_agent_find() {
    for approved in [true, false] {
        let db = db().await;
        let server = MockServer::start().await;
        let agent = Docs {
            context: context_for(&server),
        };
        let call_id = format!("approval_{approved}");
        let record = parked_docs_chat(&db, &agent, &call_id).await;
        let (restored, mut restored_chat) = ChatRecord::find_for_agent(&db, record.id(), &agent)
            .await
            .unwrap();
        let pending = restored
            .pending_approvals(&db, &mut restored_chat)
            .await
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].remote);
        assert_eq!(pending[0].arguments, Some(json!({ "query": "Ruby" })));
        assert!(
            restored
                .is_awaiting_approval(&db, &mut restored_chat)
                .await
                .unwrap()
        );
        if approved {
            restored
                .approve(&db, &mut restored_chat, &call_id)
                .await
                .unwrap();
        } else {
            restored
                .deny(&db, &mut restored_chat, &call_id)
                .await
                .unwrap();
        }

        let (resumed, mut resumed_chat) = ChatRecord::find_for_agent(&db, record.id(), &agent)
            .await
            .unwrap();
        resumed.run_tools(&db, &mut resumed_chat).await.unwrap();
        resumed.run_tools(&db, &mut resumed_chat).await.unwrap();

        let tools: Vec<messages::Model> = resumed
            .messages(&db)
            .await
            .unwrap()
            .into_iter()
            .filter(|m| m.role == "tool")
            .collect();
        assert_eq!(tools.len(), 1, "approved={approved}");
        let result = message_to_llm(&db, &tools[0]).await.unwrap();
        let response = json!([{ "type": "mcp_approval_response", "approval_request_id": call_id, "approve": approved }]);
        assert_eq!(result.raw_content, Some(response.clone()));
        assert_eq!(result.tool_call_id.as_deref(), Some(call_id.as_str()));
        let input = ChatRecord::find_for_agent(&db, record.id(), &agent)
            .await
            .unwrap()
            .1
            .render()
            .unwrap()["input"]
            .clone();
        let input = input.as_array().unwrap();
        assert!(input.contains(&raw_approval(&call_id)), "{input:?}");
        assert!(input.contains(&response[0]), "{input:?}");
    }
}

/// Ruby turns the query cache on and writes the decision through a second SQLite connection.
/// SeaORM has no query cache; the second process here is a second connection to the same file.
// spec: active_record/chat_methods_server_approval_spec.rb:53 reads a decision written by another worker while the query cache is enabled
#[tokio::test]
async fn reads_a_decision_another_worker_wrote() {
    let dir = std::env::temp_dir().join(format!("rust_llm_approval_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&dir);
    let url = format!("sqlite://{}?mode=rwc", dir.display());
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
    });
    let db = Database::connect(&url).await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    let server = MockServer::start().await;
    let agent = Docs {
        context: context_for(&server),
    };
    let record = parked_docs_chat(&db, &agent, "approval_worker").await;
    let mut chat = record.to_llm(&db).await.unwrap();
    let pending = record.pending_approvals(&db, &mut chat).await.unwrap();
    assert!(record.is_awaiting_approval(&db, &mut chat).await.unwrap());

    let worker = Database::connect(&url).await.unwrap();
    rust_llm_tool_calls::Entity::update_many()
        .col_expr(
            rust_llm_tool_calls::Column::Approval,
            sea_orm::sea_query::Expr::value("approved"),
        )
        .filter(rust_llm_tool_calls::Column::Id.eq(pending[0].id))
        .exec(&worker)
        .await
        .unwrap();

    assert!(!record.is_awaiting_approval(&db, &mut chat).await.unwrap());
    assert!(
        record
            .pending_approvals(&db, &mut chat)
            .await
            .unwrap()
            .is_empty()
    );
    let _ = std::fs::remove_file(&dir);
}

// ---- acts_as_batch_spec ------------------------------------------------------------------------

const HAIKU: &str = "claude-haiku-4-5";

fn anthropic_answer(text: &str) -> Value {
    json!({ "id": "msg_1", "type": "message", "role": "assistant", "model": HAIKU,
            "content": [{ "type": "text", "text": text }], "stop_reason": "end_turn",
            "usage": { "input_tokens": 5, "output_tokens": 1 } })
}

/// Anthropic's batch endpoints for `id`: created in progress, then ended, with `answers` as the
/// results (`[index, answer]`).
async fn anthropic_batch(id: &str, answers: &[(usize, &str)]) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/v1/messages/batches"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "id": id, "processing_status": "in_progress" })),
        )
        .mount(&server)
        .await;
    Mock::given(matchers::method("GET"))
        .and(matchers::path(format!("/v1/messages/batches/{id}")))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "id": id, "processing_status": "ended" })),
        )
        .mount(&server)
        .await;
    let lines: Vec<String> = answers
        .iter()
        .map(|(i, text)| {
            json!({ "custom_id": i.to_string(), "result": { "type": "succeeded", "message": anthropic_answer(text) } })
                .to_string()
        })
        .collect();
    Mock::given(matchers::method("GET"))
        .and(matchers::path(format!("/v1/messages/batches/{id}/results")))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(lines.join("\n"), "application/octet-stream"),
        )
        .mount(&server)
        .await;
    server
}

/// `Chat.create!(model:).ask_later(text)`, built with `config` (the batch store configured).
async fn staged(
    db: &DatabaseConnection,
    config: &Arc<Config>,
    text: &str,
) -> (ChatRecord, rust_llm::Chat) {
    let record = ChatRecord::create(db, HAIKU, None).await.unwrap();
    let mut chat = record.to_llm_with(db, config.clone()).await.unwrap();
    record.ask_later(db, &mut chat, text).await.unwrap();
    (record, chat)
}

fn store_config(server: &MockServer, db: &DatabaseConnection) -> Arc<Config> {
    let mut c = (*config_for(server)).clone();
    c.batch_store = Some(Arc::new(rust_llm_loco::BatchStore::new(db.clone())));
    Arc::new(c)
}

/// Ruby's Railtie sets `config.batch_store = RubyLLM::ActiveRecord::Batch`; the Rust app sets
/// `config.batch_store` to [`rust_llm_loco::BatchStore`], which is the `rust_llm_batches` record
/// adapter itself (there is no separate store class).
// spec: active_record/acts_as_batch_spec.rb:10 uses the record itself as the Rails persistence adapter
#[tokio::test]
async fn the_record_is_the_persistence_adapter() {
    let db = db().await;
    let server = anthropic_batch("msgbatch_0", &[(0, "4")]).await;
    let config = store_config(&server, &db);
    let (record, chat) = staged(&db, &config, "What is 2 + 2?").await;
    let batch = rust_llm_loco::batch::submit(&db, vec![(record, chat)])
        .await
        .unwrap();

    let found = rust_llm::Batch::find_with_config(config, batch.id(), None)
        .await
        .unwrap();
    assert_eq!(found.id(), "msgbatch_0");
    assert_eq!(
        found.chats().map(<[_]>::len),
        Some(1),
        "answered from the row"
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|r| r.method.as_str() == "POST"),
        "found without asking the provider"
    );
}

// spec: active_record/acts_as_batch_spec.rb:26 submits, persists the batch and its chats, and routes answers home
#[tokio::test]
async fn submits_persists_and_routes_answers_home() {
    let db = db().await;
    let server = anthropic_batch("msgbatch_1", &[(0, "4"), (1, "Jupiter")]).await;
    let config = store_config(&server, &db);
    let first = staged(&db, &config, "What is 2 + 2? Just the number.").await;
    let second = staged(&db, &config, "Name the largest planet. One word.").await;
    let ids = [first.0.id(), second.0.id()];
    let (first_record, second_record) = (first.0.clone(), second.0.clone());

    let batch = rust_llm_loco::batch::submit(&db, vec![first, second])
        .await
        .unwrap();

    assert_eq!(batch.id(), "msgbatch_1");
    let row = rust_llm_loco::batch::find_record(&db, batch.id(), None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(row.provider, "anthropic");
    assert_eq!(rust_llm_loco::batch::chat_ids(&row), ids);
    let records = rust_llm_loco::batch::records(&db, &row).await.unwrap();
    assert_eq!(
        records
            .iter()
            .map(|r| r.as_ref().map(ChatRecord::id))
            .collect::<Vec<_>>(),
        [Some(ids[0]), Some(ids[1])]
    );

    // Poll from a fresh batch, the way a job in another process would.
    let mut polled = rust_llm::Batch::find_with_config(config, batch.id(), None)
        .await
        .unwrap();
    polled.refresh().await.unwrap();
    assert!(polled.is_complete());
    assert_eq!(polled.status(), rust_llm::BatchStatus::Succeeded);
    assert_eq!(polled.raw_status(), Some("ended"));

    rust_llm_loco::batch::collect(&db, &mut polled)
        .await
        .unwrap();

    let rows = first_record.messages(&db).await.unwrap();
    assert_eq!(
        rows.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
        ["user", "assistant"]
    );
    assert_eq!(rows.last().unwrap().content.as_deref(), Some("4"));
    let answer = message_to_llm(&db, rows.last().unwrap()).await.unwrap();
    assert_eq!(answer.tokens().input, Some(5));
    assert_eq!(
        contents(&second_record, &db).await.last().unwrap(),
        "Jupiter"
    );
}

/// Anthropic reports no batch invoice, so this runs on OpenRouter, whose batch endpoint reports
/// `usage.cost` the way Ruby's stubbed `find_batch` returns `reported_cost`.
// spec: active_record/acts_as_batch_spec.rb:59 persists a reported batch invoice through refresh and fresh Batch.find calls
#[tokio::test]
async fn persists_a_reported_invoice_through_refresh_and_find() {
    let db = db().await;
    let server = MockServer::start().await;
    Mock::given(matchers::method("GET"))
        .and(matchers::path("/api/beta/batches/batch_invoice"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "batch_invoice", "status": "completed", "usage": { "cost": 0.001 }
        })))
        .mount(&server)
        .await;
    let mut c = (*store_config(&server, &db)).clone();
    c.set("openrouter_api_base", format!("{}/api/v1", server.uri()));
    c.set("openrouter_api_key", "test");
    let config = Arc::new(c);
    let (record, _) = staged(&db, &config, "What is 2 + 2?").await;
    // `RubyLLM.batch(chat)` with `create_batch` reporting a zero invoice.
    let submitted = rust_llm::Batch::from_attributes(
        config.clone(),
        "openrouter",
        rust_llm::batch::BatchAttributes {
            id: "batch_invoice".into(),
            raw_status: Some("in_progress".into()),
            reported_cost: Some(Cost::from_h(&json!({ "total": 0 }), None)),
            ..Default::default()
        },
    )
    .unwrap();
    rust_llm_loco::batch::persist(&db, &submitted, &[record.id()])
        .await
        .unwrap();

    let mut pending = rust_llm::Batch::find_with_config(config.clone(), "batch_invoice", None)
        .await
        .unwrap();
    assert_eq!(pending.reported_cost().unwrap().total(), Some(0.0));
    assert_eq!(pending.cost().await.unwrap().total(), None);

    pending.refresh().await.unwrap();
    let mut restored = rust_llm::Batch::find_with_config(config.clone(), "batch_invoice", None)
        .await
        .unwrap();
    assert_eq!(restored.reported_cost().unwrap().total(), Some(0.001));
    assert_eq!(restored.cost().await.unwrap().total(), Some(0.001));
    let stored = rust_llm_loco::batch::find_record(&db, "batch_invoice", None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.reported_cost, Some(json!({ "total": 0.001 })));

    // `Batch.sync(stale)`: a batch without an invoice keeps the stored one.
    let stale = rust_llm::Batch::from_attributes(
        config.clone(),
        "openrouter",
        rust_llm::batch::BatchAttributes {
            id: "batch_invoice".into(),
            raw_status: Some("completed".into()),
            completed: true,
            ..Default::default()
        },
    )
    .unwrap();
    rust_llm_loco::batch::sync(&db, &stale).await.unwrap();
    let found = rust_llm::Batch::find_with_config(config, "batch_invoice", None)
        .await
        .unwrap();
    assert_eq!(found.reported_cost().unwrap().total(), Some(0.001));
}

// spec: active_record/acts_as_batch_spec.rb:85 is idempotent: re-collecting never appends an answer twice
#[tokio::test]
async fn re_collecting_never_appends_an_answer_twice() {
    let db = db().await;
    let server = anthropic_batch("msgbatch_2", &[(0, "4")]).await;
    let config = store_config(&server, &db);
    let (record, chat) = staged(&db, &config, "What is 2 + 2?").await;
    let batch = rust_llm_loco::batch::submit(&db, vec![(record.clone(), chat)])
        .await
        .unwrap();

    let mut first = rust_llm::Batch::find_with_config(config.clone(), batch.id(), None)
        .await
        .unwrap();
    first.refresh().await.unwrap();
    rust_llm_loco::batch::collect(&db, &mut first)
        .await
        .unwrap();
    let mut retry = rust_llm::Batch::find_with_config(config, batch.id(), None)
        .await
        .unwrap();
    rust_llm_loco::batch::collect(&db, &mut retry)
        .await
        .unwrap();

    let assistants = record
        .messages(&db)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| m.role == "assistant")
        .count();
    assert_eq!(assistants, 1);
}

// spec: active_record/acts_as_batch_spec.rb:100 keeps answers aligned when a chat was deleted before collection
#[tokio::test]
async fn keeps_answers_aligned_when_a_chat_was_deleted() {
    let db = db().await;
    let server = anthropic_batch("msgbatch_3", &[(0, "first"), (1, "second")]).await;
    let config = store_config(&server, &db);
    let first = staged(&db, &config, "First question.").await;
    let second = staged(&db, &config, "Second question.").await;
    let (doomed, survivor) = (first.0.clone(), second.0.clone());
    let batch = rust_llm_loco::batch::submit(&db, vec![first, second])
        .await
        .unwrap();

    doomed.destroy(&db).await.unwrap(); // gone by the time the job polls

    let mut polled = rust_llm::Batch::find_with_config(config, batch.id(), None)
        .await
        .unwrap();
    polled.refresh().await.unwrap();
    rust_llm_loco::batch::collect(&db, &mut polled)
        .await
        .unwrap();

    assert_eq!(contents(&survivor, &db).await.last().unwrap(), "second");
}

// spec: active_record/acts_as_spec.rb:127 usage persistence persists a batch response as one usage attempt
#[tokio::test]
async fn persists_a_batch_response_as_one_usage_attempt() {
    let db = db().await;
    let server = anthropic_batch("batch_test", &[(0, "Hello")]).await;
    let config = store_config(&server, &db);
    let (record, chat) = staged(&db, &config, "Hello").await;
    let batch = rust_llm_loco::batch::submit(&db, vec![(record.clone(), chat)])
        .await
        .unwrap();
    let mut polled = rust_llm::Batch::find_with_config(config, batch.id(), None)
        .await
        .unwrap();
    polled.refresh().await.unwrap();

    rust_llm_loco::batch::collect(&db, &mut polled)
        .await
        .unwrap();

    let usages = record.usages(&db).await.unwrap();
    assert_eq!(usages.len(), 1);
    let last = record.messages(&db).await.unwrap().pop().unwrap();
    assert_eq!(usages[0].status, "succeeded");
    assert_eq!(usages[0].message_id, Some(last.id as i64));
    assert_eq!(
        (usages[0].input_tokens, usages[0].output_tokens),
        (Some(5), Some(1))
    );
    assert_eq!(
        record.cost(&db).await.unwrap().total(),
        usages[0].total_cost
    );
}

// ---- agent_rails_spec --------------------------------------------------------------------------

/// `instructions ...` declarations evaluated for a chat (`config`, the record as `chat`).
type Declarations = fn(&Arc<Config>, &Value) -> rust_llm::Result<Vec<InstructionDeclaration>>;

/// A named agent reading `app/prompts/<name>/` under a temporary prompt root.
struct Named {
    name: &'static str,
    context: Option<rust_llm::Context>,
    declarations: Option<Declarations>,
    model: &'static str,
    provider: Option<&'static str>,
    assume: bool,
    protocol: Option<ProtocolName>,
    locals: Value,
}

impl Named {
    fn new(name: &'static str) -> Named {
        Named {
            name,
            context: None,
            declarations: None,
            model: MODEL,
            provider: None,
            assume: false,
            protocol: None,
            locals: json!({}),
        }
    }
}

impl Agent for Named {
    fn model(&self) -> Option<&str> {
        Some(self.model)
    }
    fn provider(&self) -> Option<&str> {
        self.provider
    }
    fn assume_model_exists(&self) -> bool {
        self.assume
    }
    fn protocol(&self) -> Option<ProtocolName> {
        self.protocol
    }
    fn name(&self) -> String {
        self.name.into()
    }
    fn context(&self) -> Option<rust_llm::Context> {
        self.context.clone()
    }
    fn prompt_locals(&self) -> Value {
        self.locals.clone()
    }
    fn instruction_declarations(
        &self,
        config: &Arc<Config>,
        chat: &Value,
    ) -> rust_llm::Result<Vec<InstructionDeclaration>> {
        match self.declarations {
            Some(f) => f(config, chat),
            None => {
                let default = DefaultDeclarations(self);
                default.instruction_declarations(config, chat)
            }
        }
    }
}

/// Delegates to the trait's default `instruction_declarations` for a [`Named`] agent.
struct DefaultDeclarations<'a>(&'a Named);

impl Agent for DefaultDeclarations<'_> {
    fn name(&self) -> String {
        self.0.name.into()
    }
    fn prompt_locals(&self) -> Value {
        self.0.locals.clone()
    }
}

/// A context whose prompt root is a fresh directory holding `<agent>/instructions.txt.jinja`.
fn prompt_context(agent: &str, template: Option<&str>) -> rust_llm::Context {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let root = std::env::temp_dir().join(format!(
        "rust_llm_agent_rails_{}_{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    let dir = root.join(agent);
    std::fs::create_dir_all(&dir).unwrap();
    if let Some(template) = template {
        std::fs::write(dir.join("instructions.txt.jinja"), template).unwrap();
    }
    rust_llm::context(|c| {
        c.set("prompt_root", root.display().to_string());
    })
}

// spec: agent_rails_spec.rb:8 uses block-configured credentials when creating and finding a chat
#[tokio::test]
async fn uses_block_configured_credentials_when_creating_and_finding() {
    let db = db().await;
    let global_key = rust_llm::config().get("openai_api_key").map(str::to_string);
    let mut agent = Named::new("credentialed_agent");
    agent.provider = Some("openai");
    agent.context = Some(rust_llm::context(|c| {
        c.set("openai_api_key", "agent-key");
        c.set("openai_api_base", "https://example.com/v1");
    }));

    let (created, chat) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    assert_eq!(chat.config().get("openai_api_key"), Some("agent-key"));
    let (_, found) = ChatRecord::find_for_agent(&db, created.id(), &agent)
        .await
        .unwrap();
    assert_eq!(
        found.config().get("openai_api_base"),
        Some("https://example.com/v1")
    );
    assert_eq!(
        rust_llm::config().get("openai_api_key").map(str::to_string),
        global_key,
        "the global configuration is untouched"
    );
}

/// Ruby's bare `instructions` macro names the conventional prompt, which is what an agent with
/// no declared instructions renders in Rust.
// spec: agent_rails_spec.rb:57 loads instructions.txt.erb when instructions is called without arguments
// spec: agent_rails_spec.rb:77 loads instructions.txt.erb automatically when a named agent has no instructions macro
#[tokio::test]
async fn loads_the_conventional_prompt_with_the_record_as_chat() {
    let db = db().await;
    let mut agent = Named::new("spec_default_prompt_agent");
    agent.context = Some(prompt_context(
        "spec_default_prompt_agent",
        Some("Default prompt for chat {{ chat.id }}"),
    ));

    let (record, _) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();

    assert_eq!(
        system_rows(&record, &db).await,
        [(format!("Default prompt for chat {}", record.id()), false)]
    );
}

// spec: agent_rails_spec.rb:108 raises when an explicitly referenced prompt file is missing
#[tokio::test]
async fn raises_when_an_explicitly_referenced_prompt_file_is_missing() {
    let db = db().await;
    let mut agent = Named::new("spec_missing_prompt_agent");
    agent.context = Some(prompt_context("spec_missing_prompt_agent", None));
    // `instructions { prompt('instructions') }`: an explicit reference renders through the
    // prompt roots, where no such file exists.
    agent.declarations = Some(|config, _| {
        rust_llm::Prompt::with_config(config.clone(), "spec_missing_prompt_agent/instructions")
            .render(json!({}))
            .map(|text| vec![InstructionDeclaration::new(text)])
    });

    let err = ChatRecord::create_for_agent(&db, &agent).await.unwrap_err();

    assert!(
        matches!(err, rust_llm_loco::Error::Llm(rust_llm::Error::PromptNotFound(ref m)) if m.contains("Prompt file not found")),
        "{err:?}"
    );
}

// spec: agent_rails_spec.rb:201 combines persisted and unpersisted instruction declarations
#[tokio::test]
async fn combines_persisted_and_unpersisted_instruction_declarations() {
    let db = db().await;
    let mut agent = Named::new("spec_layered_instructions_agent");
    agent.declarations = Some(|_, chat| {
        Ok(vec![
            InstructionDeclaration {
                cache_until_here: true,
                ..InstructionDeclaration::new("Stable policy")
            },
            InstructionDeclaration {
                append: true,
                persist: false,
                ..InstructionDeclaration::new(format!("Current chat {}", chat["id"]))
            },
        ])
    });

    let (created, chat) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    assert_eq!(
        system_rows(&created, &db).await,
        [("Stable policy".to_string(), true)]
    );
    assert_eq!(
        texts(&chat),
        [
            "Stable policy".to_string(),
            format!("Current chat {}", created.id())
        ]
    );

    let (loaded, chat) = ChatRecord::find_for_agent(&db, created.id(), &agent)
        .await
        .unwrap();
    assert_eq!(
        system_rows(&loaded, &db).await,
        [("Stable policy".to_string(), true)]
    );
    assert_eq!(
        texts(&chat),
        [
            "Stable policy".to_string(),
            format!("Current chat {}", created.id())
        ]
    );
    assert_eq!(
        chat.messages()
            .iter()
            .map(|m| m.cache_until_here)
            .collect::<Vec<_>>(),
        [true, false]
    );
}

thread_local! {
    static VERSION: std::cell::RefCell<&'static str> = const { std::cell::RefCell::new("one") };
}

// spec: agent_rails_spec.rb:235 syncs only persistent instruction declarations
#[tokio::test]
async fn syncs_only_persistent_instruction_declarations() {
    let db = db().await;
    let mut agent = Named::new("spec_instruction_persistence_agent");
    agent.declarations = Some(|_, _| {
        let version = VERSION.with(|v| *v.borrow());
        Ok(vec![
            InstructionDeclaration::new(format!("Stable {version}")),
            InstructionDeclaration {
                append: true,
                persist: false,
                ..InstructionDeclaration::new(format!("Runtime {version}"))
            },
        ])
    });

    let (record, _) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    VERSION.with(|v| *v.borrow_mut() = "two");
    ChatRecord::sync_instructions(&db, record.id(), &agent)
        .await
        .unwrap();

    assert_eq!(
        system_rows(&record, &db).await,
        [("Stable two".to_string(), false)]
    );
}

/// `instructions display_name: -> { display_name }` with `inputs :display_name`: the input is
/// the agent's prompt local here.
fn display_agent(name: &'static str, display_name: &str) -> Named {
    let mut agent = Named::new(name);
    agent.context = Some(prompt_context(
        name,
        Some("System for {{ display_name }} on chat {{ chat.id }}"),
    ));
    agent.locals = json!({ "display_name": display_name });
    agent
}

// spec: agent_rails_spec.rb:299 keeps runtime instructions on repeated to_llm calls after find
#[tokio::test]
async fn keeps_runtime_instructions_on_repeated_to_llm_after_find() {
    let db = db().await;
    let ava = display_agent("spec_runtime_reuse_agent", "Ava");
    let (record, _) = ChatRecord::create_for_agent(&db, &ava).await.unwrap();
    let mut bea = display_agent("spec_runtime_reuse_agent", "Bea");
    bea.context = ava.context.clone();

    let (loaded, _) = ChatRecord::find_for_agent(&db, record.id(), &bea)
        .await
        .unwrap();

    let expected = format!("System for Bea on chat {}", record.id());
    for _ in 0..2 {
        let chat = loaded.to_llm(&db).await.unwrap();
        assert_eq!(chat.messages()[0].content(), expected);
    }
}

// spec: agent_rails_spec.rb:323 syncs instructions explicitly via .sync_instructions
#[tokio::test]
async fn syncs_instructions_explicitly() {
    let db = db().await;
    let ava = display_agent("spec_sync_agent", "Ava");
    let (record, _) = ChatRecord::create_for_agent(&db, &ava).await.unwrap();
    let system = |who: &str| vec![(format!("System for {who} on chat {}", record.id()), false)];
    assert_eq!(system_rows(&record, &db).await, system("Ava"));

    let mut bea = display_agent("spec_sync_agent", "Bea");
    bea.context = ava.context.clone();
    ChatRecord::find_for_agent(&db, record.id(), &bea)
        .await
        .unwrap();
    assert_eq!(system_rows(&record, &db).await, system("Ava"));

    ChatRecord::sync_instructions(&db, record.id(), &bea)
        .await
        .unwrap();
    assert_eq!(system_rows(&record, &db).await, system("Bea"));

    let mut cia = display_agent("spec_sync_agent", "Cia");
    cia.context = ava.context.clone();
    ChatRecord::sync_instructions(&db, record.id(), &cia)
        .await
        .unwrap();
    assert_eq!(system_rows(&record, &db).await, system("Cia"));
}

fn assumed_agent() -> Named {
    let mut agent = Named::new("spec_assume_exists_agent");
    agent.model = "not-a-real-model";
    agent.provider = Some("openai");
    agent.assume = true;
    agent.declarations = Some(|_, _| Ok(vec![InstructionDeclaration::new("Hello")]));
    agent
}

// spec: agent_rails_spec.rb:363 propagates assume_model_exists from class config when using find
#[tokio::test]
async fn propagates_assume_model_exists_through_find() {
    let db = db().await;
    let agent = assumed_agent();
    let (created, _) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();

    let (found, chat) = ChatRecord::find_for_agent(&db, created.id(), &agent)
        .await
        .unwrap();

    assert!(found.assume_model_exists);
    assert_eq!(chat.model().id, "not-a-real-model");
}

// spec: agent_rails_spec.rb:376 propagates assume_model_exists from class config when using sync_instructions with id
#[tokio::test]
async fn propagates_assume_model_exists_through_sync_instructions() {
    let db = db().await;
    let agent = assumed_agent();
    let (created, _) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();

    let (synced, _) = ChatRecord::sync_instructions(&db, created.id(), &agent)
        .await
        .unwrap();

    assert!(synced.assume_model_exists);
}

// spec: agent_rails_spec.rb:389 propagates assume_model_exists from class config when initializing with a reloaded chat record
#[tokio::test]
async fn propagates_assume_model_exists_to_a_reloaded_record() {
    let db = db().await;
    let agent = assumed_agent();
    let (created, _) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    let reloaded = ChatRecord::find(&db, created.id()).await.unwrap();
    assert!(!reloaded.assume_model_exists, "not persisted");

    let (record, chat) = reloaded.apply_agent(&db, &agent, true).await.unwrap();

    assert!(record.assume_model_exists);
    assert_eq!(chat.model().id, "not-a-real-model");
}

// spec: agent_rails_spec.rb:403 forwards the protocol model option to created and found Rails chat records
#[tokio::test]
async fn forwards_the_protocol_to_created_and_found_records() {
    let db = db().await;
    let mut agent = Named::new("spec_protocol_agent");
    agent.model = "gpt-5-nano";
    agent.protocol = Some(ProtocolName::ChatCompletions);
    agent.declarations = Some(|_, _| Ok(vec![InstructionDeclaration::new("Hello")]));

    let (created, _) = ChatRecord::create_for_agent(&db, &agent).await.unwrap();
    assert_eq!(created.protocol, Some(ProtocolName::ChatCompletions));
    assert_eq!(
        created.to_llm(&db).await.unwrap().protocol(),
        Some(ProtocolName::ChatCompletions)
    );

    let (found, _) = ChatRecord::find_for_agent(&db, created.id(), &agent)
        .await
        .unwrap();
    assert_eq!(
        found.to_llm(&db).await.unwrap().protocol(),
        Some(ProtocolName::ChatCompletions)
    );
}
