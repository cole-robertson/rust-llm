//! Port of `spec/ruby_llm/active_record/provider_file_spec.rb`: `rust_llm_provider_files`, the
//! provider uploads of stored attachments, reused by a chat loaded in another process.
//!
//! Ruby stubs Anthropic's inline limit down to 16 bytes; here each attachment really is past the
//! 24 MB limit. Ruby's Active Storage blob is a `rust_llm_attachments` row and its key the row's
//! `blob_key`. `new_process` clears the process's confirmed uploads, which every test in this
//! binary shares, so the tests run one at a time.

use std::sync::Arc;

use rust_llm::{Attachment, Config, Provider};
use rust_llm_loco::entities::{rust_llm_attachments, rust_llm_provider_files};
use rust_llm_loco::{ChatRecord, migrations, purge_attachments};
use sea_orm::{
    ColumnTrait, ConnectionTrait, Database, DatabaseConnection, EntityTrait, PaginatorTrait,
    QueryFilter, QueryOrder,
};
use sea_orm_migration::SchemaManager;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const MODEL: &str = "claude-haiku-4-5";
const MB: usize = 1024 * 1024;

static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn db() -> DatabaseConnection {
    rust_llm::configure(|c| {
        c.set("anthropic_api_key", "test");
    });
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

fn config(server: &MockServer, key: &str) -> Arc<Config> {
    let mut c = Config::default();
    c.set("anthropic_api_base", server.uri());
    c.set("anthropic_api_key", key);
    c.max_retries = 0;
    Arc::new(c)
}

fn json_response(body: Value, status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(body)
}

fn reply() -> Value {
    json!({ "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{ "type": "text", "text": "Noted." }], "stop_reason": "end_turn",
            "usage": { "input_tokens": 9, "output_tokens": 2 } })
}

fn file_body(id: &str) -> Value {
    json!({ "id": id, "type": "file", "filename": "notes.txt", "mime_type": "text/plain", "size_bytes": 40,
            "created_at": "2026-10-01T09:00:00Z", "downloadable": false })
}

/// Answers each request at `route` with `responses` in order, the last repeating.
async fn in_order(server: &MockServer, verb: &str, route: &str, responses: Vec<ResponseTemplate>) {
    let next = Arc::new(std::sync::Mutex::new(0usize));
    Mock::given(method(verb))
        .and(path(route))
        .respond_with(move |_: &Request| {
            let mut n = next.lock().unwrap();
            let r = responses[(*n).min(responses.len() - 1)].clone();
            *n += 1;
            r
        })
        .mount(server)
        .await;
}

/// The spec's `before` plus `stub_uploads(*ids)`.
async fn server_with_uploads(ids: &[&str]) -> MockServer {
    let server = MockServer::start().await;
    in_order(
        &server,
        "POST",
        "/v1/messages",
        vec![json_response(reply(), 200)],
    )
    .await;
    in_order(
        &server,
        "POST",
        "/v1/files",
        ids.iter()
            .map(|id| json_response(file_body(id), 200))
            .collect(),
    )
    .await;
    server
}

/// `stub_lookup(id, status:)`.
async fn stub_lookup(server: &MockServer, id: &str, status: u16) {
    let body = if status == 200 {
        file_body(id)
    } else {
        json!({ "type": "error", "error": { "type": "not_found_error", "message": "Not found" } })
    };
    Mock::given(method("GET"))
        .and(path(format!("/v1/files/{id}")))
        .respond_with(json_response(body, status))
        .mount(server)
        .await;
}

/// A text file past the inline limit, its first line saying which one it is.
fn notes_named(text: &str, filename: &str) -> Attachment {
    let mut bytes = text.as_bytes().to_vec();
    bytes.resize(25 * MB, b' ');
    Attachment::from_bytes(bytes, filename, Some("text/plain"))
}

fn notes() -> Attachment {
    notes_named("Meeting notes long enough to upload.", "notes.txt")
}

/// `RubyLLM::Protocol::StoredUploads.files.clear`.
fn new_process() {
    rust_llm::files::clear_confirmed_uploads();
}

async fn requests(server: &MockServer, verb: &str, route: &str) -> Vec<Request> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|r| r.method.as_str() == verb && r.url.path() == route)
        .collect()
}

/// `sent_file(id)`: requests to the messages endpoint that reference the file.
async fn sent_file(server: &MockServer, id: &str) -> usize {
    requests(server, "POST", "/v1/messages")
        .await
        .iter()
        .filter(|r| String::from_utf8_lossy(&r.body).contains(&format!("\"file_id\":\"{id}\"")))
        .count()
}

async fn rows(db: &DatabaseConnection) -> Vec<rust_llm_provider_files::Model> {
    rust_llm_provider_files::Entity::find()
        .order_by_asc(rust_llm_provider_files::Column::Id)
        .all(db)
        .await
        .unwrap()
}

async fn ask(
    db: &DatabaseConnection,
    record: &ChatRecord,
    config: &Arc<Config>,
    text: &str,
    with: Vec<Attachment>,
) {
    let mut chat = record.to_llm_with(db, config.clone()).await.unwrap();
    record.ask_with(db, &mut chat, text, with).await.unwrap();
}

/// `Chat.find(chat.id).ask(...)`: the record loaded again, as another process would.
async fn ask_again(db: &DatabaseConnection, record: &ChatRecord, config: &Arc<Config>, text: &str) {
    let found = ChatRecord::find(db, record.id()).await.unwrap();
    ask(db, &found, config, text, Vec::new()).await;
}

async fn attachment_rows(db: &DatabaseConnection) -> Vec<rust_llm_attachments::Model> {
    rust_llm_attachments::Entity::find()
        .order_by_asc(rust_llm_attachments::Column::Id)
        .all(db)
        .await
        .unwrap()
}

/// A stored blob, not yet uploaded anywhere: the attachment of a message another chat persisted,
/// read back as `to_llm` rebuilds it.
async fn stored_blob(db: &DatabaseConnection, config: &Arc<Config>, text: &str) -> Attachment {
    let holder = ChatRecord::create(db, MODEL, None).await.unwrap();
    let mut chat = holder.to_llm_with(db, config.clone()).await.unwrap();
    holder
        .ask_later_with(
            db,
            &mut chat,
            "Keep this",
            vec![notes_named(text, "notes.txt")],
        )
        .await
        .unwrap();
    let reloaded = holder.to_llm_with(db, config.clone()).await.unwrap();
    reloaded.messages()[0].attachments[0].clone()
}

// spec: active_record/provider_file_spec.rb:68 records the upload of a file asked about against its blob
#[tokio::test]
async fn records_the_upload_of_a_file_asked_about_against_its_blob() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1"]).await;
    let config = config(&server, "pf-68");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();

    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;

    let stored = rows(&db).await;
    assert_eq!(stored.len(), 1);
    assert_eq!(
        Some(stored[0].blob_key.clone()),
        attachment_rows(&db).await[0].blob_key
    );
    assert_eq!(
        (stored[0].provider.as_str(), stored[0].file_id.as_str()),
        ("anthropic", "file_1")
    );
    assert_eq!(
        Some(stored[0].account.clone()),
        Provider::Anthropic.account_identity(&config)
    );
}

// spec: active_record/provider_file_spec.rb:80 tells apart files of the same size asked about together
#[tokio::test]
async fn tells_apart_files_of_the_same_size_asked_about_together() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1", "file_2"]).await;
    let config = config(&server, "pf-80");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    let first = notes_named("First notes, long enough to upload.", "first.txt");
    let other = notes_named("Other notes, long enough to upload.", "other.txt");

    ask(
        &db,
        &record,
        &config,
        "Compare these notes",
        vec![first, other],
    )
    .await;

    let blobs: std::collections::HashMap<String, String> = attachment_rows(&db)
        .await
        .into_iter()
        .map(|a| (a.filename, a.blob_key.unwrap()))
        .collect();
    let mut stored: Vec<(String, String)> = rows(&db)
        .await
        .into_iter()
        .map(|r| (r.blob_key, r.file_id))
        .collect();
    stored.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(
        stored,
        [
            (blobs["first.txt"].clone(), "file_1".to_string()),
            (blobs["other.txt"].clone(), "file_2".to_string())
        ]
    );
}

// spec: active_record/provider_file_spec.rb:93 reuses the upload when the chat is loaded again in another process
#[tokio::test]
async fn reuses_the_upload_when_the_chat_is_loaded_again_in_another_process() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1"]).await;
    stub_lookup(&server, "file_1", 200).await;
    let config = config(&server, "pf-93");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;
    new_process();

    for _ in 0..2 {
        ask_again(&db, &record, &config, "And the action items?").await;
    }

    assert_eq!(requests(&server, "POST", "/v1/files").await.len(), 1);
    assert_eq!(requests(&server, "GET", "/v1/files/file_1").await.len(), 1);
    assert_eq!(sent_file(&server, "file_1").await, 3);
    // Ruby also counts zero Active Storage downloads; the bytes live in `rust_llm_attachments`
    // here, so there is no storage service to download from.
}

// spec: active_record/provider_file_spec.rb:108 uploads again when the provider no longer has the file
#[tokio::test]
async fn uploads_again_when_the_provider_no_longer_has_the_file() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1", "file_2"]).await;
    stub_lookup(&server, "file_1", 404).await;
    let config = config(&server, "pf-108");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;
    new_process();

    ask_again(&db, &record, &config, "And the action items?").await;

    assert_eq!(sent_file(&server, "file_2").await, 1);
    let stored = rows(&db).await;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].file_id, "file_2");
}

// spec: active_record/provider_file_spec.rb:121 uploads again once the stored file has expired
#[tokio::test]
async fn uploads_again_once_the_stored_file_has_expired() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1", "file_2"]).await;
    stub_lookup(&server, "file_1", 200).await;
    let config = config(&server, "pf-121");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;
    let an_hour_ago: chrono::DateTime<chrono::FixedOffset> =
        (chrono::Utc::now() - chrono::Duration::hours(1)).into();
    rust_llm_provider_files::Entity::update_many()
        .col_expr(
            rust_llm_provider_files::Column::ExpiresAt,
            sea_orm::sea_query::Expr::value(an_hour_ago),
        )
        .exec(&db)
        .await
        .unwrap();

    ask_again(&db, &record, &config, "And the action items?").await;

    assert_eq!(requests(&server, "POST", "/v1/files").await.len(), 2);
    assert!(
        requests(&server, "GET", "/v1/files/file_1")
            .await
            .is_empty()
    );
    let stored = rows(&db).await;
    assert_eq!(stored.len(), 1);
    assert_eq!(
        (stored[0].file_id.as_str(), stored[0].expires_at),
        ("file_2", None)
    );
}

// spec: active_record/provider_file_spec.rb:135 keeps the uploads of each account apart
#[tokio::test]
async fn keeps_the_uploads_of_each_account_apart() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1", "file_2"]).await;
    let config = config(&server, "pf-135");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;
    let other = self::config(&server, "other-tenant");

    ask_again(&db, &record, &other, "And the action items?").await;

    assert_eq!(requests(&server, "POST", "/v1/files").await.len(), 2);
    let mut ids: Vec<String> = rows(&db).await.into_iter().map(|r| r.file_id).collect();
    ids.sort();
    assert_eq!(ids, ["file_1", "file_2"]);
}

// spec: active_record/provider_file_spec.rb:147 records a stored blob passed to ask against that blob
#[tokio::test]
async fn records_a_stored_blob_passed_to_ask_against_that_blob() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1"]).await;
    let config = config(&server, "pf-147");
    let blob = stored_blob(&db, &config, "Shared notes long enough to upload.").await;
    let blob_key = attachment_rows(&db).await[0].blob_key.clone().unwrap();

    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(&db, &record, &config, "Summarize these notes", vec![blob]).await;

    let stored = rows(&db).await;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].blob_key, blob_key);
}

// spec: active_record/provider_file_spec.rb:157 replaces the recorded upload when the provider deletes it mid-process
#[tokio::test]
async fn replaces_the_recorded_upload_when_the_provider_deletes_it_mid_process() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = MockServer::start().await;
    in_order(
        &server,
        "POST",
        "/v1/messages",
        vec![
            json_response(reply(), 200),
            json_response(
                json!({ "type": "error", "error": { "type": "not_found_error", "message": "File not found: file_1" } }),
                404,
            ),
            json_response(reply(), 200),
        ],
    )
    .await;
    in_order(
        &server,
        "POST",
        "/v1/files",
        vec![
            json_response(file_body("file_1"), 200),
            json_response(file_body("file_2"), 200),
        ],
    )
    .await;
    let config = config(&server, "pf-157");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;

    ask_again(&db, &record, &config, "And the action items?").await;

    assert_eq!(sent_file(&server, "file_2").await, 1);
    assert!(
        requests(&server, "GET", "/v1/files/file_1")
            .await
            .is_empty()
    );
    let stored = rows(&db).await;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].file_id, "file_2");
}

// spec: active_record/provider_file_spec.rb:174 never reuses the upload of a deleted blob for a blob that takes its id
#[tokio::test]
async fn never_reuses_the_upload_of_a_deleted_blob_for_a_blob_that_takes_its_id() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1", "file_2"]).await;
    let config = config(&server, "pf-174");
    let deleted = stored_blob(&db, &config, "First notes, long enough to upload.").await;
    let deleted_id = attachment_rows(&db).await[0].id;
    let first = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(&db, &first, &config, "Summarize these notes", vec![deleted]).await;
    // `delete_all` skips the destroy hook, so the deleted blob's upload row stays behind.
    rust_llm_attachments::Entity::delete_by_id(deleted_id)
        .exec(&db)
        .await
        .unwrap();
    let replacement = stored_blob(&db, &config, "Other notes, long enough to upload.").await;
    let replacement_id = attachment_rows(&db).await.last().unwrap().id;
    db.execute_unprepared(&format!(
        "UPDATE rust_llm_attachments SET id = {deleted_id} WHERE id = {replacement_id}"
    ))
    .await
    .unwrap();

    let second = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &second,
        &config,
        "Summarize these notes",
        vec![replacement],
    )
    .await;

    assert_eq!(sent_file(&server, "file_2").await, 1);
}

// spec: active_record/provider_file_spec.rb:188 forgets the uploads of a blob Active Storage purges
#[tokio::test]
async fn forgets_the_uploads_of_a_purged_blob() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1"]).await;
    let config = config(&server, "pf-188");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;
    let first = record.messages(&db).await.unwrap()[0].id;

    purge_attachments(&db, vec![i64::from(first)])
        .await
        .unwrap();

    assert_eq!(
        rust_llm_provider_files::Entity::find()
            .count(&db)
            .await
            .unwrap(),
        0
    );
    assert!(attachment_rows(&db).await.is_empty());
}

// spec: active_record/provider_file_spec.rb:198 purges blobs before the table exists
#[tokio::test]
async fn purges_blobs_before_the_table_exists() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    let server = server_with_uploads(&["file_1"]).await;
    let config = config(&server, "pf-198");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;
    let first = record.messages(&db).await.unwrap()[0].id;
    db.execute_unprepared("DROP TABLE rust_llm_provider_files")
        .await
        .unwrap();

    purge_attachments(&db, vec![i64::from(first)])
        .await
        .unwrap();

    assert!(attachment_rows(&db).await.is_empty());
}

// spec: active_record/provider_file_spec.rb:211 keeps uploads in memory until the table exists
#[tokio::test]
async fn keeps_uploads_in_memory_until_the_table_exists() {
    let _serial = SERIAL.lock().await;
    let db = db().await;
    db.execute_unprepared("DROP TABLE rust_llm_provider_files")
        .await
        .unwrap();
    let server = server_with_uploads(&["file_1", "file_2"]).await;
    let config = config(&server, "pf-211");
    let record = ChatRecord::create(&db, MODEL, None).await.unwrap();
    ask(
        &db,
        &record,
        &config,
        "Summarize these notes",
        vec![notes()],
    )
    .await;

    ask_again(&db, &record, &config, "And the action items?").await;

    assert_eq!(requests(&server, "POST", "/v1/files").await.len(), 2);
    // Nothing was written anywhere: the table is still absent.
    assert!(
        !SchemaManager::new(&db)
            .has_table("rust_llm_provider_files")
            .await
            .unwrap()
    );
    // The blob rows were still stored.
    assert_eq!(
        rust_llm_attachments::Entity::find()
            .filter(rust_llm_attachments::Column::BlobKey.is_not_null())
            .count(&db)
            .await
            .unwrap(),
        1
    );
}
