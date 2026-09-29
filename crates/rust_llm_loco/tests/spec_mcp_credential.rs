//! `McpCredentialStore`, mirroring RubyLLM's `spec/ruby_llm/active_record/mcp_credential_spec.rb`
//! on SQLite.

use std::sync::Arc;

use rust_llm::mcp::{CredentialStore, Mcp, OAuthSettings};
use rust_llm_loco::{ChatRecord, McpCredentialStore, migrations};
use sea_orm::{ConnectionTrait, Database, DatabaseConnection, Statement};
use sea_orm_migration::SchemaManager;
use serde_json::json;

const KEY: [u8; 32] = [7; 32];

async fn db() -> DatabaseConnection {
    rust_llm::configure(|c| {
        c.set("openai_api_key", "test");
    });
    let db = Database::connect("sqlite::memory:").await.unwrap();
    let manager = SchemaManager::new(&db);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    db
}

// spec: active_record/mcp_credential_spec.rb:8 is the credential store in Rails
// Rails installs the store from its Railtie; a Loco app sets it in its configuration, and an MCP
// built with that configuration keeps its credentials in the table.
#[tokio::test]
async fn is_the_credential_store_an_mcp_uses() {
    let db = db().await;
    let store = Arc::new(McpCredentialStore::new(db.clone(), KEY));
    let mut config = (*rust_llm::config()).clone();
    config.mcp_credential_store = Some(store.clone());
    let linear = Mcp::url("https://mcp.linear.app/mcp").oauth(OAuthSettings::new().owner("ada")).config(Arc::new(config)).build().unwrap();

    store.write("ada@https://mcp.linear.app/mcp", json!({ "access_token": "secret-token" }), None).await.unwrap();

    assert!(linear.is_authorized().await.unwrap());
    linear.deauthorize().await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}

// spec: active_record/mcp_credential_spec.rb:12 keeps credentials encrypted, with their owner
#[tokio::test]
async fn keeps_credentials_encrypted_with_their_owner() {
    let db = db().await;
    let store = McpCredentialStore::new(db.clone(), KEY);
    let owner = ChatRecord::create(&db, "gpt-4.1-nano", None).await.unwrap();
    let owner_key = McpCredentialStore::owner_key(rust_llm_loco::CHAT_TYPE, owner.id() as i64);

    store.write("key", json!({ "access_token": "secret-token" }), Some(&owner_key)).await.unwrap();

    assert_eq!(store.read("key").await.unwrap(), Some(json!({ "access_token": "secret-token" })));
    let record = store.find_by_key("key").await.unwrap().unwrap();
    assert_eq!((record.owner_type.as_deref(), record.owner_id), (Some("Chat"), Some(owner.id() as i64)));
    let raw = db
        .query_one_raw(Statement::from_string(db.get_database_backend(), "SELECT data FROM rust_llm_mcp_credentials WHERE key = 'key'"))
        .await
        .unwrap()
        .unwrap();
    let data: String = raw.try_get("", "data").unwrap();
    assert!(!data.contains("secret-token"), "{data}");
    assert!(McpCredentialStore::new(db.clone(), [8; 32]).read("key").await.is_err(), "another key must not decrypt");
}

// spec: active_record/mcp_credential_spec.rb:21 replaces and deletes credentials
#[tokio::test]
async fn replaces_and_deletes_credentials() {
    let db = db().await;
    let store = McpCredentialStore::new(db, KEY);
    store.write("key", json!({ "access_token": "first" }), None).await.unwrap();
    store.write("key", json!({ "access_token": "second" }), None).await.unwrap();

    assert_eq!(store.read("key").await.unwrap(), Some(json!({ "access_token": "second" })));
    let before = store.count().await.unwrap();
    store.delete("key").await.unwrap();
    assert_eq!(store.count().await.unwrap(), before - 1);
}
