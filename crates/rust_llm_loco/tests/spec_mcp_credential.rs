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
    let linear = Mcp::url("https://mcp.linear.app/mcp")
        .oauth(OAuthSettings::new().owner("ada"))
        .config(Arc::new(config))
        .build()
        .unwrap();

    store
        .write(
            "ada@https://mcp.linear.app/mcp",
            json!({ "access_token": "secret-token" }),
            None,
        )
        .await
        .unwrap();

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

    store
        .write(
            "key",
            json!({ "access_token": "secret-token" }),
            Some(&owner_key),
        )
        .await
        .unwrap();

    assert_eq!(
        store.read("key").await.unwrap(),
        Some(json!({ "access_token": "secret-token" }))
    );
    let record = store.find_by_key("key").await.unwrap().unwrap();
    assert_eq!(
        (record.owner_type.as_deref(), record.owner_id),
        (Some("Chat"), Some(owner.id() as i64))
    );
    let raw = db
        .query_one_raw(Statement::from_string(
            db.get_database_backend(),
            "SELECT data FROM rust_llm_mcp_credentials WHERE key = 'key'",
        ))
        .await
        .unwrap()
        .unwrap();
    let data: String = raw.try_get("", "data").unwrap();
    assert!(!data.contains("secret-token"), "{data}");
    assert!(
        McpCredentialStore::new(db.clone(), [8; 32])
            .read("key")
            .await
            .is_err(),
        "another key must not decrypt"
    );
}

// spec: active_record/mcp_credential_spec.rb:72 replaces and deletes credentials
#[tokio::test]
async fn replaces_and_deletes_credentials() {
    let db = db().await;
    let store = McpCredentialStore::new(db, KEY);
    store
        .write("key", json!({ "access_token": "first" }), None)
        .await
        .unwrap();
    store
        .write("key", json!({ "access_token": "second" }), None)
        .await
        .unwrap();

    assert_eq!(
        store.read("key").await.unwrap(),
        Some(json!({ "access_token": "second" }))
    );
    let before = store.count().await.unwrap();
    store.delete("key").await.unwrap();
    assert_eq!(store.count().await.unwrap(), before - 1);
}

/// A file-backed database two connections share, like the forked workers of the Ruby example.
async fn shared_db(name: &str) -> (DatabaseConnection, DatabaseConnection, std::path::PathBuf) {
    let file = std::env::temp_dir().join(format!("rust_llm_{name}_{}.sqlite3", std::process::id()));
    let _ = std::fs::remove_file(&file);
    let url = format!("sqlite://{}?mode=rwc", file.display());
    let first = Database::connect(&url).await.unwrap();
    let manager = SchemaManager::new(&first);
    for m in migrations() {
        m.up(&manager).await.unwrap();
    }
    let second = Database::connect(&url).await.unwrap();
    (first, second, file)
}

/// The token endpoint: counts refreshes, answers each after 200 ms.
struct SlowRefresh {
    refreshes: Arc<std::sync::atomic::AtomicUsize>,
}

impl wiremock::Respond for SlowRefresh {
    fn respond(&self, _request: &wiremock::Request) -> wiremock::ResponseTemplate {
        self.refreshes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        wiremock::ResponseTemplate::new(200)
            .set_body_json(json!({ "access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600 }))
            .set_delay(std::time::Duration::from_millis(200))
    }
}

// spec: active_record/mcp_credential_spec.rb:21 lets one process refresh a grant while the others wait for its token
// Ruby forks two workers that each refresh through the table; here two OAuth clients refresh
// concurrently through two stores on separate connections to one SQLite file. They also share
// this process's per-key lock, which Ruby's forks would not, but that lock alone is not what
// passes this: with it removed the row lock `synchronize` takes still keeps it to one refresh,
// and with both removed the second worker refreshes again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lets_one_process_refresh_a_grant_while_the_others_wait_for_its_token() {
    use rust_llm::mcp::OAuth;

    let (first, second, file) = shared_db("refresh").await;
    let auth = wiremock::MockServer::start().await;
    let refreshes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/token"))
        .respond_with(SlowRefresh {
            refreshes: refreshes.clone(),
        })
        .mount(&auth)
        .await;
    let owner = McpCredentialStore::owner_key(rust_llm_loco::CHAT_TYPE, 1);
    let server_url = "http://127.0.0.1:1/mcp";
    let key = format!("{owner}@{server_url}");
    let stores = [
        Arc::new(McpCredentialStore::new(first, KEY)),
        Arc::new(McpCredentialStore::new(second, KEY)),
    ];
    stores[0]
        .write(
            &key,
            json!({
                "access_token": "access-1", "refresh_token": "refresh-1", "expires_at": 0,
                "client_id": "client", "issuer": auth.uri(),
                "server": { "issuer": auth.uri(), "token_endpoint": format!("{}/token", auth.uri()) }
            }),
            Some(&owner),
        )
        .await
        .unwrap();

    let worker = |store: &Arc<McpCredentialStore>| {
        let mut config = (*rust_llm::config()).clone();
        config.mcp_credential_store = Some(store.clone());
        let oauth = OAuth::new(
            server_url,
            Some(owner.clone()),
            None,
            None,
            None,
            Arc::new(config),
        )
        .unwrap();
        tokio::spawn(async move { oauth.access_token().await.unwrap() })
    };
    let (one, two) = (worker(&stores[0]), worker(&stores[1]));
    let tokens = [one.await.unwrap(), two.await.unwrap()];

    assert_eq!(
        tokens,
        [Some("access-2".to_string()), Some("access-2".to_string())]
    );
    assert_eq!(refreshes.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        stores[0].read(&key).await.unwrap().unwrap()["refresh_token"],
        "refresh-2"
    );
    let _ = std::fs::remove_file(file);
}

/// A P-256 key, as `OpenSSL::PKey::EC.generate('prime256v1').private_to_pem`, and the base64url
/// X coordinate of its public point.
const EC_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgT3wOt+pNOqFCTSW4\nobzAWeGUx5YMPT9yXPhJOQX9Ix6hRANCAASualQCGPmiMrhMn86Cx4y1PDHdT6UX\npxher9ER5Vv5x9fQa4XsUpssH9POeC+VdMQnrevJRZ1ti8potRDWZBqf\n-----END PRIVATE KEY-----";
const EC_KEY_X: &str = "rmpUAhj5ojK4TJ_OgseMtTwx3U-lF6cYXq_REeVb-cc";

// spec: active_record/mcp_credential_spec.rb:56 keeps the key of DPoP-bound tokens with them, encrypted
#[tokio::test]
async fn keeps_the_key_of_dpop_bound_tokens_with_them_encrypted() {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use rust_llm::mcp::OAuth;

    let db = db().await;
    let store = Arc::new(McpCredentialStore::new(db.clone(), KEY));
    let owner = McpCredentialStore::owner_key(rust_llm_loco::CHAT_TYPE, 1);
    let server_url = "https://mcp.example.com/mcp";
    store
        .write(
            &format!("{owner}@{server_url}"),
            json!({ "access_token": "access-1", "token_type": "DPoP", "dpop_key": EC_KEY }),
            Some(&owner),
        )
        .await
        .unwrap();
    let mut config = (*rust_llm::config()).clone();
    config.mcp_credential_store = Some(store.clone());

    let headers = OAuth::new(server_url, Some(owner), None, None, None, Arc::new(config))
        .unwrap()
        .authorization_headers("POST")
        .await
        .unwrap();

    let header = |name: &str| {
        headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .unwrap()
    };
    let proof_header = header("DPoP").split('.').next().unwrap().to_string();
    let jwk: serde_json::Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(proof_header).unwrap()).unwrap();
    assert_eq!(header("Authorization"), "DPoP access-1");
    assert_eq!(jwk["jwk"]["x"], EC_KEY_X);
    let raw = db
        .query_one_raw(Statement::from_string(
            db.get_database_backend(),
            "SELECT data FROM rust_llm_mcp_credentials",
        ))
        .await
        .unwrap()
        .unwrap();
    let data: String = raw.try_get("", "data").unwrap();
    assert!(!data.contains("PRIVATE KEY"), "{data}");
}
