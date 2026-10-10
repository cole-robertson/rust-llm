//! Port of `lib/ruby_llm/active_record/mcp_credential.rb`: RustLLM's persistence for MCP OAuth
//! credentials, in `rust_llm_mcp_credentials`, encrypted.
//!
//! ```ruby
//! RubyLLM.config.mcp_credential_store # => RubyLLM::ActiveRecord::MCPCredential (set by the Railtie)
//! ```
//!
//! ```no_run
//! # async fn run(db: sea_orm::DatabaseConnection, key: [u8; 32]) {
//! let store = rust_llm_loco::McpCredentialStore::new(db, key);
//! rust_llm::configure(|c| c.mcp_credential_store = Some(std::sync::Arc::new(store)));
//! # }
//! ```
//!
//! Rails encrypts `data` with Active Record encryption (AES-256-GCM, a random IV per write, and
//! the `{"p": ciphertext, "h": {"iv":, "at":}}` message layout). Loco has no equivalent, so the
//! store takes the 32-byte key itself and writes the same layout. The owner is the polymorphic
//! `owner_type`/`owner_id` of a record: pass an owner such as `gid://app/Chat/1` (a GlobalID, as
//! Ruby's `owner.to_gid`), or [`McpCredentialStore::owner_key`]; other owners (plain strings)
//! leave the columns empty, like Ruby's `owner.is_a?(ActiveRecord::Base) ? owner : nil`.
//!
//! `synchronize(key)` touches the row in a transaction, which locks it on every database (SQLite
//! takes its write lock), so one process refreshes a grant while the others wait for its token.
//! The store's own reads and writes inside the block run in that transaction, as Active Record's
//! do on the transaction's connection.

use std::sync::Arc;

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rust_llm::mcp::{CredentialStore, McpError, Synchronized};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, DatabaseTransaction,
    EntityTrait, QueryFilter, TransactionTrait,
};
use serde_json::{Value, json};

use crate::entities::rust_llm_mcp_credentials;

const TAG_LENGTH: usize = 16;

tokio::task_local! {
    /// The transaction `synchronize` holds while its block runs.
    static TRANSACTION: Arc<DatabaseTransaction>;
}

/// Runs `$body` with `$conn` bound to the transaction of a running `synchronize`, or the store's
/// connection.
macro_rules! on_connection {
    ($self:ident, |$conn:ident| $body:expr) => {
        match TRANSACTION.try_with(Arc::clone) {
            Ok(transaction) => {
                let $conn = transaction.as_ref();
                $body
            }
            Err(_) => {
                let $conn = &$self.db;
                $body
            }
        }
    };
}

/// `RubyLLM::ActiveRecord::MCPCredential` as a [`CredentialStore`].
#[derive(Clone)]
pub struct McpCredentialStore {
    db: DatabaseConnection,
    cipher: Aes256Gcm,
}

impl std::fmt::Debug for McpCredentialStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("McpCredentialStore")
    }
}

fn failure(message: impl std::fmt::Display) -> rust_llm::Error {
    McpError::new(format!("MCP credentials: {message}")).into()
}

impl McpCredentialStore {
    /// A store in `db`, encrypting with the 32-byte `key` (`active_record_encryption.primary_key`).
    pub fn new(db: DatabaseConnection, key: [u8; 32]) -> McpCredentialStore {
        McpCredentialStore {
            db,
            cipher: Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key)),
        }
    }

    /// The owner key of a record, `gid://rust_llm/<type>/<id>`, which the store keeps as its
    /// polymorphic owner.
    pub fn owner_key(owner_type: &str, owner_id: i64) -> String {
        format!("gid://rust_llm/{owner_type}/{owner_id}")
    }

    /// `find_by(key:)`, for reading the owner and the stored ciphertext.
    pub async fn find_by_key(
        &self,
        key: &str,
    ) -> Result<Option<rust_llm_mcp_credentials::Model>, sea_orm::DbErr> {
        let query = rust_llm_mcp_credentials::Entity::find()
            .filter(rust_llm_mcp_credentials::Column::Key.eq(key));
        on_connection!(self, |conn| query.one(conn).await)
    }

    /// `count`.
    pub async fn count(&self) -> Result<u64, sea_orm::DbErr> {
        use sea_orm::PaginatorTrait;
        rust_llm_mcp_credentials::Entity::find()
            .count(&self.db)
            .await
    }

    fn encrypt(&self, data: &Value) -> rust_llm::Result<String> {
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let mut sealed = self
            .cipher
            .encrypt(&nonce, data.to_string().as_bytes())
            .map_err(failure)?;
        let tag = sealed.split_off(sealed.len() - TAG_LENGTH);
        Ok(json!({ "p": STANDARD.encode(sealed), "h": { "iv": STANDARD.encode(nonce), "at": STANDARD.encode(tag) } }).to_string())
    }

    fn decrypt(&self, stored: &str) -> rust_llm::Result<Value> {
        let message: Value = serde_json::from_str(stored).map_err(failure)?;
        let field = |pointer: &str| {
            message
                .pointer(pointer)
                .and_then(Value::as_str)
                .and_then(|v| STANDARD.decode(v).ok())
                .ok_or_else(|| failure("unreadable"))
        };
        let (mut sealed, iv, tag) = (field("/p")?, field("/h/iv")?, field("/h/at")?);
        if iv.len() != 12 {
            return Err(failure("unreadable"));
        }
        sealed.extend(tag);
        let plain = self
            .cipher
            .decrypt(Nonce::from_slice(&iv), sealed.as_slice())
            .map_err(failure)?;
        serde_json::from_slice(&plain).map_err(failure)
    }
}

/// `owner.is_a?(ActiveRecord::Base) ? owner : nil`: a GlobalID names a record.
fn polymorphic_owner(owner: Option<&str>) -> (Option<String>, Option<i64>) {
    let Some(path) = owner.and_then(|o| o.strip_prefix("gid://")) else {
        return (None, None);
    };
    let mut parts = path.splitn(3, '/').skip(1);
    match (parts.next(), parts.next().and_then(|id| id.parse().ok())) {
        (Some(owner_type), Some(id)) if !owner_type.is_empty() => {
            (Some(owner_type.to_string()), Some(id))
        }
        _ => (None, None),
    }
}

#[async_trait]
impl CredentialStore for McpCredentialStore {
    /// `find_by(key:)&.data`.
    async fn read(&self, key: &str) -> rust_llm::Result<Option<Value>> {
        let record = self.find_by_key(key).await.map_err(failure)?;
        record
            .and_then(|r| r.data)
            .map(|data| self.decrypt(&data))
            .transpose()
    }

    /// `find_or_initialize_by(key:).update!(data:, owner:)`.
    async fn write(&self, key: &str, data: Value, owner: Option<&str>) -> rust_llm::Result<()> {
        let (owner_type, owner_id) = polymorphic_owner(owner);
        let now: sea_orm::prelude::DateTimeWithTimeZone = chrono::Utc::now().into();
        let encrypted = self.encrypt(&data)?;
        let result = match self.find_by_key(key).await.map_err(failure)? {
            Some(existing) => {
                let mut record: rust_llm_mcp_credentials::ActiveModel = existing.into();
                record.data = Set(Some(encrypted));
                record.owner_type = Set(owner_type);
                record.owner_id = Set(owner_id);
                record.updated_at = Set(now);
                on_connection!(self, |conn| record.update(conn).await.map(|_| ()))
            }
            None => {
                let record = rust_llm_mcp_credentials::ActiveModel {
                    key: Set(key.to_string()),
                    data: Set(Some(encrypted)),
                    owner_type: Set(owner_type),
                    owner_id: Set(owner_id),
                    created_at: Set(now),
                    updated_at: Set(now),
                    ..Default::default()
                };
                on_connection!(self, |conn| record.insert(conn).await.map(|_| ()))
            }
        };
        result.map_err(failure)
    }

    /// `where(key:).delete_all`.
    async fn delete(&self, key: &str) -> rust_llm::Result<()> {
        let query = rust_llm_mcp_credentials::Entity::delete_many()
            .filter(rust_llm_mcp_credentials::Column::Key.eq(key));
        on_connection!(self, |conn| query.exec(conn).await)
            .map(|_| ())
            .map_err(failure)
    }

    /// `synchronize(key)`: `transaction { where(key:).touch_all; yield }`. Updating the row locks
    /// it on every database; SQLite ignores `FOR UPDATE`. The block's error rolls it back.
    async fn synchronize<'a>(
        &'a self,
        key: &'a str,
        block: Synchronized<'a>,
    ) -> rust_llm::Result<bool> {
        if TRANSACTION.try_with(|_| ()).is_ok() {
            return block.await;
        }
        let transaction = self.db.begin().await.map_err(failure)?;
        let now: sea_orm::prelude::DateTimeWithTimeZone = chrono::Utc::now().into();
        rust_llm_mcp_credentials::Entity::update_many()
            .col_expr(
                rust_llm_mcp_credentials::Column::UpdatedAt,
                Expr::value(now),
            )
            .filter(rust_llm_mcp_credentials::Column::Key.eq(key))
            .exec(&transaction)
            .await
            .map_err(failure)?;
        let transaction = Arc::new(transaction);
        let result = TRANSACTION.scope(transaction.clone(), block).await;
        let transaction = Arc::into_inner(transaction)
            .ok_or_else(|| failure("the transaction outlived its block"))?;
        match result {
            Ok(value) => {
                transaction.commit().await.map_err(failure)?;
                Ok(value)
            }
            Err(e) => {
                let _ = transaction.rollback().await;
                Err(e)
            }
        }
    }
}
