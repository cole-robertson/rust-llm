//! Port of `lib/ruby_llm/active_record/provider_file.rb`: the record of the provider files each
//! stored attachment was uploaded to (`rust_llm_provider_files`), so a chat loaded in another
//! process sends the file it already uploaded instead of uploading the bytes again. Rows name the
//! attachment by its `rust_llm_attachments.blob_key`, which is never handed out twice, so a row
//! left by a deleted attachment can never match a newer one.

use std::sync::Arc;

use rust_llm::UploadedFile;
use rust_llm::files::ProviderFileStore;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, ConnectionTrait, DatabaseConnection,
    EntityTrait, QueryFilter,
};

use crate::entities::rust_llm_provider_files;

/// `RubyLLM::ActiveRecord::ProviderFile`.
pub struct ProviderFile;

impl ProviderFile {
    /// `ProviderFile.store { blob_key }`: a store for the attachment whose blob key this is, or
    /// `None` while the table has not been migrated (uploads then stay in memory).
    pub async fn store(
        db: &DatabaseConnection,
        blob_key: Option<String>,
    ) -> Option<Arc<dyn ProviderFileStore>> {
        if !table_exists(db).await {
            return None;
        }
        Some(store_for(db, blob_key))
    }

    /// `ProviderFile.forget_blob(blob)`: deletes the uploads recorded for a blob as it is
    /// destroyed (the `after_destroy` hook on Active Storage blobs). A schema without the table has
    /// nothing to forget.
    pub async fn forget_blob(db: &impl ConnectionTrait, blob_key: &str) -> crate::Result<()> {
        if !table_exists(db).await {
            return Ok(());
        }
        rust_llm_provider_files::Entity::delete_many()
            .filter(rust_llm_provider_files::Column::BlobKey.eq(blob_key))
            .exec(db)
            .await?;
        Ok(())
    }
}

/// The store of one blob, once the table is known to exist.
pub(crate) fn store_for(
    db: &DatabaseConnection,
    blob_key: Option<String>,
) -> Arc<dyn ProviderFileStore> {
    Arc::new(Store {
        db: db.clone(),
        blob_key,
    })
}

/// `table_exists?`.
pub(crate) async fn table_exists(db: &impl ConnectionTrait) -> bool {
    let backend = db.get_database_backend();
    let sql = match backend {
        sea_orm::DbBackend::Sqlite => {
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rust_llm_provider_files'"
        }
        _ => "SELECT 1 FROM information_schema.tables WHERE table_name = 'rust_llm_provider_files'",
    };
    db.query_one_raw(sea_orm::Statement::from_string(backend, sql))
        .await
        .is_ok_and(|row| row.is_some())
}

/// `ProviderFile::Store`: the uploads of one blob. A write failure is logged, since the upload it
/// would record already succeeded.
struct Store {
    db: DatabaseConnection,
    blob_key: Option<String>,
}

impl Store {
    fn scoped(
        &self,
        provider: &str,
        account: &str,
    ) -> Option<sea_orm::Select<rust_llm_provider_files::Entity>> {
        let key = self.blob_key.as_ref()?;
        Some(
            rust_llm_provider_files::Entity::find()
                .filter(rust_llm_provider_files::Column::BlobKey.eq(key.as_str()))
                .filter(rust_llm_provider_files::Column::Provider.eq(provider))
                .filter(rust_llm_provider_files::Column::Account.eq(account)),
        )
    }
}

#[async_trait::async_trait]
impl ProviderFileStore for Store {
    async fn fetch(&self, provider: &str, account: &str) -> Option<UploadedFile> {
        let row = self.scoped(provider, account)?.one(&self.db).await.ok()??;
        Some(UploadedFile {
            id: row.file_id,
            provider: provider.to_string(),
            filename: None,
            byte_size: None,
            created_at: None,
            expires_at: row.expires_at.map(|t| t.with_timezone(&chrono::Utc)),
            status: None,
            mime_type: None,
            purpose: None,
            uri: None,
            downloadable: None,
            metadata: serde_json::Value::Null,
        })
    }

    /// `find_or_initialize_by(blob_key:, provider:, account:).update!(file_id:, expires_at:)`.
    /// Another process recording an upload of the same blob first is fine: its file serves too.
    async fn store(&self, upload: &UploadedFile, provider: &str, account: &str) {
        let Some(key) = &self.blob_key else { return };
        let expires_at = upload.expires_at.map(Into::into);
        let result = match self.scoped(provider, account) {
            Some(scope) => scope.one(&self.db).await,
            None => return,
        };
        let now = crate::now();
        let written = match result {
            Ok(Some(row)) => {
                let mut row: rust_llm_provider_files::ActiveModel = row.into();
                row.file_id = Set(upload.id.clone());
                row.expires_at = Set(expires_at);
                row.updated_at = Set(now);
                row.update(&self.db).await.map(|_| ())
            }
            Ok(None) => rust_llm_provider_files::ActiveModel {
                blob_key: Set(key.clone()),
                provider: Set(provider.to_string()),
                account: Set(account.to_string()),
                file_id: Set(upload.id.clone()),
                expires_at: Set(expires_at),
                created_at: Set(now),
                updated_at: Set(now),
                ..Default::default()
            }
            .insert(&self.db)
            .await
            .map(|_| ()),
            Err(e) => Err(e),
        };
        match written {
            Err(e)
                if matches!(
                    e.sql_err(),
                    Some(sea_orm::SqlErr::UniqueConstraintViolation(_))
                ) => {}
            Err(e) => tracing::warn!("RustLLM could not record the upload of {key}: {e}"),
            Ok(()) => {}
        }
    }

    /// `forget`: only a row that still names the missing file goes, since another process may
    /// have recorded a newer upload meanwhile.
    async fn forget(&self, id: &str, provider: &str, account: &str) {
        let Some(key) = &self.blob_key else { return };
        if let Err(e) = rust_llm_provider_files::Entity::delete_many()
            .filter(rust_llm_provider_files::Column::BlobKey.eq(key.as_str()))
            .filter(rust_llm_provider_files::Column::Provider.eq(provider))
            .filter(rust_llm_provider_files::Column::Account.eq(account))
            .filter(rust_llm_provider_files::Column::FileId.eq(id))
            .exec(&self.db)
            .await
        {
            tracing::warn!("RustLLM could not forget the upload of {key}: {e}");
        }
    }
}
