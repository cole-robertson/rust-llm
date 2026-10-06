
// ---- accounts (`rust-llm generate chat_ui`) ------------------------------------------------
// A chat belongs to an account through `chats.account_id` (the chat UI's migration). The chat UI
// reads and writes chats only through these, so another account's chat is never found: a 404.

/// `chats.account_id`.
fn account_column() -> sea_orm::sea_query::Alias {
    sea_orm::sea_query::Alias::new("account_id")
}

/// `Current.account.chats.create!(model:)`.
///
/// # Errors
/// An unknown model (`rust_llm_loco::Error::Llm`) or a database error; a chat that could not be
/// given its account is destroyed again, so no chat is left without one.
pub async fn create_in_account(
    db: &DatabaseConnection,
    account_id: i64,
    model: &str,
    provider: Option<&str>,
) -> rust_llm_loco::Result<ChatRecord> {
    let record = ChatRecord::create(db, model, provider).await?;
    let assigned = Entity::update_many()
        .col_expr(account_column(), Expr::value(account_id))
        .filter(Column::Id.eq(record.id()))
        .exec(db)
        .await;
    if let Err(e) = assigned {
        let id = record.id();
        if let Err(cleanup) = record.destroy(db).await {
            tracing::error!(chat_id = id, error = %cleanup, "could not remove a chat without an account");
        }
        return Err(e.into());
    }
    Ok(record)
}

/// `Current.account.chats.find_by(id:)`: `None` for a chat of another account.
///
/// # Errors
/// Database errors.
pub async fn find_in_account(
    db: &DatabaseConnection,
    account_id: i64,
    id: i32,
) -> Result<Option<Model>, DbErr> {
    Entity::find_by_id(id)
        .filter(Expr::col(account_column()).eq(account_id))
        .one(db)
        .await
}

/// `Current.account.chats.order(created_at: :desc)` as page props, each with its model and
/// message count.
///
/// # Errors
/// Database errors.
pub async fn list_in_account(db: &DatabaseConnection, account_id: i64) -> Result<Vec<Value>, DbErr> {
    let chats = Entity::find()
        .filter(Expr::col(account_column()).eq(account_id))
        .order_by_desc(Column::CreatedAt)
        .order_by_desc(Column::Id)
        .all(db)
        .await?;
    let chat_ids: Vec<i32> = chats.iter().map(|c| c.id).collect();
    let model_ids: Vec<i32> = chats.iter().map(|c| c.rust_llm_model_id).collect();
    let models = rust_llm_models::Entity::find()
        .filter(rust_llm_models::Column::Id.is_in(model_ids))
        .all(db)
        .await?;
    let counts: Vec<(i32, i64)> = messages::Entity::find()
        .select_only()
        .column(messages::Column::ChatId)
        .column_as(Expr::col(messages::Column::Id).count(), "count")
        .filter(messages::Column::ChatId.is_in(chat_ids))
        .group_by(messages::Column::ChatId)
        .into_tuple()
        .all(db)
        .await?;
    Ok(chats
        .iter()
        .map(|chat| {
            let model = models.iter().find(|m| m.id == chat.rust_llm_model_id);
            let count = counts
                .iter()
                .find(|(id, _)| *id == chat.id)
                .map_or(0, |(_, n)| *n);
            json!({
                "id": chat.id,
                "model_label": model.map(model_label),
                "message_count": count,
                "created_at": chat.created_at.to_rfc3339(),
            })
        })
        .collect())
}
