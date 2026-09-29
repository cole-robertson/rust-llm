//! `ChatResponseJob` (RubyLLM's chat_ui `chat_response_job.rb.tt`) as a Loco worker. Generated
//! by `rust-llm generate chat_ui`.
//!
//! RubyLLM's job calls `chat.ask(content)` and broadcasts each chunk over Turbo. Here the
//! controller stores the user message first (`ask_later`) and this worker runs `complete`, which
//! persists every step (tool calls, tool results, the answer) as it lands. The chat page polls
//! for them; there is no token-by-token streaming.

use loco_rs::prelude::*;
use sea_orm::DatabaseConnection;
use serde::{Deserialize, Serialize};

use crate::models::chats::ChatRecord;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseArgs {
    pub chat_id: i32,
}

pub struct ChatResponseWorker {
    pub ctx: AppContext,
}

#[async_trait]
impl BackgroundWorker<ChatResponseArgs> for ChatResponseWorker {
    fn build(ctx: &AppContext) -> Self {
        Self { ctx: ctx.clone() }
    }

    async fn perform(&self, args: ChatResponseArgs) -> Result<()> {
        respond(&self.ctx.db, args.chat_id).await
    }
}

/// Reloads the chat from its rows and completes it. A chat deleted in the meantime is skipped;
/// a failed completion is logged and fails the job.
async fn respond(db: &DatabaseConnection, chat_id: i32) -> Result<()> {
    let record = match ChatRecord::find(db, chat_id).await {
        Ok(record) => record,
        Err(rust_llm_loco::Error::NotFound(_)) => {
            tracing::info!(chat_id, "chat is gone; not responding");
            return Ok(());
        }
        Err(e) => return Err(Error::wrap(e)),
    };
    let mut chat = record.to_llm(db).await.map_err(Error::wrap)?;
    // Give the chat tools here, e.g. `chat = chat.with_tool(crate::tools::weather_tool::WeatherTool);`
    if let Err(e) = record.complete(db, &mut chat).await {
        tracing::error!(chat_id, error = %e, "chat response failed");
        return Err(Error::wrap(e));
    }
    Ok(())
}
