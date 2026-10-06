//! `ChatResponseJob` (RubyLLM's chat_ui `chat_response_job.rb.tt`) as a Loco worker. Generated
//! by `rust-llm generate chat_ui`.
//!
//! RubyLLM's job calls `chat.ask(content) { |chunk| ... }` and appends each chunk to the last
//! message over Turbo. Here the controller stores the user message first (`ask_later`) and this
//! worker runs `complete_stream`, which creates each assistant row before its first chunk and
//! writes it (with tool calls and usage) when it ends. The chunks go out on `ChatChannel`,
//! collected and sent at most every `CHUNK_INTERVAL` so a fast model doesn't send hundreds of
//! tiny events; the page appends them and reloads `messages` on `message_end`.

use std::time::{Duration, Instant};

use loco_rs::prelude::*;
use rust_llm_loco::StreamEvent;
use sea_orm::DatabaseConnection;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    channels::chat::ChatChannel,
    models::chats::{self, ChatRecord},
};

/// The longest a chunk waits before it is sent with the ones after it.
pub const CHUNK_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatResponseArgs {
    pub account_id: i64,
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
        respond(&self.ctx.db, args.account_id, args.chat_id).await
    }
}

/// Text waiting to be broadcast for one message.
struct Pending {
    message_id: i32,
    text: String,
    since: Instant,
}

/// Everything streamed into each message so far, to recognise the final chunk some protocols
/// send with the whole message again.
#[derive(Default)]
struct Streamed {
    message_id: i32,
    text: String,
}

/// Reloads the chat from its rows and streams the reply. A chat deleted in the meantime (or not
/// in the account) is skipped; a failed reply is broadcast as an error, logged, and fails the job.
async fn respond(db: &DatabaseConnection, account_id: i64, chat_id: i32) -> Result<()> {
    if chats::find_in_account(db, account_id, chat_id).await?.is_none() {
        tracing::info!(chat_id, "chat is gone; not responding");
        return Ok(());
    }
    let record = ChatRecord::find(db, chat_id).await.map_err(Error::wrap)?;
    let mut chat = record.to_llm(db).await.map_err(Error::wrap)?;
    // Give the chat tools here, e.g. `chat = chat.with_tool(crate::tools::weather_tool::WeatherTool);`

    let send = |payload: serde_json::Value| ChatChannel::broadcast_to(account_id, chat_id, payload);
    let mut pending: Option<Pending> = None;
    let mut streamed = Streamed::default();
    let flush = |pending: &mut Option<Pending>| {
        if let Some(p) = pending.take().filter(|p| !p.text.is_empty()) {
            send(json!({ "type": "chunk", "message_id": p.message_id, "content": p.text }));
        }
    };
    let outcome = record
        .complete_stream(db, &mut chat, |event| match event {
            StreamEvent::NewMessage(row) => {
                flush(&mut pending);
                streamed = Streamed {
                    message_id: row.id,
                    text: String::new(),
                };
                send(json!({ "type": "message_start", "message_id": row.id, "role": row.role }));
            }
            StreamEvent::Chunk { message_id, chunk } => {
                let text = chunk.content();
                let repeat = streamed.message_id == message_id
                    && !streamed.text.is_empty()
                    && text == streamed.text;
                if text.is_empty() || repeat {
                    return;
                }
                streamed.text.push_str(text);
                let p = pending.get_or_insert_with(|| Pending {
                    message_id,
                    text: String::new(),
                    since: Instant::now(),
                });
                p.text.push_str(text);
                if p.since.elapsed() >= CHUNK_INTERVAL {
                    flush(&mut pending);
                }
            }
            StreamEvent::EndMessage(row) => {
                flush(&mut pending);
                send(json!({ "type": "message_end", "message_id": row.id }));
            }
        })
        .await;
    if let Err(e) = outcome {
        tracing::error!(chat_id, error = %e, "chat response failed");
        send(json!({ "type": "error", "message": e.to_string() }));
        return Err(Error::wrap(e));
    }
    Ok(())
}
