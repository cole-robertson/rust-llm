//! The chat UI (`rust-llm generate chat_ui`): chats belong to an account, a reply streams over
//! `ChatChannel` token by token, and nobody outside the account can see a chat or its stream.
//! Replies come from `anthropic_stub` (a local stand-in for Anthropic's streaming API).

use std::time::Duration;

use futures_util::{Stream, StreamExt};
use {{pkg_name}}::{
    auth::CurrentSession,
    channels::chat::ChatChannel,
    live,
    models::{chats, messages, sessions as session_model},
};
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QueryOrder};
use serde_json::{json, Value};
use serial_test::serial;

use super::anthropic_stub::{self, MODEL};
use super::*;

const TWO: &str = "two@example.com";
/// Seeded accounts: Acme (one@ owner, two@ member) and Globex (two@ owner).
const ACME: i64 = 1;
const GLOBEX: i64 = 2;

async fn session_of(ctx: &AppContext, email: &str) -> CurrentSession {
    let user = user(ctx, email).await;
    let session = session_model::Model::create_for_user(
        &ctx.db,
        &user,
        &session_model::RequestDetails::default(),
    )
    .await
    .unwrap();
    CurrentSession { session, user }
}

/// The next frame within `wait`, if any.
async fn next(stream: &mut (impl Stream<Item = Value> + Unpin), wait: Duration) -> Option<Value> {
    tokio::time::timeout(wait, stream.next())
        .await
        .ok()
        .flatten()
}

fn chat_channel(slug: &str, id: i32) -> String {
    live::identifier(ChatChannel::NAME, &json!({ "account": slug, "id": id }))
}

/// `POST /{slug}/chats` as whoever is signed in; the new chat's id (the worker has answered:
/// tests run jobs in the foreground).
async fn create_chat(server: &TestServer, slug: &str, prompt: &str) -> i32 {
    let res = server
        .post(&format!("/{slug}/chats"))
        .json(&json!({ "model": format!("anthropic:{MODEL}"), "prompt": prompt }))
        .await;
    assert!(res.status_code().is_redirection(), "{}", res.text());
    let location = res.header(header::LOCATION);
    let location = location.to_str().unwrap();
    let prefix = format!("/{slug}/chats/");
    assert!(location.starts_with(&prefix), "{location}");
    location[prefix.len()..].parse().unwrap()
}

async fn rows(ctx: &AppContext, chat_id: i32) -> Vec<(String, String)> {
    messages::Entity::find()
        .filter(messages::Column::ChatId.eq(chat_id))
        .order_by_asc(messages::Column::Id)
        .all(&ctx.db)
        .await
        .unwrap()
        .into_iter()
        .map(|m| (m.role, m.content.unwrap_or_default()))
        .collect()
}

#[tokio::test]
#[serial]
async fn signed_out_visitors_are_sent_to_sign_in() {
    with_app(|server, _ctx| async move {
        let res = server.get("/acme/chats").await;
        assert_redirect(&res, route_table::SIGN_IN);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn a_reply_streams_over_the_chat_channel_and_is_saved() {
    anthropic_stub::use_stub().await;
    with_app(|mut server, ctx| async move {
        sign_in(&mut server, &ctx, ONE).await;
        let chat_id = create_chat(&server, "acme", "Hello").await;
        assert_eq!(
            rows(&ctx, chat_id).await,
            [
                ("user".into(), "Hello".into()),
                ("assistant".into(), "You said: Hello".into())
            ]
        );
        let page = inertia_get(&server, &ctx, &format!("/acme/chats/{chat_id}")).await;
        assert_eq!(page["component"], "chats/show");
        assert_eq!(page["props"]["messages"].as_array().unwrap().len(), 2);
        assert_eq!(page["props"]["awaiting_response"], false);

        // one@ watches the chat while sending a follow-up.
        let one = session_of(&ctx, ONE).await;
        let mut stream = live::subscribe(&ctx, &one, vec![chat_channel("acme", chat_id)]).await;
        assert_eq!(
            next(&mut stream, Duration::from_secs(1)).await.unwrap()["type"],
            "confirm_subscription"
        );
        let res = server
            .post(&format!("/acme/chats/{chat_id}/messages"))
            .json(&json!({ "content": "Tell me about streaming tokens" }))
            .await;
        assert_redirect(&res, &format!("/acme/chats/{chat_id}"));

        let mut events = Vec::new();
        while let Some(frame) = next(&mut stream, Duration::from_secs(2)).await {
            let message = frame["message"].clone();
            let done = message["type"] == "message_end";
            events.push(message);
            if done {
                break;
            }
        }
        let start = &events[0];
        assert_eq!(start["type"], "message_start", "{events:?}");
        let id = start["message_id"].clone();
        let chunks: Vec<&str> = events
            .iter()
            .filter(|e| e["type"] == "chunk")
            .map(|e| {
                assert_eq!(e["message_id"], id);
                e["content"].as_str().unwrap()
            })
            .collect();
        assert!(
            chunks.len() > 1,
            "the reply arrives in pieces, not at once: {events:?}"
        );
        let end = events.last().unwrap();
        assert_eq!(end["type"], "message_end");
        assert_eq!(end["message_id"], id);
        // The pieces add up to the saved reply, in the row the stream named.
        let reply = "You said: Tell me about streaming tokens";
        assert_eq!(chunks.concat(), reply);
        let saved = messages::Entity::find_by_id(i32::try_from(id.as_i64().unwrap()).unwrap())
            .one(&ctx.db)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(saved.content.as_deref(), Some(reply));
        assert_eq!(rows(&ctx, chat_id).await.len(), 4);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn chats_are_listed_per_account() {
    anthropic_stub::use_stub().await;
    with_app(|mut server, ctx| async move {
        sign_in(&mut server, &ctx, TWO).await; // in Acme and Globex
        let acme = create_chat(&server, "acme", "in acme").await;
        let globex = create_chat(&server, "globex", "in globex").await;
        let ids = |page: Value| -> Vec<i64> {
            page["props"]["chats"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c["id"].as_i64().unwrap())
                .collect()
        };
        assert_eq!(
            ids(inertia_get(&server, &ctx, "/acme/chats").await),
            [i64::from(acme)]
        );
        assert_eq!(
            ids(inertia_get(&server, &ctx, "/globex/chats").await),
            [i64::from(globex)]
        );
        assert!(chats::find_in_account(&ctx.db, ACME, globex)
            .await
            .unwrap()
            .is_none());
        assert!(chats::find_in_account(&ctx.db, GLOBEX, globex)
            .await
            .unwrap()
            .is_some());
    })
    .await;
}

#[tokio::test]
#[serial]
async fn another_accounts_chat_is_a_404() {
    anthropic_stub::use_stub().await;
    with_app(|mut server, ctx| async move {
        sign_in(&mut server, &ctx, TWO).await;
        let globex = create_chat(&server, "globex", "secret plans").await;

        // one@ is in Acme only.
        server.clear_cookies();
        sign_in(&mut server, &ctx, ONE).await;
        for path in [
            format!("/globex/chats/{globex}"),
            format!("/acme/chats/{globex}"),
            "/globex/chats".to_owned(),
        ] {
            assert_eq!(server.get(&path).await.status_code(), 404, "GET {path}");
        }
        let res = server
            .post(&format!("/acme/chats/{globex}/messages"))
            .json(&json!({ "content": "let me in" }))
            .await;
        assert_eq!(res.status_code(), 404);
        assert_eq!(
            server
                .delete(&format!("/acme/chats/{globex}"))
                .await
                .status_code(),
            404
        );
        // Nothing was added or removed.
        assert_eq!(rows(&ctx, globex).await.len(), 2);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn a_non_member_cannot_subscribe_to_a_chat() {
    anthropic_stub::use_stub().await;
    with_app(|mut server, ctx| async move {
        sign_in(&mut server, &ctx, TWO).await;
        let globex = create_chat(&server, "globex", "secret plans").await;

        let one = session_of(&ctx, ONE).await; // in Acme only
        let mut stream = live::subscribe(
            &ctx,
            &one,
            vec![
                chat_channel("globex", globex),
                // Accepted in Acme, asking for Globex's chat id.
                chat_channel("acme", globex),
            ],
        )
        .await;
        for _ in 0..2 {
            let frame = next(&mut stream, Duration::from_secs(1)).await.unwrap();
            assert_eq!(frame["type"], "reject_subscription", "{frame}");
        }
        ChatChannel::broadcast_to(GLOBEX, globex, json!({ "type": "chunk", "content": "x" }));
        assert_eq!(next(&mut stream, Duration::from_millis(300)).await, None);

        // A member of Globex is accepted.
        let two = session_of(&ctx, TWO).await;
        let mut stream = live::subscribe(&ctx, &two, vec![chat_channel("globex", globex)]).await;
        assert_eq!(
            next(&mut stream, Duration::from_secs(1)).await.unwrap()["type"],
            "confirm_subscription"
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn a_blank_message_comes_back_with_an_error() {
    anthropic_stub::use_stub().await;
    with_app(|mut server, ctx| async move {
        sign_in(&mut server, &ctx, ONE).await;
        let chat_id = create_chat(&server, "acme", "Hello").await;
        let res = server
            .post(&format!("/acme/chats/{chat_id}/messages"))
            .json(&json!({ "content": "  " }))
            .await;
        assert_redirect(&res, &format!("/acme/chats/{chat_id}"));
        let page = inertia_get(&server, &ctx, &format!("/acme/chats/{chat_id}")).await;
        assert_eq!(page["props"]["errors"]["content"][0], "can't be blank");
        assert_eq!(rows(&ctx, chat_id).await.len(), 2);
    })
    .await;
}
