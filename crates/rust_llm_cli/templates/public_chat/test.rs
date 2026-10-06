//! The public chat (`rust-llm generate public_chat`): no sign-in, the reply streams back in the
//! POST's own response, guests never see each other's conversations, and every limit holds.
//! Replies come from `anthropic_stub` (a local stand-in for Anthropic's streaming API).

use std::time::Duration;

use axum_test::TestResponse;
use {{pkg_name}}::controllers::public_chat::{self, Limits, COOKIE, SLOW_DOWN};
use serde_json::{json, Value};
use serial_test::serial;

use super::anthropic_stub::{self, MODEL};
use super::*;

fn limits() -> Limits {
    Limits {
        ip_messages: 20,
        session_messages: 10,
        window: Duration::from_secs(600),
        max_input_chars: 4_000,
        max_output_tokens: 1_024,
        max_turns: 20,
        model: MODEL.to_owned(),
        provider: Some("anthropic".to_owned()),
        instructions: "Be brief.".to_owned(),
    }
}

/// [`with_app`] with the public chat's `limits`, the stub, and no conversations yet.
async fn with_public_chat<F, Fut>(limits: Limits, f: F)
where
    F: FnOnce(TestServer, AppContext) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    anthropic_stub::use_stub().await;
    public_chat::reset();
    with_app(|server, ctx| async move {
        ctx.shared_store.insert(limits);
        f(server, ctx).await;
    })
    .await;
}

/// The `data:` frames of a Server-Sent Events body.
fn frames(res: &TestResponse) -> Vec<Value> {
    res.text()
        .split("\n\n")
        .filter_map(|event| {
            event
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .map(|data| serde_json::from_str(data).unwrap())
        })
        .collect()
}

async fn say(server: &TestServer, content: &str) -> TestResponse {
    server
        .post(route_table::PUBLIC_CHAT_MESSAGES)
        .json(&json!({ "content": content }))
        .await
}

async fn say_as(server: &TestServer, ip: &str, content: &str) -> TestResponse {
    server
        .post(route_table::PUBLIC_CHAT_MESSAGES)
        .add_header("x-forwarded-for", ip)
        .json(&json!({ "content": content }))
        .await
}

/// The `/chat` page's messages, as `(role, content)`.
async fn conversation(server: &TestServer, ctx: &AppContext) -> Vec<(String, String)> {
    let page = inertia_get(server, ctx, route_table::PUBLIC_CHAT).await;
    assert_eq!(page["component"], "public_chat/show");
    page["props"]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["role"].as_str().unwrap().to_owned(),
                m["content"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[tokio::test]
#[serial]
async fn a_guest_chats_without_signing_in_and_the_reply_streams() {
    with_public_chat(limits(), |server, ctx| async move {
        assert!(conversation(&server, &ctx).await.is_empty());
        let res = say(&server, "Hello there").await;
        assert_eq!(res.status_code(), 200, "{}", res.text());
        assert!(res
            .header(header::CONTENT_TYPE)
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let frames = frames(&res);
        let chunks: Vec<&str> = frames
            .iter()
            .filter(|f| f["type"] == "chunk")
            .map(|f| f["content"].as_str().unwrap())
            .collect();
        assert!(chunks.len() > 1, "token by token: {frames:?}");
        assert_eq!(chunks.concat(), "You said: Hello there");
        assert_eq!(
            frames.last().unwrap(),
            &json!({ "type": "end", "content": "You said: Hello there" })
        );
        // Kept for this browser session: the page shows it, and the next message has it as history.
        assert_eq!(
            conversation(&server, &ctx).await,
            [
                ("user".into(), "Hello there".into()),
                ("assistant".into(), "You said: Hello there".into())
            ]
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn guests_never_see_each_others_conversations() {
    with_public_chat(limits(), |server, ctx| async move {
        let res = say(&server, "my secret").await;
        assert_eq!(res.status_code(), 200);
        let cookie = res.cookie(COOKIE);
        assert!(cookie.http_only().unwrap_or(false), "not readable from JS");
        assert_eq!(cookie.value().len(), 43, "256 random bits");

        // A second browser: no cookie, so a conversation of its own.
        let mut other = server.clone();
        other.clear_cookies();
        assert!(conversation(&other, &ctx).await.is_empty());
        let res = say(&other, "hello").await;
        assert_eq!(
            frames(&res).last().unwrap()["content"],
            "You said: hello",
            "the reply carries only its own conversation"
        );
        assert_ne!(res.cookie(COOKIE).value(), cookie.value());
        assert_eq!(conversation(&other, &ctx).await.len(), 2);

        // A made-up conversation id finds nothing (and gets a fresh one).
        let mut forged = server.clone();
        forged.clear_cookies();
        forged.add_cookie(axum_test::CookieBuilder::new(COOKIE, "x".repeat(43)).build());
        assert!(conversation(&forged, &ctx).await.is_empty());

        // The first browser still has only its own.
        assert_eq!(conversation(&server, &ctx).await[0].1, "my secret");
    })
    .await;
}

#[tokio::test]
#[serial]
async fn a_message_over_the_length_limit_is_refused() {
    let mut limits = limits();
    limits.max_input_chars = 10;
    with_public_chat(limits, |server, ctx| async move {
        let res = say(&server, "this is far too long").await;
        assert_eq!(res.status_code(), 413);
        assert!(res.json::<Value>()["error"]
            .as_str()
            .unwrap()
            .contains("under 10 characters"));
        assert_eq!(say(&server, "   ").await.status_code(), 422);
        assert!(conversation(&server, &ctx).await.is_empty());
    })
    .await;
}

#[tokio::test]
#[serial]
async fn a_conversation_stops_at_its_turn_limit() {
    let mut limits = limits();
    limits.max_turns = 2;
    with_public_chat(limits, |server, ctx| async move {
        assert_eq!(say(&server, "one").await.status_code(), 200);
        assert_eq!(say(&server, "two").await.status_code(), 200);
        let res = say(&server, "three").await;
        assert_eq!(res.status_code(), 429);
        assert!(res.json::<Value>()["error"]
            .as_str()
            .unwrap()
            .contains("Start a new one"));
        let page = inertia_get(&server, &ctx, route_table::PUBLIC_CHAT).await;
        assert_eq!(page["props"]["limits"]["turns_left"], 0);
        // Starting over gives a fresh conversation.
        server.delete(route_table::PUBLIC_CHAT).await;
        assert!(conversation(&server, &ctx).await.is_empty());
        assert_eq!(say(&server, "four").await.status_code(), 200);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn one_conversation_is_rate_limited() {
    let mut limits = limits();
    limits.session_messages = 2;
    with_public_chat(limits, |server, _ctx| async move {
        assert_eq!(say(&server, "one").await.status_code(), 200);
        assert_eq!(say(&server, "two").await.status_code(), 200);
        let res = say(&server, "three").await;
        assert_eq!(res.status_code(), 429);
        assert_eq!(res.json::<Value>()["error"], SLOW_DOWN);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn one_ip_is_rate_limited_across_conversations() {
    use loco_rs::controller::middleware::remote_ip::{ClientIpSource, RemoteIpMiddleware};
    anthropic_stub::use_stub().await;
    public_chat::reset();
    let trust_the_proxy = |config: &mut loco_rs::config::Config| {
        config.server.middlewares.remote_ip = Some(RemoteIpMiddleware {
            enable: true,
            source: ClientIpSource::RightmostXForwardedFor,
        });
    };
    with_app_config(trust_the_proxy, |server, ctx| async move {
        let mut limits = limits();
        limits.ip_messages = 3;
        ctx.shared_store.insert(limits);
        // A new browser (no cookie) for each message: only the IP links them.
        for n in 0..3 {
            let mut fresh = server.clone();
            fresh.clear_cookies();
            let res = say_as(&fresh, "203.0.113.7", &format!("hi {n}")).await;
            assert_eq!(res.status_code(), 200, "{}", res.text());
        }
        let mut fresh = server.clone();
        fresh.clear_cookies();
        let res = say_as(&fresh, "203.0.113.7", "one more").await;
        assert_eq!(res.status_code(), 429);
        assert_eq!(res.json::<Value>()["error"], SLOW_DOWN);
        // Another address is not affected.
        let res = say_as(&fresh, "203.0.113.8", "hello").await;
        assert_eq!(res.status_code(), 200);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn the_output_token_cap_is_sent_to_the_model() {
    let mut limits = limits();
    limits.max_output_tokens = 77;
    with_public_chat(limits, |server, _ctx| async move {
        let res = say(&server, "count to ten").await;
        assert_eq!(res.status_code(), 200);
        let request = anthropic_stub::last_request().expect("the stub was called");
        assert_eq!(request["max_tokens"], 77);
        assert_eq!(request["system"][0]["text"], "Be brief.");
    })
    .await;
}
