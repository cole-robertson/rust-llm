//! Ports of RubyLLM 2.0's `protocol/stream_accumulator_spec.rb`: chunks folded by
//! `protocols::StreamAccumulator` exactly as the spec builds them with `RubyLLM::Chunk.new`.
//! Ruby's integer stream keys are strings here (the port keys tool-call fragments by string).

use rust_llm::message::indexmap_lite::IndexMap;
use rust_llm::message::{RawResponse, ServerToolCall};
use rust_llm::protocols::StreamAccumulator;
use rust_llm::{Citation, Error, FinishReason, Message, Role, Thinking, ToolCall};
use serde_json::{Map, Value, json};

/// `RubyLLM::Chunk.new(role: :assistant, content:)`.
fn chunk(content: Option<&str>) -> Message {
    Message::new(Role::Assistant, content.map(str::to_string))
}

/// `RubyLLM::Chunk.new(role: :assistant, content: nil, tool_calls: { key => call, ... })`.
fn tool_chunk(content: Option<&str>, calls: Vec<(&str, ToolCall)>) -> Message {
    let mut c = chunk(content);
    c.tool_calls = Some(
        calls
            .into_iter()
            .map(|(k, call)| (k.to_string(), call))
            .collect::<IndexMap<ToolCall>>(),
    );
    c
}

/// `ToolCall.new(id:, name:, arguments: '<json text>')` as a streamed opening piece.
fn opening(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall::opening(id.into(), name.into(), arguments.into())
}

/// `ToolCall.new(id: nil, name: nil, arguments: '<json text>')`.
fn fragment(arguments: &str) -> ToolCall {
    ToolCall::fragment(arguments.into())
}

fn args(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

fn finish(acc: StreamAccumulator) -> Message {
    acc.into_message(RawResponse::default()).unwrap()
}

fn call<'a>(message: &'a Message, id: &str) -> &'a ToolCall {
    message.tool_calls.as_ref().unwrap().get(id).unwrap()
}

// spec: protocol/stream_accumulator_spec.rb:7 #add > ignores an empty model id from an initial chunk
#[test]
fn ignores_an_empty_model_id_from_an_initial_chunk() {
    let mut acc = StreamAccumulator::default();
    let mut first = chunk(Some(""));
    first.model = Some(String::new());
    let mut second = chunk(Some("Hi"));
    second.model = Some("gpt-5.4-2026-03-05".into());
    acc.add(&first);
    acc.add(&second);
    assert_eq!(finish(acc).model.as_deref(), Some("gpt-5.4-2026-03-05"));
}

// spec: protocol/stream_accumulator_spec.rb:17 #add > keeps the first non-empty model id
#[test]
fn keeps_the_first_non_empty_model_id() {
    let mut acc = StreamAccumulator::default();
    for (text, model) in [("Hi", "model-a"), ("!", "model-b")] {
        let mut c = chunk(Some(text));
        c.model = Some(model.into());
        acc.add(&c);
    }
    assert_eq!(finish(acc).model.as_deref(), Some("model-a"));
}

// spec: protocol/stream_accumulator_spec.rb:27 #add > handles tool call deltas that omit arguments
#[test]
fn handles_tool_call_deltas_that_omit_arguments() {
    let mut acc = StreamAccumulator::default();
    // `arguments: nil` on an opening piece: the port's streamed pieces carry text, so nil is "".
    acc.add(&tool_chunk(
        None,
        vec![("call_1", opening("call_1", "weather", ""))],
    ));
    let message = finish(acc);
    assert_eq!(call(&message, "call_1").arguments(), Map::new());
}

// spec: protocol/stream_accumulator_spec.rb:38 #add > keeps interleaved tool call fragments separate by stream key
#[test]
fn keeps_interleaved_tool_call_fragments_separate_by_stream_key() {
    let mut acc = StreamAccumulator::default();
    let chunks = [
        ("1", ToolCall::new("call_1", "market_data", Map::new())),
        ("2", ToolCall::new("call_2", "search", Map::new())),
        ("1", fragment(r#"{"symbol":"MNQM26","#)),
        ("2", fragment(r#"{"query":"market news","#)),
        ("1", fragment(r#""interval":"minute"}"#)),
        ("2", fragment(r#""date":"2026-03-31"}"#)),
    ];
    for (key, c) in chunks {
        acc.add(&tool_chunk(None, vec![(key, c)]));
    }
    let message = finish(acc);
    assert_eq!(
        call(&message, "call_1").arguments(),
        args(json!({ "symbol": "MNQM26", "interval": "minute" }))
    );
    assert_eq!(
        call(&message, "call_2").arguments(),
        args(json!({ "query": "market news", "date": "2026-03-31" }))
    );
}

// spec: protocol/stream_accumulator_spec.rb:66 #add > wraps malformed streamed tool call arguments in a RubyLLM error
#[test]
fn wraps_malformed_streamed_tool_call_arguments_in_a_rust_llm_error() {
    let mut acc = StreamAccumulator::default();
    acc.add(&tool_chunk(
        None,
        vec![("0", opening("call_1", "weather", r#"{"city":"Berlin""#))],
    ));
    let mut last = chunk(None);
    last.finish_reason = Some(FinishReason::from_symbol("length"));
    acc.add(&last);
    let response = RawResponse {
        status: 200,
        body: json!({ "stream": "body" }),
        ..Default::default()
    };

    let error = acc.into_message(response.clone()).unwrap_err();

    let Error::ToolCallParse {
        finish_reason,
        cause,
        ..
    } = &error
    else {
        panic!("expected ToolCallParse, got {error:?}");
    };
    // `error.response == response`
    let seen = error.response().expect("the error keeps the response");
    assert_eq!(
        (seen.status, seen.body.as_str()),
        (response.status, response.body.to_string().as_str())
    );
    assert_eq!(finish_reason.as_deref(), Some("length"));
    // `error.cause` is the JSON parser error.
    assert!(cause.as_ref().is_some_and(serde_json::Error::is_eof));
    assert!(std::error::Error::source(&error).is_some());
}

// spec: protocol/stream_accumulator_spec.rb:94 #add > deduplicates citations repeated across chunks
#[test]
fn deduplicates_citations_repeated_across_chunks() {
    let mut acc = StreamAccumulator::default();
    let citation = Citation {
        url: Some("https://example.com".into()),
        title: Some("Example".into()),
        ..Default::default()
    };
    for text in ["Hello", " world"] {
        let mut c = chunk(Some(text));
        c.citations = vec![citation.clone()];
        acc.add(&c);
    }
    assert_eq!(finish(acc).citations, vec![citation]);
}

// spec: protocol/stream_accumulator_spec.rb:119 #add > retains distinct server events without ids and replaces repeated identified events
#[test]
fn retains_distinct_server_events_without_ids_and_replaces_repeated_identified_events() {
    let mut acc = StreamAccumulator::default();
    let server_call = |kind: &str, id: Option<&str>, result: Option<&str>| ServerToolCall {
        kind: kind.into(),
        name: None,
        id: id.map(str::to_string),
        input: None,
        result: result.map(Value::from),
        raw: json!({}),
    };
    let calls = [
        server_call("tool_result", None, Some("first")),
        server_call("tool_result", None, Some("second")),
        server_call("remote_call", Some("call_1"), None),
        server_call("remote_call", Some("call_1"), Some("done")),
    ];
    for call in calls {
        let mut c = chunk(None);
        c.server_tool_calls = vec![call];
        acc.add(&c);
    }
    let results: Vec<Option<Value>> = finish(acc)
        .server_tool_calls
        .into_iter()
        .map(|c| c.result)
        .collect();
    assert_eq!(
        results,
        vec![
            Some(json!("first")),
            Some(json!("second")),
            Some(json!("done"))
        ]
    );
}

// spec: protocol/stream_accumulator_spec.rb:133 #add > resolves citation text spans from the accumulated content
#[test]
fn resolves_citation_text_spans_from_the_accumulated_content() {
    let mut acc = StreamAccumulator::default();
    acc.add(&chunk(Some("Hello cited world")));
    let mut c = chunk(None);
    c.citations = vec![Citation {
        url: Some("https://example.com".into()),
        start_index: Some(6),
        end_index: Some(11),
        ..Default::default()
    }];
    acc.add(&c);
    let message = finish(acc);
    assert_eq!(message.citations[0].text.as_deref(), Some("cited"));
    assert_eq!(
        message.citations[0].url.as_deref(),
        Some("https://example.com")
    );
}

// spec: protocol/stream_accumulator_spec.rb:145 #add > preserves the final non-nil finish reason
#[test]
fn preserves_the_final_non_nil_finish_reason() {
    let mut acc = StreamAccumulator::default();
    acc.add(&chunk(Some("Hello")));
    let mut last = chunk(None);
    last.finish_reason = Some(FinishReason::from_symbol("tool_use"));
    acc.add(&last);
    assert_eq!(
        finish(acc).finish_reason,
        Some(FinishReason::Other("tool_use".into()))
    );
}

// spec: protocol/stream_accumulator_spec.rb:170 content accumulation > leaves markup in the content alone
#[test]
fn leaves_markup_in_the_content_alone() {
    let mut acc = StreamAccumulator::default();
    acc.add(&chunk(Some("<think>reasoning</think>answer")));
    let message = finish(acc);
    assert_eq!(
        message.content.as_deref(),
        Some("<think>reasoning</think>answer")
    );
    assert_eq!(message.thinking, None);
}

// spec: protocol/stream_accumulator_spec.rb:179 thinking deltas > keeps the first signature and joins the text
#[test]
fn keeps_the_first_signature_and_joins_the_text() {
    let mut acc = StreamAccumulator::default();
    let deltas = [
        (Some("one "), None),
        (None, Some("sig-1")),
        (Some("two"), Some("sig-2")),
    ];
    for (text, signature) in deltas {
        let mut c = chunk(None);
        c.thinking = Some(Thinking {
            text: text.map(str::to_string),
            signature: signature.map(str::to_string),
        });
        acc.add(&c);
    }
    let thinking = finish(acc).thinking.unwrap();
    assert_eq!(thinking.text.as_deref(), Some("one two"));
    assert_eq!(thinking.signature.as_deref(), Some("sig-1"));
}

// spec: protocol/stream_accumulator_spec.rb:212 tool call fragments > keeps parallel id-less tool calls separate and keys them by their generated ids
#[test]
fn keeps_parallel_id_less_tool_calls_separate_and_keys_them_by_their_generated_ids() {
    let mut acc = StreamAccumulator::default();
    acc.add(&tool_chunk(
        None,
        vec![
            ("0", opening("", "weather", "{}")),
            ("1", opening("", "news", "{}")),
        ],
    ));
    let calls = acc.tool_calls();
    let names: Vec<&str> = calls.values().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["weather", "news"]);
    let keys: Vec<&String> = calls.keys().collect();
    let ids: Vec<&String> = calls.values().map(|c| &c.id).collect();
    assert_eq!(keys, ids);
    assert!(ids.iter().all(|id| id.len() == 36), "{ids:?}");
    assert_ne!(ids[0], ids[1]);
}

// spec: protocol/stream_accumulator_spec.rb:228 tool call fragments > keeps text arriving in the same chunk as a tool call
#[test]
fn keeps_text_arriving_in_the_same_chunk_as_a_tool_call() {
    let mut acc = StreamAccumulator::default();
    acc.add(&tool_chunk(
        Some("Checking the weather."),
        vec![("0", opening("call_1", "weather", "{}"))],
    ));
    let message = finish(acc);
    assert_eq!(message.content.as_deref(), Some("Checking the weather."));
    let keys: Vec<&String> = message.tool_calls.as_ref().unwrap().keys().collect();
    assert_eq!(keys, ["call_1"]);
}

// spec: protocol/stream_accumulator_spec.rb:255 tool call fragments > treats a nil argument fragment as empty
#[test]
fn treats_a_nil_argument_fragment_as_empty() {
    let mut acc = StreamAccumulator::default();
    acc.add(&tool_chunk(
        None,
        vec![("0", opening("call_1", "weather", r#"{"a":"#))],
    ));
    // `arguments: nil`: the port's fragment of nothing.
    acc.add(&tool_chunk(None, vec![("0", fragment(""))]));
    acc.add(&tool_chunk(None, vec![("0", fragment("1}"))]));
    let message = finish(acc);
    assert_eq!(
        call(&message, "call_1").arguments(),
        args(json!({ "a": 1 }))
    );
}

// spec: protocol/stream_accumulator_spec.rb:279 tool call fragments > adopts a thought signature that arrives with a later fragment
#[test]
fn adopts_a_thought_signature_that_arrives_with_a_later_fragment() {
    let mut acc = StreamAccumulator::default();
    acc.add(&tool_chunk(
        None,
        vec![("0", opening("call_1", "weather", "{}"))],
    ));
    let mut signed = fragment("");
    signed.thought_signature = Some("sig".into());
    acc.add(&tool_chunk(None, vec![("0", signed)]));
    let message = finish(acc);
    assert_eq!(
        call(&message, "call_1").thought_signature.as_deref(),
        Some("sig")
    );
}

// spec: protocol/stream_accumulator_spec.rb:297 tool call fragments > keeps hash arguments as they arrived
#[test]
fn keeps_hash_arguments_as_they_arrived() {
    let mut acc = StreamAccumulator::default();
    acc.add(&tool_chunk(
        None,
        vec![(
            "0",
            ToolCall::new("call_1", "weather", args(json!({ "city": "Rome" }))),
        )],
    ));
    let message = finish(acc);
    assert_eq!(
        call(&message, "call_1").arguments(),
        args(json!({ "city": "Rome" }))
    );
}

// spec: protocol/stream_accumulator_spec.rb:328 citation spans > leaves a citation alone when the span falls outside the content
#[test]
fn leaves_a_citation_alone_when_the_span_falls_outside_the_content() {
    let mut acc = StreamAccumulator::default();
    let mut c = chunk(Some("short"));
    c.citations = vec![Citation {
        url: Some("https://example.test".into()),
        start_index: Some(100),
        end_index: Some(200),
        ..Default::default()
    }];
    acc.add(&c);
    assert_eq!(finish(acc).citations[0].text, None);
}
