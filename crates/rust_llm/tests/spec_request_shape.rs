//! `Error#request_shape` and `RequestShape` (RubyLLM 2.1): `request_shape_spec.rb`,
//! `protocol/request_shapes_spec.rb`, `protocol/request_shapes/analysis_spec.rb`,
//! `chat_request_shape_spec.rb`, each protocol's `request_shapes_spec.rb`, and
//! `error_spec.rb`'s `#request_shape`.

mod spec_helpers;

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use rust_llm::message::{Operation, indexmap_lite::IndexMap};
use rust_llm::request_shape::analysis::{Analysis, Pairing, PartSpec, Rule, Speaker, TurnSpec};
use rust_llm::request_shape::{
    Part, PartKind, Problem, ProblemKind, RequestShape, Source, ToolRound, Turn, Unit, describe,
};
use rust_llm::{
    Attachment, Chat, Config, Cost, Error, ErrorKind, Message, Parameter, ProtocolName, Role,
    Thinking, Tokens, Tool, ToolCall, ToolError, ToolResult, UsageEntry, UsageStatus,
};
use serde_json::{Map, Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn b64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// `Base64.encode64`: lines of 60 characters, each ending in a newline.
fn b64_lines(data: &[u8]) -> String {
    let encoded = b64(data);
    encoded
        .as_bytes()
        .chunks(60)
        .map(|c| format!("{}\n", String::from_utf8_lossy(c)))
        .collect()
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn turns_of(shape: &RequestShape) -> Vec<String> {
    shape.turns.iter().map(ToString::to_string).collect()
}

fn problems_of(shape: &RequestShape) -> Vec<String> {
    shape.problems.iter().map(ToString::to_string).collect()
}

fn settings(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

// ---- RequestShapeHelpers --------------------------------------------------------------------

/// `everything_shown(shape)`: its text, its Hash, and the inspection of everything in it.
fn everything_shown(shape: &RequestShape) -> String {
    let mut pieces = vec![
        shape.to_string(),
        shape.to_h().to_string(),
        format!("{shape:?}"),
    ];
    pieces.extend(shape.turns.iter().map(|t| format!("{t:?}")));
    pieces.extend(shape.instructions.iter().map(|p| format!("{p:?}")));
    pieces.extend(
        shape
            .turns
            .iter()
            .flat_map(|t| t.parts.iter().map(|p| format!("{p:?}"))),
    );
    pieces.extend(shape.tool_rounds.iter().map(|r| format!("{r:?}")));
    pieces.extend(shape.problems.iter().map(|p| format!("{p:?} {p}")));
    pieces.join("\n")
}

fn expect_no_secrets(shape: &RequestShape) {
    let shown = everything_shown(shape).to_lowercase();
    assert!(!shown.contains("secret"), "{shown}");
    assert!(!shown.contains("example.com"), "{shown}");
    let data = b64(&std::fs::read(fixture("ruby.png")).unwrap());
    assert!(!everything_shown(shape).contains(&data[..32]));
}

/// `REFUSAL`.
fn refusal() -> Value {
    json!({ "error": { "code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT" } })
}

async fn refusing_server(status: u16) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(ResponseTemplate::new(status).set_body_json(refusal()))
        .mount(&server)
        .await;
    server
}

/// `include_context 'with configured RubyLLM'`, with the providers the shape specs call.
fn config(server: &MockServer) -> Arc<Config> {
    let mut c = (*spec_helpers::config(server)).clone();
    for (provider, path) in [("mistral", "/v1"), ("perplexity", "")] {
        c.set(
            format!("{provider}_api_base"),
            format!("{}{path}", server.uri()),
        );
        c.set(format!("{provider}_api_key"), "test");
    }
    Arc::new(c)
}

struct Lookup;

#[async_trait]
impl Tool for Lookup {
    fn name(&self) -> String {
        "lookup".into()
    }
    fn description(&self) -> String {
        "Looks things up".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("query").description("What to look up")]
    }
    async fn execute(&self, a: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(a["query"].as_str().unwrap_or("").into())
    }
}

fn entry(provider: &str, model: &str) -> UsageEntry {
    UsageEntry {
        id: UsageEntry::next_id(),
        owner: None,
        operation: Operation::Chat,
        provider: provider.into(),
        model: model.into(),
        status: UsageStatus::Succeeded,
        tokens: Tokens::default(),
        cost: Cost::default(),
    }
}

fn calls(call: ToolCall) -> IndexMap<ToolCall> {
    [(call.id.clone(), call)].into_iter().collect()
}

/// `secret_conversation(provider, model)`.
fn secret_conversation(provider: &str, model: &str) -> Vec<Message> {
    let produced = |mut m: Message| {
        m.usage_entries = vec![entry(provider, model)];
        m.model = Some(model.into());
        m
    };
    let mut question = Message::user("secret question");
    question.attachments = vec![Attachment::new(fixture("ruby.png"))];
    let mut call = ToolCall::new(
        "secret-call-1",
        "lookup",
        spec_helpers::args(json!({ "query": "secret argument" })),
    );
    call.thought_signature = Some("secret-call-signature".into());
    // `Message.new(role: :assistant, content: nil, tool_calls:)` normalizes the content to "".
    let mut asks = Message::new(Role::Assistant, Some(String::new()));
    asks.tool_calls = Some(calls(call));
    asks.thinking = Thinking::build(
        Some("secret thought".into()),
        Some("secret-signature".into()),
    );
    let mut result = Message::new(Role::Tool, Some("secret result".into()));
    result.tool_call_id = Some("secret-call-1".into());
    let mut answer = Message::assistant("secret answer");
    answer.thinking = Thinking::build(Some("secret plan".into()), Some("secret-signature".into()));
    vec![
        question,
        produced(asks),
        result,
        produced(answer),
        Message::user("secret follow-up"),
    ]
}

/// `secret_chat(provider:, model:, protocol:)`.
fn secret_chat(
    server: &MockServer,
    provider: &str,
    model: &str,
    protocol: Option<ProtocolName>,
) -> Chat {
    let mut chat = Chat::with_config(config(server), Some(model), Some(provider), false)
        .expect("chat")
        .with_instructions("secret instructions")
        .with_tool(Lookup);
    if let Some(p) = protocol {
        chat = chat.with_protocol(p);
    }
    for m in secret_conversation(provider, model) {
        chat.add_message(m);
    }
    chat
}

/// `refused_request_shape(chat)`: the shape of the `BadRequestError` `complete` raises.
async fn refused_shape(chat: &mut Chat) -> RequestShape {
    let error = chat.complete().await.expect_err("refused");
    assert_eq!(error.kind(), ErrorKind::BadRequest, "{error}");
    error.request_shape().cloned().expect("shape")
}

async fn shape_of_secret_chat(
    provider: &str,
    model: &str,
    protocol: Option<ProtocolName>,
) -> RequestShape {
    let server = refusing_server(400).await;
    let mut chat = secret_chat(&server, provider, model, protocol);
    refused_shape(&mut chat).await
}

fn shape(protocol: ProtocolName, payload: Value) -> Option<RequestShape> {
    describe(protocol, None, None, &payload)
}

// ---- request_shape_spec.rb ------------------------------------------------------------------

fn example_shape() -> RequestShape {
    let mut shape = RequestShape::new(vec![
        Turn::new(0, Some("user"), vec![Part::new(PartKind::Text).chars(17)]),
        Turn::new(
            1,
            Some("model"),
            vec![
                Part::new(PartKind::Thinking).signed(),
                Part::new(PartKind::ToolCall).name("find_tasks").chars(42),
            ],
        ),
        Turn::new(
            2,
            Some("user"),
            vec![
                Part::new(PartKind::ToolResult)
                    .name("find_tasks")
                    .chars(6859),
            ],
        ),
        Turn::new(
            70,
            Some("user"),
            vec![
                Part::new(PartKind::Image)
                    .bytes(1)
                    .source(Source::Inline)
                    .mime_type("image/png"),
            ],
        ),
    ]);
    shape.provider = Some("gemini".into());
    shape.model = Some("gemini-2.5-flash".into());
    shape.step = Some(3);
    shape.payload_keys = vec!["contents".into(), "generationConfig".into(), "tools".into()];
    shape.thinking_settings = settings(json!({ "thinkingBudget": -1 }));
    shape.instructions = vec![Part::new(PartKind::Text).chars(240)];
    shape.turn_count = 71;
    shape.omitted_turns = 67;
    shape.tool_names = vec!["find_tasks".into()];
    shape.tool_rounds = vec![ToolRound::new(1, 1, 1, true)];
    shape.problems = vec![
        Problem::new(
            ProblemKind::PartWithoutData,
            "thinking part carries no data",
        )
        .at(1, Some(0)),
    ];
    shape
}

// spec: request_shape_spec.rb:35 #to_s > spells out the request, one turn per line, with units
#[test]
fn to_s_spells_out_the_request() {
    assert_eq!(
        example_shape().to_string(),
        "gemini, gemini-2.5-flash, model call 3 of the turn, after 2 tool rounds\n\
         payload keys: contents, generationConfig, tools\n\
         thinking: thinkingBudget -1\n\
         instructions: text (240 chars)\n\
         #0 user: text (17 chars)\n\
         #1 model: thinking (no text), signed, call find_tasks (args 42 chars)\n\
         #2 user: result find_tasks (6859 chars)\n\
         (67 turns omitted)\n\
         #70 user: image/png (1 byte)\n\
         tools: find_tasks\n\
         tool round at #1: 1 call, 1 result, paired\n\
         problem at #1, part 0: thinking part carries no data"
    );
}

// spec: request_shape_spec.rb:52 #to_s > says when it finds no problems, and leaves out what the request does not send
#[test]
fn to_s_says_when_it_finds_no_problems() {
    let mut shape = RequestShape::new(vec![Turn::new(0, None, vec![])]);
    shape.step = Some(1);
    assert_eq!(
        shape.to_string(),
        "model call 1 of the turn\n#0: no parts\nno problems found"
    );
}

// spec: request_shape_spec.rb:60 #to_h > keeps the same keys whatever the request sends
#[test]
fn to_h_keeps_the_same_keys() {
    let keys: Vec<String> = example_shape()
        .to_h()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(
        keys,
        [
            "provider",
            "model",
            "step",
            "payload_keys",
            "thinking_settings",
            "instructions",
            "turn_count",
            "omitted_turns",
            "turns",
            "tool_names",
            "tool_rounds",
            "problems"
        ]
    );
    assert_eq!(
        RequestShape::new(vec![]).to_h(),
        json!({
            "provider": null, "model": null, "step": null, "payload_keys": [], "thinking_settings": {},
            "instructions": [], "turn_count": 0, "omitted_turns": 0, "turns": [], "tool_names": [],
            "tool_rounds": [], "problems": []
        })
    );
}

// spec: request_shape_spec.rb:71 #to_h > describes every turn, round, and problem, with sizes under their units
#[test]
fn to_h_describes_turns_rounds_and_problems() {
    let h = example_shape().to_h();
    assert_eq!(h["provider"], "gemini");
    assert_eq!(h["model"], "gemini-2.5-flash");
    assert_eq!(h["step"], 3);
    assert_eq!(h["turn_count"], 71);
    assert_eq!(h["omitted_turns"], 67);
    assert_eq!(
        h["tool_rounds"],
        json!([{ "turn": 1, "calls": 1, "results": 1, "paired": true }])
    );
    assert_eq!(
        h["problems"],
        json!([{ "kind": "part_without_data", "turn": 1, "part": 0, "message": "thinking part carries no data" }])
    );
    assert_eq!(
        h["turns"][1],
        json!({ "index": 1, "role": "model",
                "parts": [{ "kind": "thinking", "signed": true }, { "kind": "tool_call", "name": "find_tasks", "chars": 42 }] })
    );
    assert_eq!(
        h["turns"][3]["parts"],
        json!([{ "kind": "image", "bytes": 1, "source": "inline", "mime_type": "image/png" }])
    );
}

// spec: request_shape_spec.rb:87 #inspect > summarizes the shape on one line
#[test]
fn inspect_summarizes_on_one_line() {
    let shape = example_shape();
    assert_eq!(
        format!("{shape:?}"),
        r#"#<RustLLM::RequestShape provider: "gemini", model: "gemini-2.5-flash", turns: 71, problems: 1>"#
    );
    assert_eq!(
        format!("{:?}", shape.turns[0]),
        r#"#<RustLLM::RequestShape::Turn index: 0, role: "user", parts: "text (17 chars)">"#
    );
}

// spec: request_shape_spec.rb:96 RubyLLM::RequestShape::Part > names media by its MIME type, or by its kind and where it comes from
#[test]
fn part_names_media() {
    let pdf = Part::new(PartKind::Document)
        .bytes(120)
        .source(Source::Inline)
        .mime_type("application/pdf");
    assert_eq!(pdf.to_string(), "application/pdf (120 bytes)");
    assert_eq!(
        Part::new(PartKind::Image).source(Source::Url).to_string(),
        "image (url)"
    );
    assert_eq!(
        Part::new(PartKind::Document)
            .source(Source::File)
            .mime_type("application/pdf")
            .to_string(),
        "application/pdf (file)"
    );
    assert_eq!(
        Part::new(PartKind::Audio)
            .source(Source::Inline)
            .to_string(),
        "audio (no data)"
    );
}

// spec: request_shape_spec.rb:105 RubyLLM::RequestShape::Part > names calls, results, and pieces RubyLLM does not classify
#[test]
fn part_names_calls_results_and_others() {
    let call = Part::new(PartKind::ToolCall)
        .name("lookup")
        .chars(1)
        .signed();
    assert_eq!(call.to_string(), "call lookup (args 1 char), signed");
    assert_eq!(
        Part::new(PartKind::ToolResult).chars(7).to_string(),
        "result (7 chars)"
    );
    assert_eq!(
        Part::new(PartKind::Other).name("cachePoint").to_string(),
        "cachePoint"
    );
    assert_eq!(
        Part::new(PartKind::Other).to_string(),
        "part of no known kind"
    );
}

// spec: request_shape_spec.rb:114 RubyLLM::RequestShape::Part > reports whether it is signed
#[test]
fn part_reports_whether_it_is_signed() {
    assert!(Part::new(PartKind::Text).chars(3).signed().is_signed());
    assert!(!Part::new(PartKind::Text).chars(3).is_signed());
    assert_eq!(
        Part::new(PartKind::Text).chars(3).to_h(),
        json!({ "kind": "text", "chars": 3 })
    );
    assert_eq!(Part::new(PartKind::Text).chars(3).unit, Some(Unit::Chars));
}

// spec: request_shape_spec.rb:122 RubyLLM::RequestShape::ToolRound > counts calls and results, and says whether they pair up
#[test]
fn tool_round_counts_and_pairs() {
    let round = ToolRound::new(4, 2, 1, false);
    assert_eq!(
        round.to_string(),
        "tool round at #4: 2 calls, 1 result, not paired"
    );
    assert!(!round.is_paired());
}

// spec: request_shape_spec.rb:131 RubyLLM::RequestShape::Problem > names the one id a shape shows, and where the problem is
#[test]
fn problem_names_its_id_and_location() {
    let problem = Problem::new(
        ProblemKind::UnmatchedResult,
        "result answers no call in the request",
    )
    .at(6, Some(0))
    .call_id("toolu_01");
    assert_eq!(
        problem.to_string(),
        "problem at #6, part 0: result answers no call in the request (call id toolu_01)"
    );
    assert_eq!(
        Problem::new(ProblemKind::EmptyTurn, "turn has no parts")
            .at(2, None)
            .to_string(),
        "problem at #2: turn has no parts"
    );
}

// ---- protocol/request_shapes/analysis_spec.rb -----------------------------------------------

fn text() -> PartSpec {
    PartSpec {
        measure: Some(5),
        ..PartSpec::new(PartKind::Text)
    }
}

fn call(name: &str, id: Option<&str>, signed: bool) -> PartSpec {
    PartSpec {
        name: Some(name.into()),
        call_id: id.map(str::to_string),
        measure: Some(2),
        signed,
        ..PartSpec::new(PartKind::ToolCall)
    }
}

fn result(name: Option<&str>, id: Option<&str>) -> PartSpec {
    PartSpec {
        name: name.map(str::to_string),
        result_id: id.map(str::to_string),
        measure: Some(9),
        ..PartSpec::new(PartKind::ToolResult)
    }
}

fn turn(index: usize, speaker: Speaker, parts: Vec<PartSpec>) -> TurnSpec {
    let role = format!("{speaker:?}").to_lowercase();
    TurnSpec {
        index,
        role: Some(role),
        speaker,
        parts,
    }
}

fn problem_hashes(turns: Vec<TurnSpec>, pairing: Pairing, rules: &[Rule]) -> Vec<Value> {
    Analysis::new(turns, pairing, rules)
        .problems()
        .iter()
        .map(|p| {
            let mut h = p.to_h();
            h.as_object_mut().unwrap().remove("message");
            h
        })
        .collect()
}

// spec: protocol/request_shapes/analysis_spec.rb:30 pairs results with calls by id, names them after their calls, and shows the id of a result with no call
#[test]
fn analysis_pairs_by_id() {
    let turns = vec![
        turn(0, Speaker::User, vec![text()]),
        turn(
            1,
            Speaker::Model,
            vec![
                call("lookup", Some("a"), false),
                call("fetch", Some("b"), false),
            ],
        ),
        turn(
            2,
            Speaker::Tool,
            vec![result(None, Some("b")), result(None, Some("a"))],
        ),
        turn(3, Speaker::Model, vec![call("lookup", Some("c"), false)]),
        turn(4, Speaker::Tool, vec![result(None, Some("z"))]),
    ];
    let analysis = Analysis::new(turns, Pairing::Id, &[]);
    let rounds: Vec<Value> = analysis.tool_rounds().iter().map(ToolRound::to_h).collect();
    assert_eq!(
        rounds,
        [
            json!({ "turn": 1, "calls": 2, "results": 2, "paired": true }),
            json!({ "turn": 3, "calls": 1, "results": 1, "paired": false })
        ]
    );
    let names: Vec<_> = analysis.turns()[2]
        .parts
        .iter()
        .map(|p| p.name.clone())
        .collect();
    assert_eq!(
        names,
        [Some("fetch".to_string()), Some("lookup".to_string())]
    );
    let problems: Vec<String> = analysis
        .problems()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        problems,
        [
            "problem at #3: 1 call and 1 result that do not pair up",
            "problem at #4, part 0: result answers no call in the request (call id z)"
        ]
    );
}

// spec: protocol/request_shapes/analysis_spec.rb:45 pairs results with calls by position and name where the provider sends no ids
#[test]
fn analysis_pairs_by_position() {
    let turns = vec![
        turn(0, Speaker::User, vec![text()]),
        turn(
            1,
            Speaker::Model,
            vec![call("lookup", None, false), call("fetch", None, false)],
        ),
        turn(
            2,
            Speaker::Tool,
            vec![result(Some("fetch"), None), result(Some("lookup"), None)],
        ),
        turn(3, Speaker::Model, vec![call("lookup", None, false)]),
    ];
    let problems: Vec<String> = Analysis::new(turns, Pairing::Position, &[])
        .problems()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        problems,
        [
            "problem at #1: 2 calls and 2 results that do not pair up",
            "problem at #3: 1 call but 0 results"
        ]
    );
}

// spec: protocol/request_shapes/analysis_spec.rb:54 counts the model calls of the current turn
#[test]
fn analysis_counts_the_steps_of_the_current_turn() {
    let turns = vec![
        turn(0, Speaker::User, vec![text()]),
        turn(1, Speaker::Model, vec![call("a", Some("1"), false)]),
        turn(2, Speaker::Tool, vec![result(None, Some("1"))]),
        turn(3, Speaker::Model, vec![call("a", Some("2"), false)]),
        turn(4, Speaker::Tool, vec![result(None, Some("2"))]),
    ];
    assert_eq!(Analysis::new(turns, Pairing::Id, &[]).step(), Some(3));
    assert_eq!(
        Analysis::new(
            vec![turn(0, Speaker::Model, vec![text()])],
            Pairing::Id,
            &[]
        )
        .step(),
        None
    );
}

// spec: protocol/request_shapes/analysis_spec.rb:62 wants the first call of each step in the current turn signed, when the protocol requires it
#[test]
fn analysis_wants_signed_steps() {
    let turns = vec![
        turn(0, Speaker::User, vec![text()]),
        turn(1, Speaker::Model, vec![call("a", None, false)]),
        turn(2, Speaker::Tool, vec![result(Some("a"), None)]),
        turn(3, Speaker::User, vec![text()]),
        turn(
            4,
            Speaker::Model,
            vec![call("a", None, true), call("b", None, false)],
        ),
        turn(
            5,
            Speaker::Tool,
            vec![result(Some("a"), None), result(Some("b"), None)],
        ),
        turn(6, Speaker::Model, vec![text(), call("c", None, false)]),
    ];
    assert_eq!(
        problem_hashes(turns, Pairing::Position, &[Rule::SignedSteps]),
        [
            json!({ "kind": "unpaired_round", "turn": 6 }),
            json!({ "kind": "unsigned_call", "turn": 6, "part": 1 })
        ]
    );
}

// spec: protocol/request_shapes/analysis_spec.rb:71 finds parts with no data, empty turns, unsigned thinking, and roles out of order
#[test]
fn analysis_finds_the_opt_in_problems() {
    let thinking = PartSpec {
        measure: Some(4),
        ..PartSpec::new(PartKind::Thinking)
    };
    let empty_thought = PartSpec {
        signed: true,
        without_data: true,
        ..PartSpec::new(PartKind::Thinking)
    };
    let turns = vec![
        turn(0, Speaker::System, vec![text()]),
        turn(1, Speaker::Model, vec![thinking]),
        turn(2, Speaker::User, vec![]),
        turn(3, Speaker::User, vec![empty_thought]),
    ];
    assert_eq!(
        problem_hashes(
            turns,
            Pairing::Id,
            &[Rule::SignedThinking, Rule::AlternatingRoles]
        ),
        [
            json!({ "kind": "role_order", "turn": 1 }),
            json!({ "kind": "unsigned_thinking", "turn": 1, "part": 0 }),
            json!({ "kind": "empty_turn", "turn": 2 }),
            json!({ "kind": "role_order", "turn": 3 }),
            json!({ "kind": "part_without_data", "turn": 3, "part": 0 })
        ]
    );
}

// spec: protocol/request_shapes/analysis_spec.rb:81 keeps the first 10 and last 50 turns of a long conversation, and finds problems in all of them
#[test]
fn analysis_keeps_first_and_last_turns() {
    let mut turns: Vec<TurnSpec> = (0..70)
        .map(|i| {
            turn(
                i,
                if i % 2 == 1 {
                    Speaker::Model
                } else {
                    Speaker::User
                },
                vec![text()],
            )
        })
        .collect();
    turns[31] = turn(31, Speaker::Model, vec![call("lookup", Some("x"), false)]);
    let analysis = Analysis::new(turns, Pairing::Id, &[]);
    let kept: Vec<usize> = analysis.kept_turns().iter().map(|t| t.index).collect();
    assert_eq!(kept, (0..10).chain(20..70).collect::<Vec<_>>());
    assert_eq!(
        analysis
            .tool_rounds()
            .iter()
            .map(|r| r.turn)
            .collect::<Vec<_>>(),
        [31]
    );
    assert_eq!(
        analysis
            .problems()
            .iter()
            .map(|p| p.turn)
            .collect::<Vec<_>>(),
        [Some(31)]
    );
}

// ---- protocol/request_shapes_spec.rb --------------------------------------------------------
//
// Ruby defines an ad-hoc protocol reading `payload['media']` to exercise the `shape_*` helpers.
// The Rust helpers are private to the protocol readers, so these run the same inputs through
// the Chat Completions reader, whose `input_audio` and data-URI parts use the same helpers.

fn media_payload(items: &[(&str, &str)]) -> Value {
    let content: Vec<Value> = items
        .iter()
        .map(|(data, mime)| {
            json!({ "type": "file", "file": { "file_data": format!("data:{mime};base64,{data}") } })
        })
        .collect();
    json!({ "messages": [{ "role": "user", "content": content }] })
}

// spec: protocol/request_shapes_spec.rb:22 measures base64 data in decoded bytes
#[test]
fn shapes_measure_base64_in_decoded_bytes() {
    let png = b64(b"secret");
    let wav = b64(b"secret!");
    let pdf = b64_lines("secret".repeat(20).as_bytes());
    let mut payload = media_payload(&[
        (&png, "image/png"),
        (&wav, "audio/wav"),
        (&pdf, "application/pdf"),
    ]);
    payload["messages"][0]["content"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "type": "file", "file": { "file_data": null } }));
    payload["tools"] =
        json!([{ "function": { "name": "lookup" } }, null, { "function": { "name": "" } }]);
    let shape = shape(ProtocolName::ChatCompletions, payload).unwrap();
    assert_eq!(
        turns_of(&shape),
        [
            "#0 user: image/png (6 bytes), audio/wav (7 bytes), application/pdf (120 bytes), document (url)"
        ]
    );
    assert_eq!(shape.tool_names, ["lookup"]);
}

// spec: protocol/request_shapes_spec.rb:36 keeps only thinking settings that are words and numbers
#[test]
fn shapes_keep_only_word_and_number_settings() {
    let payload = json!({ "messages": [],
                          "reasoning": { "effort": "high", "max_tokens": 2048, "exclude": "think about the secret plan" } });
    assert_eq!(
        shape(ProtocolName::ChatCompletions, payload)
            .unwrap()
            .thinking_settings,
        settings(json!({ "effort": "high", "max_tokens": 2048 }))
    );
}

// spec: protocol/request_shapes_spec.rb:43 Protocol#request_shape > describes nothing for a protocol that renders no conversation
#[test]
fn shapes_describe_nothing_without_a_conversation() {
    // An embedding or other non-conversation payload: no protocol reader finds turns in it.
    for protocol in [
        ProtocolName::ChatCompletions,
        ProtocolName::Anthropic,
        ProtocolName::Gemini,
    ] {
        assert!(
            shape(protocol, json!({ "input": "secret", "model": "x" })).is_none()
                || protocol == ProtocolName::Responses
        );
    }
}

// spec: protocol/request_shapes_spec.rb:47 Protocol#request_shape > describes nothing for a payload that is not an object
#[test]
fn shapes_describe_nothing_for_a_non_object() {
    for payload in [json!(["secret"]), json!("secret"), Value::Null] {
        assert!(shape(ProtocolName::ChatCompletions, payload).is_none());
    }
}

// spec: protocol/request_shapes_spec.rb:53 Protocol#request_shape > reads a payload that repeats a key, as string-keyed provider options can make it
#[test]
fn shapes_read_a_payload_whose_key_was_merged() {
    // Ruby's `{ :media => [], 'media' => [...] }` stringifies to the later key. A JSON object has
    // one value per key; provider options deep-merged over the rendered payload win the same way.
    let mut payload = json!({ "messages": [] });
    rust_llm::protocols::deep_merge(
        &mut payload,
        &media_payload(&[(&b64(b"secret"), "image/png")]),
    );
    assert_eq!(
        turns_of(&shape(ProtocolName::ChatCompletions, payload).unwrap()),
        ["#0 user: image/png (6 bytes)"]
    );
}

// spec: protocol/request_shapes_spec.rb:59 Protocol#request_shape > describes nothing rather than raise when reading the payload fails
#[test]
fn shapes_never_fail() {
    // A Rust reader returns `Option`, so it cannot raise; the malformed payloads below are the
    // ones Ruby's readers must survive.
    for payload in malformed() {
        for protocol in all_protocols() {
            let _ = shape(protocol, payload.clone());
        }
    }
}

fn all_protocols() -> [ProtocolName; 6] {
    [
        ProtocolName::Gemini,
        ProtocolName::Interactions,
        ProtocolName::Anthropic,
        ProtocolName::ChatCompletions,
        ProtocolName::Responses,
        ProtocolName::Conversations,
    ]
}

fn malformed() -> Vec<Value> {
    vec![
        json!({ "contents": [null, "secret", 1, { "role": 7, "parts": "secret" },
                    { "parts": [null, "secret", { "functionCall": "secret", "inlineData": { "data": 5 } },
                                { "functionResponse": { "response": { "content": "secret" }, "parts": "secret" } }] }],
                "systemInstruction": "secret", "tools": "secret", "generationConfig": { "thinkingConfig": "secret" } }),
        json!({ "generateContentRequest": "secret", "contents": [{ "role": "user", "parts": [{ "text": 5 }] }] }),
        json!({ "messages": [null, "secret", { "role": [], "content": 5, "tool_calls": "secret", "reasoning_details": "secret" },
                    { "role": "tool", "content": [null, "secret", { "type": "image_url", "image_url": 5 }] },
                    { "role": "user", "content": [null, "secret", { "type": 5, "source": "secret", "image_url": 5 },
                                                  { "type": "tool_result", "content": [null, 5] },
                                                  { "type": "file_url", "file_url": 5 }] }],
                "system": 5, "tools": [null, "secret", { "function": "secret" }], "thinking": "secret", "reasoning": 5 }),
        json!({ "input": [null, "secret", { "type": 5 }, { "type": "reasoning", "summary": "secret" },
                    { "type": "function_call_output", "output": [null, { "type": 5 }] },
                    { "role": "user", "content": [null, { "type": "input_image", "image_url": 5 }, { "type": "input_file" }] }],
                "instructions": 5, "system_instruction": [], "tools": [null, "secret"], "generation_config": "secret" }),
        json!({ "inputs": [null, { "type": "message.input", "content": 5 }, { "type": "function.call", "arguments": [] },
                    { "type": "function.result", "result": { "secret": 5 } }], "completion_args": "secret" }),
    ]
}

// spec: protocol/request_shapes_spec.rb:106 payloads a request hook reshaped > describe what they can and never raise
#[test]
fn reshaped_payloads_describe_what_they_can() {
    for protocol in all_protocols() {
        for payload in malformed() {
            if let Some(shape) = shape(protocol, payload) {
                assert!(
                    !everything_shown(&shape).contains("secret"),
                    "{protocol:?}: {shape}"
                );
            }
        }
    }
}

// ---- error_spec.rb #request_shape -----------------------------------------------------------

/// A Gemini chat whose request is refused; the error carries the secret question's payload.
async fn refused_gemini(question: &str) -> Error {
    let server = refusing_server(400).await;
    let mut chat = Chat::with_config(
        config(&server),
        Some("gemini-2.5-flash"),
        Some("gemini"),
        false,
    )
    .unwrap();
    chat.add_message(Message::user(question));
    chat.complete().await.expect_err("refused")
}

// spec: error_spec.rb:84 #request_shape > describes the request through the protocol that rendered it
#[tokio::test]
async fn error_describes_the_request_through_its_protocol() {
    let error = refused_gemini("secret question").await;
    assert_eq!(
        turns_of(error.request_shape().unwrap()),
        ["#0 user: text (15 chars)"]
    );
}

// spec: error_spec.rb:88 #request_shape > describes nothing when no protocol rendered the request
#[test]
fn error_without_a_request_describes_nothing() {
    let error = Error::BadRequest("Invalid request".into(), None);
    assert!(error.request_shape().is_none());
}

// spec: error_spec.rb:92 #request_shape > reads the payload once
#[tokio::test]
async fn error_reads_the_payload_once() {
    let error = refused_gemini("secret question").await;
    let claimed = error.response().unwrap().request.as_ref().unwrap();
    assert!(!claimed.is_read());
    let first = error.request_shape().unwrap() as *const RequestShape;
    assert!(claimed.is_read());
    let second = error.request_shape().unwrap() as *const RequestShape;
    assert_eq!(first, second);
}

// spec: error_spec.rb:99 #request_shape > leaves the message and inspect as they were
#[tokio::test]
async fn error_message_and_debug_are_unchanged() {
    let error = refused_gemini("secret question").await;
    error.request_shape();
    assert_eq!(error.to_string(), "Request contains an invalid argument.");
    assert!(!format!("{error:?}").contains("secret"));
}

// ---- chat_request_shape_spec.rb --------------------------------------------------------------

fn gemini_turns() -> Vec<&'static str> {
    vec![
        "#0 user: text (15 chars), image/png (15941 bytes)",
        "#1 model: call lookup (args 27 chars), signed",
        "#2 user: result lookup (13 chars)",
        "#3 model: thinking (11 chars), text (13 chars), signed",
        "#4 user: text (16 chars)",
    ]
}

fn gemini_stream_body() -> String {
    format!(
        "data: {}\n\n",
        json!({ "candidates": [{ "content": { "role": "model", "parts": [{ "text": "Hi" }] } }] })
    )
}

// spec: chat_request_shape_spec.rb:23 describes a refused stream
#[tokio::test]
async fn chat_describes_a_refused_stream() {
    let server = refusing_server(400).await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let error = chat.complete_stream(|_| {}).await.expect_err("refused");
    assert_eq!(turns_of(error.request_shape().unwrap()), gemini_turns());
}

// spec: chat_request_shape_spec.rb:27 describes a refused token count
#[tokio::test]
async fn chat_describes_a_refused_token_count() {
    let server = refusing_server(400).await;
    let chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let error = chat.count_tokens(None).await.expect_err("refused");
    assert_eq!(error.kind(), ErrorKind::BadRequest);
    assert_eq!(turns_of(error.request_shape().unwrap()), gemini_turns());
}

// spec: chat_request_shape_spec.rb:35 describes a refused compaction
#[tokio::test]
async fn chat_describes_a_refused_compaction() {
    let server = refusing_server(400).await;
    let mut chat = secret_chat(&server, "openai", "gpt-5-nano", None);
    let error = chat.compact().await.expect_err("refused");
    assert_eq!(error.kind(), ErrorKind::BadRequest);
    let shape = error.request_shape().unwrap();
    assert_eq!(
        shape.turns[0].to_string(),
        "#0 user: text (15 chars), image/png (15941 bytes)"
    );
    assert!(shape.tool_names.is_empty());
}

// spec: chat_request_shape_spec.rb:45 describes the request behind any provider error
#[tokio::test]
async fn chat_describes_the_request_behind_any_error() {
    let server = refusing_server(500).await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let error = chat.complete().await.expect_err("refused");
    assert_eq!(error.kind(), ErrorKind::Server);
    assert_eq!(turns_of(error.request_shape().unwrap()), gemini_turns());
}

// spec: chat_request_shape_spec.rb:53 describes a request refused after a retry
#[tokio::test]
async fn chat_describes_a_request_refused_after_a_retry() {
    let server = spec_helpers::serve_templates(vec![
        ResponseTemplate::new(500).set_body_json(refusal()),
        ResponseTemplate::new(400).set_body_json(refusal()),
    ])
    .await;
    let mut c = (*config(&server)).clone();
    c.max_retries = 1;
    c.retry_interval = 0.0;
    let mut chat = Chat::with_config(Arc::new(c), Some("gemini-2.5-flash"), Some("gemini"), false)
        .unwrap()
        .with_instructions("secret instructions")
        .with_tool(Lookup);
    for m in secret_conversation("gemini", "gemini-2.5-flash") {
        chat.add_message(m);
    }
    let error = chat.complete().await.expect_err("refused");
    assert_eq!(error.kind(), ErrorKind::BadRequest);
    assert_eq!(turns_of(error.request_shape().unwrap()), gemini_turns());
}

// spec: chat_request_shape_spec.rb:62 describes a stream the provider refuses after it starts
#[tokio::test]
async fn chat_describes_a_stream_refused_after_it_starts() {
    let body = format!("data: {}\n\n", refusal());
    let server = spec_helpers::serve_templates(vec![spec_helpers::sse(body)]).await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let error = chat.complete_stream(|_| {}).await.expect_err("refused");
    assert_eq!(turns_of(error.request_shape().unwrap()), gemini_turns());
}

// spec: chat_request_shape_spec.rb:70 describes a stream that ends with a failed response event
#[tokio::test]
async fn chat_describes_a_stream_that_ends_with_a_failed_response() {
    let failed = json!({ "type": "response.failed",
                         "response": { "status": "failed", "output": [], "model": "gpt-5-nano",
                                       "error": { "code": "invalid_image", "message": "Invalid image." } } });
    let body = format!("event: response.failed\ndata: {failed}\n\n");
    let server = spec_helpers::serve_templates(vec![spec_helpers::sse(body)]).await;
    let mut chat = secret_chat(&server, "openai", "gpt-5-nano", None);
    let error = chat.complete_stream(|_| {}).await.expect_err("failed");
    assert_eq!(error.kind(), ErrorKind::BadRequest);
    assert_eq!(error.to_string(), "Invalid image.");
    assert_eq!(error.request_shape().unwrap().turn_count, 7);
}

// spec: chat_request_shape_spec.rb:82 counts the model calls of the turn the refused request continues
#[tokio::test]
async fn chat_counts_the_model_calls_of_the_turn() {
    let server = refusing_server(400).await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let mut asks = Message::new(Role::Assistant, None);
    asks.tool_calls = Some(calls(ToolCall::new("secret-call-2", "lookup", Map::new())));
    chat.add_message(asks);
    let mut result = Message::new(Role::Tool, Some("secret".into()));
    result.tool_call_id = Some("secret-call-2".into());
    chat.add_message(result);
    let shape = refused_shape(&mut chat).await;
    assert_eq!(shape.step, Some(2));
    assert_eq!(
        shape.to_string().lines().next().unwrap(),
        "gemini, gemini-2.5-flash, model call 2 of the turn, after 1 tool round"
    );
}

// spec: chat_request_shape_spec.rb:93 keeps the first and last turns of a long conversation
#[tokio::test]
async fn chat_keeps_the_first_and_last_turns() {
    let server = refusing_server(400).await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    for turn in 0..60 {
        chat.add_message(if turn % 2 == 0 {
            Message::assistant("secret")
        } else {
            Message::user("secret")
        });
    }
    let shape = refused_shape(&mut chat).await;
    assert_eq!((shape.turn_count, shape.omitted_turns), (65, 5));
    let kept: Vec<usize> = shape.turns.iter().map(|t| t.index).collect();
    assert_eq!(kept, (0..10).chain(15..65).collect::<Vec<_>>());
    assert!(
        shape
            .to_string()
            .contains("#9 model: text (6 chars)\n(5 turns omitted)\n#15 model: text (6 chars)")
    );
    expect_no_secrets(&shape);
}

// spec: chat_request_shape_spec.rb:103 keeps the message the provider gave
#[tokio::test]
async fn chat_keeps_the_providers_message() {
    let server = refusing_server(400).await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let error = chat.complete().await.expect_err("refused");
    error.request_shape();
    assert_eq!(error.to_string(), "Request contains an invalid argument.");
    assert_eq!(error.kind(), ErrorKind::BadRequest);
}

// spec: chat_request_shape_spec.rb:113 leaves an error the streaming block raises to the operation that raised it
#[tokio::test]
async fn chat_leaves_an_inner_operations_error_to_it() {
    // The streaming block runs another chat whose request is refused: its error keeps the inner
    // chat's request, not the outer one the block was streaming.
    let server = spec_helpers::serve_templates(vec![
        spec_helpers::sse(gemini_stream_body()),
        ResponseTemplate::new(400).set_body_json(refusal()),
    ])
    .await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let mut other = Chat::with_config(
        config(&server),
        Some("gemini-2.5-flash"),
        Some("gemini"),
        false,
    )
    .unwrap();
    other.add_message(Message::user("secret?"));
    let mut inner: Option<Error> = None;
    chat.complete_stream(|_| {}).await.expect("outer stream");
    if let Err(e) = other.complete().await {
        inner = Some(e);
    }
    let inner = inner.expect("inner refusal");
    assert_eq!(
        turns_of(inner.request_shape().unwrap()),
        ["#0 user: text (7 chars)"]
    );

    // An embedding sends no conversation, so its error describes nothing.
    let embed_server = refusing_server(400).await;
    let embed = rust_llm::embed(
        "secret",
        rust_llm::EmbedOptions {
            model: Some("gemini-embedding-001"),
            provider: Some("gemini"),
            config: Some(config(&embed_server)),
            ..Default::default()
        },
    )
    .await
    .expect_err("refused");
    assert!(embed.request_shape().is_none());
}

// spec: chat_request_shape_spec.rb:128 describes nothing for an error raised before the request goes out
#[tokio::test]
async fn chat_describes_nothing_before_the_request() {
    let server = refusing_server(400).await;
    let mut chat = secret_chat(&server, "gemini", "gemini-2.5-flash", None);
    let error = chat
        .ask_with("secret", vec![Attachment::new(fixture("sample.docx"))])
        .await
        .expect_err("unsupported");
    assert!(
        matches!(error, Error::UnsupportedAttachment(_)),
        "{error:?}"
    );
    assert!(error.request_shape().is_none());
}

// spec: chat_request_shape_spec.rb:137 describes nothing for an operation that sends no conversation
#[tokio::test]
async fn chat_describes_nothing_for_an_embedding() {
    let server = refusing_server(400).await;
    let error = rust_llm::embed(
        "secret",
        rust_llm::EmbedOptions {
            model: Some("gemini-embedding-001"),
            provider: Some("gemini"),
            config: Some(config(&server)),
            ..Default::default()
        },
    )
    .await
    .expect_err("refused");
    assert_eq!(error.kind(), ErrorKind::BadRequest);
    assert!(error.request_shape().is_none());
}

// ---- protocols/*/request_shapes_spec.rb -----------------------------------------------------

// spec: protocols/anthropic/request_shapes_spec.rb:10 describes a refused conversation without its contents
#[tokio::test]
async fn anthropic_describes_a_refused_conversation() {
    let shape = shape_of_secret_chat("anthropic", "claude-haiku-4-5", None).await;
    assert_eq!(
        shape.to_string(),
        format!(
            "anthropic, {}, model call 1 of the turn\n\
             payload keys: model, messages, stream, max_tokens, tools, system\n\
             instructions: text (19 chars)\n\
             #0 user: text (15 chars), image/png (15941 bytes)\n\
             #1 assistant: thinking (14 chars), signed, call lookup (args 27 chars)\n\
             #2 user: result lookup (13 chars)\n\
             #3 assistant: thinking (11 chars), signed, text (13 chars)\n\
             #4 user: text (16 chars)\n\
             tools: lookup\n\
             tool round at #1: 1 call, 1 result, paired\n\
             no problems found",
            shape.model.as_deref().unwrap()
        )
    );
    expect_no_secrets(&shape);
}

// spec: protocols/anthropic/request_shapes_spec.rb:30 finds thinking without its signature, and a result that answers no call
#[test]
fn anthropic_finds_unsigned_thinking_and_unmatched_results() {
    let image = json!({ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": b64(b"secret") } });
    let payload = json!({
        "messages": [
            { "role": "user", "content": "secret" },
            { "role": "assistant", "content": [{ "type": "thinking", "thinking": "secret" },
                                               { "type": "tool_use", "id": "toolu_secret", "name": "lookup", "input": {} }] },
            { "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "toolu_secret", "content": [{ "type": "text", "text": "secret" }, image] },
                { "type": "tool_result", "tool_use_id": "toolu_unknown", "content": "secret!" }
            ] }
        ],
        "thinking": { "type": "enabled", "budget_tokens": 2048 }, "output_config": { "effort": "high" }
    });
    let shape = shape(ProtocolName::Anthropic, payload).unwrap();
    assert_eq!(
        shape.turns[2].to_string(),
        "#2 user: result lookup (6 chars), image/png (6 bytes), result (7 chars)"
    );
    assert_eq!(
        problems_of(&shape),
        [
            "problem at #1: 1 call but 2 results",
            "problem at #1, part 0: thinking part has no signature",
            "problem at #2, part 2: result answers no call in the request (call id toolu_unknown)"
        ]
    );
    assert_eq!(
        shape.thinking_settings,
        settings(json!({ "type": "enabled", "budget_tokens": 2048, "effort": "high" }))
    );
}

// spec: protocols/anthropic/request_shapes_spec.rb:56 describes redacted thinking, every document source, and provider tool blocks
#[test]
fn anthropic_describes_redacted_thinking_documents_and_provider_tools() {
    let payload = json!({
        "system": "secret",
        "messages": [
            { "role": "user", "content": [
                { "type": "image", "source": { "type": "url", "url": "https://example.com/secret.png" } },
                { "type": "document", "source": { "type": "file", "file_id": "file_secret" } },
                { "type": "document", "source": { "type": "text", "media_type": "text/plain", "data": "secret" } },
                { "type": "document", "source": { "type": "base64", "media_type": "application/pdf", "data": b64(b"secret!") } }
            ] },
            { "role": "assistant", "content": [
                { "type": "redacted_thinking", "data": "secret" },
                { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": { "query": "secret" } },
                { "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [] }
            ] },
            { "role": "user", "content": "secret" }
        ],
        "tools": [{ "name": "lookup", "input_schema": {} }, { "type": "web_search_20260318", "name": "web_search" },
                  { "type": "mcp_toolset", "mcp_server_name": "docs" }]
    });
    let shape = shape(ProtocolName::Anthropic, payload).unwrap();
    assert_eq!(
        turns_of(&shape),
        [
            "#0 user: image (url), document (file), text/plain (6 chars), application/pdf (7 bytes)",
            "#1 assistant: thinking (no text), signed, server_tool_use, web_search_tool_result",
            "#2 user: text (6 chars)"
        ]
    );
    assert_eq!(shape.tool_names, ["lookup", "web_search", "mcp_toolset"]);
    assert!(shape.problems.is_empty());
    expect_no_secrets(&shape);
}

// spec: protocols/chat_completions/request_shapes_spec.rb:10 describes a refused conversation without its contents
#[tokio::test]
async fn chat_completions_describes_a_refused_conversation() {
    let shape = shape_of_secret_chat("deepseek", "deepseek-v4-flash", None).await;
    assert_eq!(
        shape.to_string(),
        "deepseek, deepseek-v4-flash, model call 1 of the turn\n\
         payload keys: model, messages, stream, tools\n\
         #0 system: text (19 chars)\n\
         #1 user: text (15 chars), image/png (15941 bytes)\n\
         #2 assistant: thinking (14 chars), signed, text (0 chars), call lookup (args 27 chars), signed\n\
         #3 tool: result lookup (13 chars)\n\
         #4 assistant: thinking (11 chars), signed, text (13 chars)\n\
         #5 user: text (16 chars)\n\
         tools: lookup\n\
         tool round at #2: 1 call, 1 result, paired\n\
         no problems found"
    );
    expect_no_secrets(&shape);
}

// spec: protocols/chat_completions/request_shapes_spec.rb:41 describes linked media, uploaded files, audio, reasoning details, and a result that answers no call
#[test]
fn chat_completions_describes_media_files_audio_and_reasoning() {
    let pdf = format!("data:application/pdf;base64,{}", b64(b"secret"));
    let payload = json!({
        "messages": [
            { "role": "developer", "content": "secret" },
            { "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": "https://example.com/secret.png" } },
                { "type": "file", "file": { "file_id": "file-secret" } },
                { "type": "file", "file": { "filename": "secret.pdf", "file_data": pdf } },
                { "type": "input_audio", "input_audio": { "data": b64(b"secret!!"), "format": "wav" } }
            ] },
            { "role": "assistant", "content": null, "reasoning_details": [
                { "type": "reasoning.text", "text": "secret", "signature": "secret" },
                { "type": "reasoning.encrypted", "data": "secret" }
            ], "tool_calls": [{ "id": "secret-call", "type": "function", "function": { "name": "lookup", "arguments": "{}" } }] },
            { "role": "tool", "tool_call_id": "secret-call", "content": [{ "type": "text", "text": "secret" }] },
            { "role": "tool", "tool_call_id": "call_unknown", "content": "secret!" }
        ],
        "tools": [{ "type": "function", "function": { "name": "lookup" } }, { "type": "web_search" }],
        "reasoning_effort": "high"
    });
    let shape = shape(ProtocolName::ChatCompletions, payload).unwrap();
    assert_eq!(
        turns_of(&shape),
        [
            "#0 developer: text (6 chars)",
            "#1 user: image (url), document (file), application/pdf (6 bytes), audio (8 bytes)",
            "#2 assistant: thinking (6 chars), signed, thinking (no text), signed, call lookup (args 2 chars)",
            "#3 tool: result lookup (6 chars)",
            "#4 tool: result (7 chars)"
        ]
    );
    assert_eq!(
        problems_of(&shape),
        [
            "problem at #2: 1 call but 2 results",
            "problem at #4, part 0: result answers no call in the request (call id call_unknown)"
        ]
    );
    assert_eq!(
        shape.thinking_settings,
        settings(json!({ "reasoning_effort": "high" }))
    );
}

// spec: protocols/chat_completions/request_shapes_spec.rb:78 describes the documents Mistral and Perplexity render
#[tokio::test]
async fn chat_completions_describes_mistral_and_perplexity_documents() {
    for (provider, model, document) in [
        (
            "mistral",
            "mistral-small-latest",
            "application/pdf (18810 bytes)",
        ),
        ("perplexity", "openai/gpt-5-mini", "document (18810 bytes)"),
    ] {
        let server = refusing_server(400).await;
        let mut chat = Chat::with_config(config(&server), Some(model), Some(provider), false)
            .unwrap()
            .with_protocol(ProtocolName::ChatCompletions);
        let mut m = Message::user("secret");
        m.attachments = vec![Attachment::new(fixture("sample.pdf"))];
        chat.add_message(m);
        let shape = refused_shape(&mut chat).await;
        assert_eq!(
            turns_of(&shape),
            [format!("#0 user: text (6 chars), {document}")],
            "{provider}"
        );
    }
}

// spec: protocols/chat_completions/request_shapes_spec.rb:89 describes the documents Mistral and Perplexity send by link or as data
#[test]
fn chat_completions_describes_documents_by_link_or_data() {
    let pdf = b64(b"secret!");
    let payload = json!({ "messages": [{ "role": "user", "content": [
        { "type": "document_url", "document_url": format!("data:application/pdf;base64,{pdf}") },
        { "type": "document_url", "document_url": "https://example.com/secret.pdf" },
        { "type": "file_url", "file_url": { "url": pdf } },
        { "type": "file_url", "file_url": { "url": "https://example.com/secret.pdf" } }
    ] }] });
    assert_eq!(
        turns_of(&shape(ProtocolName::ChatCompletions, payload).unwrap()),
        ["#0 user: application/pdf (7 bytes), document (url), document (7 bytes), document (url)"]
    );
}

// spec: protocols/gemini/request_shapes_spec.rb:13 describes a refused conversation without its contents
#[tokio::test]
async fn gemini_describes_a_refused_conversation() {
    let shape = shape_of_secret_chat("gemini", "gemini-2.5-flash", None).await;
    assert_eq!(
        shape.to_string(),
        "gemini, gemini-2.5-flash, model call 1 of the turn\n\
         payload keys: contents, generationConfig, systemInstruction, tools\n\
         instructions: text (19 chars)\n\
         #0 user: text (15 chars), image/png (15941 bytes)\n\
         #1 model: call lookup (args 27 chars), signed\n\
         #2 user: result lookup (13 chars)\n\
         #3 model: thinking (11 chars), text (13 chars), signed\n\
         #4 user: text (16 chars)\n\
         tools: lookup\n\
         tool round at #1: 1 call, 1 result, paired\n\
         no problems found"
    );
    expect_no_secrets(&shape);
}

// spec: protocols/gemini/request_shapes_spec.rb:32 finds a thought part that carries only a signature, which Gemini refuses
#[test]
fn gemini_finds_a_signature_only_thought() {
    let payload = json!({
        "contents": [
            { "role": "user", "parts": [{ "text": "secret" }] },
            { "role": "model", "parts": [{ "thought": true, "thoughtSignature": "secret" }, { "text": "secret" }] },
            { "role": "user", "parts": [{ "text": "secret" }] }
        ],
        "generationConfig": { "thinkingConfig": { "includeThoughts": true, "thinkingBudget": -1 } }
    });
    let shape = shape(ProtocolName::Gemini, payload).unwrap();
    assert_eq!(
        shape.turns[1].to_string(),
        "#1 model: thinking (no text), signed, text (6 chars)"
    );
    assert_eq!(
        problems_of(&shape),
        ["problem at #1, part 0: thinking part carries no data"]
    );
    assert_eq!(
        shape.thinking_settings,
        settings(json!({ "includeThoughts": true, "thinkingBudget": -1 }))
    );
    expect_no_secrets(&shape);
}

// spec: protocols/gemini/request_shapes_spec.rb:50 finds unsigned steps and function responses that do not answer their calls
#[test]
fn gemini_finds_unsigned_steps_and_unanswered_calls() {
    let payload = json!({ "contents": [
        { "role": "user", "parts": [{ "text": "secret" }] },
        { "role": "model", "parts": [{ "functionCall": { "name": "lookup", "args": {} } },
                                     { "functionCall": { "name": "fetch", "args": {} } }] },
        { "role": "user", "parts": [{ "functionResponse": { "name": "lookup", "response": { "content": "secret" } } }] }
    ] });
    let shape = shape(ProtocolName::Gemini, payload).unwrap();
    assert_eq!(shape.step, Some(2));
    assert_eq!(
        problems_of(&shape),
        [
            "problem at #1: 2 calls but 1 result",
            "problem at #1, part 0: first call of a step in the current turn has no signature"
        ]
    );
}

// spec: protocols/gemini/request_shapes_spec.rb:69 follows a tool result with the media Gemini 3 sends beside it
#[tokio::test]
async fn gemini_follows_a_result_with_its_media() {
    let server = refusing_server(400).await;
    let mut chat = Chat::with_config(
        config(&server),
        Some("gemini-3.1-pro-preview"),
        Some("gemini"),
        false,
    )
    .unwrap();
    chat.add_message(Message::user("secret"));
    let mut asks = Message::new(Role::Assistant, None);
    asks.tool_calls = Some(calls(ToolCall::new("secret-call", "lookup", Map::new())));
    chat.add_message(asks);
    let mut result = Message::new(Role::Tool, Some("secret result".into()));
    result.tool_call_id = Some("secret-call".into());
    result.attachments = vec![Attachment::new(fixture("ruby.png"))];
    chat.add_message(result);
    let shape = refused_shape(&mut chat).await;
    assert_eq!(
        shape.turns.last().unwrap().to_string(),
        "#2 user: result lookup (13 chars), image/png (15941 bytes)"
    );
    assert_eq!(shape.step, Some(2));
    assert!(shape.problems.is_empty(), "{shape}");
    expect_no_secrets(&shape);
}

// spec: protocols/gemini/request_shapes_spec.rb:85 describes parts a model returned in camel case, and the request a token count wraps
#[test]
fn gemini_describes_camel_case_parts_and_token_count_requests() {
    let audio = json!({ "inlineData": { "mimeType": "audio/wav", "data": b64(b"secret") } });
    let pdf = json!({ "fileData": { "mimeType": "application/pdf", "fileUri": "https://example.com/secret" } });
    let payload = json!({ "generateContentRequest": {
        "contents": [
            { "role": "user", "parts": [audio, pdf] },
            { "role": "model", "parts": [{ "executableCode": { "code": "print(\"secret\")" } }, { "thoughtSignature": "secret" }] }
        ],
        "tools": [{ "functionDeclarations": [{ "name": "lookup" }] }, { "google_search": {} }]
    } });
    let shape = shape(ProtocolName::Gemini, payload).unwrap();
    assert_eq!(
        turns_of(&shape),
        [
            "#0 user: audio/wav (6 bytes), application/pdf (file)",
            "#1 model: executableCode, part of no known kind, signed"
        ]
    );
    assert_eq!(shape.payload_keys, ["generateContentRequest"]);
    assert_eq!(shape.tool_names, ["lookup", "google_search"]);
    assert_eq!(
        problems_of(&shape),
        ["problem at #1, part 1: part carries no data"]
    );
    expect_no_secrets(&shape);
}

// spec: protocols/gemini/request_shapes_spec.rb:108 describes nothing in a request that holds no conversation
#[test]
fn gemini_describes_nothing_without_contents() {
    assert!(
        shape(
            ProtocolName::Gemini,
            json!({ "requests": [{ "content": { "parts": [{ "text": "secret" }] } }] })
        )
        .is_none()
    );
    assert!(shape(ProtocolName::Gemini, json!({ "contents": "secret" })).is_none());
}

// spec: protocols/interactions/request_shapes_spec.rb:10 describes a refused conversation without its contents
#[tokio::test]
async fn interactions_describes_a_refused_conversation() {
    let shape = shape_of_secret_chat(
        "gemini",
        "gemini-3.8-flash",
        Some(ProtocolName::Interactions),
    )
    .await;
    assert_eq!(
        shape.to_string(),
        "gemini, gemini-3.8-flash, model call 1 of the turn\n\
         payload keys: model, input, stream, store, system_instruction, generation_config, tools\n\
         instructions: text (19 chars)\n\
         #0 user_input: text (15 chars), image/png (15941 bytes)\n\
         #1 function_call: call lookup (args 27 chars), signed\n\
         #2 function_result: result lookup (13 chars)\n\
         #3 model_output: text (13 chars)\n\
         #4 user_input: text (16 chars)\n\
         tools: lookup\n\
         tool round at #1: 1 call, 1 result, paired\n\
         no problems found"
    );
    expect_no_secrets(&shape);
}

// spec: protocols/interactions/request_shapes_spec.rb:30 describes replayed steps and linked media, and finds an unsigned step
#[test]
fn interactions_describes_steps_media_and_unsigned_steps() {
    let payload = json!({
        "system_instruction": "",
        "input": [
            { "type": "user_input", "content": [
                { "type": "video", "mime_type": "video/mp4", "uri": "https://example.com/secret.mp4" },
                { "type": "document", "mime_type": "application/pdf", "data": b64(b"secret") }
            ] },
            { "type": "thought", "summary": [{ "type": "text", "text": "secret" }], "signature": "secret" },
            { "type": "google_search_call", "id": "search_1", "arguments": { "queries": ["secret"] }, "signature": "secret" },
            { "type": "google_search_result", "call_id": "search_1", "result": [] },
            { "type": "function_call", "id": "secret-call", "name": "lookup", "arguments": {} },
            { "type": "function_result", "call_id": "secret-call", "result": [{ "type": "text", "text": "secret" }] }
        ],
        "tools": [{ "type": "function", "name": "lookup" }, { "type": "google_search" }],
        "generation_config": { "thinking_level": "high", "thinking_summaries": "auto" }
    });
    let shape = shape(ProtocolName::Interactions, payload).unwrap();
    let text = shape.to_string();
    let lines: Vec<&str> = text.lines().collect();
    for line in [
        "instructions: text (0 chars)",
        "#0 user_input: video/mp4 (url), application/pdf (6 bytes)",
        "#1 thought: thinking (6 chars), signed",
        "#2 google_search_call: google_search_call, signed",
        "#5 function_result: result lookup (6 chars)",
        "thinking: thinking_level high, thinking_summaries auto",
    ] {
        assert!(lines.contains(&line), "{line}\n{text}");
    }
    assert_eq!(
        problems_of(&shape),
        ["problem at #4, part 0: first call of a step in the current turn has no signature"]
    );
    expect_no_secrets(&shape);
}

// spec: protocols/responses/request_shapes_spec.rb:10 describes a refused conversation without its contents
#[tokio::test]
async fn responses_describes_a_refused_conversation() {
    let shape = shape_of_secret_chat("openai", "gpt-5-nano", None).await;
    assert_eq!(
        shape.to_string(),
        "openai, gpt-5-nano, model call 1 of the turn\n\
         payload keys: model, input, instructions, stream, store, include, tools\n\
         instructions: text (19 chars)\n\
         #0 user: text (15 chars), image/png (15941 bytes)\n\
         #1 reasoning: thinking (14 chars), signed\n\
         #2 function_call: call lookup (args 27 chars)\n\
         #3 function_call_output: result lookup (13 chars)\n\
         #4 reasoning: thinking (11 chars), signed\n\
         #5 assistant: text (13 chars)\n\
         #6 user: text (16 chars)\n\
         tools: lookup\n\
         tool round at #1: 1 call, 1 result, paired\n\
         no problems found"
    );
    expect_no_secrets(&shape);
}

// spec: protocols/responses/request_shapes_spec.rb:31 describes replayed output items, every file input, and outputs that answer no call
#[test]
fn responses_describes_items_files_and_unanswered_outputs() {
    let image = format!("data:image/png;base64,{}", b64(b"secret"));
    let payload = json!({
        "input": [
            { "role": "user", "content": [
                { "type": "input_image", "file_id": "file-secret" },
                { "type": "input_file", "file_url": "https://example.com/secret.pdf" },
                { "type": "input_file", "filename": "secret.pdf",
                  "file_data": format!("data:application/pdf;base64,{}", b64(b"secret")) }
            ] },
            { "type": "reasoning", "summary": [], "encrypted_content": "secret" },
            { "type": "web_search_call", "id": "ws_1", "status": "completed", "action": { "query": "secret" } },
            { "type": "function_call", "call_id": "secret-call", "name": "lookup", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "secret-call",
              "output": [{ "type": "input_text", "text": "secret" }, { "type": "input_image", "image_url": image }] },
            { "type": "function_call_output", "call_id": "call_unknown", "output": "secret" },
            { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "secret", "annotations": [] }] }
        ],
        "tools": [{ "type": "function", "name": "lookup" }, { "type": "web_search" }],
        "reasoning": { "effort": "medium", "summary": "auto" }
    });
    let shape = shape(ProtocolName::Responses, payload).unwrap();
    assert_eq!(
        turns_of(&shape),
        [
            "#0 user: image (file), document (url), application/pdf (6 bytes)",
            "#1 reasoning: thinking (no text), signed",
            "#2 web_search_call: web_search_call",
            "#3 function_call: call lookup (args 2 chars)",
            "#4 function_call_output: result lookup (6 chars), image/png (6 bytes)",
            "#5 function_call_output: result (6 chars)",
            "#6 assistant: text (6 chars)"
        ]
    );
    assert_eq!(
        problems_of(&shape),
        [
            "problem at #1: 1 call but 2 results",
            "problem at #5, part 0: result answers no call in the request (call id call_unknown)"
        ]
    );
    assert_eq!(
        shape.thinking_settings,
        settings(json!({ "effort": "medium", "summary": "auto" }))
    );
}

// spec: protocols/responses/request_shapes_spec.rb:68 describes nothing in a request that holds no conversation
#[test]
fn responses_describes_nothing_without_input_items() {
    assert!(
        shape(
            ProtocolName::Responses,
            json!({ "model": "gpt-5-nano", "input": "secret" })
        )
        .is_none()
    );
}

// spec: protocols/mistral/conversations/request_shapes_spec.rb:12 describes a refused conversation without its contents
#[tokio::test]
async fn conversations_describes_a_refused_conversation() {
    let shape = shape_of_secret_chat(
        "mistral",
        "mistral-small-latest",
        Some(ProtocolName::Conversations),
    )
    .await;
    assert_eq!(
        shape.to_string(),
        "mistral, mistral-small-latest, model call 1 of the turn\n\
         payload keys: model, inputs, instructions, completion_args, tools, store, stream\n\
         instructions: text (19 chars)\n\
         #0 user: text (15 chars), image/png (15941 bytes)\n\
         #1 assistant: text (0 chars)\n\
         #2 function.call: call lookup (args 27 chars)\n\
         #3 function.result: result lookup (13 chars)\n\
         #4 assistant: text (13 chars)\n\
         #5 user: text (16 chars)\n\
         tools: lookup\n\
         tool round at #1: 1 call, 1 result, paired\n\
         no problems found"
    );
    expect_no_secrets(&shape);
}

// spec: protocols/mistral/conversations/request_shapes_spec.rb:33 describes replayed output entries, and a result that answers no call
#[test]
fn conversations_describes_output_entries_and_unanswered_results() {
    let payload = json!({
        "inputs": [
            { "type": "message.input", "role": "user", "content": "secret" },
            { "type": "message.output", "content": [{ "type": "thinking", "thinking": [{ "type": "text", "text": "secret" }] },
                                                    { "type": "text", "text": "secret!" }] },
            { "type": "function.result", "tool_call_id": "call_unknown", "result": "secret" },
            { "type": "agent.handoff", "from_agent_id": "ag_1", "to_agent_id": "ag_2" }
        ],
        "tools": [{ "type": "function", "function": { "name": "lookup" } }, { "type": "web_search" }],
        "completion_args": { "reasoning_effort": "high" }
    });
    let shape = shape(ProtocolName::Conversations, payload).unwrap();
    assert_eq!(
        turns_of(&shape),
        [
            "#0 user: text (6 chars)",
            "#1 message.output: thinking (6 chars), text (7 chars)",
            "#2 function.result: result (6 chars)",
            "#3 agent.handoff: agent.handoff"
        ]
    );
    assert_eq!(
        problems_of(&shape),
        ["problem at #2, part 0: result answers no call in the request (call id call_unknown)"]
    );
    assert_eq!(
        shape.thinking_settings,
        settings(json!({ "reasoning_effort": "high" }))
    );
}
