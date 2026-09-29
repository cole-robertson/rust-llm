//! `spec/ruby_llm/chat_tools_spec.rb`, replayed from RubyLLM's recorded cassettes: parallel and
//! parameterless streaming tool calls, params-DSL schemas (array, anyOf, object), thought
//! signatures, tool choice/calls control, string results, and tool results with attachments.
//! The tools are ported exactly, since their schemas are part of every recorded request body.

#[macro_use]
mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rust_llm::{
    Attachment, Chat, Parameter, Role, ThinkingConfig, Tool, ToolCall, ToolCalls, ToolChoice,
    ToolError, ToolResult,
};
use serde_json::{Map, Value, json};
use support::CHAT_MODELS;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn arg(args: &Map<String, Value>, key: &str) -> String {
    match args.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

struct Weather;

#[async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Gets current weather for a location".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("latitude").description("Latitude (e.g., 52.5200)"),
            Parameter::new("longitude").description("Longitude (e.g., 13.4050)"),
        ]
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok(format!(
            "Current weather at {}, {}: 15°C, Wind: 10 km/h",
            arg(&args, "latitude"),
            arg(&args, "longitude")
        )
        .into())
    }
}

struct BestLanguageToLearn;

#[async_trait]
impl Tool for BestLanguageToLearn {
    fn description(&self) -> String {
        "Gets the best language to learn".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok("Ruby".into())
    }
}

struct ContentReturningTool;

#[async_trait]
impl Tool for ContentReturningTool {
    fn description(&self) -> String {
        "Returns a processed string result".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("query").description("Query to process")]
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok(format!("Processed: {}", arg(&args, "query")).into())
    }
}

struct FileFetchTool;

#[async_trait]
impl Tool for FileFetchTool {
    fn description(&self) -> String {
        "Fetches a sample text file named ruby.txt".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::with_attachments(
            "Fetched the file.",
            vec![Attachment::new(fixture("ruby.txt"))],
        ))
    }
}

struct ImageFetchTool;

#[async_trait]
impl Tool for ImageFetchTool {
    fn description(&self) -> String {
        "Fetches the requested image".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::with_attachments(
            "Fetched the image.",
            vec![Attachment::new(fixture("ruby.png"))],
        ))
    }
}

struct PdfFetchTool;

#[async_trait]
impl Tool for PdfFetchTool {
    fn description(&self) -> String {
        "Fetches the requested PDF report".into()
    }
    async fn execute(&self, _: Map<String, Value>, _: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::with_attachments(
            "Fetched the report.",
            vec![Attachment::new(fixture("sample.pdf"))],
        ))
    }
}

/// `parameters do array :tags, of: :string, description: ... end`
struct ArrayParamsTool;

#[async_trait]
impl Tool for ArrayParamsTool {
    fn description(&self) -> String {
        "Uses params DSL array support".into()
    }
    fn parameters_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "tags": { "type": "array", "description": "List of tags to combine", "items": { "type": "string" } }
            },
            "required": ["tags"],
            "additionalProperties": false
        }))
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        let tags: Vec<String> = args
            .get("tags")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|t| t.as_str().unwrap_or_default().to_string())
            .collect();
        Ok(format!("Combined tags: {}", tags.join(", ")).into())
    }
}

/// `parameters do string :task, ...; any_of :status, ... do string enum: %w[pending done]; null end end`
struct AnyOfParamsTool;

#[async_trait]
impl Tool for AnyOfParamsTool {
    fn description(&self) -> String {
        "Uses params DSL any_of support".into()
    }
    fn parameters_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "task": { "type": "string", "description": "Task description" },
                "status": {
                    "description": "Optional task status",
                    "anyOf": [{ "type": "string", "enum": ["pending", "done"] }, { "type": "null" }]
                }
            },
            "required": ["task", "status"],
            "additionalProperties": false
        }))
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        Ok(format!(
            "Task \"{}\" status {}",
            arg(&args, "task"),
            arg(&args, "status")
        )
        .into())
    }
}

/// `parameters do object :window, ... do string :start, ...; string :end, ... end end`
struct ObjectParamsTool;

#[async_trait]
impl Tool for ObjectParamsTool {
    fn description(&self) -> String {
        "Uses params DSL object support".into()
    }
    fn parameters_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "window": {
                    "type": "object",
                    "properties": {
                        "start": { "type": "string", "description": "ISO start" },
                        "end": { "type": "string", "description": "ISO end" }
                    },
                    "required": ["start", "end"],
                    "additionalProperties": false,
                    "description": "Time window to schedule"
                }
            },
            "required": ["window"],
            "additionalProperties": false
        }))
    }
    async fn execute(
        &self,
        args: Map<String, Value>,
        _: &ToolCall,
    ) -> Result<ToolResult, ToolError> {
        let window = args
            .get("window")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Ok(format!(
            "Window from {} to {}",
            arg(&window, "start"),
            arg(&window, "end")
        )
        .into())
    }
}

fn check(cond: bool, what: impl Into<String>) -> Result<(), String> {
    if cond { Ok(()) } else { Err(what.into()) }
}

fn assistant_tool_call_messages(chat: &Chat) -> Vec<&rust_llm::Message> {
    chat.messages()
        .iter()
        .filter(|m| m.role == Role::Assistant && m.is_tool_call())
        .collect()
}

/// `last_tool_call(chat)`: the last call of the last assistant message that made calls.
fn last_tool_call(chat: &Chat) -> Option<ToolCall> {
    assistant_tool_call_messages(chat)
        .last()?
        .tool_calls
        .as_ref()?
        .values()
        .last()
        .cloned()
}

fn tool_called_flag(chat: Chat) -> (Chat, Arc<Mutex<bool>>) {
    let called = Arc::new(Mutex::new(false));
    let flag = called.clone();
    (
        chat.before_tool_call(move |_| *flag.lock().unwrap() = true),
        called,
    )
}

// ---- describe 'function calling' ----------------------------------------------------------

#[tokio::test]
async fn can_use_parallel_tool_calls() {
    each_model!(
        CHAT_MODELS,
        "chat function calling",
        "can use parallel tool calls",
        |chat, provider, model| {
            chat = chat.with_tool(Weather).with_tool(BestLanguageToLearn);
            let response = chat
            .ask("What's the weather in Berlin (52.5200, 13.4050) and what's the best language to learn?")
            .await
            .map_err(|e| e.to_string())?;
            check(
                response.content().contains("15"),
                format!("content {:?}", response.content),
            )?;
            check(response.content().contains("10"), "wind")?;
            check(response.content().contains("Ruby"), "ruby")?;
            check(
                chat.messages().len() >= 5,
                format!("{} messages", chat.messages().len()),
            )?;
            let calls: usize = assistant_tool_call_messages(&chat)
                .iter()
                .map(|m| m.tool_calls.as_ref().map_or(0, |c| c.len()))
                .sum();
            check(calls >= 2, format!("{calls} tool calls"))
        }
    );
}

#[tokio::test]
async fn can_use_tools_without_parameters_in_multi_turn_streaming_conversations() {
    each_model!(
        CHAT_MODELS,
        "chat function calling",
        "can use tools without parameters in multi-turn streaming conversations",
        |chat, provider, model| {
            chat = chat
                .with_tool(BestLanguageToLearn)
                .with_instructions("You must use tools whenever possible.");
            let mut chunks = 0;
            let r1 = chat
                .ask_stream(
                    "Call best_language_to_learn and repeat the programming language it returns.",
                    |_| chunks += 1,
                )
                .await
                .map_err(|e| e.to_string())?;
            check(chunks > 0, "chunks")?;
            check(
                r1.content().contains("Ruby"),
                format!("first {:?}", r1.content),
            )?;
            let r2 = chat
                .ask_stream("Call best_language_to_learn again and repeat the programming language it returns.", |_| chunks += 1)
                .await
                .map_err(|e| e.to_string())?;
            check(
                r2.content().contains("Ruby"),
                format!("second {:?}", r2.content),
            )
        }
    );
}

#[tokio::test]
async fn handles_array_params() {
    each_model!(
        CHAT_MODELS,
        "chat function calling",
        "handles array params",
        |chat, provider, model| {
            chat = chat.with_tool(ArrayParamsTool);
            chat.ask_later("Call the array params tool with tags [\"red\",\"blue\"] and tell me the combined tags.")
            .map_err(|e| e.to_string())?;
            chat.generate().await.map_err(|e| e.to_string())?;
            let call = last_tool_call(&chat).ok_or("no tool call")?;
            check(call.name == "array_params", format!("name {}", call.name))?;
            let mut tags: Vec<String> = call
                .arguments()
                .get("tags")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|t| t.as_str().map(str::to_string))
                .collect();
            tags.sort();
            check(tags == ["blue", "red"], format!("tags {tags:?}"))
        }
    );
}

#[tokio::test]
async fn handles_any_of_params() {
    each_model!(
        CHAT_MODELS,
        "chat function calling",
        "handles anyOf params",
        |chat, provider, model| {
            chat = chat.with_tool(AnyOfParamsTool);
            chat.ask_later("Call the any-of params tool for task \"Review PR\" with status \"pending\" and report the result.")
            .map_err(|e| e.to_string())?;
            chat.generate().await.map_err(|e| e.to_string())?;
            let call = last_tool_call(&chat).ok_or("no tool call")?;
            check(call.name == "any_of_params", format!("name {}", call.name))?;
            let args = call.arguments();
            check(
                args.get("task") == Some(&json!("Review PR")),
                format!("task {:?}", args.get("task")),
            )?;
            check(
                args.get("status") == Some(&json!("pending")),
                format!("status {:?}", args.get("status")),
            )
        }
    );
}

#[tokio::test]
async fn handles_object_params() {
    each_model!(
        CHAT_MODELS,
        "chat function calling",
        "handles object params",
        |chat, provider, model| {
            chat = chat.with_tool(ObjectParamsTool);
            chat.ask_later("Call the object params tool with window start 2025-01-01 and end 2025-01-02 and include the result.")
            .map_err(|e| e.to_string())?;
            chat.generate().await.map_err(|e| e.to_string())?;
            let call = last_tool_call(&chat).ok_or("no tool call")?;
            check(call.name == "object_params", format!("name {}", call.name))?;
            let window = call
                .arguments()
                .get("window")
                .and_then(Value::as_object)
                .cloned()
                .ok_or("no window")?;
            check(
                arg(&window, "start").starts_with("2025-01-01"),
                format!("start {:?}", window.get("start")),
            )?;
            check(
                arg(&window, "end").starts_with("2025-01-02"),
                format!("end {:?}", window.get("end")),
            )
        }
    );
}

// ---- describe 'function calling' / 'thought signatures' -----------------------------------

const SIGNATURE_MODELS: &[(&str, &str)] = &[("gemini", "gemini-3.1-pro-preview")];

#[tokio::test]
async fn includes_thought_signatures_for_tool_calls() {
    each_model!(
        SIGNATURE_MODELS,
        "chat function calling thought signatures",
        "includes thought signatures for tool calls",
        |chat, provider, model| {
            chat = chat
                .with_thinking(ThinkingConfig::effort("low"))
                .with_tool(Weather);
            let response = chat
                .ask("What's the weather in Berlin? (52.5200, 13.4050)")
                .await
                .map_err(|e| e.to_string())?;
            check(
                response.content().contains("15"),
                format!("content {:?}", response.content),
            )?;
            let call = chat
                .messages()
                .iter()
                .find_map(|m| {
                    m.tool_calls
                        .as_ref()
                        .filter(|c| !c.is_empty())
                        .and_then(|c| c.values().next().cloned())
                })
                .ok_or("no tool call")?;
            check(
                call.thought_signature
                    .as_deref()
                    .is_some_and(|s| !s.is_empty()),
                "thought signature",
            )
        }
    );
}

// ---- describe 'tool attachments' ----------------------------------------------------------

#[tokio::test]
async fn returns_text_and_attachments_from_tools() {
    each_model!(
        CHAT_MODELS,
        "chat tool attachments",
        "returns text and attachments from tools",
        |chat, provider, model| {
            chat = chat.with_tool(FileFetchTool);
            if matches!(provider, "ollama" | "gpustack") {
                chat = chat.with_temperature(0.0);
            }
            let response = chat
            .ask("Call file_fetch to get ruby.txt, then repeat the contents of the attached file.")
            .await
            .map_err(|e| e.to_string())?;
            let tool_message = chat
                .messages()
                .iter()
                .find(|m| m.is_tool_result())
                .ok_or("no tool result")?;
            check(
                tool_message.content() == "Fetched the file.",
                format!("tool content {:?}", tool_message.content),
            )?;
            let filename = tool_message
                .attachments
                .first()
                .and_then(|a| a.filename.as_deref());
            check(
                filename == Some("ruby.txt"),
                format!("filename {filename:?}"),
            )?;
            check(
                response.content().contains("Ruby is the best"),
                format!("content {:?}", response.content),
            )
        }
    );
}

// ---- describe 'multimodal tool attachments' -----------------------------------------------

/// `MULTIMODAL_TOOL_RESULT_MODELS` minus unsupported providers. The PDF example only exists
/// where the Ruby list doesn't say `pdf: false`, so it runs wherever a cassette was recorded.
const MULTIMODAL_TOOL_RESULT_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-haiku-4-5"),
    ("gemini", "gemini-3-flash-preview"),
    ("gemini", "gemini-2.5-flash"),
    ("mistral", "mistral-small-latest"),
    ("openai", "gpt-5-nano"),
    ("openrouter", "gemini-2.5-flash"),
    ("xai", "grok-4-1-fast-non-reasoning"),
];

#[tokio::test]
async fn describes_images_returned_from_tools() {
    each_model!(
        MULTIMODAL_TOOL_RESULT_MODELS,
        "chat multimodal tool attachments",
        "describes images returned from tools",
        |chat, provider, model| {
            chat = chat.with_tool(ImageFetchTool);
            let response = chat
                .ask("Use the image_fetch tool. Its result includes an image attachment. Inspect the attachment and describe its colors and shape.")
                .await
                .map_err(|e| e.to_string())?;
            let content = response.content().to_lowercase();
            check(
                ["ruby", "gem", "red"].iter().any(|w| content.contains(w)),
                format!("content {:?}", response.content),
            )
        }
    );
}

#[tokio::test]
async fn reads_pdfs_returned_from_tools() {
    each_model!(
        MULTIMODAL_TOOL_RESULT_MODELS,
        "chat multimodal tool attachments",
        "reads PDFs returned from tools",
        |chat, provider, model| {
            chat = chat.with_tool(PdfFetchTool);
            let response = chat
                .ask("Use the pdf_fetch tool, then quote the first sentence of the PDF body. Exclude the title and headings.")
                .await
                .map_err(|e| e.to_string())?;
            let content = response.content().to_lowercase();
            check(
                content.contains("simple pdf file") || content.contains("lorem ipsum"),
                format!("content {:?}", response.content),
            )
        }
    );
}

// ---- describe 'string tool results' -------------------------------------------------------

#[tokio::test]
async fn preserves_strings_returned_from_tools() {
    each_model!(
        CHAT_MODELS,
        "chat string tool results",
        "preserves strings returned from tools",
        |chat, provider, model| {
            chat = chat.with_tool(ContentReturningTool);
            chat.ask("Process this query: test data")
                .await
                .map_err(|e| e.to_string())?;
            let tool_message = chat
                .messages()
                .iter()
                .find(|m| m.role == Role::Tool)
                .ok_or("no tool message")?;
            check(
                tool_message.content.as_deref() == Some("Processed: test data"),
                format!("tool content {:?}", tool_message.content),
            )
        }
    );
}

// ---- describe 'tool choice and calls control' ---------------------------------------------

#[tokio::test]
async fn respects_choice_none() {
    each_model!(
        CHAT_MODELS,
        "chat tool choice and calls control",
        "respects choice: :none",
        |chat, provider, model| {
            chat = chat
                .with_tool(Weather)
                .with_tool_choice(ToolChoice::None)
                .map_err(|e| e.to_string())?;
            let (mut chat, called) = tool_called_flag(chat);
            let response = chat
                .ask("What's the weather in Berlin? (52.5200, 13.4050)")
                .await
                .map_err(|e| e.to_string())?;
            check(!*called.lock().unwrap(), "tool was called")?;
            check(response.role == Role::Assistant, "assistant message")
        }
    );
}

#[tokio::test]
async fn respects_choice_required_for_unrelated_queries() {
    each_model!(
        CHAT_MODELS,
        "chat tool choice and calls control",
        "respects choice: :required for unrelated queries",
        |chat, provider, model| {
            chat = chat
                .with_tool(Weather)
                .with_tool_choice(ToolChoice::Required)
                .map_err(|e| e.to_string())?
                .with_max_output_tokens(4096)
                .with_instructions(
                    "Your location is Berlin, at latitude 52.5200 and longitude 13.4050.",
                );
            // DeepSeek only allows forced tool choices with thinking disabled.
            if provider == "deepseek" {
                chat = chat.with_thinking(ThinkingConfig::off());
            }
            let (mut chat, called) = tool_called_flag(chat);
            chat.ask("When was the fall of Rome?")
                .await
                .map_err(|e| e.to_string())?;
            check(*called.lock().unwrap(), "tool was not called")
        }
    );
}

#[tokio::test]
async fn respects_specific_tool_choice() {
    each_model!(
        CHAT_MODELS,
        "chat tool choice and calls control",
        "respects specific tool choice",
        |chat, provider, model| {
            chat = chat
                .with_tool(Weather)
                .with_tool_choice(ToolChoice::Tool("weather".into()))
                .map_err(|e| e.to_string())?;
            if provider == "deepseek" {
                chat = chat.with_thinking(ThinkingConfig::off());
            }
            let (mut chat, called) = tool_called_flag(chat);
            chat.ask("What's the fall of Rome?")
                .await
                .map_err(|e| e.to_string())?;
            check(*called.lock().unwrap(), "tool was not called")
        }
    );
}

/// `parallel_model = provider == :openrouter ? model_for(:openrouter, :parallel_tools) : model`.
fn parallel_models() -> Vec<(&'static str, &'static str)> {
    CHAT_MODELS
        .iter()
        .map(|&(p, m)| {
            if p == "openrouter" {
                (p, "upstage/solar-pro4")
            } else {
                (p, m)
            }
        })
        .collect()
}

#[tokio::test]
async fn respects_calls_one_for_sequential_execution() {
    each_model!(
        &parallel_models(),
        "chat tool choice and calls control",
        "respects calls: :one for sequential execution",
        |chat, provider, model| {
            let violations = Arc::new(Mutex::new(Vec::new()));
            let seen = violations.clone();
            chat = chat
                .with_tool(Weather)
                .with_tool(BestLanguageToLearn)
                .with_tool_calls(ToolCalls::One)
                .with_instructions("You must use both the weather tool for Berlin (52.5200, 13.4050) and the best language tool.")
                .after_message(move |m| {
                    let n = m.tool_calls.as_ref().map_or(0, |c| c.len());
                    if m.is_tool_call() && n != 1 {
                        seen.lock().unwrap().push(n);
                    }
                });
            chat.ask("What's the weather in Berlin and what's the best programming language?")
                .await
                .map_err(|e| e.to_string())?;
            let violations = violations.lock().unwrap().clone();
            check(
                violations.is_empty(),
                format!("messages with {violations:?} tool calls"),
            )
        }
    );
}
