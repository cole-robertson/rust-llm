# Migrating from RubyLLM

RustLLM keeps RubyLLM 2.0's names, behavior, and wire format. Its tests replay RubyLLM's own
recorded cassettes and require identical request bodies. So this page is mostly spelling: how a
Ruby idiom turns into Rust.

## Conventions

- **Predicates** ending in `?` become `is_*`: `complete?` is `is_complete()`, `awaiting_approval?`
  is `is_awaiting_approval()`, `max_tokens?` is `is_max_tokens()`.
- **Keyword arguments** become an options struct with `Default`
  (`EmbedOptions { model: Some("..."), ..Default::default() }`) or a separate method
  (`ask(msg, with:)` is `ask_with(msg, attachments)`).
- **Blocks** become closures (`before_tool_call(|call| ...)`); a block passed to `ask` for
  streaming becomes `ask_stream`.
- **`nil` to clear** becomes a `clear_*` method (`clear_tools`, `clear_mcp`,
  `clear_provider_tools`), `set_instructions(None, ..)`, or `with_schema(Value::Null)`.
- **Symbols** become enums (`ToolChoice::Required`, `FinishReason::Stop`, `ProtocolName::Responses`)
  or strings where RubyLLM accepts arbitrary values (`ThinkingConfig::effort("high")`).
- **Class DSLs** (`Tool`, `Agent`) become traits whose methods have defaults; `MCP` and `Judge`
  become builders.
- **Exceptions** become `Result<_, rust_llm::Error>`, one variant per Ruby error class.
- **Chat builders take `self`**: `with_*` returns the chat, so chain them or rebind
  (`let chat = chat.with_temperature(0.2);`). Methods that run the conversation take `&mut self`.
- **`context:`** becomes an `Arc<Config>` (`Chat::with_config`, `config` fields on options structs).
- **Everything that talks to a provider is `async`.**

## Cheat Sheet

### Setup and models

| RubyLLM | RustLLM |
|---|---|
| `RubyLLM.configure { \|c\| c.openai_api_key = k }` | `rust_llm::configure(\|c\| { c.openai_api_key(k); })` |
| `c.mistral_api_key = k` (any option) | `c.set("mistral_api_key", k)` |
| `c.default_model = "..."` | `c.default_model = "...".into()` |
| `RubyLLM.config` | `rust_llm::config()` |
| `RubyLLM.context { \|c\| ... }` | `let mut c = (*rust_llm::config()).clone(); ...; Arc::new(c)` |
| `RubyLLM.models.find(id, provider:)` | `rust_llm::models().find(id, Some(provider))` |
| `RubyLLM.models.chat_models` / `embedding_models` / `by_provider(:x)` | `models().chat_models()` / `embedding_models()` / `by_provider("x")` |
| `model.supports?(:vision)` | `model.supports("vision")` |
| `model.cost_for(tokens)` | `model.cost_for(&tokens)` |

### Chat

| RubyLLM | RustLLM |
|---|---|
| `RubyLLM.chat` | `rust_llm::chat()?` |
| `RubyLLM.chat(model: id)` | `rust_llm::chat_with(id)?` |
| `RubyLLM.chat(model:, provider:)` | `Chat::new(Some(id), Some(provider))?` |
| `RubyLLM.chat(model:, provider:, assume_model_exists: true)` | `Chat::with_config(config, Some(id), Some(p), true)?` |
| `chat.ask(msg)` / `chat.say(msg)` | `chat.ask(msg).await?` / `chat.say(msg).await?` |
| `chat.ask(msg, with: files)` | `chat.ask_with(msg, vec![...]).await?` |
| `chat.ask(msg) { \|chunk\| ... }` | `chat.ask_stream(msg, \|chunk\| ...).await?` |
| `chat.ask(mcp_prompt)` | `chat.ask_prompt(&prompt).await?` |
| `chat.ask_later(msg)` | `chat.ask_later(msg)?` (`ask_later_with`, `ask_later_prompt`) |
| `chat.complete` / `complete { }` | `chat.complete().await?` / `complete_stream(..)` |
| `chat.step` / `step { }` | `chat.step().await?` / `step_stream(..)` |
| `chat.generate` | `chat.generate().await?` |
| `chat.run_tools` | `chat.run_tools().await?` |
| `chat.complete?` | `chat.is_complete()` |
| `chat.messages` | `chat.messages()` |
| `chat.messages = list` | `chat.set_messages(list)` |
| `chat.add_message(msg)` | `chat.add_message(Message::user(..))` |
| `chat.add_completion(response)` | `chat.add_completion(message, record_usage)` |
| `chat.render` | `chat.render()?` |
| `chat.cache_until_here` | `chat.cache_until_here()?` |
| `chat.with_instructions(text)` | `.with_instructions(text)` |
| `chat.with_instructions(text, append: true, cache_until_here: true)` | `chat.set_instructions(Some(text), true, true)` |
| `chat.with_instructions(nil)` | `chat.set_instructions(None, false, false)` |
| `chat.with_model(id, provider:)` | `.with_model(id, Some(provider))?` |
| `chat.with_model(id, provider:, assume_model_exists: true)` | `.with_assumed_model(id, provider)?` |
| `chat.with_model(id, protocol: :chat_completions)` | `.with_protocol(ProtocolName::ChatCompletions)` |
| `chat.with_temperature(t)` | `.with_temperature(t)` |
| `chat.with_max_output_tokens(n)` | `.with_max_output_tokens(n)` |
| `chat.with_provider_options(h)` / `with_params(h)` | `.with_provider_options(json!(..))` / `.with_params(..)` |
| `chat.with_headers(h)` | `.with_headers([(k, v)])` |
| `chat.with_thinking` / `(false)` / `(effort:, budget:, display:)` | `.with_thinking(ThinkingConfig::on() / off() / effort(..) / budget(..))` |
| `chat.with_schema(SchemaClass)` | `.with_schema_for::<T>()` (a `schemars::JsonSchema` type) |
| `chat.with_schema(hash)` / `with_schema(nil)` | `.with_schema(json!(..))` / `.with_schema(Value::Null)` |
| `chat.with_fallbacks(a, b)` | `.with_fallbacks(["a".into(), "b".into()])` |
| `chat.with_fallbacks(a, on: [..])` | `.with_fallbacks(..).with_fallback_errors(vec![ErrorKind::..])` |
| `chat.tokens` / `chat.cost` | `chat.tokens()` / `chat.cost()` |
| `chat.cancel` / `cancelled?` | `chat.cancel()` / `is_cancelled()`; `cancel_handle()` for other tasks |

### Callbacks

| RubyLLM | RustLLM |
|---|---|
| `before_message { }` | `.before_message(\|\| ..)` |
| `after_message { \|message\| }` | `.after_message(\|message\| ..)` |
| `before_tool_call { \|call\| }` | `.before_tool_call(\|call\| ..)` |
| `after_tool_result { \|result\| }` | `.after_tool_result(\|result\| ..)` |
| `after_tool_progress { \|call, progress\| }` | `.after_tool_progress(\|call, progress\| ..)` |
| `before_fallback` / `after_fallback { \|fallback\| }` | `.before_fallback(\|f\| ..)` / `.after_fallback(\|f\| ..)` |
| `before_request { \|payload\| }` | `.before_request(\|payload\| ..)` (`&mut Value`) |

### Messages

| RubyLLM | RustLLM |
|---|---|
| `message.content` | `message.content()` (`&str`) or `message.content` (`Option<String>`) |
| `message.role` | `message.role` (`Role::User`, ...) |
| `message.parsed` | `message.parsed()?` (`Option<Value>`) |
| `message.tool_calls` | `message.tool_calls` (insertion-ordered by id) |
| `message.tool_call?` / `tool_result?` | `is_tool_call()` / `is_tool_result()` |
| `message.thinking&.text` | `message.thinking.as_ref().and_then(\|t\| t.text.as_deref())` |
| `message.citations` | `message.citations` |
| `message.finish_reason` | `message.finish_reason` (`Option<FinishReason>`) |
| `stopped?`, `max_tokens?`, `tool_call_stop?`, `content_filtered?` | `is_stopped()`, `is_max_tokens()`, `is_tool_call_stop()`, `is_content_filtered()` |
| `message.raw` | `message.raw` (`Option<RawResponse>`: status, headers, body) |
| `message.tokens` / `message.cost` | `message.tokens()` / `message.cost(None)` |
| `message.model_info` | `message.model_info()` |
| `message.to_h` | `message.to_h()` |
| `tool_call.arguments` | `tool_call.arguments()` |

### Tools

| RubyLLM | RustLLM |
|---|---|
| `class Weather < RubyLLM::Tool` | `struct Weather; #[async_trait] impl Tool for Weather` |
| `description "..."` | `fn description(&self) -> String` |
| `parameter :lat, type: :number, description:, required: false` | `Parameter::new("lat").kind("number").description(..).optional()` in `fn parameters` |
| `parameters do ... end` / `parameters(schema_hash)` | `fn parameters_schema(&self) -> Option<Value>` with `rust_llm::schema_for::<T>()` or `json!` |
| `def execute(lat:, lng:)` | `async fn execute(&self, args: Map<String, Value>, call: &ToolCall) -> Result<ToolResult, ToolError>` |
| `def execute(..., tool_call:)` | the `call` argument |
| return a String / Hash | `Ok("...".into())` / `Ok(json!(..).into())` |
| return `{ error: "..." }` | `Ok(ToolResult::error("..."))` |
| return `content, [attachment]` | `Ok(ToolResult::with_attachments(content, vec![..]))` |
| raise | `Err(e)` (surfaces as `Error::Tool`) |
| `self.tool_name` | `fn name(&self) -> String` |
| `requires_approval` | `fn requires_approval(&self) -> bool { true }` |
| `provider_options cache_control: ..` | `fn provider_options(&self) -> Map<String, Value>` |
| `progress "msg", value:, total:` | `rust_llm::progress::report(Progress { .. })` |
| `chat.with_tools(Weather, Calc)` | `.with_tool(Weather).with_tool(Calc)` or `.with_tools(vec![Arc::new(..)])` |
| `chat.with_tools(nil)` | `chat.clear_tools()` |
| `with_tool_options(choice: :required, calls: :one)` | `.with_tool_choice(ToolChoice::Required)?.with_tool_calls(ToolCalls::One)` |
| `chat.approve(call)` / `deny(call)` | `chat.approve(&call.id)` / `chat.deny(&call.id)` |
| `chat.awaiting_approval?` / `pending_approvals` | `is_awaiting_approval()` / `pending_approvals()` |
| `chat.with_provider_tools(:web_search, mcp: {..})` | `.with_provider_tools([ProviderTool::alias("web_search"), ProviderTool::with_options("mcp", json!(..))])` |

### Agents

| RubyLLM | RustLLM |
|---|---|
| `class X < RubyLLM::Agent` | `struct X; impl Agent for X` |
| `model "id", provider: :p` | `fn model` / `fn provider` |
| `instructions`, `tools`, `temperature`, `max_output_tokens`, `thinking`, `schema`, `provider_options`, `fallbacks`, `mcp`, `provider_tools` | the method of the same name |
| `tool_options choice:` | `fn tool_choice` |
| `model ..., protocol:` | `fn protocol` |
| `inputs :user` | fields on the struct |
| `X.new.ask(..)` / `X.chat` | `X.chat()?.ask(..).await?` |
| `X.new(chat: existing)` / `X.find(id)` | `X.apply(chat)?`, e.g. on `record.to_llm(db).await?` |

### MCP

| RubyLLM | RustLLM |
|---|---|
| `class Files < RubyLLM::MCP; command ...; end` / `RubyLLM.mcp(command: [..])` | `Mcp::command([..])...build()?` |
| `url "..."` / `RubyLLM.mcp(url:)` | `Mcp::url("...")...build()?` |
| `transport { }` | `Mcp::transport(name, Arc::new(t))` |
| `bearer_token`, `header`, `env`, `directory`, `timeout`, `only`, `except`, `prefix` | builder methods of the same name (`*_with` for closures) |
| `tool :x, as:, description:, fixed_arguments:, wrap:` | `.tool("x", ToolShape::new().as_name(..).description(..).fixed_argument(..).wrap(..))` |
| `requires_approval :a` / `if: :destructive?` | `.requires_approval(&["a"])` / `.requires_approval_if(&[..], \|t\| t.is_destructive())` |
| `after_progress`, `before_input_request` | same names |
| `mcp.tools` | `mcp.tools().await?` / `mcp.mcp_tools().await?` |
| `mcp.call(name, **args)` / `mcp.some_tool(..)` | `mcp.call(name, json!(..)).await?` |
| `mcp.resources` / `resource(uri)` / `resource(tpl, **vars)` | `resources()` / `resource(uri)` / `resource_from_template(tpl, json!(..))` |
| `mcp.prompts` / `prompt(:name, **args)` | `prompts()` / `prompt("name", &[(k, v)])` |
| `mcp.instructions`, `mcp.close` | `instructions().await?`, `close().await` |
| `chat.with_mcp(a, b)` / `with_mcp(nil)` | `.with_mcp(a).with_mcp(b)` / `chat.clear_mcp()` |
| `chat.mcp[:files]` | `chat.mcp().get("files")` |
| `chat.awaiting_input?`, `pending_inputs`, `answer(req, **values)`, `decline(req)` | `is_awaiting_input()`, `pending_inputs()`, `answer(&req, map)?`, `decline(&req)?` |

### One-shot operations

| RubyLLM | RustLLM |
|---|---|
| `RubyLLM.embed(text, model:, provider:, dimensions:)` | `rust_llm::embed(text, EmbedOptions { .. }).await?` |
| `embedding.vectors` | `embedding.vectors` (`Vectors::Single` / `Vectors::Batch`) |
| `RubyLLM.paint(prompt, model:, size:, count:, with:, mask:)` | `rust_llm::paint(prompt, PaintOptions { .. }).await?` (`Images`) |
| `image.save(path)` / `to_blob` / `base64?` | `image.save(path).await?` / `to_blob().await?` / `is_base64()` |
| `RubyLLM.upload(file, provider:, purpose:)` | `rust_llm::upload(file, UploadOptions { .. }).await?` |
| `RubyLLM::UploadedFile.find(id, provider:)` | `UploadedFile::find(id, FileOptions { .. }).await?` |
| `RubyLLM.download(id, provider:)` | `rust_llm::download(id, FileOptions { .. }).await?` |
| `RubyLLM::Attachment.new(path, resolution:)` | `Attachment::new(path).with_resolution(Resolution::High)` |
| `RubyLLM.batch(chats)` | `rust_llm::batch(chats).await?` |
| `RubyLLM::Batch.find(id, provider:)` | `Batch::find(id, Some(provider)).await?` |
| `batch.refresh.complete?` | `batch.refresh().await?.is_complete()` |
| `batch.messages` / `results` / `statuses` / `cancel` | `messages().await?` / `results().await?` / `statuses()` / `cancel().await?` |
| `RubyLLM.embed_later(text, ..)` | `rust_llm::embed_later(text, EmbedOptions { .. })?` |
| `class X < RubyLLM::Judge; probability :a, "..."; end` | `Judge::new().probability("a", "...")?` |
| `choice :d, "..." do ... end` / `score :s, "...", [..]` | `.choice("d", Some(..), json!({..}))?` / `.score("s", Some(..), json!([..]))?` |
| `inputs :teams` + `-> { teams }` | `.inputs(["teams"])` + `Dynamic::from_fn(\|inputs\| ..)` |
| `X.judge(input, **inputs)` | `judge.judge(input).await?` / `judge.judge_with(input, JudgeOptions { .. }).await?` |
| `RubyLLM.judge(input, questions: {..})` | `rust_llm::judge(input, json!({..}), Default::default()).await?` |
| `judgment.urgent.probability` / `judgment[:x]` / `fetch(:x)` | `judgment.probability("urgent")` / `get("x")` / `fetch("x")?` |

### Errors

| RubyLLM | RustLLM |
|---|---|
| `rescue RubyLLM::RateLimitError` | `Err(Error::RateLimit(..))` |
| `rescue RubyLLM::Error => e; e.response.status` | `e.response().map(\|r\| r.status)` |
| `rescue RubyLLM::CancelledError` | `Err(Error::Cancelled)` |

See [Errors and Retries](errors-and-retries.md) for every variant.

### Rails / Loco

| RubyLLM (Rails) | RustLLM (Loco) |
|---|---|
| `acts_as_chat`, `acts_as_message`, `acts_as_tool_call` | `rust_llm_loco::ChatRecord` and its entities |
| `Chat.create!(model:)` / `Chat.find(id)` | `ChatRecord::create(db, model, None).await?` / `ChatRecord::find(db, id).await?` |
| `chat_record.ask(msg)` | `let mut chat = record.to_llm(db).await?; record.ask(db, &mut chat, msg).await?` |
| `chat_record.complete` | `record.complete(db, &mut chat).await?` |
| `chat_record.approve(id)` / `deny(id)` | `record.approve(db, &mut chat, id).await?` / `deny(..)` |
| `chat_record.cost.total` | `record.total_cost(db).await?` |
| `rails g ruby_llm:install` (and `tool`, `agent`, `schema`, `chat_ui`, `upgrade`) | `rust-llm generate install` (same names) |

## A Full Example

```ruby
class Weather < RubyLLM::Tool
  description "Get current weather"
  parameter :latitude
  parameter :longitude
  def execute(latitude:, longitude:) = { temperature: 14.2 }
end

chat = RubyLLM.chat(model: "claude-haiku-4-5").with_tools(Weather)
chat.before_tool_call { |call| puts "-> #{call.name}" }
response = chat.ask "What's the weather in Berlin (52.52, 13.405)?"
puts response.content, chat.cost.total
```

```rust,no_run
use rust_llm::{Parameter, Tool, ToolCall, ToolError, ToolResult};
use serde_json::{Map, Value, json};

struct Weather;

#[async_trait::async_trait]
impl Tool for Weather {
    fn description(&self) -> String {
        "Get current weather".into()
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("latitude"), Parameter::new("longitude")]
    }
    async fn execute(&self, _args: Map<String, Value>, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        Ok(json!({ "temperature": 14.2 }).into())
    }
}

#[tokio::main]
async fn main() -> rust_llm::Result<()> {
    let mut chat = rust_llm::chat_with("claude-haiku-4-5")?
        .with_tool(Weather)
        .before_tool_call(|call| println!("-> {}", call.name));
    let response = chat.ask("What's the weather in Berlin (52.52, 13.405)?").await?;
    println!("{}\n{:?}", response.content(), chat.cost().total());
    Ok(())
}
```

## What Is Not Ported

- Providers: Bedrock, Vertex AI, Azure, Cohere, ElevenLabs, Deepgram.
- Operations: `animate`, `speak`, `transcribe`, `ocr`, `rerank`, `moderate`, `research`,
  `count_tokens`, `tokenize`, `RubyLLM.cache`, `RubyLLM.render_prompt`, `RubyLLM.workflow`.
- Chat options: `with_caching`, `with_citations`, `with_compaction` / `compact`, `with_end_user`,
  `with_context`, tool concurrency.
- MCP OAuth; Gemini embedding batches; multipart image edits for non-gpt-image models.
- Instrumentation events, `RUBYLLM_DEBUG` logging, `RubyLLM.models.refresh`.
- Rails-only pieces: Active Storage, persisted batches, persisted cancellation, Turbo streaming,
  agent `chat_model` mode, `rescue_from`.
