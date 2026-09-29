# MCP

Connect to [Model Context Protocol](https://modelcontextprotocol.io) servers, give their tools to
a chat, and read their resources and prompts.

## Describing a Server

RubyLLM describes a server in a `RubyLLM::MCP` subclass or inline with `RubyLLM.mcp`. In Rust the
class DSL is a builder: start from `Mcp::url`, `Mcp::command`, or `Mcp::transport`, chain settings,
and finish with `build()`. Nothing is contacted until the first request.

```ruby
class Linear < RubyLLM::MCP
  url "https://mcp.linear.app/mcp"
  bearer_token ENV.fetch("LINEAR_API_KEY")
end

class Files < RubyLLM::MCP
  command "npx", "-y", "@modelcontextprotocol/server-filesystem", "."
  directory Rails.root
  env NODE_ENV: "production"
end
```

```rust,no_run
use rust_llm::Mcp;

# fn run() -> rust_llm::Result<()> {
let linear = Mcp::url("https://mcp.linear.app/mcp")
    .bearer_token(std::env::var("LINEAR_API_KEY").unwrap_or_default())
    .build()?;

let files = Mcp::command(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])
    .directory("/srv/app")
    .env("NODE_ENV", "production")
    .build()?;
# Ok(()) }
```

`url` speaks Streamable HTTP; `command` starts a local process that speaks stdio on the first
request and stops on `close().await`. Plain HTTP is only allowed for loopback addresses, and a URL
with credentials in it is rejected.

Builder settings:

| RubyLLM | `McpBuilder` |
|---|---|
| `name:` | `.name("docs")` (defaults to the URL host or command name) |
| `bearer_token "..."` / `bearer_token { ... }` | `.bearer_token(..)` / `.bearer_token_with(\|\| Some(..))` |
| `header "X", "v"` / `header("X") { ... }` | `.header(..)` / `.header_with(..)` |
| `env`, `directory`, `timeout` | `.env(..)`, `.directory(..)`, `.timeout(Duration)` |
| `only`, `except`, `prefix` | `.only(&[..])`, `.except(&[..])`, `.prefix(..)` |
| `tool :x, as:, description:, fixed_arguments:, wrap:` | `.tool("x", ToolShape::new()...)` |
| `tool SearchAndRead` | `.add_tool(\|mcp\| Arc::new(SearchAndRead(mcp)))` |
| `requires_approval :a, :b` / `requires_approval if: :destructive?` | `.requires_approval(&[..])` / `.requires_approval_if(&[..], \|tool\| tool.is_destructive())` |
| `after_progress`, `before_input_request` | `.after_progress(..)`, `.before_input_request(..)` |
| `context:` | `.config(Arc<Config>)` |

Settings that Ruby evaluates per request from declared `inputs` are closures that capture what
they need:

```ruby
class Linear < RubyLLM::MCP
  url "https://mcp.linear.app/mcp"
  inputs :user
  bearer_token { user.linear_token }
end
```

```rust,no_run
use rust_llm::Mcp;

# fn run(user_token: String) -> rust_llm::Result<()> {
let linear = Mcp::url("https://mcp.linear.app/mcp")
    .bearer_token_with(move || Some(user_token.clone()))
    .build()?;
# Ok(()) }
```

A custom transport implements `rust_llm::mcp::Transport` (`request`, `notify`, `cancel`, `close`)
and is passed to `Mcp::transport(name, Arc::new(transport))`.

## Exploring a Server

```ruby
docs = RubyLLM.mcp(url: "https://learn.microsoft.com/api/mcp")
docs.tools
docs.call("microsoft_docs_search", query: "Azure Blob Storage").text
```

```rust,no_run
use rust_llm::Mcp;
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let docs = Mcp::url("https://learn.microsoft.com/api/mcp").build()?;

for tool in docs.mcp_tools().await? {
    println!("{} read_only={} {:?}", tool.name, tool.is_read_only(), tool.description);
}
let instructions = docs.instructions().await?;

let result = docs.call("microsoft_docs_search", json!({ "query": "Azure Blob Storage" })).await?;
println!("{}", result.text);          // text blocks
let structured = &result.structured;  // Option<Value>
let files = &result.attachments;      // images and files
let failed = result.is_error();
# Ok(()) }
```

Ruby also exposes each tool as a method (`docs.microsoft_docs_search(...)`); in Rust use `call`.
`McpTool` carries the server's hints: `is_read_only`, `is_destructive`, `is_idempotent`,
`is_open_world`.

## Shaping Tools

```ruby
class GitHub < RubyLLM::MCP
  url "https://api.githubcopilot.com/mcp/"
  only :search_issues, :get_issue
  prefix :github
  tool :search_issues, as: :search_ruby_llm_issues,
       description: "Search RubyLLM's issues.",
       fixed_arguments: { owner: "crmne", repo: "ruby_llm" }
  requires_approval if: :destructive?
end
```

```rust,no_run
use rust_llm::mcp::ToolShape;
use rust_llm::{Mcp, ToolResult};
use serde_json::json;

# fn run() -> rust_llm::Result<()> {
let github = Mcp::url("https://api.githubcopilot.com/mcp/")
    .only(&["search_issues", "get_issue"])
    .prefix("github")
    .tool(
        "search_issues",
        ToolShape::new()
            .as_name("search_rust_llm_issues")
            .description("Search RustLLM's issues.")
            .fixed_argument("owner", json!("cole-robertson"))
            .fixed_argument("repo", json!("rust-llm"))
            .wrap(|result, _args| ToolResult::from(result.text.clone())),
    )
    .requires_approval_if(&[] as &[&str], |tool| tool.is_destructive())
    .build()?;
# Ok(()) }
```

Fixed arguments are removed from the schema the model sees. A result the server marks as an error
reaches the model as an error without passing through `wrap`. A declared name the server does not
offer fails with `Error::Configuration` when the tools load.

## Using Servers in Chats

```ruby
chat = RubyLLM.chat.with_mcp(linear, files)
chat.ask "Which open issues mention the flaky login spec?"
chat.mcp[:files]
chat.with_mcp(nil)
```

```rust,no_run
# async fn run(linear: rust_llm::Mcp, files: rust_llm::Mcp) -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?.with_mcp(linear).with_mcp(files);
chat.ask("Which open issues mention the flaky login spec?").await?;

let files = chat.mcp().get("files"); // by name
chat.clear_mcp();                    // with_mcp(nil)
# Ok(()) }
```

A server is contacted the first time the chat needs its tools. Two tools with the same name fail
with `Error::Argument`. An agent declares servers with `fn mcp(&self) -> Vec<Mcp>`.

## Resources

```ruby
readme = files.resource("file:///project/README.md")
readme.content
chat.ask "Summarize this", with: readme
files.resource("file:///{+path}", path: "src/lib.rs")
```

```rust,no_run
use serde_json::json;

# async fn run(files: rust_llm::Mcp, mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
let all = files.resources().await?;
let readme = files.resource("file:///project/README.md").await?;
let text = readme.content().await?;         // ResourceContent::Text or ::Blob
readme.save("README.md").await?;
chat.ask_with("Summarize this", vec![readme.to_attachment().await?]).await?;

let templates = files.resource_templates().await?;
let lib = files.resource_from_template("file:///{+path}", json!({ "path": "src/lib.rs" })).await?;
# Ok(()) }
```

## Prompts

```ruby
chat.ask github.prompt(:code_review, code: diff, language: "Ruby")
github.prompts.first.suggest(language: "ru")
```

```rust,no_run
# async fn run(github: rust_llm::Mcp, mut chat: rust_llm::Chat, diff: &str) -> rust_llm::Result<()> {
let review = github.prompt("code_review", &[("code", diff), ("language", "Rust")]).await?;
chat.ask_prompt(&review).await?; // adds every message of the prompt, then completes

let first = github.prompts().await?.remove(0);
let suggestions = first.suggest(&[("language", "ru")]).await?; // ["ruby", "rust"]
# Ok(()) }
```

## Input Requests

A server can pause a call to ask the user something. Answer with `before_input_request`, or let the
chat pause and settle the request later:

```ruby
chat.ask "Deploy the release branch"
chat.awaiting_input? # => true
request = chat.pending_inputs.first
chat.answer(request, environment: "staging") # or chat.decline(request)
chat.complete
```

```rust,no_run
use serde_json::{Map, json};

# async fn run(mut chat: rust_llm::Chat) -> rust_llm::Result<()> {
chat.ask("Deploy the release branch").await?;
if chat.is_awaiting_input() {
    let request = chat.pending_inputs().remove(0);
    println!("{:?} {:?}", request.message, request.fields.first().map(|f| &f.choices));
    let mut values = Map::new();
    values.insert("environment".into(), json!("staging"));
    chat.answer(&request, values)?; // or chat.decline(&request)?
    chat.complete().await?;
}
# Ok(()) }
```

Calling a tool outside a chat when no callback answers fails with `Error::McpInputRequired`.

## Progress and Cancellation

```rust,no_run
use rust_llm::Mcp;

# fn run() -> rust_llm::Result<()> {
let deploys = Mcp::url("https://deploys.example.com/mcp")
    .after_progress(|progress| println!("{:?} {:?}", progress.message, progress.fraction()))
    .build()?;
let chat = rust_llm::chat()?
    .with_mcp(deploys)
    .after_tool_progress(|call, progress| println!("{}: {:?}", call.name, progress.message));
# Ok(()) }
```

Progress is only requested when a callback listens. `chat.cancel()` (or a `CancelHandle`) also
stops a server call the chat is waiting on.

## Errors

A server that answers with a protocol error fails with `Error::Mcp`, whose `McpError` has the
JSON-RPC `code` and `data`.

## Not ported

- OAuth (`oauth owner:`, `authorization_url`, `authorize`, credential stores,
  `mcp_client_id`), and the `rust-llm generate upgrade` migration for MCP credentials.
- Tool methods on the server object (`docs.microsoft_docs_search(...)`): use `call`.
- `with_mcp` on persisted chat records: build the `Chat` with `to_llm` and call `with_mcp` on it.
