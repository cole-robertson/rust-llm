# Structured Output

Get a response that matches a JSON schema, and read it as JSON or a typed struct.

## From a Rust Type

RubyLLM describes the shape with a `Schematist::Schema` class. In Rust, derive
`schemars::JsonSchema` and call `with_schema_for::<T>()`. The schema is named after the type, and
every object gets `additionalProperties: false`, as Schematist does.

```ruby
class LanguagesSchema < Schematist::Schema
  array :languages do
    object do
      string :name
      integer :year
    end
  end
end

response = RubyLLM.chat.with_schema(LanguagesSchema).ask("List 3 programming languages with their year created.")
response.parsed["languages"]
```

```rust,no_run
use serde::Deserialize;

#[derive(schemars::JsonSchema, Deserialize)]
struct Language {
    name: String,
    year: i64,
}

#[derive(schemars::JsonSchema, Deserialize)]
struct Languages {
    languages: Vec<Language>,
}

# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?.with_schema_for::<Languages>();
let response = chat.ask("List 3 programming languages with their year created.").await?;

// As JSON, like `response.parsed`:
let json = response.parsed()?; // Option<serde_json::Value>, None for empty content

// Or straight into the type:
let languages: Languages = serde_json::from_str(response.content())?;
# Ok(()) }
```

Doc comments on fields become `description`s. `Option<T>` fields become optional properties. The
type's own doc comment is dropped, since it documents Rust code rather than the model's output.

## Manual JSON Schemas

`with_schema` takes a raw JSON schema, or `{ "name", "schema", "strict" }` to name it:

```ruby
chat.with_schema({ name: 'PersonSchema', schema: { type: 'object', properties: { name: { type: 'string' } },
                                                    required: ['name'], additionalProperties: false } })
```

```rust,no_run
use serde_json::json;

# fn run() -> rust_llm::Result<()> {
let chat = rust_llm::chat()?.with_schema(json!({
    "name": "PersonSchema",
    "schema": {
        "type": "object",
        "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
        "required": ["name", "age"],
        "additionalProperties": false
    }
}));
# Ok(()) }
```

Without a name the schema is called `response`. Include `additionalProperties: false` on each
object for OpenAI's strict mode.

## Changing Schemas Mid-Conversation

```ruby
chat.with_schema(PersonSchema)
person = chat.ask("Generate a person")
chat.with_schema(nil)
analysis = chat.ask("Tell me about this person's potential career paths")
```

```rust,no_run
# #[derive(schemars::JsonSchema)] struct Person { name: String }
# async fn run() -> rust_llm::Result<()> {
let mut chat = rust_llm::chat()?.with_schema_for::<Person>();
let person = chat.ask("Generate a person").await?;

let mut chat = chat.with_schema(serde_json::Value::Null); // with_schema(nil)
let analysis = chat.ask("Tell me about this person's potential career paths").await?;
# Ok(()) }
```

The `with_*` builders take `self`, so rebind the chat when you change a setting between turns.

## Provider Support

Check the registry before relying on it:

```rust,no_run
# fn run() -> rust_llm::Result<()> {
let supported = rust_llm::models().find("gpt-5.6", None)?.supports("structured_output");
# Ok(()) }
```

## JSON Mode

For valid JSON without a fixed shape, use the provider's own option:

```rust,no_run
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
// OpenAI Responses (the default protocol for OpenAI)
let mut chat = rust_llm::chat()?.with_provider_options(json!({ "text": { "format": { "type": "json_object" } } }));
let response = chat.ask("List three programming languages. Return JSON.").await?;
let json = response.parsed()?;
# Ok(()) }
```

## Not ported

- `Schematist::Schema` and the agent `schema do ... end` DSL: use `schemars` types or JSON.
