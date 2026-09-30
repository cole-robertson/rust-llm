# Prompt Templates

Keep long prompts in files, render them with locals, and share pieces between them as partials.
This follows RubyLLM's `docs/_core_features/prompt-rendering.md`.

## Rendering a Prompt

RubyLLM renders ERB files under `app/prompts`. ERB runs Ruby, so RustLLM renders Jinja templates
([minijinja](https://docs.rs/minijinja)) named `<name>.txt.jinja` instead:

```ruby
# app/prompts/support/instructions.txt.erb: You help customers of <%= product_name %>.
RubyLLM.render_prompt("support/instructions", product_name: "BillingHub")
```

```rust,no_run
# fn run() -> rust_llm::Result<()> {
// app/prompts/support/instructions.txt.jinja: You help customers of {{ product_name }}.
let text = rust_llm::render_prompt("support/instructions", serde_json::json!({ "product_name": "BillingHub" }))?;
# Ok(()) }
```

A missing file returns `Error::PromptNotFound` with the path it looked for. Using a variable the
template was not given is an error, as it is in ERB. Output is not HTML-escaped.

## ERB to Jinja

| ERB (`.txt.erb`) | Jinja (`.txt.jinja`) |
|---|---|
| `<%= name %>` | `{{ name }}` |
| `<% if admin %>...<% end %>` | `{% if admin %}...{% endif %}` |
| `<% items.each do \|i\| %>...<% end %>` | `{% for i in items %}...{% endfor %}` |
| `<%= render "tone", name: name %>` | `{{ render("tone", name=name) }}` |
| `<%= render partial: "shared/safety", locals: { name: name } %>` | `{{ render(partial="shared/safety", locals={"name": name}) }}` |
| `<%= local_assigns[:name] \|\| "friend" %>` | `{{ local_assigns.name \| default("friend") }}` |
| `<%= local_assigns["x-y"] %>` | `{{ local_assigns["x-y"] }}` |

A local whose name is not a valid Ruby local variable (`x-y`, `Name`) is only available through
`local_assigns`, as in RubyLLM.

## Partials

A partial is a file whose name starts with `_`. A bare name is looked up next to the prompt that
renders it; a name with a path is looked up from the prompt roots. Locals do not leak into a
partial: pass the ones it needs.

## Prompt Roots

The application root is `app/prompts` under the working directory; set
`config.set("prompt_root", ...)` to move it. Libraries add their own directories to
`config.prompt_roots`, which are searched after the application root, so an application overrides
a library's prompt by shipping a file at the same path:

```rust,no_run
rust_llm::configure(|config| {
    config.prompt_roots.push("vendor/my_engine/app/prompts".into());
});
```

## Agent Instructions

An agent that declares no `instructions` uses `app/prompts/<agent>/instructions.txt.jinja` when it
exists, rendered with the agent's `prompt_locals`. `WorkAssistant` reads
`app/prompts/work_assistant/`. An empty file means no instructions. See [Agents](agents.md).

## Differences from RubyLLM

- Templates are Jinja (`.txt.jinja`), not ERB, so they cannot call Ruby constants or methods.
