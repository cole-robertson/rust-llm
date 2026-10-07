# Judgments

Ask typed questions about your data (probabilities, choices, scores) and get calibrated answers
from a System One model such as TypeSafe's Jev. A judgment is a single request: every question is
answered about the same input, nothing is generated, and no history is kept.

Configure `typesafe_api_key` (or `TYPESAFE_API_KEY`). A Jev-compatible local server works too: set
`typesafe_api_base` and use `assume_model_exists` for its model ids.

## Asking a Question

A `RubyLLM::Judge` subclass becomes a `Judge` built with chained declarations. Each declaration
returns `Result<Judge>` because it validates the question.

```ruby
class Urgency < RubyLLM::Judge
  probability :urgent, "Does this need attention today?"
end
judgment = Urgency.judge("Please refund the duplicate charge today.")
judgment.urgent.probability # => 0.96
```

```rust,no_run
use rust_llm::Judge;

# async fn run() -> rust_llm::Result<()> {
let urgency = Judge::new().probability("urgent", "Does this need attention today?")?;
let judgment = urgency.judge("Please refund the duplicate charge today.").await?;
let p = judgment.probability("urgent"); // Some(0.96)
# Ok(()) }
```

Build a judge once and reuse it for many inputs.

## Defining Questions

```ruby
class TicketTriage < RubyLLM::Judge
  probability :urgent, "Does this need attention today?" do
    yes "An explicit deadline today or an ongoing outage"
    no  "A general question with no time pressure"
  end
  choice :department, "Which team should handle this?" do
    billing   "Payments and refunds"
    technical "Bugs and integrations"
    other     "Everything else"
  end
  score :frustration, "How frustrated is the customer?", ["Calm", "Frustrated", "Angry"]
end
```

```rust,no_run
use rust_llm::Judge;
use serde_json::json;

# fn run() -> rust_llm::Result<()> {
let triage = Judge::new()
    .probability_with(
        "urgent",
        Some("Does this need attention today?".into()),
        "An explicit deadline today or an ongoing outage",
        "A general question with no time pressure",
    )?
    .choice(
        "department",
        Some("Which team should handle this?".into()),
        json!({ "billing": "Payments and refunds", "technical": "Bugs and integrations", "other": "Everything else" }),
    )?
    .score("frustration", Some("How frustrated is the customer?".into()), json!(["Calm", "Frustrated", "Angry"]))?;
# Ok(()) }
```

- `probability` is the probability of yes, between 0 and 1. Near 0.5 means yes and no are about
  equally likely, not "medium".
- `choice` takes a JSON object of option name to description (`null` for none); declared order is
  kept. Up to 255 options.
- `score` takes ordered levels (2 to 10). The answer is probability-weighted on the zero-based
  scale and can fall between levels.

Instructions, options, and levels can be strings, JSON objects, or arrays.

## Supplying Input

`judge` takes text, a JSON object, or a JSON array (one shared input, such as a list of messages):

```ruby
TicketTriage.judge do
  message "I was charged twice. Please refund the duplicate charge today."
  customer { plan "Pro"; previous_contacts 2 }
end
```

```rust,no_run
# async fn run(triage: rust_llm::Judge) -> rust_llm::Result<()> {
use serde_json::json;

let judgment = triage
    .judge(json!({
        "message": "I was charged twice. Please refund the duplicate charge today.",
        "customer": { "plan": "Pro", "previous_contacts": 2 }
    }))
    .await?;
# Ok(()) }
```

## Dynamic Values and Inputs

Declared `inputs` are required runtime values. `Dynamic::from_fn` computes an instruction, option
set, or model from them, once per judgment:

```ruby
class TeamRouter < RubyLLM::Judge
  inputs :teams
  choice :team, "Which team should handle this?", -> { teams.to_h { |t| [t.slug, t.description] } }
end
TeamRouter.judge(ticket.body, teams: Team.active.to_a)
```

```rust,no_run
use rust_llm::{Dynamic, Judge, JudgeOptions};
use serde_json::{Map, Value, json};

# async fn run(body: &str) -> rust_llm::Result<()> {
let router = Judge::new().inputs(["teams"]).choice(
    "team",
    Some("Which team should handle this?".into()),
    Dynamic::from_fn(|inputs| inputs["teams"].clone()),
)?;

let mut inputs = Map::new();
inputs.insert("teams".into(), json!({ "billing": "Charges and refunds", "platform": "Outages" }));
let judgment = router.judge_with(body, JudgeOptions { inputs, ..Default::default() }).await?;
# Ok(()) }
```

Inputs are not sent to the model unless you put them in the input or a question. Missing or unknown
inputs fail with `Error::Argument`.

## Reading Answers

```ruby
judgment.department.choice
judgment.department.probabilities
judgment.department.confidence
judgment.frustration.score
judgment[:missing]        # => nil
judgment.fetch(:missing)  # raises KeyError
```

```rust,no_run
use rust_llm::Answer;

# fn run(judgment: rust_llm::Judgment) -> rust_llm::Result<()> {
let department = judgment.choice("department");     // Option<&str>
let frustration = judgment.score("frustration");    // Option<f64>

if let Some(Answer::Choice { choice, probabilities, confidence }) = judgment.get("department") {
    println!("{choice} ({confidence:.2}) {probabilities:?}");
}
let missing = judgment.get("missing"); // None
let err = judgment.fetch("missing");   // Err(Error::Argument)

for (name, answer) in &judgment.answers {
    println!("{name}: {}", answer.to_value());
}
# Ok(()) }
```

`Answer` is `Probability { probability }`, `Choice { choice, probabilities, confidence }`, or
`Score { score, levels, probabilities, confidence }`. `to_value()` serializes an answer or a whole
judgment.

## Questions from Data

```ruby
RubyLLM.judge("Please help today.", questions: {
  urgent: { type: :probability, instructions: "Does this need attention today?" },
  department: { type: :choice, instructions: "Which team?", options: { billing: "Payments", other: "Everything else" } }
})
```

```rust,no_run
use serde_json::json;

# async fn run() -> rust_llm::Result<()> {
let judgment = rust_llm::judge(
    "Please help today.",
    json!({
        "urgent": { "type": "probability", "instructions": "Does this need attention today?" },
        "department": { "type": "choice", "instructions": "Which team?", "options": { "billing": "Payments", "other": "Everything else" } }
    }),
    Default::default(),
)
.await?;
# Ok(()) }
```

## Judging Many Inputs

A `Judge` is reusable and `Clone`, so judging a batch a few at a time is a stream of owned
futures. Give each future its own input and its own clone of the judge:

```rust,no_run
use futures::{StreamExt, stream};
use rust_llm::{Judge, Judgment, Result};

async fn judge_all(judge: &Judge, texts: Vec<String>) -> Vec<(String, Result<Judgment>)> {
    stream::iter(texts)
        .map(|text| {
            let judge = judge.clone();
            async move {
                let judgment = judge.judge(text.as_str()).await;
                (text, judgment)
            }
        })
        .buffer_unordered(8) // at most 8 requests in flight
        .collect()
        .await
}
# fn _futures_are_send(judge: &Judge, texts: Vec<String>) {
#     fn is_send<T: Send>(_: T) {}
#     is_send(judge_all(judge, texts));
# }
```

Where the surrounding future must be `Send` (a Loco worker's `perform`, an axum handler,
`tokio::spawn`), a closure that takes a reference, such as `stream::iter(&texts)` or
`texts.iter()`, fails with "implementation of `FnOnce` is not general enough"
([rust-lang/rust#102211](https://github.com/rust-lang/rust/issues/102211)). Iterate owned values
(`into_iter()`, `to_vec()`) so the closure's argument is not a reference. Cloning the judge as
above also makes each future `'static`, which `tokio::spawn` and `JoinSet` need; a clone copies the
question definitions, which is small next to the request.

Each judgment is retried on its own (see [Usage](#usage)), so size `max_retries` for the batch.

## Choosing a Model

`config.default_judgment_model` defaults to `jev-latest`. A judge overrides it with `.model(..)`
(plus `.provider(..)` and `.assume_model_exists()`), and a call overrides both with
`JudgeOptions { model: Some(Some(id)), .. }`; `Some(None)` is Ruby's `model: nil` (back to the
configured default). `rust_llm::list_judgment_models(None)` lists the provider's catalog.

## Usage

```rust,no_run
# fn run(judgment: rust_llm::Judgment) {
let model = &judgment.model;      // e.g. "jev-1.13.0"
let input = judgment.tokens().input;
let total = judgment.cost().total(); // Some(USD) for jev-latest and jev-preview
# }
```

The bundled registry prices `jev-latest` and `jev-preview` at TypeSafe's published rate, $0.042
per million input tokens with output free ([TypeSafe models](https://docs.typesafe.ai/models.md)).
A model id outside the registry (`assume_model_exists`) has no price, so its `cost().total()` is
`None`. So is a judgment where an attempt failed with unknown usage (a timeout or a 5xx), as for
chats (see [Cost and Usage](cost-and-usage.md)).

`JudgeOptions::provider_options` (or `Judge::provider_options`) is merged into the request;
`model`, `state`, and `questions` are reserved. Judgments use the shared timeouts, retries, and
error types.

Retries are per request: every judgment gets up to `max_retries` retries (default 3) on a 429, a
500, a 502-504, a 529, a timeout, or a connection failure, and none on other errors such as a 401
(see [Errors and Retries](errors-and-retries.md#automatic-retries)). During a TypeSafe outage a
batch of 300 judgments therefore sends up to 1,200 requests (300 plus 900 retries), 8 at a time
under `buffer_unordered(8)`, and each judgment spends 0.7 to 0.85 s in backoff, or longer when
TypeSafe sends `Retry-After`, before it fails. To fail a batch fast, judge it with a configuration
that retries less:

```rust,no_run
use rust_llm::JudgeOptions;

let ctx = rust_llm::context(|config| config.max_retries = 1);
let options = JudgeOptions { config: Some(ctx.config().clone()), ..Default::default() };
```

`JudgeOptions::metadata` is added to the `judgment.rust_llm` [instrumentation](instrumentation.md)
event and never sent to the provider.

## Differences from RubyLLM

- Block DSLs for questions and input (`choice ... do billing "..." end`, `judge do ... end`) become
  JSON.
- Option names are always strings; there are no Symbol-vs-String variants.
- Method-style readers (`judgment.urgent`) become `get`/`probability`/`choice`/`score`.
- RubyLLM's bundled `models.json` has no price for `jev-latest` or `jev-preview`, so its
  `judgment.cost.total` is `nil`. RustLLM's bundled copy adds TypeSafe's published price. Neither
  TypeSafe's model catalog nor models.dev carries it, so `rust_llm::models::refresh` replaces it
  with the published catalog's empty pricing.
