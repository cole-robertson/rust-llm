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
let total = judgment.cost().total(); // None: the catalog carries no pricing
# }
```

`JudgeOptions::provider_options` (or `Judge::provider_options`) is merged into the request;
`model`, `state`, and `questions` are reserved. Judgments use the shared timeouts, retries, and
error types.

## Not ported

- Block DSLs for questions and input (`choice ... do billing "..." end`, `judge do ... end`): use
  JSON.
- Symbol-vs-String option names: options are always strings.
- Method-style readers (`judgment.urgent`): use `get`/`probability`/`choice`/`score`.
- `metadata:` and the `judgment.ruby_llm` instrumentation event.
