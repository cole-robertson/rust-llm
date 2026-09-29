//! The RubyLLM judgments guide, in Rust, against TypeSafe's hosted Jev.
//!
//!   TYPESAFE_API_KEY=... cargo run -p rust_llm --example judge

use rust_llm::Judge;
use serde_json::json;

#[tokio::main]
async fn main() -> rust_llm::Result<()> {
    // class TicketTriage < RubyLLM::Judge
    //   probability :urgent, "Does this need attention today?"
    //   choice :department, "Which team should handle this?" do billing ...; technical ...; other ... end
    //   score :frustration, "How frustrated is the customer?", ["Calm", "Frustrated", "Angry"]
    // end
    let triage = Judge::new()
        .probability("urgent", "Does this need attention today?")?
        .choice(
            "department",
            Some("Which team should handle this?".into()),
            json!({ "billing": "Payments and refunds", "technical": "Bugs and integrations", "other": "Everything else" }),
        )?
        .score("frustration", Some("How frustrated is the customer?".into()), json!(["Calm", "Frustrated", "Angry"]))?;

    // TicketTriage.judge { message "..."; customer { plan "Pro"; previous_contacts 2 } }
    let judgment = triage
        .judge(json!({
            "message": "I was charged twice. Please refund the duplicate charge today.",
            "customer": { "plan": "Pro", "previous_contacts": 2 }
        }))
        .await?;

    println!("model        -> {}", judgment.model);
    println!("urgent       -> {:.2}", judgment.probability("urgent").unwrap_or_default());
    let department = judgment.get("department").expect("declared");
    println!("department   -> {} (confidence {:.2})", department.choice().unwrap_or_default(), department.confidence().unwrap_or_default());
    println!("frustration  -> {:.2} of 0..2", judgment.score("frustration").unwrap_or_default());
    println!("tokens       -> {} in / {} out", judgment.tokens().input.unwrap_or(0), judgment.tokens().output.unwrap_or(0));
    Ok(())
}
