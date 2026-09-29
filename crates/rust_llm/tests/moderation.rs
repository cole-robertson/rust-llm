//! `RubyLLM.moderate`, replayed from RubyLLM's `moderation_*` cassettes. Assertions follow
//! `spec/ruby_llm/moderation_spec.rb`.

mod support;

use rust_llm::message::Operation;
use rust_llm::{Attachment, Error, ModerateOptions, ModerationInput, UsageStatus, moderate};
use support::{Cassette, config_for};

const INPUT: &str = "This is a safe message";
const LENNA: &str = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png";

async fn start(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| panic!("missing cassette {name}; run bin/convert-cassettes 'moderation_*'"))
}

fn named(name: &str) -> String {
    format!("moderation_moderate_with_openai_provider_{name}")
}

/// "moderates content and returns a Moderation instance".
#[tokio::test]
async fn moderates_content_and_returns_a_moderation_instance() {
    let cassette = start(&named("moderates_content_and_returns_a_moderation_instance")).await;
    let config = config_for(&cassette, "openai");
    let result = moderate(INPUT, ModerateOptions { config: Some(config), ..Default::default() }).await.unwrap();

    assert!(result.id.as_deref().is_some_and(|id| !id.is_empty()));
    assert_eq!(result.model, "omni-moderation-latest");
    assert_eq!(result.results.len(), 1);
    let first = &result.results[0];
    assert!(!first.is_flagged());
    assert!(first.categories.is_empty());
    assert!(!first.category_scores.is_empty());
    assert!(first.category_scores.values().all(|v| v.is_number()));
    // The call is billed once, as a moderation operation.
    assert_eq!(result.usage_entries.len(), 1);
    assert_eq!(result.usage_entries[0].operation, Operation::Moderation);
    assert_eq!(result.usage_entries[0].status, UsageStatus::Succeeded);
    cassette.assert_all_matched().await;
}

/// "provides convenience methods for checking results".
#[tokio::test]
async fn provides_convenience_methods_for_checking_results() {
    let cassette = start(&named("provides_convenience_methods_for_checking_results")).await;
    let config = config_for(&cassette, "openai");
    let result = moderate(INPUT, ModerateOptions { config: Some(config), ..Default::default() }).await.unwrap();

    assert!(!result.is_flagged());
    assert!(result.flagged_categories().is_empty());
    let scores = result.category_scores();
    assert!(scores.contains_key("violence"));
    assert_eq!(scores, result.results[0].category_scores);
    cassette.assert_all_matched().await;
}

/// "can be called directly on the Moderation class".
#[tokio::test]
async fn can_be_called_directly_on_the_moderation_class() {
    let cassette = start(&named("can_be_called_directly_on_the_moderation_class")).await;
    let config = config_for(&cassette, "openai");
    let result = rust_llm::moderation::moderate(INPUT, ModerateOptions { config: Some(config), ..Default::default() }).await.unwrap();
    assert!(!result.results.is_empty());
    cassette.assert_all_matched().await;
}

/// "supports explicit model specification": `provider: 'openai', assume_model_exists: true`.
#[tokio::test]
async fn supports_explicit_model_specification() {
    let cassette = start(&named("supports_explicit_model_specification")).await;
    let config = config_for(&cassette, "openai");
    let options = ModerateOptions { provider: Some("openai"), assume_model_exists: true, config: Some(config), ..Default::default() };
    let result = moderate(INPUT, options).await.unwrap();
    assert_eq!(result.model, "omni-moderation-latest");
    cassette.assert_all_matched().await;
}

/// "moderates text with an image attachment": the input becomes text + image_url parts.
#[tokio::test]
async fn moderates_text_with_an_image_attachment() {
    let cassette = start(&named("moderates_text_with_an_image_attachment")).await;
    let config = config_for(&cassette, "openai");
    let options =
        ModerateOptions { with: vec![Attachment::new(LENNA)], provider: Some("openai"), config: Some(config), ..Default::default() };
    let result = moderate("check this image and caption", options).await.unwrap();
    assert!(!result.results.is_empty());
    assert!(!result.is_flagged());
    assert!(!result.category_scores().is_empty());
    cassette.assert_all_matched().await;
}

/// "moderates an image attachment without text".
#[tokio::test]
async fn moderates_an_image_attachment_without_text() {
    let cassette = start(&named("moderates_an_image_attachment_without_text")).await;
    let config = config_for(&cassette, "openai");
    let options =
        ModerateOptions { with: vec![Attachment::new(LENNA)], provider: Some("openai"), config: Some(config), ..Default::default() };
    let result = moderate(ModerationInput::None, options).await.unwrap();
    assert!(!result.results.is_empty());
    assert!(!result.category_scores().is_empty());
    cassette.assert_all_matched().await;
}

/// "raises ArgumentError when neither text nor image is provided".
#[tokio::test]
async fn needs_text_or_an_image() {
    let err = moderate(ModerationInput::None, ModerateOptions::default()).await.unwrap_err();
    assert!(matches!(&err, Error::Argument(m) if m == "must provide input text, image attachment, or both"), "{err}");
}

#[tokio::test]
async fn anthropic_does_not_moderate() {
    let mut config = rust_llm::Config::default();
    config.set("anthropic_api_key", "test");
    let options = ModerateOptions { model: Some("claude-haiku-4-5"), config: Some(config.into()), ..Default::default() };
    let err = moderate(INPUT, options).await.unwrap_err();
    assert!(err.to_string().contains("Anthropic doesn't support moderation"), "{err}");
}
