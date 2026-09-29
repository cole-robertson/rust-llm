//! `RubyLLM.ocr`, replayed from RubyLLM's `ocr_*` cassettes. Assertions follow
//! `spec/ruby_llm/ocr_spec.rb`.

mod support;

use rust_llm::message::Operation;
use rust_llm::{Attachment, Error, OcrOptions, ocr};
use support::{Cassette, config_for};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

async fn start(name: &str) -> Cassette {
    Cassette::start(name).await.unwrap_or_else(|| panic!("missing cassette {name}; run bin/convert-cassettes 'ocr_*'"))
}

fn mistral<'a>(config: std::sync::Arc<rust_llm::Config>) -> OcrOptions<'a> {
    OcrOptions { model: Some("mistral-ocr-latest"), provider: Some("mistral"), config: Some(config), ..Default::default() }
}

/// "extracts markdown from a PDF": the local PDF goes as a data URI `document_url`.
#[tokio::test]
async fn extracts_markdown_from_a_pdf() {
    let cassette = start("ocr_basic_functionality_mistral_mistral-ocr-latest_extracts_markdown_from_a_pdf").await;
    let config = config_for(&cassette, "mistral");
    let result = ocr(Attachment::new(fixture("sample.pdf")), mistral(config)).await.unwrap();

    assert!(!result.pages.is_empty());
    assert_eq!(result.pages[0].index, 0);
    assert!(result.markdown().to_lowercase().contains("sample pdf"));
    assert_eq!(result.model, "mistral-ocr-latest");
    assert_eq!(result.usage_entries.len(), 1);
    assert_eq!(result.usage_entries[0].operation, Operation::Ocr);
    cassette.assert_all_matched().await;
}

/// "extracts markdown from a CSV file".
#[tokio::test]
async fn extracts_markdown_from_a_csv_file() {
    let cassette = start("ocr_basic_functionality_mistral_mistral-ocr-latest_extracts_markdown_from_a_csv_file").await;
    let config = config_for(&cassette, "mistral");
    let result = ocr(Attachment::new(fixture("sample.csv")), mistral(config)).await.unwrap();
    assert_eq!(result.pages.len(), 1);
    assert!(result.markdown().contains("12345"));
    cassette.assert_all_matched().await;
}

/// "uses the default OCR model when none is given".
#[tokio::test]
async fn uses_the_default_ocr_model_when_none_is_given() {
    let cassette = start("ocr_basic_functionality_uses_the_default_ocr_model_when_none_is_given").await;
    let config = config_for(&cassette, "mistral");
    assert_eq!(config.default_ocr_model, "mistral-ocr-latest");
    let result = ocr(Attachment::new(fixture("sample.pdf")), OcrOptions { config: Some(config), ..Default::default() }).await.unwrap();
    assert_eq!(result.model, "mistral-ocr-latest");
    assert!(result.markdown().to_lowercase().contains("sample pdf"));
    cassette.assert_all_matched().await;
}

/// "passes options through in provider vocabulary": `pages: [0]` and `usage_info`.
#[tokio::test]
async fn passes_options_through_in_provider_vocabulary() {
    let cassette = start("ocr_basic_functionality_mistral_mistral-ocr-latest_passes_options_through_in_provider_vocabulary").await;
    let config = config_for(&cassette, "mistral");
    let result = ocr(Attachment::new(fixture("sample.pdf")), OcrOptions { pages: Some(vec![0]), ..mistral(config) }).await.unwrap();
    assert_eq!(result.pages.len(), 1);
    assert_eq!(result.usage.as_ref().and_then(|u| u.get("pages_processed")).and_then(|p| p.as_i64()), Some(1));
    cassette.assert_all_matched().await;
}

/// "raises a clear error on providers without OCR support".
#[tokio::test]
async fn providers_without_ocr_fail_clearly() {
    let mut config = rust_llm::Config::default();
    config.set("openai_api_key", "test");
    let options = OcrOptions { model: Some("gpt-5-nano"), provider: Some("openai"), config: Some(config.into()), ..Default::default() };
    let err = ocr(Attachment::new(fixture("sample.pdf")), options).await.unwrap_err();
    assert!(err.to_string().contains("doesn't support OCR"), "{err}");
}

/// "validates model existence".
#[tokio::test]
async fn validates_model_existence() {
    let options = OcrOptions { model: Some("invalid-ocr-model"), ..Default::default() };
    let err = ocr(Attachment::new(fixture("sample.pdf")), options).await.unwrap_err();
    assert!(matches!(err, Error::ModelNotFound(_)), "{err}");
}
