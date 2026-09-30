//! Ports of the render/parse examples in `spec/ruby_llm/protocols/{gemini,chat_completions}/
//! transcription_spec.rb` and `protocols/gemini/file_transcription_spec.rb`, against the private
//! seams those specs call with `send`.

use serde_json::{Value, json};

use super::*;

fn request<'a>(model: &'a str, provider_options: &'a Value) -> Request<'a> {
    Request {
        model,
        language: None,
        prompt: None,
        temperature: None,
        format: None,
        speaker_names: None,
        speaker_references: None,
        provider_options,
    }
}

fn ruby_wav() -> Attachment {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ruby.wav");
    Attachment::from_bytes(std::fs::read(path).unwrap(), "ruby.wav", None)
}

// ---- protocols/gemini/transcription_spec.rb ------------------------------------------------

// spec: protocols/gemini/transcription_spec.rb:29 #render_transcription_payload > passes the requested format and temperature through
#[test]
fn gemini_passes_the_requested_format_and_temperature_through() {
    let options = json!({});
    let r = Request {
        format: Some("application/json"),
        temperature: Some(0.2),
        ..request("gemini-2.5-flash", &options)
    };
    let payload = gemini_payload(&ruby_wav(), &r).unwrap();
    assert_eq!(
        payload["generationConfig"],
        json!({ "responseMimeType": "application/json", "temperature": 0.2 })
    );
}

// spec: protocols/gemini/transcription_spec.rb:37 #render_transcription_payload > refuses an attachment that is not audio
#[test]
fn gemini_refuses_an_attachment_that_is_not_audio() {
    let notes = Attachment::from_bytes(b"not audio".to_vec(), "notes.txt", None);
    let options = json!({});
    let error = gemini_payload(&notes, &request("gemini-2.5-flash", &options)).unwrap_err();
    assert!(
        matches!(error, Error::UnsupportedAttachment(_)),
        "{error:?}"
    );
}

// spec: protocols/gemini/transcription_spec.rb:65 #parse_transcription_response > leaves the text nil when the response carries no candidate
#[test]
fn gemini_leaves_the_text_nil_when_the_response_carries_no_candidate() {
    assert_eq!(parse_gemini(&json!({}), "gemini-2.5-flash").text, None);
    assert_eq!(
        parse_gemini(&json!("not a hash"), "gemini-2.5-flash").text,
        None
    );
}

// spec: protocols/gemini/transcription_spec.rb:70 #parse_transcription_response > leaves the text nil when the candidate carries no text parts
#[test]
fn gemini_leaves_the_text_nil_when_the_candidate_carries_no_text_parts() {
    let inline = json!({ "candidates": [{ "content": { "parts": [{ "inlineData": {} }] } }] });
    assert_eq!(parse_gemini(&inline, "gemini-2.5-flash").text, None);
    let empty = json!({ "candidates": [{ "content": {} }] });
    assert_eq!(parse_gemini(&empty, "gemini-2.5-flash").text, None);
}

// spec: protocols/gemini/transcription_spec.rb:75 #parse_transcription_response > leaves the token counts nil when the response carries no usage
#[test]
fn gemini_leaves_the_token_counts_nil_when_the_response_carries_no_usage() {
    let data = json!({ "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }] });
    let t = parse_gemini(&data, "gemini-2.5-flash");
    assert_eq!(t.tokens().input, None);
    assert_eq!(t.tokens().output, None);
}

// ---- protocols/gemini/file_transcription_spec.rb -------------------------------------------

// spec: protocols/gemini/file_transcription_spec.rb:63 preserves Interactions speaker-only annotations without fabricating timing
#[test]
fn interactions_preserve_speaker_only_annotations_without_fabricating_timing() {
    let data = json!({ "status": "completed", "steps": [{ "type": "model_output", "content": [
        { "type": "text", "text": "Hello", "annotations": [
            { "type": "word_info", "text": "Hello", "speaker": "spk:0" }
        ] }
    ] }], "usage": { "total_input_tokens": 14, "total_output_tokens": 0 } });

    let t = parse_interactions(&data, "gemini-3.5-transcribe").unwrap();

    assert_eq!(t.text.as_deref(), Some("Hello"));
    assert_eq!(
        t.words,
        Some(vec![json!({ "word": "Hello", "speaker": "spk:0" })])
    );
    assert_eq!((t.tokens().input, t.tokens().output), (Some(14), Some(0)));
    assert_eq!(t.duration, None);
}

// spec: protocols/gemini/file_transcription_spec.rb:102 rejects unknown granularities and unsupported reference clips instead of ignoring them
// (The reference-clip half transcribes through Vertex AI, a provider this port leaves out.)
#[test]
fn interactions_reject_unknown_granularities() {
    let error = Family::Interactions
        .render_options(Some(&["segment"]), None, false)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Argument(m) if m.contains("must be word")),
        "{error:?}"
    );
}

// ---- protocols/chat_completions/transcription_spec.rb --------------------------------------

// spec: protocols/chat_completions/transcription_spec.rb:19 .render_transcription_payload > keeps the format the caller asked for on a diarize model
#[test]
fn chat_completions_keeps_the_format_the_caller_asked_for_on_a_diarize_model() {
    let options = json!({});
    let r = Request {
        format: Some("json"),
        ..request("gpt-4o-transcribe-diarize", &options)
    };
    let payload = multipart_payload(Family::OpenAI, &r).unwrap();
    assert_eq!(payload["response_format"], json!("json"));
}

// spec: protocols/chat_completions/transcription_spec.rb:33 .render_transcription_payload > maps format to response_format
#[test]
fn chat_completions_maps_format_to_response_format() {
    let options = json!({});
    let r = Request {
        format: Some("verbose_json"),
        ..request("whisper-1", &options)
    };
    let payload = multipart_payload(Family::OpenAI, &r).unwrap();
    assert_eq!(payload["response_format"], json!("verbose_json"));
}

// spec: protocols/chat_completions/transcription_spec.rb:41 .render_transcription_payload > maps speaker names and references to the known speaker fields
#[test]
fn chat_completions_maps_speaker_names_and_references_to_the_known_speaker_fields() {
    let options = json!({});
    let names = vec!["Alice".to_string()];
    let references = vec![ruby_wav()];
    let r = Request {
        speaker_names: Some(&names),
        speaker_references: Some(&references),
        ..request("gpt-4o-transcribe-diarize", &options)
    };
    let payload = multipart_payload(Family::OpenAI, &r).unwrap();
    assert_eq!(payload["known_speaker_names"], json!(["Alice"]));
    let refs = payload["known_speaker_references"].as_array().unwrap();
    assert_eq!(refs.len(), 1);
    assert!(
        refs[0]
            .as_str()
            .unwrap()
            .starts_with("data:audio/wav;base64,")
    );
}

// spec: protocols/chat_completions/transcription_spec.rb:52 .render_transcription_payload > merges provider options over rendered defaults
#[test]
fn chat_completions_merges_provider_options_over_rendered_defaults() {
    let options =
        json!({ "chunking_strategy": { "type": "server_vad" }, "response_format": "json" });
    let payload = multipart_payload(
        Family::OpenAI,
        &request("gpt-4o-transcribe-diarize", &options),
    )
    .unwrap();
    assert_eq!(
        payload["chunking_strategy"],
        json!({ "type": "server_vad" })
    );
    assert_eq!(payload["response_format"], json!("json"));
}
