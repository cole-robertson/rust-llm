//! Port of `lib/ruby_llm/protocols/gemini/live_transcription.rb` (`Gemini::LiveTranscription`):
//! transcription over the Live API's `BidiGenerateContent` WebSocket. The audio must be mono
//! 16-bit PCM WAV; it is sent in 100 ms frames between explicit activity boundaries once the
//! server acknowledges the setup, and the transcript is final only when generation completes.

use base64::Engine;
use serde_json::{Value, json};

use super::wav_audio::WavAudio;
use super::{Request, Transcription, TranscriptionChunk};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::protocols::{deep_merge, int};
use crate::transport::WebsocketConnection;

/// `websocket_service`.
const WEBSOCKET_SERVICE: &str =
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

/// `LiveTranscription#transcribe`, after `Provider#transcribe` rendered the timestamp options.
pub(super) async fn transcribe(
    audio: &[u8],
    r: &Request<'_>,
    api_base: &str,
    headers: &[(String, String)],
    config: &Config,
    mut on_chunk: Option<&mut (dyn FnMut(&TranscriptionChunk) + Send)>,
) -> Result<Transcription> {
    validate_request(r)?;
    let audio = transcription_audio(audio)?;
    let setup = render_setup(r.model, r.language, r.prompt, r.provider_options)?;
    let url = websocket_url(api_base)?;
    let events = collect(&url, headers, config, &audio, &setup, &mut on_chunk).await?;
    let result = parse_events(&events, &audio, r.model);
    if let Some(on_chunk) = on_chunk {
        on_chunk(&TranscriptionChunk {
            kind: TranscriptionChunk::DONE.into(),
            delta: None,
            text: result.text.clone(),
            segment: None,
            raw: events.last().cloned().unwrap_or(Value::Null),
        });
    }
    Ok(result)
}

/// `validate_transcription_request`.
fn validate_request(r: &Request) -> Result<()> {
    if r.format.is_some()
        || r.speaker_names.is_some()
        || r.speaker_references.is_some()
        || r.temperature.is_some()
    {
        return Err(Error::Argument(
            "Google Live transcription does not accept format, diarization, or temperature".into(),
        ));
    }
    Ok(())
}

/// `transcription_audio`: non-empty mono 16-bit PCM with whole samples.
fn transcription_audio(content: &[u8]) -> Result<WavAudio> {
    let audio = WavAudio::new(content)?;
    check_audio(&audio)?;
    Ok(audio)
}

fn check_audio(audio: &WavAudio) -> Result<()> {
    if (audio.encoding, audio.channels, audio.bits_per_sample) != (1, 1, 16)
        || audio.data.is_empty()
        || !audio.data.len().is_multiple_of(2)
    {
        return Err(Error::Argument(
            "Google Live transcription requires non-empty mono 16-bit PCM WAV audio".into(),
        ));
    }
    Ok(())
}

/// `render_transcription_setup` with `validate_transcription_setup`.
fn render_setup(
    model: &str,
    language: Option<&str>,
    prompt: Option<&str>,
    provider_options: &Value,
) -> Result<Value> {
    let mut transcription = serde_json::Map::new();
    if let Some(language) = language {
        transcription.insert("languageCodes".into(), json!([language]));
    }
    if let Some(prompt) = prompt {
        transcription.insert("customVocabulary".into(), json!([prompt]));
    }
    let mut payload = json!({
        "model": format!("models/{model}"),
        "generationConfig": { "responseModalities": ["TEXT"] },
        "inputAudioTranscription": transcription,
        "realtimeInputConfig": { "automaticActivityDetection": { "disabled": true } },
    });
    if !provider_options.is_null() {
        deep_merge(&mut payload, provider_options);
    }
    let config = &payload["inputAudioTranscription"];
    let set = |key: &str| config.get(key).is_some_and(|v| !v.is_null() && v != false);
    if set("diarization") || set("wordTimestamp") {
        return Err(Error::Argument(
            "Google Live transcription does not support diarization or word timestamps".into(),
        ));
    }
    if payload.pointer("/realtimeInputConfig/automaticActivityDetection/disabled")
        != Some(&Value::Bool(true))
    {
        return Err(Error::Argument(
            "Google file transcription requires manual activity boundaries".into(),
        ));
    }
    Ok(json!({ "setup": payload }))
}

/// `transcription_websocket_url`: the API host with `ws`/`wss` and the Live service path.
fn websocket_url(api_base: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(api_base)
        .map_err(|e| Error::Configuration(format!("invalid gemini_api_base {api_base:?}: {e}")))?;
    let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
    url.set_scheme(scheme)
        .map_err(|()| Error::Configuration(format!("invalid gemini_api_base {api_base:?}")))?;
    url.set_path(&format!("/ws/{WEBSOCKET_SERVICE}"));
    Ok(url.to_string())
}

/// `collect_transcription`: the setup, then the audio once `setupComplete` arrives, reading
/// events until generation completes.
async fn collect(
    url: &str,
    headers: &[(String, String)],
    config: &Config,
    audio: &WavAudio,
    setup: &Value,
    on_chunk: &mut Option<&mut (dyn FnMut(&TranscriptionChunk) + Send)>,
) -> Result<Vec<Value>> {
    let mut events: Vec<Value> = Vec::new();
    WebsocketConnection::open(url, headers, config, async |socket| {
        socket.send_text(setup.to_string()).await?;
        let ready = tokio::sync::Notify::new();
        let write = async {
            ready.notified().await;
            send_audio(socket, audio).await
        };
        socket
            .each_message(write, |message| {
                let event: Value = serde_json::from_slice(&message)?;
                if event.get("setupComplete").is_some() {
                    ready.notify_one();
                }
                process_event(&event, on_chunk)?;
                let complete = event
                    .pointer("/serverContent/generationComplete")
                    .is_some_and(|v| !v.is_null() && v != false);
                events.push(event);
                if complete {
                    socket.close();
                }
                Ok(())
            })
            .await
    })
    .await?;
    let completed = events
        .last()
        .and_then(|e| e.pointer("/serverContent/generationComplete"))
        .is_some_and(|v| !v.is_null() && v != false);
    if !completed {
        return Err(Error::Api(
            "Google Live transcription ended before generation completed".into(),
            None,
        ));
    }
    Ok(events)
}

/// `send_transcription_audio`.
async fn send_audio(socket: &WebsocketConnection, audio: &WavAudio) -> Result<()> {
    for frame in audio_frames(audio.sample_rate, &audio.data) {
        socket.send_text(frame).await?;
    }
    Ok(())
}

/// The frames `send_transcription_audio` sends: `activityStart`, 100 ms of whole samples per
/// frame, and `activityEnd`.
fn audio_frames(sample_rate: u32, data: &[u8]) -> Vec<String> {
    let bytes = (sample_rate as usize / 10).max(1) * 2;
    let mut frames = vec![json!({ "realtimeInput": { "activityStart": {} } }).to_string()];
    for chunk in data.chunks(bytes) {
        let input = json!({ "audio": {
            "mimeType": format!("audio/pcm;rate={sample_rate}"),
            "data": base64::engine::general_purpose::STANDARD.encode(chunk),
        } });
        frames.push(json!({ "realtimeInput": input }).to_string());
    }
    frames.push(json!({ "realtimeInput": { "activityEnd": {} } }).to_string());
    frames
}

/// `process_transcription_event`: provider errors raise; interim text is a partial, final input
/// transcription a delta.
fn process_event(
    event: &Value,
    on_chunk: &mut Option<&mut (dyn FnMut(&TranscriptionChunk) + Send)>,
) -> Result<()> {
    if event.get("error").is_some_and(|e| !e.is_null()) {
        let message = event
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("Google Live transcription failed");
        return Err(Error::Api(message.into(), None));
    }
    let Some(on_chunk) = on_chunk else {
        return Ok(());
    };
    let content = event.get("serverContent").cloned().unwrap_or(json!({}));
    let text = |key: &str| {
        content
            .pointer(&format!("/{key}/text"))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    if content
        .get("interimInputTranscription")
        .is_some_and(|v| !v.is_null())
    {
        on_chunk(&TranscriptionChunk {
            kind: TranscriptionChunk::PARTIAL.into(),
            delta: None,
            text: text("interimInputTranscription"),
            segment: None,
            raw: event.clone(),
        });
    } else if content
        .get("inputTranscription")
        .is_some_and(|v| !v.is_null())
    {
        on_chunk(&TranscriptionChunk {
            kind: TranscriptionChunk::DELTA.into(),
            delta: text("inputTranscription"),
            text: None,
            segment: None,
            raw: event.clone(),
        });
    }
    Ok(())
}

/// `parse_transcription_events`: the joined input transcription, the audio's duration, and the
/// last reported usage.
fn parse_events(events: &[Value], audio: &WavAudio, model: &str) -> Transcription {
    let text: String = events
        .iter()
        .filter_map(|e| {
            e.pointer("/serverContent/inputTranscription/text")?
                .as_str()
        })
        .collect();
    let usage = events
        .iter()
        .rev()
        .find_map(|e| e.get("usageMetadata").filter(|u| !u.is_null()))
        .cloned()
        .unwrap_or(json!({}));
    let mut t = Transcription::new(Some(text), model);
    t.duration = Some(audio.duration());
    t.input_tokens = int(usage.get("promptTokenCount"));
    t.output_tokens = int(usage.get("candidatesTokenCount"));
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(encoding: u16, channels: u16, bits: u16, data: &[u8]) -> WavAudio {
        WavAudio {
            data: data.to_vec(),
            sample_rate: 24_000,
            channels,
            bits_per_sample: bits,
            encoding,
        }
    }

    // spec: protocols/gemini/live_transcription_spec.rb:68 sends complete PCM frames between explicit activity boundaries and preserves every input byte
    #[test]
    fn sends_complete_pcm_frames_between_explicit_activity_boundaries() {
        let data = b"ab".repeat(2000);
        let frames: Vec<Value> = audio_frames(11_025, &data)
            .iter()
            .map(|f| serde_json::from_str(f).unwrap())
            .collect();

        assert_eq!(
            frames[0],
            json!({ "realtimeInput": { "activityStart": {} } })
        );
        assert_eq!(
            frames[frames.len() - 1],
            json!({ "realtimeInput": { "activityEnd": {} } })
        );
        let chunks: Vec<Vec<u8>> = frames[1..frames.len() - 1]
            .iter()
            .map(|f| {
                let data = f
                    .pointer("/realtimeInput/audio/data")
                    .unwrap()
                    .as_str()
                    .unwrap();
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .unwrap()
            })
            .collect();
        assert!(chunks.iter().all(|c| c.len().is_multiple_of(2)));
        assert_eq!(chunks.concat(), data);
        assert_eq!(
            frames[1].pointer("/realtimeInput/audio/mimeType"),
            Some(&json!("audio/pcm;rate=11025"))
        );
    }

    // spec: protocols/gemini/live_transcription_spec.rb:118 rejects multiple files, stereo PCM, and incomplete sample frames before opening a socket
    // (`transcribe` takes one attachment, so the multiple-files half has no Rust counterpart.)
    #[test]
    fn rejects_stereo_pcm_and_incomplete_sample_frames() {
        let message = |audio: WavAudio| match check_audio(&audio) {
            Err(Error::Argument(m)) => m,
            other => panic!("expected an argument error, got {other:?}"),
        };
        assert!(message(wav(1, 2, 16, b"ab")).contains("mono"));
        assert!(message(wav(1, 1, 16, b"abc")).contains("16-bit"));
        assert!(message(wav(1, 1, 16, b"")).contains("non-empty"));
        assert!(check_audio(&wav(1, 1, 16, b"ab")).is_ok());
    }

    #[test]
    fn renders_the_live_setup_and_url_like_ruby() {
        let setup = render_setup(
            "gemini-3.5-transcribe-live",
            Some("en-US"),
            None,
            &json!({}),
        )
        .unwrap();
        assert_eq!(
            setup,
            json!({ "setup": {
                "model": "models/gemini-3.5-transcribe-live",
                "generationConfig": { "responseModalities": ["TEXT"] },
                "inputAudioTranscription": { "languageCodes": ["en-US"] },
                "realtimeInputConfig": { "automaticActivityDetection": { "disabled": true } },
            } })
        );
        assert_eq!(
            websocket_url("https://generativelanguage.googleapis.com/v1beta").unwrap(),
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        );
        assert!(matches!(
            render_setup(
                "m",
                None,
                None,
                &json!({ "realtimeInputConfig": { "automaticActivityDetection": { "disabled": false } } })
            ),
            Err(Error::Argument(m)) if m.contains("manual activity boundaries")
        ));
    }
}
