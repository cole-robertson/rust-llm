//! Port of `lib/ruby_llm/protocols/xai/streaming_transcription.rb` (`XAI::StreamingTranscription`):
//! xAI's `/v1/stt` WebSocket. The WAV header picks the wire encoding and sample rate, the raw
//! samples go out in 100 ms binary frames once the server creates the transcript, and interim
//! results arrive as partials until a segment is final. The transcription ends when every audio
//! channel has sent `transcript.done`.

use serde_json::{Map, Value, json};

use super::wav_audio::WavAudio;
use super::{Transcription, TranscriptionChunk};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::transport::WebsocketConnection;

/// `SAMPLE_RATES`.
const SAMPLE_RATES: [u32; 6] = [8000, 16_000, 22_050, 24_000, 44_100, 48_000];

/// `ENCODINGS`: WAVE format tag and bits per sample to xAI's encoding name.
fn encoding_for(audio: &WavAudio) -> Option<&'static str> {
    match (audio.encoding, audio.bits_per_sample) {
        (1, 16) => Some("pcm"),
        (6, 8) => Some("alaw"),
        (7, 8) => Some("mulaw"),
        _ => None,
    }
}

type OnChunk<'a> = &'a mut (dyn FnMut(&TranscriptionChunk) + Send);

/// `stream_transcription(payload, model:)`. `payload` is the rendered multipart payload without
/// its file; its fields become query parameters.
pub(super) async fn stream_transcription(
    audio: &[u8],
    payload: &Map<String, Value>,
    model: &str,
    api_base: &str,
    headers: &[(String, String)],
    config: &Config,
    on_chunk: OnChunk<'_>,
) -> Result<Transcription> {
    let audio = WavAudio::new(audio)?;
    let url = streaming_transcription_url(payload, &audio, api_base)?;
    let mut segments: Vec<Value> = Vec::new();
    let mut completed: Vec<Value> = Vec::new();
    WebsocketConnection::open(&url, headers, config, async |socket| {
        receive(socket, &audio, &mut segments, &mut completed, on_chunk).await
    })
    .await?;
    if completed.len() != usize::from(audio.channels) {
        return Err(Error::Api(
            "xAI transcription ended before its final transcript".into(),
            None,
        ));
    }
    let language = payload.get("language").and_then(Value::as_str);
    let result = build_streamed_transcription(&segments, &completed, model, language);
    on_chunk(&TranscriptionChunk {
        kind: TranscriptionChunk::DONE.into(),
        delta: None,
        text: result.text.clone(),
        segment: None,
        raw: completed.last().cloned().unwrap_or(Value::Null),
    });
    Ok(result)
}

/// `receive_streamed_transcription`.
async fn receive(
    socket: &WebsocketConnection,
    audio: &WavAudio,
    segments: &mut Vec<Value>,
    completed: &mut Vec<Value>,
    on_chunk: OnChunk<'_>,
) -> Result<()> {
    let ready = tokio::sync::Notify::new();
    let write = async {
        ready.notified().await;
        send_transcription_audio(socket, audio).await
    };
    socket
        .each_message(write, |message| {
            let event: Value = serde_json::from_slice(&message)?;
            process_transcription_event(&event, segments, completed, &ready, on_chunk)?;
            if completed.len() == usize::from(audio.channels) {
                socket.close();
            }
            Ok(())
        })
        .await
}

/// `process_transcription_event`.
fn process_transcription_event(
    event: &Value,
    segments: &mut Vec<Value>,
    completed: &mut Vec<Value>,
    ready: &tokio::sync::Notify,
    on_chunk: OnChunk<'_>,
) -> Result<()> {
    match event.get("type").and_then(Value::as_str) {
        Some("transcript.created") => ready.notify_one(),
        Some("transcript.partial") => process_transcription_segment(event, segments, on_chunk),
        Some("transcript.done") => {
            process_transcription_segment(event, segments, on_chunk);
            let channel = event.get("channel_index");
            if !completed
                .iter()
                .any(|item| item.get("channel_index") == channel)
            {
                completed.push(event.clone());
            }
        }
        Some("error") => {
            let message = event
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("xAI transcription failed");
            return Err(Error::Api(message.into(), None));
        }
        _ => {}
    }
    Ok(())
}

/// `streaming_transcription_url`: the payload fields plus the audio format as query parameters,
/// arrays as repeated keys, on the `ws`/`wss` form of the API base.
fn streaming_transcription_url(
    payload: &Map<String, Value>,
    audio: &WavAudio,
    api_base: &str,
) -> Result<String> {
    let encoding = encoding_for(audio);
    let (Some(encoding), true, true) = (
        encoding,
        SAMPLE_RATES.contains(&audio.sample_rate),
        (1..=8).contains(&audio.channels),
    ) else {
        return Err(Error::Argument(
            "xAI streaming requires 16-bit PCM or 8-bit G.711 WAV audio at a supported sample rate"
                .into(),
        ));
    };
    let mut params = payload.clone();
    params.remove("file");
    params.insert("encoding".into(), encoding.into());
    params.insert("sample_rate".into(), audio.sample_rate.into());
    params.insert("interim_results".into(), true.into());
    params.insert("channels".into(), audio.channels.into());
    params.insert("multichannel".into(), (audio.channels > 1).into());

    let base = format!("{}/", api_base.trim_end_matches('/'));
    let mut url = reqwest::Url::parse(&base)
        .and_then(|b| b.join("stt"))
        .map_err(|e| Error::Configuration(format!("invalid xai_api_base {api_base:?}: {e}")))?;
    {
        let mut query = url.query_pairs_mut();
        for (key, value) in &params {
            let items = match value {
                Value::Array(items) => items.clone(),
                Value::Null => Vec::new(),
                other => vec![other.clone()],
            };
            for item in items {
                let text = match item {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                query.append_pair(key, &text);
            }
        }
    }
    let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
    url.set_scheme(scheme)
        .map_err(|()| Error::Configuration(format!("invalid xai_api_base {api_base:?}")))?;
    Ok(url.to_string())
}

/// `send_transcription_audio`: 100 ms of audio per binary frame, paced in real time, then
/// `audio.done`.
async fn send_transcription_audio(socket: &WebsocketConnection, audio: &WavAudio) -> Result<()> {
    let bytes = (audio.sample_rate as usize
        * usize::from(audio.channels)
        * usize::from(audio.bits_per_sample)
        / 8
        / 10)
        .max(1);
    for chunk in audio.data.chunks(bytes) {
        socket.send_binary(chunk.to_vec()).await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    socket
        .send_text(json!({ "type": "audio.done" }).to_string())
        .await
}

/// `process_transcription_segment`: interim text is a partial; a final segment is a delta unless
/// it repeats one already received.
fn process_transcription_segment(event: &Value, segments: &mut Vec<Value>, on_chunk: OnChunk<'_>) {
    let text = match event.get("text") {
        Some(Value::String(s)) => s.clone(),
        None | Some(Value::Null) => String::new(),
        Some(other) => other.to_string(),
    };
    if text.is_empty() {
        return;
    }
    let is_final = event
        .get("is_final")
        .is_some_and(|v| !v.is_null() && v != false);
    if !(is_final || event.get("type").and_then(Value::as_str) == Some("transcript.done")) {
        on_chunk(&TranscriptionChunk {
            kind: TranscriptionChunk::PARTIAL.into(),
            delta: None,
            text: Some(text),
            segment: None,
            raw: event.clone(),
        });
        return;
    }
    let segment = parse_transcription_segment(event);
    if segments
        .iter()
        .any(|previous| is_duplicate_segment(previous, &segment))
    {
        return;
    }
    let delta = if segments.is_empty() {
        text
    } else {
        format!(" {text}")
    };
    segments.push(segment.clone());
    on_chunk(&TranscriptionChunk {
        kind: TranscriptionChunk::SEGMENT.into(),
        delta: Some(delta),
        text: None,
        segment: Some(segment),
        raw: event.clone(),
    });
}

/// `parse_transcription_segment`: the event's text, timing, channel, words, and language, with
/// absent fields left out.
fn parse_transcription_segment(event: &Value) -> Value {
    let start = event.get("start").filter(|v| !v.is_null());
    let end = start.and_then(Value::as_f64).map(|start| {
        let duration = event.get("duration").and_then(Value::as_f64).unwrap_or(0.0);
        Value::from(start + duration)
    });
    let mut segment = Map::new();
    for (key, value) in [
        ("text", event.get("text").cloned()),
        ("start", start.cloned()),
        ("end", end),
        ("channel", event.get("channel_index").cloned()),
        ("words", event.get("words").cloned()),
        ("language", event.get("language").cloned()),
    ] {
        if let Some(value) = value.filter(|v| !v.is_null()) {
            segment.insert(key.into(), value);
        }
    }
    Value::Object(segment)
}

/// `duplicate_transcription_segment?`: the same text, channel, and words, at the same time
/// unless the repeat carries no time.
fn is_duplicate_segment(previous: &Value, segment: &Value) -> bool {
    let same = |keys: &[&str]| keys.iter().all(|k| previous.get(*k) == segment.get(*k));
    same(&["text", "channel", "words"])
        && (segment.get("start").is_none() || same(&["start", "end"]))
}

/// `build_streamed_transcription`.
fn build_streamed_transcription(
    segments: &[Value],
    completed: &[Value],
    model: &str,
    language: Option<&str>,
) -> Transcription {
    let text = segments
        .iter()
        .map(|s| s.get("text").and_then(Value::as_str).unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ");
    let mut t = Transcription::new(Some(text), model);
    t.language = language.map(str::to_string).or_else(|| {
        segments
            .iter()
            .filter_map(|s| s.get("language").and_then(Value::as_str))
            .next_back()
            .map(str::to_string)
    });
    t.duration = completed
        .iter()
        .filter_map(|e| e.get("duration").and_then(Value::as_f64))
        .reduce(f64::max);
    t.words = Some(
        segments
            .iter()
            .flat_map(|s| {
                s.get("words")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect(),
    );
    t.segments = Some(segments.to_vec());
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment_event() -> Value {
        json!({ "type": "transcript.partial", "text": "Ruby is useful.", "start": 0.0, "duration": 1.0,
                "words": [{ "text": "Ruby", "start": 0.0, "end": 0.3, "speaker": 0 }],
                "is_final": true, "speech_final": false, "language": "en" })
    }

    fn merged(mut event: Value, extra: Value) -> Value {
        crate::protocols::deep_merge(&mut event, &extra);
        event
    }

    fn without(mut event: Value, key: &str) -> Value {
        event.as_object_mut().unwrap().remove(key);
        event
    }

    fn fixture() -> WavAudio {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ruby.wav"
        ))
        .unwrap();
        WavAudio::new(&bytes).unwrap()
    }

    // spec: protocols/xai/streaming_transcription_spec.rb:21 deduplicates repeated finalized segments while keeping interim revisions separate
    #[test]
    fn deduplicates_repeated_finalized_segments_while_keeping_interim_revisions_separate() {
        let mut segments = Vec::new();
        let mut chunks: Vec<TranscriptionChunk> = Vec::new();
        let mut push = |c: &TranscriptionChunk| chunks.push(c.clone());
        let segment = segment_event();
        for event in [
            merged(segment.clone(), json!({ "is_final": false })),
            segment.clone(),
            merged(segment.clone(), json!({ "speech_final": true })),
            json!({ "type": "transcript.done", "text": "", "duration": 1.0 }),
        ] {
            process_transcription_segment(&event, &mut segments, &mut push);
        }

        let kinds: Vec<&str> = chunks.iter().map(|c| c.kind.as_str()).collect();
        assert_eq!(
            kinds,
            [TranscriptionChunk::PARTIAL, TranscriptionChunk::SEGMENT]
        );
        assert!(chunks[0].is_partial());
        assert_eq!(chunks[0].text.as_deref(), Some("Ruby is useful."));
        assert_eq!(chunks[0].delta, None);
        assert_eq!(chunks[1].delta.as_deref(), Some("Ruby is useful."));
        assert_eq!(chunks[1].raw, segment);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0]["words"][0]["speaker"], json!(0));
    }

    // spec: protocols/xai/streaming_transcription_spec.rb:37 keeps an intentionally repeated phrase at a different audio position
    #[test]
    fn keeps_an_intentionally_repeated_phrase_at_a_different_audio_position() {
        let mut segments = Vec::new();
        let mut chunks: Vec<TranscriptionChunk> = Vec::new();
        let mut push = |c: &TranscriptionChunk| chunks.push(c.clone());
        let event = without(segment_event(), "words");
        for event in [event.clone(), merged(event, json!({ "start": 2.0 }))] {
            process_transcription_segment(&event, &mut segments, &mut push);
        }

        let deltas: String = chunks.iter().filter_map(|c| c.delta.clone()).collect();
        assert_eq!(deltas, "Ruby is useful. Ruby is useful.");
        let starts: Vec<&Value> = segments.iter().map(|s| &s["start"]).collect();
        assert_eq!(starts, [&json!(0.0), &json!(2.0)]);
    }

    // spec: protocols/xai/streaming_transcription_spec.rb:48 derives the wire encoding and sample rate from WAV data and preserves repeated key terms
    #[test]
    fn derives_the_wire_encoding_and_sample_rate_from_wav_data() {
        let payload = json!({ "language": "en", "keyterm": ["Ruby", "Rails"], "diarize": true });
        let url = streaming_transcription_url(
            payload.as_object().unwrap(),
            &fixture(),
            "https://api.x.ai/v1",
        )
        .unwrap();
        let params: Vec<(String, String)> = reqwest::Url::parse(&url)
            .unwrap()
            .query_pairs()
            .into_owned()
            .collect();

        assert!(url.starts_with("wss://api.x.ai/v1/stt?"), "{url}");
        for (key, value) in [
            ("encoding", "pcm"),
            ("sample_rate", "24000"),
            ("interim_results", "true"),
            ("keyterm", "Ruby"),
            ("keyterm", "Rails"),
            ("diarize", "true"),
        ] {
            assert!(
                params.contains(&(key.to_string(), value.to_string())),
                "{key}={value} in {params:?}"
            );
        }
    }

    // spec: protocols/xai/streaming_transcription_spec.rb:58 retains finalized text, words, language and duration when completion contains no text
    #[test]
    fn retains_finalized_text_words_language_and_duration_when_completion_contains_no_text() {
        let segments = [parse_transcription_segment(&segment_event())];
        let completed =
            [json!({ "type": "transcript.done", "text": "", "words": [], "duration": 3.7 })];
        let result = build_streamed_transcription(&segments, &completed, "grok-stt", None);

        assert_eq!(result.text.as_deref(), Some("Ruby is useful."));
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.duration, Some(3.7));
        assert_eq!(
            result.words,
            Some(segment_event()["words"].as_array().unwrap().clone())
        );
    }

    // spec: protocols/xai/streaming_transcription_spec.rb:76 rejects unsupported WAV encodings before opening a socket
    #[test]
    fn rejects_unsupported_wav_encodings_before_opening_a_socket() {
        let audio = WavAudio {
            encoding: 3,
            bits_per_sample: 32,
            ..fixture()
        };
        let error =
            streaming_transcription_url(&Map::new(), &audio, "https://api.x.ai/v1").unwrap_err();
        assert!(
            matches!(&error, Error::Argument(m) if m.contains("16-bit PCM or 8-bit G.711 WAV")),
            "{error:?}"
        );
    }

    // spec: protocols/xai/streaming_transcription_spec.rb:83 keeps completion events separate for each audio channel and surfaces server errors
    #[test]
    fn keeps_completion_events_separate_for_each_audio_channel_and_surfaces_server_errors() {
        let mut segments = Vec::new();
        let mut completed = Vec::new();
        let ready = tokio::sync::Notify::new();
        let mut ignore = |_: &TranscriptionChunk| {};
        let done =
            json!({ "type": "transcript.done", "text": "", "duration": 1.0, "channel_index": 0 });
        for event in [
            done.clone(),
            done.clone(),
            merged(done, json!({ "channel_index": 1 })),
        ] {
            process_transcription_event(&event, &mut segments, &mut completed, &ready, &mut ignore)
                .unwrap();
        }

        let channels: Vec<&Value> = completed.iter().map(|e| &e["channel_index"]).collect();
        assert_eq!(channels, [&json!(0), &json!(1)]);
        let error = process_transcription_event(
            &json!({ "type": "error", "message": "Invalid audio" }),
            &mut segments,
            &mut completed,
            &ready,
            &mut ignore,
        )
        .unwrap_err();
        assert!(
            matches!(&error, Error::Api(m, _) if m == "Invalid audio"),
            "{error:?}"
        );
    }
}
