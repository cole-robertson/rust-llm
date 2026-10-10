//! Port of `lib/ruby_llm/transport/event_stream_parser.rb`, and of the stream state in
//! `lib/ruby_llm/protocol/streaming.rb` that decides between an event stream and a bare JSON body.

use crate::error::{Error, Result};

/// One server-sent event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// The event type; `None` is the default type, `message`.
    pub event: Option<String>,
    pub data: String,
    /// The last event ID when the event was dispatched (empty when none was set).
    pub id: String,
}

/// `Transport::EventStreamParser`: interprets a `text/event-stream` as the HTML standard
/// describes ("Interpreting an event stream"), fed in whatever pieces the network delivers. Bytes
/// stay buffered until a line ends, so a CRLF, a byte order mark, or a UTF-8 character split
/// across pieces reads as if it had arrived whole.
#[derive(Debug, Default)]
pub struct EventStreamParser {
    buffer: Vec<u8>,
    /// Where the next line starts.
    position: usize,
    data: String,
    event_type: String,
    id_buffer: String,
    last_event_id: String,
    reconnection_time: Option<u64>,
    started: bool,
    after_cr: bool,
}

const BOM: &[u8] = b"\xEF\xBB\xBF";

impl EventStreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// `#last_event_id`: updated at every blank line, even when no event is dispatched.
    pub fn last_event_id(&self) -> &str {
        &self.last_event_id
    }

    /// `#reconnection_time`: the last `retry` field whose value was all ASCII digits.
    pub fn reconnection_time(&self) -> Option<u64> {
        self.reconnection_time
    }

    /// `#feed(chunk) { |type, data, id| }`: hands each event the piece completes to `on_event`.
    /// When `on_event` fails, the lines already read stay read: the next `feed` continues after
    /// them.
    pub fn feed_with<E>(
        &mut self,
        chunk: impl AsRef<[u8]>,
        mut on_event: impl FnMut(SseEvent) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        self.buffer.drain(..self.position);
        self.position = 0;
        self.buffer.extend_from_slice(chunk.as_ref());
        if !self.start() {
            return Ok(());
        }
        self.skip_line_feed_after_carriage_return();
        while let Some(offset) = self.buffer[self.position..]
            .iter()
            .position(|b| matches!(b, b'\r' | b'\n'))
        {
            let index = self.position + offset;
            let line = self.buffer[self.position..index].to_vec();
            self.position = self.line_end(index);
            if let Some(event) = self.process_line(&line) {
                on_event(event)?;
            }
        }
        Ok(())
    }

    /// `feed`, collecting the events the piece completes.
    pub fn feed(&mut self, chunk: impl AsRef<[u8]>) -> Vec<SseEvent> {
        let mut events = Vec::new();
        let _ = self.feed_with(chunk, |event| {
            events.push(event);
            Ok::<(), std::convert::Infallible>(())
        });
        events
    }

    /// The events left once the body ends: an unterminated last event, which the port dispatches
    /// when the connection closes after it. (RubyLLM discards it; no recorded stream ends that
    /// way.)
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = self.feed("\n\n");
        if !self.data.is_empty() {
            events.extend(self.dispatch());
        }
        events
    }

    /// The stream decodes as UTF-8, which drops one byte order mark at its very start, so the
    /// first bytes wait until they can be told apart.
    fn start(&mut self) -> bool {
        if self.started {
            return true;
        }
        if self.buffer.len() < BOM.len() && BOM.starts_with(&self.buffer) {
            return false;
        }
        if self.buffer.starts_with(BOM) {
            self.position = BOM.len();
        }
        self.started = true;
        true
    }

    /// A carriage return that ended the previous piece may be the first half of a CRLF pair.
    fn skip_line_feed_after_carriage_return(&mut self) {
        if !self.after_cr || self.buffer.len() <= self.position {
            return;
        }
        self.after_cr = false;
        if self.buffer[self.position] == b'\n' {
            self.position += 1;
        }
    }

    fn line_end(&mut self, index: usize) -> usize {
        if self.buffer[index] != b'\r' {
            return index + 1;
        }
        if self.buffer.get(index + 1) == Some(&b'\n') {
            return index + 2;
        }
        self.after_cr = index + 1 == self.buffer.len();
        index + 1
    }

    fn process_line(&mut self, line: &[u8]) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line[0] == b':' {
            return None;
        }
        match line.iter().position(|&b| b == b':') {
            None => self.process_field(line, b""),
            Some(colon) => {
                let offset = if line.get(colon + 1) == Some(&b' ') {
                    colon + 2
                } else {
                    colon + 1
                };
                self.process_field(&line[..colon], &line[offset..]);
            }
        }
        None
    }

    fn process_field(&mut self, field: &[u8], value: &[u8]) {
        match field {
            b"event" => self.event_type = decode(value),
            b"data" => {
                self.data.push_str(&decode(value));
                self.data.push('\n');
            }
            b"id" if !value.contains(&0) => self.id_buffer = decode(value),
            b"retry" if !value.is_empty() && value.iter().all(u8::is_ascii_digit) => {
                self.reconnection_time = decode(value).parse().ok();
            }
            _ => {}
        }
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        self.last_event_id = self.id_buffer.clone();
        let mut data = std::mem::take(&mut self.data);
        let event_type = std::mem::take(&mut self.event_type);
        if data.is_empty() {
            return None;
        }
        data.pop();
        Some(SseEvent {
            event: (!event_type.is_empty()).then_some(event_type),
            data,
            id: self.last_event_id.clone(),
        })
    }
}

/// `#decode`: UTF-8, invalid bytes replaced with U+FFFD (`String#scrub`).
fn decode(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `Protocol::Streaming::MAX_JSON_BODY_BYTES`: how much of a bare JSON (or failed) streaming
/// body is kept while waiting for it to parse.
pub(crate) const MAX_JSON_BODY_BYTES: usize = 1024 * 1024;

/// `Protocol::Streaming::StreamState`: one attempt's parser, plus the bare JSON body a provider
/// may answer a streaming request with instead of events. Each attempt starts a fresh one.
pub(crate) struct StreamState {
    pub(crate) parser: EventStreamParser,
    pub(crate) buffer: Vec<u8>,
    json_body: Option<bool>,
    limit: usize,
}

impl StreamState {
    pub(crate) fn new(limit: usize) -> Self {
        StreamState {
            parser: EventStreamParser::new(),
            buffer: Vec::new(),
            json_body: None,
            limit,
        }
    }

    /// `process_stream_chunk` for one network read: events go to `on_event`; a bare JSON body
    /// that parses to an object with an `error` is returned as its raw text, for the caller to
    /// raise. Adapters cut reads anywhere, so only the first non-blank read decides whether the
    /// body is bare JSON.
    pub(crate) fn read(
        &mut self,
        chunk: &[u8],
        on_event: impl FnMut(SseEvent) -> Result<()>,
    ) -> Result<Option<String>> {
        if self.json_body.is_none() && !trim(chunk).is_empty() {
            self.json_body = Some(trim(chunk).starts_with(b"{"));
        }
        if self.json_body != Some(true) {
            self.parser.feed_with(chunk, on_event)?;
            return Ok(None);
        }
        // `handle_json_body`: refuse the read before retaining it, in bytes.
        if self.buffer.len() + chunk.len() > self.limit {
            return Err(Error::Api(
                format!("Streaming JSON response exceeds {} bytes", self.limit),
                None,
            ));
        }
        self.buffer.extend_from_slice(chunk);
        Ok(error_body(&self.buffer))
    }
}

/// `handle_failed_response`: appends a read of a failed response's body to `buffer`, unless it
/// would take the body past `limit`; then the read is not retained (`false`), so a large HTML
/// error page cannot grow the buffer, and the response status alone raises the error.
pub(crate) fn accumulate_failed_body(buffer: &mut Vec<u8>, chunk: &[u8], limit: usize) -> bool {
    if buffer.len() + chunk.len() > limit {
        return false;
    }
    buffer.extend_from_slice(chunk);
    true
}

/// The body as text when it parses to an object whose `error` is set (not null or false).
fn error_body(buffer: &[u8]) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_slice(buffer).ok()?;
    let error = parsed.as_object()?.get("error")?;
    (!error.is_null() && error != &serde_json::Value::Bool(false))
        .then(|| String::from_utf8_lossy(buffer).into_owned())
}

/// `String#strip` on bytes: ASCII whitespace and NUL.
fn trim(bytes: &[u8]) -> &[u8] {
    let blank = |b: &u8| b.is_ascii_whitespace() || *b == 0 || *b == 0x0b;
    let start = bytes.iter().position(|b| !blank(b)).unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !blank(b))
        .map_or(start, |i| i + 1);
    &bytes[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(state: &mut StreamState, chunk: &[u8]) -> Result<Option<String>> {
        state.read(chunk, |_| Ok(()))
    }

    #[test]
    fn sse_events_split_across_reads_are_reassembled() {
        let mut p = EventStreamParser::new();
        assert!(p.feed("event: message_start\ndata: {\"a\":").is_empty());
        let events = p.feed("1}\n\nevent: ping\ndata: {}\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[0].data, "{\"a\":1}");
    }

    // spec: protocol/streaming_spec.rb:243 rejects an oversized bare JSON body before retaining it
    #[test]
    fn rejects_an_oversized_bare_json_body_before_retaining_it() {
        let mut state = StreamState::new(32);
        let body = format!(r#"{{"detail":"{}"#, "x".repeat(32));
        let error = feed_all(&mut state, body.as_bytes()).unwrap_err();
        assert!(matches!(error, Error::Api(..)), "{error:?}");
        assert!(
            error.to_string().contains("JSON response exceeds 32 bytes"),
            "{error}"
        );
        assert!(state.buffer.is_empty());
    }

    // spec: protocol/streaming_spec.rb:252 limits the accumulated JSON bytes across network reads
    #[test]
    fn limits_the_accumulated_json_bytes_across_network_reads() {
        let mut state = StreamState::new(32);
        assert_eq!(feed_all(&mut state, br#"{"detail":""#).unwrap(), None);
        // Eleven two-byte characters: 22 bytes, past the limit although only 11 characters.
        let error = feed_all(&mut state, "é".repeat(11).as_bytes()).unwrap_err();
        assert!(
            error.to_string().contains("JSON response exceeds 32 bytes"),
            "{error}"
        );
        assert_eq!(state.buffer, br#"{"detail":""#);
    }

    // spec: protocol/streaming_spec.rb:262 stops retaining a failed response body past the limit, leaving its status to raise
    #[test]
    fn stops_retaining_a_failed_response_body_past_the_limit() {
        let mut buffer = b"<html>".to_vec();
        assert!(!accumulate_failed_body(
            &mut buffer,
            "x".repeat(32).as_bytes(),
            32
        ));
        assert_eq!(buffer, b"<html>");
    }
}
