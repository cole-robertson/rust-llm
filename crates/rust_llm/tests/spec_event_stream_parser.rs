//! Ports of RubyLLM 2.1's `transport/event_stream_parser_spec.rb`: the conformance cases of the
//! HTML standard's "Interpreting an event stream", fed whole, byte by byte, and split at random,
//! plus the recorded-stream comparison. Ruby compares against the `event_stream_parser` gem; the
//! port has no such gem, so `reference_events` below, a whole-body line-by-line reading of the
//! standard, plays its part.

use rand::{Rng, SeedableRng, rngs::StdRng};
use rust_llm::transport::EventStreamParser;

type Event = (String, String, String);

fn ev(kind: &str, data: &str, id: &str) -> Event {
    (kind.into(), data.into(), id.into())
}

/// `parse(stream, pieces:, parser:)`.
fn parse_pieces(pieces: &[Vec<u8>], parser: &mut EventStreamParser) -> Vec<Event> {
    let mut events = Vec::new();
    for piece in pieces {
        for e in parser.feed(piece) {
            events.push((e.event.unwrap_or_else(|| "message".into()), e.data, e.id));
        }
    }
    events
}

fn parse(stream: &[u8]) -> Vec<Event> {
    parse_pieces(&[stream.to_vec()], &mut EventStreamParser::new())
}

fn bytewise(stream: &[u8]) -> Vec<Vec<u8>> {
    stream.iter().map(|b| vec![*b]).collect()
}

fn split_at_random(stream: &[u8], random: &mut StdRng) -> Vec<Vec<u8>> {
    let mut cuts: Vec<usize> = (0..random.gen_range(1..=8))
        .map(|_| random.gen_range(0..=stream.len()))
        .collect();
    cuts.sort();
    cuts.dedup();
    let starts = std::iter::once(0).chain(cuts.iter().copied());
    let ends = cuts.iter().copied().chain(std::iter::once(stream.len()));
    starts
        .zip(ends)
        .map(|(a, b)| stream[a..b].to_vec())
        .collect()
}

fn conformance() -> Vec<(&'static str, Vec<u8>, Vec<Event>)> {
    vec![
        ("dispatches an event at a blank line", b"data: hello\n\n".to_vec(), vec![ev("message", "hello", "")]),
        ("reports the event type, defaulting to message", b"event: add\ndata: 1\n\ndata: 2\n\n".to_vec(), vec![ev("add", "1", ""), ev("message", "2", "")]),
        ("treats an empty event type as message", b"event:\ndata: 1\n\n".to_vec(), vec![ev("message", "1", "")]),
        ("ends lines with CRLF", b"data: a\r\n\r\n".to_vec(), vec![ev("message", "a", "")]),
        ("ends lines with a lone CR", b"data: a\r\rdata: b\r\r".to_vec(), vec![ev("message", "a", ""), ev("message", "b", "")]),
        ("ends lines with any mix of CR, LF, and CRLF", b"data:test\r\ndata\ndata:test\r\n\r\n".to_vec(), vec![ev("message", "test\n\ntest", "")]),
        ("strips one byte order mark at the start of the stream", "\u{FEFF}data:1\n\n\u{FEFF}data:2\n\ndata:3\n\n".as_bytes().to_vec(), vec![ev("message", "1", ""), ev("message", "3", "")]),
        ("strips only the first of two byte order marks", "\u{FEFF}\u{FEFF}data:1\n\ndata:2\n\ndata:3\n\n".as_bytes().to_vec(), vec![ev("message", "2", ""), ev("message", "3", "")]),
        ("ignores comment lines", b": ping\ndata: a\n:\n\n".to_vec(), vec![ev("message", "a", "")]),
        ("takes a line without a colon as a field with an empty value", b"data\n\ndata\ndata\n\ndata:test\n\n".to_vec(), vec![ev("message", "", ""), ev("message", "\n", ""), ev("message", "test", "")]),
        ("strips one leading space from the value", b"data:  two\n\ndata:none\n\n".to_vec(), vec![ev("message", " two", ""), ev("message", "none", "")]),
        ("keeps a leading tab", b"data:\ttab\n\n".to_vec(), vec![ev("message", "\ttab", "")]),
        ("splits the field at the first colon only", b"data: a:b\n\n".to_vec(), vec![ev("message", "a:b", "")]),
        ("joins data lines with LF and drops the final LF", b"data: a\ndata:\ndata: b\ndata:\n\n".to_vec(), vec![ev("message", "a\n\nb\n", "")]),
        ("ignores unknown and misspelled fields", b"data:test\n data\ndata\nfoobar:xxx\njustsometext\nData:x\ndata :x\ndata:test\n\n".to_vec(), vec![ev("message", "test\n\ntest", "")]),
        ("parses fields exactly as the standard does", b"data:\0\ndata:  2\rData:1\ndata\0:2\ndata:1\r\0data:4\nda-ta:3\rdata_5\ndata:3\rdata:\r\n data:32\ndata:4\n\n".to_vec(), vec![ev("message", "\0\n 2\n1\n3\n\n4", "")]),
        ("keeps the last event ID for later events", b"id: 1\ndata: a\n\ndata: b\n\n".to_vec(), vec![ev("message", "a", "1"), ev("message", "b", "1")]),
        ("resets the last event ID with an empty id", b"id:1\ndata:x\n\nid\ndata:y\n\n".to_vec(), vec![ev("message", "x", "1"), ev("message", "y", "")]),
        ("ignores an id containing NUL", b"id:1\ndata:x\n\nid:2\0\ndata:y\n\n".to_vec(), vec![ev("message", "x", "1"), ev("message", "y", "1")]),
        ("skips dispatch when the data buffer is empty and resets the event type", b"event: x\nid: 7\n\ndata: a\n\n".to_vec(), vec![ev("message", "a", "7")]),
        ("discards an event the stream ends in the middle of", b"data: a\n\nid: 2\ndata: b".to_vec(), vec![ev("message", "a", "")]),
        ("decodes UTF-8", "data: ok…\n\n".as_bytes().to_vec(), vec![ev("message", "ok…", "")]),
        ("replaces invalid UTF-8 with U+FFFD", b"data: \xFF\xFEok\n\n".to_vec(), vec![ev("message", "\u{FFFD}\u{FFFD}ok", "")]),
    ]
}

// spec: transport/event_stream_parser_spec.rb:78 description (each conformance case)
#[test]
fn parses_each_conformance_case() {
    for (description, stream, events) in conformance() {
        assert_eq!(parse(&stream), events, "{description}");
    }
}

// spec: transport/event_stream_parser_spec.rb:83 parses every case the same way whatever pieces it arrives in
#[test]
fn parses_every_case_the_same_way_whatever_pieces_it_arrives_in() {
    let mut random = StdRng::seed_from_u64(42);
    for (description, stream, events) in conformance() {
        let mut parser = EventStreamParser::new();
        assert_eq!(
            parse_pieces(&bytewise(&stream), &mut parser),
            events,
            "{description}"
        );
        for _ in 0..20 {
            let pieces = split_at_random(&stream, &mut random);
            let mut parser = EventStreamParser::new();
            assert_eq!(
                parse_pieces(&pieces, &mut parser),
                events,
                "{description}: {pieces:?}"
            );
        }
    }
}

// spec: transport/event_stream_parser_spec.rb:92 keeps a CRLF split across pieces as one line break
#[test]
fn keeps_a_crlf_split_across_pieces_as_one_line_break() {
    let pieces: Vec<Vec<u8>> = ["data: a\r", "", "\n", "\r", "\n"]
        .iter()
        .map(|p| p.as_bytes().to_vec())
        .collect();
    assert_eq!(
        parse_pieces(&pieces, &mut EventStreamParser::new()),
        vec![ev("message", "a", "")]
    );
}

// spec: transport/event_stream_parser_spec.rb:96 keeps a byte order mark split across pieces out of the first field
#[test]
fn keeps_a_byte_order_mark_split_across_pieces_out_of_the_first_field() {
    let pieces = vec![
        b"\xEF".to_vec(),
        b"\xBB".to_vec(),
        b"\xBFdata: a\n\n".to_vec(),
    ];
    assert_eq!(
        parse_pieces(&pieces, &mut EventStreamParser::new()),
        vec![ev("message", "a", "")]
    );
}

// spec: transport/event_stream_parser_spec.rb:100 decodes characters split across pieces
#[test]
fn decodes_characters_split_across_pieces() {
    let stream = "data: é€😀\n\n".as_bytes();
    let pieces = vec![
        stream[0..7].to_vec(),
        stream[7..11].to_vec(),
        stream[11..17].to_vec(),
        stream[17..].to_vec(),
    ];
    assert_eq!(
        parse_pieces(&pieces, &mut EventStreamParser::new()),
        vec![ev("message", "é€😀", "")]
    );
}

// spec: transport/event_stream_parser_spec.rb:107 remembers the reconnection time, ignoring values that are not all digits
#[test]
fn remembers_the_reconnection_time_ignoring_values_that_are_not_all_digits() {
    let mut parser = EventStreamParser::new();
    let stream = "retry: 03000\n\nretry: 1000x\n\nretry\n\nretry: -1\n\nretry: ５\n\n";
    parse_pieces(&[stream.as_bytes().to_vec()], &mut parser);
    assert_eq!(parser.reconnection_time(), Some(3000));
}

// spec: transport/event_stream_parser_spec.rb:114 updates the last event ID even when no event is dispatched
#[test]
fn updates_the_last_event_id_even_when_no_event_is_dispatched() {
    let mut parser = EventStreamParser::new();
    assert!(parse_pieces(&[b"id: 5\n\n".to_vec()], &mut parser).is_empty());
    assert_eq!(parser.last_event_id(), "5");
}

// spec: transport/event_stream_parser_spec.rb:121 leaves the last event ID unchanged until the event completes
#[test]
fn leaves_the_last_event_id_unchanged_until_the_event_completes() {
    let mut parser = EventStreamParser::new();
    parse_pieces(
        &[b"id: 1\ndata: a\n\nid: 2\ndata: b\n".to_vec()],
        &mut parser,
    );
    assert_eq!(parser.last_event_id(), "1");
}

// spec: transport/event_stream_parser_spec.rb:128 continues after the lines it already delivered when a handler raises
#[test]
fn continues_after_the_lines_it_already_delivered_when_a_handler_raises() {
    let mut parser = EventStreamParser::new();
    let mut events = Vec::new();
    let mut handler = |event: rust_llm::transport::SseEvent| {
        if event.data == "boom" {
            return Err(event.data);
        }
        events.push(event.data);
        Ok(())
    };
    assert_eq!(
        parser.feed_with("data: boom\n\ndata: a\n\n", &mut handler),
        Err("boom".to_string())
    );
    parser.feed_with("data: b\n\n", &mut handler).unwrap();
    assert_eq!(events, ["a", "b"]);
}

/// The standard read over the whole body at once: one BOM dropped, lines split at CRLF, CR, or
/// LF, then each line handled. Stands in for the `event_stream_parser` gem.
fn reference_events(stream: &[u8]) -> Vec<Event> {
    let text = String::from_utf8_lossy(stream);
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(&text);
    let mut lines = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(['\r', '\n']) {
        lines.push(&rest[..i]);
        let skip = if rest[i..].starts_with("\r\n") { 2 } else { 1 };
        rest = &rest[i + skip..];
    }
    let (mut events, mut data, mut kind, mut id_buffer) =
        (Vec::new(), String::new(), String::new(), String::new());
    for line in lines {
        if line.is_empty() {
            let id = id_buffer.clone();
            if !data.is_empty() {
                data.pop();
                let kind = if kind.is_empty() {
                    "message".into()
                } else {
                    kind.clone()
                };
                events.push((kind, std::mem::take(&mut data), id));
            }
            kind.clear();
            continue;
        }
        if line.starts_with(':') {
            continue;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => kind = value.into(),
            "data" => {
                data.push_str(value);
                data.push('\n');
            }
            "id" if !value.contains('\0') => id_buffer = value.into(),
            _ => {}
        }
    }
    events
}

fn split_in_random_sizes(stream: &[u8], random: &mut StdRng) -> Vec<Vec<u8>> {
    let mut pieces = Vec::new();
    let mut offset = 0;
    while offset < stream.len() {
        let end = (offset + random.gen_range(1..=512)).min(stream.len());
        pieces.push(stream[offset..end].to_vec());
        offset = end;
    }
    pieces
}

/// Every recorded `text/event-stream` body among the converted cassettes.
fn recorded_streams() -> Vec<Vec<u8>> {
    let dir = format!("{}/tests/cassettes", env!("CARGO_MANIFEST_DIR"));
    let mut streams = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(path).unwrap();
        if !text.contains("text/event-stream") {
            continue;
        }
        let interactions: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        for interaction in interactions {
            let content_type = interaction["response_headers"]
                .as_object()
                .and_then(|h| {
                    h.iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
                })
                .map(|(_, v)| v.to_string())
                .unwrap_or_default();
            let body = interaction["response_body"].as_str().unwrap_or_default();
            if content_type.contains("text/event-stream") && !body.is_empty() {
                streams.push(body.as_bytes().to_vec());
            }
        }
    }
    streams
}

// spec: transport/event_stream_parser_spec.rb:178 compared with the event_stream_parser gem > agrees on every recorded stream, however it is split
#[test]
fn agrees_on_every_recorded_stream_however_it_is_split() {
    let mut random = StdRng::seed_from_u64(2026);
    let streams = recorded_streams();
    for stream in &streams {
        let expected = reference_events(stream);
        let mut splits = vec![
            vec![stream.clone()],
            stream.chunks(64).map(<[u8]>::to_vec).collect(),
            split_in_random_sizes(stream, &mut random),
        ];
        if stream.len() < 8192 {
            splits.push(bytewise(stream));
        }
        for pieces in splits {
            assert_eq!(
                parse_pieces(&pieces, &mut EventStreamParser::new()),
                expected
            );
        }
    }
    assert!(streams.len() > 100, "{} recorded streams", streams.len());
}
