//! Port of `lib/ruby_llm/mcp/content.rb`: reads MCP content blocks into text and attachments.
//! Tool results and prompt messages share the format.

use base64::Engine;
use serde_json::Value;

use crate::attachment::Attachment;

/// `Content.read`: text blocks joined by blank lines, and the files as attachments.
pub fn read(blocks: &[Value]) -> (String, Vec<Attachment>) {
    let mut texts = Vec::new();
    let mut attachments = Vec::new();
    for block in blocks {
        match part(block) {
            Some(Part::Text(text)) => texts.push(text),
            Some(Part::File(attachment)) => attachments.push(*attachment),
            None => {}
        }
    }
    (texts.join("\n\n"), attachments)
}

enum Part {
    Text(String),
    File(Box<Attachment>),
}

fn part(block: &Value) -> Option<Part> {
    let str_at = |key: &str| block.get(key).and_then(Value::as_str);
    match str_at("type")? {
        "text" => str_at("text").map(|t| Part::Text(t.into())),
        kind @ ("image" | "audio") => Some(Part::File(Box::new(attachment(
            str_at("data"),
            str_at("mimeType"),
            kind,
        )))),
        "resource" => embedded(block.get("resource").unwrap_or(&Value::Null)),
        "resource_link" => {
            let label: Vec<&str> = [str_at("title").or_else(|| str_at("name")), str_at("uri")]
                .into_iter()
                .flatten()
                .collect();
            Some(Part::Text(label.join(": ")))
        }
        _ => None,
    }
}

fn embedded(resource: &Value) -> Option<Part> {
    if let Some(text) = resource.get("text").and_then(Value::as_str) {
        return Some(Part::Text(text.into()));
    }
    let blob = resource.get("blob").and_then(Value::as_str)?;
    let uri = resource.get("uri").and_then(Value::as_str).unwrap_or("");
    Some(Part::File(Box::new(attachment(
        Some(blob),
        resource.get("mimeType").and_then(Value::as_str),
        &filename(uri),
    ))))
}

/// `Content.attachment`: names a nameless file after its kind plus the MIME type's extension.
pub(crate) fn attachment(data: Option<&str>, mime_type: Option<&str>, name: &str) -> Attachment {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.unwrap_or(""))
        .unwrap_or_default();
    let name = match mime_type.and_then(extension) {
        Some(ext) if !name.contains('.') => format!("{name}.{ext}"),
        _ => name.to_string(),
    };
    Attachment::from_bytes(bytes, name, mime_type)
}

/// Marcel's first extension for a MIME type.
fn extension(mime_type: &str) -> Option<&'static str> {
    match mime_type {
        "image/jpeg" => Some("jpeg"),
        "audio/mpeg" => Some("mp3"),
        "audio/wav" | "audio/x-wav" => Some("wav"),
        other => mime_guess::get_mime_extensions_str(other).and_then(|exts| exts.first().copied()),
    }
}

/// `Content.filename`: the last segment of a URI's path, or `"resource"` for an invalid URI.
pub(crate) fn filename(uri: &str) -> String {
    match reqwest::Url::parse(uri) {
        Ok(url) => url
            .path()
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string(),
        Err(_) => "resource".into(),
    }
}
