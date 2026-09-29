//! Port of `lib/ruby_llm/mcp/resource_template.rb`: a family of resources named by an RFC 6570
//! URI template.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value, json};

use super::Mcp;
use crate::error::Result;

static EXPRESSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{([+#./;?&]?)([^{}]+)\}").expect("valid regex"));

/// `RubyLLM::MCP::ResourceTemplate`.
#[derive(Clone)]
pub struct ResourceTemplate {
    /// The URI template, such as `"file:///{path}"`.
    pub uri: String,
    pub name: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    mcp: Mcp,
}

impl std::fmt::Debug for ResourceTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceTemplate")
            .field("uri", &self.uri)
            .field("name", &self.name)
            .finish()
    }
}

impl ResourceTemplate {
    pub(crate) fn new(mcp: Mcp, data: &Value) -> ResourceTemplate {
        let s = |key: &str| data.get(key).and_then(Value::as_str).map(str::to_string);
        ResourceTemplate {
            uri: s("uriTemplate").unwrap_or_default(),
            name: s("name"),
            title: s("title"),
            description: s("description"),
            mime_type: s("mimeType"),
            mcp,
        }
    }

    /// `suggest(**variables)`: asks the server to complete the first variable's partial value,
    /// with the rest as context.
    pub async fn suggest(&self, variables: &[(&str, &str)]) -> Result<Vec<String>> {
        self.mcp
            .suggest(
                json!({ "type": "ref/resource", "uri": self.uri }),
                variables,
            )
            .await
    }

    /// `ResourceTemplate.expand`: fills in `template` with `variables` (a JSON object whose values
    /// are strings, numbers, or arrays of them).
    pub fn expand(template: &str, variables: &Value) -> String {
        let empty = Map::new();
        let variables = variables.as_object().unwrap_or(&empty);
        EXPRESSION
            .replace_all(template, |caps: &regex::Captures| {
                let specs: Vec<&str> = caps[2].split(',').collect();
                expand_expression(&caps[1], &specs, variables)
            })
            .into_owned()
    }
}

/// `OPERATORS`: each operator's separator and whether reserved characters pass unencoded.
fn operator(op: &str) -> (&'static str, bool) {
    match op {
        "+" | "#" => (",", true),
        "/" => ("/", false),
        "." => (".", false),
        ";" => (";", false),
        "?" | "&" => ("&", false),
        _ => (",", false),
    }
}

fn expand_expression(op: &str, specs: &[&str], variables: &Map<String, Value>) -> String {
    let (separator, reserved) = operator(op);
    let values: Vec<String> = specs
        .iter()
        .filter_map(|spec| expand_variable(spec, op, separator, reserved, variables))
        .collect();
    if values.is_empty() {
        return String::new();
    }
    let prefix = if op.is_empty() || op == "+" { "" } else { op };
    format!("{prefix}{}", values.join(separator))
}

/// Expands one variable with its prefix (`:3`) and explode (`*`) modifiers.
fn expand_variable(
    spec: &str,
    op: &str,
    separator: &str,
    reserved: bool,
    variables: &Map<String, Value>,
) -> Option<String> {
    let explode = spec.ends_with('*');
    let spec = spec.trim_end_matches('*');
    let (name, length) = match spec.split_once(':') {
        Some((name, length)) => (name, length.parse::<usize>().ok()),
        None => (spec, None),
    };
    let value = variables.get(name)?;
    let items: Vec<&Value> = match value {
        Value::Array(items) => items.iter().collect(),
        Value::Null => Vec::new(),
        other => vec![other],
    };
    let parts: Vec<String> = items
        .into_iter()
        .map(|item| {
            let text = match item {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let text: String = match length {
                Some(n) => text.chars().take(n).collect(),
                None => text,
            };
            encode(&text, reserved)
        })
        .collect();
    let named = matches!(op, "?" | "&" | ";");
    Some(if explode {
        parts
            .iter()
            .map(|part| {
                if named {
                    format!("{name}={part}")
                } else {
                    part.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(separator)
    } else if named {
        format!("{name}={}", parts.join(","))
    } else {
        parts.join(",")
    })
}

fn encode(value: &str, reserved: bool) -> String {
    let allowed = |c: char| {
        c.is_ascii_alphanumeric()
            || "-._~".contains(c)
            || (reserved && ":/?#[]@!$&'()*+,;=%".contains(c))
    };
    let mut out = String::new();
    for c in value.chars() {
        if allowed(c) {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            for byte in c.encode_utf8(&mut buf).bytes() {
                out.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // spec/ruby_llm/mcp/resource_template_spec.rb
    #[test]
    fn expands_like_rubyllm() {
        let cases = [
            (
                "file:///{path}",
                json!({ "path": "docs/README.md" }),
                "file:///docs%2FREADME.md",
            ),
            (
                "file:///{+path}",
                json!({ "path": "docs/README.md" }),
                "file:///docs/README.md",
            ),
            (
                "repo://{owner}/{repo}",
                json!({ "owner": "crmne", "repo": "ruby llm" }),
                "repo://crmne/ruby%20llm",
            ),
            (
                "search{?q,limit}",
                json!({ "q": "mcp", "limit": 5 }),
                "search?q=mcp&limit=5",
            ),
            ("items{/id}", json!({ "id": 42 }), "items/42"),
            ("file:///{path}", json!({}), "file:///"),
            ("{{name}}", json!({ "name": "x" }), "{x}"),
            (
                "file:///{path*}",
                json!({ "path": "README.md" }),
                "file:///README.md",
            ),
            ("items{/path*}", json!({ "path": ["a", "b"] }), "items/a/b"),
            ("items{/path}", json!({ "path": ["a", "b"] }), "items/a,b"),
            (
                "search{?tags*}",
                json!({ "tags": ["x", "y"] }),
                "search?tags=x&tags=y",
            ),
            ("id/{id:3}", json!({ "id": "abcdef" }), "id/abc"),
        ];
        for (template, variables, expanded) in cases {
            assert_eq!(
                ResourceTemplate::expand(template, &variables),
                expanded,
                "{template} with {variables}"
            );
        }
    }
}
