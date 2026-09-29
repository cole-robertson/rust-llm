//! Port of `lib/ruby_llm/mcp/param_headers.rb`: mirrors tool arguments marked with
//! `x-mcp-header` into `Mcp-Param-*` HTTP headers, as 2026-07-28 requires, so intermediaries can
//! route on them. Tools that declare invalid headers are left out entirely.

use serde_json::{Map, Value};

const TYPES: &[&str] = &["string", "integer", "boolean"];

fn is_token(header: &str) -> bool {
    !header.is_empty()
        && header
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "!#$%&'*+-.^_`|~".contains(c))
}

/// `ParamHeaders.valid?`.
pub fn is_valid(definition: &Value) -> bool {
    let declared = declarations(definition);
    let valid = declared.iter().all(|(_, header, schema)| {
        header.as_str().is_some_and(is_token)
            && schema
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| TYPES.contains(&t))
    });
    let mut names: Vec<String> = declared
        .iter()
        .map(|(_, header, _)| header_text(header).to_lowercase())
        .collect();
    let count = names.len();
    names.sort();
    names.dedup();
    if valid && names.len() == count {
        return true;
    }
    let name = definition.get("name").and_then(Value::as_str).unwrap_or("");
    tracing::warn!("Ignoring MCP tool {name}: it declares invalid x-mcp-header values");
    false
}

/// `ParamHeaders.for`: the headers for the arguments that have values.
pub fn headers_for(definition: &Value, arguments: &Map<String, Value>) -> Vec<(String, String)> {
    declarations(definition)
        .into_iter()
        .filter_map(|(property, header, _)| {
            let value = arguments.get(&property).filter(|v| !v.is_null())?;
            let text = match value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Some((header_text(&header), text))
        })
        .collect()
}

fn header_text(header: &Value) -> String {
    match header {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn declarations(definition: &Value) -> Vec<(String, Value, Value)> {
    let Some(properties) = definition
        .pointer("/inputSchema/properties")
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    properties
        .iter()
        .filter_map(|(property, schema)| {
            let header = schema.as_object()?.get("x-mcp-header")?;
            Some((property.clone(), header.clone(), schema.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tool(properties: Value) -> Value {
        json!({ "name": "query", "inputSchema": { "type": "object", "properties": properties } })
    }

    #[test]
    fn mirrors_marked_arguments_that_have_values() {
        let definition = tool(json!({
            "region": { "type": "string", "x-mcp-header": "Region" },
            "verbose": { "type": "boolean", "x-mcp-header": "Verbose" },
            "limit": { "type": "integer", "x-mcp-header": "Limit" },
            "query": { "type": "string" }
        }));
        let arguments = json!({ "region": "us-west1", "verbose": false, "query": "SELECT 1" });
        assert_eq!(
            headers_for(&definition, arguments.as_object().unwrap()),
            vec![
                ("Region".to_string(), "us-west1".to_string()),
                ("Verbose".to_string(), "false".to_string())
            ]
        );
    }

    #[test]
    fn rejects_tools_with_invalid_declarations() {
        assert!(is_valid(&tool(
            json!({ "a": { "type": "string", "x-mcp-header": "Region" } })
        )));
        assert!(!is_valid(&tool(
            json!({ "a": { "type": "number", "x-mcp-header": "Price" } })
        )));
        assert!(!is_valid(&tool(
            json!({ "a": { "type": "string", "x-mcp-header": "Bad Name" } })
        )));
        assert!(!is_valid(&tool(json!({
            "a": { "type": "string", "x-mcp-header": "Region" },
            "b": { "type": "string", "x-mcp-header": "region" }
        }))));
    }
}
