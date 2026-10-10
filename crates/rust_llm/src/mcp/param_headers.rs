//! Port of `lib/ruby_llm/mcp/param_headers.rb`: mirrors tool arguments marked with
//! `x-mcp-header` into `Mcp-Param-*` HTTP headers, as 2026-07-28 requires, so intermediaries can
//! route on them. Tools that declare invalid headers are left out entirely.

use serde_json::{Map, Value};

const TYPES: &[&str] = &["string", "integer", "boolean"];
const SCHEMA_MAPS: &[&str] = &[
    "$defs",
    "definitions",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
];
const SCHEMA_VALUES: &[&str] = &[
    "items",
    "prefixItems",
    "additionalItems",
    "contains",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "propertyNames",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
];

/// A declared header: the property path it mirrors (`None` when the schema is reached through a
/// keyword other than `properties`, so no argument path leads to it), the header, and its schema.
type Declaration = (Option<Vec<String>>, Value, Value);

fn is_token(header: &str) -> bool {
    !header.is_empty()
        && header
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "!#$%&'*+-.^_`|~".contains(c))
}

/// `ParamHeaders.valid?`.
pub fn is_valid(definition: &Value) -> bool {
    let declared = declarations(definition);
    let valid = declared
        .iter()
        .all(|(path, header, schema)| is_valid_declaration(path.as_deref(), header, schema));
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

/// `ParamHeaders.for`: the headers for the arguments that have values, read from each
/// declaration's exact property path.
pub fn headers_for(definition: &Value, arguments: &Map<String, Value>) -> Vec<(String, String)> {
    declarations(definition)
        .into_iter()
        .filter_map(|(path, header, _)| {
            let value = argument_at(arguments, path.as_deref()?)?;
            let text = match value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Some((header_text(&header), text))
        })
        .collect()
}

/// `ParamHeaders.valid_declaration?`.
fn is_valid_declaration(path: Option<&[String]>, header: &Value, schema: &Value) -> bool {
    path.is_some_and(|p| !p.is_empty())
        && header.as_str().is_some_and(is_token)
        && schema
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| TYPES.contains(&t))
}

/// `header.to_s`: `nil` reads as an empty string.
fn header_text(header: &Value) -> String {
    match header {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn declarations(definition: &Value) -> Vec<Declaration> {
    definition
        .get("inputSchema")
        .map(|schema| schema_declarations(schema, Some(Vec::new())))
        .unwrap_or_default()
}

/// `ParamHeaders.schema_declarations`.
fn schema_declarations(schema: &Value, path: Option<Vec<String>>) -> Vec<Declaration> {
    let Some(object) = schema.as_object() else {
        return Vec::new();
    };
    let mut declared = Vec::new();
    if let Some(header) = object.get("x-mcp-header") {
        declared.push((path.clone(), header.clone(), schema.clone()));
    }
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        for (property, child) in properties {
            let child_path = path.as_ref().map(|p| {
                let mut p = p.clone();
                p.push(property.clone());
                p
            });
            declared.extend(schema_declarations(child, child_path));
        }
    }
    for child in unreachable_schemas(object) {
        declared.extend(schema_declarations(child, None));
    }
    declared
}

/// `ParamHeaders.unreachable_schemas`: subschemas no argument path leads to.
fn unreachable_schemas(schema: &Map<String, Value>) -> Vec<&Value> {
    schema
        .iter()
        .flat_map(|(keyword, children)| -> Vec<&Value> {
            if SCHEMA_MAPS.contains(&keyword.as_str()) {
                children
                    .as_object()
                    .map(|map| map.values().collect())
                    .unwrap_or_default()
            } else if SCHEMA_VALUES.contains(&keyword.as_str()) {
                match children {
                    Value::Array(items) => items.iter().collect(),
                    other => vec![other],
                }
            } else {
                Vec::new()
            }
        })
        .collect()
}

/// `ParamHeaders.argument_at`: the non-null value at `path`, if every step is an object.
fn argument_at<'a>(arguments: &'a Map<String, Value>, path: &[String]) -> Option<&'a Value> {
    let (first, rest) = path.split_first()?;
    let mut value = arguments.get(first)?;
    for property in rest {
        value = value.as_object()?.get(property)?;
    }
    (!value.is_null()).then_some(value)
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

    fn nested_tool(properties: Value) -> Value {
        tool(json!({ "routing": { "type": "object", "properties": properties } }))
    }

    fn region() -> Value {
        json!({ "type": "string", "x-mcp-header": "Region" })
    }

    fn with(mut schema: Value, key: &str, value: Value) -> Value {
        schema[key] = value;
        schema
    }

    fn headers(definition: &Value, arguments: Value) -> Vec<(String, String)> {
        headers_for(definition, arguments.as_object().unwrap())
    }

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // spec: mcp/param_headers_spec.rb:35 mirrors values from their exact property paths
    #[test]
    fn mirrors_values_from_their_exact_property_paths() {
        let definition = nested_tool(json!({ "region": region() }));
        assert_eq!(
            headers(
                &definition,
                json!({ "region": "wrong", "routing": { "region": "us-west1" } })
            ),
            pairs(&[("Region", "us-west1")])
        );
    }

    // spec: mcp/param_headers_spec.rb:42 mirrors several levels of properties alongside root arguments
    #[test]
    fn mirrors_several_levels_alongside_root_arguments() {
        let definition = tool(json!({
            "region": region(),
            "routing": { "properties": {
                "account": { "properties": {
                    "id": { "type": "string", "x-mcp-header": "Account" }
                } }
            } }
        }));
        assert_eq!(
            headers(
                &definition,
                json!({ "region": "us-west1", "routing": { "account": { "id": "account-1" } } })
            ),
            pairs(&[("Region", "us-west1"), ("Account", "account-1")])
        );
    }

    // spec: mcp/param_headers_spec.rb:56 reads mixed String and Symbol argument keys
    // (JSON arguments have only string keys; nested lookups use the same keys at every level)
    #[test]
    fn reads_nested_argument_keys() {
        let definition = nested_tool(json!({ "region": region() }));
        assert_eq!(
            headers(&definition, json!({ "routing": { "region": "us-west1" } })),
            pairs(&[("Region", "us-west1")])
        );
    }

    // spec: mcp/param_headers_spec.rb:63 keeps false and zero nested values
    #[test]
    fn keeps_false_and_zero_nested_values() {
        let definition = nested_tool(json!({
            "verbose": { "type": "boolean", "x-mcp-header": "Verbose" },
            "limit": { "type": "integer", "x-mcp-header": "Limit" }
        }));
        assert_eq!(
            headers(
                &definition,
                json!({ "routing": { "verbose": false, "limit": 0 } })
            ),
            pairs(&[("Verbose", "false"), ("Limit", "0")])
        );
    }

    // spec: mcp/param_headers_spec.rb:74 omits a nested header when its path has no value in #{arguments.inspect}
    #[test]
    fn omits_a_nested_header_when_its_path_has_no_value() {
        let definition = nested_tool(json!({ "region": region() }));
        for arguments in [
            json!({}),
            json!({ "routing": null }),
            json!({ "routing": {} }),
            json!({ "routing": { "region": null } }),
        ] {
            assert_eq!(headers(&definition, arguments), vec![]);
        }
    }

    // spec: mcp/param_headers_spec.rb:79 keeps root headers when a nested object is absent
    #[test]
    fn keeps_root_headers_when_a_nested_object_is_absent() {
        let mut definition = nested_tool(json!({ "region": region() }));
        definition["inputSchema"]["properties"]["limit"] =
            json!({ "type": "integer", "x-mcp-header": "Limit" });
        assert_eq!(
            headers(&definition, json!({ "limit": 0 })),
            pairs(&[("Limit", "0")])
        );
    }

    // spec: mcp/param_headers_spec.rb:86 does not substitute defaults for missing arguments
    #[test]
    fn does_not_substitute_defaults_for_missing_arguments() {
        let definition =
            nested_tool(json!({ "region": with(region(), "default", json!("us-west1")) }));
        assert_eq!(headers(&definition, json!({ "routing": {} })), vec![]);
    }

    // spec: mcp/param_headers_spec.rb:92 preserves an empty nested string value
    #[test]
    fn preserves_an_empty_nested_string_value() {
        let definition = nested_tool(json!({ "region": region() }));
        assert_eq!(
            headers(&definition, json!({ "routing": { "region": "" } })),
            pairs(&[("Region", "")])
        );
    }

    // spec: mcp/param_headers_spec.rb:97 preserves safe integer boundary values
    #[test]
    fn preserves_safe_integer_boundary_values() {
        let definition =
            nested_tool(json!({ "limit": { "type": "integer", "x-mcp-header": "Limit" } }));
        let maximum: i64 = (1 << 53) - 1;
        for value in [-maximum, maximum] {
            assert_eq!(
                headers(&definition, json!({ "routing": { "limit": value } })),
                vec![("Limit".to_string(), value.to_string())]
            );
        }
    }

    // spec: mcp/param_headers_spec.rb:106 accepts valid nested declarations without changing the schema
    #[test]
    fn accepts_valid_nested_declarations_without_changing_the_schema() {
        let definition = nested_tool(json!({ "region": region() }));
        let original = definition.to_string();
        assert!(is_valid(&definition));
        headers(&definition, json!({ "routing": { "region": "us-west1" } }));
        assert_eq!(definition.to_string(), original);
    }

    // spec: mcp/param_headers_spec.rb:116 rejects a nested declaration with header #{header.inspect}
    #[test]
    fn rejects_a_nested_declaration_with_a_bad_header() {
        for header in [
            json!(""),
            Value::Null,
            json!(42),
            json!("Bad Name"),
            json!("Region\r\nInjected"),
        ] {
            let definition =
                nested_tool(json!({ "region": with(region(), "x-mcp-header", header.clone()) }));
            assert!(!is_valid(&definition), "{header}");
        }
    }

    // spec: mcp/param_headers_spec.rb:124 rejects a nested declaration of type #{type}
    #[test]
    fn rejects_a_nested_declaration_of_an_unmirrored_type() {
        for kind in ["number", "object", "array"] {
            let definition = nested_tool(json!({ "region": with(region(), "type", json!(kind)) }));
            assert!(!is_valid(&definition), "{kind}");
        }
    }

    // spec: mcp/param_headers_spec.rb:129 rejects case-insensitive header duplicates between root and nested properties
    #[test]
    fn rejects_duplicates_between_root_and_nested_properties() {
        let mut definition =
            nested_tool(json!({ "region": with(region(), "x-mcp-header", json!("region")) }));
        definition["inputSchema"]["properties"]["region"] = region();
        assert!(!is_valid(&definition));
    }

    // spec: mcp/param_headers_spec.rb:136 rejects case-insensitive header duplicates in separate nested objects
    #[test]
    fn rejects_duplicates_in_separate_nested_objects() {
        let mut definition = nested_tool(json!({ "region": region() }));
        definition["inputSchema"]["properties"]["other"] =
            json!({ "properties": { "region": with(region(), "x-mcp-header", json!("REGION")) } });
        assert!(!is_valid(&definition));
    }

    // spec: mcp/param_headers_spec.rb:147 rejects an annotation on the schema root
    #[test]
    fn rejects_an_annotation_on_the_schema_root() {
        let mut definition = tool(json!({}));
        definition["inputSchema"]["x-mcp-header"] = json!("Root");
        assert!(!is_valid(&definition));
    }

    // spec: mcp/param_headers_spec.rb:156 rejects an annotation under #{keyword}
    #[test]
    fn rejects_an_annotation_under_a_schema_keyword() {
        for keyword in [
            "items",
            "additionalItems",
            "contains",
            "additionalProperties",
            "unevaluatedItems",
            "unevaluatedProperties",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "contentSchema",
        ] {
            assert!(
                !is_valid(&tool(json!({ "routing": { keyword: region() } }))),
                "{keyword}"
            );
        }
    }

    // spec: mcp/param_headers_spec.rb:162 rejects an annotation in the #{keyword} schema array
    #[test]
    fn rejects_an_annotation_in_a_schema_array() {
        for keyword in ["allOf", "anyOf", "oneOf", "prefixItems"] {
            assert!(
                !is_valid(&tool(json!({ "routing": { keyword: [region()] } }))),
                "{keyword}"
            );
        }
    }

    // spec: mcp/param_headers_spec.rb:168 rejects an annotation under the #{keyword} schema map
    #[test]
    fn rejects_an_annotation_under_a_schema_map() {
        for keyword in [
            "$defs",
            "definitions",
            "patternProperties",
            "dependentSchemas",
            "dependencies",
        ] {
            let mut definition = tool(json!({}));
            definition["inputSchema"][keyword] = json!({ "entry": region() });
            assert!(!is_valid(&definition), "{keyword}");
        }
    }

    // spec: mcp/param_headers_spec.rb:176 rejects properties reached through an array schema
    #[test]
    fn rejects_properties_reached_through_an_array_schema() {
        let item = json!({ "type": "object", "properties": { "region": region() } });
        let definition = tool(json!({ "routes": { "type": "array", "items": item } }));
        assert!(!is_valid(&definition));
    }

    // spec: mcp/param_headers_spec.rb:183 rejects properties reached through a composition schema
    #[test]
    fn rejects_properties_reached_through_a_composition_schema() {
        let branch = json!({ "type": "object", "properties": { "region": region() } });
        let definition = tool(json!({ "routing": { "anyOf": [branch] } }));
        assert!(!is_valid(&definition));
    }

    // spec: mcp/param_headers_spec.rb:190 rejects an annotation in a referenced definition
    #[test]
    fn rejects_an_annotation_in_a_referenced_definition() {
        let mut definition = tool(json!({ "routing": { "$ref": "#/$defs/routing" } }));
        definition["inputSchema"]["$defs"] = json!({
            "routing": { "type": "object", "properties": { "region": region() } }
        });
        assert!(!is_valid(&definition));
    }

    // spec: mcp/param_headers_spec.rb:199 accepts schemas without annotations under composition and array keywords
    #[test]
    fn accepts_schemas_without_annotations_under_composition_and_array_keywords() {
        let item = json!({ "anyOf": [{ "type": "string" }, { "type": "null" }] });
        let definition = tool(json!({ "routes": { "type": "array", "items": item } }));
        assert!(is_valid(&definition));
        assert_eq!(
            headers(&definition, json!({ "routes": ["us-west1"] })),
            vec![]
        );
    }

    // spec: mcp/param_headers_spec.rb:208 does not mistake #{keyword} data for a schema declaration
    #[test]
    fn does_not_mistake_data_keywords_for_declarations() {
        for keyword in ["default", "const", "enum", "examples"] {
            let data = json!({ "x-mcp-header": "Bad Name" });
            let value = if matches!(keyword, "enum" | "examples") {
                json!([data])
            } else {
                data
            };
            let definition = tool(json!({ "routing": { keyword: value } }));
            assert!(is_valid(&definition), "{keyword}");
            assert_eq!(headers(&definition, json!({})), vec![]);
        }
    }

    // spec: mcp/param_headers_spec.rb:218 does not mistake a property named x-mcp-header for an annotation
    #[test]
    fn does_not_mistake_a_property_named_x_mcp_header_for_an_annotation() {
        let definition = tool(json!({ "x-mcp-header": { "type": "string" } }));
        assert!(is_valid(&definition));
        assert_eq!(
            headers(&definition, json!({ "x-mcp-header": "Bad Name" })),
            vec![]
        );
    }

    // spec: mcp/param_headers_spec.rb:225 accepts boolean schemas without declarations
    #[test]
    fn accepts_boolean_schemas_without_declarations() {
        let definition = tool(json!({ "enabled": true, "disabled": false }));
        assert!(is_valid(&definition));
        assert_eq!(headers(&definition, json!({})), vec![]);
    }
}
