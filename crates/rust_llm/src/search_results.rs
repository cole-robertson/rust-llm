//! Port of `lib/ruby_llm/search_results.rb`: documents a tool returns so the model can cite each
//! source. Protocols that support it (Anthropic) render them in their citation format; cited
//! passages come back on `Message::citations`.
//!
//! ```ruby
//! RubyLLM::SearchResults.new(title: 'Q4 Report', url: report_url, text: report_text)
//! ```
//!
//! ```ignore
//! Ok(SearchResults::new(vec![json!({ "title": "Q4 Report", "url": report_url, "text": report_text })])?.into())
//! ```

use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::tool::ToolResult;

/// `RubyLLM::SearchResults`.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    /// Each result reduced to its `title`, `url`, and `text` entries.
    pub results: Vec<Map<String, Value>>,
}

const KEY: &str = "search_results";

impl SearchResults {
    /// `SearchResults.new(*results)`. Fails with `Error::Argument` if there are no results or one
    /// is missing `title` or `text`.
    pub fn new(results: Vec<Value>) -> Result<SearchResults> {
        if results.is_empty() {
            return Err(Error::Argument("SearchResults requires at least one result".into()));
        }
        let results = results.iter().map(normalize).collect::<Result<Vec<_>>>()?;
        Ok(SearchResults { results })
    }

    /// `SearchResults.from_content`: recognizes a tool result that serialized search results.
    pub(crate) fn from_content(content: Option<&str>) -> Option<SearchResults> {
        let content = content?;
        if !content.trim_start().starts_with('{') {
            return None;
        }
        let parsed: Value = serde_json::from_str(content).ok()?;
        let entries = parsed.get(KEY)?.as_array()?;
        if entries.is_empty() || !entries.iter().all(Value::is_object) {
            return None;
        }
        SearchResults::new(entries.clone()).ok()
    }

    /// `to_h`.
    pub fn to_h(&self) -> Value {
        json!({ KEY: self.results })
    }
}

fn normalize(entry: &Value) -> Result<Map<String, Value>> {
    let has = |k: &str| entry.get(k).is_some_and(|v| !v.is_null());
    if !has("title") || !has("text") {
        return Err(Error::Argument("Search results require :title and :text".into()));
    }
    Ok(["title", "url", "text"].iter().filter_map(|k| entry.get(*k).map(|v| (k.to_string(), v.clone()))).collect())
}

/// `Tool#result_content`: search results go to the model as their JSON.
impl From<SearchResults> for ToolResult {
    fn from(results: SearchResults) -> Self {
        ToolResult::from(results.to_h().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_tool_result_content() {
        let results = SearchResults::new(vec![json!({ "title": "A", "text": "x", "extra": 1 })]).unwrap();
        let content = ToolResult::from(results.clone()).content;
        assert_eq!(content, r#"{"search_results":[{"title":"A","text":"x"}]}"#);
        assert_eq!(SearchResults::from_content(Some(&content)), Some(results));
        assert_eq!(SearchResults::from_content(Some("plain")), None);
    }

    #[test]
    fn requires_title_and_text() {
        assert!(SearchResults::new(vec![]).is_err());
        assert!(SearchResults::new(vec![json!({ "title": "A" })]).is_err());
    }
}
