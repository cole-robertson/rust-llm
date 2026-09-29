//! Port of `lib/ruby_llm/tokens.rb`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Token counts a provider reported for one request. `None` means the provider did not report
/// that component, which RubyLLM keeps distinct from zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tokens {
    pub input: Option<i64>,
    pub output: Option<i64>,
    pub cache_read: Option<i64>,
    pub cache_write: Option<i64>,
    pub thinking: Option<i64>,
    pub server_tool_use: Option<Map<String, Value>>,
    pub reported_cost: Option<f64>,
}

impl Tokens {
    /// `Tokens.aggregate`: sums each reported component; a component stays `None` only if no
    /// entry reported it.
    pub fn aggregate<'a>(tokens: impl IntoIterator<Item = &'a Tokens>) -> Tokens {
        let tokens: Vec<&Tokens> = tokens.into_iter().collect();
        match tokens.len() {
            0 => return Tokens::default(),
            1 => return tokens[0].clone(),
            _ => {}
        }
        fn sum(values: impl Iterator<Item = Option<i64>>) -> Option<i64> {
            values.flatten().fold(None, |acc, v| Some(acc.unwrap_or(0) + v))
        }
        let reported_cost = tokens
            .iter()
            .filter_map(|t| t.reported_cost)
            .fold(None, |acc: Option<f64>, v| Some(acc.unwrap_or(0.0) + v));
        let mut server_tool_use: Option<Map<String, Value>> = None;
        for counters in tokens.iter().filter_map(|t| t.server_tool_use.as_ref()) {
            let total = server_tool_use.get_or_insert_with(Map::new);
            for (tool, count) in counters {
                let prev = total.get(tool).and_then(Value::as_i64).unwrap_or(0);
                total.insert(tool.clone(), Value::from(prev + count.as_i64().unwrap_or(0)));
            }
        }
        Tokens {
            input: sum(tokens.iter().map(|t| t.input)),
            output: sum(tokens.iter().map(|t| t.output)),
            cache_read: sum(tokens.iter().map(|t| t.cache_read)),
            cache_write: sum(tokens.iter().map(|t| t.cache_write)),
            thinking: sum(tokens.iter().map(|t| t.thinking)),
            server_tool_use,
            reported_cost,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.input.is_none()
            && self.output.is_none()
            && self.cache_read.is_none()
            && self.cache_write.is_none()
            && self.thinking.is_none()
            && self.server_tool_use.is_none()
    }

    /// Keeps the later reported value of each component, as `StreamAccumulator#count_tokens` does.
    pub(crate) fn merge_latest(&mut self, other: &Tokens) {
        if other.input.is_some() {
            self.input = other.input;
        }
        if other.output.is_some() {
            self.output = other.output;
        }
        if other.cache_read.is_some() {
            self.cache_read = other.cache_read;
        }
        if other.cache_write.is_some() {
            self.cache_write = other.cache_write;
        }
        if other.thinking.is_some() {
            self.thinking = other.thinking;
        }
        if other.server_tool_use.is_some() {
            self.server_tool_use = other.server_tool_use.clone();
        }
        if other.reported_cost.is_some() {
            self.reported_cost = other.reported_cost;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregating_keeps_unreported_components_as_none() {
        let a = Tokens { input: Some(10), output: Some(5), ..Default::default() };
        let b = Tokens { input: Some(7), cache_read: Some(3), ..Default::default() };
        let total = Tokens::aggregate([&a, &b]);
        assert_eq!(total.input, Some(17));
        assert_eq!(total.output, Some(5));
        assert_eq!(total.cache_read, Some(3));
        assert_eq!(total.thinking, None);
    }
}
