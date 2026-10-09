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
    /// `#cache_write_by_ttl`: `cache_write` split by cache lifetime (`"5m"`, `"1h"`), lifetimes
    /// with no writes left out; `None` when the provider reported no split.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_by_ttl: Option<Map<String, Value>>,
    /// `#server_tool_use`: how many times each provider-executed tool ran, keyed like
    /// `"web_search_requests"`. Tools that did not run are left out; `None` when none ran.
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
            values
                .flatten()
                .fold(None, |acc, v| Some(acc.unwrap_or(0) + v))
        }
        let reported_cost = tokens
            .iter()
            .filter_map(|t| t.reported_cost)
            .fold(None, |acc: Option<f64>, v| Some(acc.unwrap_or(0.0) + v));
        // `aggregate_counts`: sums each key across the attempts that reported any.
        fn counts<'a>(maps: impl Iterator<Item = &'a Map<String, Value>>) -> Option<Map<String, Value>> {
            let mut out: Option<Map<String, Value>> = None;
            for counters in maps {
                let total = out.get_or_insert_with(Map::new);
                for (key, count) in counters {
                    let prev = total.get(key).and_then(Value::as_i64).unwrap_or(0);
                    total.insert(key.clone(), Value::from(prev + count.as_i64().unwrap_or(0)));
                }
            }
            out
        }
        let server_tool_use = counts(tokens.iter().filter_map(|t| t.server_tool_use.as_ref()));
        let cache_write_by_ttl =
            counts(tokens.iter().filter_map(|t| t.cache_write_by_ttl.as_ref()));
        Tokens {
            input: sum(tokens.iter().map(|t| t.input)),
            output: sum(tokens.iter().map(|t| t.output)),
            cache_read: sum(tokens.iter().map(|t| t.cache_read)),
            cache_write: sum(tokens.iter().map(|t| t.cache_write)),
            thinking: sum(tokens.iter().map(|t| t.thinking)),
            cache_write_by_ttl,
            server_tool_use,
            reported_cost,
        }
    }

    /// `positive_counts`: String keys with Integer counts, keeping only counts above zero;
    /// `None` when nothing is left. `Tokens.new` applies it to `server_tool_use:` and
    /// `cache_write_by_ttl:`.
    pub fn positive_counts(counts: &Value) -> Option<Map<String, Value>> {
        let used: Map<String, Value> = counts
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(key, count)| {
                let n = count
                    .as_i64()
                    .or_else(|| count.as_f64().map(|f| f as i64))
                    .or_else(|| count.as_str().and_then(|s| s.trim().parse().ok()))
                    .unwrap_or(0);
                (n > 0).then(|| (key.clone(), Value::from(n)))
            })
            .collect();
        (!used.is_empty()).then_some(used)
    }

    /// `Tokens.new(server_tool_use:)`: sets the counts, normalized by [`Tokens::positive_counts`].
    pub fn with_server_tool_use(mut self, counts: &Value) -> Tokens {
        self.server_tool_use = Tokens::positive_counts(counts);
        self
    }

    /// `Tokens.new(cache_write_by_ttl:)`: sets the split, normalized by [`Tokens::positive_counts`].
    pub fn with_cache_write_by_ttl(mut self, counts: &Value) -> Tokens {
        self.cache_write_by_ttl = Tokens::positive_counts(counts);
        self
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
        if other.cache_write_by_ttl.is_some() {
            self.cache_write_by_ttl = other.cache_write_by_ttl.clone();
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
        let a = Tokens {
            input: Some(10),
            output: Some(5),
            ..Default::default()
        };
        let b = Tokens {
            input: Some(7),
            cache_read: Some(3),
            ..Default::default()
        };
        let total = Tokens::aggregate([&a, &b]);
        assert_eq!(total.input, Some(17));
        assert_eq!(total.output, Some(5));
        assert_eq!(total.cache_read, Some(3));
        assert_eq!(total.thinking, None);
    }
}
