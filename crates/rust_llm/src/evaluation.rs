//! Port of `lib/ruby_llm/evaluation.rb` and `lib/ruby_llm/evaluation/*` (in progress; lane EO).

use crate::message::UsageEntry;

/// `Runner#instrument` for `usage.ruby_llm`: hands a finished attempt to every evaluation
/// runner in scope.
pub(crate) fn capture_usage(_entry: &UsageEntry) {}
