//! Port of `lib/ruby_llm/error.rb` and the status mapping in
//! `lib/ruby_llm/transport/error_middleware.rb`.

use std::sync::LazyLock;

use regex::Regex;

/// The HTTP exchange an API error came from, kept like RubyLLM's `error.response`.
#[derive(Debug, Clone, Default)]
pub struct ErrorResponse {
    pub status: u16,
    pub body: String,
}

/// Every failure RubyLLM raises, one variant per Ruby error class.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Api(String, Option<ErrorResponse>),
    #[error("{0}")]
    BadRequest(String, Option<ErrorResponse>),
    #[error("{0}")]
    Unauthorized(String, Option<ErrorResponse>),
    #[error("{0}")]
    PaymentRequired(String, Option<ErrorResponse>),
    #[error("{0}")]
    Forbidden(String, Option<ErrorResponse>),
    #[error("{0}")]
    RateLimit(String, Option<ErrorResponse>),
    #[error("{0}")]
    ContextLengthExceeded(String, Option<ErrorResponse>),
    #[error("{0}")]
    Server(String, Option<ErrorResponse>),
    #[error("{0}")]
    ServiceUnavailable(String, Option<ErrorResponse>),
    #[error("{0}")]
    Overloaded(String, Option<ErrorResponse>),
    #[error("{message}")]
    ToolCallParse { message: String, finish_reason: Option<String> },
    #[error("{0}")]
    UnsupportedAttachment(String),
    #[error("{0}")]
    Configuration(String),
    #[error("{0}")]
    ModelNotFound(String),
    /// `ModelRegistryError`: the registry could not be fetched, read, or saved.
    #[error("{0}")]
    ModelRegistry(String),
    /// `PromptNotFoundError`: `render_prompt` found no template file.
    #[error("{0}")]
    PromptNotFound(String),
    /// A prompt template failed to parse or render (Ruby raises the ERB error itself).
    #[error("{0}")]
    Prompt(String),
    #[error("{0}")]
    InvalidToolChoice(String),
    #[error("{0}")]
    PendingToolCalls(String),
    #[error("Chat generation cancelled")]
    Cancelled,
    #[error("{0}")]
    Argument(String),
    #[error("timeout: {0}")]
    Timeout(String),
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    /// A tool's own failure, re-raised to the caller like an exception escaping `execute`.
    #[error("{0}")]
    Tool(String),
    /// `UnsupportedServerToolError`: a provider tool alias the protocol does not define.
    #[error("{0}")]
    UnsupportedServerTool(String),
    /// `MCP::Error`: an MCP server answered with a JSON-RPC error, a bad status, or not at all.
    #[error(transparent)]
    Mcp(Box<crate::mcp::McpError>),
    /// `MCP::InputRequiredError`: a server needs input from the user that no callback gave.
    #[error(transparent)]
    McpInputRequired(Box<crate::mcp::InputRequiredError>),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Which Ruby class an error corresponds to, for `with_fallbacks(on:)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    Api,
    BadRequest,
    Unauthorized,
    PaymentRequired,
    Forbidden,
    RateLimit,
    ContextLengthExceeded,
    Server,
    ServiceUnavailable,
    Overloaded,
    Timeout,
    ConnectionFailed,
    Other,
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        match self {
            Error::Api(..) => ErrorKind::Api,
            Error::BadRequest(..) => ErrorKind::BadRequest,
            Error::Unauthorized(..) => ErrorKind::Unauthorized,
            Error::PaymentRequired(..) => ErrorKind::PaymentRequired,
            Error::Forbidden(..) => ErrorKind::Forbidden,
            Error::RateLimit(..) => ErrorKind::RateLimit,
            Error::ContextLengthExceeded(..) => ErrorKind::ContextLengthExceeded,
            Error::Server(..) => ErrorKind::Server,
            Error::ServiceUnavailable(..) => ErrorKind::ServiceUnavailable,
            Error::Overloaded(..) => ErrorKind::Overloaded,
            Error::Timeout(_) => ErrorKind::Timeout,
            Error::ConnectionFailed(_) => ErrorKind::ConnectionFailed,
            _ => ErrorKind::Other,
        }
    }

    /// The same error class with a provider-specific message (`Provider#parse_error` overrides).
    pub(crate) fn with_message(self, message: String) -> Error {
        match self {
            Error::Api(_, r) => Error::Api(message, r),
            Error::BadRequest(_, r) => Error::BadRequest(message, r),
            Error::Unauthorized(_, r) => Error::Unauthorized(message, r),
            Error::PaymentRequired(_, r) => Error::PaymentRequired(message, r),
            Error::Forbidden(_, r) => Error::Forbidden(message, r),
            Error::RateLimit(_, r) => Error::RateLimit(message, r),
            Error::ContextLengthExceeded(_, r) => Error::ContextLengthExceeded(message, r),
            Error::Server(_, r) => Error::Server(message, r),
            Error::ServiceUnavailable(_, r) => Error::ServiceUnavailable(message, r),
            Error::Overloaded(_, r) => Error::Overloaded(message, r),
            other => other,
        }
    }

    pub fn response(&self) -> Option<&ErrorResponse> {
        match self {
            Error::Api(_, r)
            | Error::BadRequest(_, r)
            | Error::Unauthorized(_, r)
            | Error::PaymentRequired(_, r)
            | Error::Forbidden(_, r)
            | Error::RateLimit(_, r)
            | Error::ContextLengthExceeded(_, r)
            | Error::Server(_, r)
            | Error::ServiceUnavailable(_, r)
            | Error::Overloaded(_, r) => r.as_ref(),
            _ => None,
        }
    }

    /// `Transport::Connection#retry_exceptions`.
    pub(crate) fn retryable(&self) -> bool {
        matches!(
            self.kind(),
            ErrorKind::RateLimit
                | ErrorKind::Server
                | ErrorKind::ServiceUnavailable
                | ErrorKind::Overloaded
                | ErrorKind::Timeout
                | ErrorKind::ConnectionFailed
        )
    }

    pub(crate) fn tool_call_parse(finish_reason: Option<&str>) -> Self {
        let mut message = "Provider returned malformed tool call arguments".to_string();
        if let Some(reason) = finish_reason {
            message = format!("{message} (finish_reason: {reason})");
        }
        Error::ToolCallParse { message, finish_reason: finish_reason.map(str::to_string) }
    }
}

/// `Fallback::DEFAULT_ERRORS`.
pub const DEFAULT_FALLBACK_ERRORS: &[ErrorKind] = &[
    ErrorKind::RateLimit,
    ErrorKind::Server,
    ErrorKind::ServiceUnavailable,
    ErrorKind::Overloaded,
    ErrorKind::Timeout,
    ErrorKind::ConnectionFailed,
];

fn patterns(list: &[&str]) -> Vec<Regex> {
    list.iter().map(|p| Regex::new(&format!("(?i){p}")).unwrap()).collect() // patterns are constants in this file
}

static CONTEXT_LENGTH_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    patterns(&[
        "context length",
        "context window",
        "exceeds?.*context size",
        "maximum context",
        "request too large",
        "too many tokens",
        "token count exceeds",
        r"input[_\s-]?token",
        "input or output tokens? must be reduced",
        "reduce the length of messages",
        "prompt is too long",
        "context limit",
    ])
});
static RATE_LIMIT_PATTERNS: LazyLock<Vec<Regex>> =
    LazyLock::new(|| patterns(&["rate limit", "per minute", "per hour", "per day"]));
static OVERLOAD_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| patterns(&["currently overloaded"]));

fn matches_any(list: &[Regex], message: &str) -> bool {
    list.iter().any(|re| re.is_match(message))
}

/// `Provider#parse_error`: pull a human message out of an error body.
pub(crate) fn parse_error_message(body: &str) -> Option<String> {
    if body.trim().is_empty() {
        return None;
    }
    let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
        return Some(body.to_string());
    };
    fn part_message(part: &serde_json::Value) -> Option<String> {
        // `part.to_s`: `nil` is empty (and dropped), a string is itself, unquoted.
        let Some(obj) = part.as_object() else {
            return Some(match part {
                serde_json::Value::Null => String::new(),
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            });
        };
        if let Some(s) = obj.get("error").and_then(|e| e.as_str()) {
            return Some(s.to_string());
        }
        let nested = obj.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str());
        nested
            .or_else(|| obj.get("message").and_then(|m| m.as_str()))
            .or_else(|| obj.get("detail").and_then(|m| m.as_str()))
            .map(str::to_string)
    }
    match &json {
        serde_json::Value::Array(parts) => {
            let messages: Vec<String> =
                parts.iter().filter_map(part_message).filter(|m| !m.is_empty()).collect();
            (!messages.is_empty()).then(|| messages.join(". "))
        }
        serde_json::Value::Object(_) => part_message(&json),
        // `else body`: a body that parses to a bare JSON string is that string, unquoted.
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// `ErrorMiddleware.parse_error`: map a failed HTTP status to the error class.
pub(crate) fn error_for_status(status: u16, body: &str) -> Error {
    error_for_status_message(status, body, parse_error_message(body))
}

/// `ErrorMiddleware.parse_error` with the message a provider's `parse_error` already read.
pub(crate) fn error_for_status_message(status: u16, body: &str, message: Option<String>) -> Error {
    let response = Some(ErrorResponse { status, body: body.to_string() });
    let text = message.clone().unwrap_or_default();
    let msg = |default: &str| message.clone().unwrap_or_else(|| default.to_string());
    match status {
        400 => {
            if matches_any(&CONTEXT_LENGTH_PATTERNS, &text) {
                Error::ContextLengthExceeded(msg("Context length exceeded"), response)
            } else if matches_any(&OVERLOAD_PATTERNS, &text) {
                Error::Overloaded(msg("Service overloaded - please try again later"), response)
            } else {
                Error::BadRequest(msg("Invalid request - please check your input"), response)
            }
        }
        401 => Error::Unauthorized(msg("Invalid API key - check your credentials"), response),
        402 => Error::PaymentRequired(msg("Payment required - please top up your account"), response),
        403 => Error::Forbidden(
            msg("Forbidden - you do not have permission to access this resource"),
            response,
        ),
        429 => {
            if !matches_any(&RATE_LIMIT_PATTERNS, &text) && matches_any(&CONTEXT_LENGTH_PATTERNS, &text) {
                Error::ContextLengthExceeded(msg("Context length exceeded"), response)
            } else {
                Error::RateLimit(msg("Rate limit exceeded - please wait a moment"), response)
            }
        }
        500 => Error::Server(msg("API server error - please try again"), response),
        502..=504 => {
            Error::ServiceUnavailable(msg("API server unavailable - please try again later"), response)
        }
        529 => Error::Overloaded(msg("Service overloaded - please try again later"), response),
        _ => Error::Api(message.unwrap_or_else(|| body.to_string()), response),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_400_about_context_becomes_context_length_exceeded() {
        let err = error_for_status(400, r#"{"error":{"message":"prompt is too long: 250000 tokens"}}"#);
        assert_eq!(err.kind(), ErrorKind::ContextLengthExceeded);
        assert_eq!(err.to_string(), "prompt is too long: 250000 tokens");
    }

    #[test]
    fn a_429_that_mentions_rate_limits_stays_a_rate_limit_even_if_it_mentions_tokens() {
        let err = error_for_status(429, r#"{"error":{"message":"Rate limit reached for input tokens per minute"}}"#);
        assert_eq!(err.kind(), ErrorKind::RateLimit);
    }

    #[test]
    fn a_529_is_overloaded_and_retryable() {
        let err = error_for_status(529, r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#);
        assert_eq!(err.kind(), ErrorKind::Overloaded);
        assert!(err.retryable());
    }

    #[test]
    fn an_empty_401_uses_the_default_message() {
        assert_eq!(error_for_status(401, "").to_string(), "Invalid API key - check your credentials");
    }
}
