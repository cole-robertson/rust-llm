//! Port of `lib/ruby_llm/transport/connection.rb`, `error_middleware.rb`, and the SSE handling in
//! `protocol/streaming.rb`.
//!
//! Retry rules match RubyLLM's Faraday setup: up to `max_retries` retries on rate limits, 5xx,
//! overloads, timeouts, and connection failures, with exponential backoff and jitter, honoring
//! `Retry-After`/`retry-after-ms`. A stream that has already delivered a chunk is never retried.

use std::time::Duration;

use futures::StreamExt;
use serde_json::Value;

use crate::config::Config;
use crate::error::{Error, Result, error_for_status};
use crate::message::RawResponse;
use crate::providers::Provider;

#[derive(Clone)]
pub struct Connection {
    client: reqwest::Client,
    provider: Provider,
    config: std::sync::Arc<Config>,
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection").field("provider", &self.provider.slug()).finish()
    }
}

/// One server-sent event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental SSE parser (`EventStreamParser::Parser`).
#[derive(Default)]
pub(crate) struct SseParser {
    buffer: String,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    pub(crate) fn feed(&mut self, bytes: &str) -> Vec<SseEvent> {
        self.buffer.push_str(bytes);
        let mut events = Vec::new();
        while let Some(pos) = self.buffer.find('\n') {
            let mut line: String = self.buffer.drain(..=pos).collect();
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
            if line.is_empty() {
                if !self.data.is_empty() {
                    events.push(SseEvent { event: self.event.take(), data: self.data.join("\n") });
                    self.data.clear();
                }
                self.event = None;
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = match line.split_once(':') {
                Some((f, v)) => (f.to_string(), v.strip_prefix(' ').unwrap_or(v).to_string()),
                None => (line.clone(), String::new()),
            };
            match field.as_str() {
                "event" => self.event = Some(value),
                "data" => self.data.push(value),
                _ => {}
            }
        }
        events
    }

    pub(crate) fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = self.feed("\n\n");
        if !self.data.is_empty() {
            events.push(SseEvent { event: self.event.take(), data: self.data.join("\n") });
            self.data.clear();
        }
        events
    }
}

impl Connection {
    pub fn new(provider: Provider, config: std::sync::Arc<Config>) -> Result<Connection> {
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| Error::Configuration(e.to_string()))?;
        Ok(Connection { client, provider, config })
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    pub fn config(&self) -> &std::sync::Arc<Config> {
        &self.config
    }

    fn url(&self, path: &str) -> Result<String> {
        if path.starts_with("http://") || path.starts_with("https://") {
            return Ok(path.to_string());
        }
        let base = self.provider.api_base(&self.config)?;
        Ok(format!("{}/{}", base.trim_end_matches('/'), path.trim_start_matches('/')))
    }

    fn request(&self, url: &str, payload: &Value, extra: &[(String, String)]) -> reqwest::RequestBuilder {
        let mut req = self.client.post(url).json(payload);
        for (k, v) in self.provider.headers(&self.config) {
            req = req.header(k, v);
        }
        for (k, v) in extra {
            req = req.header(k.as_str(), v.as_str());
        }
        req
    }

    /// faraday-retry's `calculate_sleep_amount`: a `Retry-After` longer than `retry_max_interval`
    /// means "don't retry" (`None`); otherwise wait the longer of it and the jittered backoff.
    fn backoff(&self, attempt: u32, retry_after: Option<f64>) -> Option<Duration> {
        let current = (self.config.retry_interval * self.config.retry_backoff_factor.powi(attempt as i32))
            .min(self.config.retry_max_interval);
        let interval = current + rand::random::<f64>() * self.config.retry_interval_randomness * self.config.retry_interval;
        match retry_after {
            Some(after) if after > self.config.retry_max_interval => None,
            Some(after) if after >= interval => Some(Duration::from_secs_f64(after)),
            _ => Some(Duration::from_secs_f64(interval)),
        }
    }

    /// Whether to retry after `error` on attempt `attempt`, and how long to wait first.
    fn retry_delay(&self, error: &Error, attempt: u32, retry_after: Option<f64>) -> Option<Duration> {
        if !error.retryable() || attempt >= self.config.max_retries {
            return None;
        }
        self.backoff(attempt, retry_after)
    }

    async fn send_with_retry(
        &self,
        url: &str,
        payload: &Value,
        extra: &[(String, String)],
        on_attempt: &mut (dyn FnMut(Option<&Error>) + Send),
    ) -> Result<reqwest::Response> {
        self.send_built(&|| self.request(url, payload, extra), true, on_attempt).await
    }

    /// Sends the request `build` makes, retrying it under the rules above when `retry` is set.
    async fn send_built(
        &self,
        build: &(dyn Fn() -> reqwest::RequestBuilder + Send + Sync),
        retry: bool,
        on_attempt: &mut (dyn FnMut(Option<&Error>) + Send),
    ) -> Result<reqwest::Response> {
        let mut attempt = 0;
        let mut previous: Option<Error> = None;
        loop {
            on_attempt(previous.as_ref());
            let result = build().send().await;
            let (error, retry_after) = match result {
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let retry_after = retry_after_secs(&resp);
                    let body = resp.text().await.unwrap_or_default();
                    let body = self.provider.strip_html_error(&body).unwrap_or(body);
                    (error_for_status(status, &body), retry_after)
                }
                Err(e) if e.is_timeout() => (Error::Timeout(e.to_string()), None),
                Err(e) => (Error::ConnectionFailed(e.to_string()), None),
            };
            let delay = if retry { self.retry_delay(&error, attempt, retry_after) } else { None };
            let Some(delay) = delay else { return Err(error) };
            tracing::debug!(provider = self.provider.slug(), attempt, "retrying after {error}");
            tokio::time::sleep(delay).await;
            previous = Some(error);
            attempt += 1;
        }
    }

    /// POST JSON and parse a JSON response. `on_attempt` fires before every attempt so the usage
    /// ledger can record retries.
    pub async fn post(
        &self,
        path: &str,
        payload: &Value,
        extra: &[(String, String)],
        on_attempt: &mut (dyn FnMut(Option<&Error>) + Send),
    ) -> Result<RawResponse> {
        let url = self.url(path)?;
        let resp = self.send_with_retry(&url, payload, extra, on_attempt).await?;
        let status = resp.status().as_u16();
        let headers = header_pairs(&resp);
        let text = resp.text().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        if text.trim().is_empty() {
            return Err(Error::Api("Provider returned an empty response body".into(), None));
        }
        let body: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));
        Ok(RawResponse { status, headers, body, request_body: payload.to_string().into() })
    }

    /// `Connection#post` with a multipart body, for Files and Batches uploads. `form` builds a fresh
    /// body per attempt. Pass `idempotent: false` (`mark_non_idempotent`) for a request that creates
    /// server-side state: it is then never retried, since a retry after a lost response would
    /// create it twice.
    pub async fn post_multipart(
        &self,
        path: &str,
        form: impl Fn() -> reqwest::multipart::Form + Send + Sync,
        extra: &[(String, String)],
        idempotent: bool,
    ) -> Result<RawResponse> {
        let resp = self.send(reqwest::Method::POST, path, extra, idempotent, &|req| req.multipart(form())).await?;
        json_response(resp, Value::Null).await
    }

    /// `Connection#get`, parsing a JSON response. GETs retry like any idempotent request.
    pub async fn get(&self, path: &str, extra: &[(String, String)]) -> Result<RawResponse> {
        let resp = self.send(reqwest::Method::GET, path, extra, true, &|req| req).await?;
        json_response(resp, Value::Null).await
    }

    /// `Connection#get` for a binary body, such as a file download.
    pub async fn get_bytes(&self, path: &str, extra: &[(String, String)]) -> Result<Vec<u8>> {
        let resp = self.send(reqwest::Method::GET, path, extra, true, &|req| req).await?;
        Ok(resp.bytes().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?.to_vec())
    }

    /// `Connection#delete`, parsing a JSON response if there is one.
    pub async fn delete(&self, path: &str, extra: &[(String, String)]) -> Result<RawResponse> {
        let resp = self.send(reqwest::Method::DELETE, path, extra, true, &|req| req).await?;
        let status = resp.status().as_u16();
        let headers = header_pairs(&resp);
        let text = resp.text().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        let body = if text.trim().is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) };
        Ok(RawResponse { status, headers, body, request_body: Default::default() })
    }

    /// Any request with the provider's headers plus `extra`, with errors mapped like `post`.
    /// `body` adds the body to the builder; it runs once per attempt. `retry: false` sends once.
    pub async fn send(
        &self,
        method: reqwest::Method,
        path: &str,
        extra: &[(String, String)],
        retry: bool,
        body: &(dyn Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync),
    ) -> Result<reqwest::Response> {
        self.send_tracked(method, path, extra, retry, body, &mut |_| {}).await
    }

    /// `send` with `on_attempt` firing before every attempt, like `post`, so the usage ledger can
    /// record retries of a request whose response is not JSON (speech audio, multipart uploads).
    pub async fn send_tracked(
        &self,
        method: reqwest::Method,
        path: &str,
        extra: &[(String, String)],
        retry: bool,
        body: &(dyn Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder + Send + Sync),
        on_attempt: &mut (dyn FnMut(Option<&Error>) + Send),
    ) -> Result<reqwest::Response> {
        let url = self.url(path)?;
        let build = || {
            let mut req = self.client.request(method.clone(), &url);
            for (k, v) in self.provider.headers(&self.config) {
                req = req.header(k, v);
            }
            for (k, v) in extra {
                req = req.header(k.as_str(), v.as_str());
            }
            body(req)
        };
        self.send_built(&build, retry, on_attempt).await
    }

    /// POST JSON and feed each server-sent event to `on_event`. Errors inside the stream
    /// (`event: error`, `{"error": ...}` data) are raised through the same status mapping, and are
    /// retried like HTTP errors as long as no event has reached `on_event` yet. Once the caller has
    /// seen a chunk, a failure is final: retrying would replay it into the same accumulator.
    pub async fn stream(
        &self,
        path: &str,
        payload: &Value,
        extra: &[(String, String)],
        on_attempt: &mut (dyn FnMut(Option<&Error>) + Send),
        on_event: &mut (dyn FnMut(SseEvent, Value) -> Result<()> + Send),
        streaming_error: fn(&str) -> Option<u16>,
    ) -> Result<RawResponse> {
        let url = self.url(path)?;
        let mut attempt = 0;
        let mut stream_error: Option<Error> = None;
        loop {
            let mut first = stream_error.take();
            let mut forward = |previous: Option<&Error>| on_attempt(previous.or(first.take().as_ref()));
            let resp = self.send_with_retry(&url, payload, extra, &mut forward).await?;
            let status = resp.status().as_u16();
            let headers = header_pairs(&resp);
            let mut delivered = false;
            match self.read_stream(resp, on_event, streaming_error, &mut delivered).await {
                Ok(()) => return Ok(RawResponse { status, headers, body: Value::Null, request_body: payload.to_string().into() }),
                Err(error) if delivered => return Err(error),
                Err(error) => {
                    let Some(delay) = self.retry_delay(&error, attempt, None) else { return Err(error) };
                    tracing::debug!(provider = self.provider.slug(), attempt, "retrying stream after {error}");
                    tokio::time::sleep(delay).await;
                    stream_error = Some(error);
                    attempt += 1;
                }
            }
        }
    }

    pub(crate) async fn read_stream(
        &self,
        resp: reqwest::Response,
        on_event: &mut (dyn FnMut(SseEvent, Value) -> Result<()> + Send),
        streaming_error: fn(&str) -> Option<u16>,
        delivered: &mut bool,
    ) -> Result<()> {
        let mut parser = SseParser::default();
        let mut stream = resp.bytes_stream();
        let mut handle = |event: SseEvent| -> Result<()> {
            if event.data == "[DONE]" {
                return Ok(());
            }
            let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
                tracing::debug!("Failed to parse data chunk: {}", event.data);
                return Ok(());
            };
            let is_error = event.event.as_deref() == Some("error")
                || data.get("error").is_some()
                || data.get("type").and_then(Value::as_str) == Some("error");
            if is_error {
                let code = streaming_error(&event.data).unwrap_or(500);
                return Err(error_for_status(code, &event.data));
            }
            *delivered = true;
            on_event(event, data)
        };
        while let Some(bytes) = stream.next().await {
            let bytes = bytes.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
            for event in parser.feed(&String::from_utf8_lossy(&bytes)) {
                handle(event)?;
            }
        }
        for event in parser.finish() {
            handle(event)?;
        }
        Ok(())
    }
}

/// Reads a JSON body the way `post` does: an empty body is an error, a non-JSON one a string.
pub(crate) async fn json_response(resp: reqwest::Response, request_body: Value) -> Result<RawResponse> {
    let request_body: std::sync::Arc<str> = if request_body.is_null() { "".into() } else { request_body.to_string().into() };
    let status = resp.status().as_u16();
    let headers = header_pairs(&resp);
    let text = resp.text().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
    if text.trim().is_empty() {
        return Err(Error::Api("Provider returned an empty response body".into(), None));
    }
    let body: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));
    Ok(RawResponse { status, headers, body, request_body })
}

fn header_pairs(resp: &reqwest::Response) -> Vec<(String, String)> {
    resp.headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or_default().to_string()))
        .collect()
}

fn retry_after_secs(resp: &reqwest::Response) -> Option<f64> {
    let h = resp.headers();
    if let Some(ms) = h.get("retry-after-ms").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<f64>().ok())
        && ms.is_finite() && ms >= 0.0 {
            return Some(ms / 1000.0);
        }
    h.get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<f64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_events_split_across_reads_are_reassembled() {
        let mut p = SseParser::default();
        assert!(p.feed("event: message_start\ndata: {\"a\":").is_empty());
        let events = p.feed("1}\n\nevent: ping\ndata: {}\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[0].data, "{\"a\":1}");
    }
}
