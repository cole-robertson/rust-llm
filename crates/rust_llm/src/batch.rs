//! Port of `lib/ruby_llm/batch.rb`, `lib/ruby_llm/embedding_request.rb`, the batch methods of
//! `lib/ruby_llm/provider.rb`, and the batch protocols: `protocols/anthropic/batches.rb`,
//! `protocols/openai/batches.rb` (with `responses/batches.rb`, `chat_completions/batches.rb`,
//! `chat_completions/embedding_batches.rb`), `protocols/gemini/batches.rb`,
//! `providers/mistral/chat_completions/batches.rb`, and `providers/xai/chat_completions/batches.rb`.
//!
//! ```ruby
//! chats = documents.map { |doc| RubyLLM.chat(model: "claude-haiku-4-5").ask_later(doc.text) }
//! batch = RubyLLM.batch(chats)
//! batch.refresh.complete? # => false, check back later
//! batch.messages          # the responses, in submission order
//! ```
//!
//! The batch owns the chats it answers: read them back with `chats()` or `into_chats()`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::chat::{Chat, resolve_model};
use crate::config::Config;
use crate::cost::{Component, Cost, Tier};
use crate::embedding::{EmbedOptions, Embedding};
use crate::error::{Error, Result};
use crate::message::{Message, Operation, RawResponse, Role, UsageEntry, UsageStatus};
use crate::model::{Model, PricingCategory, PricingTier};
use crate::models;
use crate::protocols::{anthropic, chat_completions, gemini, responses};
use crate::providers::Provider;
use crate::tokens::Tokens;
use crate::transport::Connection;

/// The provider-neutral lifecycle status of a batch, and the outcome of each request in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStatus {
    Pending,
    Succeeded,
    Failed,
    Cancelled,
}

/// One collected answer: a `Message` in a chat batch, an `Embedding` in an embeddings batch.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // public result type; boxing would change the API
pub enum BatchResult {
    Message(Message),
    Embedding(Embedding),
}

impl BatchResult {
    pub fn as_message(&self) -> Option<&Message> {
        match self {
            BatchResult::Message(m) => Some(m),
            BatchResult::Embedding(_) => None,
        }
    }

    pub fn as_embedding(&self) -> Option<&Embedding> {
        match self {
            BatchResult::Embedding(e) => Some(e),
            BatchResult::Message(_) => None,
        }
    }

    pub fn tokens(&self) -> Tokens {
        match self {
            BatchResult::Message(m) => m.tokens(),
            BatchResult::Embedding(e) => e.tokens(),
        }
    }

    pub fn cost(&self) -> Cost {
        match self {
            BatchResult::Message(m) => m.cost(None),
            BatchResult::Embedding(e) => e.cost(),
        }
    }
}

/// `RubyLLM::EmbeddingRequest`: an embedding awaiting a provider-side batch. Collecting the
/// batch's results fills in `result`.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub text: String,
    pub dimensions: Option<i64>,
    /// The Embedding once the batch completed and its results were collected; `None` until
    /// then, and for requests that failed.
    pub result: Option<Embedding>,
    model: Model,
    provider: Provider,
    config: Arc<Config>,
}

impl EmbeddingRequest {
    pub fn new(text: impl Into<String>, options: EmbedOptions<'_>) -> Result<EmbeddingRequest> {
        let config = options.config.clone().unwrap_or_else(crate::config);
        let model_id = options.model.unwrap_or(&config.default_embedding_model).to_string();
        let (model, provider) = resolve_model(&model_id, options.provider, options.assume_model_exists)?;
        Ok(EmbeddingRequest { text: text.into(), dimensions: options.dimensions, result: None, model, provider, config })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// `EmbeddingRequest#render`: the embedding payload in the provider's wire format.
    pub fn render(&self) -> Result<Value> {
        match self.provider {
            Provider::Anthropic => Err(Error::Api("Anthropic doesn't support embeddings".into(), None)),
            Provider::Gemini => {
                let mut r = json!({ "model": format!("models/{}", self.model.id), "content": { "parts": [{ "text": self.text }] } });
                if let Some(d) = self.dimensions {
                    r["outputDimensionality"] = d.into();
                }
                Ok(json!({ "requests": [r] }))
            }
            Provider::Mistral => {
                let mut p = json!({ "model": self.model.id, "input": self.text });
                if let Some(d) = self.dimensions {
                    p["output_dimension"] = d.into();
                }
                Ok(p)
            }
            _ => {
                let mut p = json!({ "model": self.model.id, "input": self.text });
                if let Some(d) = self.dimensions {
                    p["dimensions"] = d.into();
                }
                Ok(p)
            }
        }
    }
}

/// `RubyLLM.embed_later(text, model:, provider:, dimensions:)`.
pub fn embed_later(text: impl Into<String>, options: EmbedOptions<'_>) -> Result<EmbeddingRequest> {
    EmbeddingRequest::new(text, options)
}

/// What `RubyLLM.batch` accepts: chats, or embedding requests, never both.
pub enum Submission {
    Chats(Vec<Chat>),
    Embeddings(Vec<EmbeddingRequest>),
}

impl From<Vec<Chat>> for Submission {
    fn from(chats: Vec<Chat>) -> Self {
        Submission::Chats(chats)
    }
}

impl From<Chat> for Submission {
    fn from(chat: Chat) -> Self {
        Submission::Chats(vec![chat])
    }
}

impl From<Vec<EmbeddingRequest>> for Submission {
    fn from(requests: Vec<EmbeddingRequest>) -> Self {
        Submission::Embeddings(requests)
    }
}

impl From<EmbeddingRequest> for Submission {
    fn from(request: EmbeddingRequest) -> Self {
        Submission::Embeddings(vec![request])
    }
}

/// `RubyLLM.batch(chats)`.
pub async fn batch(items: impl Into<Submission>) -> Result<Batch> {
    Batch::submit(items).await
}

/// `RubyLLM::Batch`.
pub struct Batch {
    http: Http,
    chats: Option<Vec<Chat>>,
    requests: Option<Vec<EmbeddingRequest>>,
    batch_protocol: Option<Kind>,
    id: String,
    status: BatchStatus,
    raw_status: Option<String>,
    completed: bool,
    request_counts: Option<Value>,
    request_count: Option<usize>,
    statuses: Vec<Option<BatchStatus>>,
    delivered: HashSet<usize>,
    cached: Option<Vec<Option<BatchResult>>>,
}

impl std::fmt::Debug for Batch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("id", &self.id)
            .field("status", &self.status)
            .field("raw_status", &self.raw_status)
            .field("chats", &self.chats.as_ref().map(Vec::len))
            .field("requests", &self.requests.as_ref().map(Vec::len))
            .finish()
    }
}

const AWAITING_ROLES: [Role; 2] = [Role::User, Role::Tool];

impl Batch {
    /// `Batch.submit`: submits chats (each staged with `ask_later`) or embedding requests to their
    /// shared provider as one batch.
    pub async fn submit(items: impl Into<Submission>) -> Result<Batch> {
        match items.into() {
            Submission::Chats(chats) => Batch::submit_chats(chats).await,
            Submission::Embeddings(requests) => Batch::submit_embeddings(requests).await,
        }
    }

    /// `Batch.find(id, provider:)`: the provider's current state for `id`, from any process.
    pub async fn find(id: &str, provider: Option<&str>) -> Result<Batch> {
        Batch::find_with_config(crate::config(), id, provider).await
    }

    /// `Batch.find(id, provider:, context:)`.
    pub async fn find_with_config(config: Arc<Config>, id: &str, provider: Option<&str>) -> Result<Batch> {
        let Some(provider) = provider else {
            return Err(Error::Argument("Provider must be specified to find a batch that is not persisted by RustLLM".into()));
        };
        let provider = Provider::resolve_or_err(provider)?;
        let kind = default_kind(provider)?;
        let mut batch = Batch::new(provider, config)?;
        let attrs = batch.http.find(kind, id).await?;
        batch.apply(attrs);
        Ok(batch)
    }

    /// Hands a found batch the chats it answers, in submission order, so collecting appends to
    /// them (`Batch.new(provider:, chats:, ...)`). Answers already in a chat are not appended again.
    pub fn with_chats(mut self, chats: Vec<Chat>) -> Self {
        self.chats = Some(chats);
        self
    }

    fn new(provider: Provider, config: Arc<Config>) -> Result<Batch> {
        Ok(Batch {
            http: Http::new(provider, config)?,
            chats: None,
            requests: None,
            batch_protocol: None,
            id: String::new(),
            status: BatchStatus::Pending,
            raw_status: None,
            completed: false,
            request_counts: None,
            request_count: None,
            statuses: Vec::new(),
            delivered: HashSet::new(),
            cached: None,
        })
    }

    async fn submit_chats(chats: Vec<Chat>) -> Result<Batch> {
        if chats.is_empty() {
            return Err(Error::Argument("Cannot submit an empty batch".into()));
        }
        if !chats.iter().all(awaiting_model) {
            return Err(Error::Argument(
                "Every chat in a batch must be awaiting the model; stage one with ask_later, or run_tools first".into(),
            ));
        }
        let provider = shared_provider(chats.iter().map(Chat::provider))?;
        let requests = chats
            .iter()
            .enumerate()
            .map(|(i, chat)| Ok(Req { custom_id: i.to_string(), model: chat.model().id.clone(), payload: chat.render()?, text: None }))
            .collect::<Result<Vec<_>>>()?;
        let mut batch = Batch::new(provider, chats[0].config().clone())?;
        batch.create(&requests).await?;
        batch.chats = Some(chats);
        Ok(batch)
    }

    async fn submit_embeddings(requests: Vec<EmbeddingRequest>) -> Result<Batch> {
        if requests.is_empty() {
            return Err(Error::Argument("Cannot submit an empty batch".into()));
        }
        let provider = shared_provider(requests.iter().map(EmbeddingRequest::provider))?;
        let lines = requests
            .iter()
            .enumerate()
            .map(|(i, r)| Ok(Req { custom_id: i.to_string(), model: r.model.id.clone(), payload: r.render()?, text: Some(r.text.clone()) }))
            .collect::<Result<Vec<_>>>()?;
        let mut batch = Batch::new(provider, requests[0].config.clone())?;
        batch.create(&lines).await?;
        batch.requests = Some(requests);
        Ok(batch)
    }

    /// `Provider#create_batch`: picks the batch protocol for the requests, then submits.
    async fn create(&mut self, requests: &[Req]) -> Result<()> {
        let kind = kind_for(self.http.provider, requests)?;
        let attrs = self.http.create(kind, requests).await?;
        self.batch_protocol = Some(kind);
        self.apply(attrs);
        Ok(())
    }

    /// The provider's batch id. Persist it to load the batch again with `find`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The slug of the provider running the batch.
    pub fn provider(&self) -> &'static str {
        self.http.provider.slug()
    }

    pub fn status(&self) -> BatchStatus {
        self.status
    }

    /// The provider-reported status string, such as "in_progress".
    pub fn raw_status(&self) -> Option<&str> {
        self.raw_status.as_deref()
    }

    /// The provider-reported request tallies by state, when the provider reports them.
    pub fn request_counts(&self) -> Option<&Value> {
        self.request_counts.as_ref()
    }

    /// The submitted chats in order; `None` for a batch loaded with `find` or holding embeddings.
    pub fn chats(&self) -> Option<&[Chat]> {
        self.chats.as_deref()
    }

    pub fn into_chats(self) -> Option<Vec<Chat>> {
        self.chats
    }

    /// The submitted embedding requests in order; `None` for a chat batch or a found batch.
    pub fn requests(&self) -> Option<&[EmbeddingRequest]> {
        self.requests.as_deref()
    }

    /// The outcome of each collected request, in submission order.
    pub fn statuses(&self) -> &[Option<BatchStatus>] {
        &self.statuses
    }

    /// The protocol the batch was submitted through, e.g. "responses".
    pub fn batch_protocol(&self) -> Option<&'static str> {
        self.batch_protocol.map(Kind::name)
    }

    /// `complete?`: as of the last state fetched. Never contacts the provider; poll with `refresh`.
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    pub fn is_succeeded(&self) -> bool {
        self.status == BatchStatus::Succeeded
    }

    pub fn is_failed(&self) -> bool {
        self.status == BatchStatus::Failed
    }

    pub fn is_cancelled(&self) -> bool {
        self.status == BatchStatus::Cancelled
    }

    /// Re-fetches the batch, updating `status`, `raw_status`, `request_counts`, and `is_complete`.
    pub async fn refresh(&mut self) -> Result<&mut Self> {
        let kind = default_kind(self.http.provider)?;
        let attrs = self.http.find(kind, &self.id).await?;
        self.apply(attrs);
        Ok(self)
    }

    /// Asks the provider to cancel the batch. Requests already processed still return results.
    pub async fn cancel(&mut self) -> Result<&mut Self> {
        let kind = default_kind(self.http.provider)?;
        let attrs = self.http.cancel(kind, &self.id).await?;
        self.apply(attrs);
        Ok(self)
    }

    /// `messages`/`results`: the answers in submission order, `None` where a request failed. Chat
    /// answers are also appended to their chats; embeddings fill their request's `result`. Cached
    /// once the batch is complete, so collecting early keeps reading fresh.
    pub async fn results(&mut self) -> Result<Vec<Option<BatchResult>>> {
        if let Some(cached) = &self.cached {
            return Ok(cached.clone());
        }
        let collected = self.collect_results().await?;
        if self.completed {
            self.cached = Some(collected.clone());
        }
        Ok(collected)
    }

    /// The chat answers of `results`.
    pub async fn messages(&mut self) -> Result<Vec<Option<Message>>> {
        Ok(self.results().await?.into_iter().map(|r| r.and_then(|r| r.as_message().cloned())).collect())
    }

    /// Token usage aggregated across the collected responses.
    pub async fn tokens(&mut self) -> Result<Tokens> {
        let tokens: Vec<Tokens> = self.results().await?.iter().flatten().map(BatchResult::tokens).collect();
        Ok(Tokens::aggregate(tokens.iter()))
    }

    /// The collected responses' cost at batch rates; the total is `None` until the batch ends.
    pub async fn cost(&mut self) -> Result<Cost> {
        if !self.completed {
            return Ok(Cost::aggregate(std::iter::empty::<&Cost>(), false));
        }
        let costs: Vec<Cost> = self.results().await?.iter().flatten().map(BatchResult::cost).collect();
        Ok(Cost::aggregate(costs.iter(), true))
    }

    fn apply(&mut self, attrs: Attrs) {
        if attrs.protocol.is_some() {
            self.batch_protocol = attrs.protocol;
        }
        self.id = attrs.id;
        self.raw_status = attrs.raw_status;
        self.completed = attrs.completed;
        self.request_counts = attrs.request_counts;
        self.request_count = attrs.request_count;
        let kind = self.batch_protocol.or_else(|| default_kind(self.http.provider).ok());
        self.status = kind.map_or(BatchStatus::Pending, |k| k.status(self.raw_status.as_deref(), self.completed));
    }

    async fn results_kind(&self) -> Result<Kind> {
        if let Some(kind) = self.batch_protocol {
            return Ok(kind);
        }
        // `OpenAI#batch_protocol_for_stored_batch`: the endpoint says which protocol ran it.
        if self.http.provider == Provider::OpenAI {
            let data = self.http.get(&format!("batches/{}", self.id)).await?;
            return Ok(openai_kind_for_endpoint(data.get("endpoint")).unwrap_or(Kind::Responses));
        }
        default_kind(self.http.provider)
    }

    async fn collect_results(&mut self) -> Result<Vec<Option<BatchResult>>> {
        let kind = self.results_kind().await?;
        let rows = self.http.results(kind, &self.id).await?;
        let known = self.known_request_count();
        let mut seen = HashSet::new();
        for (index, _, _) in &rows {
            if *index < 0 || known.is_some_and(|count| *index as usize >= count) {
                return Err(Error::Api(format!("Invalid batch result index: {index}"), None));
            }
            if !seen.insert(*index) {
                return Err(Error::Api(format!("Duplicate batch result index: {index}"), None));
            }
        }
        let size = known.unwrap_or_else(|| rows.iter().map(|(i, _, _)| *i as usize + 1).max().unwrap_or(0));
        let mut slots: Vec<Option<BatchResult>> = vec![None; size];
        if self.statuses.len() < size {
            self.statuses.resize(size, None);
        }
        for (index, result, failure) in rows {
            let index = index as usize;
            self.statuses[index] = Some(if result.is_some() { BatchStatus::Succeeded } else { failure });
            if let Some(result) = result {
                slots[index] = Some(self.deliver(index, result)?);
            }
        }
        if self.completed {
            let missing = if self.is_cancelled() { BatchStatus::Cancelled } else { BatchStatus::Failed };
            for status in self.statuses.iter_mut().take(size) {
                status.get_or_insert(missing);
            }
        }
        Ok(slots)
    }

    fn known_request_count(&self) -> Option<usize> {
        self.chats.as_ref().map(Vec::len).or_else(|| self.requests.as_ref().map(Vec::len)).or(self.request_count)
    }

    /// Collecting early keeps reading fresh, so a result already delivered comes back on every
    /// later poll: hand each one over once.
    fn deliver(&mut self, index: usize, result: BatchResult) -> Result<BatchResult> {
        match result {
            BatchResult::Embedding(mut embedding) => {
                let model = self.requests.as_ref().and_then(|r| r.get(index)).map(|r| r.model.clone());
                if embedding.usage_entries.is_empty() {
                    let entry = self.batch_usage(Operation::Embedding, Some(embedding.model.as_str()), embedding.tokens(), model)?;
                    embedding.usage_entries = vec![entry];
                }
                if self.delivered.insert(index)
                    && let Some(request) = self.requests.as_mut().and_then(|r| r.get_mut(index))
                {
                    request.result = Some(embedding.clone());
                }
                Ok(BatchResult::Embedding(embedding))
            }
            BatchResult::Message(mut message) => {
                let chat_model = self.chats.as_ref().and_then(|c| c.get(index)).map(|c| c.model().clone());
                if message.usage_entries.is_empty() {
                    let entry = self.batch_usage(Operation::Chat, message.model.as_deref(), message.tokens.clone(), chat_model)?;
                    message.usage_entries = vec![entry];
                }
                let delivered = self.delivered.contains(&index)
                    || self.chats.as_ref().and_then(|c| c.get(index)).is_some_and(|chat| already_in_chat(chat, &message));
                if !delivered {
                    self.delivered.insert(index);
                    if let Some(chat) = self.chats.as_mut().and_then(|c| c.get_mut(index)) {
                        chat.add_completion(message.clone(), true);
                    }
                }
                Ok(BatchResult::Message(message))
            }
        }
    }

    /// `attach_batch_usage`: one succeeded entry priced at batch rates.
    fn batch_usage(&self, operation: Operation, result_model: Option<&str>, tokens: Tokens, model: Option<Model>) -> Result<UsageEntry> {
        let provider = self.http.provider;
        let model = match model {
            Some(m) => m,
            None => models::models().find(result_model.unwrap_or_default(), Some(provider.slug()))?,
        };
        let cost = batch_cost(provider, &tokens, &model);
        Ok(UsageEntry {
            id: UsageEntry::next_id(),
            operation,
            provider: provider.slug().into(),
            model: result_model.map(str::to_string).unwrap_or_else(|| model.id.clone()),
            status: UsageStatus::Succeeded,
            tokens,
            cost,
        })
    }
}

fn awaiting_model(chat: &Chat) -> bool {
    !chat.is_complete() && chat.messages().last().is_some_and(|m| AWAITING_ROLES.contains(&m.role))
}

/// A plain answer is the chat's last message once it arrives. A tool-call answer is not: running
/// its tools adds messages after it, so match on its tool-call ids instead.
fn already_in_chat(chat: &Chat, message: &Message) -> bool {
    match &message.tool_calls {
        Some(calls) if message.is_tool_call() => chat.messages().iter().any(|m| {
            m.is_tool_call() && m.tool_calls.as_ref().is_some_and(|c| c.keys().any(|k| calls.contains_key(k)))
        }),
        _ => !chat.messages().last().is_some_and(|m| AWAITING_ROLES.contains(&m.role)),
    }
}

fn shared_provider(providers: impl Iterator<Item = Provider>) -> Result<Provider> {
    let mut slugs: Vec<Provider> = Vec::new();
    for p in providers {
        if !slugs.contains(&p) {
            slugs.push(p);
        }
    }
    if slugs.len() > 1 {
        let names: Vec<&str> = slugs.iter().map(Provider::slug).collect();
        return Err(Error::Argument(format!("A batch takes one provider per submission, got: {}", names.join(", "))));
    }
    let provider = slugs[0];
    default_kind(provider)?;
    Ok(provider)
}

// ---- pricing ------------------------------------------------------------------------------------

const COMPONENTS: [Component; 5] =
    [Component::Input, Component::Output, Component::CacheRead, Component::CacheWrite, Component::Thinking];

/// `Provider#batch_cost_multiplier`: the provider's batch discount per component, applied when
/// the model lists no batch price.
fn batch_cost_multiplier(provider: Provider, component: Component) -> Option<f64> {
    match provider {
        Provider::OpenAI | Provider::Anthropic | Provider::Mistral => Some(0.5),
        Provider::Gemini => Some(if matches!(component, Component::CacheRead | Component::CacheWrite) { 1.0 } else { 0.5 }),
        _ => None,
    }
}

fn batch_rate(tier: &PricingTier, component: Component) -> Option<f64> {
    match component {
        Component::Input => tier.input_per_million,
        Component::Output => tier.output_per_million,
        Component::CacheRead => tier.cache_read_input_per_million,
        Component::CacheWrite => tier.cache_write_input_per_million,
        Component::Thinking => tier.reasoning_output_per_million,
    }
}

fn long_context_pricing(pricing: &PricingCategory, tokens: &Tokens) -> bool {
    match (&pricing.long_context, pricing.long_context_threshold) {
        (Some(_), Some(threshold)) => {
            tokens.input.unwrap_or(0) + tokens.cache_read.unwrap_or(0) + tokens.cache_write.unwrap_or(0) > threshold
        }
        _ => false,
    }
}

/// `Provider#batch_cost`: the model's batch tier where it lists one, otherwise the standard price
/// times the provider's batch discount. A provider-reported cost wins outright. Like the rest of
/// this port's `Cost`, embeddings are priced from the model's text-token prices.
pub fn batch_cost(provider: Provider, tokens: &Tokens, model: &Model) -> Cost {
    let standard = Cost::new(tokens, Some(model), Tier::Standard);
    if tokens.reported_cost.is_some() {
        return standard;
    }
    let pricing = model.pricing.text_tokens();
    let batch_tier = if long_context_pricing(&pricing, tokens) { None } else { pricing.batch.clone() };
    let batch = batch_tier.as_ref().map(|_| Cost::new(tokens, Some(model), Tier::Batch));
    let mut amounts = [None; 5];
    let mut missing = Vec::new();
    for (i, component) in COMPONENTS.into_iter().enumerate() {
        amounts[i] = match (&batch_tier, &batch) {
            (Some(tier), Some(batch)) if batch_rate(tier, component).is_some() => batch.get(component),
            _ => standard.get(component).zip(batch_cost_multiplier(provider, component)).map(|(v, m)| v * m),
        };
        if amounts[i].is_none()
            && (standard.missing().contains(&component) || standard.get(component).is_some_and(|v| v > 0.0))
        {
            missing.push(component);
        }
    }
    Cost::from_amounts(amounts, missing, standard.is_reported())
}

// ---- protocols ----------------------------------------------------------------------------------

/// The batch protocol a batch runs through. RubyLLM composes `Batches` modules into protocol
/// classes; here each dialect is a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Anthropic,
    Responses,
    ChatCompletions,
    Embeddings,
    Gemini,
    Mistral,
    XAI,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Anthropic => "anthropic",
            Kind::Responses => "responses",
            Kind::ChatCompletions | Kind::Mistral | Kind::XAI => "chat_completions",
            Kind::Embeddings => "embeddings",
            Kind::Gemini => "gemini",
        }
    }

    fn is_openai(self) -> bool {
        matches!(self, Kind::Responses | Kind::ChatCompletions | Kind::Embeddings)
    }

    /// `parse_batch_status`.
    fn status(self, raw: Option<&str>, completed: bool) -> BatchStatus {
        if !completed {
            return BatchStatus::Pending;
        }
        let raw = raw.unwrap_or("");
        match self {
            Kind::Anthropic if raw == "ended" => BatchStatus::Succeeded,
            Kind::Anthropic => BatchStatus::Pending,
            Kind::Responses | Kind::ChatCompletions | Kind::Embeddings => match raw {
                "completed" => BatchStatus::Succeeded,
                "cancelled" => BatchStatus::Cancelled,
                _ => BatchStatus::Failed,
            },
            Kind::Gemini if raw.ends_with("SUCCEEDED") => BatchStatus::Succeeded,
            Kind::Gemini if raw.ends_with("CANCELLED") => BatchStatus::Cancelled,
            Kind::Gemini => BatchStatus::Failed,
            Kind::Mistral => match raw {
                "SUCCESS" => BatchStatus::Succeeded,
                "CANCELLED" => BatchStatus::Cancelled,
                _ => BatchStatus::Failed,
            },
            Kind::XAI if raw == "failed" => BatchStatus::Failed,
            Kind::XAI => BatchStatus::Succeeded,
        }
    }
}

/// `Provider#batch_protocol`: the provider's default batch protocol, or an error when it has none.
fn default_kind(provider: Provider) -> Result<Kind> {
    match provider {
        Provider::OpenAI => Ok(Kind::Responses),
        Provider::Anthropic => Ok(Kind::Anthropic),
        Provider::Gemini => Ok(Kind::Gemini),
        Provider::Mistral => Ok(Kind::Mistral),
        Provider::XAI => Ok(Kind::XAI),
        _ => Err(Error::Api(format!("{} doesn't support batch requests", provider.slug()), None)),
    }
}

/// `Provider#batch_protocol_for`: OpenAI routes by payload shape; everyone else has one protocol.
fn kind_for(provider: Provider, requests: &[Req]) -> Result<Kind> {
    if provider != Provider::OpenAI {
        return default_kind(provider);
    }
    let mut kinds: Vec<Kind> = Vec::new();
    for r in requests {
        let input = r.payload.get("input");
        let kind = match input {
            Some(Value::String(_)) => Kind::Embeddings,
            Some(Value::Array(a)) if !a.is_empty() && a.iter().all(Value::is_string) => Kind::Embeddings,
            Some(_) => Kind::Responses,
            None if r.payload.get("messages").is_some() => Kind::ChatCompletions,
            None => {
                return Err(Error::Api("openai batch requests only support chat, responses, or embedding payloads".into(), None));
            }
        };
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    }
    match kinds.as_slice() {
        [kind] => Ok(*kind),
        _ => Err(Error::Api("openai batch requests must target one endpoint per submission".into(), None)),
    }
}

fn openai_kind_for_endpoint(endpoint: Option<&Value>) -> Option<Kind> {
    match endpoint?.as_str()?.trim_start_matches('/') {
        "v1/responses" | "responses" => Some(Kind::Responses),
        "v1/chat/completions" | "chat/completions" => Some(Kind::ChatCompletions),
        "v1/embeddings" | "embeddings" => Some(Kind::Embeddings),
        _ => None,
    }
}

fn openai_endpoint(kind: Kind) -> &'static str {
    match kind {
        Kind::ChatCompletions => "/v1/chat/completions",
        Kind::Embeddings => "/v1/embeddings",
        _ => "/v1/responses",
    }
}

/// One staged request: `{ custom_id:, model:, payload:, text: }`.
struct Req {
    custom_id: String,
    model: String,
    payload: Value,
    text: Option<String>,
}

/// The attributes `find_batch`/`create_batch` return.
#[derive(Default)]
struct Attrs {
    id: String,
    raw_status: Option<String>,
    completed: bool,
    request_counts: Option<Value>,
    request_count: Option<usize>,
    protocol: Option<Kind>,
}

/// One collected row: `[index, result, failure_status]`.
type Row = (i64, Option<BatchResult>, BatchStatus);

/// `Batch::Helpers#batch_payload`: the rendered payload without `stream` and `except`.
fn batch_payload(request: &Req, except: &[&str]) -> Value {
    let mut payload = request.payload.clone();
    if let Some(o) = payload.as_object_mut() {
        o.remove("stream");
        for key in except {
            o.remove(*key);
        }
    }
    payload
}

/// `single_batch_model!`.
fn single_batch_model<'a>(requests: &'a [Req], provider_name: &str) -> Result<&'a str> {
    let first = requests[0].model.as_str();
    if requests.iter().all(|r| r.model == first) {
        return Ok(first);
    }
    Err(Error::Api(format!("{provider_name} batch requests must use one model per submission"), None))
}

/// `batch_result_index`: Ruby's `Integer(id)`.
fn batch_result_index(id: &str) -> Result<i64> {
    id.trim().parse::<i64>().map_err(|_| Error::Argument(format!("invalid value for Integer(): {id:?}")))
}

/// `batch_failure`: logs the failed request and normalizes its status.
fn batch_failure(custom_id: &str, detail: Option<String>, status: &str) -> BatchStatus {
    let mut line = format!("Batch request {custom_id} {status}");
    if let Some(detail) = detail {
        line.push_str(&format!(": {detail}"));
    }
    tracing::warn!("{line}");
    if status.to_lowercase().contains("cancel") { BatchStatus::Cancelled } else { BatchStatus::Failed }
}

/// `batch_error_message`.
fn batch_error_message(line: &Value) -> Option<String> {
    fn value(error: &Value) -> Option<String> {
        match error {
            Value::Object(o) => o.get("message").and_then(Value::as_str).map(str::to_string),
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }
    let response = line.get("response");
    line.get("error")
        .and_then(value)
        .or_else(|| line.get("error_message").and_then(Value::as_str).map(str::to_string))
        .or_else(|| response.and_then(|r| r.get("body")).and_then(|b| b.get("error")).and_then(value))
        .or_else(|| response.and_then(|r| r.get("error")).and_then(value))
}

fn count(v: Option<&Value>) -> Option<usize> {
    let v = v?;
    v.as_u64().map(|n| n as usize).or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn str_at(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}

fn random_hex() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..16].to_string()
}

fn raw(body: &Value) -> RawResponse {
    RawResponse { status: 200, headers: Vec::new(), body: body.clone(), request_body: Default::default() }
}

fn anthropic_attrs(data: &Value) -> Attrs {
    let request_counts = data.get("request_counts").filter(|v| !v.is_null()).cloned();
    let request_count = request_counts.as_ref().and_then(Value::as_object).map(|o| o.values().filter_map(Value::as_u64).sum::<u64>() as usize);
    let raw_status = str_at(data, "processing_status");
    Attrs {
        id: str_at(data, "id").unwrap_or_default(),
        completed: raw_status.as_deref() == Some("ended"),
        raw_status,
        request_counts,
        request_count,
        protocol: None,
    }
}

fn openai_attrs(data: &Value) -> Attrs {
    const TERMINAL: [&str; 4] = ["completed", "failed", "expired", "cancelled"];
    let request_counts = data.get("request_counts").filter(|v| !v.is_null()).cloned();
    let raw_status = str_at(data, "status");
    Attrs {
        id: str_at(data, "id").unwrap_or_default(),
        completed: raw_status.as_deref().is_some_and(|s| TERMINAL.contains(&s)),
        raw_status,
        request_count: count(request_counts.as_ref().and_then(|c| c.get("total"))),
        request_counts,
        protocol: openai_kind_for_endpoint(data.get("endpoint")),
    }
}

/// A batch starts as an Operation wrapping the batch in `metadata`; polling returns the batch
/// directly. Read either shape.
fn gemini_attrs(data: &Value) -> Attrs {
    const TERMINAL: [&str; 4] = ["SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"];
    let batch = data.get("metadata").unwrap_or(data);
    let raw_status = str_at(batch, "state");
    let request_counts = batch.get("batchStats").cloned();
    Attrs {
        id: str_at(data, "name").or_else(|| str_at(batch, "name")).unwrap_or_default(),
        completed: raw_status.as_deref().is_some_and(|s| TERMINAL.iter().any(|t| s.ends_with(t))),
        raw_status,
        request_count: count(request_counts.as_ref().and_then(|c| c.get("requestCount"))),
        request_counts,
        protocol: None,
    }
}

fn mistral_attrs(data: &Value) -> Attrs {
    const TERMINAL: [&str; 4] = ["SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"];
    let raw_status = str_at(data, "status");
    let mut counts = Map::new();
    for (key, field) in [("total", "total_requests"), ("completed", "completed_requests"), ("succeeded", "succeeded_requests"), ("failed", "failed_requests")] {
        if let Some(v) = data.get(field).filter(|v| !v.is_null()) {
            counts.insert(key.into(), v.clone());
        }
    }
    Attrs {
        id: str_at(data, "id").unwrap_or_default(),
        completed: raw_status.as_deref().is_some_and(|s| TERMINAL.contains(&s)),
        raw_status,
        request_count: count(data.get("total_requests")),
        request_counts: Some(Value::Object(counts)),
        protocol: None,
    }
}

fn xai_attrs(data: &Value) -> Attrs {
    let empty = json!({});
    let state = data.get("state").filter(|s| s.is_object()).unwrap_or(&empty);
    let num = |key: &str| state.get(key).and_then(Value::as_i64).unwrap_or(0);
    let completed = num("num_requests") > 0 && num("num_pending") == 0;
    let raw_status = if data.get("state").is_some() {
        let error = state.get("error").map(|e| e.as_str().map(str::to_string).unwrap_or_else(|| if e.is_null() { String::new() } else { e.to_string() }));
        Some(if error.is_some_and(|e| !e.is_empty()) { "failed" } else if completed { "completed" } else { "processing" }.to_string())
    } else {
        str_at(data, "status")
    };
    Attrs {
        id: str_at(data, "batch_id").or_else(|| str_at(data, "id")).unwrap_or_default(),
        raw_status,
        completed,
        request_count: count(state.get("num_requests")),
        request_counts: Some(state.clone()),
        protocol: None,
    }
}

/// `batch_schema_payload`: batchGenerateContent ignores `responseJsonSchema` but honors the legacy
/// `responseSchema`, so batches carry the schema in Gemini's Schema dialect.
fn gemini_batch_schema_payload(mut payload: Value) -> Value {
    if let Some(config) = payload.get_mut("generationConfig").and_then(Value::as_object_mut)
        && let Some(schema) = config.remove("responseJsonSchema")
    {
        config.insert("responseSchema".into(), gemini_response_schema(&schema));
    }
    payload
}

fn gemini_response_schema(node: &Value) -> Value {
    const JSON_SCHEMA_ONLY_KEYS: [&str; 4] = ["$schema", "$id", "additionalProperties", "strict"];
    match node {
        Value::Array(a) => Value::Array(a.iter().map(gemini_response_schema).collect()),
        Value::Object(o) => {
            let mut schema = Map::new();
            for (key, value) in o {
                if JSON_SCHEMA_ONLY_KEYS.contains(&key.as_str()) {
                    continue;
                }
                let converted = match value {
                    Value::Object(props) if key == "properties" => {
                        Value::Object(props.iter().map(|(k, v)| (k.clone(), gemini_response_schema(v))).collect())
                    }
                    _ => gemini_response_schema(value),
                };
                schema.insert(key.clone(), converted);
            }
            // `nullable_type`: ["string", "null"] becomes "string" plus nullable: true.
            let types: Vec<Value> = match schema.get("type") {
                Some(Value::Array(a)) => a.clone(),
                Some(t) => vec![t.clone()],
                None => Vec::new(),
            };
            if types.len() > 1 && types.contains(&json!("null")) {
                let first = types.into_iter().find(|t| t != &json!("null")).unwrap_or(Value::Null);
                schema.insert("type".into(), first);
                schema.insert("nullable".into(), true.into());
            }
            Value::Object(schema)
        }
        other => other.clone(),
    }
}

/// OpenAI-compatible embeddings body to an `Embedding` (`parse_embedding_response`).
fn embedding_result(body: &Value, array_input: bool) -> BatchResult {
    BatchResult::Embedding(Embedding::from_openai_body(body, !array_input))
}

// ---- HTTP ---------------------------------------------------------------------------------------

/// The batch endpoints' HTTP calls, through the shared `Connection`: GETs retry like any idempotent
/// request, while creating a batch or its input file is sent once (`mark_non_idempotent`), as in
/// RubyLLM.
struct Http {
    provider: Provider,
    connection: Connection,
}

impl Http {
    fn new(provider: Provider, config: Arc<Config>) -> Result<Http> {
        Ok(Http { provider, connection: Connection::new(provider, config)? })
    }

    async fn get(&self, path: &str) -> Result<Value> {
        Ok(self.connection.get(path, &[]).await?.body)
    }

    async fn get_text(&self, path: &str) -> Result<String> {
        let bytes = self.connection.get_bytes(path, &[]).await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value> {
        let resp = self.connection.send(reqwest::Method::POST, path, &[], false, &|req| req.json(&body)).await?;
        let text = resp.text().await.map_err(|e| Error::ConnectionFailed(e.to_string()))?;
        Ok(if text.trim().is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) })
    }

    /// The JSONL input file for an OpenAI batch, uploaded to `files` with `purpose=batch`
    /// (`Provider#upload_file` for a batch).
    async fn upload_batch_file(&self, kind: Kind, requests: &[Req]) -> Result<String> {
        let jsonl = requests
            .iter()
            .map(|r| {
                json!({ "custom_id": r.custom_id, "method": "POST", "url": openai_endpoint(kind), "body": batch_payload(r, &[]) })
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let form = || {
            let file = reqwest::multipart::Part::bytes(jsonl.clone().into_bytes())
                .file_name("ruby_llm_batch.jsonl")
                .mime_str("application/jsonl")
                .expect("static mime type");
            reqwest::multipart::Form::new().part("file", file).text("purpose", "batch")
        };
        let data = self.connection.post_multipart("files", form, &[], false).await?.body;
        str_at(&data, "id").ok_or_else(|| Error::Api("File upload returned no id".into(), None))
    }

    /// `create_batch`.
    async fn create(&self, kind: Kind, requests: &[Req]) -> Result<Attrs> {
        match kind {
            Kind::Anthropic => {
                let rows: Vec<Value> = requests.iter().map(|r| json!({ "custom_id": r.custom_id, "params": r.payload })).collect();
                Ok(anthropic_attrs(&self.post("v1/messages/batches", json!({ "requests": rows })).await?))
            }
            Kind::Responses | Kind::ChatCompletions | Kind::Embeddings => {
                single_batch_model(requests, self.provider.slug())?;
                let file_id = self.upload_batch_file(kind, requests).await?;
                let body = json!({ "input_file_id": file_id, "endpoint": openai_endpoint(kind), "completion_window": "24h" });
                Ok(openai_attrs(&self.post("batches", body).await?))
            }
            Kind::Gemini => {
                let model = single_batch_model(requests, "gemini")?;
                if requests.iter().any(|r| r.text.is_some()) {
                    return Err(Error::Api("Gemini embedding batches are not supported by rust_llm yet".into(), None));
                }
                let rows: Vec<Value> = requests
                    .iter()
                    .map(|r| {
                        let mut request = gemini_batch_schema_payload(batch_payload(r, &[]));
                        request["model"] = format!("models/{model}").into();
                        json!({ "request": request, "metadata": { "custom_id": r.custom_id } })
                    })
                    .collect();
                let body = json!({
                    "batch": {
                        "displayName": format!("ruby_llm_{}", random_hex()),
                        "inputConfig": { "requests": { "requests": rows } }
                    }
                });
                Ok(gemini_attrs(&self.post(&format!("models/{model}:batchGenerateContent"), body).await?))
            }
            Kind::Mistral => {
                let model = single_batch_model(requests, "mistral")?;
                let is_embedding = |r: &Req| r.payload.get("input").is_some();
                let endpoint = if requests.iter().all(is_embedding) {
                    "/v1/embeddings"
                } else if requests.iter().any(is_embedding) {
                    return Err(Error::Api("Mistral batches cannot mix chat and embedding requests".into(), None));
                } else {
                    "/v1/chat/completions"
                };
                let rows: Vec<Value> = requests
                    .iter()
                    .map(|r| {
                        let body = batch_payload(r, &["model"]);
                        let custom_id =
                            if body.get("input").is_some_and(Value::is_array) { format!("{}:array", r.custom_id) } else { r.custom_id.clone() };
                        json!({ "custom_id": custom_id, "body": body })
                    })
                    .collect();
                let body = json!({ "endpoint": endpoint, "model": model, "requests": rows });
                Ok(mistral_attrs(&self.post("batch/jobs", body).await?))
            }
            Kind::XAI => {
                let batch = self.post("batches", json!({ "name": format!("ruby_llm_{}", random_hex()) })).await?;
                let id = str_at(&batch, "batch_id")
                    .or_else(|| str_at(&batch, "id"))
                    .ok_or_else(|| Error::Api("xAI returned no batch id".into(), None))?;
                let rows: Vec<Value> = requests
                    .iter()
                    .map(|r| {
                        let payload = batch_payload(r, &[]);
                        let kind = if payload.get("input").is_some() { "responses" } else { "chat_get_completion" };
                        let mut request = Map::new();
                        request.insert(kind.into(), payload);
                        json!({ "batch_request_id": r.custom_id, "batch_request": request })
                    })
                    .collect();
                self.post(&format!("batches/{id}/requests"), json!({ "batch_requests": rows })).await?;
                self.find(kind, &id).await
            }
        }
    }

    /// `find_batch`.
    async fn find(&self, kind: Kind, id: &str) -> Result<Attrs> {
        match kind {
            Kind::Anthropic => Ok(anthropic_attrs(&self.get(&format!("v1/messages/batches/{id}")).await?)),
            k if k.is_openai() => Ok(openai_attrs(&self.get(&format!("batches/{id}")).await?)),
            Kind::Gemini => Ok(gemini_attrs(&self.get(&gemini_batch_name(id)).await?)),
            // A just-created Mistral job can 404 for a moment; retry that twice.
            Kind::Mistral => {
                let mut attempts = 0;
                loop {
                    match self.get(&format!("batch/jobs/{id}")).await {
                        Ok(data) => return Ok(mistral_attrs(&data)),
                        Err(e) => {
                            attempts += 1;
                            if !(e.response().is_some_and(|r| r.status == 404) && attempts < 3) {
                                return Err(e);
                            }
                            tokio::time::sleep(Duration::from_secs_f64(0.5 * attempts as f64)).await;
                        }
                    }
                }
            }
            _ => Ok(xai_attrs(&self.get(&format!("batches/{id}")).await?)),
        }
    }

    /// `cancel_batch`.
    async fn cancel(&self, kind: Kind, id: &str) -> Result<Attrs> {
        match kind {
            Kind::Anthropic => Ok(anthropic_attrs(&self.post(&format!("v1/messages/batches/{id}/cancel"), json!({})).await?)),
            k if k.is_openai() => Ok(openai_attrs(&self.post(&format!("batches/{id}/cancel"), json!({})).await?)),
            Kind::Gemini => {
                self.post(&format!("{}:cancel", gemini_batch_name(id)), json!({})).await?;
                self.find(kind, id).await
            }
            Kind::Mistral => Ok(mistral_attrs(&self.post(&format!("batch/jobs/{id}/cancel"), json!({})).await?)),
            _ => Ok(xai_attrs(&self.post(&format!("batches/{id}:cancel"), json!({})).await?)),
        }
    }

    /// `batch_results`: `[index, result, failure_status]` rows, in no particular order.
    async fn results(&self, kind: Kind, id: &str) -> Result<Vec<Row>> {
        match kind {
            Kind::Anthropic => {
                let text = self.get_text(&format!("v1/messages/batches/{id}/results")).await?;
                let mut rows = Vec::new();
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let line: Value = serde_json::from_str(line)?;
                    let custom_id = str_at(&line, "custom_id").unwrap_or_default();
                    let index = batch_result_index(&custom_id)?;
                    let result = line.get("result").cloned().unwrap_or(Value::Null);
                    let result_type = str_at(&result, "type").unwrap_or_default();
                    if result_type == "succeeded" {
                        let body = result.get("message").cloned().unwrap_or(Value::Null);
                        let message = anthropic::parse_completion_body(&body, raw(&body))?;
                        rows.push((index, Some(BatchResult::Message(message)), BatchStatus::Failed));
                    } else {
                        let detail = result.pointer("/error/error/message").and_then(Value::as_str).map(str::to_string);
                        rows.push((index, None, batch_failure(&custom_id, detail, &result_type)));
                    }
                }
                Ok(rows)
            }
            k if k.is_openai() => {
                let batch = self.get(&format!("batches/{id}")).await?;
                let mut rows = Vec::new();
                for key in ["output_file_id", "error_file_id"] {
                    let Some(file_id) = batch.get(key).and_then(Value::as_str).filter(|f| !f.is_empty()) else { continue };
                    let text = self.get_text(&format!("files/{file_id}/content")).await?;
                    for line in text.lines().filter(|l| !l.trim().is_empty()) {
                        let line: Value = serde_json::from_str(line)?;
                        rows.push(self.openai_row(kind, &line)?);
                    }
                }
                Ok(rows)
            }
            Kind::Gemini => {
                let body = self.get(&gemini_batch_name(id)).await?;
                let batch = body.get("metadata").unwrap_or(&body);
                if batch.get("@type").and_then(Value::as_str).is_some_and(|t| t.ends_with(".EmbedContentBatch")) {
                    return Err(Error::Api("Gemini embedding batches are not supported by rust_llm yet".into(), None));
                }
                let model_id = batch.get("model").and_then(Value::as_str).unwrap_or_default().trim_start_matches("models/");
                let model = Model::default_for(model_id, "gemini");
                let inlined = ["/response/inlinedResponses/inlinedResponses", "/output/inlinedResponses/inlinedResponses", "/metadata/output/inlinedResponses/inlinedResponses"]
                    .iter()
                    .find_map(|p| body.pointer(p).and_then(Value::as_array))
                    .cloned()
                    .unwrap_or_default();
                let mut rows = Vec::new();
                for (position, inline) in inlined.iter().enumerate() {
                    let key = inline.pointer("/metadata/custom_id").or_else(|| inline.pointer("/metadata/key")).and_then(Value::as_str);
                    let index = match key {
                        Some(k) => batch_result_index(k)?,
                        None => position as i64,
                    };
                    if let Some(response) = inline.get("response") {
                        let message = gemini::parse_completion_body(&model, response, raw(response))?;
                        rows.push((index, Some(BatchResult::Message(message)), BatchStatus::Failed));
                    } else {
                        let detail = inline.pointer("/error/message").and_then(Value::as_str).map(str::to_string);
                        let label = key.map(str::to_string).unwrap_or_else(|| index.to_string());
                        rows.push((index, None, batch_failure(&label, detail, "failed")));
                    }
                }
                Ok(rows)
            }
            Kind::Mistral => {
                let data = self.get(&format!("batch/jobs/{id}?inline=true")).await?;
                let outputs = data.get("outputs").and_then(Value::as_array).cloned().unwrap_or_default();
                let mut rows = Vec::new();
                for line in &outputs {
                    let full_id = str_at(line, "custom_id").unwrap_or_default();
                    let (custom_id, shape) = full_id.split_once(':').map_or((full_id.as_str(), None), |(a, b)| (a, Some(b)));
                    let index = batch_result_index(custom_id)?;
                    match line.pointer("/response/body").filter(|b| !b.is_null()) {
                        Some(body) if body.get("data").is_some_and(Value::is_array) => {
                            rows.push((index, Some(embedding_result(body, shape == Some("array"))), BatchStatus::Failed));
                        }
                        Some(body) => {
                            let message = chat_completions::parse_completion_body(self.provider, body, raw(body))?;
                            rows.push((index, Some(BatchResult::Message(message)), BatchStatus::Failed));
                        }
                        None => rows.push((index, None, batch_failure(&full_id, batch_error_message(line), "failed"))),
                    }
                }
                Ok(rows)
            }
            _ => {
                let mut rows = Vec::new();
                let mut token: Option<String> = None;
                loop {
                    let mut path = format!("batches/{id}/results?limit=100");
                    if let Some(t) = &token {
                        path.push_str(&format!("&pagination_token={t}"));
                    }
                    let response = self.get(&path).await?;
                    let page = response.get("results").or_else(|| response.get("batch_results")).and_then(Value::as_array).cloned().unwrap_or_default();
                    for result in &page {
                        let request_id = str_at(result, "batch_request_id").or_else(|| str_at(result, "custom_id")).unwrap_or_default();
                        let index = batch_result_index(&request_id)?;
                        let body = result
                            .pointer("/batch_result/response/chat_get_completion")
                            .or_else(|| result.pointer("/response/chat_get_completion"));
                        match body {
                            Some(body) => {
                                let message = chat_completions::parse_completion_body(self.provider, body, raw(body))?;
                                rows.push((index, Some(BatchResult::Message(message)), BatchStatus::Failed));
                            }
                            None => rows.push((index, None, batch_failure(&request_id, batch_error_message(result), "failed"))),
                        }
                    }
                    token = str_at(&response, "pagination_token").or_else(|| str_at(&response, "next_page_token"));
                    if token.is_none() {
                        break;
                    }
                }
                Ok(rows)
            }
        }
    }

    fn openai_row(&self, kind: Kind, line: &Value) -> Result<Row> {
        let custom_id = str_at(line, "custom_id").unwrap_or_default();
        let index = batch_result_index(&custom_id)?;
        let response = line.get("response").filter(|r| !r.is_null());
        let ok = response.and_then(|r| r.get("status_code")).and_then(Value::as_i64).is_some_and(|s| (200..=299).contains(&s));
        let body = response.and_then(|r| r.get("body"));
        match body {
            Some(body) if ok => {
                let result = match kind {
                    Kind::Embeddings => embedding_result(body, false),
                    Kind::ChatCompletions => BatchResult::Message(chat_completions::parse_completion_body(self.provider, body, raw(body))?),
                    _ => BatchResult::Message(responses::parse_completion_body(self.provider, body, raw(body))?),
                };
                Ok((index, Some(result), BatchStatus::Failed))
            }
            _ => Ok((index, None, batch_failure(&custom_id, batch_error_message(line), "failed"))),
        }
    }
}

fn gemini_batch_name(id: &str) -> String {
    if id.starts_with("batches/") { id.to_string() } else { format!("batches/{id}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_schema_drops_json_schema_only_keys_and_folds_nullable_types() {
        let payload = json!({ "generationConfig": { "responseJsonSchema": {
            "type": "object", "additionalProperties": false, "strict": true,
            "properties": { "name": { "type": ["string", "null"] } }
        } } });
        let out = gemini_batch_schema_payload(payload);
        assert_eq!(
            out["generationConfig"],
            json!({ "responseSchema": { "type": "object", "properties": { "name": { "type": "string", "nullable": true } } } })
        );
    }

    #[test]
    fn a_cancelled_request_type_is_normalized_to_cancelled() {
        assert_eq!(batch_failure("0", None, "canceled"), BatchStatus::Cancelled);
        assert_eq!(batch_failure("0", Some("boom".into()), "errored"), BatchStatus::Failed);
    }
}
