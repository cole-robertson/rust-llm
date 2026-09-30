//! Port of `lib/ruby_llm/context.rb` and `RubyLLM.context`: an isolated configuration scope with
//! the same entry points as the crate root, reading its own `Config` copy instead of the global.
//!
//! ```ruby
//! ctx = RubyLLM.context { |config| config.openai_api_key = ENV.fetch("TENANT_OPENAI_API_KEY") }
//! ctx.chat.ask "Explain Ruby blocks."
//! ```

use std::sync::Arc;

use serde_json::Value;

use crate::attachment::Attachment;
use crate::chat::Chat;
use crate::config::Config;
use crate::embedding::{EmbedInput, EmbedOptions, Embedding};
use crate::error::Result;
use crate::files::{DownloadedFile, FileOptions, UploadOptions, UploadedFile};
use crate::image::{Images, PaintOptions};
use crate::judge::{Judge, JudgeOptions, Judgment};
use crate::moderation::{ModerateOptions, Moderation, ModerationInput};
use crate::ocr::{Ocr, OcrOptions};
use crate::rerank::{Rerank, RerankOptions};
use crate::speech::{SpeakOptions, Speech, SpeechChunk};
use crate::tokenization::{Tokenization, TokenizeOptions};
use crate::video::{AnimateOptions, Video, VideoJob};

/// `RubyLLM::Context`. The global configuration is left untouched.
#[derive(Debug, Clone)]
pub struct Context {
    config: Arc<Config>,
}

/// `RubyLLM.context { |config| ... }`: a copy of the global configuration, changed by `f`.
pub fn context(f: impl FnOnce(&mut Config)) -> Context {
    let mut config = (*crate::config()).clone();
    f(&mut config);
    Context::new(config)
}

impl Context {
    pub fn new(config: Config) -> Context {
        Context {
            config: Arc::new(config),
        }
    }

    /// The context's configuration.
    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }

    /// `ctx.chat(model:, provider:)`.
    pub fn chat(&self, model: Option<&str>, provider: Option<&str>) -> Result<Chat> {
        Chat::with_config(self.config.clone(), model, provider, false)
    }

    /// `ctx.count_tokens(text, model:, provider:)`.
    pub async fn count_tokens(
        &self,
        text: &str,
        model: Option<&str>,
        provider: Option<&str>,
    ) -> Result<i64> {
        self.chat(model, provider)?.count_tokens(Some(text)).await
    }

    /// `ctx.tokenize(text, ...)`.
    pub async fn tokenize(&self, text: &str, options: TokenizeOptions<'_>) -> Result<Tokenization> {
        crate::tokenization::tokenize(
            text,
            TokenizeOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.embed(text, ...)`.
    pub async fn embed(
        &self,
        input: impl Into<EmbedInput>,
        options: EmbedOptions<'_>,
    ) -> Result<Embedding> {
        crate::embedding::embed(
            input,
            EmbedOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.embed_later(text, model:, provider:, dimensions:)`: an embedding request staged for a
    /// batch, carrying this context's configuration.
    pub fn embed_later(
        &self,
        text: impl Into<EmbedInput>,
        options: EmbedOptions<'_>,
    ) -> Result<crate::batch::EmbeddingRequest> {
        crate::batch::EmbeddingRequest::new(
            text,
            EmbedOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
    }

    /// `ctx.mcp(url:, ...)`: `MCP.define(...).new(context: self)`, a builder that connects with
    /// this context's configuration.
    pub fn mcp(&self, builder: crate::mcp::McpBuilder) -> crate::mcp::McpBuilder {
        builder.config(self.config.clone())
    }

    /// `ctx.paint(prompt, ...)`.
    pub async fn paint(&self, prompt: &str, options: PaintOptions<'_>) -> Result<Images> {
        crate::image::paint(
            prompt,
            PaintOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.animate(prompt, ...)`.
    pub async fn animate(
        &self,
        prompt: Option<&str>,
        options: AnimateOptions<'_>,
    ) -> Result<Video> {
        crate::video::animate(
            prompt,
            AnimateOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.animate_later(prompt, ...)`.
    pub async fn animate_later(
        &self,
        prompt: Option<&str>,
        options: AnimateOptions<'_>,
    ) -> Result<VideoJob> {
        crate::video::animate_later(
            prompt,
            AnimateOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.speak(input, ...)`.
    pub async fn speak(&self, input: &str, options: SpeakOptions<'_>) -> Result<Speech> {
        crate::speech::speak(
            input,
            SpeakOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.speak(input, ...) { |chunk| ... }`.
    pub async fn speak_stream(
        &self,
        input: &str,
        options: SpeakOptions<'_>,
        on_chunk: impl FnMut(&SpeechChunk) + Send,
    ) -> Result<Speech> {
        crate::speech::speak_stream(
            input,
            SpeakOptions {
                config: Some(self.config.clone()),
                ..options
            },
            on_chunk,
        )
        .await
    }

    /// `ctx.moderate(input, ...)`.
    pub async fn moderate(
        &self,
        input: impl Into<ModerationInput>,
        options: ModerateOptions<'_>,
    ) -> Result<Moderation> {
        crate::moderation::moderate(
            input,
            ModerateOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.judge(input, questions:, ...)`.
    pub async fn judge(
        &self,
        input: impl Into<Value>,
        questions: Value,
        options: JudgeOptions,
    ) -> Result<Judgment> {
        let Value::Object(questions) = questions else {
            return Err(crate::Error::Argument("Questions must be a Hash".into()));
        };
        Judge::new()
            .with_config(self.config.clone())
            .judge_with(
                input,
                JudgeOptions {
                    questions,
                    ..options
                },
            )
            .await
    }

    /// `ctx.ocr(file, ...)`.
    pub async fn ocr(&self, file: impl Into<Attachment>, options: OcrOptions<'_>) -> Result<Ocr> {
        crate::ocr::ocr(
            file,
            OcrOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.rerank(query, documents, model:, ...)`.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        model: &str,
        options: RerankOptions<'_>,
    ) -> Result<Rerank> {
        crate::rerank::rerank(
            query,
            documents,
            model,
            RerankOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.upload(file, ...)`: `UploadedFile.upload(..., context: self)`. Without `provider`,
    /// the provider of this context's default model is used.
    pub async fn upload(
        &self,
        file: impl Into<Attachment>,
        options: UploadOptions<'_>,
    ) -> Result<UploadedFile> {
        UploadedFile::upload(
            file,
            UploadOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }

    /// `ctx.download(id, ...)`: `UploadedFile.download(..., context: self)`.
    pub async fn download(&self, id: &str, options: FileOptions<'_>) -> Result<DownloadedFile> {
        UploadedFile::download(
            id,
            FileOptions {
                config: Some(self.config.clone()),
                ..options
            },
        )
        .await
    }
}
