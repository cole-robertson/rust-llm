//! Port of `lib/ruby_llm/mcp/resource.rb`: a file, a record, or any data the server names with a
//! URI. Pass one to a chat as an attachment, or save it.

use std::path::Path;

use base64::Engine;
use serde_json::Value;

use super::{Mcp, content};
use crate::attachment::Attachment;
use crate::error::Result;

/// A resource's content: text for text resources, bytes for binary ones.
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceContent {
    Text(String),
    Blob(Vec<u8>),
}

impl ResourceContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ResourceContent::Text(text) => Some(text),
            ResourceContent::Blob(_) => None,
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            ResourceContent::Text(text) => text.into_bytes(),
            ResourceContent::Blob(bytes) => bytes,
        }
    }
}

/// `RubyLLM::MCP::Resource`. Resources from `Mcp::resources` are read from the server the first
/// time their content is needed.
#[derive(Clone)]
pub struct Resource {
    pub uri: String,
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    data: Value,
    mcp: Mcp,
}

impl std::fmt::Debug for Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resource")
            .field("uri", &self.uri)
            .field("name", &self.name)
            .field("mime_type", &self.mime_type)
            .finish()
    }
}

impl Resource {
    pub(crate) fn new(mcp: Mcp, data: Value) -> Resource {
        let s = |key: &str| data.get(key).and_then(Value::as_str).map(str::to_string);
        let uri = s("uri").unwrap_or_default();
        Resource {
            name: s("name").unwrap_or_else(|| content::filename(&uri)),
            title: s("title"),
            description: s("description"),
            mime_type: s("mimeType"),
            uri,
            data,
            mcp,
        }
    }

    /// `content`: embedded content, or read from the server.
    pub async fn content(&self) -> Result<ResourceContent> {
        if let Some(text) = self.data.get("text").and_then(Value::as_str) {
            return Ok(ResourceContent::Text(text.to_string()));
        }
        if let Some(blob) = self.data.get("blob").and_then(Value::as_str) {
            return Ok(ResourceContent::Blob(
                base64::engine::general_purpose::STANDARD
                    .decode(blob)
                    .unwrap_or_default(),
            ));
        }
        let read = self.mcp.resource(&self.uri).await?;
        if read.data.get("text").is_none() && read.data.get("blob").is_none() {
            return Err(super::McpError::new(format!(
                "{} returned no content for {}",
                self.mcp.name(),
                self.uri
            ))
            .into());
        }
        Box::pin(read.content()).await
    }

    /// `to_blob`: the content as bytes.
    pub async fn to_blob(&self) -> Result<Vec<u8>> {
        Ok(self.content().await?.into_bytes())
    }

    /// `save(path)`: writes the content to `path`.
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_blob().await?)?;
        Ok(())
    }

    /// `to_attachment`: how chats take a resource (`ask_with(msg, vec![resource.to_attachment()])`).
    pub async fn to_attachment(&self) -> Result<Attachment> {
        let filename = content::filename(&self.uri);
        let filename = if filename.is_empty() {
            self.name.clone()
        } else {
            filename
        };
        Ok(Attachment::from_bytes(
            self.to_blob().await?,
            filename,
            None,
        ))
    }
}
