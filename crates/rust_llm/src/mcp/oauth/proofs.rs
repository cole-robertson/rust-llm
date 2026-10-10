//! Port of `lib/ruby_llm/mcp/oauth/proofs.rb`: signs DPoP proofs (RFC 9449 section 4.2) with the
//! key a token is bound to, carrying the latest nonce each server supplied. Nonces from the
//! authorization server and the resource server stay apart, as section 9 requires.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use reqwest::Url;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::key::Key;
use super::{lock, now};
use crate::error::Result;

type Origin = (String, Option<String>, Option<u16>);

/// `MCP::OAuth::Proofs`.
#[derive(Default)]
pub(crate) struct Proofs {
    nonces: Mutex<HashMap<Origin, String>>,
    keys: Mutex<HashMap<String, Arc<Key>>>,
}

fn origin(url: &str) -> Origin {
    match Url::parse(url) {
        Ok(uri) => (
            uri.scheme().to_string(),
            uri.host_str().map(str::to_string),
            uri.port_or_known_default(),
        ),
        Err(_) => (String::new(), None, None),
    }
}

impl Proofs {
    /// `sign(pem, url, token:, verb:)`: a proof for a `verb` request to `url` with the
    /// PEM-encoded key, and with the hash of the access `token` it accompanies.
    pub(crate) fn sign(
        &self,
        pem: &str,
        url: &str,
        token: Option<&str>,
        verb: &str,
    ) -> Result<String> {
        let cached = lock(&self.keys).get(pem).cloned();
        let key = match cached {
            Some(key) => key,
            None => {
                let key = Arc::new(Key::new(pem)?);
                lock(&self.keys).insert(pem.to_string(), key.clone());
                key
            }
        };
        let htu = url.split(['?', '#']).next().unwrap_or(url);
        let mut claims = Map::new();
        claims.insert("jti".into(), uuid::Uuid::new_v4().to_string().into());
        claims.insert("htm".into(), verb.into());
        claims.insert("htu".into(), htu.into());
        claims.insert("iat".into(), now().into());
        if let Some(token) = token {
            let ath = URL_SAFE_NO_PAD.encode(Sha256::digest(token.as_bytes()));
            claims.insert("ath".into(), ath.into());
        }
        if let Some(nonce) = lock(&self.nonces).get(&origin(url)) {
            claims.insert("nonce".into(), nonce.clone().into());
        }
        let mut header = Map::new();
        header.insert("typ".into(), "dpop+jwt".into());
        header.insert("jwk".into(), key.jwk().unwrap_or(Value::Null));
        key.jwt(&Value::Object(claims), key.algorithm(&[]), header)
    }

    /// `remember(url, nonce)`.
    pub(crate) fn remember(&self, url: &str, nonce: Option<&str>) {
        if let Some(nonce) = nonce {
            lock(&self.nonces).insert(origin(url), nonce.to_string());
        }
    }
}
