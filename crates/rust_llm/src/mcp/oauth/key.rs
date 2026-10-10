//! Port of `lib/ruby_llm/mcp/oauth/key.rb`: a private key that signs JSON Web Tokens with JWS
//! (RFC 7515): the client assertions of `private_key_jwt` (RFC 7523 section 2.2) and DPoP proofs
//! (RFC 9449 section 4.2). Takes an RSA or elliptic curve key as a PEM string: PKCS#8
//! (`PRIVATE KEY`, what `private_to_pem` writes) for both, or PKCS#1 (`RSA PRIVATE KEY`).
//!
//! Signing uses `ring`, which signs on P-256 and P-384 but not P-521, so a P-521 key is refused
//! like any other unsupported key; generated keys are P-256, as in Ruby.

use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use ring::rand::SystemRandom;
use ring::signature::{self, EcdsaKeyPair, KeyPair, RsaKeyPair};
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};

const RSA_ALGORITHMS: &[&str] = &["RS256", "PS256", "RS384", "PS384", "RS512", "PS512"];

enum Pair {
    Ec {
        pair: EcdsaKeyPair,
        crv: &'static str,
        alg: &'static str,
    },
    Rsa(RsaKeyPair),
}

/// `MCP::OAuth::Key`.
pub(crate) struct Key {
    pair: Pair,
    pem: String,
}

fn invalid() -> Error {
    Error::Argument("OAuth private keys must be RSA or EC keys on P-256, P-384, or P-521".into())
}

fn encode(data: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(data)
}

/// The label and DER bytes of the first PEM block.
fn decode_pem(pem: &str) -> Option<(String, Vec<u8>)> {
    let start = pem.find("-----BEGIN ")? + "-----BEGIN ".len();
    let label_end = start + pem[start..].find("-----")?;
    let label = pem[start..label_end].to_string();
    let body_start = label_end + "-----".len();
    let body_end = body_start + pem[body_start..].find("-----END ")?;
    let body: String = pem[body_start..body_end]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    Some((label, STANDARD.decode(body).ok()?))
}

/// `private_to_pem`: PKCS#8 DER as a `PRIVATE KEY` PEM block, 64 characters a line.
fn encode_pem(der: &[u8]) -> String {
    let body = STANDARD.encode(der);
    let lines: Vec<&str> = body
        .as_bytes()
        .chunks(64)
        .map(|line| std::str::from_utf8(line).unwrap_or(""))
        .collect();
    format!(
        "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n",
        lines.join("\n")
    )
}

impl Key {
    /// `Key.generate`: a new P-256 key.
    pub(crate) fn generate() -> Result<Key> {
        let document = EcdsaKeyPair::generate_pkcs8(
            &signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            &SystemRandom::new(),
        )
        .map_err(|_| Error::Configuration("Could not generate a DPoP key".into()))?;
        Key::new(&encode_pem(document.as_ref()))
    }

    /// `Key.new(pem)`: raises `Error::Argument` for anything but an RSA key or an EC key on a
    /// supported curve.
    pub(crate) fn new(pem: &str) -> Result<Key> {
        let (label, der) = decode_pem(pem).ok_or_else(invalid)?;
        let rng = SystemRandom::new();
        let pair = match label.as_str() {
            "PRIVATE KEY" => {
                let ec = |algorithm, crv, alg| {
                    EcdsaKeyPair::from_pkcs8(algorithm, &der, &rng)
                        .ok()
                        .map(|pair| Pair::Ec { pair, crv, alg })
                };
                ec(
                    &signature::ECDSA_P256_SHA256_FIXED_SIGNING,
                    "P-256",
                    "ES256",
                )
                .or_else(|| {
                    ec(
                        &signature::ECDSA_P384_SHA384_FIXED_SIGNING,
                        "P-384",
                        "ES384",
                    )
                })
                .or_else(|| RsaKeyPair::from_pkcs8(&der).ok().map(Pair::Rsa))
            }
            "RSA PRIVATE KEY" => RsaKeyPair::from_der(&der).ok().map(Pair::Rsa),
            _ => None,
        }
        .ok_or_else(invalid)?;
        Ok(Key {
            pair,
            pem: pem.to_string(),
        })
    }

    /// `algorithm(supported)`: the JWS algorithm this key signs with, preferring one of
    /// `supported` when the authorization server lists them.
    pub(crate) fn algorithm(&self, supported: &[String]) -> &'static str {
        let candidates: &[&'static str] = match &self.pair {
            Pair::Rsa(_) => RSA_ALGORITHMS,
            Pair::Ec { alg, .. } => std::slice::from_ref(alg),
        };
        candidates
            .iter()
            .find(|c| supported.iter().any(|s| s == *c))
            .or(candidates.first())
            .copied()
            .unwrap_or("ES256")
    }

    /// `jwt(claims, algorithm:, **header)`.
    pub(crate) fn jwt(
        &self,
        claims: &Value,
        algorithm: &str,
        mut header: Map<String, Value>,
    ) -> Result<String> {
        header.insert("alg".into(), algorithm.into());
        let input = format!(
            "{}.{}",
            encode(Value::Object(header).to_string().as_bytes()),
            encode(claims.to_string().as_bytes())
        );
        let signature = self.sign(input.as_bytes(), algorithm)?;
        Ok(format!("{input}.{}", encode(&signature)))
    }

    /// `jwk`: the public half of an elliptic curve key as a JSON Web Key (RFC 7518 section
    /// 6.2.1). `None` for RSA keys, which never bind tokens.
    pub(crate) fn jwk(&self) -> Option<Value> {
        let Pair::Ec { pair, crv, .. } = &self.pair else {
            return None;
        };
        let point = pair.public_key().as_ref();
        let size = (point.len() - 1) / 2;
        Some(json!({
            "kty": "EC", "crv": crv, "x": encode(&point[1..=size]), "y": encode(&point[size + 1..])
        }))
    }

    /// `to_pem`.
    pub(crate) fn to_pem(&self) -> &str {
        &self.pem
    }

    /// JWS carries ECDSA signatures as R and S side by side (RFC 7518 section 3.4), which is
    /// what ring's fixed-length signing algorithms return.
    fn sign(&self, input: &[u8], algorithm: &str) -> Result<Vec<u8>> {
        let rng = SystemRandom::new();
        let failed = || Error::Argument(format!("Could not sign with {algorithm}"));
        match &self.pair {
            Pair::Ec { pair, alg, .. } if *alg == algorithm => pair
                .sign(&rng, input)
                .map(|s| s.as_ref().to_vec())
                .map_err(|_| failed()),
            Pair::Rsa(pair) => {
                let encoding: &'static dyn signature::RsaEncoding = match algorithm {
                    "RS256" => &signature::RSA_PKCS1_SHA256,
                    "RS384" => &signature::RSA_PKCS1_SHA384,
                    "RS512" => &signature::RSA_PKCS1_SHA512,
                    "PS256" => &signature::RSA_PSS_SHA256,
                    "PS384" => &signature::RSA_PSS_SHA384,
                    "PS512" => &signature::RSA_PSS_SHA512,
                    _ => return Err(failed()),
                };
                let mut signature = vec![0; pair.public().modulus_len()];
                pair.sign(encoding, &rng, input, &mut signature)
                    .map_err(|_| failed())?;
                Ok(signature)
            }
            Pair::Ec { .. } => Err(failed()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_a_p256_key_that_reads_back_from_its_pem() {
        let key = Key::generate().unwrap();
        assert!(key.to_pem().starts_with("-----BEGIN PRIVATE KEY-----"));
        let again = Key::new(key.to_pem()).unwrap();
        assert_eq!(again.jwk(), key.jwk());
        assert_eq!(key.algorithm(&["RS256".into()]), "ES256");
        assert_eq!(key.jwk().unwrap()["crv"], "P-256");
    }

    #[test]
    fn refuses_what_is_not_a_supported_private_key() {
        assert!(matches!(Key::new("not a key"), Err(Error::Argument(_))));
    }
}
