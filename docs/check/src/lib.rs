//! Compiles every ```rust block in `docs/*.md` as a doctest (`cargo test -p rust_llm_docs --doc`).
//! Samples define `async fn`s that are never called, so the doctests compile the code without
//! touching the network.

#[doc = include_str!("../../getting-started.md")]
pub mod getting_started {}
#[doc = include_str!("../../configuration.md")]
pub mod configuration {}
#[doc = include_str!("../../chat.md")]
pub mod chat {}
#[doc = include_str!("../../tools.md")]
pub mod tools {}
#[doc = include_str!("../../agents.md")]
pub mod agents {}
#[doc = include_str!("../../streaming.md")]
pub mod streaming {}
#[doc = include_str!("../../structured-output.md")]
pub mod structured_output {}
#[doc = include_str!("../../thinking.md")]
pub mod thinking {}
#[doc = include_str!("../../attachments-and-files.md")]
pub mod attachments_and_files {}
#[doc = include_str!("../../embeddings.md")]
pub mod embeddings {}
#[doc = include_str!("../../images.md")]
pub mod images {}
#[doc = include_str!("../../batches.md")]
pub mod batches {}
#[doc = include_str!("../../mcp.md")]
pub mod mcp {}
#[doc = include_str!("../../judgments.md")]
pub mod judgments {}
#[doc = include_str!("../../persistence-loco.md")]
pub mod persistence_loco {}
#[doc = include_str!("../../generators.md")]
pub mod generators {}
#[doc = include_str!("../../errors-and-retries.md")]
pub mod errors_and_retries {}
#[doc = include_str!("../../cost-and-usage.md")]
pub mod cost_and_usage {}
#[doc = include_str!("../../migrating-from-rubyllm.md")]
pub mod migrating_from_rubyllm {}
#[doc = include_str!("../../instrumentation.md")]
pub mod instrumentation {}
#[doc = include_str!("../../prompts.md")]
pub mod prompts {}
