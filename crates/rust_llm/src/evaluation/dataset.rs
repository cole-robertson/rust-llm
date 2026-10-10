//! Port of `lib/ruby_llm/evaluation/dataset.rb`: datasets from files, data, or code.

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::Value;

use super::Case;
use crate::config::Config;
use crate::error::{Error, Result};

/// Where an evaluation's cases come from (`dataset path | enumerable | { block }`).
#[derive(Clone)]
pub enum Dataset {
    /// A `.yml`, `.yaml`, `.json`, or `.jsonl` file.
    Path(PathBuf),
    /// Cases built in code.
    Cases(Vec<Case>),
    /// Rows as data: an array of case hashes, or `{ "cases": [...] }`.
    Data(Value),
    /// `dataset { ... }`: computed once per run.
    Block(Arc<dyn Fn() -> Result<Dataset> + Send + Sync>),
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dataset::Path(p) => f.debug_tuple("Path").field(p).finish(),
            Dataset::Cases(c) => f.debug_tuple("Cases").field(c).finish(),
            Dataset::Data(v) => f.debug_tuple("Data").field(v).finish(),
            Dataset::Block(_) => f.write_str("Block"),
        }
    }
}

impl Dataset {
    /// `dataset { ... }`.
    pub fn block(f: impl Fn() -> Result<Dataset> + Send + Sync + 'static) -> Dataset {
        Dataset::Block(Arc::new(f))
    }
}

impl From<Vec<Case>> for Dataset {
    fn from(cases: Vec<Case>) -> Dataset {
        Dataset::Cases(cases)
    }
}

impl From<&[Case]> for Dataset {
    fn from(cases: &[Case]) -> Dataset {
        Dataset::Cases(cases.to_vec())
    }
}

impl From<PathBuf> for Dataset {
    fn from(path: PathBuf) -> Dataset {
        Dataset::Path(path)
    }
}

impl From<&Path> for Dataset {
    fn from(path: &Path) -> Dataset {
        Dataset::Path(path.to_path_buf())
    }
}

impl From<&str> for Dataset {
    fn from(path: &str) -> Dataset {
        Dataset::Path(PathBuf::from(path))
    }
}

impl From<Value> for Dataset {
    fn from(data: Value) -> Dataset {
        Dataset::Data(data)
    }
}

/// `Dataset.load(source, name:)`.
pub(crate) fn load(source: Option<&Dataset>, name: Option<&str>, config: &Config) -> Result<Vec<Case>> {
    let cases = rows(source.cloned(), name, config)?;
    if cases.is_empty() {
        return Err(Error::Argument("A dataset cannot be empty".into()));
    }
    let mut names: Vec<&str> = cases.iter().map(Case::name).collect();
    names.sort_unstable();
    names.dedup();
    if names.len() != cases.len() {
        return Err(Error::Argument("Dataset case names must be unique".into()));
    }
    Ok(cases)
}

fn rows(source: Option<Dataset>, name: Option<&str>, config: &Config) -> Result<Vec<Case>> {
    let mut source = match source {
        Some(Dataset::Block(f)) => Some(f()?),
        other => other,
    };
    if source.is_none() {
        source = Some(Dataset::Path(discover(name, config)?));
    }
    let data = match source {
        Some(Dataset::Cases(cases)) => return Ok(cases),
        Some(Dataset::Path(path)) => read(&path)?,
        Some(Dataset::Data(data)) => data,
        Some(Dataset::Block(_)) | None => {
            return Err(Error::Argument(
                "A dataset must contain enumerable cases".into(),
            ));
        }
    };
    let data = match data {
        Value::Object(h) => cases_from_hash(h)?,
        other => other,
    };
    let Value::Array(rows) = data else {
        return Err(Error::Argument(
            "A dataset must contain enumerable cases".into(),
        ));
    };
    rows.iter().map(Case::from_value).collect()
}

fn cases_from_hash(source: serde_json::Map<String, Value>) -> Result<Value> {
    let unknown: Vec<&str> = source
        .keys()
        .map(String::as_str)
        .filter(|k| !["name", "cases"].contains(k))
        .collect();
    if !unknown.is_empty() {
        return Err(Error::Argument(format!(
            "Unsupported dataset fields: {}; declare evaluators in Rust",
            unknown.join(", ")
        )));
    }
    source
        .get("cases")
        .cloned()
        .ok_or_else(|| Error::Argument("key not found: \"cases\"".into()))
}

static ACRONYM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([A-Z]+)([A-Z][a-z])").expect("valid regex"));
static CAMEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([a-z\d])([A-Z])").expect("valid regex"));

/// `Dataset.discover(name)`: `app/evals/<name underscored>.{yml,yaml,json,jsonl}` beside the
/// prompt root (`Prompt.root.parent.join("evals")`).
fn discover(name: Option<&str>, config: &Config) -> Result<PathBuf> {
    let Some(name) = name else {
        return Err(Error::Argument(
            "An anonymous evaluation needs an explicit dataset".into(),
        ));
    };
    let filename = name.replace("::", "/");
    let filename = ACRONYM.replace_all(&filename, "${1}_${2}");
    let filename = CAMEL.replace_all(&filename, "${1}_${2}").to_lowercase();
    let prompts = crate::prompt::roots(config)
        .into_iter()
        .next()
        .unwrap_or_else(crate::prompt::root);
    let root = prompts
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
        .join("evals");
    let paths: Vec<PathBuf> = ["yml", "yaml", "json", "jsonl"]
        .iter()
        .map(|ext| root.join(format!("{filename}.{ext}")))
        .filter(|p| p.is_file())
        .collect();
    match paths.as_slice() {
        [] => Err(Error::Argument(format!(
            "Dataset not found: {}.{{yml,yaml,json,jsonl}}",
            root.join(&filename).display()
        ))),
        [path] => Ok(path.clone()),
        _ => Err(Error::Argument(format!(
            "Ambiguous dataset: {}",
            paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// `Dataset.read(path)`. YAML is read safely: tagged nodes (`!ruby/object`) are rejected
/// rather than constructed, like `YAML.safe_load_file`.
fn read(path: &Path) -> Result<Value> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let body = || std::fs::read_to_string(path);
    match ext {
        "yml" | "yaml" => {
            let yaml: serde_yaml::Value = serde_yaml::from_str(&body()?)
                .map_err(|e| Error::Argument(format!("Invalid YAML in {}: {e}", path.display())))?;
            yaml_to_json(yaml)
        }
        "json" => Ok(serde_json::from_str(&body()?)?),
        "jsonl" => body()?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(Error::from))
            .collect::<Result<Vec<Value>>>()
            .map(Value::Array),
        _ => Err(Error::Argument(format!(
            "Unsupported dataset format: {}",
            path.display()
        ))),
    }
}

/// Plain YAML data as JSON. A tag is a request to construct an object (`Psych::DisallowedClass`).
fn yaml_to_json(value: serde_yaml::Value) -> Result<Value> {
    use serde_yaml::Value as Y;
    Ok(match value {
        Y::Null => Value::Null,
        Y::Bool(b) => Value::Bool(b),
        Y::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into()
            } else if let Some(u) = n.as_u64() {
                u.into()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .ok_or_else(|| Error::Argument("Judgment data must contain finite numbers".into()))?
            }
        }
        Y::String(s) => Value::String(s),
        Y::Sequence(items) => Value::Array(items.into_iter().map(yaml_to_json).collect::<Result<_>>()?),
        Y::Mapping(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                let key = match k {
                    Y::String(s) => s,
                    Y::Bool(b) => b.to_string(),
                    Y::Number(n) => n.to_string(),
                    Y::Null => String::new(),
                    other => return Err(Error::Argument(format!("Unsupported YAML key: {other:?}"))),
                };
                out.insert(key, yaml_to_json(v)?);
            }
            Value::Object(out)
        }
        Y::Tagged(tagged) => {
            return Err(Error::Argument(format!(
                "Tried to load unspecified class: {}",
                tagged.tag
            )));
        }
    })
}
