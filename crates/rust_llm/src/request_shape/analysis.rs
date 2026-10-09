//! Port of `lib/ruby_llm/protocol/request_shapes/analysis.rb` and the `PartSpec`/`TurnSpec`
//! structs of `lib/ruby_llm/protocol/request_shapes.rb`.
//!
//! Finds what a provider would refuse in the turns a protocol read: parts with no data, empty
//! turns, tool rounds whose calls and results do not pair up, and results that answer no call,
//! plus the rules a protocol opts into.

use std::collections::{HashMap, HashSet};

use super::{Part, PartKind, Problem, ProblemKind, Source, ToolRound, Turn, Unit};

pub const KEPT_FIRST_TURNS: usize = 10;
pub const KEPT_LAST_TURNS: usize = 50;

/// How the provider matches results to calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    Id,
    /// Compares tool names in order.
    Position,
}

/// Rules a protocol opts into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// The first call of each step in the current turn needs a signature.
    SignedSteps,
    /// Every thinking part needs a signature.
    SignedThinking,
    /// User and model turns alternate, starting with the user.
    AlternatingRoles,
}

/// `TurnSpec#speaker`: whose turn it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    System,
    User,
    Model,
    Tool,
}

/// `PartSpec`: a piece of a turn as a protocol reads it, with the ids and flags the analysis
/// needs and a shape never shows.
#[derive(Debug, Clone)]
pub struct PartSpec {
    pub kind: PartKind,
    pub measure: Option<usize>,
    pub unit: Option<Unit>,
    pub name: Option<String>,
    pub source: Option<Source>,
    pub mime_type: Option<String>,
    pub signed: bool,
    pub call_id: Option<String>,
    pub result_id: Option<String>,
    pub without_data: bool,
}

impl PartSpec {
    pub fn new(kind: PartKind) -> PartSpec {
        PartSpec {
            kind,
            measure: None,
            unit: None,
            name: None,
            source: None,
            mime_type: None,
            signed: false,
            call_id: None,
            result_id: None,
            without_data: false,
        }
    }

    pub fn is_call(&self) -> bool {
        self.kind == PartKind::ToolCall
    }

    pub fn is_result(&self) -> bool {
        self.kind == PartKind::ToolResult
    }

    pub fn to_part(&self) -> Part {
        Part {
            kind: self.kind,
            size: self.measure,
            unit: self.measure.map(|_| self.unit.unwrap_or(Unit::Chars)),
            name: self.name.clone(),
            source: self.source,
            mime_type: self.mime_type.clone(),
            signed: self.signed,
        }
    }
}

/// `TurnSpec`.
#[derive(Debug, Clone)]
pub struct TurnSpec {
    pub index: usize,
    pub role: Option<String>,
    pub speaker: Speaker,
    pub parts: Vec<PartSpec>,
}

impl TurnSpec {
    pub fn is_model(&self) -> bool {
        self.speaker == Speaker::Model
    }

    pub fn to_turn(&self) -> Turn {
        Turn {
            index: self.index,
            role: self.role.clone(),
            parts: self.parts.iter().map(PartSpec::to_part).collect(),
        }
    }
}

/// A run of model turns (by position in the turns) and the results that answer it.
struct Round {
    turn: usize,
    calls: Vec<(usize, usize)>,
    results: Vec<(usize, usize)>,
}

/// `Protocol::RequestShapes::Analysis`.
pub struct Analysis {
    turns: Vec<TurnSpec>,
    pairing: Pairing,
    rules: Vec<Rule>,
    rounds: Vec<Round>,
}

fn counted(count: usize, noun: &str) -> String {
    format!("{count} {noun}{}", if count == 1 { "" } else { "s" })
}

impl Analysis {
    pub fn new(mut turns: Vec<TurnSpec>, pairing: Pairing, rules: &[Rule]) -> Analysis {
        if pairing == Pairing::Id {
            name_results(&mut turns);
        }
        let rounds = rounds(&turns);
        Analysis {
            turns,
            pairing,
            rules: rules.to_vec(),
            rounds,
        }
    }

    /// The turns, with result names filled in from their calls.
    pub fn turns(&self) -> &[TurnSpec] {
        &self.turns
    }

    /// `kept_turns`: the first 10 and last 50 of a long conversation.
    pub fn kept_turns(&self) -> Vec<&TurnSpec> {
        let n = self.turns.len();
        if n <= KEPT_FIRST_TURNS + KEPT_LAST_TURNS {
            return self.turns.iter().collect();
        }
        self.turns[..KEPT_FIRST_TURNS]
            .iter()
            .chain(&self.turns[n - KEPT_LAST_TURNS..])
            .collect()
    }

    fn last_user_turn(&self) -> Option<usize> {
        self.turns.iter().rposition(|t| t.speaker == Speaker::User)
    }

    /// `step`: the model call after every round of the current turn, which starts after the last
    /// user turn.
    pub fn step(&self) -> Option<usize> {
        let last = self.turns[self.last_user_turn()?].index;
        Some(
            self.rounds
                .iter()
                .filter(|r| self.turns[r.turn].index > last)
                .count()
                + 1,
        )
    }

    pub fn tool_rounds(&self) -> Vec<ToolRound> {
        let kept: HashSet<usize> = self.kept_turns().iter().map(|t| t.index).collect();
        self.rounds
            .iter()
            .filter(|r| kept.contains(&self.turns[r.turn].index))
            .map(|r| {
                ToolRound::new(
                    self.turns[r.turn].index,
                    r.calls.len(),
                    r.results.len(),
                    self.is_paired(r),
                )
            })
            .collect()
    }

    pub fn problems(&self) -> Vec<Problem> {
        let mut found: Vec<Problem> = Vec::new();
        found.extend(self.parts_without_data());
        found.extend(self.empty_turns());
        found.extend(self.unpaired_rounds());
        found.extend(self.unmatched_results());
        found.extend(self.unsigned_calls());
        found.extend(self.unsigned_thinking());
        found.extend(self.roles_out_of_order());
        // `sort_by { [turn, part || -1, order] }`: stable, so `order` is the original position.
        found.sort_by_key(|p| (p.turn, p.part.map_or(-1, |part| part as i64)));
        found
    }

    fn part(&self, (t, p): (usize, usize)) -> &PartSpec {
        &self.turns[t].parts[p]
    }

    fn is_paired(&self, round: &Round) -> bool {
        if round.calls.len() != round.results.len() {
            return false;
        }
        let ids = self.pairing == Pairing::Id
            && round.calls.iter().all(|&c| self.part(c).call_id.is_some())
            && round
                .results
                .iter()
                .all(|&r| self.part(r).result_id.is_some());
        if !ids {
            let calls: Vec<_> = round.calls.iter().map(|&c| &self.part(c).name).collect();
            let results: Vec<_> = round.results.iter().map(|&r| &self.part(r).name).collect();
            return calls == results;
        }
        let mut calls: Vec<&Option<String>> =
            round.calls.iter().map(|&c| &self.part(c).call_id).collect();
        let mut results: Vec<&Option<String>> = round
            .results
            .iter()
            .map(|&r| &self.part(r).result_id)
            .collect();
        // `tally == tally`: the same ids, as many times each.
        calls.sort();
        results.sort();
        calls == results
    }

    fn problem(
        &self,
        kind: ProblemKind,
        message: String,
        turn: usize,
        part: Option<usize>,
    ) -> Problem {
        Problem::new(kind, message).at(self.turns[turn].index, part)
    }

    fn parts_without_data(&self) -> Vec<Problem> {
        let mut out = Vec::new();
        for (t, turn) in self.turns.iter().enumerate() {
            for (p, part) in turn.parts.iter().enumerate() {
                if part.without_data {
                    let noun = if part.kind == PartKind::Other {
                        "part".to_string()
                    } else {
                        format!("{} part", part.kind.as_str())
                    };
                    out.push(self.problem(
                        ProblemKind::PartWithoutData,
                        format!("{noun} carries no data"),
                        t,
                        Some(p),
                    ));
                }
            }
        }
        out
    }

    fn empty_turns(&self) -> Vec<Problem> {
        (0..self.turns.len())
            .filter(|&t| self.turns[t].parts.is_empty())
            .map(|t| self.problem(ProblemKind::EmptyTurn, "turn has no parts".into(), t, None))
            .collect()
    }

    fn unpaired_rounds(&self) -> Vec<Problem> {
        self.rounds
            .iter()
            .filter(|r| !self.is_paired(r))
            .map(|r| {
                let calls = counted(r.calls.len(), "call");
                let results = counted(r.results.len(), "result");
                let message = if r.calls.len() == r.results.len() {
                    format!("{calls} and {results} that do not pair up")
                } else {
                    format!("{calls} but {results}")
                };
                self.problem(ProblemKind::UnpairedRound, message, r.turn, None)
            })
            .collect()
    }

    /// The one place a shape shows an id: a result naming a call the request never made.
    fn unmatched_results(&self) -> Vec<Problem> {
        if self.pairing != Pairing::Id {
            return Vec::new();
        }
        let mut called: HashSet<&Option<String>> = HashSet::new();
        let mut out = Vec::new();
        for (t, turn) in self.turns.iter().enumerate() {
            for (p, part) in turn.parts.iter().enumerate() {
                if part.is_call() && part.call_id.is_some() {
                    called.insert(&part.call_id);
                }
                if part.is_result() && !called.contains(&part.result_id) {
                    let mut problem = self.problem(
                        ProblemKind::UnmatchedResult,
                        "result answers no call in the request".into(),
                        t,
                        Some(p),
                    );
                    problem.call_id = part.result_id.clone();
                    out.push(problem);
                }
            }
        }
        out
    }

    fn unsigned_calls(&self) -> Vec<Problem> {
        if !self.rules.contains(&Rule::SignedSteps) {
            return Vec::new();
        }
        let start = self.last_user_turn().map_or(0, |u| u + 1);
        let mut out = Vec::new();
        for run in runs(&self.turns, start) {
            if !self.turns[run[0]].is_model() {
                continue;
            }
            let first = run.iter().find_map(|&t| {
                self.turns[t]
                    .parts
                    .iter()
                    .position(PartSpec::is_call)
                    .map(|p| (t, p))
            });
            if let Some((t, p)) = first
                && !self.turns[t].parts[p].signed
            {
                out.push(self.problem(
                    ProblemKind::UnsignedCall,
                    "first call of a step in the current turn has no signature".into(),
                    t,
                    Some(p),
                ));
            }
        }
        out
    }

    fn unsigned_thinking(&self) -> Vec<Problem> {
        if !self.rules.contains(&Rule::SignedThinking) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (t, turn) in self.turns.iter().enumerate() {
            for (p, part) in turn.parts.iter().enumerate() {
                if part.kind == PartKind::Thinking && !part.signed {
                    out.push(self.problem(
                        ProblemKind::UnsignedThinking,
                        "thinking part has no signature".into(),
                        t,
                        Some(p),
                    ));
                }
            }
        }
        out
    }

    fn roles_out_of_order(&self) -> Vec<Problem> {
        if !self.rules.contains(&Rule::AlternatingRoles) {
            return Vec::new();
        }
        let turns: Vec<usize> = (0..self.turns.len())
            .filter(|&t| self.turns[t].speaker != Speaker::System)
            .collect();
        let mut out = Vec::new();
        if let Some(&first) = turns.first()
            && self.turns[first].is_model()
        {
            out.push(self.problem(
                ProblemKind::RoleOrder,
                "conversation starts with a model turn".into(),
                first,
                None,
            ));
        }
        for pair in turns.windows(2) {
            let (a, b) = (&self.turns[pair[0]], &self.turns[pair[1]]);
            if a.is_model() == b.is_model() {
                let who = if b.is_model() { "model" } else { "user" };
                out.push(self.problem(
                    ProblemKind::RoleOrder,
                    format!("follows another {who} turn"),
                    pair[1],
                    None,
                ));
            }
        }
        out
    }
}

/// `name_results`: a result names the call it answers by id.
fn name_results(turns: &mut [TurnSpec]) {
    let mut names: HashMap<String, Option<String>> = HashMap::new();
    for part in turns.iter_mut().flat_map(|t| t.parts.iter_mut()) {
        if part.is_call()
            && let Some(id) = &part.call_id
        {
            names.insert(id.clone(), part.name.clone());
        }
        if part.is_result() && part.name.is_none() {
            part.name = part
                .result_id
                .as_ref()
                .and_then(|id| names.get(id).cloned().flatten());
        }
    }
}

/// `runs`: consecutive turns from `start` grouped by whether they are model turns.
fn runs(turns: &[TurnSpec], start: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    for t in start..turns.len() {
        match out.last_mut() {
            Some(run) if turns[run[0]].is_model() == turns[t].is_model() => run.push(t),
            _ => out.push(vec![t]),
        }
    }
    out
}

/// `rounds`: consecutive model turns make one round, answered by the results in the turns that
/// follow them up to the next model turn.
fn rounds(turns: &[TurnSpec]) -> Vec<Round> {
    let runs = runs(turns, 0);
    let mut out = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        if !turns[run[0]].is_model() {
            continue;
        }
        let located = |run: &Vec<usize>, want: fn(&PartSpec) -> bool| -> Vec<(usize, usize)> {
            run.iter()
                .flat_map(|&t| {
                    turns[t]
                        .parts
                        .iter()
                        .enumerate()
                        .filter(move |(_, p)| want(p))
                        .map(move |(p, _)| (t, p))
                })
                .collect()
        };
        let calls = located(run, PartSpec::is_call);
        if calls.is_empty() {
            continue;
        }
        let results = runs
            .get(i + 1)
            .map(|answers| located(answers, PartSpec::is_result))
            .unwrap_or_default();
        out.push(Round {
            turn: run[0],
            calls,
            results,
        });
    }
    out
}
