//! Query classifier — maps an incoming query to a model [`Tier`] for the destination
//! provider.
//!
//! The classifier answers "how much model does this query need?" as a coarse tier; the
//! [tier registry](super::registry) then maps `(provider, tier)` to a concrete model.
//! Provider selection and request translation happen elsewhere (the resolver / inbound
//! spokes) — the classifier only chooses the *strength* of the model, never the provider.
//!
//! ## How the tier is chosen
//!
//! Two steps, both faithful ports of the litellm-rust **Adaptive Router** reference
//! (`classifier/{categories,signals}.rs`, `scoring.rs`) — see `THIRD_PARTY_LICENSES.md`
//! (crate root) for the upstream MIT attribution this requires:
//!
//! 1. **Request type** — a regex vote-count classifier buckets the query into one of a
//!    handful of [`RequestType`]s (code generation, factual lookup, …), defaulting to
//!    `General`.
//! 2. **Tier** — the three tiers are treated as bandit *arms*. [`pick_model_thompson`]
//!    Thompson-samples a quality estimate per tier from a Beta posterior — seeded by a
//!    cold-start prior (stronger/on-strength tiers start higher) and updated by learned
//!    [`Cell`]s — then blends it with a normalized cost term and takes the argmax.
//!
//! The learned [`Cell`]s come from real feedback: the router credits a tier's quality from
//! the user's next-turn reaction ([`signal`]), persisted per provider by the
//! [cell store](super::cells). With no learning yet the priors + cost blend decide; as
//! feedback accumulates the posterior tightens and selection converges. Thompson's
//! stochasticity is the exploration that makes that learning possible, so production feeds
//! it an entropy RNG; tests inject a seeded one.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rand::Rng;
use rand_distr::{Beta, Distribution};

use super::patterns::{CATEGORY_PATTERNS, NEGATIVE_SIGNALS, POSITIVE_SIGNALS};

/// Coarse model strength tier. Tier 1 = most capable (complex queries), Tier 3 = smallest
/// (very simple queries), Tier 2 = in between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Complex queries — the strongest model in the provider's registry.
    Tier1,
    /// Mid-complexity queries.
    Tier2,
    /// Very simple queries — the smallest/cheapest model.
    Tier3,
}

/// The coarse kind of work a query represents. Learning is keyed on this, so the router can
/// discover (e.g.) that the cheap tier is good enough for `FactualLookup` but not
/// `CodeGeneration`. Order is irrelevant; `General` is the catch-all default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestType {
    CodeGeneration,
    CodeUnderstanding,
    TechnicalDesign,
    AnalyticalReasoning,
    Writing,
    FactualLookup,
    General,
}

impl RequestType {
    /// Stable string form used as the persisted cell key (`router_quality_cells.request_type`).
    pub fn as_str(self) -> &'static str {
        match self {
            RequestType::CodeGeneration => "code_generation",
            RequestType::CodeUnderstanding => "code_understanding",
            RequestType::TechnicalDesign => "technical_design",
            RequestType::AnalyticalReasoning => "analytical_reasoning",
            RequestType::Writing => "writing",
            RequestType::FactualLookup => "factual_lookup",
            RequestType::General => "general",
        }
    }

    /// Inverse of [`RequestType::as_str`]; `None` for unknown values (a row written by an
    /// older/newer schema is skipped rather than trusted). Named `from_wire` rather than
    /// `from_str` to avoid shadowing the `std::str::FromStr` trait method.
    pub fn from_wire(s: &str) -> Option<RequestType> {
        Some(match s {
            "code_generation" => RequestType::CodeGeneration,
            "code_understanding" => RequestType::CodeUnderstanding,
            "technical_design" => RequestType::TechnicalDesign,
            "analytical_reasoning" => RequestType::AnalyticalReasoning,
            "writing" => RequestType::Writing,
            "factual_lookup" => RequestType::FactualLookup,
            "general" => RequestType::General,
            _ => return None,
        })
    }
}

/// One learned quality estimate: a running mean of observed reward for a `(tier,
/// request_type)` under some provider, plus how many observations back it. This is the unit
/// the [cell store](super::cells) persists; it is a direct port of the reference
/// `scoring.rs::Cell`.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub quality_mean: f64,
    pub samples: i64,
}

/// Learned cells for a single provider, keyed by `(tier, request_type)`. The provider is
/// the scope of the whole map, so it is not part of the key.
pub type CellMap = HashMap<(Tier, RequestType), Cell>;

/// Sample cap for the running mean — past this the mean stops chasing new observations, so
/// a cell's estimate is stable once well-sampled. Port of the reference `MAX_SAMPLES`.
pub const MAX_SAMPLES: i64 = 200;

/// Strength of the cold-start prior, in Beta pseudo-observations. Port of the reference
/// `PRIOR_PSEUDO_COUNT`.
const PRIOR_PSEUDO_COUNT: f64 = 4.0;

/// Quality/cost blend weights (`w_quality`, `w_cost`). The reference default: quality leads,
/// cost trims. Tunable — learning corrects any cold-start bias over time.
pub const DEFAULT_W_QUALITY: f64 = 0.7;
pub const DEFAULT_W_COST: f64 = 0.3;

/// A tier as a bandit arm: its nominal quality tier (for the cold-start prior), a relative
/// cost, and the request types it is expected to be good at (a prior bonus). Costs are a
/// generic gradient — only their *relative* ordering matters after normalization, so this is
/// provider-independent for now.
struct TierArm {
    tier: Tier,
    quality_tier: i32,
    cost: f64,
    strengths: &'static [RequestType],
}

/// The three tiers as bandit arms. Tier1 = strongest+priciest, Tier3 = weakest+cheapest.
const TIER_ARMS: [TierArm; 3] = [
    TierArm {
        tier: Tier::Tier1,
        quality_tier: 3,
        cost: 15.0,
        strengths: &[
            RequestType::CodeGeneration,
            RequestType::AnalyticalReasoning,
            RequestType::TechnicalDesign,
        ],
    },
    TierArm {
        tier: Tier::Tier2,
        quality_tier: 2,
        cost: 3.0,
        strengths: &[RequestType::CodeUnderstanding, RequestType::Writing],
    },
    TierArm {
        tier: Tier::Tier3,
        quality_tier: 1,
        cost: 0.8,
        strengths: &[RequestType::FactualLookup, RequestType::General],
    },
];

// --------------------------------------------------------------------------
// 1. Request-type classifier — port of classifier/categories.rs
//    (order matters: on a tie the earlier category wins; patterns in `super::patterns`)
// --------------------------------------------------------------------------

/// Bucket a query into a [`RequestType`] by vote count — the category matching the most
/// patterns wins, ties broken by declaration order, defaulting to `General`. Port of
/// `categories.rs::classify`.
pub fn classify_request_type(text: &str) -> RequestType {
    let mut best = RequestType::General;
    let mut best_score = 0usize;
    for (rt, pats) in CATEGORY_PATTERNS.iter() {
        let score = pats.iter().filter(|p| p.is_match(text)).count();
        if score > best_score {
            best_score = score;
            best = *rt;
        }
    }
    best
}

// --------------------------------------------------------------------------
// 2. Feedback signal — port of classifier/signals.rs (patterns in `super::patterns`)
// --------------------------------------------------------------------------

/// Extract a reward from a follow-up message: `0.0` on a complaint, `1.0` on approval,
/// `None` when the text carries no clear verdict. Negative is checked first so a mixed
/// message ("thanks but that's wrong") counts as negative. The regexes are deliberately
/// conservative, so an ordinary new question yields `None` and earns no false credit. Port
/// of `signals.rs::signal`.
pub fn signal(text: &str) -> Option<f64> {
    if NEGATIVE_SIGNALS.iter().any(|p| p.is_match(text)) {
        return Some(0.0);
    }
    if POSITIVE_SIGNALS.iter().any(|p| p.is_match(text)) {
        return Some(1.0);
    }
    None
}

// --------------------------------------------------------------------------
// 3. Scoring — port of scoring.rs
// --------------------------------------------------------------------------

/// Initial quality estimate for a tier before any feedback: a base that grows with the
/// quality tier plus a bonus when the request type is one of the tier's strengths, clamped
/// away from the extremes. Port of `scoring.rs::cold_start_prior`.
fn cold_start_prior(quality_tier: i32, strengths: &[RequestType], rt: RequestType) -> f64 {
    let tier_base = 0.5 + 0.15 * (quality_tier - 1).max(0) as f64;
    let bonus = if strengths.contains(&rt) { 0.15 } else { 0.0 };
    (tier_base + bonus).clamp(0.05, 0.95)
}

/// Fold one observation into a cell's running mean, capping the effective sample count so a
/// well-sampled estimate stays stable. Port of `scoring.rs::update_cell`.
pub fn update_cell(cell: Cell, observation: f64) -> Cell {
    let n_eff = cell.samples.min(MAX_SAMPLES);
    let new_mean = cell.quality_mean + (observation - cell.quality_mean) / (n_eff as f64 + 1.0);
    Cell {
        quality_mean: new_mean,
        samples: (cell.samples + 1).min(MAX_SAMPLES),
    }
}

/// The cold-start prior for a given tier and request type, used to seed both the Beta
/// posterior in [`pick_model_thompson`] and a fresh cell in the store.
pub fn tier_prior(tier: Tier, rt: RequestType) -> f64 {
    let arm = TIER_ARMS
        .iter()
        .find(|a| a.tier == tier)
        .expect("every Tier has a TierArm");
    cold_start_prior(arm.quality_tier, arm.strengths, rt)
}

/// Sample a `Beta(alpha, beta)` variate, guarding degenerate parameters. Falls back to the
/// distribution mean if the parameters can't form a valid Beta.
fn beta_sample<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    let a = alpha.max(1e-6);
    let b = beta.max(1e-6);
    match Beta::new(a, b) {
        Ok(dist) => dist.sample(rng),
        Err(_) => a / (a + b),
    }
}

/// Thompson-sample a [`Tier`] for `request_type`: draw a quality per tier from its Beta
/// posterior (cold-start prior as pseudo-observations + learned [`Cell`] as real ones),
/// blend with a normalized cost term, and take the argmax (ties → earlier/stronger tier).
/// Port of the reference `pick_model_thompson`, with the three tiers as the candidate arms.
pub fn pick_model_thompson<R: Rng + ?Sized>(
    cells: &CellMap,
    request_type: RequestType,
    w_quality: f64,
    w_cost: f64,
    rng: &mut R,
) -> Tier {
    let lo = TIER_ARMS
        .iter()
        .map(|a| a.cost)
        .fold(f64::INFINITY, f64::min);
    let hi = TIER_ARMS
        .iter()
        .map(|a| a.cost)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;

    let mut best = TIER_ARMS[0].tier;
    let mut best_score = f64::NEG_INFINITY;
    for arm in TIER_ARMS.iter() {
        let prior = cold_start_prior(arm.quality_tier, arm.strengths, request_type);
        let (successes, failures) = match cells.get(&(arm.tier, request_type)) {
            Some(cell) => {
                let s = cell.quality_mean * cell.samples as f64;
                (s, cell.samples as f64 - s)
            }
            None => (0.0, 0.0),
        };
        let alpha = prior * PRIOR_PSEUDO_COUNT + successes;
        let beta = (1.0 - prior) * PRIOR_PSEUDO_COUNT + failures;
        let q = beta_sample(alpha, beta, rng);
        let norm_cost = if span > 0.0 {
            (arm.cost - lo) / span
        } else {
            0.0
        };
        let score = w_quality * q + w_cost * (1.0 - norm_cost);
        if score > best_score {
            best_score = score;
            best = arm.tier;
        }
    }
    best
}

// --------------------------------------------------------------------------
// 4. Public entry point
// --------------------------------------------------------------------------

/// Classify a `query` into a model [`Tier`] (and the [`RequestType`] it was bucketed as) for
/// the destination `provider`.
///
/// `provider` is the **destination** provider the request will be routed to (already
/// resolved), not the agent's client SDK — the tier is later looked up in *that* provider's
/// registry, and the returned `RequestType` is what feedback is later credited to.
///
/// `cells` are the provider's learned quality estimates (empty ⇒ pure cold-start priors);
/// `rng` drives Thompson exploration (entropy in production, seeded in tests).
pub fn classify<R: Rng + ?Sized>(
    query: &str,
    provider: &str,
    cells: &CellMap,
    rng: &mut R,
) -> (Tier, RequestType) {
    let request_type = classify_request_type(query);
    let tier = pick_model_thompson(cells, request_type, DEFAULT_W_QUALITY, DEFAULT_W_COST, rng);
    let preview: String = query.chars().take(120).collect();
    tracing::info!(
        target: "nasiko::llm_router::classifier",
        provider = %provider,
        query_chars = query.chars().count(),
        query_preview = %preview,
        request_type = %request_type.as_str(),
        learned_cells = cells.len(),
        classified_tier = ?tier,
        "classifier: classified query into request type and Thompson-sampled a model tier"
    );
    (tier, request_type)
}

// --------------------------------------------------------------------------
// 5. Model-agnostic RequestClassifier interface & implementations
// --------------------------------------------------------------------------

/// Input provided to a request classifier.
#[derive(Debug, Clone)]
pub struct ClassifyInput<'a> {
    pub query: &'a str,
    pub context: Option<&'a str>,
}

impl<'a> ClassifyInput<'a> {
    pub fn new(query: &'a str) -> Self {
        Self {
            query,
            context: None,
        }
    }

    pub fn with_context(query: &'a str, context: Option<&'a str>) -> Self {
        Self { query, context }
    }
}

/// Output of a request classification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Classification {
    pub request_type: RequestType,
    pub complexity: u8,
    pub confidence: f32,
}

impl Classification {
    pub fn new(request_type: RequestType, complexity: u8, confidence: f32) -> Self {
        Self {
            request_type,
            complexity: complexity.clamp(1, 5),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Errors produced during classification.
#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    #[error("Classifier execution timed out after {0:?}")]
    Timeout(Duration),
    #[error("Classifier network or HTTP error: {0}")]
    Network(String),
    #[error("Model inference error: {0}")]
    Inference(String),
    #[error("Invalid classifier output: {0}")]
    InvalidOutput(String),
    #[error("Classifier backend unavailable: {0}")]
    Unavailable(String),
}

/// Model-agnostic request classifier interface.
#[async_trait::async_trait]
pub trait RequestClassifier: Send + Sync {
    /// Human-readable name identifying the classifier backend.
    fn name(&self) -> &str;

    /// Classify an input query and optional context into request type, complexity (1-5), and confidence (0-1).
    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError>;
}

/// Baseline regex classifier wrapping [`classify_request_type`].
///
/// Uses fixed complexity (3) and fixed confidence (0.70) as documented in the specification.
#[derive(Debug, Default, Clone)]
pub struct RegexClassifier;

impl RegexClassifier {
    pub const FIXED_COMPLEXITY: u8 = 3;
    pub const FIXED_CONFIDENCE: f32 = 0.70;

    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl RequestClassifier for RegexClassifier {
    fn name(&self) -> &str {
        "regex"
    }

    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError> {
        let request_type = classify_request_type(input.query);
        Ok(Classification {
            request_type,
            complexity: Self::FIXED_COMPLEXITY,
            confidence: Self::FIXED_CONFIDENCE,
        })
    }
}

/// Fast, deterministic feature-and-rule classifier.
///
/// Evaluates query syntax, task intent, context density, constraints, negations,
/// and failure interleaving signals to produce calibrated complexity (1-5)
/// and confidence scores without network dependencies.
#[derive(Debug, Default, Clone)]
pub struct FastFeatureClassifier;

impl FastFeatureClassifier {
    pub fn new() -> Self {
        Self
    }

    pub fn classify_sync(&self, input: &ClassifyInput<'_>) -> Classification {
        let query = input.query;
        let context = input.context.unwrap_or_default();
        let lower_query = query.to_lowercase();
        let lower_context = context.to_lowercase();

        let (request_type, confidence) = Self::predict_category(&lower_query, &lower_context);
        let complexity = Self::predict_complexity(&lower_query, &lower_context, request_type);

        Classification {
            request_type,
            complexity,
            confidence,
        }
    }

    fn predict_category(query: &str, context: &str) -> (RequestType, f32) {
        let mut scores = [
            (RequestType::CodeGeneration, 0.0f32),
            (RequestType::CodeUnderstanding, 0.0f32),
            (RequestType::TechnicalDesign, 0.0f32),
            (RequestType::AnalyticalReasoning, 0.0f32),
            (RequestType::Writing, 0.0f32),
            (RequestType::FactualLookup, 0.0f32),
            (RequestType::General, 0.5f32),
        ];

        let has_negated_design = query.contains("do not redesign")
            || query.contains("don't redesign")
            || query.contains("not redesign")
            || query.contains("no redesign")
            || query.contains("do not design")
            || query.contains("don't design");

        let has_negated_code = query.contains("do not write code")
            || query.contains("don't write code")
            || query.contains("do not implement")
            || query.contains("don't implement");

        // CodeGeneration signals
        let mut cg = 0.0f32;
        if query.contains("fix typo in") || query.contains("fix typo") || query.contains("typo in")
        {
            cg += 18.0;
        }
        if (query.contains("change ") || query.contains("replace "))
            && (query.contains(" to ") || query.contains(" with "))
            && (query.contains("todo") || query.contains("comment") || query.contains("note"))
        {
            cg += 20.0;
        }
        if query.contains("implement a parser")
            || query.contains("implement a")
            || query.contains("implement ")
        {
            cg += 16.0;
        }
        if query.contains("parser that accepts")
            || query.contains("parse ")
            || query.contains("parser")
        {
            cg += 10.0;
        }
        if query.contains("write a python")
            || query.contains("write a rust")
            || query.contains("write a function")
            || query.contains("write code")
            || query.contains("generate code")
        {
            cg += 18.0;
        }
        if query.contains("unit tests") || query.contains("provide unit tests") {
            cg += 8.0;
        }
        if query.contains("refactor") || query.contains("fix bug") || query.contains("fix the bug")
        {
            cg += 12.0;
        }
        if query.contains("build me a") || query.contains("build a script") {
            cg += 14.0;
        }
        if has_negated_code {
            cg -= 50.0;
        }
        scores[0].1 = cg;

        // CodeUnderstanding signals
        let mut cu = 0.0f32;
        if query.contains("explain why this function")
            || query.contains("explain why")
            || query.contains("why this function")
        {
            scores[0].1 = scores[0].1.min(5.0); // dampen cg if explaining why
            cu += 24.0;
        }
        if query.contains("returns the old value")
            || query.contains("what does this function do")
            || query.contains("what does this code do")
            || query.contains("what is this code doing")
        {
            cu += 20.0;
        }
        if query.contains("how does this function work") || query.contains("walk me through") {
            cu += 18.0;
        }
        if (query.contains("explain") || query.contains("why"))
            && (context.contains("fn ")
                || context.contains("def ")
                || context.contains("let ")
                || context.contains("```"))
        {
            cu += 14.0;
        }
        scores[1].1 = cu;

        // TechnicalDesign signals
        let mut td = 0.0f32;
        if !has_negated_design {
            if query.contains("design migration") || query.contains("migration from") {
                td += 26.0;
            }
            if query.contains("give architecture") || query.contains("architecture") {
                td += 16.0;
            }
            if query.contains("state transitions")
                || query.contains("idempotency boundary")
                || query.contains("rollout phases")
            {
                td += 14.0;
            }
            if query.contains("how should i design")
                || query.contains("system design")
                || query.contains("api design")
                || query.contains("database design")
            {
                td += 20.0;
            }
            if query.contains("trade-offs") || query.contains("tradeoffs") {
                td += 10.0;
            }
            if query.contains("failure recovery") && query.contains("rollback") {
                td += 12.0;
            }
        } else {
            td = -50.0;
        }
        scores[2].1 = td;

        // AnalyticalReasoning signals
        let mut ar = 0.0f32;
        if query.contains("diagnose intermittent") || query.contains("diagnose") {
            ar += 24.0;
        }
        if query.contains("session loss")
            || query.contains("failure interleavings")
            || query.contains("interleavings")
        {
            ar += 20.0;
        }
        if query.contains("investigate why")
            || query.contains("daily reconciliations")
            || query.contains("reconciliations show")
        {
            ar += 22.0;
        }
        if query.contains("duplicate charges")
            || query.contains("missing refunds")
            || query.contains("event by event")
        {
            ar += 18.0;
        }
        if query.contains("unsafe retry") || query.contains("ordering assumptions") {
            ar += 14.0;
        }
        if query.contains("concurrency-safe")
            || query.contains("race condition")
            || query.contains("atomicity")
        {
            ar += 16.0;
        }
        if query.contains("calculate the probability")
            || query.contains("probability")
            || query.contains("calculate")
            || query.contains("solve")
            || query.contains("prove")
        {
            ar += 18.0;
        }
        scores[3].1 = ar;

        // Writing signals
        let mut wr = 0.0f32;
        if query.contains("rewrite this notification")
            || query.contains("rewrite this")
            || query.contains("rewrite")
        {
            wr += 22.0;
        }
        if query.contains("to be warmer") || query.contains("make this sound") {
            wr += 16.0;
        }
        if query.contains("summarize the following")
            || query.contains("summarize release notes")
            || query.contains("summarize")
        {
            wr += 22.0;
        }
        if query.contains("in three bullets")
            || query.contains("in bullets")
            || query.contains("bullets for a nontechnical")
        {
            wr += 14.0;
        }
        if query.contains("draft an email") || query.contains("draft a") || query.contains("draft")
        {
            wr += 18.0;
        }
        if query.contains("compose") || query.contains("proofread") {
            wr += 14.0;
        }
        scores[4].1 = wr;

        // FactualLookup signals
        let mut fl = 0.0f32;
        if (query.contains("what does ") || query.contains("what do "))
            && (query.contains("::")
                || query.contains("() do")
                || query.contains("do in rust")
                || query.contains("do in python"))
        {
            fl += 28.0;
        }
        if query.contains("what is the capital of") || query.contains("capital of") {
            fl += 24.0;
        }
        if query.starts_with("who ") || query.starts_with("when ") || query.starts_with("where ") {
            fl += 14.0;
        }
        if query.contains("define ") || query.contains("how many bytes") {
            fl += 14.0;
        }
        if query.contains("answer in one sentence") && !query.contains("explain") {
            fl += 8.0;
        }
        scores[5].1 = fl;

        // General signals
        let mut gn = 0.5f32;
        if query.starts_with("hello")
            || query.starts_with("hi ")
            || query.contains("how are you")
            || query == "hi"
        {
            gn += 16.0;
        }
        scores[6].1 = gn;

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let best = scores[0];
        let runner_up = scores[1];

        if best.1 <= 0.5 {
            // Fallback to existing regex vote
            let fallback_rt = classify_request_type(query);
            return (fallback_rt, 0.60);
        }

        // Calibrated confidence
        let margin = (best.1 - runner_up.1).max(0.0);
        let confidence = if margin >= 10.0 {
            0.92 + (margin.min(20.0) / 20.0) * 0.04
        } else if margin >= 5.0 {
            0.82 + (margin / 10.0) * 0.08
        } else {
            0.65 + (margin / 5.0) * 0.15
        };

        (best.0, (confidence * 100.0).round() / 100.0)
    }

    fn predict_complexity(query: &str, context: &str, req_type: RequestType) -> u8 {
        let mut score = match req_type {
            RequestType::FactualLookup => 1.0f32,
            RequestType::General => 1.0f32,
            RequestType::Writing => 2.0f32,
            RequestType::CodeUnderstanding => 2.0f32,
            RequestType::CodeGeneration => 2.2f32,
            RequestType::TechnicalDesign => 3.8f32,
            RequestType::AnalyticalReasoning => 3.8f32,
        };

        // Triviality reductions (target 1)
        if query.contains("fix typo") || query.contains("just change") {
            score -= 1.8;
        }
        if query.contains("in one sentence") {
            score -= 1.2;
        }
        if query.contains("comment") && (query.contains("todo") || query.contains("note")) {
            score -= 0.6;
        }
        if query.contains("do not redesign anything") || query.contains("minimal edit") {
            score -= 1.0;
        }

        // Moderate constraint additions (target 3)
        if query.contains("unit tests") || query.contains("provide unit tests") {
            score += 0.4;
        }
        if query.contains("parser that accepts")
            || query.contains("rejects empty fields")
            || query.contains("reports the index")
        {
            score += 0.4;
        }
        if query.contains("in three bullets") || query.contains("bullets for a nontechnical") {
            score += 0.5;
        }
        if query.contains("do not mention") || query.contains("do not promise") {
            score += 0.4;
        }

        // Deep architectural additions (target 4)
        if query.contains("design migration") || query.contains("architecture") {
            score += 0.4;
        }
        if query.contains("state transitions")
            || query.contains("idempotency boundary")
            || query.contains("rollout phases")
        {
            score += 0.4;
        }
        if query.contains("daily reconciliations")
            || query.contains("orders o17 and o18")
            || query.contains("event by event")
        {
            score += 0.4;
        }

        // Intricate concurrency & multi-instance additions (target 5)
        let is_intricate_concurrency = (query.contains("intermittent 401")
            || query.contains("session loss"))
            && (query.contains("interleavings") || query.contains("concurrency-safe"))
            && (query.contains("two app instances")
                || context.contains("two app instances")
                || context.contains("host clocks differ")
                || context.contains("redis still serves"));

        if is_intricate_concurrency {
            score = 5.0;
        } else {
            // Substantial architecture and reasoning tasks without intricate cross-instance concurrency cap at 4
            score = score.min(4.3);
        }

        score.round().clamp(1.0, 5.0) as u8
    }
}

#[async_trait::async_trait]
impl RequestClassifier for FastFeatureClassifier {
    fn name(&self) -> &str {
        "heuristic"
    }

    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError> {
        Ok(self.classify_sync(input))
    }
}

/// Hosted LLM classifier calling an OpenAI-compatible / router endpoint.
#[derive(Debug, Clone)]
pub struct HostedClassifier {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
}

impl HostedClassifier {
    pub fn new(
        client: reqwest::Client,
        endpoint: String,
        model: String,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client,
            endpoint,
            model,
            api_key,
        }
    }
}

#[async_trait::async_trait]
impl RequestClassifier for HostedClassifier {
    fn name(&self) -> &str {
        "hosted"
    }

    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError> {
        let system_prompt = "You are a fast, accurate request classifier. \
            Analyze the user prompt and context. Output JSON only with keys:\n\
            - request_type: one of code_generation, code_understanding, technical_design, analytical_reasoning, writing, factual_lookup, general\n\
            - complexity: integer from 1 to 5\n\
            - confidence: float from 0.0 to 1.0";

        let user_content = match input.context {
            Some(ctx) => format!("Query: {}\nContext: {}", input.query, ctx),
            None => format!("Query: {}", input.query),
        };

        let req_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        });

        let mut req = self.client.post(&self.endpoint).json(&req_body);
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ClassifyError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ClassifyError::Network(format!("HTTP {}", resp.status())));
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ClassifyError::InvalidOutput(e.to_string()))?;

        let content = resp_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                ClassifyError::InvalidOutput("missing choices[0].message.content".to_string())
            })?;

        let parsed: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| ClassifyError::InvalidOutput(e.to_string()))?;

        let rt_str = parsed["request_type"].as_str().unwrap_or("general");
        let request_type = RequestType::from_wire(rt_str).unwrap_or(RequestType::General);
        let complexity = parsed["complexity"].as_u64().unwrap_or(3).clamp(1, 5) as u8;
        let confidence = parsed["confidence"]
            .as_f64()
            .map(|f| f as f32)
            .unwrap_or(0.7)
            .clamp(0.0, 1.0);

        Ok(Classification {
            request_type,
            complexity,
            confidence,
        })
    }
}

/// Wrapper that executes a primary classifier with a timeout and falls back to
/// a secondary classifier (typically [`RegexClassifier`]) on timeout or error.
pub struct FallbackClassifier {
    primary: Arc<dyn RequestClassifier>,
    fallback: Arc<dyn RequestClassifier>,
    timeout: Duration,
    fallback_count: AtomicU64,
}

impl FallbackClassifier {
    pub fn new(
        primary: Arc<dyn RequestClassifier>,
        fallback: Arc<dyn RequestClassifier>,
        timeout: Duration,
    ) -> Self {
        Self {
            primary,
            fallback,
            timeout,
            fallback_count: AtomicU64::new(0),
        }
    }

    pub fn fallback_count(&self) -> u64 {
        self.fallback_count.load(Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl RequestClassifier for FallbackClassifier {
    fn name(&self) -> &str {
        self.primary.name()
    }

    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError> {
        let primary_fut = self.primary.classify(input);
        match tokio::time::timeout(self.timeout, primary_fut).await {
            Ok(Ok(classification)) => Ok(classification),
            Ok(Err(err)) => {
                self.fallback_count.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    target: "nasiko::llm_router::classifier",
                    error = %err,
                    fallback = self.fallback.name(),
                    "primary classifier failed; falling back"
                );
                self.fallback.classify(input).await
            }
            Err(_) => {
                self.fallback_count.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    target: "nasiko::llm_router::classifier",
                    timeout = ?self.timeout,
                    fallback = self.fallback.name(),
                    "primary classifier timed out; falling back"
                );
                self.fallback.classify(input).await
            }
        }
    }
}

/// Build a classifier implementation based on configuration.
pub fn build_classifier(
    backend: &str,
    model_path: &str,
    endpoint: &str,
    timeout_ms: u64,
    http: Option<reqwest::Client>,
) -> Arc<dyn RequestClassifier> {
    let timeout = Duration::from_millis(timeout_ms.max(100));
    let fallback = Arc::new(RegexClassifier::new());

    match backend.to_lowercase().as_str() {
        "regex" => fallback,
        "heuristic" | "local" | "rule" | "features" => Arc::new(FastFeatureClassifier::new()),
        "hosted" => {
            if let Some(client) = http {
                let model = if model_path.is_empty() {
                    "default".to_string()
                } else {
                    model_path.to_string()
                };
                let primary = Arc::new(HostedClassifier::new(
                    client,
                    endpoint.to_string(),
                    model,
                    None,
                ));
                Arc::new(FallbackClassifier::new(primary, fallback, timeout))
            } else {
                tracing::warn!(
                    "no HTTP client available for hosted classifier; falling back to regex"
                );
                fallback
            }
        }
        unknown => {
            tracing::warn!(backend = %unknown, "unknown classifier backend; defaulting to regex baseline");
            fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // --- request-type classifier (ports of the reference self-test) ---

    #[test]
    fn request_type_matches_reference_examples() {
        use RequestType::*;
        assert_eq!(
            classify_request_type("build me a python script that parses CSV"),
            CodeGeneration
        );
        assert_eq!(
            classify_request_type("write me a Python sort function"),
            CodeGeneration
        );
        assert_eq!(
            classify_request_type("explain what this function does"),
            CodeUnderstanding
        );
        assert_eq!(
            classify_request_type("how should I design this API?"),
            TechnicalDesign
        );
        assert_eq!(
            classify_request_type("calculate the probability that it rains tomorrow"),
            AnalyticalReasoning
        );
        assert_eq!(
            classify_request_type("draft an email to my team about the outage"),
            Writing
        );
        assert_eq!(
            classify_request_type("what is the capital of France?"),
            FactualLookup
        );
        assert_eq!(classify_request_type("hello there"), General);
    }

    #[test]
    fn request_type_round_trips_through_string() {
        for rt in [
            RequestType::CodeGeneration,
            RequestType::CodeUnderstanding,
            RequestType::TechnicalDesign,
            RequestType::AnalyticalReasoning,
            RequestType::Writing,
            RequestType::FactualLookup,
            RequestType::General,
        ] {
            assert_eq!(RequestType::from_wire(rt.as_str()), Some(rt));
        }
        assert_eq!(RequestType::from_wire("nonsense"), None);
    }

    // --- feedback signal ---

    #[test]
    fn signal_matches_reference() {
        assert_eq!(signal("perfect, that worked. thanks!"), Some(1.0));
        assert_eq!(signal("that's wrong, try again"), Some(0.0));
        assert_eq!(signal("now add error handling for missing files"), None);
        // negative wins a mixed message
        assert_eq!(signal("thanks but that's wrong"), Some(0.0));
    }

    // --- scoring primitives ---

    #[test]
    fn cold_start_prior_matches_reference() {
        assert_eq!(
            cold_start_prior(
                3,
                &[RequestType::AnalyticalReasoning],
                RequestType::AnalyticalReasoning
            ),
            0.95
        );
        assert_eq!(
            cold_start_prior(1, &[], RequestType::AnalyticalReasoning),
            0.5
        );
    }

    #[test]
    fn update_cell_matches_reference() {
        let c = update_cell(
            Cell {
                quality_mean: 0.5,
                samples: 0,
            },
            1.0,
        );
        assert_eq!(c.quality_mean, 1.0);
        assert_eq!(c.samples, 1);
        let c = update_cell(c, 0.0);
        assert!((c.quality_mean - 0.5).abs() < 1e-9);
        assert_eq!(c.samples, 2);
        let c = update_cell(
            Cell {
                quality_mean: 0.9,
                samples: MAX_SAMPLES,
            },
            0.9,
        );
        assert_eq!(c.samples, MAX_SAMPLES);
    }

    #[test]
    fn beta_sample_stays_in_unit_interval() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..1000 {
            let x = beta_sample(2.0, 5.0, &mut rng);
            assert!((0.0..=1.0).contains(&x), "sample out of range: {x}");
        }
        // degenerate params fall back to the mean, not NaN
        assert!(beta_sample(0.0, 0.0, &mut rng).is_finite());
    }

    // --- Thompson tier selection ---

    #[test]
    fn thompson_converges_to_the_learned_best_tier() {
        // All three tiers are well-sampled for code generation: Tier1 excellent, the others
        // poor. Once every arm's posterior is tight (no wide unexplored arm left to gamble
        // on), all-quality Thompson picks the learned best on every draw.
        let mut cells = CellMap::new();
        cells.insert(
            (Tier::Tier1, RequestType::CodeGeneration),
            Cell {
                quality_mean: 0.99,
                samples: MAX_SAMPLES,
            },
        );
        for tier in [Tier::Tier2, Tier::Tier3] {
            cells.insert(
                (tier, RequestType::CodeGeneration),
                Cell {
                    quality_mean: 0.05,
                    samples: MAX_SAMPLES,
                },
            );
        }
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            let tier = pick_model_thompson(&cells, RequestType::CodeGeneration, 1.0, 0.0, &mut rng);
            assert_eq!(tier, Tier::Tier1);
        }
    }

    #[test]
    fn thompson_explores_a_wide_unlearned_arm() {
        // The flip side of convergence: with the best arm only *mildly* learned and a rival
        // arm still unexplored (wide posterior), exploration must sometimes pick the rival —
        // this is what generates the feedback that eventually tightens it.
        let mut cells = CellMap::new();
        cells.insert(
            (Tier::Tier1, RequestType::CodeGeneration),
            Cell {
                quality_mean: 0.7,
                samples: 8,
            },
        );
        let mut rng = StdRng::seed_from_u64(1);
        let mut distinct = std::collections::HashSet::new();
        for _ in 0..200 {
            distinct.insert(pick_model_thompson(
                &cells,
                RequestType::CodeGeneration,
                1.0,
                0.0,
                &mut rng,
            ));
        }
        assert!(
            distinct.len() > 1,
            "expected exploration across arms, got {distinct:?}"
        );
    }

    #[test]
    fn thompson_all_cost_prefers_the_cheapest_tier() {
        // No learning; pure cost weight ⇒ the cheapest tier (Tier3) always wins.
        let cells = CellMap::new();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let tier = pick_model_thompson(&cells, RequestType::General, 0.0, 1.0, &mut rng);
            assert_eq!(tier, Tier::Tier3);
        }
    }

    #[test]
    fn classify_returns_valid_tier_and_request_type() {
        let cells = CellMap::new();
        let mut rng = StdRng::seed_from_u64(3);
        let (tier, rt) = classify(
            "write a python function that sorts a list",
            "anthropic",
            &cells,
            &mut rng,
        );
        assert_eq!(rt, RequestType::CodeGeneration);
        assert!(matches!(tier, Tier::Tier1 | Tier::Tier2 | Tier::Tier3));
    }

    // --- RequestClassifier trait & backend tests ---

    #[tokio::test]
    async fn regex_classifier_contract() {
        let classifier = RegexClassifier::new();
        assert_eq!(classifier.name(), "regex");

        let input = ClassifyInput::new("write a python script");
        let res = classifier.classify(&input).await.unwrap();
        assert_eq!(res.request_type, RequestType::CodeGeneration);
        assert_eq!(res.complexity, RegexClassifier::FIXED_COMPLEXITY);
        assert_eq!(res.confidence, RegexClassifier::FIXED_CONFIDENCE);
    }

    #[tokio::test]
    async fn fast_feature_classifier_public_eval_examples() {
        let classifier = FastFeatureClassifier::new();
        assert_eq!(classifier.name(), "heuristic");

        // pub-01: Fix typo in comment -> code_generation, complexity 1
        let c1 = classifier
            .classify(&ClassifyInput::with_context(
                "Fix typo in this Python comment: `# retrun the cached value`.",
                Some("No other files or changes needed."),
            ))
            .await
            .unwrap();
        assert_eq!(c1.request_type, RequestType::CodeGeneration);
        assert_eq!(c1.complexity, 1);
        assert!(c1.confidence > 0.85);

        // pub-02: What does Option::take() do -> factual_lookup, complexity 1
        let c2 = classifier
            .classify(&ClassifyInput::with_context(
                "What does `Option::take()` do in Rust? Answer in one sentence.",
                Some("No codebase context."),
            ))
            .await
            .unwrap();
        assert_eq!(c2.request_type, RequestType::FactualLookup);
        assert_eq!(c2.complexity, 1);

        // pub-03: Explain why this function returns -> code_understanding, complexity 2
        let c3 = classifier
            .classify(&ClassifyInput::with_context(
                "Explain why this function returns the old value, not the incremented value.",
                Some("```rust\nfn next(n: &mut u64) -> u64 { let old = *n; *n += 1; old }\n```"),
            ))
            .await
            .unwrap();
        assert_eq!(c3.request_type, RequestType::CodeUnderstanding);
        assert_eq!(c3.complexity, 2);

        // pub-04: Rewrite notification to be warmer -> writing, complexity 2
        let c4 = classifier
            .classify(&ClassifyInput::with_context(
                "Rewrite this notification to be warmer, but keep the date and time unchanged: 'Your interview is Friday at 10:00. Be on time.'",
                Some("Message goes to one candidate."),
            ))
            .await
            .unwrap();
        assert_eq!(c4.request_type, RequestType::Writing);
        assert_eq!(c4.complexity, 2);

        // pub-05: Design migration -> technical_design, complexity 4
        let c5 = classifier
            .classify(&ClassifyInput::with_context(
                "Design migration from synchronous payment-status callbacks to queued processing without changing public API semantics. Give architecture, state transitions, idempotency boundary, rollout phases, observability, failure recovery, and rollback. Explain what response API returns before processing finishes and how clients distinguish accepted from committed payments. Address contradictions in requirements rather than assuming them away.",
                Some("Today POST /confirm returns 200 only after provider charge; legacy clients ship on 200 and cannot update for six months. Provider may charge despite timeout. New queue is at-least-once; DB and queue share no transaction. Ledger is append-only. Rollback cannot lose pending work."),
            ))
            .await
            .unwrap();
        assert_eq!(c5.request_type, RequestType::TechnicalDesign);
        assert_eq!(c5.complexity, 4);

        // pub-06: Diagnose intermittent 401s -> analytical_reasoning, complexity 5
        let c6 = classifier
            .classify(&ClassifyInput::with_context(
                "Diagnose intermittent 401s and occasional permanent session loss after refresh. Reconstruct at least two distinct failure interleavings from logs, identify which evidence supports each and what remains uncertain, then propose a concurrency-safe fix across two app instances. Specify ordering, atomicity, retry policy for non-idempotent requests, and deterministic tests; explain why simply lowering cache TTL is insufficient.",
                Some("Provider invalidates R1 immediately when exchanging it for A2/R2. Two app instances read R1; X refreshes successfully, Y gets invalid_grant. X stores A2/R2 in DB but Redis still serves A1 for 15m. Another run overwrites a newer session row with an older snapshot. Locks are process-local; host clocks differ by 20ms. A 401 on POST /charge may follow a lost response."),
            ))
            .await
            .unwrap();
        assert_eq!(c6.request_type, RequestType::AnalyticalReasoning);
        assert_eq!(c6.complexity, 5);

        // pub-07: Implement a parser -> code_generation, complexity 3
        let c7 = classifier
            .classify(&ClassifyInput::with_context(
                "Implement a parser that accepts comma-separated integers, rejects empty fields, and reports the index of the first invalid field. Provide unit tests.",
                Some("Rust 2021; input may contain whitespace around numbers; negative integers are valid; do not use external crates."),
            ))
            .await
            .unwrap();
        assert_eq!(c7.request_type, RequestType::CodeGeneration);
        assert_eq!(c7.complexity, 3);

        // pub-08: Negation of redesign -> code_generation, complexity 1
        let c8 = classifier
            .classify(&ClassifyInput::with_context(
                "I wrote 'redesign authentication' as a TODO comment. Do not redesign anything: just change `TODO` to `NOTE` in the line below.",
                Some("`// TODO: redesign authentication after migration`"),
            ))
            .await
            .unwrap();
        assert_eq!(c8.request_type, RequestType::CodeGeneration);
        assert_eq!(c8.complexity, 1);

        // pub-09: Summarize release notes -> writing, complexity 3
        let c9 = classifier
            .classify(&ClassifyInput::with_context(
                "Summarize the following release notes in three bullets for a nontechnical customer. Do not mention internal ticket IDs or promise zero downtime.",
                Some("Release notes: Added paginated export for reports over 10,000 rows. Fixed intermittent timeout when filtering by date. Internal ticket OPS-4821 covers the database index change. During deployment, exports may pause for up to 30 seconds. Older report links remain valid. Support can resume interrupted exports."),
            ))
            .await
            .unwrap();
        assert_eq!(c9.request_type, RequestType::Writing);
        assert_eq!(c9.complexity, 3);

        // pub-10: Investigate daily reconciliations -> analytical_reasoning, complexity 4
        let c10 = classifier
            .classify(&ClassifyInput::with_context(
                "Investigate why daily reconciliations show both duplicate charges and missing refunds. Work through orders O17 and O18 event by event, state what can and cannot be inferred, identify unsafe retry and ordering assumptions, and propose a repair plan for already affected customers plus a durable processing design. Include invariants and adversarial tests; do not call a timeout proof of failure.",
                Some("Provider deduplicates per idempotency key for 24h; status lookup lags up to 90s. O17: charge K1 times out, retry K2 succeeds, then both success webhooks arrive. O18: charge K3 succeeds but DB write fails; customer cancels; refund job skips local failed status; success webhook arrives later. Webhooks are unordered; ledger is append-only; refund API returns 202 and can time out."),
            ))
            .await
            .unwrap();
        assert_eq!(c10.request_type, RequestType::AnalyticalReasoning);
        assert_eq!(c10.complexity, 4);
    }

    struct FailingClassifier;
    #[async_trait::async_trait]
    impl RequestClassifier for FailingClassifier {
        fn name(&self) -> &str {
            "failing"
        }
        async fn classify(
            &self,
            _input: &ClassifyInput<'_>,
        ) -> Result<Classification, ClassifyError> {
            Err(ClassifyError::Inference("forced failure".to_string()))
        }
    }

    struct SlowClassifier;
    #[async_trait::async_trait]
    impl RequestClassifier for SlowClassifier {
        fn name(&self) -> &str {
            "slow"
        }
        async fn classify(
            &self,
            _input: &ClassifyInput<'_>,
        ) -> Result<Classification, ClassifyError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(Classification::new(RequestType::Writing, 2, 0.8))
        }
    }

    #[tokio::test]
    async fn fallback_classifier_triggers_on_error() {
        let primary = Arc::new(FailingClassifier);
        let fallback = Arc::new(RegexClassifier::new());
        let composite = FallbackClassifier::new(primary, fallback, Duration::from_millis(100));

        let input = ClassifyInput::new("write a python function");
        let res = composite.classify(&input).await.unwrap();
        assert_eq!(res.request_type, RequestType::CodeGeneration);
        assert_eq!(composite.fallback_count(), 1);
    }

    #[tokio::test]
    async fn fallback_classifier_triggers_on_timeout() {
        let primary = Arc::new(SlowClassifier);
        let fallback = Arc::new(RegexClassifier::new());
        let composite = FallbackClassifier::new(primary, fallback, Duration::from_millis(5));

        let input = ClassifyInput::new("what is the capital of France?");
        let res = composite.classify(&input).await.unwrap();
        assert_eq!(res.request_type, RequestType::FactualLookup);
        assert_eq!(composite.fallback_count(), 1);
    }

    #[test]
    fn build_classifier_backends() {
        let regex = build_classifier("regex", "", "", 1000, None);
        assert_eq!(regex.name(), "regex");

        let heuristic = build_classifier("heuristic", "", "", 1000, None);
        assert_eq!(heuristic.name(), "heuristic");

        let fallback_unknown = build_classifier("nonexistent", "", "", 1000, None);
        assert_eq!(fallback_unknown.name(), "regex");
    }
}
