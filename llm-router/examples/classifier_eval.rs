//! Request classifier evaluation harness.
//!
//! Reads an evaluation set from the path in `EVAL_SET` and evaluates each example using
//! the configured [`RequestClassifier`] backend (default: `FastFeatureClassifier`, or
//! selected via `CLASSIFIER_BACKEND=regex|heuristic|hosted`).
//!
//! Outputs one JSONL line per case to the path in `OUT`:
//! ```json
//! {"id":"pub-01","request_type":"code_generation","complexity":1,"confidence":0.92,"latency_us":840}
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use nasiko_llm_router::routing::{
    ClassifyInput, FastFeatureClassifier, RegexClassifier, RequestClassifier, RequestType,
    build_classifier,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct EvalSet {
    #[serde(default)]
    examples: Vec<EvalExample>,
}

#[derive(Debug, Deserialize)]
struct EvalExample {
    id: String,
    query: String,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    request_type: Option<String>,
    #[serde(default)]
    complexity: Option<u8>,
    #[serde(default)]
    #[allow(dead_code)]
    tier_hypothesis: Option<String>,
}

#[derive(Debug, Serialize)]
struct EvalOutputLine<'a> {
    id: &'a str,
    request_type: &'a str,
    complexity: u8,
    confidence: f32,
    latency_us: u128,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eval_set_path = std::env::var("EVAL_SET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("classifier-eval.json"));

    let out_path = std::env::var("OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("classifier-out.jsonl"));

    let backend = std::env::var("CLASSIFIER_BACKEND").unwrap_or_else(|_| "heuristic".to_string());
    let model_path = std::env::var("CLASSIFIER_MODEL_PATH").unwrap_or_default();
    let endpoint = std::env::var("CLASSIFIER_ENDPOINT").unwrap_or_default();
    let timeout_ms = std::env::var("CLASSIFIER_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    eprintln!("=== Nasiko Request Classifier Eval Harness ===");
    eprintln!("EVAL_SET: {}", eval_set_path.display());
    eprintln!("OUT:      {}", out_path.display());
    eprintln!("BACKEND:  {}", backend);

    if !eval_set_path.exists() {
        eprintln!(
            "Error: evaluation dataset file not found at: {}",
            eval_set_path.display()
        );
        eprintln!("Download it via:");
        eprintln!(
            "  curl -fsSL https://registry.nasiko.dev/r/nasiko/classifier-eval -o {}",
            eval_set_path.display()
        );
        std::process::exit(1);
    }

    let file = File::open(&eval_set_path)?;
    let reader = BufReader::new(file);
    let eval_set: EvalSet = serde_json::from_reader(reader)?;

    // Load classifier once before the loop (per-call latency excludes model load)
    let load_start = Instant::now();
    let classifier: Arc<dyn RequestClassifier> = match backend.as_str() {
        "regex" => Arc::new(RegexClassifier::new()),
        "heuristic" | "local" | "rule" => Arc::new(FastFeatureClassifier::new()),
        "hosted" => build_classifier(
            "hosted",
            &model_path,
            &endpoint,
            timeout_ms,
            Some(reqwest::Client::new()),
        ),
        unknown => {
            eprintln!("Unknown backend '{}', using heuristic", unknown);
            Arc::new(FastFeatureClassifier::new())
        }
    };
    let load_duration = load_start.elapsed();
    eprintln!(
        "Loaded classifier backend '{}' in {:?}",
        classifier.name(),
        load_duration
    );

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let out_file = File::create(&out_path)?;
    let mut writer = BufWriter::new(out_file);

    let mut total_cases = 0usize;
    let mut rt_correct = 0usize;
    let mut complexity_exact = 0usize;
    let mut total_complexity_err = 0u64;
    let mut total_confidence = 0.0f64;
    let mut latencies_us = Vec::with_capacity(eval_set.examples.len());

    let eval_start = Instant::now();

    for example in &eval_set.examples {
        let input = ClassifyInput::with_context(&example.query, example.context.as_deref());

        let call_start = Instant::now();
        let classification = match classifier.classify(&input).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "Warning: classifier error on {}: {}; falling back to baseline",
                    example.id, e
                );
                RegexClassifier::new()
                    .classify(&input)
                    .await
                    .unwrap_or_else(|_| {
                        nasiko_llm_router::routing::Classification::new(
                            RequestType::General,
                            3,
                            0.5,
                        )
                    })
            }
        };
        let call_latency = call_start.elapsed().as_micros();
        latencies_us.push(call_latency);

        total_cases += 1;
        total_confidence += classification.confidence as f64;

        if let Some(ref expected_rt) = example.request_type
            && classification.request_type.as_str() == expected_rt
        {
            rt_correct += 1;
        }

        if let Some(expected_comp) = example.complexity {
            if classification.complexity == expected_comp {
                complexity_exact += 1;
            }
            total_complexity_err +=
                (classification.complexity as i64 - expected_comp as i64).unsigned_abs();
        }

        let line = EvalOutputLine {
            id: &example.id,
            request_type: classification.request_type.as_str(),
            complexity: classification.complexity,
            confidence: classification.confidence,
            latency_us: call_latency,
        };

        serde_json::to_writer(&mut writer, &line)?;
        writer.write_all(b"\n")?;
    }

    writer.flush()?;
    let total_eval_duration = eval_start.elapsed();

    // Summary statistics
    latencies_us.sort();
    let p50 = latencies_us
        .get(latencies_us.len() / 2)
        .copied()
        .unwrap_or(0);
    let p95_idx =
        ((latencies_us.len() as f64 * 0.95) as usize).min(latencies_us.len().saturating_sub(1));
    let p95 = latencies_us.get(p95_idx).copied().unwrap_or(0);
    let mean_lat = if !latencies_us.is_empty() {
        latencies_us.iter().sum::<u128>() / latencies_us.len() as u128
    } else {
        0
    };

    eprintln!("\n=== Evaluation Results ===");
    eprintln!("Total examples evaluated: {}", total_cases);
    eprintln!("Total runtime:            {:?}", total_eval_duration);
    if total_cases > 0 {
        eprintln!(
            "Request type accuracy:    {}/{} ({:.1}%)",
            rt_correct,
            total_cases,
            (rt_correct as f64 / total_cases as f64) * 100.0
        );
        eprintln!(
            "Complexity exact match:   {}/{} ({:.1}%)",
            complexity_exact,
            total_cases,
            (complexity_exact as f64 / total_cases as f64) * 100.0
        );
        eprintln!(
            "Complexity MAE:           {:.3}",
            total_complexity_err as f64 / total_cases as f64
        );
        eprintln!(
            "Mean confidence:          {:.3}",
            total_confidence / total_cases as f64
        );
    }
    eprintln!(
        "Latency (us):             mean={} us, p50={} us, p95={} us",
        mean_lat, p50, p95
    );
    eprintln!("Output written to:        {}", out_path.display());

    Ok(())
}
