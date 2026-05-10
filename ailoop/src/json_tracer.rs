//! NDJSON file sink for run/step/tool/chunk events.
//!
//! `JsonTracer` is the file-based counterpart to `TracingMiddleware`.
//! Where `TracingMiddleware` routes through the `tracing` crate, this
//! middleware appends one JSON object per line to a caller-supplied
//! path — useful for post-mortem inspection and for hosts that ban the
//! `tracing` dependency. No feature flag: the sink uses only `serde_json`,
//! `std::fs`, and a tokio `Mutex`, all already present in this crate.
//!
//! Every line shares an envelope:
//!
//! ```text
//! { "schema": 1, "ts_ms": <u64>, "kind": "<event>", ...payload }
//! ```
//!
//! `schema: 1` is the format version — bump it on any breaking change to
//! a payload shape so downstream consumers can branch. `ts_ms` is
//! milliseconds since the Unix epoch (no timezone ambiguity, no extra
//! dep). Run/step correlation comes via `run_id` / `step_id` fields when
//! the underlying hook carries them.
//!
//! By default deltas are logged as character counts only and tool
//! `args` / `result` payloads are summarised — full content is reserved
//! for [`JsonTracer::verbose`], which is documented as dev-only because
//! it leaks user data into the log.
//!
//! IO failures (a disk that fills up mid-run, a reader holding the file
//! open in a way that breaks the writer) never panic and never abort
//! the run. They surface as a single `eprintln!` per `JsonTracer`
//! instance — a telemetry middleware that takes down a production run
//! is worse than one that drops events.

use std::fs::File;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ailoop_core::{
    ChatMiddleware, ChatRequest, FinishReason, HookAction, Message, RunConfig, RunId, StepId,
    StreamChunk, ToolDecision, ToolResultContent, Usage,
};
use serde_json::{Value, json};
use tokio::sync::Mutex;

/// Current schema version embedded on every line. Bump on any
/// payload-shape break so downstream consumers can match on it.
const SCHEMA_VERSION: u32 = 1;

/// Append-only NDJSON sink. Holds the open file behind a tokio
/// `Mutex` so concurrent runs sharing the same `Arc<JsonTracer>` cannot
/// interleave bytes within a line. The inner handle is `std::fs::File`:
/// the writes are small (one JSON object) and `tokio::fs::File` buffers
/// internally, which would silently swallow events on drop without an
/// explicit `flush().await` between every emit.
pub struct JsonTracer {
    file: Mutex<File>,
    error_emitted: AtomicBool,
    verbose: bool,
}

impl JsonTracer {
    /// Open `path` for append (creating it if missing) and return a
    /// tracer that logs counts only — delta lengths instead of contents,
    /// no tool `args` / `result` bodies. Fails synchronously if the file
    /// cannot be opened so a misconfigured path surfaces at construction
    /// rather than silently dropping every event.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::build(path, false)
    }

    /// Same as [`new`](Self::new) but also writes the full content of
    /// text/reasoning deltas and tool `args` / `result` payloads. Intended
    /// for local debugging only — these lines contain user data, model
    /// output, and tool inputs verbatim and should not be shipped to a
    /// shared destination.
    pub fn verbose(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::build(path, true)
    }

    fn build(path: impl AsRef<Path>, verbose: bool) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        Ok(Self {
            file: Mutex::new(file),
            error_emitted: AtomicBool::new(false),
            verbose,
        })
    }

    async fn emit(&self, kind: &str, mut payload: serde_json::Map<String, Value>) {
        payload.insert("schema".into(), json!(SCHEMA_VERSION));
        payload.insert("ts_ms".into(), json!(now_ms()));
        payload.insert("kind".into(), json!(kind));
        let value = Value::Object(payload);
        let mut bytes = match serde_json::to_vec(&value) {
            Ok(b) => b,
            Err(e) => {
                self.warn_once(format_args!("serialize: {e}"));
                return;
            }
        };
        bytes.push(b'\n');
        let mut guard = self.file.lock().await;
        if let Err(e) = guard.write_all(&bytes) {
            self.warn_once(format_args!("write: {e}"));
        }
    }

    fn warn_once(&self, args: std::fmt::Arguments<'_>) {
        if !self.error_emitted.swap(true, Ordering::Relaxed) {
            eprintln!("JsonTracer IO failure (further errors suppressed): {args}");
        }
    }
}

impl std::fmt::Debug for JsonTracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonTracer")
            .field("verbose", &self.verbose)
            .finish_non_exhaustive()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn finish_reason_str(r: &FinishReason) -> &'static str {
    match r {
        FinishReason::EndTurn => "end_turn",
        FinishReason::ToolUse => "tool_use",
        FinishReason::MaxTokens => "max_tokens",
        FinishReason::StopSequence => "stop_sequence",
        FinishReason::Aborted(_) => "aborted",
        FinishReason::Other(_) => "other",
        _ => "unknown",
    }
}

fn finish_reason_payload(r: &FinishReason) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("kind".into(), json!(finish_reason_str(r)));
    if let FinishReason::Aborted(reason) | FinishReason::Other(reason) = r {
        o.insert("detail".into(), json!(reason));
    }
    Value::Object(o)
}

fn usage_payload(u: &Usage) -> Value {
    json!({
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cached_input_tokens": u.cached_input_tokens,
        "cache_creation_input_tokens": u.cache_creation_input_tokens,
        "cache_creation_5m_tokens": u.cache_creation_5m_tokens,
        "cache_creation_1h_tokens": u.cache_creation_1h_tokens,
    })
}

fn tool_result_outcome(r: &ToolResultContent) -> &'static str {
    match r {
        ToolResultContent::Text(_) => "text",
        ToolResultContent::Error(_) => "error",
        _ => "unknown",
    }
}

fn tool_result_body(r: &ToolResultContent) -> &str {
    match r {
        ToolResultContent::Text(s) | ToolResultContent::Error(s) => s,
        _ => "",
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for JsonTracer {
    async fn on_run_start(
        &self,
        run_id: &RunId,
        messages: &[Message],
        config: &RunConfig,
    ) -> HookAction {
        let mut p = serde_json::Map::new();
        p.insert("run_id".into(), json!(run_id.to_string()));
        p.insert("messages".into(), json!(messages.len()));
        p.insert("max_iterations".into(), json!(config.max_iterations));
        p.insert("max_tokens".into(), json!(config.max_tokens));
        self.emit("run_started", p).await;
        HookAction::Continue
    }

    async fn on_chat_request(&self, run_id: &RunId, step_id: &StepId, req: &mut ChatRequest) {
        let mut p = serde_json::Map::new();
        p.insert("run_id".into(), json!(run_id.to_string()));
        p.insert("step_id".into(), json!(step_id.to_string()));
        p.insert("messages".into(), json!(req.messages.len()));
        p.insert(
            "tools".into(),
            json!(req.tools.as_ref().map(|t| t.len()).unwrap_or(0)),
        );
        p.insert("max_tokens".into(), json!(req.max_tokens));
        self.emit("chat_request", p).await;
    }

    async fn on_chunk(&self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::TextDelta { delta } => {
                let mut p = serde_json::Map::new();
                p.insert("chars".into(), json!(delta.len()));
                if self.verbose {
                    p.insert("text".into(), json!(delta));
                }
                self.emit("text_delta", p).await;
            }
            StreamChunk::ReasoningDelta { delta } => {
                let mut p = serde_json::Map::new();
                p.insert("chars".into(), json!(delta.len()));
                if self.verbose {
                    p.insert("text".into(), json!(delta));
                }
                self.emit("reasoning_delta", p).await;
            }
            StreamChunk::ReasoningEnd { signature } => {
                let mut p = serde_json::Map::new();
                p.insert("has_signature".into(), json!(signature.is_some()));
                self.emit("reasoning_end", p).await;
            }
            StreamChunk::RedactedReasoningBlock { data } => {
                let mut p = serde_json::Map::new();
                p.insert("bytes".into(), json!(data.len()));
                self.emit("redacted_reasoning", p).await;
            }
            StreamChunk::ToolCallStart { id, name } => {
                let mut p = serde_json::Map::new();
                p.insert("call_id".into(), json!(id));
                p.insert("name".into(), json!(name));
                self.emit("tool_call_start", p).await;
            }
            StreamChunk::ToolCallArgsDelta { .. } => {
                // Suppressed for parity with TracingMiddleware: the
                // accumulated args land on `ToolCallEnd` and per-delta
                // entries would dominate the log.
            }
            StreamChunk::ToolCallEnd { id, name, args } => {
                let mut p = serde_json::Map::new();
                p.insert("call_id".into(), json!(id));
                p.insert("name".into(), json!(name));
                if self.verbose {
                    p.insert("args".into(), args.clone());
                }
                self.emit("tool_call_end", p).await;
            }
            StreamChunk::ToolResult {
                run_id,
                step_id,
                call_id,
                content,
            } => {
                let mut p = serde_json::Map::new();
                p.insert("run_id".into(), json!(run_id.to_string()));
                p.insert("step_id".into(), json!(step_id.to_string()));
                p.insert("call_id".into(), json!(call_id));
                p.insert("outcome".into(), json!(tool_result_outcome(content)));
                if self.verbose {
                    p.insert("body".into(), json!(tool_result_body(content)));
                }
                self.emit("tool_result", p).await;
            }
            StreamChunk::StepStarted {
                run_id,
                step_id,
                iteration,
            } => {
                let mut p = serde_json::Map::new();
                p.insert("run_id".into(), json!(run_id.to_string()));
                p.insert("step_id".into(), json!(step_id.to_string()));
                p.insert("iteration".into(), json!(*iteration));
                self.emit("step_started", p).await;
            }
            StreamChunk::StepFinished {
                run_id,
                step_id,
                iteration,
                new_messages_so_far,
            } => {
                let mut p = serde_json::Map::new();
                p.insert("run_id".into(), json!(run_id.to_string()));
                p.insert("step_id".into(), json!(step_id.to_string()));
                p.insert("iteration".into(), json!(*iteration));
                p.insert("new_messages".into(), json!(new_messages_so_far.len()));
                self.emit("step_finished", p).await;
            }
            StreamChunk::TurnFinished {
                reason,
                usage,
                service_tier,
            } => {
                let mut p = serde_json::Map::new();
                p.insert("reason".into(), finish_reason_payload(reason));
                p.insert("usage".into(), usage_payload(usage));
                p.insert("service_tier".into(), json!(service_tier));
                self.emit("turn_finished", p).await;
            }
            StreamChunk::HistoryCompacted {
                run_id,
                before_count,
                after_count,
                strategy,
            } => {
                let mut p = serde_json::Map::new();
                p.insert("run_id".into(), json!(run_id.to_string()));
                p.insert("before".into(), json!(*before_count));
                p.insert("after".into(), json!(*after_count));
                p.insert("strategy".into(), json!(*strategy));
                self.emit("history_compacted", p).await;
            }
            // RunStarted / RunFinished have dedicated hooks with richer
            // context (config, messages, usage) — log there, not here.
            // Wildcard covers the `#[non_exhaustive]` future variants.
            StreamChunk::RunStarted { .. } | StreamChunk::RunFinished { .. } => {}
            _ => {}
        }
    }

    async fn on_run_finished(
        &self,
        run_id: &RunId,
        reason: &FinishReason,
        usage: &Usage,
        new_messages: &[Message],
    ) {
        let mut p = serde_json::Map::new();
        p.insert("run_id".into(), json!(run_id.to_string()));
        p.insert("reason".into(), finish_reason_payload(reason));
        p.insert("usage".into(), usage_payload(usage));
        p.insert("new_messages".into(), json!(new_messages.len()));
        self.emit("run_finished", p).await;
    }

    async fn on_run_error(&self, run_id: &RunId, err: &(dyn std::error::Error + Send + Sync)) {
        let mut p = serde_json::Map::new();
        p.insert("run_id".into(), json!(run_id.to_string()));
        p.insert("error".into(), json!(err.to_string()));
        self.emit("run_error", p).await;
    }

    async fn on_before_tool_call(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
    ) -> ToolDecision {
        let mut p = serde_json::Map::new();
        p.insert("run_id".into(), json!(run_id.to_string()));
        p.insert("step_id".into(), json!(step_id.to_string()));
        p.insert("name".into(), json!(name));
        if self.verbose {
            p.insert("args".into(), args.clone());
        }
        self.emit("before_tool_call", p).await;
        ToolDecision::Continue
    }

    async fn on_after_tool_call(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
        result: &ToolResultContent,
    ) {
        let mut p = serde_json::Map::new();
        p.insert("run_id".into(), json!(run_id.to_string()));
        p.insert("step_id".into(), json!(step_id.to_string()));
        p.insert("name".into(), json!(name));
        p.insert("outcome".into(), json!(tool_result_outcome(result)));
        if self.verbose {
            p.insert("args".into(), args.clone());
            p.insert("result".into(), json!(tool_result_body(result)));
        }
        self.emit("after_tool_call", p).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::testing::ScriptedModel;
    use ailoop_core::{Message, RunConfig};
    use ailoop_tools::ToolRegistry;
    use futures::StreamExt;
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn read_lines(path: &PathBuf) -> Vec<Value> {
        let raw = std::fs::read_to_string(path).expect("log file should exist");
        raw.lines()
            .map(|l| {
                serde_json::from_str::<Value>(l).unwrap_or_else(|e| {
                    panic!("each line must be valid JSON; got `{l}` (err: {e})")
                })
            })
            .collect()
    }

    fn close_and_read(tracer: JsonTracer, path: PathBuf) -> Vec<Value> {
        // Drop the tracer to release the file handle (and let the OS
        // flush the page cache for the read below).
        drop(tracer);
        read_lines(&path)
    }

    #[tokio::test]
    async fn schema_envelope_present_on_every_line() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace.ndjson");
        let tracer = JsonTracer::new(&path).expect("open should succeed");
        let run_id = RunId::new();
        let step_id = StepId::new();

        tracer
            .on_run_start(&run_id, &[Message::user("hi")], &RunConfig::default())
            .await;
        tracer
            .on_before_tool_call(&run_id, &step_id, "echo", &Value::Null)
            .await;
        tracer
            .on_after_tool_call(
                &run_id,
                &step_id,
                "echo",
                &Value::Null,
                &ToolResultContent::Text("ok".into()),
            )
            .await;
        tracer
            .on_chunk(&StreamChunk::HistoryCompacted {
                run_id: run_id.clone(),
                before_count: 12,
                after_count: 4,
                strategy: "truncate",
            })
            .await;
        tracer
            .on_run_finished(&run_id, &FinishReason::EndTurn, &Usage::default(), &[])
            .await;

        let lines = close_and_read(tracer, path);
        assert_eq!(lines.len(), 5);
        for line in &lines {
            let obj = line.as_object().expect("line must be an object");
            assert_eq!(obj.get("schema"), Some(&json!(SCHEMA_VERSION)));
            assert!(obj.get("ts_ms").and_then(|v| v.as_u64()).is_some());
            assert!(obj.get("kind").and_then(|v| v.as_str()).is_some());
        }
        let kinds: Vec<&str> = lines.iter().map(|l| l["kind"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            vec![
                "run_started",
                "before_tool_call",
                "after_tool_call",
                "history_compacted",
                "run_finished",
            ]
        );
        let run_id_str = run_id.to_string();
        for line in &lines {
            if let Some(rid) = line.get("run_id") {
                assert_eq!(rid.as_str(), Some(run_id_str.as_str()));
            }
        }
    }

    #[tokio::test]
    async fn new_returns_err_when_path_unopenable() {
        let dir = tempdir().unwrap();
        // Pointing at a path whose parent is a regular file (not a
        // directory) is guaranteed to fail at open without ever needing
        // to write a byte. Cheaper than a read-only filesystem and works
        // identically across CI hosts.
        let blocker = dir.path().join("not-a-dir");
        std::fs::write(&blocker, b"x").unwrap();
        let bad = blocker.join("trace.ndjson");
        let result = JsonTracer::new(&bad);
        assert!(result.is_err(), "expected open to fail, got Ok at {bad:?}");
    }

    #[tokio::test]
    async fn default_logs_counts_verbose_logs_full_content() {
        let dir = tempdir().unwrap();
        let default_path = dir.path().join("default.ndjson");
        let verbose_path = dir.path().join("verbose.ndjson");

        let default_tracer = JsonTracer::new(&default_path).unwrap();
        let verbose_tracer = JsonTracer::verbose(&verbose_path).unwrap();
        let chunk = StreamChunk::TextDelta {
            delta: "hello world".into(),
        };
        default_tracer.on_chunk(&chunk).await;
        verbose_tracer.on_chunk(&chunk).await;

        let default_lines = close_and_read(default_tracer, default_path);
        let verbose_lines = close_and_read(verbose_tracer, verbose_path);

        assert_eq!(default_lines[0]["chars"], json!("hello world".len()));
        assert!(
            default_lines[0].get("text").is_none(),
            "default mode must not include delta text"
        );
        assert_eq!(verbose_lines[0]["chars"], json!("hello world".len()));
        assert_eq!(verbose_lines[0]["text"], json!("hello world"));
    }

    #[tokio::test]
    async fn verbose_includes_tool_args_and_result() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace.ndjson");
        let tracer = JsonTracer::verbose(&path).unwrap();
        let run_id = RunId::new();
        let step_id = StepId::new();
        let args = json!({ "city": "Lima" });

        tracer
            .on_before_tool_call(&run_id, &step_id, "weather", &args)
            .await;
        tracer
            .on_after_tool_call(
                &run_id,
                &step_id,
                "weather",
                &args,
                &ToolResultContent::Text("sunny".into()),
            )
            .await;

        let lines = close_and_read(tracer, path);
        assert_eq!(lines[0]["args"], args);
        assert_eq!(lines[1]["args"], args);
        assert_eq!(lines[1]["result"], json!("sunny"));
    }

    #[tokio::test]
    async fn engine_run_writes_lifecycle_events_in_order() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace.ndjson");
        let tracer: Arc<dyn ChatMiddleware> =
            Arc::new(JsonTracer::new(&path).expect("open should succeed"));

        let model = ScriptedModel::new([vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }]]);
        let registry = ToolRegistry::new();
        let run_id = RunId::new();
        let mut config = RunConfig::default();
        config.middlewares = vec![Arc::clone(&tracer)];
        config.run_id = Some(run_id.clone());

        let stream = crate::run_chat(&model, vec![Message::user("hi")], &registry, config)
            .await
            .expect("run_chat should start");
        let _: Vec<_> = stream.collect().await;

        // Drop the only remaining strong ref so the file is closed
        // before we read.
        drop(tracer);

        let lines = read_lines(&path);
        let kinds: Vec<&str> = lines.iter().map(|l| l["kind"].as_str().unwrap()).collect();
        for required in [
            "run_started",
            "step_started",
            "step_finished",
            "run_finished",
        ] {
            assert!(
                kinds.contains(&required),
                "missing `{required}` in log; saw: {kinds:?}"
            );
        }
        let run_id_str = run_id.to_string();
        let any_with_run_id = lines
            .iter()
            .any(|l| l.get("run_id").and_then(|v| v.as_str()) == Some(run_id_str.as_str()));
        assert!(
            any_with_run_id,
            "no line carried run_id `{run_id_str}`; saw: {lines:?}"
        );
    }

    #[tokio::test]
    async fn concurrent_runs_share_tracer_without_interleaving_lines() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace.ndjson");
        let tracer: Arc<JsonTracer> =
            Arc::new(JsonTracer::new(&path).expect("open should succeed"));

        // Each task pumps a stream of synthetic events through the
        // shared tracer; the Mutex<File> guarantees that even when they
        // race, no two write_all calls interleave bytes.
        let make_task = |t: Arc<JsonTracer>, label: usize| async move {
            let run_id = RunId::new();
            let step_id = StepId::new();
            for i in 0..50 {
                t.on_chunk(&StreamChunk::TextDelta {
                    delta: format!("task{label}-msg{i}"),
                })
                .await;
                t.on_chunk(&StreamChunk::StepStarted {
                    run_id: run_id.clone(),
                    step_id: step_id.clone(),
                    iteration: i,
                })
                .await;
            }
        };

        tokio::join!(
            make_task(Arc::clone(&tracer), 0),
            make_task(Arc::clone(&tracer), 1),
            make_task(Arc::clone(&tracer), 2),
        );
        drop(tracer);

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 3 * 50 * 2);
        // Every line must still parse as a complete JSON object —
        // interleaving would corrupt that.
        for line in &lines {
            assert!(
                line.is_object(),
                "concurrent writes corrupted a line: {line}"
            );
            assert_eq!(line["schema"], json!(SCHEMA_VERSION));
        }
    }
}
