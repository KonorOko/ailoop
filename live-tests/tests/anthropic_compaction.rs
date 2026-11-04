//! Compaction against the real Anthropic API.
//!
//! Requires `ANTHROPIC_API_KEY`, from the environment or a `.env` file.
//! `AILOOP_LIVE_MODEL` picks the model. Each test prints what it got, so
//! run with `--nocapture` and read the summaries: the assertions only
//! catch clear failures.
//!
//! The `cheap_` tests send short texts, plus one oversized request that
//! the API rejects:
//!
//! ```text
//! cargo test -p ailoop-live-tests -- --ignored --nocapture cheap_
//! ```
//!
//! The `costly_` test bills every part of a history larger than the
//! context window, so it only runs with an explicit size. The rejected
//! request prints `prompt is too long: X tokens > Y maximum` for
//! 12 000 000 characters, so 1.3 times the limit is about
//! `12_000_000 * 1.3 * Y / X` characters:
//!
//! ```text
//! AILOOP_LIVE_OVERFLOW_CHARS=<characters> \
//!   cargo test -p ailoop-live-tests -- --ignored --nocapture costly_
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use ailoop::{
    AbortReason, CancellationToken, ChatRequest, CompactionError, CompletionModel, Conversation,
    FinishReason, History, HistoryBuilder, Message, RunOptions, StreamChunk, SummarizeStrategy,
    async_trait,
};
use ailoop_anthropic::{AnthropicChatModel, AnthropicClient};
use futures::StreamExt;
use futures::stream::BoxStream;

const DEFAULT_MODEL: &str = "claude-haiku-5-5";
/// Size of the rejected request, well past a window of a million tokens
/// even at seven characters per token. Under the API's 32 MB body limit.
const REJECTED_CHARS: usize = 12_000_000;
const CODE_NAME: &str = "BLUE-HERON-47";
const LATE_CODE_NAME: &str = "RED-OTTER-92";

fn model() -> AnthropicChatModel {
    let name = std::env::var("AILOOP_LIVE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    println!("model: {name}");
    AnthropicClient::from_env()
        .expect("ANTHROPIC_API_KEY is required")
        .model(name)
}

fn summarizing(budget: usize) -> HistoryBuilder {
    History::builder(budget)
        .preserve_n_last(1)
        .strategy(Box::new(SummarizeStrategy::new(Arc::new(model()))))
}

fn history_with(messages: Vec<Message>) -> History {
    let mut history = summarizing(1_000_000).build();
    history.extend(messages);
    history
}

/// A conversation whose oldest part states a fact and whose newest part
/// asks for things nobody answered. A summarizer that answers the
/// transcript instead of describing it gives "391" or a poem.
fn transcript() -> Vec<Message> {
    vec![
        Message::user(format!("The project's code name is {CODE_NAME}.")),
        Message::assistant_text("Understood, I will use that code name."),
        Message::user(
            "Before we continue: what is 17 times 23? Also write a four-line poem about the sea.",
        ),
        Message::assistant_text("I will get to both in a moment."),
        Message::user("Thanks. Moving on to the deployment plan."),
    ]
}

/// `chars` characters of work log across many messages, with a fact at
/// each end of the part that gets summarized.
fn oversized_history(chars: usize) -> Vec<Message> {
    const JOBS: [&str; 6] = ["billing", "ingest", "search", "reports", "backup", "auth"];
    const OWNERS: [&str; 5] = ["Ana", "Luis", "Mei", "Omar", "Sofia"];
    const ALERTS: [&str; 4] = ["latency", "disk usage", "error rate", "queue depth"];
    const PER_MESSAGE: usize = 40_000;

    let count = (chars / PER_MESSAGE).max(4) & !1;
    let mut messages = Vec::with_capacity(count + 1);
    let mut n: u64 = 7;
    for i in 0..count {
        let mut text = match i {
            0 => format!("The project's code name is {CODE_NAME}.\n"),
            _ if i == count - 2 => format!("The backup's code name is {LATE_CODE_NAME}.\n"),
            _ => String::new(),
        };
        while text.len() < PER_MESSAGE {
            n = n.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let r = (n >> 33) as usize;
            text.push_str(&format!(
                "Ticket OPS-{}: the {} job on host-{} took {} ms after {} retries; {} is following the {} alert.\n",
                r % 90_000,
                JOBS[r % JOBS.len()],
                r % 40,
                r % 9_000,
                r % 4,
                OWNERS[r % OWNERS.len()],
                ALERTS[r % ALERTS.len()],
            ));
        }
        messages.push(if i % 2 == 0 {
            Message::user(text)
        } else {
            Message::assistant_text(text)
        });
    }
    messages.push(Message::user("Carry on."));
    messages
}

/// A model that prints each answer it streams, to see what every
/// request of a split produced.
struct Logged<M>(M);

#[async_trait]
impl<M: CompletionModel> CompletionModel for Logged<M> {
    type Error = M::Error;

    fn name(&self) -> &str {
        self.0.name()
    }

    fn model(&self) -> &str {
        self.0.model()
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        let stream = self.0.chat_stream(req).await.inspect_err(|error| {
            println!("--- request rejected: {error}");
        })?;
        let mut answer = String::new();
        Ok(stream
            .inspect(move |chunk| match chunk {
                Ok(StreamChunk::TextDelta { delta }) => answer.push_str(delta),
                Ok(StreamChunk::TurnFinished { usage, .. }) => {
                    println!(
                        "--- request answered, {} input tokens, early fact: {}, late fact: {}\n{answer}",
                        usage.input_tokens,
                        answer.contains(CODE_NAME),
                        answer.contains(LATE_CODE_NAME),
                    );
                }
                Ok(_) => {}
                Err(error) => println!("--- stream failed: {error}"),
            })
            .boxed())
    }
}

/// The history's summary as printable text.
fn rendered(history: &History) -> String {
    format!("{:?}", history.messages()[0]).replace("\\n", "\n")
}

/// The prefix being summarized ends in an assistant message. Models that
/// do not accept prefill answer a request ending that way with a 400.
#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn cheap_summary_of_a_prefix_ending_in_assistant_is_accepted() {
    let mut history = history_with(vec![
        Message::user("Please remember that the deadline is Friday."),
        Message::assistant_text("Noted: the deadline is Friday."),
        Message::user("Good."),
    ]);

    let stats = history.force_compact().await.expect("request is accepted");

    println!("{stats:?}\n{}", rendered(&history));
    assert!(stats.after < stats.before, "{stats:?}");
    assert!(stats.usage.input_tokens > 0, "{stats:?}");
    assert!(stats.usage.output_tokens > 0, "{stats:?}");
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn cheap_summary_describes_the_transcript_instead_of_answering_it() {
    let mut history = history_with(transcript());

    history.force_compact().await.expect("summary");

    let summary = rendered(&history);
    println!("{summary}");
    assert!(summary.contains(CODE_NAME), "lost the fact: {summary}");
    assert!(!summary.contains("391"), "answered the question: {summary}");
}

/// The counterpart: a result somebody did state is state worth keeping.
#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn cheap_summary_keeps_the_results_the_transcript_states() {
    let mut history = history_with(vec![
        Message::user("How many rows did the migration touch?"),
        Message::assistant_text("It touched 48213 rows and failed on /srv/data/orders.csv."),
        Message::user("Thanks."),
    ]);

    history.force_compact().await.expect("summary");

    let summary = rendered(&history);
    println!("{summary}");
    assert!(
        summary.contains("48213") || summary.contains("48,213"),
        "lost the result: {summary}"
    );
    assert!(summary.contains("/srv/data/orders.csv"), "{summary}");
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn cheap_run_compacts_first_and_counts_the_spend() {
    let mut chat = Conversation::builder(model())
        .history(summarizing(10))
        .max_tokens(64)
        .build()
        .unwrap();
    chat.extend_messages(transcript());

    let mut stream = chat
        .stream_with_options(
            "What is the project's code name? Reply with only the name.",
            RunOptions::new(),
        )
        .await
        .expect("run starts");

    let mut order = Vec::new();
    let mut compaction = None;
    let mut answer = String::new();
    let mut total = None;
    while let Some(chunk) = stream.next().await {
        match chunk.expect("no error") {
            StreamChunk::RunStarted { .. } => order.push("started"),
            StreamChunk::HistoryCompacted { usage, .. } => {
                order.push("compacted");
                compaction = Some(usage);
            }
            StreamChunk::StepStarted { .. } => order.push("step"),
            StreamChunk::TextDelta { delta } => answer.push_str(&delta),
            StreamChunk::RunFinished { usage, reason, .. } => {
                assert_eq!(reason, FinishReason::EndTurn);
                total = Some(usage);
            }
            _ => {}
        }
    }

    let compaction = compaction.expect("the run compacted");
    let total = total.expect("the run finished");
    println!("{order:?}\ncompaction: {compaction:?}\ntotal: {total:?}\nanswer: {answer}");
    assert_eq!(order[..2], ["started", "compacted"]);
    assert!(compaction.input_tokens > 0, "{compaction:?}");
    assert!(total.input_tokens > compaction.input_tokens, "{total:?}");
    assert!(total.output_tokens > compaction.output_tokens, "{total:?}");
    assert!(answer.contains(CODE_NAME), "fact lost: {answer}");
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn cheap_cancelling_during_the_summary_aborts_and_keeps_the_history() {
    let mut chat = Conversation::builder(model())
        .history(summarizing(10))
        .build()
        .unwrap();
    chat.extend_messages(transcript());
    let before = chat.messages().len();

    let token = CancellationToken::new();
    let canceller = token.clone();
    tokio::spawn(async move {
        // Well before a summary can finish streaming.
        tokio::time::sleep(Duration::from_millis(300)).await;
        canceller.cancel();
    });

    let started = Instant::now();
    let mut stream = chat
        .stream_with_options("Carry on.", RunOptions::new().cancellation(token))
        .await
        .expect("run starts");
    let mut finish = None;
    while let Some(chunk) = stream.next().await {
        match chunk.expect("an abort is not an error") {
            StreamChunk::HistoryCompacted { .. } => panic!("the summary beat the cancellation"),
            StreamChunk::RunFinished { reason, .. } => finish = Some(reason),
            _ => {}
        }
    }
    drop(stream);

    println!("finished in {:?} with {finish:?}", started.elapsed());
    assert_eq!(finish, Some(FinishReason::Aborted(AbortReason::Cancelled)),);
    // Everything is still there, plus the kickoff message.
    assert_eq!(chat.messages().len(), before + 1);
}

/// The first oversized request fails the compaction. The message it
/// prints holds the model's real limit, which sizes the costly test.
#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn cheap_oversized_prefix_with_one_request_reports_the_overflow() {
    let strategy = SummarizeStrategy::new(Arc::new(model())).with_max_requests(1);
    let mut history = History::builder(1_000_000)
        .preserve_n_last(1)
        .strategy(Box::new(strategy))
        .build();
    history.extend(oversized_history(REJECTED_CHARS));
    let before = history.messages().len();

    let error = history.force_compact().await.expect_err("too large");

    println!("{error}");
    assert!(
        matches!(error, CompactionError::SummarizationFailed { .. }),
        "{error:?}"
    );
    assert!(error.is_context_overflow(), "{error:?}");
    assert_eq!(history.messages().len(), before, "history must not change");
}

#[tokio::test]
#[ignore = "bills a history larger than the context window"]
async fn costly_oversized_prefix_is_split_until_it_fits() {
    let chars: usize = std::env::var("AILOOP_LIVE_OVERFLOW_CHARS")
        .expect("set AILOOP_LIVE_OVERFLOW_CHARS; see the top of this file")
        .parse()
        .expect("a number of characters");
    let mut history = History::builder(1_000_000)
        .preserve_n_last(1)
        .strategy(Box::new(SummarizeStrategy::new(Arc::new(Logged(model())))))
        .build();
    history.extend(oversized_history(chars));

    let started = Instant::now();
    let stats = history.force_compact().await.expect("the split completes");

    let summary = rendered(&history);
    println!("{stats:?} in {:?}\n{summary}", started.elapsed());
    // Each end of the history lands in a different request, so the early
    // fact only survives by being carried through every summary.
    assert!(summary.contains(CODE_NAME), "lost the early fact");
    assert!(summary.contains(LATE_CODE_NAME), "lost the late fact");
    assert!(stats.after < stats.before, "{stats:?}");
    assert!(stats.usage.input_tokens > 0, "{stats:?}");
}
