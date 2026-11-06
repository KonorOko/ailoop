//! End-to-end tests against a fake HTTP server: requests go through
//! reqwest, SSE parsing and the state machine exactly as in production.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use ailoop::{Conversation, Message, RetryConfig, RetryingModel, Source, UserBlock, ailoop_tool};
use ailoop_core::{
    Capability, ChatRequest, CompletionModel, FinishReason, ProviderError, ProviderErrorKind,
    RetryClassification, Retryable, StreamChunk, ToolDefinition,
};
use ailoop_openai::{
    AzureAuth, CachingTokenProvider, ModelCapabilities, OpenAIApiErrorKind, OpenAIClient,
    OpenAIError, TokenProvider,
};
use futures::StreamExt;
use serde_json::json;
use support::{FakeServer, Reply, closed_local_url};

const CHAT: &str = "/v1/chat/completions";

fn text_stream(text: &str) -> Reply {
    Reply::sse(&[
        json!({"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}),
        json!({"choices":[{"index":0,"delta":{"content":text}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4}}),
    ])
}

async fn collect(
    model: &impl CompletionModel<Error = OpenAIError>,
    req: ChatRequest,
) -> Result<(Vec<StreamChunk>, Option<OpenAIError>), OpenAIError> {
    let mut stream = model.chat_stream(req).await?;
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(c) => chunks.push(c),
            Err(e) => return Ok((chunks, Some(e))),
        }
    }
    Ok((chunks, None))
}

fn hello_request() -> ChatRequest {
    ChatRequest::new(vec![Message::user("hola")], 256)
}

async fn setup_error(server: &FakeServer, req: ChatRequest) -> OpenAIError {
    let model = OpenAIClient::new(server.base_url()).model("m");
    match collect(&model, req).await {
        Err(e) => e,
        Ok(other) => panic!("expected the request to fail, got {other:?}"),
    }
}

// ---------------------------------------------------------------- streaming

#[tokio::test]
async fn streams_text_without_a_key() {
    let server = FakeServer::start().await;
    server.push(CHAT, text_stream("¡Hola!"));
    let model = OpenAIClient::new(server.base_url()).model("llama3.2");

    let (chunks, err) = collect(&model, hello_request()).await.unwrap();
    assert!(err.is_none(), "{err:?}");
    assert!(matches!(&chunks[0], StreamChunk::TextDelta { delta } if delta == "¡Hola!"));
    match chunks.last().unwrap() {
        StreamChunk::TurnFinished { reason, usage, .. } => {
            assert!(matches!(reason, FinishReason::EndTurn));
            assert_eq!((usage.input_tokens, usage.output_tokens), (11, 4));
        }
        other => panic!("{other:?}"),
    }

    let request = &server.requests_to(CHAT)[0];
    assert_eq!(request.method, "POST");
    assert!(!request.headers.contains_key("authorization"));
    let body = request.json();
    assert_eq!(body["model"], json!("llama3.2"));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["max_tokens"], json!(256));
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
}

#[tokio::test]
async fn sends_key_and_extra_headers() {
    let server = FakeServer::start().await;
    server.push(CHAT, text_stream("ok"));
    let model = OpenAIClient::new(server.base_url())
        .api_key("sk-or-test")
        .header("HTTP-Referer", "https://konor.me")
        .header("X-Title", "apunte")
        .model("openai/gpt-4o-mini");
    collect(&model, hello_request()).await.unwrap();

    let request = &server.requests_to(CHAT)[0];
    assert_eq!(request.headers["authorization"], "Bearer sk-or-test");
    assert_eq!(request.headers["http-referer"], "https://konor.me");
    assert_eq!(request.headers["x-title"], "apunte");
}

#[tokio::test]
async fn conversation_runs_end_to_end() {
    let server = FakeServer::start().await;
    server.push(CHAT, text_stream("Una nota corta."));
    let model = OpenAIClient::new(server.base_url()).model("m");
    let mut chat = Conversation::builder(model)
        .system_prompt("Responde en español.")
        .build()
        .unwrap();
    let outcome = chat.run("Resume esto").await.unwrap();
    assert_eq!(outcome.final_text.as_deref(), Some("Una nota corta."));
    let body = server.requests_to(CHAT)[0].json();
    assert_eq!(body["messages"][0]["role"], json!("system"));
    assert_eq!(body["messages"][1]["content"], json!("Resume esto"));
}

// -------------------------------------------------------------------- tools

#[ailoop_tool(description = "Search the user's notes")]
async fn search_notes(query: String) -> String {
    format!("2 notes match {query}")
}

#[tokio::test]
async fn tool_call_round_trip_through_conversation() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::sse(&[
            json!({"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search_notes","arguments":""}}]}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\":"}}]}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rust\"}"}}]}}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
        ]),
    );
    server.push(CHAT, text_stream("Encontré 2 notas."));

    let model = OpenAIClient::new(server.base_url()).model("qwen2.5");
    let mut chat = Conversation::builder(model)
        .tool(SearchNotes)
        .build()
        .unwrap();
    let outcome = chat.run("¿Qué notas tengo de rust?").await.unwrap();
    assert_eq!(outcome.final_text.as_deref(), Some("Encontré 2 notas."));

    let requests = server.requests_to(CHAT);
    assert_eq!(requests.len(), 2);
    let first = requests[0].json();
    assert_eq!(first["tools"][0]["function"]["name"], json!("search_notes"));
    let second = requests[1].json();
    let messages = second["messages"].as_array().unwrap();
    let assistant = messages
        .iter()
        .find(|m| m["role"] == json!("assistant"))
        .unwrap();
    assert_eq!(assistant["tool_calls"][0]["id"], json!("call_1"));
    let tool = messages
        .iter()
        .find(|m| m["role"] == json!("tool"))
        .unwrap();
    assert_eq!(tool["tool_call_id"], json!("call_1"));
    assert_eq!(tool["content"], json!("2 notes match rust"));
}

// ------------------------------------------------------------------- images

#[tokio::test]
async fn sends_jpeg_screenshot_as_data_uri() {
    let server = FakeServer::start().await;
    server.push(CHAT, text_stream("Veo una ventana."));
    let model = OpenAIClient::new(server.base_url()).model("llava");
    let req = ChatRequest::new(
        vec![Message::user_with_blocks([
            UserBlock::text("¿Qué hay en la pantalla?"),
            UserBlock::image(Source::Base64 {
                media_type: "image/jpeg".into(),
                data: "/9j/4AAQSkZJRg==".into(),
            }),
        ])],
        256,
    );
    let (_, err) = collect(&model, req).await.unwrap();
    assert!(err.is_none());
    let body = server.requests_to(CHAT)[0].json();
    let content = &body["messages"][0]["content"];
    assert_eq!(content[0]["type"], json!("text"));
    assert_eq!(content[1]["type"], json!("image_url"));
    assert_eq!(
        content[1]["image_url"]["url"],
        json!("data:image/jpeg;base64,/9j/4AAQSkZJRg==")
    );
}

// ------------------------------------------------------------------- errors

#[tokio::test]
async fn local_service_not_running() {
    let url = closed_local_url().await;
    let model = OpenAIClient::new(&url).model("llama3.2");
    let started = std::time::Instant::now();
    let err = collect(&model, hello_request()).await.unwrap_err();
    let one_attempt = started.elapsed();
    assert_eq!(err.kind(), ProviderErrorKind::Network);
    assert!(model.client().is_local());
    assert!(
        matches!(&err, OpenAIError::Unreachable { url: u, local: true, .. } if *u == format!("{url}/chat/completions")),
        "{err:?}"
    );
    // Not retried: the service is not running. A refused connection
    // costs about the same each time (around 2 s on Windows), so a
    // retry would at least double the time.
    let retrying = RetryingModel::new(OpenAIClient::new(&url).model("m"));
    let started = std::time::Instant::now();
    assert!(retrying.chat_stream(hello_request()).await.is_err());
    let elapsed = started.elapsed();
    assert!(elapsed < one_attempt * 2 + Duration::from_millis(200));
}

#[tokio::test]
async fn remote_host_unreachable_is_offline() {
    // `.invalid` never resolves (RFC 2606).
    let model = OpenAIClient::new("https://api.example.invalid/v1").model("m");
    let err = collect(&model, hello_request()).await.unwrap_err();
    assert_eq!(err.kind(), ProviderErrorKind::Network);
    assert!(
        matches!(&err, OpenAIError::Unreachable { url, local: false, .. } if url == "https://api.example.invalid/v1/chat/completions"),
        "{err:?}"
    );
}

#[tokio::test]
async fn rejected_key_401_and_403() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::json(
            401,
            json!({"error":{"message":"Incorrect API key provided","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}),
        ),
    );
    let err = setup_error(&server, hello_request()).await;
    assert_eq!(err.kind(), ProviderErrorKind::Auth);
    assert!(matches!(&err, OpenAIError::Api { status, .. } if status.as_u16() == 401));

    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::json(403, json!({"error":{"code":403,"message":"Key disabled"}})),
    );
    let err = setup_error(&server, hello_request()).await;
    assert_eq!(err.kind(), ProviderErrorKind::Permission);
    assert!(matches!(&err, OpenAIError::Api { status, .. } if status.as_u16() == 403));
}

#[tokio::test]
async fn rate_limit_with_retry_after() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::status_with_headers(
            429,
            &[("retry-after", "20"), ("content-type", "application/json")],
            r#"{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}"#,
        ),
    );
    let err = setup_error(&server, hello_request()).await;
    assert_eq!(err.kind(), ProviderErrorKind::RateLimited);
    assert_eq!(
        err.retry_classification(),
        RetryClassification::Transient {
            retry_after: Some(Duration::from_secs(20))
        }
    );
}

#[tokio::test]
async fn rate_limit_with_openai_reset_headers() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::status_with_headers(
            429,
            &[
                ("x-ratelimit-remaining-tokens", "0"),
                ("x-ratelimit-reset-tokens", "1m30s"),
                ("x-ratelimit-reset-requests", "120ms"),
            ],
            r#"{"error":{"message":"Rate limit reached for tokens","type":"tokens","code":"rate_limit_exceeded"}}"#,
        ),
    );
    let err = setup_error(&server, hello_request()).await;
    assert_eq!(
        err.retry_classification(),
        RetryClassification::Transient {
            retry_after: Some(Duration::from_secs(90))
        }
    );
}

#[tokio::test]
async fn retrying_model_waits_out_a_rate_limit() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::status_with_headers(429, &[("retry-after-ms", "10")], "{}"),
    );
    server.push(CHAT, text_stream("ok"));
    let mut config = RetryConfig::default();
    config.jitter = false;
    let model = RetryingModel::with_config(OpenAIClient::new(server.base_url()).model("m"), config);
    let (chunks, err) = collect(&model, hello_request()).await.unwrap();
    assert!(err.is_none());
    assert!(matches!(&chunks[0], StreamChunk::TextDelta { .. }));
    assert_eq!(server.requests_to(CHAT).len(), 2);
}

#[tokio::test]
async fn quota_exhausted_is_not_retried() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::json(
            429,
            json!({"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}),
        ),
    );
    let model = RetryingModel::new(OpenAIClient::new(server.base_url()).model("m"));
    let err = collect(&model, hello_request()).await.unwrap_err();
    assert_eq!(err.kind(), ProviderErrorKind::QuotaExhausted);
    assert_eq!(server.requests_to(CHAT).len(), 1);
}

#[tokio::test]
async fn model_without_tool_support() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::json(
            400,
            json!({"error":{"message":"registry.ollama.ai/library/gemma2:2b does not support tools","type":"api_error","param":null,"code":null}}),
        ),
    );
    let mut req = hello_request();
    req.tools = Some(vec![ToolDefinition::new(
        "search_notes",
        "Search",
        json!({"type":"object","properties":{}}),
        vec![],
    )]);
    let err = setup_error(&server, req).await;
    assert_eq!(
        err.kind(),
        ProviderErrorKind::Unsupported(Capability::Tools)
    );
}

#[tokio::test]
async fn model_without_image_support() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::json(
            404,
            json!({"error":{"message":"No endpoints found that support image input","code":404}}),
        ),
    );
    let req = ChatRequest::new(
        vec![Message::user_with_blocks([UserBlock::image(
            Source::Base64 {
                media_type: "image/jpeg".into(),
                data: "/9j/".into(),
            },
        )])],
        64,
    );
    let err = setup_error(&server, req).await;
    assert_eq!(
        err.kind(),
        ProviderErrorKind::Unsupported(Capability::Images)
    );
}

#[tokio::test]
async fn model_not_found() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::json(
            404,
            json!({"error":{"message":"model \"llama9\" not found, try pulling it first","type":"api_error","param":null,"code":null}}),
        ),
    );
    let err = setup_error(&server, hello_request()).await;
    assert_eq!(err.kind(), ProviderErrorKind::ModelNotFound);
}

#[tokio::test]
async fn connection_cut_mid_response() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::sse_unfinished(
            &[json!({"choices":[{"index":0,"delta":{"content":"La respuesta emp"}}]})],
            true,
        ),
    );
    let model = OpenAIClient::new(server.base_url()).model("m");
    let (chunks, err) = collect(&model, hello_request()).await.unwrap();
    assert!(matches!(&chunks[0], StreamChunk::TextDelta { .. }));
    let err = err.expect("the cut must surface as an error");
    assert_eq!(err.kind(), ProviderErrorKind::Interrupted);
    assert!(
        matches!(
            &err,
            OpenAIError::Interrupted {
                source: Some(_),
                ..
            }
        ),
        "{err:?}"
    );
}

#[tokio::test]
async fn stream_closed_without_finishing() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::sse_unfinished(
            &[json!({"choices":[{"index":0,"delta":{"content":"Hasta aq"}}]})],
            false,
        ),
    );
    let model = OpenAIClient::new(server.base_url()).model("m");
    let (_, err) = collect(&model, hello_request()).await.unwrap();
    let err = err.expect("an unfinished stream must surface as an error");
    assert!(
        matches!(&err, OpenAIError::Interrupted { source: None, .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn error_event_mid_stream() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::sse(&[
            json!({"choices":[{"index":0,"delta":{"content":"Hol"}}]}),
            json!({"error":{"message":"model runner has unexpectedly stopped","type":"api_error"}}),
        ]),
    );
    let model = OpenAIClient::new(server.base_url()).model("m");
    let (_, err) = collect(&model, hello_request()).await.unwrap();
    match err {
        Some(OpenAIError::Provider { message, .. }) => {
            assert_eq!(message, "model runner has unexpectedly stopped")
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn context_overflow_is_flagged_for_compaction() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::json(
            400,
            json!({"error":{"message":"This model's maximum context length is 8192 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}),
        ),
    );
    let err = setup_error(&server, hello_request()).await;
    assert!(err.is_context_overflow());
    assert_eq!(err.kind(), ProviderErrorKind::ContextOverflow);
}

#[tokio::test]
async fn unparseable_error_body_keeps_the_status_category() {
    let server = FakeServer::start().await;
    server.push(
        CHAT,
        Reply::status_with_headers(500, &[], "<html>Bad Gateway</html>"),
    );
    let err = setup_error(&server, hello_request()).await;
    match err {
        OpenAIError::Api { kind, message, .. } => {
            assert_eq!(kind, OpenAIApiErrorKind::ServerError);
            assert_eq!(message, "<html>Bad Gateway</html>");
        }
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------------------- models

#[tokio::test]
async fn lists_models() {
    let server = FakeServer::start().await;
    server.push(
        "/v1/models",
        Reply::json(
            200,
            json!({"object":"list","data":[
                {"id":"llama3.2:latest","object":"model","created":1730000000,"owned_by":"library"},
                {"id":"llava:7b","object":"model","created":1730000001,"owned_by":"library"}
            ]}),
        ),
    );
    let client = OpenAIClient::new(server.base_url()).api_key("k");
    let models = client.list_models().await.unwrap();
    let ids: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["llama3.2:latest", "llava:7b"]);
    assert_eq!(models[0].capabilities, ModelCapabilities::default());
    let request = &server.requests_to("/v1/models")[0];
    assert_eq!(request.method, "GET");
    assert_eq!(request.headers["authorization"], "Bearer k");
}

#[tokio::test]
async fn list_models_errors_are_classified() {
    let server = FakeServer::start().await;
    server.push(
        "/v1/models",
        Reply::json(401, json!({"error":{"code":401,"message":"No auth"}})),
    );
    let err = OpenAIClient::new(server.base_url())
        .list_models()
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ProviderErrorKind::Auth);

    let url = closed_local_url().await;
    let err = OpenAIClient::new(&url).list_models().await.unwrap_err();
    assert_eq!(err.kind(), ProviderErrorKind::Network);
    assert!(matches!(err, OpenAIError::Unreachable { local: true, .. }));
}

#[tokio::test]
async fn capabilities_from_openrouter_listing() {
    let server = FakeServer::start().await;
    server.push(
        "/v1/models",
        Reply::json(
            200,
            json!({"data":[{
                "id":"google/gemma-3-27b-it","name":"Google: Gemma 3 27B","context_length":131072,
                "architecture":{"input_modalities":["text","image"]},
                "supported_parameters":["max_tokens","temperature"]
            }]}),
        ),
    );
    let client = OpenAIClient::new(server.base_url());
    let caps = client
        .model_capabilities("google/gemma-3-27b-it")
        .await
        .unwrap();
    assert_eq!(caps, ModelCapabilities::new(Some(true), Some(false)));
    // Complete from the listing: no native lookups.
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn capabilities_from_ollama_show() {
    let server = FakeServer::start().await;
    server.push(
        "/v1/models",
        Reply::json(200, json!({"data":[{"id":"llava:7b","object":"model"}]})),
    );
    server.push(
        "/api/show",
        Reply::json(200, json!({"capabilities":["completion","vision"]})),
    );
    let client = OpenAIClient::new(server.base_url());
    let caps = client.model_capabilities("llava:7b").await.unwrap();
    assert_eq!(caps, ModelCapabilities::new(Some(true), Some(false)));
    let show = &server.requests_to("/api/show")[0];
    assert_eq!(show.json(), json!({"model":"llava:7b"}));
}

#[tokio::test]
async fn capabilities_from_lm_studio() {
    let server = FakeServer::start().await;
    server.push(
        "/v1/models",
        Reply::json(200, json!({"data":[{"id":"qwen2.5-vl-7b-instruct"}]})),
    );
    server.push(
        "/api/v1/models",
        Reply::json(
            200,
            json!({"models":[{
                "type":"llm","key":"qwen2.5-vl-7b-instruct",
                "capabilities":{"vision":true,"trained_for_tool_use":true}
            }]}),
        ),
    );
    let client = OpenAIClient::new(server.base_url());
    let caps = client
        .model_capabilities("qwen2.5-vl-7b-instruct")
        .await
        .unwrap();
    assert_eq!(caps, ModelCapabilities::new(Some(true), Some(true)));
}

#[tokio::test]
async fn capabilities_stay_unknown_when_nobody_says() {
    let server = FakeServer::start().await;
    server.push(
        "/v1/models",
        Reply::json(200, json!({"data":[{"id":"some-model"}]})),
    );
    let client = OpenAIClient::new(server.base_url());
    let caps = client.model_capabilities("some-model").await.unwrap();
    assert_eq!(caps, ModelCapabilities::default());
}

// -------------------------------------------------------------------- azure

const AZURE_CHAT: &str = "/openai/v1/chat/completions";

fn azure_root(server: &FakeServer) -> String {
    format!("http://{}", server.addr)
}

#[tokio::test]
async fn azure_api_key_goes_in_the_api_key_header() {
    let server = FakeServer::start().await;
    server.push(AZURE_CHAT, text_stream("hi"));
    let model = OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("az-key".into()))
        .model("my-deployment");
    assert_eq!(model.name(), "azure-openai");
    let (_, err) = collect(&model, hello_request()).await.unwrap();
    assert!(err.is_none(), "{err:?}");
    let request = &server.requests_to(AZURE_CHAT)[0];
    assert_eq!(request.headers["api-key"], "az-key");
    assert!(!request.headers.contains_key("authorization"));
    assert!(!request.path.contains("api-version"));
    let body = request.json();
    assert_eq!(body["model"], "my-deployment");
    assert_eq!(body["max_completion_tokens"], 256);
    assert!(body.get("max_tokens").is_none());
}

#[tokio::test]
async fn azure_fixed_token_is_a_bearer() {
    let server = FakeServer::start().await;
    server.push(AZURE_CHAT, text_stream("hi"));
    let model = OpenAIClient::azure(azure_root(&server), AzureAuth::Token("tok".into())).model("d");
    collect(&model, hello_request()).await.unwrap();
    let request = &server.requests_to(AZURE_CHAT)[0];
    assert_eq!(request.headers["authorization"], "Bearer tok");
    assert!(!request.headers.contains_key("api-key"));
}

struct CountingProvider(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl TokenProvider for CountingProvider {
    async fn token(&self) -> Result<String, OpenAIError> {
        let n = self.0.fetch_add(1, Ordering::SeqCst);
        Ok(format!("entra-{n}"))
    }
}

#[tokio::test]
async fn azure_token_provider_is_asked_per_request_and_can_be_cached() {
    let server = FakeServer::start().await;
    server.push(AZURE_CHAT, text_stream("hi"));
    let calls = Arc::new(AtomicUsize::new(0));
    let model = OpenAIClient::azure(
        azure_root(&server),
        AzureAuth::provider(CountingProvider(Arc::clone(&calls))),
    )
    .model("d");
    collect(&model, hello_request()).await.unwrap();
    collect(&model, hello_request()).await.unwrap();
    let auth: Vec<_> = server
        .requests_to(AZURE_CHAT)
        .iter()
        .map(|r| r.headers["authorization"].clone())
        .collect();
    assert_eq!(auth, ["Bearer entra-0", "Bearer entra-1"]);

    let cached_calls = Arc::new(AtomicUsize::new(0));
    let cached = CachingTokenProvider::new(
        CountingProvider(Arc::clone(&cached_calls)),
        Duration::from_secs(3600),
        Duration::from_secs(300),
    );
    let model = OpenAIClient::azure(azure_root(&server), AzureAuth::provider(cached)).model("d");
    collect(&model, hello_request()).await.unwrap();
    collect(&model, hello_request()).await.unwrap();
    assert_eq!(cached_calls.load(Ordering::SeqCst), 1);
}

struct FailingProvider;

#[async_trait::async_trait]
impl TokenProvider for FailingProvider {
    async fn token(&self) -> Result<String, OpenAIError> {
        Err(OpenAIError::Config("token fetch failed".into()))
    }
}

#[tokio::test]
async fn azure_token_provider_error_fails_before_sending() {
    let server = FakeServer::start().await;
    let model =
        OpenAIClient::azure(azure_root(&server), AzureAuth::provider(FailingProvider)).model("d");
    let err = collect(&model, hello_request()).await.unwrap_err();
    assert!(
        matches!(&err, OpenAIError::Config(m) if m.contains("token fetch failed")),
        "{err:?}"
    );
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn azure_rate_limit_honours_retry_after_ms() {
    let server = FakeServer::start().await;
    server.push(
        AZURE_CHAT,
        Reply::status_with_headers(
            429,
            &[("retry-after-ms", "10"), ("retry-after", "60")],
            r#"{"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#,
        ),
    );
    server.push(AZURE_CHAT, text_stream("ok"));
    let client = OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("k".into()));
    let err = collect(&client.clone().model("d"), hello_request())
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ProviderErrorKind::RateLimited);
    assert_eq!(
        err.retry_classification(),
        RetryClassification::Transient {
            retry_after: Some(Duration::from_millis(10))
        }
    );
    let mut config = RetryConfig::default();
    config.jitter = false;
    let model = RetryingModel::with_config(client.model("d"), config);
    let (_, err) = collect(&model, hello_request()).await.unwrap();
    assert!(err.is_none());
}

#[tokio::test]
async fn azure_deployment_not_found_and_content_filter() {
    let server = FakeServer::start().await;
    server.push(
        AZURE_CHAT,
        Reply::json(
            404,
            json!({"error":{"code":"DeploymentNotFound","message":"The API deployment for this resource does not exist."}}),
        ),
    );
    let model =
        OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("k".into())).model("nope");
    let err = collect(&model, hello_request()).await.unwrap_err();
    assert_eq!(err.kind(), ProviderErrorKind::ModelNotFound);
    assert_eq!(err.retry_classification(), RetryClassification::Permanent);

    let server = FakeServer::start().await;
    server.push(
        AZURE_CHAT,
        Reply::json(
            400,
            json!({"error":{"code":"content_filter","message":"The response was filtered due to the prompt triggering Azure OpenAI's content management policy.","param":"prompt","status":400}}),
        ),
    );
    let model = OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("k".into())).model("d");
    let err = collect(&model, hello_request()).await.unwrap_err();
    assert_eq!(err.kind(), ProviderErrorKind::ContentFilter);
    assert_eq!(err.retry_classification(), RetryClassification::Permanent);
}

#[tokio::test]
async fn azure_context_overflow_is_flagged() {
    let server = FakeServer::start().await;
    server.push(
        AZURE_CHAT,
        Reply::json(
            400,
            json!({"error":{"message":"Your input exceeds the context window of this model. Please adjust your input and try again.","type":"invalid_request_error","param":"input","code":"context_length_exceeded"}}),
        ),
    );
    let model = OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("k".into())).model("d");
    let err = collect(&model, hello_request()).await.unwrap_err();
    assert!(err.is_context_overflow());
}

#[tokio::test]
async fn azure_recorded_text_stream() {
    let server = FakeServer::start().await;
    server.push(
        AZURE_CHAT,
        Reply::sse_recorded(include_str!("fixtures/chat_text.sse")),
    );
    let model = OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("k".into())).model("d");
    let (chunks, err) = collect(&model, hello_request()).await.unwrap();
    assert!(err.is_none(), "{err:?}");
    let text: String = chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::TextDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello, what's up?");
    match chunks.last() {
        Some(StreamChunk::TurnFinished { reason, usage, .. }) => {
            assert!(matches!(reason, FinishReason::EndTurn));
            assert_eq!((usage.input_tokens, usage.output_tokens), (13, 6));
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn azure_recorded_tool_call_stream() {
    let server = FakeServer::start().await;
    server.push(
        AZURE_CHAT,
        Reply::sse_recorded(include_str!("fixtures/chat_tool_call.sse")),
    );
    let model = OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("k".into())).model("d");
    let (chunks, err) = collect(&model, hello_request()).await.unwrap();
    assert!(err.is_none(), "{err:?}");
    let mut started = Vec::new();
    let mut finished = Vec::new();
    let mut reason = None;
    for chunk in &chunks {
        match chunk {
            StreamChunk::ToolCallStarted { call_id, name } => {
                started.push((call_id.clone(), name.clone()))
            }
            StreamChunk::ToolCallFinished { args, .. } => finished.push(args.clone()),
            StreamChunk::TurnFinished { reason: r, .. } => reason = Some(r.clone()),
            _ => {}
        }
    }
    assert_eq!(
        started,
        [("call_FIXTURE".to_string(), "get_weather".to_string())]
    );
    assert_eq!(finished.len(), 1);
    assert!(finished[0].is_object(), "{:?}", finished[0]);
    assert!(matches!(reason, Some(FinishReason::ToolUse)));
}

#[tokio::test]
async fn azure_capabilities_are_unknown_without_requests() {
    let server = FakeServer::start().await;
    let client = OpenAIClient::azure(azure_root(&server), AzureAuth::ApiKey("k".into()));
    let caps = client.model_capabilities("my-deployment").await.unwrap();
    assert_eq!(caps, ModelCapabilities::default());
    assert!(server.requests().is_empty());
}
