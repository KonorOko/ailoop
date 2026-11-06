//! A minimal HTTP/1.1 server for the adapter's tests. It records every
//! request and answers from a per-path queue of scripted replies, which
//! lets a test stream SSE, send error statuses with headers, or drop the
//! connection in the middle of a body. One request per connection.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Recorded {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("request body is JSON")
    }
}

#[derive(Debug, Clone)]
pub enum Reply {
    /// A complete response with `Content-Length`.
    Full {
        status: u16,
        headers: Vec<(String, String)>,
        body: String,
    },
    /// A chunked SSE response, one chunk per event. With `cut`, the
    /// connection closes before the terminating chunk, which the client
    /// sees as a transport failure mid-body.
    Sse { events: Vec<String>, cut: bool },
}

impl Reply {
    pub fn json(status: u16, body: Value) -> Self {
        Reply::Full {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: body.to_string(),
        }
    }

    pub fn status_with_headers(status: u16, headers: &[(&str, &str)], body: &str) -> Self {
        Reply::Full {
            status,
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            body: body.to_string(),
        }
    }

    /// SSE from JSON payloads, with the `[DONE]` sentinel at the end.
    pub fn sse(payloads: &[Value]) -> Self {
        let mut events: Vec<String> = payloads.iter().map(|p| format!("data: {p}\n\n")).collect();
        events.push("data: [DONE]\n\n".into());
        Reply::Sse { events, cut: false }
    }

    /// SSE replayed from a recorded body (`data: ...` blocks separated
    /// by blank lines), one chunk per event.
    pub fn sse_recorded(text: &str) -> Self {
        let events = text
            .split("\n\n")
            .map(str::trim)
            .filter(|e| !e.is_empty())
            .map(|e| format!("{e}\n\n"))
            .collect();
        Reply::Sse { events, cut: false }
    }

    /// SSE from JSON payloads, without `[DONE]`, with the body closed
    /// cleanly (`cut: false`) or abruptly (`cut: true`).
    pub fn sse_unfinished(payloads: &[Value], cut: bool) -> Self {
        let events = payloads.iter().map(|p| format!("data: {p}\n\n")).collect();
        Reply::Sse { events, cut }
    }
}

#[derive(Clone)]
pub struct FakeServer {
    pub addr: std::net::SocketAddr,
    requests: Arc<Mutex<Vec<Recorded>>>,
    routes: Arc<Mutex<HashMap<String, VecDeque<Reply>>>>,
}

impl FakeServer {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = FakeServer {
            addr,
            requests: Arc::default(),
            routes: Arc::default(),
        };
        let handle = server.clone();
        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    break;
                };
                let handle = handle.clone();
                tokio::spawn(async move { handle.serve(socket).await });
            }
        });
        server
    }

    /// Base URL in the form the client expects (`.../v1`).
    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// Queue `reply` for the next request to `path` (e.g.
    /// `/v1/chat/completions`). The last reply of a path repeats.
    pub fn push(&self, path: &str, reply: Reply) -> &Self {
        self.routes
            .lock()
            .unwrap()
            .entry(path.to_string())
            .or_default()
            .push_back(reply);
        self
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    pub fn requests_to(&self, path: &str) -> Vec<Recorded> {
        self.requests()
            .into_iter()
            .filter(|r| r.path == path)
            .collect()
    }

    fn next_reply(&self, path: &str) -> Reply {
        let mut routes = self.routes.lock().unwrap();
        match routes.get_mut(path) {
            Some(queue) if queue.len() > 1 => queue.pop_front().unwrap(),
            Some(queue) if queue.len() == 1 => queue.front().unwrap().clone(),
            _ => Reply::json(404, serde_json::json!({"error": {"message": "no route"}})),
        }
    }

    async fn serve(&self, mut socket: TcpStream) {
        let Some(request) = read_request(&mut socket).await else {
            return;
        };
        let path = request.path.clone();
        self.requests.lock().unwrap().push(request);
        let reply = self.next_reply(&path);
        let _ = write_reply(&mut socket, reply).await;
    }
}

async fn read_request(socket: &mut TcpStream) -> Option<Recorded> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = socket.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut first = lines.next()?.split(' ');
    let method = first.next()?.to_string();
    let path = first.next()?.to_string();
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[header_end..].to_vec();
    while body.len() < length {
        let n = socket.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Some(Recorded {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).to_string(),
    })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Status",
    }
}

async fn write_reply(socket: &mut TcpStream, reply: Reply) -> std::io::Result<()> {
    match reply {
        Reply::Full {
            status,
            headers,
            body,
        } => {
            let mut head = format!(
                "HTTP/1.1 {status} {}\r\ncontent-length: {}\r\nconnection: close\r\n",
                reason(status),
                body.len()
            );
            for (k, v) in headers {
                head.push_str(&format!("{k}: {v}\r\n"));
            }
            head.push_str("\r\n");
            socket.write_all(head.as_bytes()).await?;
            socket.write_all(body.as_bytes()).await?;
            socket.shutdown().await
        }
        Reply::Sse { events, cut } => {
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                )
                .await?;
            for event in events {
                let chunk = format!("{:x}\r\n{event}\r\n", event.len());
                socket.write_all(chunk.as_bytes()).await?;
                socket.flush().await?;
                // Separate writes so the client sees real streaming.
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            if cut {
                // Drop without the terminating chunk.
                return Ok(());
            }
            socket.write_all(b"0\r\n\r\n").await?;
            socket.shutdown().await
        }
    }
}

/// A local address nothing listens on: bind a port, then free it.
pub async fn closed_local_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}/v1")
}
