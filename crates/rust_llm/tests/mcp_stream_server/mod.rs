//! Port of RubyLLM's `spec/support/mcp_stream_server.rb`: a loopback Streamable HTTP server for
//! specs. Each connection is answered by the closure given to [`Server::new`], which receives the
//! [`ServerRequest`] and a [`Reply`]. Streams opened with [`Reply::stream`] stay open, so a spec
//! can send events down them, drop them, or end them as a server would.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

/// A request the server received: its HTTP verb, lowercased headers, and JSON body.
#[derive(Clone, Debug)]
pub struct ServerRequest {
    pub verb: String,
    pub headers: HashMap<String, String>,
    pub body: Option<Value>,
}

impl ServerRequest {
    pub fn rpc_method(&self) -> Option<&str> {
        self.body.as_ref()?.get("method")?.as_str()
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }
}

type Respond = dyn Fn(&ServerRequest, Reply) + Send + Sync;

#[derive(Default)]
struct State {
    requests: Vec<ServerRequest>,
    /// Open streams by connection number: the socket and the request it answers.
    streams: HashMap<usize, (TcpStream, ServerRequest)>,
    closed: bool,
}

pub struct Server {
    pub url: String,
    state: Arc<Mutex<State>>,
    listener: TcpListener,
}

impl Server {
    pub fn new(respond: impl Fn(&ServerRequest, Reply) + Send + Sync + 'static) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!(
            "http://127.0.0.1:{}/mcp",
            listener.local_addr().unwrap().port()
        );
        let state = Arc::new(Mutex::new(State::default()));
        let respond: Arc<Respond> = Arc::new(respond);
        let accepting = listener.try_clone().unwrap();
        let shared = state.clone();
        std::thread::spawn(move || {
            for (number, socket) in accepting.incoming().enumerate() {
                if shared.lock().unwrap().closed {
                    break;
                }
                let Ok(socket) = socket else { break };
                let (state, respond) = (shared.clone(), respond.clone());
                std::thread::spawn(move || serve(number, socket, state, respond));
            }
        });
        Server {
            url,
            state,
            listener,
        }
    }

    pub fn requests(&self) -> Vec<ServerRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    pub fn requests_for(&self, rpc_method: &str) -> Vec<ServerRequest> {
        self.requests()
            .into_iter()
            .filter(|r| r.rpc_method() == Some(rpc_method))
            .collect()
    }

    pub fn open_streams(&self) -> usize {
        self.state.lock().unwrap().streams.len()
    }

    pub fn push(&self, message: Value) {
        self.write_event(&format!("data: {message}"));
    }

    pub fn write_event(&self, event: &str) {
        for mut socket in self.sockets() {
            chunk(&mut socket, &format!("{event}\n\n"));
        }
    }

    pub fn drop_streams(&self) {
        for socket in self.sockets() {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }

    pub fn finish(&self) {
        let streams: Vec<(TcpStream, ServerRequest)> = {
            let state = self.state.lock().unwrap();
            state
                .streams
                .values()
                .filter_map(|(s, r)| Some((s.try_clone().ok()?, r.clone())))
                .collect()
        };
        for (mut socket, request) in streams {
            let id = request.body.as_ref().and_then(|b| b.get("id")).cloned();
            let answer =
                json!({ "jsonrpc": "2.0", "id": id, "result": { "resultType": "complete" } });
            chunk(&mut socket, &format!("data: {answer}\n\n"));
            let _ = socket.write_all(b"0\r\n\r\n");
            let _ = socket.shutdown(Shutdown::Both);
        }
    }

    pub fn close(&self) {
        self.state.lock().unwrap().closed = true;
        // Wake the acceptor so it sees `closed`.
        let _ = TcpStream::connect(self.listener.local_addr().unwrap());
        self.drop_streams();
    }

    fn sockets(&self) -> Vec<TcpStream> {
        let state = self.state.lock().unwrap();
        state
            .streams
            .values()
            .filter_map(|(s, _)| s.try_clone().ok())
            .collect()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.close();
    }
}

/// A client may close a stream as soon as it reads the answer, so writes that follow find it gone.
fn chunk(socket: &mut TcpStream, data: &str) {
    let _ = socket.write_all(format!("{:x}\r\n{data}\r\n", data.len()).as_bytes());
}

fn serve(number: usize, socket: TcpStream, state: Arc<Mutex<State>>, respond: Arc<Respond>) {
    let Ok(reader) = socket.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let Some(verb) = line.split_whitespace().next().map(str::to_string) else {
        return;
    };
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_lowercase(), value.trim().to_string());
        }
    }
    let length: usize = headers
        .get("content-length")
        .and_then(|l| l.parse().ok())
        .unwrap_or(0);
    let body = (length > 0)
        .then(|| {
            let mut buffer = vec![0; length];
            reader.read_exact(&mut buffer).ok()?;
            serde_json::from_slice(&buffer).ok()
        })
        .flatten();
    let request = ServerRequest {
        verb,
        headers,
        body,
    };
    state.lock().unwrap().requests.push(request.clone());
    let reply = Reply {
        number,
        socket,
        reader,
        request: request.clone(),
        state,
    };
    respond(&request, reply);
}

/// How the server answers one request.
pub struct Reply {
    number: usize,
    socket: TcpStream,
    reader: BufReader<TcpStream>,
    request: ServerRequest,
    state: Arc<Mutex<State>>,
}

impl Reply {
    /// `json(status:, headers:, **fields)`: a JSON-RPC message answering the request.
    pub fn json(mut self, status: u16, headers: &[(&str, &str)], fields: Value) {
        let id = self
            .request
            .body
            .as_ref()
            .and_then(|b| b.get("id"))
            .cloned();
        let mut body = json!({ "jsonrpc": "2.0", "id": id });
        if let (Some(body), Value::Object(fields)) = (body.as_object_mut(), fields) {
            body.extend(fields);
        }
        let mut headers = headers.to_vec();
        headers.push(("Content-Type", "application/json"));
        self.respond(status, &body.to_string(), &headers);
    }

    pub fn status(mut self, code: u16) {
        self.respond(code, "", &[]);
    }

    /// `stream(*messages)`: an event stream that stays open until the client or the spec closes
    /// it.
    pub fn stream(mut self, messages: &[Value]) {
        let _ = self.socket.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
        );
        {
            let mut state = self.state.lock().unwrap();
            if let Ok(clone) = self.socket.try_clone() {
                state
                    .streams
                    .insert(self.number, (clone, self.request.clone()));
            }
            for message in messages {
                chunk(&mut self.socket, &format!("data: {message}\n\n"));
            }
        }
        let mut sink = Vec::new();
        let _ = self.reader.read_to_end(&mut sink);
        self.state.lock().unwrap().streams.remove(&self.number);
    }

    fn respond(&mut self, status: u16, body: &str, headers: &[(&str, &str)]) {
        let mut head = format!("HTTP/1.1 {status} Status\r\n");
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str(&format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ));
        let _ = self.socket.write_all(head.as_bytes());
        let _ = self.socket.shutdown(Shutdown::Write);
    }
}
