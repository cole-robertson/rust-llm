"""Serves one recorded RubyLLM Anthropic response for every POST /v1/messages, and logs each
request body to stdout, so a generated chat UI can be driven end to end without a real key.

Usage: python3 anthropic_replay.py CASSETTE.json PORT
"""

import http.server
import json
import sys

cassette, port = sys.argv[1], int(sys.argv[2])
body = json.load(open(cassette))[0]["response_body"].encode()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        request = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        print(f"POST {self.path} {request.decode()}", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
