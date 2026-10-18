#!/usr/bin/env python3
"""Multi-port fake upstream for verifying where ccs routes a request.

Each port impersonates one provider. Every request is logged to stdout and
answered with a valid response whose text names the port that served it, so
the answer you see in the client tells you which upstream handled it.

    python scripts/probe_servers.py                # ports 7901 + 7902
    python scripts/probe_servers.py 7901 7902 7903 # any ports you like

Point a ccs provider at http://127.0.0.1:<port>:

    api_format "anthropic"                 -> ccs POSTs /v1/messages
    api_format "openai" + chat_completions -> ccs POSTs /v1/chat/completions
    api_format "openai" + responses        -> ccs POSTs /v1/responses

Switch providers by editing `current` in ~/.ccs/config.json; a running
`ccs serve` reloads it automatically. The client's reply then names the port
that served the request, and this script's stdout logs the full path, model
and stream flag for every hit.

Ctrl+C stops every listener.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORTS = [7901, 7902]


def label_for(port: int) -> str:
    return f"p{port}"


class ProbeHandler(BaseHTTPRequestHandler):
    """Answers Anthropic / OpenAI requests with a marker naming this port."""

    protocol_version = "HTTP/1.1"
    label = "?"
    hits = 0

    # The default handler writes to stderr; keep the trace on one clean line.
    def log_message(self, fmt, *args):  # noqa: D401 - silence stdlib logging
        pass

    # ── helpers ──────────────────────────────────────────────────────────
    def _body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        return self.rfile.read(n) if n > 0 else b""

    def _json(self, code: int, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _write(self, payload: str) -> None:
        try:
            self.wfile.write(payload.encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self, event: str, obj) -> None:
        self._write(f"event: {event}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n")

    def _trace(self, body: bytes):
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}
        model = req.get("model") or "-"
        stream = bool(req.get("stream"))
        has_auth = bool(
            self.headers.get("authorization") or self.headers.get("x-api-key")
        )
        cls = type(self)
        cls.hits += 1
        print(
            f"[{time.strftime('%H:%M:%S')}] {self.label:>6} "
            f"#{cls.hits:<3} {self.command} {self.path} "
            f"model={model} stream={'yes' if stream else 'no'} "
            f"auth={'yes' if has_auth else 'NO'}",
            flush=True,
        )
        return req, model, stream

    # ── routes ───────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        self._trace(b"")
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/v1/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": f"probe-{self.label}",
                            "object": "model",
                            "created": 0,
                            "owned_by": self.label,
                        }
                    ],
                },
            )
        else:
            self._json(404, {"error": {"message": f"{self.label}: no GET {self.path}"}})

    def do_POST(self) -> None:
        body = self._body()
        _req, model, stream = self._trace(body)
        path = self.path.split("?")[0].rstrip("/")
        text = f"[ccs-probe] routed to {self.label}"
        if path.endswith("/v1/messages"):
            self._anthropic(model, stream, text)
        elif path.endswith("/v1/chat/completions"):
            self._openai_chat(model, stream, text)
        elif path.endswith("/v1/responses"):
            self._openai_responses(model, stream, text)
        else:
            self._json(404, {"error": {"message": f"{self.label}: no POST {self.path}"}})

    # ── wire formats ─────────────────────────────────────────────────────
    def _anthropic(self, model: str, stream: bool, text: str) -> None:
        model = model or "probe"
        out_tokens = len(text) // 4 + 1
        if not stream:
            self._json(
                200,
                {
                    "id": "msg_probe",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": out_tokens},
                },
            )
            return
        self._sse_headers()
        self._sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_probe",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                },
            },
        )
        self._sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        self._sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
        self._sse(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        )
        self._sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": out_tokens},
            },
        )
        self._sse("message_stop", {"type": "message_stop"})

    def _openai_chat(self, model: str, stream: bool, text: str) -> None:
        model = model or "probe"
        created = int(time.time())
        out_tokens = len(text) // 4 + 1
        if not stream:
            self._json(
                200,
                {
                    "id": "chatcmpl-probe",
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": out_tokens,
                        "total_tokens": 12 + out_tokens,
                    },
                },
            )
            return
        self._sse_headers()

        def chunk(delta, finish=None):
            return {
                "id": "chatcmpl-probe",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        self._write(f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n")
        self._write(f"data: {json.dumps(chunk({'content': text}))}\n\n")
        self._write(f"data: {json.dumps(chunk({}, 'stop'))}\n\n")
        self._write("data: [DONE]\n\n")

    def _openai_responses(self, model: str, stream: bool, text: str) -> None:
        model = model or "probe"
        created = int(time.time())
        out_tokens = len(text) // 4 + 1
        item = {
            "type": "message",
            "id": "msg_probe",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        usage = {
            "input_tokens": 12,
            "output_tokens": out_tokens,
            "total_tokens": 12 + out_tokens,
        }
        if not stream:
            self._json(
                200,
                {
                    "id": "resp_probe",
                    "object": "response",
                    "created_at": created,
                    "status": "completed",
                    "model": model,
                    "output": [item],
                    "usage": usage,
                },
            )
            return
        self._sse_headers()
        self._sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_probe",
                    "object": "response",
                    "created_at": created,
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                },
            },
        )
        self._sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_probe",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
        )
        self._sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": "msg_probe",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )
        self._sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_probe",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
        )
        self._sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": "msg_probe",
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
        )
        self._sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": "msg_probe",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        )
        self._sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": item,
            },
        )
        self._sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_probe",
                    "object": "response",
                    "created_at": created,
                    "status": "completed",
                    "model": model,
                    "output": [item],
                    "usage": usage,
                },
            },
        )


def start(port: int) -> ThreadingHTTPServer:
    handler = type(f"ProbeHandler{port}", (ProbeHandler,), {"label": label_for(port)})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str]) -> int:
    ports = [int(a) for a in argv[1:] if a.isdigit()] or DEFAULT_PORTS
    started = []
    for port in ports:
        try:
            started.append(start(port))
            print(f"probe listening on http://127.0.0.1:{port}  (label {label_for(port)})", flush=True)
        except OSError as e:
            print(f"cannot bind {port}: {e}", flush=True)
    if not started:
        return 1
    print(
        "\nPoint a ccs provider at one of these base URLs; each request is "
        "logged above and answers with its own label.\nCtrl+C to stop.\n",
        flush=True,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
