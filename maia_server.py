"""Shared Maia move server — one per anchor Elo.

Loads ONE Maia engine (via the maia3-uci launcher, on GPU by default) and serves move
requests over a Unix-domain socket. Many concurrent game workers connect to it, so the GPU
holds only ~N_anchors Maia instances (~1.2 GB each) no matter how many games run at once.

A single UCI engine is single-threaded, so engine access is serialised with a lock; Maia is
fast (~0.2 s/move), so even dozens of concurrent games per anchor barely keep it busy.

Protocol (one JSON line each way, per connection):
    request : {"moves": ["e2e4", "e7e5", ...]}   # UCI moves from the start position
    response: {"move": "g1f3"}  or  {"error": "..."}

Run:
    python maia_server.py --elo 1000 --socket /tmp/maia-1000.sock \
        --maia-path /workspace/llm-venv/bin/maia3-uci

It prints `READY` to stdout once the socket is serving.
"""

import argparse
import json
import os
import socket
import socketserver
import sys
import threading

import chess
import chess.engine

_ENGINE = None
_LOCK = threading.Lock()
_TIME = 0.2


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline()
            if not line:
                return  # bare connect (e.g. readiness probe) — nothing to do
            req = json.loads(line.decode())
            board = chess.Board()
            for mv in req.get("moves", []):
                board.push_uci(mv)
            with _LOCK:
                result = _ENGINE.play(board, chess.engine.Limit(time=_TIME))
            resp = {"move": result.move.uci()}
        except Exception as e:  # noqa: BLE001 — report any failure back to the worker
            resp = {"error": f"{type(e).__name__}: {e}"}
        try:
            self.wfile.write((json.dumps(resp) + "\n").encode())
        except Exception:
            pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def request_move(socket_path: str, moves, timeout: float = 120) -> str:
    """Client: ask the Maia server at `socket_path` for the move after `moves` (UCI list)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(socket_path)
        s.sendall((json.dumps({"moves": list(moves)}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf:
        raise RuntimeError("empty response from Maia server")
    resp = json.loads(buf.decode())
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["move"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", type=int, required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--maia-path", default="maia3-uci")
    ap.add_argument("--maia-model", default="maia3-79m")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--time", type=float, default=0.2)
    ap.add_argument("--device", default=None, help="e.g. 'cpu' to run Maia off-GPU")
    args = ap.parse_args()

    global _ENGINE, _TIME
    _TIME = args.time
    cmd = [args.maia_path, "--model", args.maia_model, "--elo", str(args.elo),
           "--use-uci-history", "--temperature", str(args.temperature),
           "--top-p", str(args.top_p)]
    if args.device:
        cmd += ["--device", args.device]
        if args.device == "cpu":
            cmd.append("--no-use-amp")
    _ENGINE = chess.engine.SimpleEngine.popen_uci(cmd)

    try:
        os.unlink(args.socket)
    except FileNotFoundError:
        pass

    server = _Server(args.socket, _Handler)

    def _cleanup(*_a):
        try:
            _ENGINE.quit()
        except Exception:
            pass
        try:
            os.unlink(args.socket)
        except Exception:
            pass
        os._exit(0)

    import signal
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    print("READY", flush=True)
    try:
        server.serve_forever()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
