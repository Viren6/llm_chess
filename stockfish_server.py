"""Shared Stockfish move-ordering server — ONE instance for the whole run.

Loads ONE Stockfish 18 engine and serves "rank these legal moves" requests over a Unix-domain
socket. Many concurrent game workers connect to it, but a single global lock serialises every
evaluation, so exactly one engine search runs at any moment across all workers combined (as the
covert-ordering trick requires). This is a deliberate throughput bottleneck — see the cost note
in run_maia_concurrent.py.

For a position, EACH legal move is searched independently for `nodes` nodes (default 1,000,000)
with a freshly-cleared hash (a `ucinewgame` is sent before every move, so no transposition-table
state leaks between moves). The moves are returned ordered best→worst from the side-to-move's
point of view. The full legal-move set is always returned (a move whose search errors is appended
last) so the caller never loses a legal option.

Protocol (one JSON line each way, per connection):
    request : {"fen": "<FEN>", "nodes": 1000000}      # nodes optional, defaults to server's
    response: {"ordered": ["g1f3", ...], "scores": {"g1f3": 34, ...}}   # cp, mover's POV
              or {"error": "..."}

Run:
    python stockfish_server.py --socket /tmp/sf-order.sock \
        --sf-path /workspace/engines/stockfish18 --hash 128 --threads 1 --nodes 1000000

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
_NODES = 1_000_000
_MATE = 1_000_000  # finite stand-in for mate scores when sorting


def _rank_moves(fen, nodes):
    """Return (ordered_uci, scores) for the position; one `nodes`-node search per legal move,
    each with a fresh hash. Always covers every legal move."""
    board = chess.Board(fen)
    scored, failed = [], []
    for mv in board.legal_moves:
        u = mv.uci()
        try:
            # game=object() -> python-chess sends `ucinewgame`, clearing the hash for this move.
            info = _ENGINE.analyse(
                board, chess.engine.Limit(nodes=nodes),
                root_moves=[mv], game=object(),
            )
            cp = info["score"].relative.score(mate_score=_MATE)  # mover's POV, higher = better
            scored.append((u, int(cp)))
        except Exception:  # noqa: BLE001 — keep the move in the list, just rank it last
            failed.append(u)
    scored.sort(key=lambda kv: kv[1], reverse=True)
    ordered = [u for u, _ in scored] + failed
    return ordered, {u: c for u, c in scored}


def _eval_root(fen, nodes):
    """One search of the position; return (best_move_uci, cp) from the side-to-move's POV."""
    board = chess.Board(fen)
    info = _ENGINE.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
    pv = info.get("pv") or []
    best = pv[0].uci() if pv else None
    return best, int(info["score"].relative.score(mate_score=_MATE))


def _eval_move(fen, move_uci, nodes):
    """One search restricted to `move_uci`; return cp from the side-to-move's POV."""
    board = chess.Board(fen)
    info = _ENGINE.analyse(board, chess.engine.Limit(nodes=nodes),
                           root_moves=[chess.Move.from_uci(move_uci)], game=object())
    return int(info["score"].relative.score(mate_score=_MATE))


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline()
            if not line:
                return  # bare connect (readiness probe)
            req = json.loads(line.decode())
            nodes = int(req.get("nodes") or _NODES)
            op = req.get("op")
            if op == "root":            # correction: position eval + best move (1 search)
                with _LOCK:
                    best, cp = _eval_root(req["fen"], nodes)
                resp = {"root_move": best, "root_cp": cp}
            elif op == "move":          # correction: eval of one specific move (1 search)
                with _LOCK:
                    cp = _eval_move(req["fen"], req["move"], nodes)
                resp = {"cp": cp}
            else:                       # nudge: rank ALL legal moves (1 search each)
                with _LOCK:  # exactly one engine search at a time, across all workers
                    ordered, scores = _rank_moves(req["fen"], nodes)
                resp = {"ordered": ordered, "scores": scores}
        except Exception as e:  # noqa: BLE001
            resp = {"error": f"{type(e).__name__}: {e}"}
        try:
            self.wfile.write((json.dumps(resp) + "\n").encode())
        except Exception:
            pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def request_ranked(socket_path: str, fen: str, nodes=None, timeout: float = 1800):
    """Client: rank the legal moves of `fen`. Returns (ordered_uci, scores) where scores maps
    each scored UCI move -> centipawns from the side-to-move's POV (mate = ±1_000_000)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(socket_path)
        payload = {"fen": fen}
        if nodes is not None:
            payload["nodes"] = nodes
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf:
        raise RuntimeError("empty response from Stockfish server")
    resp = json.loads(buf.decode())
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["ordered"], resp.get("scores", {})


def request_order(socket_path: str, fen: str, nodes=None, timeout: float = 1800) -> list:
    """Client: ask the server to rank the legal moves of `fen`. Returns best→worst UCI list."""
    return request_ranked(socket_path, fen, nodes=nodes, timeout=timeout)[0]


def _request(socket_path: str, payload: dict, timeout: float = 1800) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(socket_path)
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf:
        raise RuntimeError("empty response from Stockfish server")
    resp = json.loads(buf.decode())
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp


def request_root(socket_path: str, fen: str, nodes=None, timeout: float = 1800):
    """Client (correction): position eval + best move. Returns (best_move_uci, cp) STM-POV."""
    p = {"op": "root", "fen": fen}
    if nodes is not None:
        p["nodes"] = nodes
    r = _request(socket_path, p, timeout)
    return r.get("root_move"), r["root_cp"]


def request_move_eval(socket_path: str, fen: str, move_uci: str, nodes=None, timeout: float = 1800):
    """Client (correction): eval of one specific move. Returns cp from the side-to-move's POV."""
    p = {"op": "move", "fen": fen, "move": move_uci}
    if nodes is not None:
        p["nodes"] = nodes
    return _request(socket_path, p, timeout)["cp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--sf-path", default="/workspace/engines/stockfish18")
    ap.add_argument("--hash", type=int, default=128, help="hash size in MB (cleared per move)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--nodes", type=int, default=1_000_000, help="nodes per legal move")
    args = ap.parse_args()

    global _ENGINE, _NODES
    _NODES = args.nodes
    _ENGINE = chess.engine.SimpleEngine.popen_uci(args.sf_path)
    _ENGINE.configure({"Hash": args.hash, "Threads": args.threads})

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
