"""One shared LC0 CPU policy-head engine, serving independent UCI positions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socketserver
import threading

import chess
import chess.engine

WEIGHTS_URL = "https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-tf13tune.pb.gz"
WEIGHTS_SHA256 = "2696d3f49ad56412ad24bdcbf14c81c9c459197f93d7d0d6f95cb61d3b51e435"
DEFAULT_LC0 = str(Path.home() / ".local/share/llm-chess/lc0/source/build/lc0")
DEFAULT_WEIGHTS = str(Path.home() / ".local/share/llm-chess/networks/BT4-tf13tune.pb.gz")


def engine_command(lc0_path, weights):
    return [str(Path(lc0_path).resolve()), "policyhead", f"--weights={Path(weights).resolve()}",
            "--backend=blas", "--backend-opts=batch_size=1", "--nncache=0", "--policy-softmax-temp=1",
            f"--config={os.devnull}", "--preload"]


def engine_metadata(lc0_path, weights):
    binary, network = Path(lc0_path), Path(weights)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"LC0 executable missing: {binary}")
    if not network.is_file():
        raise ValueError(f"BT4 weights missing: {network}")
    digest = hashlib.sha256()
    with network.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("BT4 weights SHA256 does not match the requested BT4-tf13tune network")
    binary_digest = hashlib.sha256()
    with binary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            binary_digest.update(chunk)
    return {"opponent": "bt4-policy", "backend": "blas", "device": "cpu", "search": "policyhead",
            "nodes": 1, "weights": str(network.resolve()), "weights_sha256": digest.hexdigest(),
            "weights_url": WEIGHTS_URL, "lc0_sha256": binary_digest.hexdigest(),
            "lc0_path": str(binary.resolve()), "command": engine_command(binary, network)}


def policy_move(engine, moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    if board.is_game_over():
        raise ValueError("Cannot request a move in a finished position")
    # New game token resets engine state between interleaved games; full history
    # is still supplied by python-chess for the network's history input planes.
    result = engine.play(board, chess.engine.Limit(nodes=1), game=object(), info=chess.engine.INFO_BASIC)
    if result.info.get("nodes") != 1:
        raise RuntimeError(f"LC0 policyhead must report exactly 1 node, got {result.info.get('nodes')}")
    if result.move not in board.legal_moves:
        raise RuntimeError("LC0 returned an illegal move")
    return result.move.uci()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if not line:
            return
        try:
            request = json.loads(line)
            with self.server.engine_lock:
                move = policy_move(self.server.engine, request["moves"])
            response = {"move": move}
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        self.wfile.write((json.dumps(response) + "\n").encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--lc0-path", default=DEFAULT_LC0)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    metadata = engine_metadata(args.lc0_path, args.weights)
    print(json.dumps(metadata), flush=True)
    engine = chess.engine.SimpleEngine.popen_uci(metadata["command"], timeout=120)
    try:
        engine.ping()
        with Server(args.socket, Handler) as server:
            server.engine = engine
            server.engine_lock = threading.Lock()
            def stop(*_args):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            print("READY", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.quit()
        Path(args.socket).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
