"""Resume failed Astra-max/simple BT4 games in a separate, concurrent recovery batch.

Default is a read-only plan. --execute starts new OAuth requests and a private LC0
server. Original logs/processes/sockets are never modified. Use a fresh --out for
each batch; to recover a failed continuation, scan the previous recovery directory.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import chess
import chess.pgn


def checkpoint_board(record):
    if record.get("reason") != "ERROR OCCURED" or record.get("prompt_type") != "simple":
        raise ValueError("Only failed simple-UCI games can be resumed")
    stream = io.StringIO(record["pgn"])
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or chess.pgn.read_game(stream) is not None:
        raise ValueError("Invalid checkpoint PGN")
    if game.board().fen() != chess.Board().fen():
        raise ValueError("Nonstandard starting position is unsupported")
    board = game.end().board()
    if len(board.move_stack) != record["number_of_moves"]:
        raise ValueError("Checkpoint move count mismatch")
    if board.is_game_over() or len(board.move_stack) >= 200:
        raise ValueError("Checkpoint is already terminal or at the move limit")
    return board


def load_checkpoint(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    record = json.loads(raw)
    checkpoint_board(record)
    record.setdefault("_source", str(path))
    record.setdefault("_source_sha256", hashlib.sha256(raw).hexdigest())
    return record


def discover(logs, run):
    jobs = []
    for folder in sorted(Path(logs).glob(run + "_*")):
        if not folder.is_dir():
            continue
        transcript = folder / "output.txt"
        if not transcript.exists():
            continue
        text = transcript.read_text()
        if "RuntimeError: Game failed;" not in text:
            continue  # Active and completed games are left alone.
        candidates = []
        for path in folder.glob("*.json"):
            record = json.loads(path.read_text())
            if isinstance(record, dict) and record.get("reason") == "ERROR OCCURED":
                candidates.append(path)
        if len(candidates) != 1:
            raise ValueError(f"{folder}: expected one failed checkpoint")
        source = candidates[0]
        record = load_checkpoint(source)
        board = checkpoint_board(record)
        logged = re.findall(r"MADE MOVE (\d+): \S+ ([a-h][1-8][a-h][1-8][nbrq]?)", text)
        for ply, move in logged:
            if int(ply) < 1 or int(ply) > len(board.move_stack) or board.move_stack[int(ply)-1].uci() != move:
                raise ValueError(f"{folder}: transcript disagrees with saved PGN")
        config = json.loads((folder / "run_config.json").read_text())
        if (config.get("model") != "gpt-6-astra" or config.get("reasoning_effort") != "max"
                or config.get("prompt_type") != "simple"
                or config.get("llm_config", {}).get("auth") != "chatgpt_oauth"
                or config.get("engine", {}).get("opponent") != "bt4-policy"
                or config.get("llm_color") not in ("white", "black")):
            raise ValueError(f"{folder}: unsupported original configuration")
        color = config["llm_color"]
        if record[f"player_{color}"].get("model") != "gpt-6-astra":
            raise ValueError(f"{folder}: model/color mismatch")
        jobs.append((source, record, config))
    return jobs


def execute(jobs, out, concurrency=20):
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not jobs:
        return 0
    # Heavy harness imports and all engine/model activity occur only after --execute.
    from types import SimpleNamespace
    import run_maia_concurrent as runner
    from bt4_policy_server import engine_metadata
    first = jobs[0][2]["engine"]
    metadata = engine_metadata(first["lc0_path"], first["weights"])
    for _, _, config in jobs:
        for key in ("lc0_sha256", "weights_sha256", "nodes", "search", "backend"):
            if config["engine"].get(key) != metadata[key]:
                raise ValueError(f"Engine configuration mismatch: {key}")
    out = Path(out).resolve()
    for source, _, _ in jobs:
        if out == source.parent or out in source.parents or source.parent in out.parents:
            raise ValueError("Recovery output must be separate from original game folders")
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for side in ("W", "B"):
        env[f"MODEL_KIND_{side}"] = "openai_oauth"
        env[f"OPENAI_MODEL_NAME_{side}"] = "gpt-6-astra"
    repo = Path(__file__).resolve().parent
    previous_cwd = Path.cwd()
    servers = {}
    failures = 0
    active = []
    try:
        # Isolates the server log as well as its PID-specific socket and process.
        os.chdir(out)
        servers = runner._start_bt4_server(SimpleNamespace(
            lc0_path=first["lc0_path"], bt4_weights=first["weights"], server_ready_timeout=120))
        pending = iter(jobs)
        exhausted = False
        while active or not exhausted:
            while len(active) < concurrency and not exhausted:
                try:
                    source, record, config = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                destination = out / source.parent.name
                destination.mkdir()
                # Snapshot outside the game folder, so stats readers cannot double-count it.
                snapshot = out / (source.parent.name + ".checkpoint")
                snapshot.write_text(json.dumps(record, indent=2))
                command = [sys.executable, str(repo / "run_maia_concurrent.py"),
                           "--worker", "--opponent", "bt4-policy", "--elo", "0",
                           "--color", config["llm_color"], "--socket", servers[0][0],
                           "--out", str(destination), "--resume-json", str(snapshot),
                           "--lc0-path", first["lc0_path"], "--bt4-weights", first["weights"],
                           "--prompt", "simple", "--reasoning-effort", "max", "--no-thinking"]
                with (destination / "output.txt").open("x") as log:
                    child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                active.append((child, source, record))
            for child, source, record in active[:]:
                rc = child.poll()
                if rc is None:
                    continue
                child.wait()
                active.remove((child, source, record))
                failures += rc != 0
                print(f"{source.parent.name}: resumed from ply {record['number_of_moves']}, rc={rc}", flush=True)
            if active:
                time.sleep(0.1)
    finally:
        for child, _, _ in active:
            if child.poll() is None:
                child.terminate()
        for child, _, _ in active:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        runner._stop_servers(servers)
        os.chdir(previous_cwd)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--run", required=True, help="exact original run prefix")
    parser.add_argument("--out", type=Path, required=True, help="new, separate recovery directory")
    parser.add_argument("--concurrency", type=int, default=20, help="maximum simultaneous recovery games (default: 20)")
    parser.add_argument("--execute", action="store_true", help="start recovery; default only prints plan")
    args = parser.parse_args()
    if not args.logs.is_dir() or not re.fullmatch(r"[A-Za-z0-9-]+", args.run):
        parser.error("existing --logs and a literal run prefix are required")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    jobs = discover(args.logs, args.run)
    for source, record, _ in jobs:
        board = checkpoint_board(record)
        print(f"{source.parent.name}: {len(board.move_stack)} plies saved; next={'white' if board.turn else 'black'}")
    print(f"{len(jobs)} failed games eligible; originals untouched; recovery concurrency={min(args.concurrency, len(jobs))}.")
    if args.execute and jobs:
        return execute(jobs, args.out, args.concurrency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
