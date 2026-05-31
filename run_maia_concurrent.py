"""Concurrent Maia-anchor runner using shared GPU Maia servers.

Why this exists: llm_chess.run() uses module-level globals (board, player types, ...), so it
is NOT thread-safe — concurrent games must be separate PROCESSES. If each game spawned its own
Maia, the GPU would hold one ~1.2 GB Maia per concurrent game (≈240 GB for 200 games). Instead:

  - One Maia SERVER per anchor Elo (maia_server.py) loads Maia once on the GPU and serves moves
    over a Unix socket. GPU use = N_anchors × ~1.2 GB, independent of game concurrency.
  - Up to --concurrency game-WORKER processes run at once. Each worker plays ONE game, gets
    Maia moves from its anchor's server (no local Maia, no GPU), and writes its OWN folder:
        _logs/engine_vs_llm/maia-elo-<N>/<llm>/<ts>_<color>_j<idx>/
    with output.txt (full transcript) + the per-game JSON + _aggregate_results.json.

Output never interleaves: each worker is its own process and the launcher redirects that
worker's stdout/stderr straight to its own output.txt. The launcher console shows only short
progress lines. data/maia_elo.py sums the per-game aggregates exactly as before.

Usage:
    python run_maia_concurrent.py --reps 39 --concurrency 150 \
        --maia-path /workspace/llm-venv/bin/maia3-uci

(Set both .env sides to your model first, same as run_maia_anchors.py.)
"""

import argparse
import concurrent.futures
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

import llm_chess
from utils import get_llms

MAIA_ELOS = [600, 800, 1000, 1200, 1400]
MAIA = llm_chess.PlayerType.CHESS_ENGINE_MAIA


# --------------------------------------------------------------------------- helpers
def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-") or "llm"


def _model_of(player) -> str:
    """Model id for a player, or '' for the engine. AG2's llm_config is an LLMConfig object
    (subscriptable) or a dict — both support ['config_list'][0]['model']."""
    cfg = getattr(player, "llm_config", None)
    if not cfg:
        return ""
    try:
        return cfg["config_list"][0]["model"]
    except Exception:
        try:
            return getattr(cfg.config_list[0], "model", "") or ""
        except Exception:
            return ""


def _model_of_config(cfg) -> str:
    try:
        return cfg["config_list"][0].get("model", "")
    except (KeyError, IndexError, TypeError):
        return ""


def _hyperparams(args):
    """Per-side LLM overrides (temperature pin + reasoning/thinking/provider via extra_body)."""
    hp = {}
    if args.llm_temperature is not None:
        hp["hyperparams"] = {"temperature": args.llm_temperature}
    if getattr(args, "reasoning_effort", None) is not None:
        hp["reasoning_effort"] = args.reasoning_effort
    extra_body = {}
    provider = {}
    if args.quant is not None:
        provider["quantizations"] = [args.quant]
    if args.provider is not None:
        provider["order"] = [args.provider]
        provider["allow_fallbacks"] = False
    if provider:
        extra_body["provider"] = provider
    if args.thinking:
        # Different providers toggle native thinking differently:
        #  - chat_template (DeepInfra gemma/qwen): extra_body.chat_template_kwargs.enable_thinking
        #  - thinking_block (DeepSeek v4):         extra_body.thinking = {"type": "enabled"}
        style = getattr(args, "thinking_style", "chat_template")
        if style == "thinking_block":
            extra_body["thinking"] = {"type": "enabled"}
        else:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        hp["provider_overrides"] = {"extra_body": extra_body}
    return hp or None


def _add_llm_args(ap):
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="enable the model's native thinking mode (default: on)")
    ap.add_argument("--thinking-style", choices=["chat_template", "thinking_block"],
                    default="chat_template",
                    help="how --thinking is expressed: chat_template (DeepInfra gemma/qwen "
                         "enable_thinking) or thinking_block (DeepSeek extra_body.thinking)")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning_effort to send (e.g. high); provider must support it")
    ap.add_argument("--llm-temperature", type=float, default=None,
                    help="pin the LLM sampling temperature")
    ap.add_argument("--provider", default=None, help="force one OpenRouter provider endpoint")
    ap.add_argument("--quant", default=None, help="force OpenRouter provider precision, e.g. bf16")


# --------------------------------------------------------------------------- worker
def _worker(args):
    """Play ONE game vs the shared Maia server; print transcript to stdout; write aggregate."""
    llm_chess.maia_server = args.socket          # -> ChessEngineMaiaAgent goes remote
    llm_chess.maia_elo = args.elo                # for metadata only (server owns strength)
    if args.color == "white":
        llm_chess.white_player_type = llm_chess.PlayerType.LLM_WHITE
        llm_chess.black_player_type = MAIA
    else:
        llm_chess.white_player_type = MAIA
        llm_chess.black_player_type = llm_chess.PlayerType.LLM_BLACK
    llm_chess.remove_text = llm_chess.DEFAULT_REMOVE_TEXT_REGEX
    llm_chess.max_api_retries = 6
    llm_chess.api_retry_delay = 2.0

    hp = _hyperparams(args)
    cfg_w, cfg_b = get_llms(white_hyperparams=hp, black_hyperparams=hp)

    os.makedirs(args.out, exist_ok=True)
    stats, pw, pb = llm_chess.run(log_dir=args.out, llm_config_white=cfg_w, llm_config_black=cfg_b)

    winner = stats.get("winner")
    agg = {"total_games": 1, "white_wins": 0, "black_wins": 0, "draws": 0}
    if winner == pw.name:
        agg["white_wins"] = 1
    elif winner == pb.name:
        agg["black_wins"] = 1
    else:
        agg["draws"] = 1
    agg["player_white"] = {"name": pw.name, "model": _model_of(pw)}
    agg["player_black"] = {"name": pb.name, "model": _model_of(pb)}
    with open(os.path.join(args.out, "_aggregate_results.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)


# --------------------------------------------------------------------------- servers
def _start_servers(elos, args):
    """Launch one maia_server.py per Elo; return {elo: (socket_path, Popen)} once all ready."""
    servers = {}
    for elo in elos:
        sock = f"/tmp/maia-{elo}-{os.getpid()}.sock"
        log = open(f"maia_server_{elo}.log", "w")
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "maia_server.py"),
               "--elo", str(elo), "--socket", sock,
               "--maia-path", args.maia_path, "--maia-model", args.maia_model,
               "--temperature", str(args.maia_temperature), "--top-p", str(args.maia_top_p),
               "--time", str(args.maia_time)]
        if args.maia_device:
            cmd += ["--device", args.maia_device]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        servers[elo] = (sock, proc)

    deadline = time.time() + args.server_ready_timeout
    for elo, (sock, proc) in servers.items():
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"Maia server for elo {elo} exited early "
                                   f"(code {proc.returncode}); see maia_server_{elo}.log")
            try:
                c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                c.settimeout(2)
                c.connect(sock)
                c.close()
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError(f"Maia server for elo {elo} not ready in "
                                       f"{args.server_ready_timeout}s; see maia_server_{elo}.log")
                time.sleep(0.3)
        print(f"[server] maia-elo-{elo} ready ({sock})", flush=True)
    return servers


def _stop_servers(servers):
    for _elo, (sock, proc) in servers.items():
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
    for _elo, (sock, proc) in servers.items():
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            os.unlink(sock)
        except Exception:
            pass


# --------------------------------------------------------------------------- worker dispatch
def _run_child(args, elo, color, socket_path, folder):
    """In a forked child: redirect this game's output to its own output.txt and play one game.
    Uses os._exit so the child never runs the parent's atexit/finally (which would tear down
    the shared Maia servers)."""
    import copy
    import traceback
    fd = os.open(os.path.join(folder, "output.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    # Fresh line-buffered wrappers on the redirected fds so print() in the game lands in the file.
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8")
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8")
    ca = copy.copy(args)
    ca.elo, ca.color, ca.socket, ca.out = elo, color, socket_path, folder
    try:
        _worker(ca)
        sys.stdout.flush()
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _run_forked(args, jobs, servers, folder_for):
    """Fork one child per game from this already-imported parent. The ~300MB+ of autogen/
    llm_chess module pages are shared copy-on-write, so the cgroup (which counts each physical
    page once) sees total RAM stay ~flat as concurrency grows — unlike spawning a fresh
    interpreter per game (~350MB EACH). This is what lets high --concurrency (e.g. 200) fit a
    memory-capped pod. Forks happen from the single-threaded main loop (fork + threads is unsafe)."""
    import gc
    gc.collect()
    try:
        gc.freeze()  # move the shared heap into the permanent gen so GC won't dirty its CoW pages
    except Exception:
        pass

    total = len(jobs)
    job_iter = iter(jobs)
    running = {}  # pid -> (elo, color, folder)
    done = 0
    exhausted = False
    try:
        while not exhausted or running:
            while len(running) < args.concurrency and not exhausted:
                try:
                    elo, color, jidx = next(job_iter)
                except StopIteration:
                    exhausted = True
                    break
                folder = folder_for(elo, color, jidx)
                sys.stdout.flush()
                sys.stderr.flush()
                pid = os.fork()
                if pid == 0:
                    _run_child(args, elo, color, servers[elo][0], folder)
                    os._exit(0)  # unreachable: _run_child always exits
                running[pid] = (elo, color, folder)
            if not running:
                break
            pid, status = os.waitpid(-1, 0)
            if pid not in running:
                continue
            elo, color, folder = running.pop(pid)
            rc = os.waitstatus_to_exitcode(status)
            done += 1
            tag = "ok" if rc == 0 else f"FAILED(rc={rc})"
            print(f"[{done}/{total}] maia-elo-{elo} {color} {tag}", flush=True)
    finally:
        for pid in list(running):  # on error/interrupt, don't leave orphaned games
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        for pid in list(running):
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass


# --------------------------------------------------------------------------- launcher
def _launch(args):
    colors = ["white", "black"] if args.colors == "both" else [args.colors]

    # Validate .env + model slug once (fail fast before spinning up servers).
    cfg_w, cfg_b = get_llms(white_hyperparams=_hyperparams(args), black_hyperparams=_hyperparams(args))
    model_slug = _slug(_model_of_config(cfg_b) or _model_of_config(cfg_w))

    jobs = [(elo, color, i)
            for elo in args.elos for color in colors for i in range(args.reps)]
    # Per-launch id: timestamp + launcher PID, so folders are unique within a run (idx+color)
    # AND across re-runs (different PID) — the per-game JSON is only minute-resolution, so each
    # game must get its own folder to avoid clobbering.
    ts = time.strftime("%Y-%m-%d-%H-%M-%S") + f"-p{os.getpid()}"
    mode = "spawn" if args.spawn else "fork"
    print(f"[plan] {len(jobs)} games "
          f"({len(args.elos)} anchors x {len(colors)} colors x {args.reps} reps), "
          f"concurrency={args.concurrency}, mode={mode}, model={model_slug}", flush=True)

    servers = _start_servers(args.elos, args)
    self_path = os.path.abspath(__file__)

    def folder_for(elo, color, idx):
        folder = os.path.join(args.logs_root, "engine_vs_llm", f"maia-elo-{elo}",
                              model_slug, f"{ts}_{color}_j{idx}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def run_one_spawn(job):
        elo, color, idx = job
        folder = folder_for(elo, color, idx)
        cmd = [sys.executable, self_path, "--worker",
               "--elo", str(elo), "--color", color, "--socket", servers[elo][0],
               "--out", folder]
        cmd += ["--thinking"] if args.thinking else ["--no-thinking"]
        if args.llm_temperature is not None:
            cmd += ["--llm-temperature", str(args.llm_temperature)]
        if args.provider is not None:
            cmd += ["--provider", args.provider]
        if args.quant is not None:
            cmd += ["--quant", args.quant]
        with open(os.path.join(folder, "output.txt"), "w", encoding="utf-8") as out:
            rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
        return elo, color, folder, rc

    try:
        if args.spawn:
            done = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [ex.submit(run_one_spawn, j) for j in jobs]
                for fut in concurrent.futures.as_completed(futures):
                    elo, color, folder, rc = fut.result()
                    done += 1
                    tag = "ok" if rc == 0 else f"FAILED(rc={rc})"
                    print(f"[{done}/{len(jobs)}] maia-elo-{elo} {color} {tag}", flush=True)
        else:
            _run_forked(args, jobs, servers, folder_for)
    finally:
        _stop_servers(servers)
    print("=== all games complete ===  now run: python data/maia_elo.py", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    # worker-only:
    ap.add_argument("--elo", type=int)
    ap.add_argument("--color", choices=["white", "black"])
    ap.add_argument("--socket")
    ap.add_argument("--out")
    # launcher:
    ap.add_argument("--elos", type=int, nargs="+", default=MAIA_ELOS)
    ap.add_argument("--reps", type=int, default=1, help="games per color per anchor")
    ap.add_argument("--colors", choices=["both", "white", "black"], default="both")
    ap.add_argument("--concurrency", type=int, default=100,
                    help="max concurrent game workers (~= concurrent API requests)")
    ap.add_argument("--spawn", action="store_true",
                    help="spawn a fresh interpreter per game (~350MB RAM EACH) instead of the "
                         "default fork model (shared copy-on-write heap, ~flat RAM). Use only if "
                         "fork misbehaves; fork is what makes high --concurrency fit a RAM cap.")
    ap.add_argument("--logs-root", default="_logs")
    ap.add_argument("--server-ready-timeout", type=float, default=120)
    ap.add_argument("--maia-path", default="/workspace/llm-venv/bin/maia3-uci")
    ap.add_argument("--maia-model", default="maia3-79m")
    ap.add_argument("--maia-temperature", type=float, default=1.0)
    ap.add_argument("--maia-top-p", type=float, default=1.0)
    ap.add_argument("--maia-time", type=float, default=0.2)
    ap.add_argument("--maia-device", default=None, help="e.g. 'cpu' to keep Maia off the GPU")
    _add_llm_args(ap)
    args = ap.parse_args()

    if args.worker:
        _worker(args)
    else:
        _launch(args)


if __name__ == "__main__":
    main()
