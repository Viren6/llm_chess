"""Run the LLM against the Maia 3 79M anchor ladder, balancing colors.

Self-contained: this drives ``llm_chess.run()`` directly (one call per game) rather
than going through ``run_multiple_games.run_games``. That keeps the shared harness
untouched and avoids its multi-game aggregation entirely, so a single game per color
is fine.

For each Elo in MAIA_ELOS, plays ``--reps`` games with the LLM as White (Maia Black)
and ``--reps`` games with the LLM as Black (Maia White). The default ``--reps 1`` is
2 games per anchor — one per color. Each (anchor, color) batch writes its own
``_aggregate_results.json`` under
``_logs/engine_vs_llm/maia-elo-<N>/<llm>/<ts>_<color>/`` (re-running accumulates).
Estimate the color-balanced Elo afterwards with:

    python3 data/maia_elo.py

The LLM under test is resolved from .env: the White side (``_W`` keys) when it plays
White, the Black side (``_B`` keys) when it plays Black — so set BOTH to your model.

Usage:
    python3 run_maia_anchors.py                   # 600..1400, 1 game/color/anchor (2/anchor)
    python3 run_maia_anchors.py --reps 25         # 25 games/color/anchor
    python3 run_maia_anchors.py --elos 800 1000   # subset of anchors
    python3 run_maia_anchors.py --colors black    # single color only (no balancing)
"""

import argparse
import contextlib
import json
import os
import re
import sys
import time

import llm_chess
from utils import get_llms

MAIA_ELOS = [600, 800, 1000, 1200, 1400]
REPS_PER_COLOR = 1  # games per color per anchor (default => 2 games/anchor)

MAIA = llm_chess.PlayerType.CHESS_ENGINE_MAIA

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Current batch's output.txt (mutable holder). The _Tee objects are installed ONCE for the
# whole run (see _install_tee) and never closed, so anything that caches sys.stdout — notably
# AG2's logging handler — keeps a valid reference; only the file they point at is swapped per
# batch. An earlier version closed a per-batch tee instead, which left AG2 writing to a closed
# file on the next batch ("I/O operation on closed file").
_CUR_LOGFILE = [None]


class _Tee:
    """Forward writes to the real console and to the current batch file (ANSI stripped)."""

    def __init__(self, console):
        self._console = console

    def write(self, data):
        self._console.write(data)
        f = _CUR_LOGFILE[0]
        if f is not None and not f.closed:
            try:
                f.write(_ANSI_RE.sub("", data))
            except ValueError:
                pass
        return len(data)

    def flush(self):
        try:
            self._console.flush()
        except Exception:
            pass
        f = _CUR_LOGFILE[0]
        if f is not None and not f.closed:
            try:
                f.flush()
            except ValueError:
                pass

    def __getattr__(self, name):  # delegate isatty/encoding/fileno/... to the console
        return getattr(self._console, name)


@contextlib.contextmanager
def _install_tee():
    """Install persistent stdout/stderr tees for the whole run; restore them at the end."""
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(orig_out), _Tee(orig_err)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        f = _CUR_LOGFILE[0]
        if f is not None and not f.closed:
            f.close()
        _CUR_LOGFILE[0] = None


@contextlib.contextmanager
def _batch_log(log_folder, filename="output.txt"):
    """Point the persistent tee at <log_folder>/output.txt for one batch."""
    os.makedirs(log_folder, exist_ok=True)
    f = open(os.path.join(log_folder, filename), "w", encoding="utf-8")
    prev = _CUR_LOGFILE[0]
    _CUR_LOGFILE[0] = f
    try:
        yield
    finally:
        _CUR_LOGFILE[0] = prev
        try:
            f.flush()
            f.close()
        except Exception:
            pass


def _model_of(player) -> str:
    """Model id for a player, or '' for the engine (which has no llm_config)."""
    cfg = getattr(player, "llm_config", None)
    if isinstance(cfg, dict):
        lst = cfg.get("config_list") or [{}]
        return (lst[0] or {}).get("model", "") if lst else ""
    return ""


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-") or "llm"


def _run_batch(elo, llm_color, reps, cfg_w, cfg_b, model_slug):
    """Play `reps` games at one Maia Elo with the LLM on `llm_color`; write aggregate."""
    if llm_color == "white":
        llm_chess.white_player_type = llm_chess.PlayerType.LLM_WHITE
        llm_chess.black_player_type = MAIA
    else:  # black
        llm_chess.white_player_type = MAIA
        llm_chess.black_player_type = llm_chess.PlayerType.LLM_BLACK
    llm_chess.maia_elo = elo

    ts = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = os.path.join("_logs", "engine_vs_llm", f"maia-elo-{elo}",
                              model_slug, f"{ts}_{llm_color}")

    agg = {"total_games": 0, "white_wins": 0, "black_wins": 0, "draws": 0}
    pw_info = pb_info = {"name": "", "model": ""}
    # Tee this batch's full console (every model turn) into <log_folder>/output.txt.
    with _batch_log(log_folder):
        for _ in range(reps):
            stats, pw, pb = llm_chess.run(
                log_dir=log_folder,
                llm_config_white=cfg_w,
                llm_config_black=cfg_b,
            )
            winner = stats.get("winner")
            if winner == pw.name:
                agg["white_wins"] += 1
            elif winner == pb.name:
                agg["black_wins"] += 1
            else:
                agg["draws"] += 1
            agg["total_games"] += 1
            pw_info = {"name": pw.name, "model": _model_of(pw)}
            pb_info = {"name": pb.name, "model": _model_of(pb)}

        agg["player_white"] = pw_info
        agg["player_black"] = pb_info
        with open(os.path.join(log_folder, "_aggregate_results.json"), "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

    # Printed after stdout is restored, so the summary goes to the real console.
    print(f"  -> {log_folder}: white_wins={agg['white_wins']} "
          f"black_wins={agg['black_wins']} draws={agg['draws']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS_PER_COLOR,
                    help="games per color per anchor (total per anchor = reps x #colors)")
    ap.add_argument("--elos", type=int, nargs="+", default=MAIA_ELOS,
                    help="Maia Elo anchors to sweep")
    ap.add_argument("--colors", choices=["both", "white", "black"], default="both",
                    help="LLM color(s); 'both' (default) balances White's advantage")
    ap.add_argument("--maia-path", default=None,
                    help="override llm_chess.maia_path (the maia3-uci launcher)")
    ap.add_argument("--maia-model", default=None,
                    help="Maia model alias/HF repo via --model (default: llm_chess.maia_model = maia3-79m)")
    ap.add_argument("--maia-time", type=float, default=None,
                    help="override llm_chess.maia_time_per_move (s)")
    ap.add_argument("--maia-temperature", type=float, default=None,
                    help="Maia move-sampling temp; >0 = diverse human-like games, "
                         "0 = deterministic argmax (default: llm_chess.maia_temperature = 1.0)")
    ap.add_argument("--maia-top-p", type=float, default=None,
                    help="Maia nucleus-sampling threshold (1.0 = disabled)")
    ap.add_argument("--llm-temperature", type=float, default=None,
                    help="pin the LLM sampling temperature (default: provider default, ~1.0)")
    ap.add_argument("--provider", default=None,
                    help="pin a single OpenRouter provider endpoint, e.g. 'venice/bf16' "
                         "(provider.order=[<slug>], allow_fallbacks=false — prone to that "
                         "provider's rate limits)")
    ap.add_argument("--quant", default=None,
                    help="force provider precision, e.g. 'bf16' (provider.quantizations=[<q>]); "
                         "allows ANY provider at that precision so load spreads across them")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="enable the model's native thinking mode via "
                         "chat_template_kwargs.enable_thinking (DeepInfra/vLLM); reasoning goes to "
                         "the reasoning_content field, content stays clean (default: on)")
    ap.add_argument("--reason", action=argparse.BooleanOptionalAction, default=False,
                    help="prompt-based reasoning: instruct the LLM to write analysis then its move "
                         "in one message. Fallback for models without a thinking mode (default: off)")
    args = ap.parse_args()

    if args.maia_path is not None:
        llm_chess.maia_path = args.maia_path
    if args.maia_model is not None:
        llm_chess.maia_model = args.maia_model
    if args.maia_time is not None:
        llm_chess.maia_time_per_move = args.maia_time
    if args.maia_temperature is not None:
        llm_chess.maia_temperature = args.maia_temperature
    if args.maia_top_p is not None:
        llm_chess.maia_top_p = args.maia_top_p

    # Helpful defaults for API opponents: strip <think> blocks; tolerate rate limits.
    llm_chess.remove_text = llm_chess.DEFAULT_REMOVE_TEXT_REGEX
    llm_chess.max_api_retries = 6
    llm_chess.api_retry_delay = 2.0
    llm_chess.require_reasoning = args.reason  # write analysis, then make_move, in one message

    # Per-side LLM overrides: optionally pin temperature, and/or force an OpenRouter
    # provider. provider_overrides merges into config_list[0]; AG2 forwards `extra_body`
    # to the OpenAI create() call, and OpenRouter reads the `provider` field from it.
    hp = {}
    if args.llm_temperature is not None:
        hp["hyperparams"] = {"temperature": args.llm_temperature}
    # extra_body carries non-standard request fields that AG2 forwards to the OpenAI create()
    # call: OpenRouter provider routing (--provider/--quant) and the thinking-mode toggle
    # (chat_template_kwargs.enable_thinking, read by DeepInfra/vLLM).
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
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        hp["provider_overrides"] = {"extra_body": extra_body}
    cfg_w, cfg_b = get_llms(white_hyperparams=hp or None, black_hyperparams=hp or None)
    model_slug = _slug(_model_of_config(cfg_b) or _model_of_config(cfg_w))

    colors = ["white", "black"] if args.colors == "both" else [args.colors]
    with _install_tee():
        for elo in args.elos:
            for color in colors:
                print(f"\n\033[95m=== Maia Elo {elo} | LLM as {color} "
                      f"({args.reps} game(s)) ===\033[0m")
                _run_batch(elo, color, args.reps, cfg_w, cfg_b, model_slug)


def _model_of_config(cfg) -> str:
    try:
        return cfg["config_list"][0].get("model", "")
    except (KeyError, IndexError, TypeError):
        return ""


if __name__ == "__main__":
    main()
