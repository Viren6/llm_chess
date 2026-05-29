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
import json
import os
import re
import time

import llm_chess
from utils import get_llms

MAIA_ELOS = [600, 800, 1000, 1200, 1400]
REPS_PER_COLOR = 1  # games per color per anchor (default => 2 games/anchor)

MAIA = llm_chess.PlayerType.CHESS_ENGINE_MAIA


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
    os.makedirs(log_folder, exist_ok=True)

    agg = {"total_games": 0, "white_wins": 0, "black_wins": 0, "draws": 0}
    pw_info = pb_info = {"name": "", "model": ""}
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
    ap.add_argument("--maia-path", default=None, help="override llm_chess.maia_path")
    ap.add_argument("--maia-time", type=float, default=None,
                    help="override llm_chess.maia_time_per_move (s)")
    args = ap.parse_args()

    if args.maia_path is not None:
        llm_chess.maia_path = args.maia_path
    if args.maia_time is not None:
        llm_chess.maia_time_per_move = args.maia_time

    # Helpful defaults for API opponents: strip <think> blocks; tolerate rate limits.
    llm_chess.remove_text = llm_chess.DEFAULT_REMOVE_TEXT_REGEX
    llm_chess.max_api_retries = 6
    llm_chess.api_retry_delay = 2.0

    cfg_w, cfg_b = get_llms()  # reads .env (_W and _B); both sides must be your model
    model_slug = _slug(_model_of_config(cfg_b) or _model_of_config(cfg_w))

    colors = ["white", "black"] if args.colors == "both" else [args.colors]
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
