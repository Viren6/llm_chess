"""Run GPT-6 Astra (max reasoning) vs Maia 3 Elo 2400 using ChatGPT OAuth."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from openai_oauth import AppServer


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local_maia = Path.home() / ".local/share/llm-chess/maia-env/bin/maia3-uci"
    parser.add_argument("--maia-path", default=shutil.which("maia3-uci") or
                        (str(local_maia) if local_maia.is_file() else "maia3-uci"))
    parser.add_argument("--reps", type=positive_int, default=1, help="games per color (default: 2 games total)")
    parser.add_argument("--colors", choices=["both", "white", "black"], default="both")
    parser.add_argument("--max-plies", type=positive_int, default=200)
    parser.add_argument("--timeout", type=positive_int, default=7200, help="seconds per LLM request")
    parser.add_argument("--logs-root", type=Path, default=Path("_logs"))
    parser.add_argument("--check", action="store_true", help="check OAuth, model access and Maia executable; no games")
    args = parser.parse_args()
    with AppServer(timeout=60) as server:
        server.require_chatgpt()
        server.check_model("gpt-6-astra", "max")
    maia_path = shutil.which(args.maia_path)
    if maia_path is None:
        parser.error("Maia 3 executable missing; install Maia 3 and pass --maia-path /path/to/maia3-uci (README)")
    if args.check:
        print(f"Ready: ChatGPT OAuth, gpt-6-astra/max, Maia 3 Elo 2400 ({maia_path}), simple UCI")
        return

    import llm_chess
    from utils import get_llms
    from termination_reasons import TerminationReason

    for side in ("W", "B"):
        os.environ[f"MODEL_KIND_{side}"] = "openai_oauth"
        os.environ[f"OPENAI_MODEL_NAME_{side}"] = "gpt-6-astra"
    hp = {"reasoning_effort": "max"}
    cfg_w, cfg_b = get_llms(white_hyperparams=hp, black_hyperparams=hp, timeout=args.timeout)
    llm_chess.maia_path = maia_path
    llm_chess.maia_model = "maia3-79m"
    llm_chess.maia_elo = 2400
    llm_chess.maia_server = None
    llm_chess.maia_temperature = 1.0
    llm_chess.maia_top_p = 1.0
    llm_chess.maia_use_uci_history = True
    llm_chess.reset_maia_history = False
    llm_chess.max_game_moves = args.max_plies
    root = args.logs_root / "engine_vs_llm/maia-elo-2400/gpt-6-astra-max-oauth-simple" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    colors = ["white", "black"] if args.colors == "both" else [args.colors]
    for rep in range(args.reps):
        for color in colors:
            out = root / f"{color}-{rep + 1}"
            out.mkdir(parents=True, exist_ok=False)
            llm_chess.white_player_type = (llm_chess.PlayerType.LLM_WHITE if color == "white"
                                          else llm_chess.PlayerType.CHESS_ENGINE_MAIA)
            llm_chess.black_player_type = (llm_chess.PlayerType.LLM_BLACK if color == "black"
                                          else llm_chess.PlayerType.CHESS_ENGINE_MAIA)
            metadata = {"model": "gpt-6-astra", "reasoning_effort": "max", "auth": "chatgpt_oauth",
                        "transport": "codex_app_server", "prompt_type": "simple", "notation": "uci",
                        "maia_model": "maia3-79m", "maia_elo": 2400, "maia_temperature": 1.0,
                        "maia_top_p": 1.0, "maia_use_uci_history": True, "llm_color": color,
                        "max_plies": args.max_plies, "timeout": args.timeout}
            (out / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Playing Astra/max ({color}) vs Maia 2400; transcript: {out / 'output.txt'}", flush=True)
            with (out / "output.txt").open("w") as transcript, redirect_stdout(transcript), redirect_stderr(transcript):
                stats, pw, pb = llm_chess.run_simple(log_dir=str(out), llm_config_white=cfg_w,
                    llm_config_black=cfg_b, notation="uci", explain=False, api_timeout=args.timeout)
            if stats["reason"] == TerminationReason.ERROR.value:
                raise RuntimeError(f"Game failed; see {out / 'output.txt'} (not counted as a draw)")
            aggregate = {"total_games": 1, "white_wins": int(stats["winner"] == pw.name),
                         "black_wins": int(stats["winner"] == pb.name),
                         "draws": int(stats["winner"] not in (pw.name, pb.name)),
                         "prompt_type": "simple", "reasoning_effort": "max", "auth": "chatgpt_oauth",
                         "player_white": {"name": pw.name, "model": "gpt-6-astra" if color == "white" else ""},
                         "player_black": {"name": pb.name, "model": "gpt-6-astra" if color == "black" else ""}}
            (out / "_aggregate_results.json").write_text(json.dumps(aggregate, indent=2) + "\n")
            print(f"Finished: {stats['winner']} — {stats['reason']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
