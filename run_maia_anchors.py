"""Run the Black-side LLM against the Maia 3 79M anchor ladder.

Plays NUM_REPETITIONS games against Maia at each Elo in MAIA_ELOS, writing one
run folder per anchor under _logs/engine_vs_llm/maia-elo-<N>/<llm>/<ts>/. After
the sweep finishes, estimate a single anchored Elo with:

    python3 data/maia_elo.py

The LLM under test is whatever the .env BLACK side resolves to (same as the rest
of the harness). To test several models, set the model in .env and re-run this
script per model. Maia plays White; the LLM plays Black (matching the existing
engine-vs-LLM convention, so the +35 white-advantage correction in the Elo
estimator applies unchanged).

Usage:
    python3 run_maia_anchors.py                 # 600 800 1000 1200 1400, NUM_REPETITIONS each
    python3 run_maia_anchors.py --reps 50
    python3 run_maia_anchors.py --elos 800 1000 1200
"""

import argparse

import llm_chess
import run_multiple_games as rmg

# Anchor ladder requested for the experiment.
MAIA_ELOS = [600, 800, 1000, 1200, 1400]
NUM_REPETITIONS = 50  # games per anchor (per color is not split here; Maia is always White)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=NUM_REPETITIONS,
                    help="games per Maia anchor")
    ap.add_argument("--elos", type=int, nargs="+", default=MAIA_ELOS,
                    help="Maia Elo anchors to sweep")
    ap.add_argument("--maia-path", default=None,
                    help="override llm_chess.maia_path")
    ap.add_argument("--maia-time", type=float, default=None,
                    help="override llm_chess.maia_time_per_move (s)")
    args = ap.parse_args()

    if args.maia_path is not None:
        llm_chess.maia_path = args.maia_path
    if args.maia_time is not None:
        llm_chess.maia_time_per_move = args.maia_time

    rmg.WHITE_PLAYER_TYPE = llm_chess.PlayerType.CHESS_ENGINE_MAIA
    rmg.NUM_REPETITIONS = args.reps

    for elo in args.elos:
        print(f"\n\033[95m=== Maia anchor: Elo {elo} ({args.reps} games) ===\033[0m")
        rmg.ENGINE_LEVEL = elo          # ENGINE_LEVEL doubles as the Maia Elo
        rmg.LOG_FOLDER = None           # force a fresh per-anchor log folder
        rmg.run_games()


if __name__ == "__main__":
    main()
