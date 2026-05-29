"""Estimate an anchored Elo for each LLM from games against the Maia 3 anchor ladder.

This uses maxim's exact rating math: `estimate_elo_from_blocks` below is copied
verbatim from get_refined_csv.py (Bradley-Terry / logistic MLE solved with Brent's
method, closed-form Fisher-information 95% CI, +/-35 white-advantage correction).
It is inlined rather than imported so this script depends only on scipy+stdlib,
not the full refiner's import chain (orjson/pandas/etc.). The only difference from
the Dragon pipeline is the anchor: Maia's Elo is used directly as the opponent
rating (no level->Elo formula), since Maia 3 is rating-conditioned.

It scans engine-vs-LLM run folders produced by run_maia_anchors.py:

    _logs/engine_vs_llm/maia-elo-<N>/<llm>/<ts>/_aggregate_results.json

and, per LLM, fits one Elo across all Maia anchors that model played.

Usage:
    python3 data/maia_elo.py
    python3 data/maia_elo.py --logs _logs/engine_vs_llm --out data/maia_elo.csv
"""

import argparse
import csv
import json
import math
import os
import re

MAIA_ELO_RE = re.compile(r"maia-elo-(\d+)")

# Elo points added after solving the (Black-only) MLE; mirrors get_refined_csv.
ELO_WHITE_ADVANTAGE = 35.0


def estimate_elo_from_blocks(blocks, white_advantage=ELO_WHITE_ADVANTAGE):
    """Estimate Elo from aggregated blocks where the model is always Black.

    Mirrors get_refined_csv.estimate_elo_from_blocks so this script stays
    dependency-light. blocks: list of (opponent_elo, wins, draws, losses).
    Returns (R_true, se_true), R_true including white_advantage. NaN if undefined.

    One guard is added vs. the original: an explicit perfect-separation check
    (overall score 0% or 100%). There the MLE diverges to +/-inf and must be NaN,
    but the original's bracket-expansion can let 10^x underflow to exactly 0.0,
    making f() hit zero so brentq returns a spurious finite root.
    """
    if not blocks:
        return float("nan"), float("nan")

    opp_elos, Ns, Ss = [], [], []
    for opp_elo, wins, draws, losses in blocks:
        N = int(wins) + int(draws) + int(losses)
        if N <= 0 or not isinstance(opp_elo, (int, float)):
            continue
        S = (int(wins) + 0.5 * int(draws)) / N
        opp_elos.append(float(opp_elo))
        Ns.append(N)
        Ss.append(S)

    if not opp_elos:
        return float("nan"), float("nan")

    # Perfect separation -> MLE has no finite solution.
    total_n = sum(Ns)
    s_bar = sum(S * N for S, N in zip(Ss, Ns)) / total_n
    if s_bar <= 0.0 or s_bar >= 1.0:
        return float("nan"), float("nan")

    def f(R):
        total = 0.0
        for opp, N, S in zip(opp_elos, Ns, Ss):
            E = 1.0 / (1.0 + 10.0 ** ((opp - R) / 400.0))
            total += N * (S - E)
        return total

    a = min(opp_elos) - 800.0
    b = max(opp_elos) + 800.0
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        width = 800.0
        for _ in range(6):
            a -= width
            b += width
            fa, fb = f(a), f(b)
            if fa * fb <= 0:
                break
            width *= 2.0
    if fa * fb > 0:
        return float("nan"), float("nan")

    from scipy.optimize import root_scalar

    root = root_scalar(f, bracket=(a, b), method="brentq").root

    info = 0.0
    for opp, N in zip(opp_elos, Ns):
        E = 1.0 / (1.0 + 10.0 ** ((opp - root) / 400.0))
        info += N * E * (1.0 - E)
    info *= (math.log(10.0) / 400.0) ** 2
    se_black = (1.0 / math.sqrt(info)) if info > 0 else float("nan")

    return float(root + white_advantage), float(se_black)


def _maia_elo_for_run(run_dir: str, agg: dict):
    """Anchor Elo for a run: prefer the path segment, fall back to _run metadata."""
    m = MAIA_ELO_RE.search(run_dir.replace(os.sep, "/").lower())
    if m:
        return int(m.group(1))
    maia = (agg.get("run_metadata", {}).get("chess_engines", {}) or {}).get("maia")
    if isinstance(maia, dict) and isinstance(maia.get("elo"), int):
        return maia["elo"]
    return None


def _llm_side(agg: dict):
    """Return (llm_model, llm_is_white) if this is a Maia-opponent run, else None."""
    white = agg.get("player_white", {})
    black = agg.get("player_black", {})
    white_is_maia = "maia" in str(white.get("name", "")).lower()
    black_is_maia = "maia" in str(black.get("name", "")).lower()
    if white_is_maia == black_is_maia:
        return None  # both or neither are Maia -> not a clean Maia-vs-LLM run
    if white_is_maia:
        model = black.get("model") or black.get("name", "unknown")
        return model, False  # LLM is Black
    model = white.get("model") or white.get("name", "unknown")
    return model, True  # LLM is White


def collect(logs_root: str):
    """model -> {"elos": {maia_elo: [w, d, l]}, "colors": set("W"/"B")}."""
    out = {}
    for dirpath, _dirs, files in os.walk(logs_root):
        if "_aggregate_results.json" not in files:
            continue
        path = os.path.join(dirpath, "_aggregate_results.json")
        try:
            with open(path, encoding="utf-8") as f:
                agg = json.load(f)
        except Exception as e:
            print(f"WARNING: could not read {path}: {e}")
            continue

        side = _llm_side(agg)
        elo = _maia_elo_for_run(dirpath, agg)
        if side is None or elo is None:
            continue
        model, llm_is_white = side

        white_wins = int(agg.get("white_wins", 0))
        black_wins = int(agg.get("black_wins", 0))
        draws = int(agg.get("draws", 0))
        if llm_is_white:
            wins, losses, color = white_wins, black_wins, "W"
        else:
            wins, losses, color = black_wins, white_wins, "B"

        rec = out.setdefault(model, {"elos": {}, "colors": set()})
        rec["colors"].add(color)
        bucket = rec["elos"].setdefault(elo, [0, 0, 0])
        bucket[0] += wins
        bucket[1] += draws
        bucket[2] += losses
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="_logs/engine_vs_llm",
                    help="root to scan for Maia anchor runs")
    ap.add_argument("--out", default="data/maia_elo.csv")
    args = ap.parse_args()

    data = collect(args.logs)
    if not data:
        print(f"No Maia anchor runs found under {args.logs}. "
              f"Run run_maia_anchors.py first.")
        return

    rows = []
    for model in sorted(data):
        rec = data[model]
        colors = rec["colors"]
        # white_advantage sign mirrors get_refined_csv: the LLM-as-Black solve adds
        # +35; an LLM-as-White solve subtracts it. Mixed colors can't share one sign.
        if colors == {"B"}:
            white_advantage = ELO_WHITE_ADVANTAGE
        elif colors == {"W"}:
            white_advantage = -ELO_WHITE_ADVANTAGE
        else:
            white_advantage = ELO_WHITE_ADVANTAGE
            print(f"WARNING: {model} has mixed LLM colors {colors}; "
                  f"white-advantage sign is ambiguous, defaulting to Black (+35).")

        blocks = [(elo, w, d, l) for elo, (w, d, l) in sorted(rec["elos"].items())]
        R, se = estimate_elo_from_blocks(blocks, white_advantage=white_advantage)
        moe = 1.96 * se if se == se else float("nan")  # se != se => NaN
        total = sum(w + d + l for _e, w, d, l in blocks)

        # Per-anchor score% (helps spot 0%/100% saturation that yields a blank Elo).
        anchor_str = "  ".join(
            f"{elo}:{((w + 0.5 * d) / (w + d + l)):.2f}({w + d + l})"
            for elo, (w, d, l) in sorted(rec["elos"].items())
        )
        rows.append({
            "model": model,
            "elo": "" if R != R else f"{R:.1f}",
            "elo_moe_95": "" if moe != moe else f"{moe:.1f}",
            "games": total,
            "colors": "".join(sorted(colors)),
            "per_anchor_score": anchor_str,
        })

    # Console table
    print(f"\n=== Maia-anchored Elo (anchors from {args.logs}) ===\n")
    w_model = max(len(r["model"]) for r in rows + [{"model": "model"}])
    print(f"{'model':<{w_model}}  {'elo':>8}  {'+/-95%':>7}  {'games':>5}  col  per-anchor score(games)")
    for r in rows:
        elo_disp = r["elo"] or "(none)"
        print(f"{r['model']:<{w_model}}  {elo_disp:>8}  {r['elo_moe_95'] or '':>7}  "
              f"{r['games']:>5}  {r['colors']:>3}  {r['per_anchor_score']}")
    print("\n(blank Elo = score is 0% or 100% across all anchors, so the MLE diverges; "
          "add an anchor the model scores ~35-65% against.)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
