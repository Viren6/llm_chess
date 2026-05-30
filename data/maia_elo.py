"""Estimate a color-balanced anchored Elo for each LLM from games vs the Maia 3 ladder.

Balanced colors are FORCED, not corrected for. Per Maia anchor, the LLM's score is
the equal-weight average of its White-side score and its Black-side score, so
White's first-move advantage cancels by construction and NO white-advantage fudge
is applied. Consequences:
  - Each color contributes exactly 50% to an anchor, even if the two colors ended
    up with unequal game counts (the larger color is down-weighted to the smaller).
  - An anchor with games on only ONE color cannot be balanced, so it is skipped
    (with a warning). If every anchor for a model is single-color, it gets no Elo.

Rating math otherwise mirrors get_refined_csv: a Bradley-Terry / logistic MLE
solved with Brent's method, plus a closed-form Fisher-information 95% CI. Maia's
Elo is used directly as the opponent rating (Maia 3 is rating-conditioned).

It scans engine-vs-LLM run folders produced by run_maia_anchors.py:

    _logs/engine_vs_llm/maia-elo-<N>/<llm>/<ts>/_aggregate_results.json

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


def fit_elo(opp_elos, Ns, Ss):
    """Logistic-MLE Elo from per-anchor (opponent Elo, weight N, score S in [0,1]).

    Solves Σ N_k (S_k - E_k(R)) = 0 for R via Brent; CI from Fisher information.
    There is NO color/white-advantage term — balancing is the caller's job. Returns
    (R, se); NaN if undefined (no anchors, or perfectly separated 0%/100%).
    """
    opp_elos = [float(o) for o in opp_elos]
    if not opp_elos:
        return float("nan"), float("nan")

    total_n = sum(Ns)
    if total_n <= 0:
        return float("nan"), float("nan")
    s_bar = sum(S * N for S, N in zip(Ss, Ns)) / total_n
    if s_bar <= 0.0 or s_bar >= 1.0:        # perfect separation -> no finite MLE
        return float("nan"), float("nan")

    def f(R):
        return sum(N * (S - 1.0 / (1.0 + 10.0 ** ((opp - R) / 400.0)))
                   for opp, N, S in zip(opp_elos, Ns, Ss))

    a, b = min(opp_elos) - 800.0, max(opp_elos) + 800.0
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
    se = (1.0 / math.sqrt(info)) if info > 0 else float("nan")
    return float(root), float(se)


def _maia_elo_for_run(run_dir, agg):
    """Anchor Elo for a run: prefer the path segment, fall back to _run metadata."""
    m = MAIA_ELO_RE.search(run_dir.replace(os.sep, "/").lower())
    if m:
        return int(m.group(1))
    maia = (agg.get("run_metadata", {}).get("chess_engines", {}) or {}).get("maia")
    if isinstance(maia, dict) and isinstance(maia.get("elo"), int):
        return maia["elo"]
    return None


def _llm_is_white(agg):
    """True if the LLM is White (Maia Black), False if LLM Black, None if not a Maia-vs-LLM run.

    Detected from player names (the engine is always 'Chess_Engine_Maia_*'), which is reliable
    even when the LLM model field is missing from the aggregate.
    """
    w = "maia" in str(agg.get("player_white", {}).get("name", "")).lower()
    b = "maia" in str(agg.get("player_black", {}).get("name", "")).lower()
    if w == b:
        return None  # both or neither are Maia -> not a clean Maia-vs-LLM run
    return not w     # LLM is White iff White is NOT Maia


def _model_key(dirpath):
    """The model is the folder segment between maia-elo-<N> and the per-game run folder, e.g.
    _logs/engine_vs_llm/maia-elo-1000/<MODEL-SLUG>/<run>/. Using the slug (not the aggregate's
    model field) groups White and Black games of the same model together and is stable across
    the per-batch and per-game folder layouts — and robust to a missing model field."""
    return os.path.basename(os.path.dirname(dirpath)) or "llm"


def collect(logs_root):
    """model -> {maia_elo -> {"W": [w,d,l], "B": [w,d,l]}} from the LLM's perspective."""
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

        llm_is_white = _llm_is_white(agg)
        elo = _maia_elo_for_run(dirpath, agg)
        if llm_is_white is None or elo is None:
            continue
        model = _model_key(dirpath)

        ww = int(agg.get("white_wins", 0))
        bw = int(agg.get("black_wins", 0))
        d = int(agg.get("draws", 0))
        if llm_is_white:
            wins, losses, key = ww, bw, "W"
        else:
            wins, losses, key = bw, ww, "B"

        anc = out.setdefault(model, {}).setdefault(elo, {"W": [0, 0, 0], "B": [0, 0, 0]})
        anc[key][0] += wins
        anc[key][1] += d
        anc[key][2] += losses
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
        opp_elos, Ns, Ss = [], [], []
        anchor_lines = []
        balanced_games = 0
        for elo, sides in sorted(data[model].items()):
            wW, dW, lW = sides["W"]
            wB, dB, lB = sides["B"]
            nW, nB = wW + dW + lW, wB + dB + lB
            if nW == 0 or nB == 0:
                print(f"WARNING: {model} anchor {elo} skipped — single color "
                      f"(B{nB}/W{nW}); both colors are required to force balance.")
                continue
            sW = (wW + 0.5 * dW) / nW
            sB = (wB + 0.5 * dB) / nB
            S = (sW + sB) / 2.0          # equal weight per color -> no white-advantage term
            n = min(nW, nB)              # down-weight the larger color to keep it balanced
            opp_elos.append(elo)
            Ss.append(S)
            Ns.append(2 * n)
            balanced_games += 2 * n
            anchor_lines.append(f"{elo}:{S:.2f}(B{nB}/W{nW})")

        R, se = fit_elo(opp_elos, Ns, Ss)
        moe = 1.96 * se if se == se else float("nan")  # se != se => NaN
        rows.append({
            "model": model,
            "elo": "" if R != R else f"{R:.1f}",
            "elo_moe_95": "" if moe != moe else f"{moe:.1f}",
            "balanced_games": balanced_games,
            "per_anchor_score": "  ".join(anchor_lines) if anchor_lines else "(no balanced anchors)",
        })

    # Console table
    print(f"\n=== Maia-anchored Elo — color-balanced, no white-advantage term "
          f"(anchors from {args.logs}) ===\n")
    w_model = max(len(r["model"]) for r in rows + [{"model": "model"}])
    print(f"{'model':<{w_model}}  {'elo':>8}  {'+/-95%':>7}  {'bal.games':>9}  per-anchor score(games)")
    for r in rows:
        elo_disp = r["elo"] or "(none)"
        print(f"{r['model']:<{w_model}}  {elo_disp:>8}  {r['elo_moe_95'] or '':>7}  "
              f"{r['balanced_games']:>9}  {r['per_anchor_score']}")
    print("\nPer anchor: balanced score = mean(White score, Black score); (Bn/Wn) = games "
          "per color. Blank Elo = no balanced anchor, or score 0%/100% across anchors.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
