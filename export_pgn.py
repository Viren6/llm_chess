"""Export validated benchmark game logs into one multi-game PGN file."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import tempfile

import chess.pgn


def player_label(record, side, source, aggregate):
    player = record[f"player_{side}"]
    if "maia" in player.get("name", "").lower():
        anchor = re.search(r"maia-elo-(\d+)", source.as_posix())
        return f"Maia 3 (Elo {anchor[1]})" if anchor else "Maia 3"
    model = player.get("model")
    label = model if model and model != "N/A" else player["name"]
    effort = aggregate.get("reasoning_effort")
    return f"{label} ({effort})" if effort else label


def load_game(source, logs_root):
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or "pgn" not in record:
        return None
    if record.get("reason") == "ERROR OCCURED":
        raise ValueError(f"{source}: infrastructure-error game cannot be exported as a result")
    if not isinstance(record["pgn"], str) or not record["pgn"].strip():
        raise ValueError(f"{source}: empty or invalid PGN")
    stream = io.StringIO(record["pgn"])
    game = chess.pgn.read_game(stream)
    if game is None or game.errors:
        raise ValueError(f"{source}: invalid PGN: {game.errors if game else 'no game'}")
    if chess.pgn.read_game(stream) is not None:
        raise ValueError(f"{source}: expected exactly one PGN game per log")
    moves = list(game.mainline_moves())
    if len(moves) != record["number_of_moves"]:
        raise ValueError(f"{source}: PGN has {len(moves)} plies; log reports {record['number_of_moves']}")
    winner = record["winner"]
    if winner == record["player_white"]["name"]:
        result = "1-0"
    elif winner == record["player_black"]["name"]:
        result = "0-1"
    elif winner == "NONE":
        result = "1/2-1/2"
    else:
        raise ValueError(f"{source}: unrecognized winner {winner!r}")
    if game.headers.get("Result") != result:
        raise ValueError(f"{source}: PGN result disagrees with logged winner")
    outcome = game.end().board().outcome()
    if outcome is not None and outcome.result() != result:
        raise ValueError(f"{source}: final board disagrees with logged result")
    aggregate_path = source.parent / "_aggregate_results.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    game.headers["Event"] = f"LLM Chess ({record.get('prompt_type', 'standard')})"
    game.headers["Site"] = "Local benchmark"
    for side in ("white", "black"):
        game.headers[side.title()] = player_label(record, side, source, aggregate)
    game.headers["PlyCount"] = str(len(moves))
    game.headers["BenchmarkTermination"] = record.get("reason", "")
    game.headers["SourceLog"] = source.relative_to(logs_root).as_posix()
    return game


def export_games(logs_root, output, *, run=None, expected_games=None):
    logs_root, output = Path(logs_root), Path(output)
    if not logs_root.is_dir():
        raise ValueError(f"Log directory not found: {logs_root}")
    games = []
    for source in sorted(logs_root.rglob("*.json")):
        if source.name.startswith("_") or source.name == "run_config.json":
            continue
        if run is not None and not source.parent.name.startswith(run + "_"):
            continue
        game = load_game(source, logs_root)
        if game is not None:
            games.append(game)
    if not games:
        raise ValueError("No games with PGN found in the selected logs")
    if expected_games is not None and len(games) != expected_games:
        raise ValueError(f"Expected {expected_games} games, found {len(games)}; output was not written")
    # Validate everything before replacing the output; never leave a partial PGN.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".pgn.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for index, game in enumerate(games, 1):
                game.headers["Round"] = str(index)
                handle.write(game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True)))
                handle.write("\n\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(games)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="directory to scan recursively for game JSON logs")
    parser.add_argument("--out", type=Path, required=True, help="destination multi-game PGN")
    parser.add_argument("--run", help="exact run prefix, e.g. 2026-09-05-21-47-07-p135893")
    parser.add_argument("--expected-games", type=int, help="refuse to write unless this many games are found")
    args = parser.parse_args()
    try:
        count = export_games(args.logs, args.out, run=args.run, expected_games=args.expected_games)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {count} validated games to {args.out}")


if __name__ == "__main__":
    main()
