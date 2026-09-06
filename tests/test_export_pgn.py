import json

import chess.pgn
import pytest

from export_pgn import export_games


def write_game(root, name="run_white_j0", **changes):
    folder = root / "maia-elo-2400" / "astra" / name
    folder.mkdir(parents=True, exist_ok=True)
    record = {"pgn": '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1',
              "number_of_moves": 4, "winner": "Maia_Black", "reason": "Checkmate",
              "player_white": {"name": "Player_White", "model": "gpt-6-astra"},
              "player_black": {"name": "Maia_Black", "model": "N/A"}, "prompt_type": "simple"}
    record.update(changes)
    (folder / "game.json").write_text(json.dumps(record))
    (folder / "_aggregate_results.json").write_text('{"reasoning_effort":"max"}')
    return folder


def test_export_names_moves_and_run_selection(tmp_path):
    logs = tmp_path / "logs"
    write_game(logs)
    write_game(logs, "other_white_j0")
    out = tmp_path / "all.pgn"
    assert export_games(logs, out, run="run", expected_games=1) == 1
    with out.open() as handle:
        game = chess.pgn.read_game(handle)
        assert chess.pgn.read_game(handle) is None
    assert game.headers["White"] == "gpt-6-astra (max)"
    assert game.headers["Black"] == "Maia 3 (Elo 2400)"
    assert game.end().board().is_checkmate()
    assert game.headers["Result"] == "0-1"


@pytest.mark.parametrize("changes", [{"number_of_moves": 3}, {"winner": "Player_White"},
                                    {"reason": "ERROR OCCURED"}, {"pgn": ""}])
def test_bad_log_preserves_existing_output(tmp_path, changes):
    logs = tmp_path / "logs"
    write_game(logs, **changes)
    out = tmp_path / "all.pgn"
    out.write_text("existing")
    with pytest.raises(ValueError):
        export_games(logs, out)
    assert out.read_text() == "existing"


def test_count_mismatch_does_not_write(tmp_path):
    logs = tmp_path / "logs"
    write_game(logs)
    out = tmp_path / "all.pgn"
    with pytest.raises(ValueError, match="Expected 40 games, found 1"):
        export_games(logs, out, expected_games=40)
    assert not out.exists()
