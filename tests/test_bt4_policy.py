"""BT4 transport and scheduling regressions; all inference is mocked."""
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json

import chess
import pytest

import bt4_policy_server as bt4
import run_maia_concurrent as runner
from tests.test_openai_oauth import fake_codex


def test_policy_uses_one_node_full_history_and_fresh_state():
    engine = Mock()
    engine.play.return_value = SimpleNamespace(move=chess.Move.from_uci("e7e5"), info={"nodes": 1})
    assert bt4.policy_move(engine, ["e2e4"]) == "e7e5"
    first = engine.play.call_args
    assert [m.uci() for m in first.args[0].move_stack] == ["e2e4"]
    assert first.args[1].nodes == 1 and first.args[1].time is None
    bt4.policy_move(engine, ["e2e4"])
    assert first.kwargs["game"] is not engine.play.call_args.kwargs["game"]


@pytest.mark.parametrize("nodes,move", [(2, "e2e4"), (None, "e2e4"), (1, "e7e5")])
def test_bad_engine_response_is_error(nodes, move):
    engine = Mock()
    engine.play.return_value = SimpleNamespace(move=chess.Move.from_uci(move), info={"nodes": nodes})
    with pytest.raises(RuntimeError):
        bt4.policy_move(engine, [])


def test_metadata_pins_cpu_policy_and_exact_weights(tmp_path, monkeypatch):
    binary = tmp_path / "lc0"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    weights = tmp_path / "BT4.pb.gz"
    weights.write_bytes(b"weights")
    with pytest.raises(ValueError, match="SHA256"):
        bt4.engine_metadata(binary, weights)
    monkeypatch.setattr(bt4, "WEIGHTS_SHA256", hashlib.sha256(b"weights").hexdigest())
    meta = bt4.engine_metadata(binary, weights)
    assert meta["nodes"] == 1 and meta["device"] == "cpu"
    assert meta["command"][1] == "policyhead"
    assert "--backend=blas" in meta["command"]
    assert "--nncache=0" in meta["command"]
    assert "elo" not in meta


@pytest.mark.parametrize("color", ["white", "black"])
def test_simple_harness_bt4_both_colors(fake_codex, monkeypatch, color):
    import llm_chess
    import maia_server
    from utils import get_llms
    for side in ("W", "B"):
        monkeypatch.setenv(f"MODEL_KIND_{side}", "openai_oauth")
        monkeypatch.setenv(f"OPENAI_MODEL_NAME_{side}", "gpt-6-astra")
    white, black = get_llms({"reasoning_effort": "max"}, {"reasoning_effort": "max"})
    monkeypatch.setattr(llm_chess, "white_player_type", llm_chess.PlayerType.LLM_WHITE if color == "white" else llm_chess.PlayerType.CHESS_ENGINE_BT4_POLICY)
    monkeypatch.setattr(llm_chess, "black_player_type", llm_chess.PlayerType.LLM_BLACK if color == "black" else llm_chess.PlayerType.CHESS_ENGINE_BT4_POLICY)
    monkeypatch.setattr(llm_chess, "bt4_server", "/test/socket")
    monkeypatch.setattr(llm_chess, "max_game_moves", 2)
    monkeypatch.setattr(llm_chess, "visualize_board", False)
    request = Mock(return_value="e7e5" if color == "white" else "e2e4")
    monkeypatch.setattr(maia_server, "request_move", request)
    stats, pw, pb = llm_chess.run_simple(log_dir=None, llm_config_white=white, llm_config_black=black)
    assert stats["number_of_moves"] == 2 and stats["reason"] == "Max moves reached"
    assert stats["prompt_type"] == "simple"
    engine_player = pb if color == "white" else pw
    assert "BT4_Policy" in engine_player.name and "Maia" not in engine_player.name
    assert request.call_args.args[1] == (["e2e4"] if color == "white" else [])


def test_40_game_plan_one_shared_server_and_bt4_labels(tmp_path, monkeypatch):
    cfg = {"config_list": [{"model": "gpt-6-astra"}]}
    monkeypatch.setattr(runner, "get_llms", lambda **kw: (cfg, cfg))
    monkeypatch.setattr(runner, "_hyperparams", lambda args: None)
    monkeypatch.setattr(bt4, "engine_metadata", lambda *args: {"opponent": "bt4-policy"})
    start = Mock(return_value={0: ("/test/socket", None)})
    monkeypatch.setattr(runner, "_start_servers", start)
    monkeypatch.setattr(runner, "_stop_servers", Mock())
    plans = []
    def plan(args, jobs, servers, folder_for):
        plans.extend(jobs)
        assert args.concurrency == 20
        for job in jobs:
            assert "/bt4-policy/" in folder_for(*job)
    monkeypatch.setattr(runner, "_run_forked", plan)
    args = SimpleNamespace(opponent="bt4-policy", colors="both", prompt="simple", sf_order=False,
                           lc0_path="lc0", bt4_weights="BT4", reasoning_effort="max", reps=20,
                           concurrency=20, spawn=False, logs_root=str(tmp_path))
    runner._launch(args)
    assert len(plans) == 40
    assert sum(color == "white" for _, color, _ in plans) == 20
    assert sum(color == "black" for _, color, _ in plans) == 20
    start.assert_called_once_with([0], args)
