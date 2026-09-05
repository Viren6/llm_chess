"""Exercise the real subprocess transport and AG2/simple-harness boundary offline."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from openai_oauth import AppServer, OpenAIOAuthClient, codex_env


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / "codex"
    script.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys, time
def emit(x):
    print(json.dumps(x), flush=True)
for line in sys.stdin:
    m = json.loads(line)
    if 'id' not in m:
        continue
    method, p = m.get('method'), m.get('params', {})
    with open(os.environ['OAUTH_TEST_REQUESTS'], 'a') as log:
        log.write(json.dumps(m) + '\\n')
    result = {}
    if method == 'account/read':
        result = {'account': {'type': os.environ.get('OAUTH_TEST_AUTH', 'chatgpt')}}
    elif method == 'model/list':
        result = {'data': [{'model': 'gpt-6-astra', 'supportedReasoningEfforts': [{'reasoningEffort': 'max'}]}]}
    elif method == 'thread/start':
        result = {'thread': {'id': 'thread1'}, 'model': p['model']}
    elif method == 'turn/start':
        mode = os.environ.get('OAUTH_TEST_MODE', '')
        if mode == 'timeout':
            time.sleep(20)
        if mode == 'exit':
            sys.exit(1)
        result = {'turn': {'id': 'turn1'}}
        # Intentionally send events BEFORE the RPC response.
        move = 'e7e5' if 'as black' in p['input'][0]['text'] else 'e2e4'
        item = {'type': 'agentMessage', 'text': 'make_move ' + move}
        if mode == 'tool':
            item = {'type': 'commandExecution'}
        emit({'method': 'item/completed', 'params': {'threadId': 'thread1', 'item': item}})
        emit({'method': 'thread/tokenUsage/updated', 'params': {'threadId': 'thread1',
            'tokenUsage': {'total': {'inputTokens': 31, 'outputTokens': 7}}}})
        turn = {'id': 'turn1', 'status': 'failed' if mode == 'failed' else 'completed'}
        emit({'method': 'turn/completed', 'params': {'threadId': 'thread1', 'turn': turn}})
    emit({'id': m['id'], 'result': result})
''')
    script.chmod(0o755)
    monkeypatch.setenv("LLM_CHESS_CODEX_BIN", str(script))
    monkeypatch.setenv("LLM_CHESS_CODEX_HOME", str(tmp_path / "auth"))
    requests = tmp_path / "requests.jsonl"
    monkeypatch.setenv("OAUTH_TEST_REQUESTS", str(requests))
    return requests


def test_oauth_env_strips_api_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CHESS_CODEX_HOME", str(tmp_path / "auth"))
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_ACCESS_TOKEN"):
        monkeypatch.setenv(key, "do-not-use")
    env = codex_env()
    assert not any(k in env for k in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_ACCESS_TOKEN"))
    assert env["CODEX_HOME"] == str(tmp_path / "auth")


@pytest.mark.parametrize("auth", ["apiKey", "", "amazonBedrock"])
def test_refuse_non_subscription(fake_codex, monkeypatch, auth):
    monkeypatch.setenv("OAUTH_TEST_AUTH", auth)
    with AppServer(2) as server, pytest.raises(RuntimeError, match="OAuth login required"):
        server.require_chatgpt()
    assert not any(json.loads(line)["method"] == "turn/start" for line in fake_codex.read_text().splitlines())


def test_completion_and_stateless_config(fake_codex):
    client = OpenAIOAuthClient({"model": "gpt-6-astra", "reasoning_effort": "max", "timeout": 2})
    for _ in range(2):
        response = client.create({"messages": [{"role": "system", "content": "Chess only"},
                                               {"role": "user", "content": "Legal moves: e2e4"}]})
        assert client.message_retrieval(response) == ["make_move e2e4"]
        assert client.get_usage(response) == {"prompt_tokens": 31, "completion_tokens": 7,
                                              "total_tokens": 38, "cost": 0.0, "model": "gpt-6-astra"}
    requests = [json.loads(line) for line in fake_codex.read_text().splitlines()]
    starts = [m["params"] for m in requests if m["method"] == "thread/start"]
    assert len(starts) == 2 and starts[0]["cwd"] != starts[1]["cwd"]
    assert all(m["ephemeral"] and m["baseInstructions"] == "Chess only" for m in starts)
    assert all(m["config"]["features.shell_tool"] is False for m in starts)
    assert all(m["params"]["effort"] == "max" for m in requests if m["method"] == "turn/start")


@pytest.mark.parametrize("mode,pattern", [("tool", "tool-free"), ("failed", "turn failed"), ("exit", "exited")])
def test_failed_turns(fake_codex, monkeypatch, mode, pattern):
    monkeypatch.setenv("OAUTH_TEST_MODE", mode)
    with AppServer(2) as server, pytest.raises(RuntimeError, match=pattern):
        server.complete("gpt-6-astra", "max", [{"role": "user", "content": "Move"}])
    assert server.proc.poll() is not None


def test_timeout_cleans_process(fake_codex, monkeypatch):
    monkeypatch.setenv("OAUTH_TEST_MODE", "timeout")
    with AppServer(0.3) as server, pytest.raises(TimeoutError):
        server.complete("gpt-6-astra", "max", [{"role": "user", "content": "Move"}])
    assert server.proc.poll() is not None


def test_model_preflight(fake_codex):
    with AppServer(2) as server:
        server.require_chatgpt()
        server.check_model("gpt-6-astra", "max")
        with pytest.raises(ValueError, match="not available"):
            server.check_model("astra0max", "max")
        with pytest.raises(ValueError, match="does not advertise"):
            server.check_model("gpt-6-astra", "ultra")


@pytest.mark.parametrize("color", ["white", "black"])
def test_simple_harness_uses_oauth_and_records_usage(fake_codex, monkeypatch, color):
    import llm_chess
    from utils import get_llms
    for side in ("W", "B"):
        monkeypatch.setenv(f"MODEL_KIND_{side}", "openai_oauth")
        monkeypatch.setenv(f"OPENAI_MODEL_NAME_{side}", "gpt-6-astra")
    cfg_w, cfg_b = get_llms({"reasoning_effort": "max"}, {"reasoning_effort": "max"})
    monkeypatch.setattr(llm_chess, "white_player_type", llm_chess.PlayerType.LLM_WHITE if color == "white" else llm_chess.PlayerType.CHESS_ENGINE_MAIA)
    monkeypatch.setattr(llm_chess, "black_player_type", llm_chess.PlayerType.LLM_BLACK if color == "black" else llm_chess.PlayerType.CHESS_ENGINE_MAIA)
    monkeypatch.setattr(llm_chess, "max_game_moves", 2)
    monkeypatch.setattr(llm_chess, "visualize_board", False)
    monkeypatch.setattr(llm_chess.ChessEngineMaiaAgent, "generate_reply", lambda self, **kw: "make_move e7e5" if color == "white" else "make_move e2e4")
    stats, pw, pb = llm_chess.run_simple(log_dir=None, llm_config_white=cfg_w, llm_config_black=cfg_b)
    assert stats["number_of_moves"] == 2
    assert stats["reason"] == "Max moves reached"
    assert stats["prompt_type"] == "simple"
    player = pw if color == "white" else pb
    assert player.client.actual_usage_summary["gpt-6-astra"]["prompt_tokens"] == 31


def test_reject_custom_codex_config(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CHESS_CODEX_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text('model_provider="other"')
    with pytest.raises(ValueError, match="dedicated"):
        codex_env()


def test_launcher_pins_matchup_and_keeps_errors_out_of_aggregate(fake_codex, monkeypatch, tmp_path):
    import llm_chess
    import run_oauth_maia
    for side in ("W", "B"):
        for prefix in ("MODEL_KIND", "OPENAI_MODEL_NAME"):
            key = f"{prefix}_{side}"
            monkeypatch.setenv(key, os.environ.get(key, ""))
    for name in ("white_player_type", "black_player_type", "maia_path", "maia_model", "maia_elo",
                 "maia_server", "maia_temperature", "maia_top_p", "maia_use_uci_history",
                 "reset_maia_history", "max_game_moves"):
        monkeypatch.setattr(llm_chess, name, getattr(llm_chess, name))
    monkeypatch.setattr(sys, "argv", ["run_oauth_maia.py", "--maia-path", sys.executable,
                                     "--max-plies", "2", "--logs-root", str(tmp_path / "logs")])
    monkeypatch.setattr(llm_chess, "visualize_board", False)
    monkeypatch.setattr(llm_chess.ChessEngineMaiaAgent, "generate_reply",
                        lambda self, **kw: "make_move " + ("e2e4" if self.board.turn else "e7e5"))
    run_oauth_maia.main()
    files = list((tmp_path / "logs").rglob("_aggregate_results.json"))
    assert len(files) == 2
    for file in files:
        aggregate = json.loads(file.read_text())
        assert aggregate["draws"] == 1
        metadata = json.loads((file.parent / "run_config.json").read_text())
        assert (metadata["model"], metadata["reasoning_effort"], metadata["maia_elo"], metadata["notation"]) == ("gpt-6-astra", "max", 2400, "uci")
    monkeypatch.setenv("OAUTH_TEST_MODE", "failed")
    with pytest.raises(RuntimeError, match="not counted as a draw"):
        run_oauth_maia.main()
    assert len(list((tmp_path / "logs").rglob("_aggregate_results.json"))) == 2
