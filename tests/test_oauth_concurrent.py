"""Offline regression tests; never start a real model request or chess engine."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import openai_oauth
import run_maia_concurrent as runner


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_terminal_without_codex_path_finds_editor_binary(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_CHESS_CODEX_BIN", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(openai_oauth.platform, "system", lambda: "Linux")
    monkeypatch.setattr(openai_oauth.platform, "machine", lambda: "x86_64")
    binary = executable(tmp_path / ".vscode-server/extensions/openai.chatgpt-26.9/bin/linux-x86_64/codex")
    assert openai_oauth.codex_command()[0] == str(binary)


@pytest.mark.parametrize("override", ["./codex", "./bin/codex"])
def test_relative_override_is_resolved_before_subprocess_cwd_change(tmp_path, monkeypatch, override):
    monkeypatch.chdir(tmp_path)
    binary = executable(tmp_path / override)
    monkeypatch.setenv("LLM_CHESS_CODEX_BIN", override)
    assert openai_oauth.resolve_codex_binary() == str(binary)


def test_bad_explicit_override_does_not_silently_fallback(tmp_path, monkeypatch):
    executable(tmp_path / "codex")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("LLM_CHESS_CODEX_BIN", "/missing/codex")
    with pytest.raises(FileNotFoundError, match="LLM_CHESS_CODEX_BIN is not executable"):
        openai_oauth.resolve_codex_binary()


def test_no_installation_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_CHESS_CODEX_BIN", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="Install Codex CLI"):
        openai_oauth.resolve_codex_binary()


@pytest.fixture
def oauth_config():
    return {"config_list": [{"model": "gpt-6-astra", "model_client_cls": "OpenAIOAuthClient",
                             "oauth_reasoning_effort": "max"}]}


def test_preflight_pins_binary_and_checks_each_unique_model(oauth_config, monkeypatch):
    monkeypatch.setenv("LLM_CHESS_CODEX_BIN", "original")
    monkeypatch.setattr(openai_oauth, "resolve_codex_binary", lambda: "/absolute/codex")
    server = Mock()
    server.__enter__ = Mock(return_value=server)
    server.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(openai_oauth, "AppServer", Mock(return_value=server))
    runner._preflight_oauth((oauth_config, oauth_config), "simple")
    assert openai_oauth.os.environ["LLM_CHESS_CODEX_BIN"] == "/absolute/codex"
    server.require_chatgpt.assert_called_once()
    server.check_model.assert_called_once_with("gpt-6-astra", "max")
    server.__exit__.assert_called_once()


@pytest.mark.parametrize("failure", [FileNotFoundError("codex missing"), RuntimeError("login required")])
def test_preflight_failure_starts_no_servers_or_workers(oauth_config, monkeypatch, failure):
    monkeypatch.setattr(runner, "get_llms", lambda **kw: (oauth_config, oauth_config))
    monkeypatch.setattr(runner, "_hyperparams", lambda args: None)
    monkeypatch.setattr(runner, "_preflight_oauth", Mock(side_effect=failure))
    servers, workers = Mock(), Mock()
    monkeypatch.setattr(runner, "_start_servers", servers)
    monkeypatch.setattr(runner, "_run_forked", workers)
    with pytest.raises(type(failure), match=str(failure)):
        runner._launch(SimpleNamespace(colors="both", prompt="simple"))
    servers.assert_not_called()
    workers.assert_not_called()


def test_worker_error_is_not_aggregated_as_draw(tmp_path, monkeypatch):
    for name in ("maia_server", "maia_elo", "white_player_type", "black_player_type",
                 "remove_text", "max_api_retries", "api_retry_delay"):
        monkeypatch.setattr(runner.llm_chess, name, getattr(runner.llm_chess, name))
    monkeypatch.setattr(runner, "_hyperparams", lambda args: None)
    monkeypatch.setattr(runner, "get_llms", lambda **kw: ({}, {}))
    monkeypatch.setattr(runner.llm_chess, "run_simple", Mock(return_value=(
        {"reason": runner.llm_chess.TerminationReason.ERROR.value}, None, None)))
    args = SimpleNamespace(socket="unused", elo=2400, color="white", out=str(tmp_path), prompt="simple")
    with pytest.raises(RuntimeError, match="not counted as a draw"):
        runner._worker(args)
    assert not (tmp_path / "_aggregate_results.json").exists()


@pytest.mark.parametrize("status", [0, 256])
def test_fork_worker_exit_status_is_propagated(monkeypatch, tmp_path, status):
    monkeypatch.setattr(runner.os, "fork", lambda: 123)
    monkeypatch.setattr(runner.os, "waitpid", lambda *args: (123, status))
    monkeypatch.setattr("gc.freeze", lambda: None)
    args = SimpleNamespace(concurrency=1)
    call = lambda: runner._run_forked(args, [(2400, "white", 0)], {2400: ("unused", None)},
                                    lambda *args: str(tmp_path))
    if status:
        with pytest.raises(RuntimeError, match="1/1 games failed"):
            call()
    else:
        call()


@pytest.mark.parametrize("rc", [0, 1])
def test_spawn_preserves_effort_and_propagates_failure(tmp_path, monkeypatch, rc):
    monkeypatch.setattr(runner, "get_llms", lambda **kw: ({"config_list": [{"model": "gpt-6-astra"}]}, {}))
    monkeypatch.setattr(runner, "_hyperparams", lambda args: None)
    monkeypatch.setattr(runner, "_start_servers", lambda *args: {2400: ("unused", None)})
    stopped = Mock()
    monkeypatch.setattr(runner, "_stop_servers", stopped)
    subprocess = Mock(return_value=SimpleNamespace(returncode=rc))
    monkeypatch.setattr(runner.subprocess, "run", subprocess)
    args = SimpleNamespace(colors="white", prompt="simple", reasoning_effort="max", elos=[2400],
                           reps=1, spawn=True, concurrency=1, logs_root=str(tmp_path), sf_order=False,
                           thinking=False, thinking_style="reasoning", service_tier=None,
                           llm_temperature=None, provider=None, quant=None)
    if rc:
        with pytest.raises(RuntimeError, match="1/1 games failed"):
            runner._launch(args)
    else:
        runner._launch(args)
    command = subprocess.call_args.args[0]
    assert command[command.index("--reasoning-effort") + 1] == "max"
    assert command[command.index("--thinking-style") + 1] == "reasoning"
    stopped.assert_called_once()
