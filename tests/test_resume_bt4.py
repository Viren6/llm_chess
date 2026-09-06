import io
import json
from unittest.mock import Mock

import chess
import chess.pgn
import pytest

import resume_bt4 as recovery
from tests.test_openai_oauth import fake_codex


def record_for(moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return {"reason": "ERROR OCCURED", "prompt_type": "simple",
            "number_of_moves": len(moves), "pgn": str(chess.pgn.Game.from_board(board)),
            "player_white": {"model": "gpt-6-astra"}, "player_black": {"model": "gpt-6-astra"}}


def test_history_includes_repetition_castling_and_en_passant():
    moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2 + ['e2e4']
    board = recovery.checkpoint_board(record_for(moves))
    assert [m.uci() for m in board.move_stack] == moves
    assert board.ep_square == chess.E3
    assert board.has_kingside_castling_rights(chess.WHITE)
    board.pop()
    assert board.is_repetition(3)


@pytest.mark.parametrize('change', [
    {'reason': 'Checkmate'}, {'number_of_moves': 2}, {'pgn': ''},
    {'prompt_type': 'standard'},
])
def test_reject_bad_checkpoint(change):
    record = record_for(['e2e4'])
    record.update(change)
    with pytest.raises(ValueError):
        recovery.checkpoint_board(record)


def test_reject_terminal():
    with pytest.raises(ValueError, match='terminal'):
        recovery.checkpoint_board(record_for(['f2f3', 'e7e5', 'g2g4', 'd8h4']))


@pytest.mark.parametrize('moves,color,expected', [
    (['e2e4'], 'black', ['e2e4', 'e7e5']),
    (['g1f3', 'g8f6'], 'white', ['g1f3', 'g8f6', 'e2e4']),
])
def test_resume_turn_full_pgn_and_original_move_limit(fake_codex, monkeypatch, moves, color, expected):
    import llm_chess
    import maia_server
    from utils import get_llms
    for side in ('W', 'B'):
        monkeypatch.setenv(f'MODEL_KIND_{side}', 'openai_oauth')
        monkeypatch.setenv(f'OPENAI_MODEL_NAME_{side}', 'gpt-6-astra')
    white, black = get_llms({'reasoning_effort': 'max'}, {'reasoning_effort': 'max'})
    monkeypatch.setattr(llm_chess, 'white_player_type', llm_chess.PlayerType.LLM_WHITE if color == 'white' else llm_chess.PlayerType.CHESS_ENGINE_BT4_POLICY)
    monkeypatch.setattr(llm_chess, 'black_player_type', llm_chess.PlayerType.LLM_BLACK if color == 'black' else llm_chess.PlayerType.CHESS_ENGINE_BT4_POLICY)
    monkeypatch.setattr(llm_chess, 'bt4_server', '/never-connect')
    monkeypatch.setattr(llm_chess, 'max_game_moves', len(expected))
    monkeypatch.setattr(llm_chess, 'visualize_board', False)
    request = Mock(side_effect=AssertionError('must resume with LLM, not replay engine move'))
    monkeypatch.setattr(maia_server, 'request_move', request)
    stats, _, _ = llm_chess.run_simple(log_dir=None, llm_config_white=white,
        llm_config_black=black, resume_record=record_for(moves))
    assert stats['reason'] == 'Max moves reached'
    assert stats['number_of_moves'] == len(expected)
    game = chess.pgn.read_game(io.StringIO(stats['pgn']))
    assert [m.uci() for m in game.mainline_moves()] == expected
    assert stats['resume']['starting_ply'] == len(moves)
    request.assert_not_called()
    calls = [json.loads(line) for line in fake_codex.read_text().splitlines()]
    assert len([c for c in calls if c['method'] == 'turn/start']) == 1


def make_failed(tmp_path, name='run_white_j0'):
    folder = tmp_path / name
    folder.mkdir()
    (folder / 'game.json').write_text(json.dumps(record_for(['e2e4'])))
    (folder / 'output.txt').write_text('MADE MOVE 1: Player_White e2e4\nRuntimeError: Game failed;')
    (folder / 'run_config.json').write_text(json.dumps({
        'model': 'gpt-6-astra', 'reasoning_effort': 'max', 'prompt_type': 'simple',
        'llm_config': {'auth': 'chatgpt_oauth'}, 'engine': {'opponent': 'bt4-policy'},
        'llm_color': 'black'}))
    return folder


def test_discover_skips_active_and_completed_and_checks_transcript(tmp_path):
    failed = make_failed(tmp_path)
    active = make_failed(tmp_path, 'run_white_j1')
    (active / 'output.txt').write_text('working')
    complete = make_failed(tmp_path, 'run_white_j2')
    record = record_for(['e2e4'])
    record['reason'] = 'Max moves reached'
    (complete / 'game.json').write_text(json.dumps(record))
    (complete / 'output.txt').write_text('finished')
    assert len(recovery.discover(tmp_path, 'run')) == 1
    (failed / 'output.txt').write_text('MADE MOVE 1: Player_White d2d4\nRuntimeError: Game failed;')
    with pytest.raises(ValueError, match='disagrees'):
        recovery.discover(tmp_path, 'run')


def test_plan_never_executes_or_writes(tmp_path, monkeypatch):
    make_failed(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    execute = Mock(side_effect=AssertionError('no execution'))
    monkeypatch.setattr(recovery, 'execute', execute)
    monkeypatch.setattr('sys.argv', ['resume_bt4.py', '--logs', str(tmp_path), '--run', 'run', '--out', str(tmp_path/'out')])
    assert recovery.main() == 0
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert not (tmp_path / 'out').exists()
    execute.assert_not_called()


def test_execute_uses_private_server_and_new_logs(tmp_path, monkeypatch):
    import run_maia_concurrent as runner
    import bt4_policy_server
    source_root = tmp_path / 'original'
    source_root.mkdir()
    make_failed(source_root)
    jobs = recovery.discover(source_root, 'run')
    meta = {'lc0_path': '/fake/lc0', 'weights': '/fake/weights',
            'lc0_sha256': 'binary', 'weights_sha256': 'weights', 'nodes': 1,
            'search': 'policyhead', 'backend': 'blas'}
    jobs[0][2]['engine'] = meta
    monkeypatch.setattr(bt4_policy_server, 'engine_metadata', lambda *a: meta)
    out = tmp_path / 'recovery'
    server = {0: ('/private-test-socket', object())}
    def start(args):
        from pathlib import Path
        assert Path.cwd() == out
        return server
    monkeypatch.setattr(runner, '_start_bt4_server', start)
    stop = Mock()
    monkeypatch.setattr(runner, '_stop_servers', stop)
    commands = []
    def spawn(command, **kwargs):
        commands.append(command)
        assert kwargs['env']['MODEL_KIND_W'] == 'openai_oauth'
        assert kwargs['env']['OPENAI_MODEL_NAME_B'] == 'gpt-6-astra'
        assert command[command.index('--socket')+1] == '/private-test-socket'
        checkpoint = command[command.index('--resume-json')+1]
        assert recovery.load_checkpoint(checkpoint)['number_of_moves'] == 1
        kwargs['stdout'].write('offline worker evidence')
        return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))
    monkeypatch.setattr(recovery.subprocess, 'Popen', spawn)
    before = {str(p): p.read_bytes() for p in source_root.rglob('*') if p.is_file()}
    assert recovery.execute(jobs, out) == 0
    assert before == {str(p): p.read_bytes() for p in source_root.rglob('*') if p.is_file()}
    assert len(commands) == 1
    stop.assert_called_once_with(server)
    with pytest.raises(FileExistsError):
        recovery.execute(jobs, out)
    assert len(commands) == 1


def test_immediate_capacity_error_keeps_checkpoint(fake_codex, monkeypatch):
    import llm_chess
    from utils import get_llms, calculate_material_count
    for side in ('W', 'B'):
        monkeypatch.setenv(f'MODEL_KIND_{side}', 'openai_oauth')
        monkeypatch.setenv(f'OPENAI_MODEL_NAME_{side}', 'gpt-6-astra')
    monkeypatch.setenv('OAUTH_TEST_MODE', 'failed')
    white, black = get_llms({'reasoning_effort': 'max'}, {'reasoning_effort': 'max'})
    monkeypatch.setattr(llm_chess, 'white_player_type', llm_chess.PlayerType.CHESS_ENGINE_BT4_POLICY)
    monkeypatch.setattr(llm_chess, 'black_player_type', llm_chess.PlayerType.LLM_BLACK)
    monkeypatch.setattr(llm_chess, 'bt4_server', '/never-connect')
    monkeypatch.setattr(llm_chess, 'max_api_retries', 0)
    monkeypatch.setattr(llm_chess, 'visualize_board', False)
    record = record_for(['e2e4'])
    stats, _, _ = llm_chess.run_simple(log_dir=None, llm_config_white=white,
        llm_config_black=black, resume_record=record)
    assert stats['reason'] == 'ERROR OCCURED'
    assert recovery.checkpoint_board(stats).fen() == recovery.checkpoint_board(record).fen()
    mw, mb = calculate_material_count(recovery.checkpoint_board(record))
    assert stats['material_count'] == {'white': mw, 'black': mb}


@pytest.mark.parametrize('concurrency', [1, 3, 20])
def test_concurrent_recovery_fills_slots_and_collects_errors(tmp_path, monkeypatch, concurrency):
    import run_maia_concurrent as runner
    import bt4_policy_server
    root = tmp_path / 'source'
    root.mkdir()
    for i in range(8):
        make_failed(root, f'run_black_j{i}')
    jobs = recovery.discover(root, 'run')
    meta = {'lc0_path': '/fake/lc0', 'weights': '/fake/weights',
            'lc0_sha256': 'binary', 'weights_sha256': 'weights', 'nodes': 1,
            'search': 'policyhead', 'backend': 'blas'}
    for _, _, config in jobs:
        config['engine'] = meta
    monkeypatch.setattr(bt4_policy_server, 'engine_metadata', lambda *a: meta)
    start = Mock(return_value={0: ('/private-socket', object())})
    stop = Mock()
    monkeypatch.setattr(runner, '_start_bt4_server', start)
    monkeypatch.setattr(runner, '_stop_servers', stop)
    state = {'live': 0, 'peak': 0, 'spawned': 0}
    def spawn(command, **kwargs):
        state['live'] += 1
        state['spawned'] += 1
        state['peak'] = max(state['peak'], state['live'])
        rc = int(state['spawned'] == 2)
        def wait():
            state['live'] -= 1
            return rc
        return Mock(poll=Mock(return_value=rc), wait=wait)
    monkeypatch.setattr(recovery.subprocess, 'Popen', spawn)
    assert recovery.execute(jobs, tmp_path/'out', concurrency) == 1
    assert state == {'live': 0, 'peak': min(concurrency, 8), 'spawned': 8}
    start.assert_called_once()
    stop.assert_called_once()


def test_launch_failure_cleans_only_private_children(tmp_path, monkeypatch):
    import run_maia_concurrent as runner
    import bt4_policy_server
    from pathlib import Path
    root = tmp_path/'source'
    root.mkdir()
    for i in range(2):
        make_failed(root, f'run_black_j{i}')
    jobs = recovery.discover(root, 'run')
    meta = {'lc0_path': '/fake/lc0', 'weights': '/fake/weights',
            'lc0_sha256': 'binary', 'weights_sha256': 'weights', 'nodes': 1,
            'search': 'policyhead', 'backend': 'blas'}
    for _, _, config in jobs:
        config['engine'] = meta
    monkeypatch.setattr(bt4_policy_server, 'engine_metadata', lambda *a: meta)
    server = {0: ('/private-socket', object())}
    monkeypatch.setattr(runner, '_start_bt4_server', Mock(return_value=server))
    stop = Mock()
    monkeypatch.setattr(runner, '_stop_servers', stop)
    child = Mock(poll=Mock(return_value=None), wait=Mock(return_value=0))
    monkeypatch.setattr(recovery.subprocess, 'Popen', Mock(side_effect=[child, OSError('spawn failed')]))
    cwd = Path.cwd()
    with pytest.raises(OSError, match='spawn failed'):
        recovery.execute(jobs, tmp_path/'out', 20)
    child.terminate.assert_called_once()
    child.wait.assert_called_once_with(timeout=10)
    stop.assert_called_once_with(server)
    assert Path.cwd() == cwd


def test_reject_nonpositive_concurrency_before_launch(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        recovery.execute([], tmp_path/'out', 0)
    assert not (tmp_path/'out').exists()
