import time
import traceback
import re
import chess
from typing import Any, Dict, Tuple
from enum import Enum
from custom_agents import (
    GameAgent,
    RandomPlayerAgent,
    AutoReplyAgent,
    ChessEngineStockfishAgent,
    ChessEngineDragonAgent,
    ChessEngineMaiaAgent,
    NonGameAgent,
    build_termination_predicate,
    extract_message_text,
    is_retryable_error,
)
from utils import calculate_material_count, generate_game_stats, get_llms, display_board, display_store_game_video_and_stats
# Re-export so existing `from llm_chess import TerminationReason` callers keep working.
# The canonical definition lives in termination_reasons.py (intentionally dep-free) so
# lightweight consumers like data/get_refined_csv.py can skip llm_chess's heavy import chain.
from termination_reasons import TerminationReason  # noqa: F401

class BoardRepresentation(Enum):
    FEN_ONLY = 1
    UNICODE_ONLY = 2
    UNICODE_WITH_PGN = 3


class PlayerType(Enum):
    LLM_WHITE = 1  # Represents a white player controlled by an LLM and using *_W config keys from .env
    LLM_BLACK = 2  # Represents a black player controlled by an LLM and using *_B config keys from .env
    RANDOM_PLAYER = 3  # Represents a player making random moves
    CHESS_ENGINE_STOCKFISH = 5
    CHESS_ENGINE_DRAGON = 6  # Add this new entry for Dragon engine
    LLM_NON = 7  # Represents a mixture of agents player using multiple LLMs
    CHESS_ENGINE_MAIA = 8  # Maia 3 human-like, Elo-calibrated anchor opponent


white_player_type = PlayerType.RANDOM_PLAYER
black_player_type = PlayerType.LLM_BLACK
enable_reflection = False  # Whether to offer the LLM time to think and evaluate moves
# Whether to instruct the model to write step-by-step analysis and then its move in ONE
# message. Unlike enable_reflection (a separate 'do_reflection' action), this needs no extra
# turn and avoids the proxy matching 'do_reflection' ahead of 'make_move' and swallowing the
# move. Reasoning must end with a single 'make_move <UCI>' line (see common_prompt).
require_reasoning = False
board_representation_mode = BoardRepresentation.UNICODE_ONLY  # What kind of board is printed in response to get_current_board
rotate_board_for_white = False # Whether to rotate the Uicode board for the white player so it gets it's pieces at the bottom

# Game configuration
max_game_moves = 200  # maximum number of game moves before terminating, dafault 200
max_llm_turns = 10  # how many conversation turns can an LLM make deciding on a move, e.g. repeating valid actions many times, default 10
max_failed_attempts = 3  # number of wrong replies within a dialog (e.g. non existing action) before stopping/giving a loss, default 3
throttle_delay = 0 # some LLM providers might thorttle frequent API reuqests, make a delay (in seconds) between moves
dialog_turn_delay = 1  # adds a delay in seconds inside LLM agent, i.e. delays between turns in a dialog happenning within a move

# API Retry configuration
max_api_retries = 3  # Maximum number of retries for API errors (e.g., "Service is not available"), default 3
api_retry_delay = 2.0  # Base delay in seconds between retries (uses exponential backoff), default 2.0

random_print_board = (
    False  # if set to True the random player will also print it's board to Console
)
visualize_board = False  # You can skip board visualization (animated board in popup window) to speed up execution

# Default hyperparameters are temperature 0.3, top_p 1.0
# o1-mini fails with any temp params other than 1.0 or not present, R1 distil recomends 0.5-0.7, kimi-k1.5-preview 0.3
# For thinking mode (temperature will be removed automatically if thinking_budget is set):
default_hyperparams = {
    "temperature": 0.3,
    "top_p": 1.0,
}


reasoning_effort = None # Default is None, used with OpenAI models low, medium, or high

thinking_budget = None # Default is None, if set will enable extended thinking with Anthropic models, min 1024 for Claude 3.7

# Tell AutoReply agent to remove given pieces of text from BOTH agents history when processing replies
# (using re.sub(self.ignore_text, '', action_choice, flags=re.DOTALL))
# It is needed to remove isolating thinking tokens. E.g. Deepseek R1 32B uses <think> tags that can have actions mentioned breaking execution (r"<think>.*?</think>")
# r"<think>.*?</think>" - Deepseek R1 Distil
# r"◁think▷.*?◁/think▷ - Kimi 1.5
# r"<reasoning>.*?</reasoning>" - Reka Flash
# Default None

# Unified default: remove everything from start up to and including any known
# closing tag used by thinking/reasoning models, plus trailing whitespace.
# Covers:
# - </think>          (DeepSeek/Qwen/Phi-4 thinking tokens)
# - ◁/think▷          (Kimi 1.5 thinking tokens)
# - </reasoning>      (Reka Flash reasoning tokens)
DEFAULT_REMOVE_TEXT_REGEX: str = r"^.*?(?:</think>|◁/think▷|</reasoning>)\s*"

# Per-player regex for stripping text from message history
remove_text_white: str | None = None  # e.g. r"<think>.*?</think>"
remove_text_black: str | None = None
# Backward-compat global (applies to both sides when per-player not set)
remove_text: str | None = None

stockfish_path = "/opt/homebrew/bin/stockfish"
reset_stockfish_history = True  # If True, Stockfish will get no history before making a move, default is True
stockfish_level = 1  # Set to an integer (0-20) to override Stockfish skill level, or None to use default
stockfish_time_per_move = 0.1  # Time limit (in seconds) for Stockfish to think per move, default is 0.1

# Komodo Dragon chess engine configuration
# Download Dragon 1 from https://komodochess.com
dragon_path = "./dragon/dragon-osx"  # Path to Komodo Dragon executable
reset_dragon_history = True  # If True, Dragon will get no history before making a move
dragon_level = 1  # Skill level (1-25) for Komodo Dragon
dragon_time_per_move = 0.1  # Time limit (in seconds) for Dragon to think per move

# Maia 3 human-like chess engine configuration (Elo-calibrated anchor opponent).
# Launched as a UCI subprocess: `maia_path --model <maia_model> --elo <maia_elo> [...]`.
# We use the generic `maia3-uci` launcher (not a preset binary like maia3-79m) so behaviour
# matches maia3-uci's documented defaults. Maia's "level" is its Elo directly, so maia_elo
# doubles as the rating anchor used by data/maia_elo.py. History is kept
# (reset_maia_history=False) because the --use-uci-history flag expects the move history.
maia_path = "maia3-uci"  # generic Maia UCI launcher; resolved via PATH (or set an absolute path)
maia_model = "maia3-79m"  # model alias / HF repo passed via --model
maia_elo = 1000  # Target playing strength passed to Maia via --elo (also the anchor Elo)
maia_time_per_move = 0.2  # Time limit (in seconds) for Maia to think per move
maia_use_uci_history = True  # Pass --use-uci-history so Maia conditions on move history
reset_maia_history = False  # If True, drop move history (reconstruct board from FEN)
# Maia move-policy sampling, passed to maia3-uci via --temperature/--top-p. maia3-uci
# defaults temperature to 1.0 (samples the human-move distribution at the target Elo =>
# diverse, human-like games); 0 = argmax/deterministic (identical games).
maia_temperature = 1.0  # 1.0 = full human-policy sampling (maia3-uci default); 0 = argmax
maia_top_p = 1.0  # Nucleus sampling threshold for Maia (1.0 = disabled)
# If set to a Unix socket path, the Maia agent talks to a shared maia_server.py instead of
# spawning its own local Maia subprocess. Lets many concurrent game workers share a few GPU
# Maia instances (see run_maia_concurrent.py). None = spawn a local subprocess per agent.
maia_server = None

## Actions

board = chess.Board()
san_moves = []

def get_current_board() -> str:
    """
    Returns:
        str: A text representation of the current board state.
    """
    orientation = not (rotate_board_for_white and board.turn) # True is default orientation, False is rotated
    if board_representation_mode == BoardRepresentation.FEN_ONLY:
        return board.fen()
    elif board_representation_mode == BoardRepresentation.UNICODE_ONLY:
        return board.unicode(orientation=orientation)
    elif board_representation_mode == BoardRepresentation.UNICODE_WITH_PGN:
        pgn_header = (
            "[Event \"Chess Game\"]\n"
            f"[Date \"{time.strftime('%Y.%m.%d')}\"]\n"
            "[White \"Player White\"]\n"
            "[Black \"Player Black\"]\n"
            "[Result \"*\"]\n\n"
        )

        pgn_moves = ""
        for i, move in enumerate(san_moves):
            if i % 2 == 0:
                pgn_moves += f"{(i // 2) + 1}. {move} "
            else:
                pgn_moves += f"{move} "

            if i > 0 and i % 10 == 9:
                pgn_moves += "\n"

        return f"{board.unicode(orientation=orientation)}\n\nPGN:\n{pgn_header}{pgn_moves}"


def get_legal_moves() -> str:
    """
    Returns:
        str: A list of legal moves in UCI format, separated by commas.
    """
    if board.legal_moves.count() == 0:
        return None
    return ",".join([str(move) for move in board.legal_moves])


def make_move(move: str):
    """
    Args:
        move (str): A move in UCI format.

    """
    san_board = board.copy() # make a copy of the board in order not to spoil it with san() call

    move_obj = chess.Move.from_uci(move) # this conversation will fail if the move is invalid, proxy will bubble the error to counterpart agent
    board.push_uci(str(move_obj))

    san_move = san_board.san(move_obj)
    san_moves.append(san_move)
    
    # Visualize the board if enabled
    if visualize_board:
        display_board(board, move_obj)

def run(
    log_dir="_logs",
    llm_config_white=None,
    llm_config_black=None,
    non_llm_configs_white=None,
    non_llm_configs_black=None,
) -> Tuple[Dict[str, Any], GameAgent, GameAgent]:
    """
    Runs the chess game simulation.

    Args:
        log_dir (str): Directory to save log file with game result. Set to NONE to not create one
        llm_config_white (dict, optional): LLM config for white. If None, uses default from get_llms_autogen.
        llm_config_black (dict, optional): LLM config for black. If None, uses default from get_llms_autogen.
        non_llm_configs_white (list, optional): List of configs for non-LLM white. If None, uses default.
        non_llm_configs_black (list, optional): List of configs for non-LLM black. If None, uses default.

    Returns:
        tuple: A tuple containing game statistics, the white player, and the black player.
    """
    # Set up configs if not provided
    if llm_config_white is None or llm_config_black is None:
        WHITE_MODEL_CONFIG = {
            "hyperparams": default_hyperparams,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            **({"thinking_budget": thinking_budget} if thinking_budget else {}),
        }
        BLACK_MODEL_CONFIG = WHITE_MODEL_CONFIG.copy()
        _llm_config_white, _llm_config_black = get_llms(
            white_hyperparams=WHITE_MODEL_CONFIG,
            black_hyperparams=BLACK_MODEL_CONFIG,
        )
        if llm_config_white is None:
            llm_config_white = _llm_config_white
        if llm_config_black is None:
            llm_config_black = _llm_config_black

    if non_llm_configs_white is None:
        non_llm_configs_white = [
            {**llm_config_white, "temperature": 0.0},
            {**llm_config_white, "temperature": 1.0},
        ]
    if non_llm_configs_black is None:
        non_llm_configs_black = [
            {**llm_config_black, "temperature": 0.0},
            {**llm_config_black, "temperature": 1.0},
        ]

    time_started = time.strftime("%Y.%m.%d_%H:%M")

    # Action names
    get_current_board_action = "get_current_board"
    get_legal_moves_action = "get_legal_moves"
    make_move_action = "make_move"

    # Init chess board and game state
    material_count = {"white": 0, "black": 0}
    global board, san_moves
    board.reset()
    san_moves.clear()
    winner = None
    reason = None

    move_was_made = "Move made, switching player"
    termination_conditions = [
        move_was_made.lower(),
        TerminationReason.TOO_MANY_WRONG_ACTIONS.value.lower(),
    ]

    # Action names
    get_current_board_action = "get_current_board"
    get_legal_moves_action = "get_legal_moves"
    reflect_action = "do_reflection"
    make_move_action = "make_move"

    common_prompt = (
        "Now is your turn to make a move. Before making a move you can pick one of the following actions:\n"
        f"- '{get_current_board_action}' to get the schema and current status of the board\n"
        f"- '{get_legal_moves_action}' to get a UCI formatted list of available moves\n"
        + (
            f"- '{reflect_action}' to take a moment to think about your strategy\n"
            if enable_reflection
            else ""
        )
        + f"- '{make_move_action} <UCI formatted move>' when you are ready to complete your turn (e.g., '{make_move_action} e2e4')"
        + (
            f"\n\nReason in depth before you move — do not be brief. First use "
            f"'{get_current_board_action}' and '{get_legal_moves_action}' to see the position and "
            f"your legal options. Then send ONE message with a thorough analysis:\n"
            f"- Evaluate the position: material, king safety, pawn structure, piece activity, and "
            f"the opponent's immediate threats.\n"
            f"- Choose your 3-5 strongest candidate moves (in algebraic notation like Nf3, never "
            f"as '{make_move_action}').\n"
            f"- For EACH candidate, calculate the most forcing line several plies deep, give the "
            f"opponent's best reply, and evaluate the resulting position.\n"
            f"- Compare the candidates and explain which you choose and why.\n"
            f"Take as much space as you need. Then finish the message with your committing action "
            f"on its own final line: '{make_move_action} <UCI>'. Write '{make_move_action}' exactly "
            f"once, as the last line, and do not write the other action names in your analysis."
            if require_reasoning
            else ""
        )
    )

    reflect_prompt = (
        "Before deciding on the next move you can reflect on your current situation, write down notes and evaluate.\n"
        "Here're a few recommendations that you can follow to make a better move decision:\n"
        "- Shortlist the most valuable next moves\n"
        "- Consider how they affect the situation\n"
        "- What could be the next moves from your opponent in each case\n"
        "- Is there any strategy fitting the situation and your choice of moves\n"
        "- Rerank the shortlisted moves based on the previous steps\n"
    )

    reflection_followup_prompt = (
        "Now that you reflected please choose any of the valid actions: "
        f"{get_current_board_action}, {get_legal_moves_action}, {reflect_action}, "
        f"{make_move_action} <UCI formatted move>"
    )

    invalid_action_message = (
        f"Invalid action. Pick one, reply exactly with the name and space delimitted argument: "
        f"{get_current_board_action}, {get_legal_moves_action}"
        f"{', ' + reflect_action if enable_reflection else ''}"
        f", {make_move_action} <UCI formatted move>"
    )

    # Spent hours debuging circular loops in termination message and prompt and figuring out None is not good for system message
    is_termination_message = build_termination_predicate(termination_conditions)

    llm_white = GameAgent(
        name="Player_White",
        # Not using system message as some LLMs can ignore it
        system_message="",
        description="You are a professional chess player and you play as white. "
        + common_prompt,
        llm_config=llm_config_white,
        is_termination_msg=is_termination_message,
        human_input_mode="NEVER",
        dialog_turn_delay=dialog_turn_delay,
        max_retries=max_api_retries,
        retry_delay=api_retry_delay,
    )

    llm_black = GameAgent(
        name="Player_Black",
        system_message="",
        description="You are a professional chess player and you play as black. "
        + common_prompt,
        llm_config=llm_config_black,
        is_termination_msg=is_termination_message,
        human_input_mode="NEVER",
        dialog_turn_delay=dialog_turn_delay,
        max_retries=max_api_retries,
        retry_delay=api_retry_delay,
    )

    random_player = RandomPlayerAgent(
        name="Random_Player",
        system_message="",
        description="You are a random chess player.",
        human_input_mode="NEVER",
        is_termination_msg=is_termination_message,
        make_move_action=make_move_action,
        get_legal_moves_action=get_legal_moves_action,
        get_current_board_action=(
            get_current_board_action if random_print_board else None
        ),
    )

    # Proxy mediates between board functions and either player
    proxy_agent = AutoReplyAgent(
        name="Proxy",
        human_input_mode="NEVER",
        is_termination_msg=is_termination_message,
        max_failed_attempts=max_failed_attempts,
        get_legal_moves=get_legal_moves,
        get_current_board=get_current_board,
        make_move=make_move,
        move_was_made_message=move_was_made,
        invalid_action_message=invalid_action_message,
        too_many_failed_actions_message=TerminationReason.TOO_MANY_WRONG_ACTIONS.value,
        get_current_board_action=get_current_board_action,
        reflect_action=reflect_action,
        get_legal_moves_action=get_legal_moves_action,
        reflect_prompt=reflect_prompt,
        reflection_followup_prompt=reflection_followup_prompt,
        make_move_action=make_move_action,
        remove_text=remove_text,
    )

    player_white = {
        PlayerType.LLM_WHITE: llm_white,
        PlayerType.LLM_BLACK: llm_black,
        PlayerType.RANDOM_PLAYER: random_player,
        PlayerType.CHESS_ENGINE_STOCKFISH: ChessEngineStockfishAgent(
            name="Chess_Engine_Stockfish_White",
            board=board,
            make_move_action=make_move_action,
            stockfish_path=stockfish_path,
            remove_history=reset_stockfish_history,
            is_termination_msg=is_termination_message,
            level=stockfish_level,
            time_limit=stockfish_time_per_move,
        ),
        PlayerType.CHESS_ENGINE_DRAGON: ChessEngineDragonAgent(
            name="Chess_Engine_Dragon_White",
            board=board,
            make_move_action=make_move_action,
            dragon_path=dragon_path,
            remove_history=reset_dragon_history,
            is_termination_msg=is_termination_message,
            level=dragon_level,
            time_limit=dragon_time_per_move,
        ),
        PlayerType.CHESS_ENGINE_MAIA: ChessEngineMaiaAgent(
            name="Chess_Engine_Maia_White",
            board=board,
            make_move_action=make_move_action,
            maia_path=maia_path,
            maia_model=maia_model,
            maia_server=maia_server,
            elo=maia_elo,
            use_uci_history=maia_use_uci_history,
            remove_history=reset_maia_history,
            temperature=maia_temperature,
            top_p=maia_top_p,
            is_termination_msg=is_termination_message,
            time_limit=maia_time_per_move,
        ),
        PlayerType.LLM_NON: NonGameAgent(
            name="Player_Non_White",
            system_message="",
            description="You are a professional chess player and you play as white. " + common_prompt,
            llm_config=llm_config_white,
            llm_configs=non_llm_configs_white,
            is_termination_msg=is_termination_message,
            human_input_mode="NEVER",
            dialog_turn_delay=dialog_turn_delay,
            max_retries=max_api_retries,
            retry_delay=api_retry_delay,
        ),
    }.get(white_player_type)

    player_black = {
        PlayerType.LLM_WHITE: llm_white,
        PlayerType.LLM_BLACK: llm_black,
        PlayerType.RANDOM_PLAYER: random_player,
        PlayerType.CHESS_ENGINE_STOCKFISH: ChessEngineStockfishAgent(
            name="Chess_Engine_Stockfish_Black",
            board=board,
            make_move_action=make_move_action,
            stockfish_path=stockfish_path,
            remove_history=reset_stockfish_history,
            is_termination_msg=is_termination_message,
            level=stockfish_level,
            time_limit=stockfish_time_per_move,
        ),
        PlayerType.CHESS_ENGINE_DRAGON: ChessEngineDragonAgent(
            name="Chess_Engine_Dragon_Black",
            board=board,
            make_move_action=make_move_action,
            dragon_path=dragon_path,
            remove_history=reset_dragon_history,
            is_termination_msg=is_termination_message,
            level=dragon_level,
            time_limit=dragon_time_per_move,
        ),
        PlayerType.CHESS_ENGINE_MAIA: ChessEngineMaiaAgent(
            name="Chess_Engine_Maia_Black",
            board=board,
            make_move_action=make_move_action,
            maia_path=maia_path,
            maia_model=maia_model,
            maia_server=maia_server,
            elo=maia_elo,
            use_uci_history=maia_use_uci_history,
            remove_history=reset_maia_history,
            temperature=maia_temperature,
            top_p=maia_top_p,
            is_termination_msg=is_termination_message,
            time_limit=maia_time_per_move,
        ),
        PlayerType.LLM_NON: NonGameAgent(
            name="Player_Non_Black",
            system_message="",
            description="You are a professional chess player and you play as black. " + common_prompt,
            llm_config=llm_config_black,
            llm_configs=non_llm_configs_black,
            is_termination_msg=is_termination_message,
            human_input_mode="NEVER",
            dialog_turn_delay=dialog_turn_delay,
            max_retries=max_api_retries,
            retry_delay=api_retry_delay,
        ),
    }.get(black_player_type)

    for player in [player_white, player_black]:
        # Reflection counts are global
        player.reflections_used = 0
        player.reflections_used_before_board = 0
        player.material_count = {"white": 0, "black": 0}

    # Function to generate PGN string
    def get_pgn_string():
        # Determine result based on game state
        result = "*"  # Default for in-progress games
        if winner == "NONE":
            result = "1/2-1/2"  # Draw
        elif winner == player_white.name:
            result = "1-0"  # White wins
        elif winner == player_black.name:
            result = "0-1"  # Black wins
            
        # Create PGN header
        pgn_header = (
            "[Event \"Chess Game\"]\n"
            f"[Date \"{time.strftime('%Y.%m.%d')}\"]\n"
            "[White \"Player White\"]\n"
            "[Black \"Player Black\"]\n"
            f"[Result \"{result}\"]\n\n"
        )
        
        # Format the moves list into PGN format
        pgn_moves = ""
        for i, move in enumerate(san_moves):
            if i % 2 == 0:  # White's move - add move number
                pgn_moves += f"{(i // 2) + 1}. {move} "
            else:  # Black's move
                pgn_moves += f"{move} "
                
            # Add newline every 5 full moves for readability
            if i > 0 and i % 10 == 9:
                pgn_moves += "\n"
        
        # Add result at the end
        pgn_moves += f" {result}"
        
        return pgn_header + pgn_moves
    
    try:
        current_move = 0
        reason = None

        while current_move < max_game_moves and not reason:
            for player in [player_white, player_black]:
                # Set per-player remove_text dynamically
                if player is player_white and remove_text_white is not None:
                    proxy_agent.remove_text = remove_text_white
                elif player is player_black and remove_text_black is not None:
                    proxy_agent.remove_text = remove_text_black
                else:
                    proxy_agent.remove_text = remove_text
                # Reset player state variables before each move: has_requested_board, failed_action_attempts
                player.prep_to_move()
                chat_result = proxy_agent.initiate_chat(
                    recipient=player,
                    message=player.description,
                    max_turns=max_llm_turns,
                    cache=None
                )
                current_move += 1
                last_message = chat_result.summary

                print(f"\033[94mMADE MOVE {current_move}\033[0m")
                # print(f"\033[94mLast Message: {last_message}\033[0m")
                _, last_usage = list(
                    chat_result.cost["usage_including_cached_inference"].items()
                )[-1]

                white_material, black_material = calculate_material_count(board)
                print(
                    f"\033[94mMaterial Count - White: {white_material}, Black: {black_material}\033[0m"
                )
                material_count["white"] = white_material
                material_count["black"] = black_material

                # Get token usage stats - different for NoN agents
                if isinstance(player, NonGameAgent):
                    # NoN agents store stats directly in the agent
                    prompt_tokens = player.total_prompt_tokens
                    completion_tokens = player.total_completion_tokens
                else:
                    # Standard agents get stats from chat result
                    prompt_tokens = (
                        last_usage["prompt_tokens"] if isinstance(last_usage, dict) else 0
                    )
                    completion_tokens = (
                        last_usage["completion_tokens"]
                        if isinstance(last_usage, dict)
                        else 0
                    )
                print(f"\033[94mPrompt Tokens: {prompt_tokens}\033[0m")
                print(f"\033[94mCompletion Tokens: {completion_tokens}\033[0m")
                if (
                    last_message.lower().strip()
                    == TerminationReason.TOO_MANY_WRONG_ACTIONS.value.lower().strip()
                ):
                    winner = (
                        player_black.name
                        if player == player_white
                        else player_white.name
                    )
                    reason = TerminationReason.TOO_MANY_WRONG_ACTIONS.value
                elif board.is_game_over():
                    if board.is_checkmate():
                        winner = player_black.name if board.turn else player_white.name
                        reason = TerminationReason.CHECKMATE.value
                    elif board.is_stalemate():
                        winner = "NONE"
                        reason = TerminationReason.STALEMATE.value
                    elif board.is_insufficient_material():
                        winner = "NONE"
                        reason = TerminationReason.INSUFFICIENT_MATERIAL.value
                    elif board.is_seventyfive_moves():
                        winner = "NONE"
                        reason = TerminationReason.SEVENTYFIVE_MOVES.value
                    elif board.is_fivefold_repetition():
                        winner = "NONE"
                        reason = TerminationReason.FIVEFOLD_REPETITION.value
                elif (
                    last_message.lower().strip() != move_was_made.lower().strip()
                    # TODO: fix max llm turns for NoN, seems like missing chat history broke this check
                    and len(chat_result.chat_history) >= max_llm_turns * 2
                ):
                    winner = (
                        player_black.name
                        if player == player_white
                        else player_white.name
                    )
                    reason = TerminationReason.MAX_TURNS.value
                elif current_move >= max_game_moves:
                    winner = "NONE"
                    reason = TerminationReason.MAX_MOVES.value
                elif (
                    last_message.lower().strip() not in [
                        move_was_made.lower().strip(),
                        invalid_action_message.lower().strip()
                    ]
                    and not last_message.lower().startswith("failed to make move:")
                ):
                    winner = "NONE"
                    reason = TerminationReason.UNKNOWN_ISSUE.value

                proxy_agent.clear_history()
                time.sleep(throttle_delay)
                if reason:
                    break


    except Exception as e:
        print("\033[91mExecution was halted due to error.\033[0m")
        print(f"Exception details: {e}")
        traceback.print_exc()
        winner = "NONE"
        reason = TerminationReason.ERROR.value

    # Generate the PGN string for the complete game
    pgn_string = get_pgn_string()
    
    game_stats = generate_game_stats(
        time_started,
        winner,
        reason,
        current_move,
        player_white,
        player_black,
        material_count,
        pgn_string,
    )

    game_stats["prompt_type"] = "standard"
    display_store_game_video_and_stats(game_stats, log_dir)
    return game_stats, player_white, player_black


def run_simple(
    log_dir="_logs",
    llm_config_white=None,
    llm_config_black=None,
) -> Tuple[Dict[str, Any], GameAgent, GameAgent]:
    """Token-lean single-prompt-per-move harness (LLM vs Maia).

    Unlike run()'s multi-turn dialog (separate get_board / get_legal_moves / make_move turns
    with accumulating history), each LLM move here is ONE stateless request: a single prompt
    carrying the board + legal moves, and the model replies directly with 'make_move <uci>'.
    No conversation history is carried between moves, which cuts token usage dramatically.
    Native thinking / reasoning_effort still applies (it lives in the API config, not the
    prompt). Stats, token usage, reasoning capture and logging reuse the same machinery as
    run(); games are tagged prompt_type='simple'.
    """
    if llm_config_white is None or llm_config_black is None:
        WHITE_MODEL_CONFIG = {
            "hyperparams": default_hyperparams,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            **({"thinking_budget": thinking_budget} if thinking_budget else {}),
        }
        _w, _b = get_llms(white_hyperparams=WHITE_MODEL_CONFIG,
                          black_hyperparams=WHITE_MODEL_CONFIG.copy())
        llm_config_white = llm_config_white or _w
        llm_config_black = llm_config_black or _b

    time_started = time.strftime("%Y.%m.%d_%H:%M")
    make_move_action = "make_move"
    global board, san_moves
    board.reset()
    san_moves.clear()
    material_count = {"white": 0, "black": 0}
    winner = None
    reason = None

    is_termination_message = build_termination_predicate(
        ["move made, switching player", TerminationReason.TOO_MANY_WRONG_ACTIONS.value.lower()]
    )

    def _make_player(ptype, color):
        if ptype in (PlayerType.LLM_WHITE, PlayerType.LLM_BLACK):
            return GameAgent(
                name="Player_White" if color == "white" else "Player_Black",
                system_message="You are a precise chess engine. You always reply with a single legal move.",
                llm_config=llm_config_white if color == "white" else llm_config_black,
                is_termination_msg=is_termination_message,
                human_input_mode="NEVER", dialog_turn_delay=0,
                max_retries=max_api_retries, retry_delay=api_retry_delay,
            )
        if ptype == PlayerType.CHESS_ENGINE_MAIA:
            return ChessEngineMaiaAgent(
                name="Chess_Engine_Maia_White" if color == "white" else "Chess_Engine_Maia_Black",
                board=board, make_move_action=make_move_action,
                maia_path=maia_path, maia_model=maia_model, maia_server=maia_server,
                elo=maia_elo, use_uci_history=maia_use_uci_history, remove_history=reset_maia_history,
                temperature=maia_temperature, top_p=maia_top_p,
                is_termination_msg=is_termination_message, time_limit=maia_time_per_move,
            )
        raise ValueError(f"run_simple supports LLM-vs-Maia only; got {ptype} for {color}")

    player_white = _make_player(white_player_type, "white")
    player_black = _make_player(black_player_type, "black")
    for p in (player_white, player_black):
        p.reflections_used = 0
        p.reflections_used_before_board = 0
        p.material_count = {"white": 0, "black": 0}

    def _parse_move(text, legal):
        text = text or ""
        # prefer the move after the last explicit 'make_move'
        for cand in reversed(re.findall(r"make_move\s*[:=]?\s*([a-h][1-8][a-h][1-8][nbrq]?)",
                                        text, re.IGNORECASE)):
            if cand.lower() in legal:
                return cand.lower()
        # fall back to any legal UCI token (last one mentioned)
        for cand in reversed(re.findall(r"\b([a-h][1-8][a-h][1-8][nbrq]?)\b", text, re.IGNORECASE)):
            if cand.lower() in legal:
                return cand.lower()
        return None

    def _llm_call(player, messages):
        """One completion through the agent's client (tracks usage + reasoning + retries).
        Returns (content, prompt_tokens, completion_tokens)."""
        for attempt in range(player.max_retries + 1):
            try:
                t0 = time.time()
                player._last_reasoning = None
                resp = player.client.create(messages=messages)
                player.accumulated_reply_time_seconds += time.time() - t0
                player._emit_reasoning()
                choice = (getattr(resp, "choices", None) or [None])[0]
                content = getattr(getattr(choice, "message", None), "content", "") or ""
                usage = getattr(resp, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                return content, pt, ct
            except Exception as e:
                if attempt < player.max_retries and is_retryable_error(e):
                    print(f"\033[93mAPI error (simple) attempt {attempt+1} for {player.name}: {e}\033[0m")
                    time.sleep(player.retry_delay * (2 ** attempt))
                    continue
                raise

    def _llm_move(player, color):
        legal = set(get_legal_moves().split(","))
        note = ""
        for _ in range(max_failed_attempts + 1):
            user = (
                f"You are playing chess as {color}. Choose the strongest legal move.\n\n"
                f"Board:\n{get_current_board()}\n\n"
                f"FEN: {board.fen()}\n\n"
                f"Legal moves (UCI): {get_legal_moves()}\n\n"
                f"{note}"
                "Reply with your move as the FINAL line, exactly as: make_move <uci>  "
                "(e.g. make_move e2e4). You may reason first, but the last line must be that."
            )
            print(f"\n================ PROMPT -> {player.name} ({color}) ================")
            print(user, flush=True)
            text, pt, ct = _llm_call(player, [
                {"role": "system", "content": player.system_message},
                {"role": "user", "content": user},
            ])
            print(f"---------------- RESPONSE <- {player.name} ----------------")
            print(text)
            print(f"[tokens] prompt={pt}  completion={ct}  total={pt + ct}", flush=True)
            mv = _parse_move(text, legal)
            if mv:
                return mv
            player.wrong_moves += 1
            note = ("Your previous reply had no legal move. Pick strictly one move from the "
                    "Legal moves list and end with 'make_move <uci>'.\n")
        return None

    def _maia_move(player):
        reply = player.generate_reply(messages=[{"role": "user", "content": "Your move."}])
        text = reply if isinstance(reply, str) else extract_message_text(reply)
        m = re.search(r"([a-h][1-8][a-h][1-8][nbrq]?)", text or "", re.IGNORECASE)
        return m.group(1).lower() if m else None

    try:
        current_move = 0
        while current_move < max_game_moves and not reason:
            for player in (player_white, player_black):
                player.prep_to_move()
                if board.is_game_over() or get_legal_moves() is None:
                    break
                if isinstance(player, ChessEngineMaiaAgent):
                    mv = _maia_move(player)
                else:
                    mv = _llm_move(player, "white" if player is player_white else "black")

                if mv is None:
                    winner = player_black.name if player is player_white else player_white.name
                    reason = TerminationReason.TOO_MANY_WRONG_ACTIONS.value
                    break

                make_move(mv)
                player.make_move_count += 1
                current_move += 1
                mw, mb = calculate_material_count(board)
                material_count["white"], material_count["black"] = mw, mb
                print(f"\033[94mMADE MOVE {current_move}: {player.name} {mv}\033[0m", flush=True)

                if board.is_game_over():
                    if board.is_checkmate():
                        winner = player_black.name if board.turn else player_white.name
                        reason = TerminationReason.CHECKMATE.value
                    elif board.is_stalemate():
                        winner, reason = "NONE", TerminationReason.STALEMATE.value
                    elif board.is_insufficient_material():
                        winner, reason = "NONE", TerminationReason.INSUFFICIENT_MATERIAL.value
                    elif board.is_seventyfive_moves():
                        winner, reason = "NONE", TerminationReason.SEVENTYFIVE_MOVES.value
                    elif board.is_fivefold_repetition():
                        winner, reason = "NONE", TerminationReason.FIVEFOLD_REPETITION.value
                    else:
                        winner, reason = "NONE", TerminationReason.UNKNOWN_ISSUE.value
                elif current_move >= max_game_moves:
                    winner, reason = "NONE", TerminationReason.MAX_MOVES.value
                if reason:
                    break
    except Exception as e:
        print("\033[91mExecution was halted due to error.\033[0m")
        print(f"Exception details: {e}")
        traceback.print_exc()
        winner, reason = "NONE", TerminationReason.ERROR.value
    finally:
        for p in (player_white, player_black):
            if isinstance(p, ChessEngineMaiaAgent):
                p.close()

    # PGN (same format as run())
    result = "1/2-1/2" if winner == "NONE" else ("1-0" if winner == player_white.name else
                                                 ("0-1" if winner == player_black.name else "*"))
    pgn_header = ("[Event \"Chess Game\"]\n" f"[Date \"{time.strftime('%Y.%m.%d')}\"]\n"
                  "[White \"Player White\"]\n[Black \"Player Black\"]\n" f"[Result \"{result}\"]\n\n")
    pgn_moves = ""
    for i, mv in enumerate(san_moves):
        pgn_moves += (f"{(i // 2) + 1}. {mv} " if i % 2 == 0 else f"{mv} ")
        if i > 0 and i % 10 == 9:
            pgn_moves += "\n"
    pgn_string = pgn_header + pgn_moves + f" {result}"

    game_stats = generate_game_stats(time_started, winner, reason, current_move,
                                     player_white, player_black, material_count, pgn_string)
    game_stats["prompt_type"] = "simple"
    display_store_game_video_and_stats(game_stats, log_dir)
    return game_stats, player_white, player_black


if __name__ == "__main__":
    run()
