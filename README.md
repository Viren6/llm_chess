# LLM Chess: Benchmarking Reasoning and Instruction-Following in LLMs

[![Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-%20🏆-blueviolet)](https://maxim-saplin.github.io/llm_chess/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%20FoRLM%202025-green)](https://arxiv.org/abs/2512.01992)

LLM Chess is a benchmark that evaluates Large Language Models (LLMs) on their reasoning and instruction-following abilities in an agentic setting. LLMs engage in multi-turn dialogs to play chess against opponents like a Random Player or the Komodo Dragon chess engine. This setup tests both strategic reasoning (chess skill) and protocol adherence (sustained interaction without errors).

Key insights from the benchmark:
- Early models (2024) struggled with basic instruction following, often hallucinating illegal moves or failing dialogs.
- Advanced reasoning models (e.g., o1, o3, o4-mini) in 2025 saturated random-based evaluations, prompting the addition of Dragon as a stronger opponent for Elo anchoring.
- Metrics separate chess skill (Win/Loss, Elo) from durability (Game Duration), revealing trade-offs in model capabilities.

See the [live leaderboard](https://maxim-saplin.github.io/llm_chess/) for rankings and the [NeurIPS FoRLM 2025 paper](docs/LLM%20CHESS%2C%20Benchmarking%20Reasoning%20and%20Instruction-Following%20in%20LLMs%20through%20Chess%20-%20NeurIPS%20FoRLM%202025.pdf) for full details.

<img width="2118" height="1582" alt="image" src="https://github.com/user-attachments/assets/4375a8a8-e226-4ed1-820f-86006d0404e2" />

## Installation and Setup

1. **Clone the repository**:
   ```
   git clone https://github.com/maxim-saplin/llm_chess.git
   cd llm_chess
   ```

2. **Create a virtual environment** (recommended):
   ```
   # Using uv (recommended)
   uv sync
   ```

3. **Install dependencies**:
   ```
   # Already handled by `uv sync` above
   ```

4. **Configure LLMs**:
   - Copy `.env.sample` to `.env` and add your API keys.
   - Suffixes like `_W` (white) and `_B` (black) distinguish configs for multi-LLM setups.
   - Supports Azure OpenAI chat completions (`MODEL_KIND=azure`), Azure OpenAI Responses API (`MODEL_KIND=azure_responses`), OpenAI, Anthropic, Google, Groq, and local models via Autogen.
   - For local models, ensure Ollama or LM Studio is running.

5. **Chess Engines** (optional, for stronger opponents):
   - **Komodo Dragon**: Download binaries from [komodochess.com](https://komodochess.com/installation.htm) and place in `dragon/`. Set `llm_chess.dragon_path`.
   - **Stockfish**: Install via `brew install stockfish` (macOS) or equivalent. Set `llm_chess.stockfish_path` (default: `/opt/homebrew/bin/stockfish`).
   - **Maia 3** (human-like, Elo-calibrated anchor — used by `CHESS_ENGINE_MAIA`): a rating-conditioned neural engine from [CSSLab/maia3](https://github.com/CSSLab/maia3). Unlike Dragon/Stockfish it plays *human-like* moves at a target Elo, which makes it a natural fixed anchor for rating LLMs (see the [Maia anchor sweep](#maia-anchor-sweep-elo)). It runs as a **UCI subprocess**, so it can live in its own Python environment — it pulls in PyTorch and does **not** need to be installed into the `llm_chess` (uv) environment.
     1. **Install** the package (depends on PyTorch, so use an environment with a compatible Python, 3.10–3.12):
        ```
        git clone https://github.com/CSSLab/maia3.git
        cd maia3
        python -m pip install .          # or: python -m pip install -e .
        ```
        This creates the UCI entry-point commands `maia3-79m`, `maia3-23m`, `maia3-5m`, `maia3-uci`, and `maia3-cache`. Weights **auto-download from Hugging Face** (`UofTCSSLab/Maia3-79M`) on first run and cache under `~/.cache/huggingface`. Optionally pre-fetch them: `maia3-cache --model maia3-79m`.
     2. **Verify** it launches as a UCI engine (loads the net, prints `Maia3 ready`, then waits for UCI input — `Ctrl+C` to exit):
        ```
        maia3-79m --elo 1000 --use-uci-history
        ```
        A GPU is used automatically when available; for CPU-only add `--device cpu --no-use-amp`.
     3. **Point the harness at it**: the default `llm_chess.maia_path` is `maia3-79m`, resolved via your `PATH` — so a fresh clone works out of the box when `maia3` is installed in the same environment that runs the harness (or is otherwise on `PATH`). If Maia lives in a *separate* environment, set `llm_chess.maia_path` to its absolute path instead (find it with `which maia3-79m`). Strength is set per run via `llm_chess.maia_elo` (Maia's "level" *is* its Elo); history is passed through `--use-uci-history`, so leave `reset_maia_history = False`.

## Running Games

### Single Game
Run a single chess simulation:
```
uv run llm-chess
```
- Default: Random Player (white) vs. LLM (black).
- Logs saved to `_logs/` with JSON details and optional video recordings.

### Multiple Games
For benchmarking, run multiple simulations:
```
uv run python run_multiple_games.py
```
- Default: 42 games.
- Customize in the script:
  - `NUM_REPETITIONS`: Number of games (e.g., 30+ for reliable stats).
  - `LOG_FOLDER`: Output directory (e.g., `_logs/random_vs_llm/`).
  - `STORE_INDIVIDUAL_LOGS`: Set to `False` for aggregate JSON only.
- Aggregates results in `aggregate_results.json` and individual logs in `{timestamp}.json`.

### Maia anchor sweep (Elo)
Play the LLM against the **Maia 3 anchor ladder** with **balanced colors** and estimate a single anchored Elo. Requires Maia set up (Installation §5) with `llm_chess.maia_path` pointing at `maia3-79m`. Configure BOTH `.env` sides (`_W` and `_B`) to your model — the LLM plays each color.
```
uv run python run_maia_anchors.py --reps 25      # 25 games/color/anchor across Elo 600..1400
uv run python data/maia_elo.py                   # fit color-balanced Elo, write data/maia_elo.csv
```
- `--reps N` plays N games as White and N as Black per anchor; the default plays both colors so White's first-move advantage is **balanced out, not corrected for**.
- Per-anchor runs land in `_logs/engine_vs_llm/maia-elo-<N>/<llm>/<ts>_<color>/` (re-running accumulates). Each folder holds `output.txt` (the full console transcript — every model turn, ANSI-stripped), the per-game stats/PGN JSON, and `_aggregate_results.json`. `data/maia_elo.py` fits Elo with the same MLE as Dragon (Bradley-Terry + 95% CI) using each Maia Elo directly as the anchor, but with **colors forced balanced**: each anchor's score is the equal-weight mean of the White-side and Black-side scores (no white-advantage term), and any single-color anchor is skipped. Per-anchor scores are printed so you can see which rungs sit in the informative 35–65% band.

## Game Rules

- **Players**: Random (white) vs. LLM (black) by default. Supports LLM vs. LLM, engine vs. LLM.
- **Constraints**:
  - Max 200 moves (100 per player).
  - Max 10 turns per LLM move (user/assistant pairs).
  - Max 3 mistakes per dialog (illegal moves/actions); exceeds → LLM loss.
- **Outcomes**:
  - **Win/Loss**: Checkmate or opponent errors/timeouts.
  - **Draw**: Max moves reached, stalemate, insufficient material, repetition, or 75-move rule.
  - **Errors**: Programmatic issues → Draw (manual review for API throttles/model failures → discard or LLM loss).
- Games use UCI notation for moves and Unicode boards for visualization.

## Configurations

Edit globals in `llm_chess.py` or pass via `run_multiple_games.py`:

- Provider choice comes from `MODEL_KIND_W` / `MODEL_KIND_B` in `.env`.
- Use `azure` for classic Azure chat-completions deployments.
- Use `azure_responses` for Azure deployments that require the Responses API. Keep `AZURE_OPENAI_ENDPOINT_*` at the resource root such as `https://your-resource.openai.azure.com`; the runtime will normalize it to the Responses base path automatically.

- `white_player_type` / `black_player_type`: `RANDOM_PLAYER`, `LLM`, `CHESS_ENGINE_DRAGON`, `CHESS_ENGINE_STOCKFISH`, `CHESS_ENGINE_MAIA`.
- Maia options: `maia_path` (path to the `maia3-79m` command), `maia_elo` (target strength / Elo anchor), `maia_time_per_move`, `maia_use_uci_history`, `reset_maia_history`.
- `enable_reflection`: Enable "reflect" action for strategic thinking (extra tokens).
- `use_fen_board`: Use FEN notation instead of Unicode board (default: False).
- `max_game_moves`: Max moves (default: 200).
- Per-move LLM limits:
  - `max_llm_turns`: Max dialog turns (default: 10).
  - `max_failed_attempts`: Max errors before loss (default: 3).
- `throttle_delay_moves`: API delay (default: 1s) to avoid rate limits.

## Agents

- **LLM Agent**: Autogen `ConversableAgent` for dialog-based moves. Prompts guide actions: `get_current_board`, `get_legal_moves`, `make_move <UCI>`.
- **Random Agent**: Custom; requests legal moves, picks randomly. Always white.
- **Proxy Agent**: Custom `AutoReplyAgent`; orchestrates dialogs, provides board/moves.
- **Chess Engines**:
  - **Dragon**: Elo-rated. Binaries in `dragon/`.
    - Level 1: 250 Elo
    - Level 2: 375 Elo
    - Level 3: 500 Elo
    - Level 4: 625 Elo
    - Level 5: 750 Elo
    - Level 6: 875 Elo
    - Level 7: 1000 Elo
    - Formula: Elo = 125 × (level + 1)
    - Practical rule of thumb for stronger models:
      - Use completed games at the strongest Dragon level already tested and compute `S = (wins + 0.5 * draws) / N`.
      - If `35% <= S <= 65%`, stay at that level; this is the informative range for Elo estimation.
      - If `S > 65%` across roughly 15 to 20 clean games, test a higher Dragon level.
      - If `65% <= S < 80%`, move up 1 level. If `80% <= S < 90%`, move up 2 levels. If `S >= 90%`, move up 3 levels.
      - If the model is at `100%` wins on its strongest tested level, treat the current Elo as under-resolved and keep raising Dragon until the strongest-level score drops back near `35%` to `65%`.
  - **Stockfish**: Strong engine; install separately.
  - **Maia 3**: Human-like, rating-conditioned engine ([CSSLab/maia3](https://github.com/CSSLab/maia3)); its target Elo *is* its strength (no level→Elo formula). Runs as a UCI subprocess via `maia3-79m --elo <N> --use-uci-history`. Setup in Installation §5; anchored-Elo workflow in [Maia anchor sweep](#maia-anchor-sweep-elo). The same informative-band (35–65%) guidance as Dragon applies when choosing anchors.

## Processing Logs

Logs in `_logs/` contain JSON per game. Aggregate and refine:

   ```
   uv run get-refined-csv
   ```
   - Handles multiple directories (Random vs. LLM, Dragon vs. LLM).
   - Computes Elo (anchored to Dragon levels: Elo ≈ 125 × (level + 1)), Win/Loss %, Game Duration %.
   - Filters low-sample models; supports overrides/aliases.
   - Output: CSV with player stats, usage (tokens/cost), interruptions.

For games against the **Maia 3** anchor ladder, compute Elo with the dedicated estimator instead (it uses each Maia Elo directly as the anchor):
   ```
   uv run python data/maia_elo.py
   ```
   - Scans `_logs/engine_vs_llm/maia-elo-<N>/...`, fits one Elo per LLM (Bradley-Terry MLE + 95% CI), writes `data/maia_elo.csv`.



Manual review: Check logs for API errors (discard) vs. model failures (LLM loss).

## Tests

Run the test suite (parallel by default):

```
uv run pytest -q -n auto tests
```

## Metrics

From refined CSV/leaderboard:

- **Elo**: Estimated rating (±95% CI), anchored to Dragon/chess.com. Combines Random/Dragon data.
- **Win/Loss**: (Wins - Losses) / Total % (0-100%). Blends skill + instruction following. 50% = balanced.
- **Game Duration**: % of max moves completed (0-100%). Measures dialog stability (100% = no interruptions).
- **Tokens**: Completions per move. Indicates verbosity/efficiency.
- **Other**: Mistakes/1000 moves, cost/game, material diff, interruptions.

Primary sort: Elo (DESC), then Win/Loss (DESC), Duration (DESC), Tokens (ASC). Dragon-tested models marked with *.

Matrix View (in leaderboard): Win Rate (skill) vs. Duration (following) for 2D clustering.

## Sample Dialog (Single Move)

Example with GPT-4o Mini (Black):

```
Proxy (to Player_Black):
You are a professional chess player and you play as black. Now is your turn to make a move. Before making a move you can pick one of the following actions:
- 'get_current_board' to get the schema and current status of the board
- 'get_legal_moves' to get a UCI formatted list of available moves
- 'make_move <UCI formatted move>' when you are ready to complete your turn (e.g., 'make_move e2e4')
Respond with the action.

Player_Black (to Proxy):
get_current_board

Proxy (to Player_Black):
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
· · · · · · · ·
· · · · · · · ·
· · · · · · · ·
· · · · · · · ·
♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖


Player_Black (to Proxy):
get_legal_moves

Proxy (to Player_Black):
a7a6, b7b6, c7c6, d7d6, e7e6, f7f6, g7g6, h7h6, a7a5, b7b5, c7c5, d7d5, e7e5, f7f5, g7g5, h7h5

Player_Black (to Proxy):
make_move e7e5

Proxy (to Player_Black):
Move made, switching player
```

## References

- **Libraries**: `chess` (board rules), Autogen (agents/dialogs), Stockfish/Dragon (engines).
- **Data**: Logs in `_logs/`; analysis in `data/`.
- **Notes/Changelog**: [docs/notes.md](docs/notes.md) for updates, model tiers, and insights.
- **License**: MIT (see LICENSE).
- **Contribute**: Fork, PR improvements to setup, agents, or analysis.

For issues or questions, open a GitHub issue.
