# Astra max versus BT4 policy: final match Elo difference

All 40 games finished: 20 per Astra color. Simple UCI harness, ChatGPT subscription OAuth, BT4-tf13tune on LC0 CPU policyhead with one node. Recovery results replace their original failed attempts.

| Player | Games | Wins | Losses | Draws | Score | Relative Elo |
|---|---:|---:|---:|---:|---:|---:|
| GPT-6 Astra (max) | 40 | 7 | 25 | 8 | 27.50% | -168.40 |
| BT4 policy (CPU, 1 node) | 40 | 25 | 7 | 8 | 72.50% | +0.00 |

BT4 is set to zero only as a reference; this is not an absolute Elo rating. The match-implied difference is `400 * log10(score / (1 - score))`, with draws worth half a point. Equal numbers of games per color make the pooled score color-balanced, consistent with `data.maia_elo.fit_elo`.

| Astra color | Wins | Losses | Draws |
|---|---:|---:|---:|
| White | 4 | 14 | 2 |
| Black | 3 | 11 | 6 |

User adjudication: Black j11 reached the 200-ply cap and was scored as a White/BT4 win instead of its original draw. This table includes that override.

Validation: 40 unique game records, 20 per color, PGN ply counts and results agree with game JSON and per-game aggregates. All eight recovered games counted once.

## Source game records

- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j0/2026.09.06_08:03.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j1/2026.09.06_08:03.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j10/2026.09.06_11:34.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j11/2026.09.06_11:45.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_black_j12/2026.09.06_14:32.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_black_j13/2026.09.06_14:32.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_black_j14/2026.09.06_14:32.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j15/2026.09.06_12:04.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_black_j16/2026.09.06_14:32.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j17/2026.09.06_12:04.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j18/2026.09.06_12:04.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j19/2026.09.06_12:04.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j2/2026.09.06_10:34.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j3/2026.09.06_10:55.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j4/2026.09.06_10:55.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_black_j5/2026.09.06_14:32.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j6/2026.09.06_10:56.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_black_j7/2026.09.06_14:32.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j8/2026.09.06_11:21.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_black_j9/2026.09.06_11:26.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j0/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j1/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j10/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j11/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j12/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j13/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j14/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j15/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j16/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j17/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j18/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j19/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j2/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j3/2026.09.06_06:02.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_white_j4/2026.09.06_14:32.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j5/2026.09.06_06:02.json`
- `_logs/bt4-recovery/2026-09-06-06-02-23-p1225456_white_j6/2026.09.06_14:32.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j7/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j8/2026.09.06_06:02.json`
- `_logs/oauth-simple/engine_vs_llm/bt4-policy/gpt-6-astra-max/2026-09-06-06-02-23-p1225456_white_j9/2026.09.06_06:02.json`
