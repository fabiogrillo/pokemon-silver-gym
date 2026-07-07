# RL-3: Visited-Coordinates Observation (agent_090, cold) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attempt RL-3 per `docs/superpowers/specs/2026-07-06-final-attempt-findings.md` §2-R3/§4: re-add the visited-coordinates observation the 10e ablation removed while it was drawn TRANSPOSED (the ram x/y swap), as a proper Dict-obs key routed through the un-swap convention — cold `agent_090` run with the full structural stack.

**Architecture:** New gated env feature (`visited_obs`): a 48×48 uint8 binary crop of THIS EPISODE's visited tiles, centered on the player, in TRUE axes (un-swap at the boundary, cite `agents/rl/map_layout.ram_to_image_px`), exposed as a new `visited` key in the Dict observation space (separate from the RGB image, so battle/menu screens don't corrupt it — arXiv:2502.19920 §II-C). SB3's MultiInputPolicy picks it up automatically as an extra CNN/flatten branch. Cold start required (obs-space change) — that is the point of RL-3.

**Tech Stack:** as RL-1/2.

## Global Constraints

- ALL code/comments/commit messages in English.
- env/ change ADDITIVE + GATED (`visited_obs=False` default): with the flag OFF the observation space is BIT-IDENTICAL to today (old checkpoints keep loading; add a test for this).
- Visited tiles are already tracked (`self.visited_tiles` in `pokemon_env_cnn.py`) — reuse; the crop is per-episode, keyed by (map, x, y); tiles on OTHER maps than the current one are simply not in the crop (same-map filter).
- De-transposition is THE feature: the crop must be `crop[row=y][col=x]` with TRUE x/y (RAM local_x holds wYCoord — un-swap when indexing). Include a regression test that FAILS if axes are swapped (walk east via monkeypatched RAM: the mark must move along the crop's COLUMN axis).
- An Ollama eval run may be live (GPU + CPU light emulator tests fine); never broad-pkill. DO NOT LAUNCH the real run — controller launches when the GPU frees (same protocol as RL-2).
- Config for agent_090 (Task 2): COLD (no warm start), TOTAL 100M, kill gate `nav/reach_cherrygrove < 0.5 at 30M → stop`; stack: CONFINE_TO_CORRIDOR=True, waypoint frontier ON, EXPLORATION_SCALE=4.0, VISITED_OBS=True, and R2 SOFTENED per the agent_089 post-mortem (training_log.md): DYNAMIC_EPISODE_BUDGET base 32768 (not 16384; add `DYN_BUDGET_BASE` to config as an overridable, env reads it), curriculum split 9× egg_delivered_clean + 1× crossing + 1× route31 + 1× violet_city (12 envs; pure-start majority restored).
- Tests command: `.venv/bin/python -m pytest tests/test_visited_obs.py tests/test_confinement.py tests/test_dynamic_budget.py tests/test_frontier_score.py -q` (never the whole tests/ root — collection hangs on interactive scripts).

---

### Task 1: visited-coords Dict-obs key (gated env feature)

**Files:**
- Modify: `env/pokemon_env_cnn.py` (obs space + `_visited_crop()` + step/reset wiring), `agents/rl/config.py` (`VISITED_OBS=False`, `DYN_BUDGET_BASE=16384` default preserved), `agents/rl/train_cnn.py` (wire both like the other flags)
- Test: `tests/test_visited_obs.py`

**Interfaces:**
- `PokemonEnvCNN(..., visited_obs: bool = False, dyn_budget_base: int = 16384)`; when visited_obs: observation dict gains `"visited": spaces.Box(0, 1, (48, 48), np.uint8)`; the crop marks episode-visited tiles of the CURRENT map within ±24 tiles of the player (player at center (24,24)); un-swap when reading RAM coords.
- Tests (monkeypatched-RAM pattern from tests/test_confinement.py): (a) flag off → obs keys unchanged (exact set equality with today's); (b) flag on → `visited` present, uint8, (48,48), center marked after reset+step; (c) THE de-transposition regression: fake RAM walks true-EAST 3 tiles (increment `local_y`, the wXCoord holder) → the marks appear along increasing crop COLUMNS at the center row (assert specific cells; this test MUST fail if someone indexes `crop[x][y]`); (d) dyn_budget_base override honored when dynamic_episode_budget=True.

- [ ] Steps: failing tests → implement → 4-suite command green → commit `feat(rl): gated visited-coordinates observation, de-transposed (R3)`.

---

### Task 2: agent_090 config + smoke (NO launch)

**Files:** `agents/rl/config.py`, `training_log.md`

Same protocol as RL-2's Task 2 (smoke: boot ~3 min, checkpoint-free cold init, 12 workers, TB dir, kill by exact PID; NO real launch — controller's step). Config per Global Constraints above; header documents: RL-3 hypothesis (the paper's indispensable input was ablated while transposed — 10e verdict was a bug artifact), the R2 softening rationale (agent_089 post-mortem), kill gate. training_log entry "Agent 090 — CORRIDOR FINAL ATTEMPT RL-3 (R3 + R1 + softened R2 + R4) — LAUNCH PENDING".

- [ ] Steps: config → smoke → log entry → commit `feat(rl): agent_090 config - visited obs cold run (RL-3, launch pending)`.

---

## Self-Review

R3 de-transposition is regression-tested by construction (test c). Old-checkpoint compatibility guarded by test (a) + flag default. R2 softening is a documented, evidence-based deviation from the findings table (agent_089 regression post-mortem), not a silent change. Launch gating mirrors RL-2's successful handoff. No placeholders.
