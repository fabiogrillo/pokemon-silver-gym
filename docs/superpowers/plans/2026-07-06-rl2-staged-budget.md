# RL-2: Staged Resets + Event-Scaled Episode Budget (agent_089) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attempt RL-2 per `docs/superpowers/specs/2026-07-06-final-attempt-findings.md` §4: keep RL-1's stack (CONFINE_TO_CORRIDOR + waypoint frontier), add technique R2 — curriculum env slots resetting from on-corridor intermediate saves, and a dynamic per-episode step budget (16,384 base + 16,384 per waypoint reached, capped at 65,536) — as `agent_089`, warm from `runs/checkpoints/agent_088/agent_088_39999936_steps.zip`.

**Architecture:** R2(a) is config-only — `CURRICULUM_STATES_CNN` already maps (state_path, n_envs) per worker slot. R2(b) hooks the existing per-episode truncation (`self._max_steps`, checked in `step()` ~line 305 of `env/pokemon_env_cnn.py`): a gated `dynamic_episode_budget` flag makes the cap grow when the episode reaches a NEW waypoint ordinal — long episodes must be EARNED (the Pokémon-Red paper's budget trick, arXiv:2502.19920).

**Tech Stack:** as RL-1 (SB3 PPO, PyBoy, pytest, RTX 5080).

## Global Constraints

- ALL code/comments/commit messages in English.
- env/ changes ADDITIVE + GATED (default off); no observation-space change (agent_088 checkpoint must load).
- RL-1 verdict (training_log.md Agent 088 section) is the baseline: wp 0→2 consolidated, route31 flat 0. RL-2's hypothesis: reset pressure AT the lagging segment + earned budget breaks the wp-2 ceiling.
- DO NOT LAUNCH the real run: implement + test + smoke only. The controller launches when the GPU is free (LLM-2 runs in progress). The launch command and training_log entry are prepared but the `nohup` step is explicitly the controller's.
- Tests: `.venv/bin/python -m pytest tests/test_confinement.py tests/test_frontier_score.py tests/test_dynamic_budget.py -q` (do NOT run the slow legacy root suite).
- Save states for stages exist: `saves/crossing.state`, `saves/route31.state`, `saves/violet_city.state` (all egg-delivered, on-corridor).

---

### Task 1: Dynamic episode budget (gated env feature)

**Files:**
- Modify: `env/pokemon_env_cnn.py`
- Modify: `agents/rl/config.py` (flag, default False), `agents/rl/train_cnn.py` (wiring, mirrors confine_to_corridor)
- Test: `tests/test_dynamic_budget.py`

**Interfaces:**
- Produces: `PokemonEnvCNN(..., dynamic_episode_budget: bool = False)`. When True: `reset()` sets `self._max_steps = DYN_BUDGET_BASE` (=16384, module constant next to MAX_STEPS) instead of MAX_STEPS; in `step()`, when the episode's max waypoint ordinal INCREASES, `self._max_steps = min(MAX_STEPS, DYN_BUDGET_BASE * (1 + episode_max_waypoint))`. Frontier-origin episodes keep their existing shorter `frontier_max_steps` cap (budget applies to start/curriculum episodes only — if `self._from_frontier`, leave the cap alone).
- The episode's waypoint ordinal already exists in the env (the same machinery feeding `nav/ep_max_waypoint` / frontier scoring — grep `max_waypoint` in `pokemon_env_cnn.py` and reuse it; do NOT recompute).
- `config.DYNAMIC_EPISODE_BUDGET = False` here (True in Task 2).

- [ ] **Step 1: Failing test** — `tests/test_dynamic_budget.py`, same monkeypatched-RAM pattern as `tests/test_confinement.py` (copy its ROM/STATE constants and `requires_rom` marker):

```python
@requires_rom
def test_budget_starts_at_base_and_grows_on_waypoint(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True, dynamic_episode_budget=True)
    env.reset()
    assert env._max_steps == 16384
    real_read = env.ram_reader.read_all
    def fake_read():
        s = real_read()
        s["map_bank"], s["map_number"] = 26, 3   # Cherrygrove = waypoint 1
        return s
    monkeypatch.setattr(env.ram_reader, "read_all", fake_read)
    env.step(0)
    assert env._max_steps == 32768
    env.pyboy.pyboy.stop(save=False)


@requires_rom
def test_budget_off_by_default(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True)
    env.reset()
    assert env._max_steps == 65536
    env.pyboy.pyboy.stop(save=False)
```

(Adapt the waypoint-ordinal expectation to the env's actual waypoint machinery — if Cherrygrove's ordinal in `WAYPOINT_ORDER` differs from 1, use the real value and compute 16384*(1+ordinal) accordingly; the assertion values above assume ordinal 1.)

- [ ] **Step 2:** verify failure (unknown kwarg). **Step 3:** implement. **Step 4:** the three-suite test command green. **Step 5:** commit `feat(rl): gated dynamic episode budget - long episodes must be earned (R2b)`.

---

### Task 2: agent_089 config + smoke (NO launch)

**Files:**
- Modify: `agents/rl/config.py`
- Modify: `training_log.md` (entry prepared, marked "LAUNCH PENDING")

**Interfaces:**
- Produces config: `RUN_NAME="agent_089"`; warm from `runs/checkpoints/agent_088/agent_088_39999936_steps.zip`; `TOTAL_TIMESTEPS=60_000_000`; RL-1 flags kept (CONFINE_TO_CORRIDOR=True, EXPLORATION_SCALE=4.0, frontier on); `DYNAMIC_EPISODE_BUDGET=True`; `CURRICULUM_STATES_CNN` re-weighted to add the staged saves — keep the majority of envs on `saves/egg_delivered_clean.state` and assign a minority across `crossing/route31/violet_city` (exact split: preserve the TOTAL env count `N_ENVS_CNN`; e.g. with 12 envs: 6× egg_delivered_clean, 2× crossing, 2× route31, 2× violet_city — adjust proportionally to the real N_ENVS_CNN and document the split). Header comment: RL-2 hypothesis + kill criterion `no new segment (nav/reach_violet == 0.0) after 60M → stop; also stop early at 30M if reach_route31 still 0.0` (tighter early gate since staged resets should crack route31 fast).
- Smoke: same protocol as RL-1 Task 3 Step 2 (3-minute background boot, checkpoint loads, workers up, TB dir `runs/agent_089_1/` appears, clean kill of the smoke process by exact PID). GPU is shared with an Ollama run — a 3-minute smoke is acceptable, a real launch is NOT.
- training_log entry: "Agent 089 — CORRIDOR FINAL ATTEMPT RL-2 (RL-1 + R2)" with hypothesis, config deltas vs 088, kill criteria, and the line `**LAUNCH PENDING** (controller launches when the GPU frees up after the LLM-2 runs).`

- [ ] **Step 1:** config. **Step 2:** smoke + clean kill. **Step 3:** training_log entry. **Step 4:** commit `feat(rl): agent_089 config - staged resets + dynamic budget (RL-2, launch pending)`.

---

## Self-Review

**Spec coverage:** R2(a) staged saves → Task 2 curriculum split; R2(b) budget → Task 1; warm-from-088 handoff per the RL-1 verdict → Task 2; controller-gated launch → explicit in both Global Constraints and Task 2. **Placeholder scan:** ordinal-dependent assertion flagged for adaptation with the real value — intentional. **Type consistency:** `dynamic_episode_budget` kwarg + `DYNAMIC_EPISODE_BUDGET` config named consistently; DYN_BUDGET_BASE constant defined once.
