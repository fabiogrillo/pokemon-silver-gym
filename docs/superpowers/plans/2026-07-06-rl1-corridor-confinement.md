# RL-1: Corridor Confinement + Waypoint-Scored Frontier (agent_088) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement attempt RL-1 of the final-corridor schedule (`docs/superpowers/specs/2026-07-06-final-attempt-findings.md` §4): techniques R1 (CONFINE_TO_CORRIDOR) + R4 (waypoint-scored frontier), then launch a 60M warm run (`agent_088`) from `agent_079_129999792_steps.zip`.

**Architecture:** Mirror the proven `CONFINE_TO_GYM` pattern (gated config flag → termination in `pokemon_env_cnn.py`) with a corridor-legal map set; re-score Phase-2 (egg-delivered) frontier cells by `max_waypoint` ordinal so ε-greedy resets concentrate at the leading edge; restore the agent_079 corridor recipe in `agents/rl/config.py` as `agent_088`, warm-started, with the new flags on.

**Tech Stack:** Python 3.12 (`.venv`), SB3 PPO + SubprocVecEnv, PyBoy, pytest, TensorBoard, RTX 5080 (16 GB, currently idle).

## Global Constraints

- ALL code/comments/commit messages in English.
- env/ changes here are ADDITIVE + GATED (default off) — exactly like `CONFINE_TO_GYM` was; existing checkpoints must stay loadable (NO observation-space changes in this attempt).
- Run tests with `.venv/bin/python -m pytest <path> -v` from repo root. Existing suites (`tests/llm`, `tests/test_comparison_gif.py`, `tests/test_map_layout.py`) must stay green.
- The training launch is BACKGROUND (`nohup`, log to `runs/agent_088_launch.log`); do not block on it.
- Kill criteria (enforced later by the controller, record them in training_log.md): `nav/reach_route31` still 0.0 at 40M → stop the run.
- Document the attempt in `training_log.md` following the existing per-agent entry format.

---

### Task 1: CONFINE_TO_CORRIDOR (env flag, gated) — technique R1

**Files:**
- Modify: `env/rewards.py` (define `CORRIDOR_LEGAL` map set)
- Modify: `env/pokemon_env_cnn.py` (constructor arg + termination, next to the CONFINE_TO_GYM block at ~line 239)
- Modify: `agents/rl/config.py` (new flag, default False here — turned on in Task 3)
- Test: `tests/test_confinement.py` (new)

**Interfaces:**
- Produces: `rewards.CORRIDOR_LEGAL: set[tuple[int,int]]` = `CORRIDOR_WHITELIST ∪ {ELM_LAB, VIOLET_GATEHOUSE, GYM_MAP}` (use the exact constants already defined in `env/rewards.py`; check their names with grep — the whitelist at line ~79 lists the overworld corridor; add the legal interiors: Elm's lab (24,5), the Route31→Violet gatehouse (26,11), the gym (10,7)). If a constant for one of these is missing in rewards.py, define it there with the RAM-verified (bank,num) from `agents/rl/map_layout.py`.
- Produces: `PokemonEnvCNN(..., confine_to_corridor: bool = False)`; when True and `current_map not in CORRIDOR_LEGAL` → `terminated = True` (same placement as the confine_to_gym check; both flags must be independently usable).
- Produces: `config.CONFINE_TO_CORRIDOR: bool` wired through wherever `CONFINE_TO_GYM` is passed to the env constructor (grep `confine_to_gym` in `agents/rl/` to find every wiring point: train_cnn.py, evaluate_cnn.py build_vec_env, etc. — wire the new flag the same way; **eval must NOT confine** — copy exactly how eval handles confine_to_gym today).

- [ ] **Step 1: Write the failing test**

`tests/test_confinement.py` (uses the emulator fixture pattern from `tests/llm/conftest.py` — copy the fixture inline here since tests/ root has no conftest with it; skip cleanly if ROM missing):

```python
import os
import pytest

from env.pokemon_env_cnn import PokemonEnvCNN
from env.rewards import CORRIDOR_LEGAL, CORRIDOR_WHITELIST

ROM = "pokemon_rom.gbc"
STATE = "saves/egg_delivered_clean.state"

requires_rom = pytest.mark.skipif(
    not (os.path.exists(ROM) and os.path.exists(STATE)), reason="ROM/state not available")


def test_corridor_legal_superset_of_whitelist():
    assert CORRIDOR_WHITELIST <= CORRIDOR_LEGAL
    assert (24, 5) in CORRIDOR_LEGAL   # Elm's lab
    assert (10, 7) in CORRIDOR_LEGAL   # gym
    assert (26, 11) in CORRIDOR_LEGAL  # Violet gatehouse
    assert (3, 70) not in CORRIDOR_LEGAL  # Dark Cave stays illegal


@requires_rom
def test_confinement_terminates_on_illegal_map(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True, confine_to_corridor=True)
    env.reset()
    # Simulate the RAM reporting an off-corridor map (Dark Cave) without walking there:
    real_read = env.ram_reader.read_all
    def fake_read():
        s = real_read()
        s["map_bank"], s["map_number"] = 3, 70
        return s
    monkeypatch.setattr(env.ram_reader, "read_all", fake_read)
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True
    env.pyboy.pyboy.stop(save=False)


@requires_rom
def test_no_confinement_by_default(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True)
    env.reset()
    real_read = env.ram_reader.read_all
    def fake_read():
        s = real_read()
        s["map_bank"], s["map_number"] = 3, 70
        return s
    monkeypatch.setattr(env.ram_reader, "read_all", fake_read)
    _, _, terminated, _, _ = env.step(0)
    assert terminated is False or True  # replaced in Step 3 — see note
    env.pyboy.pyboy.stop(save=False)
```

NOTE on the third test: the default-off case must assert that confinement specifically did not fire. Inspect how `terminated` is computed (zephyr/hp) — with the fake map the hp/zephyr terms are untouched, so assert `terminated is False`. Write it that way (the placeholder `or True` above MUST be removed; the committed test asserts `terminated is False`).

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_confinement.py -v` → ImportError (`CORRIDOR_LEGAL`) / TypeError (unknown kwarg).

- [ ] **Step 3: Implement** — `CORRIDOR_LEGAL` in rewards.py (with a comment citing the findings doc R1); `confine_to_corridor` arg + termination in pokemon_env_cnn.py (comment mirrors the confine_to_gym one, cites agent_079's off-path-lure diagnosis); `CONFINE_TO_CORRIDOR = False` in config.py next to `CONFINE_TO_GYM`; wire through train/eval the same way confine_to_gym is wired (eval: NOT confined).

- [ ] **Step 4: Tests pass** — new file 3/3 (2 skipped without ROM; on this machine they run) + `.venv/bin/python -m pytest tests/ -v --ignore=tests/llm -q` green.

- [ ] **Step 5: Commit** — `git add env/rewards.py env/pokemon_env_cnn.py agents/rl/config.py tests/test_confinement.py agents/rl/train_cnn.py agents/rl/evaluate_cnn.py && git commit -m "feat(rl): gated CONFINE_TO_CORRIDOR flag (R1 — structural off-path pruning)"`

---

### Task 2: Waypoint-scored frontier (technique R4)

**Files:**
- Modify: `env/frontier_archive.py` (`frontier_score` Phase-2 branch)
- Test: `tests/test_frontier_score.py` (new; pure function, no emulator)

**Interfaces:**
- Consumes: `frontier_score(egg_received, egg_delivered, return_progress, max_waypoint, gym)` — currently returns flat `1.0` for all delivered cells.
- Produces: delivered cells return `1.0 + max_waypoint` (so gym-adjacent cells outrank New Bark cells: waypoint ordinals 0..5 per `WAYPOINT_ORDER` in rewards.py — grep the exact name/levels); carry stays `2.0 + max(0, ...)`? NO — carry tier is Phase-1 machinery, LEAVE IT UNTOUCHED (`2.0`); pre-egg unchanged (`0.0`). The docstring's flat-tier rationale (agent_061) must be UPDATED, not deleted: flat-across-depth remains true WITHIN a waypoint level; the new scoring is flat-per-waypoint-tier, which preserves both original fixes (score of a cell still never changes: max_waypoint here must be the CELL's own waypoint at capture time, not the episode max — check what add() currently passes and, if it passes the episode max, change the call site to pass the cell's own value).

- [ ] **Step 1: Failing test**

```python
from env.frontier_archive import frontier_score


def test_delivered_cells_ranked_by_waypoint():
    base = dict(egg_received=True, return_progress=0, gym=False)
    s0 = frontier_score(egg_delivered=True, max_waypoint=0, **base)
    s3 = frontier_score(egg_delivered=True, max_waypoint=3, **base)
    s5 = frontier_score(egg_delivered=True, max_waypoint=5, **base)
    assert s0 < s3 < s5


def test_carry_and_preegg_tiers_unchanged():
    assert frontier_score(True, False, 0, 0, False) == 2.0
    assert frontier_score(False, False, 0, 0, False) == 0.0
```

(Adapt the positional/keyword form to the real signature.)

- [ ] **Step 2: Run to verify failure** (delivered cells all equal today).
- [ ] **Step 3: Implement** (+ audit the `add()` call site for cell-own vs episode-max waypoint; fix call site if needed and say so in the commit message).
- [ ] **Step 4: Tests pass** (new + any existing frontier tests).
- [ ] **Step 5: Commit** — `feat(rl): rank delivered frontier cells by waypoint ordinal (R4)`

---

### Task 3: agent_088 config + smoke + launch

**Files:**
- Modify: `agents/rl/config.py`
- Modify: `training_log.md` (new agent_088 entry)

**Interfaces:**
- Consumes: the agent_079 corridor recipe. Recover it from git history (`git log --all -S "agent_079" --oneline -- agents/rl/config.py`, then `git show <sha>:agents/rl/config.py`) and `training_log.md`'s agent_079 entry (bidirectional frontier ON, Violet curriculum anchors, `EXPLORATION_SCALE = 4.0`). Current config.py is in agent_087 gym-slice state — every gym-specific override must be reverted to the corridor values (state path `saves/egg_delivered_clean.state`, CONFINE_TO_GYM=False, gym reward gates off, corridor curriculum on, episode length as 079 used).
- Produces: `RUN_NAME = "agent_088"`, `WARM_START = "runs/checkpoints/agent_079/agent_079_129999792_steps.zip"` (use the config's actual warm-start mechanism — grep how 085/087 warm-started), `TOTAL_TIMESTEPS = 60_000_000`, `CONFINE_TO_CORRIDOR = True`, header comment: RL-1 hypothesis + kill criterion.

- [ ] **Step 1: Write the config** (header comment format: mirror the agent_087 header style; cite findings §4 RL-1).
- [ ] **Step 2: Smoke run** — launch the trainer for ~3 minutes in foreground with a tiny override if the trainer supports it, else background + kill: verify it loads the 079 checkpoint (log line), workers start, TensorBoard dir `runs/agent_088_1` appears, no crash. `pkill` the smoke run cleanly.
- [ ] **Step 3: Real launch (background)** — `nohup .venv/bin/python -m agents.rl.train_cnn > runs/agent_088_launch.log 2>&1 &` then confirm: process alive after 60 s, log shows checkpoint loaded + env workers up, `nvidia-smi` shows the process. Record PID in the report.
- [ ] **Step 4: training_log.md entry** — new "Agent 088 — CORRIDOR FINAL ATTEMPT RL-1 (R1+R4)" section: hypothesis (findings §2 R1/R4 one-liners), config deltas vs 079, kill criterion (`nav/reach_route31` 0.0 at 40M → stop), launch timestamp.
- [ ] **Step 5: Commit** — `git add agents/rl/config.py training_log.md && git commit -m "feat(rl): launch agent_088 — corridor confinement + waypoint frontier (attempt RL-1)"`

---

## Self-Review

**Spec coverage:** R1 → Task 1; R4 → Task 2; RL-1 run config/launch → Task 3; kill criteria recorded → Task 3 Step 4. LLM-side techniques are a separate plan (parallel track).
**Placeholder scan:** the third confinement test's `or True` placeholder is explicitly flagged for replacement in its NOTE — intentional instruction, not a leftover. Recipe values recovered from git/training_log at execution (procedure given).
**Type consistency:** `confine_to_corridor` kwarg name used in Tasks 1/3; `CORRIDOR_LEGAL` in rewards.py consumed by env; `frontier_score` signature preserved.
**Executor notes:** Task 3 requires reading git history + training_log to reconstruct the 079 recipe — implementer must be careful and document every config delta in the training_log entry. GPU is idle and reserved for this run.
