# LLM Spatial Compass (Attempts LLM-1/LLM-2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement techniques L2 (coordinate un-swap in perception) and L1 (`navigate_to` tool backed by A* over a pret-derived walkability grid) from `docs/superpowers/specs/2026-07-06-final-attempt-findings.md` §3, then run attempts LLM-1 (calibration baseline) and LLM-2 per the §4 schedule.

**Architecture:** Perception reports true (x, y) by un-swapping the frozen ram_reader fields (the `map_layout.ram_to_image_px` convention, applied at the label level). An offline extraction script turns pret/pokegold map data (`maps/*.blk` + tileset `collision.bin` + `collision_permissions.asm`) into per-map walkability grids committed as JSON; an A* router over those grids (+ map connections already in `agents/rl/map_layout.py`) powers a new `navigate_to(x, y)` tool that EXECUTES the button presses via the existing move loop, stopping on battle/map-change. The LLM chooses where; the tool does how.

**Tech Stack:** Python 3.12 (`.venv`), pret/pokegold raw files via curl (extraction time only), pytest, Ollama qwen3-vl:8b (runs), existing `agents/llm/` stack.

## Global Constraints

- ALL code/comments/commit messages in English.
- `env/` MUST NOT change (frozen semantics; compensate at consumers — same rule as map_layout).
- ram_reader fields: `local_x` holds wYCoord, `local_y` holds wXCoord. In agents/llm code, un-swap AT THE BOUNDARY and use true (x, y) internally; every un-swap site cites `map_layout.ram_to_image_px`.
- Tool count stays ≤ 5: `navigate_to` joins move/press/get_state/wait_frames (overworld set only).
- Tests: `.venv/bin/python -m pytest tests/llm <new files> -v`; existing suites stay green. An RL training run (agent_088) is LIVE — do not touch `agents/rl/config.py`, do not run anything GPU-heavy, do not `pkill` python.
- Corridor maps for grids: the 6 overworld maps (24,4),(24,3),(26,3),(26,1),(26,2),(10,5) + gym (10,7) interior. Grid coordinate convention: `grid[y][x]` with true x east / y south, sizes in tiles per `agents/rl/map_layout.py` `MAP_INFO` (tiles = blocks×2).
- Ollama runs (LLM-1/LLM-2) start from `saves/egg_delivered_clean.state`, confinement OFF, `max_steps=500`.

---

### Task 1: L2 — true (x, y) in perception + waypoint tracking in the run summary

**Files:**
- Modify: `agents/llm/perception.py` (`format_state_text`)
- Modify: `agents/llm/config.py` (SYSTEM_PROMPT axis convention line)
- Modify: `agents/llm/agent.py` (summary: `max_waypoint`)
- Test: `tests/llm/test_perception.py` (update), `tests/llm/test_waypoint_tracking.py` (new)

**Interfaces:**
- Produces: `format_state_text` prints `at (x, y)` where `x = state["local_y"]`, `y = state["local_x"]` (un-swap; comment cites ram_to_image_px). SYSTEM_PROMPT gains: `"Coordinates: x grows EAST, y grows SOUTH; moving up decreases y."`
- Produces: `WAYPOINT_MAPS: list[tuple[tuple[int,int], int]]` in `agents/llm/perception.py` mapping map ids to ordinals `{(26,3):1, (26,1):2, (26,2):3, (10,5):4, (10,7):5}` (Cherrygrove=1 … gym=5; New Bark/Route29 = 0), helper `waypoint_ordinal(bank, num) -> int`; `ReActAgent.run` summary gains `"max_waypoint": int`.

- [ ] **Step 1: Update/write failing tests**

In `tests/llm/test_perception.py`, the base state has `local_x=4, local_y=6` (RAM order → true x=6, y=4). Update the assertion `assert "(4, 6)" in txt` to `assert "(6, 4)" in txt` with the comment `# ram local_x/local_y are swapped (wYCoord/wXCoord); perception must report TRUE (x, y)`. New `tests/llm/test_waypoint_tracking.py`:

```python
from agents.llm.perception import waypoint_ordinal


def test_waypoint_ordinals():
    assert waypoint_ordinal(24, 4) == 0   # New Bark
    assert waypoint_ordinal(24, 3) == 0   # Route 29
    assert waypoint_ordinal(26, 3) == 1   # Cherrygrove
    assert waypoint_ordinal(26, 1) == 2   # Route 30
    assert waypoint_ordinal(26, 2) == 3   # Route 31
    assert waypoint_ordinal(10, 5) == 4   # Violet City
    assert waypoint_ordinal(10, 7) == 5   # gym
    assert waypoint_ordinal(3, 70) == 0   # unknown maps -> 0
```

- [ ] **Step 2: Run to verify failure** — perception test fails on "(6, 4)"; new file ImportError.
- [ ] **Step 3: Implement** — un-swap in `format_state_text`; prompt line in config.py; `waypoint_ordinal` + `WAYPOINT_MAPS`; in `ReActAgent.run` track `max_wp = max(max_wp, waypoint_ordinal(state["map_bank"], state["map_number"]))` per step and add to the summary dict.
- [ ] **Step 4: Full LLM suite green** — `.venv/bin/python -m pytest tests/llm -q` (26+ tests: existing plus updates).
- [ ] **Step 5: Commit** — `feat(llm): report true (x,y) coordinates and track corridor waypoint (L2)`

---

### Task 2: LLM-1 calibration run

Run 1 episode: `.venv/bin/python -m agents.llm.run` from `saves/egg_delivered_clean.state` (edit nothing; `LLMConfig.state_path` default — check it still points at egg_delivered_clean; if config default was changed to the gym state during earlier phases, pass/patch it back for this run only). Record in `training_log.md` (new "LLM final attempt" section): summary line with steps/tokens/max_waypoint/battles. Expectation per findings: navigation still fails (calibration point). ~50 min wall clock. No commit of run logs (untracked); commit the training_log entry: `docs: LLM-1 calibration run result`.

---

### Task 3: L1a — walkability grid extraction (offline script + committed grids)

**Files:**
- Create: `agents/llm/extract_collision.py` (offline extractor, run once)
- Create: `assets/collision/<MAP>.json` (7 grids, committed — small)
- Test: `tests/llm/test_collision_grids.py`

**Interfaces:**
- Produces: `assets/collision/{new_bark,route_29,cherrygrove,route_30,route_31,violet_city,gym}.json`, each `{"bank": int, "num": int, "width": int, "height": int, "walkable": [[0|1,...],...]}` (row-major `grid[y][x]`, true axes, sizes matching `MAP_INFO` tiles).
- Extraction sources (fetch with curl at run time, do NOT vendor the repo): pret/pokegold `maps/<Name>.blk` (W×H block ids, sizes from constants/map_constants.asm — same numbers already comment-documented in `agents/rl/map_layout.py`), tileset collision tables `gfx/tilesets/*_collision.bin` (4 bytes per block: quadrant collision ids for the 2×2 tiles: order UL,UR,DL,DR — VERIFY against pokecrystal wiki "Add a new tileset"), walkability classes from `data/collision/collision_permissions.asm` (LAND* walkable; WATER/WALL non-walkable; ledges: mark NON-walkable in v1 — conservative, A* routes around them). Which tileset each map uses: `data/maps/maps.asm` map headers.
- The gym (10,7) is an indoor map — same pipeline, tileset johto/indoor per its header.

- [ ] **Step 1: Failing test** (grids don't exist yet):

```python
import json, os
import pytest
from agents.rl import map_layout as ml

GRIDS = {
    (24, 4): "assets/collision/new_bark.json", (24, 3): "assets/collision/route_29.json",
    (26, 3): "assets/collision/cherrygrove.json", (26, 1): "assets/collision/route_30.json",
    (26, 2): "assets/collision/route_31.json", (10, 5): "assets/collision/violet_city.json",
    (10, 7): "assets/collision/gym.json",
}


@pytest.mark.parametrize("key,path", GRIDS.items())
def test_grid_exists_and_matches_map_size(key, path):
    assert os.path.exists(path)
    g = json.load(open(path))
    assert (g["bank"], g["num"]) == key
    box = ml.MAP_INFO[key]
    assert g["width"] == box.size[0] and g["height"] == box.size[1]
    assert len(g["walkable"]) == g["height"] and len(g["walkable"][0]) == g["width"]
    flat = [c for row in g["walkable"] for c in row]
    assert 0 < sum(flat) < len(flat)  # some walkable, some not
```

- [ ] **Step 2-3: Write the extractor, run it, tests green.**
- [ ] **Step 4: GROUND-TRUTH CHECK against the emulator** — write a one-off check (inside the extractor as `--verify`): boot `saves/violet_city.state`, use the existing walkability probe machinery (`agents/llm/agent.py`'s save/restore 4-direction probe — factor the probe into a reusable function if needed) at the current position: probe result must MATCH the grid's 4 neighbors. Repeat for `saves/route31.state` and `saves/egg_delivered_clean.state`. 3 positions × 4 directions = 12 ground-truth cells; require ≥ 11/12 agreement (document any mismatch — tile events like doors can differ). THIS STEP GATES THE WHOLE APPROACH: if agreement is poor, STOP and return BLOCKED with the mismatch table.
- [ ] **Step 5: Commit** — `feat(llm): pret-derived walkability grids for the corridor (L1a)`

---

### Task 4: L1b — A* router + `navigate_to` tool

**Files:**
- Create: `agents/llm/pathfind.py`
- Modify: `agents/llm/tools.py` (schema + validate + execute), `agents/llm/config.py` (prompt: tool usage guidance)
- Test: `tests/llm/test_pathfind.py`, `tests/llm/test_tools_validate.py` (add cases)

**Interfaces:**
- Produces: `pathfind.load_grids() -> dict[(bank,num) -> Grid]`; `pathfind.astar(grid, start_xy, goal_xy) -> list[str] | None` (returns button directions `["up","left",...]` in TRUE axes; None if unreachable); `pathfind.plan(bank, num, start_true_xy, goal_true_xy)` same-map v1 (cross-map legs are the harness's job later — YAGNI now).
- Produces: tool `navigate_to(x: int, y: int)` (overworld only): validates target within current map bounds + walkable; executes the A* path with the existing press loop (`frames_per_press`, settle), re-reading RAM each step; stops early on battle_type>0, map change, or 3 consecutive non-moves (desync guard → observation `{"ok": False, "note": "path blocked at (x,y)"}`); observation reports final true (x,y).
- `validate_tool_call("navigate_to", ...)` coerces ints, rejects out-of-map targets with `ToolValidationError`.

- [ ] **Step 1: Failing tests** — A* unit tests on a tiny synthetic grid (reachable path length, wall detour, unreachable → None) + validate cases (bad coords) + an emulator test (fixture from tests/llm/conftest.py): from `saves/violet_city_gym.state` navigate 3 tiles north, assert position changed toward the target. Write them concretely (synthetic 5×5 grid literals).
- [ ] **Step 2-4: TDD to green** (full `tests/llm` + new files).
- [ ] **Step 5: Commit** — `feat(llm): A* navigate_to tool over walkability grids (L1b)`

---

### Task 5: LLM-2 runs + log

Two runs from `saves/egg_delivered_clean.state` with the new tool (~50 min each). Kill criterion (findings §4): tool works but the agent never calls it sensibly across both runs → stop, note for LLM-3 (harness-driven goals). Record both summaries (steps/tokens/max_waypoint/navigate_to call count) in `training_log.md`; commit `docs: LLM-2 runs — navigate_to tool results`.

---

## Self-Review

**Spec coverage:** L2 → Task 1; LLM-1 → Task 2; L1 grid → Task 3 (with the ground-truth gate); L1 tool → Task 4; LLM-2 → Task 5. LLM-3..5 (macro-waypoints, minimap, scratchpad, alt models) intentionally deferred to their own plan after LLM-2's outcome — the findings schedule is adaptive.
**Placeholder scan:** extraction byte-order marked VERIFY against the pret wiki (verification step included); no TBDs.
**Type consistency:** true-(x,y) convention stated once in Global Constraints and cited per task; `waypoint_ordinal` (T1) reused by T5's logging; grid JSON schema (T3) consumed by `pathfind.load_grids` (T4).
**Executor notes:** T2/T5 are ~1h wall-clock Ollama runs (background + poll, frames not needed); T3's extractor needs network; agent_088 training must not be disturbed (CPU headroom is fine: PyBoy single runs are light).
