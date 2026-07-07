# LLM-3: Harness-Driven Leg Goals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attempt LLM-3 per `docs/superpowers/specs/2026-07-06-final-attempt-findings.md` §3-L4/§4: the HARNESS owns the corridor route as an ordered leg checklist (auto-advancing on map change); each turn's prompt carries ONLY the current leg's goal ("you are on <map> at (x,y); reach the exit at (tx,ty) — call navigate_to(tx,ty)"). The model's residual job: battles + leg-local decisions. LLM-2 evidence this addresses: 50/56 navigate_to calls perseverated on one mid-town coordinate; the model adopts the tool but cannot pick strategic targets.

**Architecture:** A new `agents/llm/legs.py` derives each corridor map's EXIT TILES from the committed collision grids (walkable tiles on the map border, grouped by side) — no hand-typed coordinates. A `LegTracker` holds the ordered corridor legs (map id + exit side [+ explicit override for the gym door / Falkner]), advances on map change, and renders the per-turn goal line. `agent.py` consumes it behind a config flag (`leg_mode`, default True for the corridor; False restores LLM-2 behavior).

**Tech Stack:** existing agents/llm stack; grids in `assets/collision/`; pytest; Ollama runs at the end (GPU shared with agent_089 training — runs are allowed under contention, ~13 s/step).

## Global Constraints

- ALL code/comments/commit messages in English. env/ untouched; agents/rl/ untouched (agent_089 training LIVE, PID in `.superpowers/sdd/progress.md` — never broad-pkill; kill only exact PIDs you started).
- True-(x,y) convention everywhere in agents/llm (RAM swap compensated at boundaries, cite `agents/rl/map_layout.ram_to_image_px`).
- Corridor legs use these map ids in order: (24,4) New Bark → west side; (24,3) Route 29 → west; (26,3) Cherrygrove → north; (26,1) Route 30 → north; (26,2) Route 31 → west; (26,11) gatehouse → west; (10,5) Violet City → gym door; (10,7) gym → Falkner. The gym-door tile on (10,5) and the Falkner tile on (10,7) are NOT border exits: mark them as explicit overrides — gym door: the door tile of the gym building in Violet City (find it from the grid: the walkable tile whose NORTH neighbor is the building edge at the known gym location — or simpler and acceptable: the tile where `saves/violet_city_gym.state` spawns MINUS the warp, i.e. document how you derived it); Falkner: the top-center walkable tile of the gym grid.
- Tests: `.venv/bin/python -m pytest tests/llm -q` green throughout.
- Kill criterion for the runs (findings §4): <2 legs completed per run across 2 runs → inspect cause (battle deaths vs navigation) before LLM-4.

---

### Task 1: legs.py — exit derivation + LegTracker

**Files:**
- Create: `agents/llm/legs.py`
- Test: `tests/llm/test_legs.py`

**Interfaces:**
- `border_exits(grid: dict, side: str) -> list[tuple[int,int]]` — walkable tiles on the given border (`"west"`: x==0; `"east"`: x==width-1; `"north"`: y==0; `"south"`: y==height-1), true (x,y).
- `LEGS: list[Leg]` — dataclass `Leg(map_key: tuple[int,int], name: str, target: tuple[int,int], hint: str)`; targets resolved at import from the grids (median border-exit tile of the leg's side) or the explicit overrides above.
- `LegTracker(legs=LEGS)`: `.current(bank, num) -> Leg | None` (first leg matching the current map; None if off-route), `.goal_note(bank, num, x, y) -> str` (one line: leg name, current position, target, `navigate_to` instruction; if off-route: "return to the corridor" + previous leg), `.completed_count(visited_map_ids) -> int`.
- Unit tests: border_exits on a synthetic grid literal; LEGS resolve (every leg target is walkable in its grid and on the declared side or an override); LegTracker.current transitions across the leg order; goal_note contains the target coordinates.

- [ ] Steps: failing tests → implement → green → commit `feat(llm): corridor leg tracker with grid-derived exit targets (L4)`.

---

### Task 2: wire leg_mode into the agent loop

**Files:**
- Modify: `agents/llm/agent.py`, `agents/llm/config.py`
- Test: `tests/llm/test_leg_mode.py`

**Interfaces:**
- `LLMConfig.leg_mode: bool = True` (comment: LLM-3 harness-owned goals; False = LLM-2 free-form).
- In `ReActAgent.run`, overworld turns with `leg_mode`: append `LegTracker.goal_note(...)` to the memory note (AFTER the walkable-directions line). The summary dict gains `"legs_completed": int`.
- Test with the emulator+StubClient pattern: 2-step run asserts the note passed to the client contains the New Bark leg target (spy on client.chat's messages) and summary has `legs_completed`.

- [ ] Steps: failing test → implement → green (full tests/llm) → commit `feat(llm): leg-goal prompting mode (LLM-3)`.

---

### Task 3: LLM-3 runs (2× under GPU contention) + verdict

Same runner protocol as LLM-2 (background `.venv/bin/python -m agents.llm.run`, poll the newest trace; ~13 s/step under training contention, ~110 min/run; run 2 only after run 1 ends). Collect per run: summary numbers (now incl. legs_completed), navigate_to usage + distinct targets, maps visited. Verdict vs the kill criterion. Append `### LLM-3 — harness leg goals (L4)` to training_log.md and commit `docs: LLM-3 runs - harness-driven leg goals results` + trailer. If a run crashes on a code error: capture traceback, return BLOCKED.

---

## Self-Review

Grid-derived exits avoid hand-typed coordinates (single source of truth = collision grids); overrides limited to the two non-border targets with derivation documented. leg_mode default True flips the corridor harness to LLM-3 semantics while keeping LLM-2 reproducible (flag off). Task 3's contention note matches the live agent_089 run. No placeholders; interfaces named consistently across tasks.
