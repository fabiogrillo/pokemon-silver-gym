# LLM Agent — Design Spec

**Date:** 2026-06-29
**Status:** Approved (pre-implementation)
**Author:** Fabio Grillo

## Goal

Build a local, text+vision LLM agent that plays Pokémon Silver through the same shared
environment as the RL agent, with the same win condition: navigate from the
*egg-delivered* state (`saves/egg_delivered_clean.state`) through Elm's Lab → Cherrygrove →
Route 30/31 → Violet City and **beat Gym Leader Falkner for the Zephyr Badge**.

The agent reasons in a ReAct loop, perceives the game as **text (RAM→prompt) + a screenshot**,
and acts through a small set of tools. It is evaluated with the same metrics as the RL agent so
the two can be compared quantitatively (the spine of the eventual blog post).

## Non-Goals (YAGNI)

- No fine-tuning. The agent is purely prompt-based.
- No ASCII mini-map in v1 (deferred to Plan B — see Risks).
- No multi-model comparison. A single model is used (`qwen3-vl:8b`); `mistral-small3.1` was
  dropped because it OOMs on the 16 GB GPU.
- No new game logic. The agent reuses the existing `PyBoyWrapper` and `RAMReader` unchanged.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| First milestone | Full corridor navigation (egg-delivered → Falkner) | Direct 1:1 comparison with RL v2 |
| Perception | Text (RAM→prompt) + screenshot (vision) | Most "human-like"; strong blog narrative |
| Action granularity | Hybrid: macros in overworld, single button in battle | Macros cut LLM calls 10–50× (run feasible); single-button in battle where each decision matters |
| Mode switching | Driven by `battle_type` (RAM `0xD116`) | Deterministic, no fragile heuristics |
| Model | `qwen3-vl:8b` via Ollama OpenAI-compatible API | Fits 16 GB comfortably (~6 GB), native vision + tools, fast iteration |
| Tools | `move`, `press`, `get_state`, `wait_frames` | Interaction is `press("a")`; no separate `talk` tool needed |
| ASCII mini-map | Plan B only | Honest test of vision-only navigation first |

Hardware: RTX 5080, 16 GB VRAM. Ollama 0.30.11.

## Architecture

The LLM agent attaches to the existing environment via the same "socket" the RL policy used:
`PyBoyWrapper` (emulator step/reset/state) + `RAMReader.read_all()` (state dict). Neither changes.

```
agents/llm/
├── config.py          # model name, base_url, max_steps, token budget, anti-stuck thresholds
├── perception.py      # RAM dict + frame → text prompt + screenshot (base64 PNG)
├── tools.py           # move(dir, steps) | press(button) | get_state() | wait_frames(n)
├── memory.py          # short-term memory: last N actions, seen landmarks, stuck detector
├── agent.py           # ReAct loop + overworld/battle mode switch
├── run.py             # single-run entrypoint; logs every Thought/Action/Observation (JSONL)
└── evaluate_llm.py    # N runs, same JSONL metrics as RL eval
agents/comparison.py   # join RL + LLM JSONL logs → one comparison table
```

### Data flow

```
PyBoyWrapper (GBC) → RAMReader.read_all() ─┐
                     screen.ndarray ───────┤→ perception → prompt + image
                                           │                     ↓
                            qwen3-vl:8b (Ollama, OpenAI-compatible chat + tools)
                                           │                     ↓
                                   tool_call → tools.execute() → advance emulator
                                           │                     ↓
                                   memory.update() + JSONL log → loop
```

## Components

### perception.py
- `to_prompt(state: dict) -> str`: render `RAMReader.read_all()` as compact text — position,
  map id, party HP/levels, enemy HP/level (in battle), quest flags, trainers beaten, and the
  current sub-goal. Mirrors PokéLLMon prompt design (in-context feedback, consistent actions).
- `screenshot(frame) -> str`: encode the PyBoy RGB frame as a base64 PNG for the vision model.
- System prompt encodes the overall objective and the ReAct contract.

### tools.py (≤4, JSON validated/repaired before execution)
- `move(direction, steps)` — overworld macro: press a direction `steps` times; **stop early** if
  the map id changes or a battle starts. One LLM call = many frames.
- `press(button)` — single button (A/B/Start/direction): battle actions, dialogue, warps.
- `get_state()` — return current textual state (when the model wants to "look again").
- `wait_frames(n)` — advance through animations/dialogue without input.
- All tool-call arguments are validated; malformed JSON is repaired or the call is re-requested.

### agent.py — ReAct loop + hybrid modes
```
reset(egg_delivered_clean.state)
loop until zephyr badge or max_steps:
    state = RAMReader.read_all()
    mode  = OVERWORLD if state["battle_type"] == 0 else BATTLE
    tools = MACRO_TOOLS if mode == OVERWORLD else SINGLE_ACTION_TOOLS
    prompt = perception(state, screenshot)
    thought, tool_call = ollama.chat(prompt, tools)
    obs = execute(tool_call)            # advance emulator
    memory.update(thought, tool_call, obs)
    log(thought, tool_call, obs, tokens)
```

### memory.py — short-term memory & stuck detection
LLM navigation agents get stuck (oscillating against a wall — the "Claude Plays Pokémon" lesson).
`memory.py` keeps the last N actions and seen maps/landmarks; a detector flags position loops and
injects a warning into the prompt ("you've been stuck here for K steps, try another direction").
Without this, runs don't terminate.

## Error Handling
- **Bad tool JSON**: validate args against the tool schema; repair (coerce types, clamp ranges) or
  re-request once; otherwise fall back to a no-op `wait_frames` and log the failure.
- **Model returns no tool call** (prose only): re-prompt once asking for a tool call; then no-op.
- **Ollama unreachable / timeout**: fail the run with a clear error; the eval harness records it.
- **Stuck loop**: stuck detector escalates the prompt; after a hard cap, end the run as "stuck".

## Evaluation
`evaluate_llm.py` mirrors `agents/rl/evaluate_cnn.py`: same N runs from
`saves/egg_delivered_clean.state`, same per-episode JSONL fields —
`badge`, `steps`, `tiles`, `battles_won`, wall-clock, **tokens**. Success criteria are defined
*before* running so the comparison is honest. `agents/comparison.py` joins RL + LLM JSONL logs
into one table (badge rate, mean steps, tile coverage, tokens, wall-clock).

## Development Roadmap (phased)

| Phase | What | Verifiable output |
|---|---|---|
| 0. Feasibility spike | One Ollama call with vision + a dummy tool, on `qwen3-vl:8b`. Confirm it returns a valid `tool_call` while reading a screenshot. | Script prints a valid tool_call from a screenshot |
| 1. Perception | `perception.py`: `to_prompt()` + base64 screenshot. No model yet. | Prints prompt + image from a save state |
| 2. Tools | `tools.py`: `move`, `press`, `get_state`, `wait_frames`, with JSON validation/repair. | Each tool moves the emulator as expected |
| 3. Minimal loop | `agent.py` + `run.py`: overworld-only ReAct, logs every Thought. Sub-goal: leave Elm's Lab → reach Cherrygrove. | A logged run that reaches Cherrygrove |
| 4. Battle mode | Mode switch on `battle_type`; single-button tools; win a Route 30 trainer. | A run that wins ≥1 battle |
| 5. Anti-stuck + memory | `memory.py` + loop detector. | A run that recovers from a stall |
| 6. Full run | Tune prompt/budget until it (sometimes) reaches Falkner. | ≥1 run that earns the Zephyr Badge |
| 7. Eval + comparison | `evaluate_llm.py` + `comparison.py`; RL vs LLM table. | Metrics table + writeup |

## Risks & Mitigations
1. **Local model can't combine vision + tools** → Phase 0 spike de-risks this before any agent
   code is written. Fallback: text + ASCII mini-map (drop vision).
2. **Overworld navigation never gets going** → macros + anti-stuck; if still failing, add the
   ASCII mini-map (Plan B) for spatial context.
3. **Cost / latency** → macros reduce calls 10–50×; a token budget in `config.py` caps runs.
4. **`battle_type` semantics unverified for all cases** → verify empirically (the address is used
   by the RL reward but its full enum should be confirmed) during Phase 4.

## References
- `docs/LLM_AGENT_RESOURCES.md` — study path (ReAct, Ollama tools, PokéLLMon)
- ReAct (Yao et al., 2022); PokéLLMon (arXiv 2402.01118)
- Ollama OpenAI compatibility + tool support
