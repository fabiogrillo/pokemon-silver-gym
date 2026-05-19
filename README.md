# Pokemon Silver Gym — RL Agent vs AI Agent

A personal portfolio project exploring two fundamentally different approaches to playing **Pokemon Silver**: a **Reinforcement Learning agent** trained via PPO, and a **text-based AI agent** powered by a local LLM (Ollama). Both agents share the same game environment and compete for the same win condition: **earning the Zephyr Badge** from Gym Leader Falkner in Violet City.

The goal is not just to play Pokemon — it's to understand, from the ground up, how RL training works, how an LLM-based agentic loop is built, and how to compare two fundamentally different AI approaches quantitatively. Every line of code is written by hand (GitHub Copilot for autocomplete only) to maximize learning.

> **Status**: Work in progress — Week 4 of a 4-week build.

---

## What This Project Demonstrates

- Building a custom **Gymnasium-compatible environment** from a Game Boy ROM using PyBoy
- Reading and using **internal ROM memory** (RAM addresses) for game state extraction and reward shaping
- Training a **PPO agent** (Stable Baselines3) with parallel environments and tracking training via TensorBoard
- Implementing a **ReAct loop** with tool-calling using the OpenAI-compatible Ollama API
- **Quantitative comparison** of RL vs LLM agent: badge rate, steps per episode, map exploration heatmaps
- Full **Docker deployment** so anyone can reproduce the training and agent runs

---

## Architecture

```
pokemon-silver-gym/
│
├── env/                          # Shared layer used by both agents
│   ├── pyboy_wrapper.py          # PyBoy emulator wrapper: step, reset, save/load state, GIF capture
│   ├── ram_reader.py             # ROM RAM reader: position, HP, badges, battle state
│   ├── pokemon_env.py            # Gymnasium Env: obs space + action space + reward function
│   └── actions.py                # Action space definition (8 GBC buttons)
│
├── agents/
│   ├── rl/                       # Reinforcement Learning Agent (PPO)
│   │   ├── config.py             # Hyperparameters
│   │   ├── train.py              # PPO training loop (SubprocVecEnv + TensorBoard)
│   │   └── evaluate.py           # Evaluation over N episodes
│   └── llm/                      # LLM Agent (ReAct loop)
│       ├── tools.py              # Tool definitions in OpenAI tool-calling format
│       ├── prompts.py            # System prompt + state serialization
│       ├── agent.py              # Main ReAct loop
│       └── run.py                # Entry point with full logging
│
├── evaluation/
│   ├── benchmark.py              # Runs both agents, collects metrics
│   ├── metrics.py                # Badge rate, steps, tile coverage
│   └── visualize.py             # GIF generation, heatmaps, reward plots
│
├── notebooks/                    # Exploration and analysis
│   ├── 01_explore_pyboy.ipynb    # PyBoy + RAM address verification
│   └── 02_training_analysis.ipynb
│
├── tests/                        # Unit tests
├── saves/                        # Initial game save state (New Bark Town, Totodile)
├── runs/                         # Training output (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

**Data flow**:
```
PyBoy (GBC emulator) → RAM Reader → Gymnasium Env → RL Agent (PPO) or LLM Agent (ReAct)
                                                              ↓
                                                    PyBoy.button(action) → next frame
```

---

## Win Condition

Both agents start from New Bark Town with Totodile as their starter Pokemon.  
**Goal**: navigate to Violet City and defeat Gym Leader Falkner to earn the **Zephyr Badge**.

Sub-milestones tracked:
- Tiles explored (navigation progress)
- Trainer battles won
- Badge obtained (yes/no per episode)

---

## Agents

### RL Agent — Proximal Policy Optimization (PPO)
- Algorithm: PPO via Stable Baselines3
- Policy: MLP (observation is a numeric vector, not image frames)
- Training: 8 parallel environments (SubprocVecEnv), GPU-accelerated policy updates
- Observation: `[map_id, player_x, player_y, badge_count, lead_hp, lead_max_hp, in_battle, unique_tiles]`
- Reward: exploration bonus + badge win condition + trainer defeat signals

### LLM Agent — ReAct Loop with Ollama
- Pattern: ReAct (Reason + Act) — think in natural language, call a tool, repeat
- LLM: local Ollama (`llama3.1:8b` or `llama3.3:70b` in 4-bit)
- API: OpenAI-compatible SDK (`from openai import OpenAI` with `base_url="http://localhost:11434/v1"`)
- Tools: `press_button`, `get_game_state`, `wait_frames`
- Each step: RAM state → serialized text → LLM reasons → tool call → execute

---

## Comparison Metrics

| Metric | RL Agent | LLM Agent |
|--------|----------|-----------|
| Badge rate | % over 100 eval episodes | % over 20 runs |
| Steps per episode | mean ± std | mean ± std |
| Unique tiles explored | heatmap | heatmap |
| Wall-clock time per run | minutes | minutes |
| Tokens per run | N/A | Ollama token count |

---

## Setup

### Prerequisites

- Python 3.12+ (project uses Python 3.14)
- NVIDIA GPU with CUDA (for RL training)
- [Ollama](https://ollama.com) installed and running locally (for LLM agent)
- Pokemon Silver ROM (`*.gbc`) — not included for copyright reasons. You must provide your own legal copy.

### Install

```bash
git clone https://github.com/YOUR_USERNAME/pokemon-silver-gym.git
cd pokemon-silver-gym
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Place your ROM

```bash
cp /path/to/your/pokemon_silver.gbc pokemon_rom.gbc
```

### Run RL training

```bash
python agents/rl/train.py
tensorboard --logdir ./runs/   # open http://localhost:6006
```

### Run LLM agent

```bash
ollama pull llama3.1:8b
python agents/llm/run.py
```

### Docker

```bash
docker-compose up
```

---

## Build Checklist

### Phase 0 — Project Setup

- [x] Create GitHub repository `pokemon-silver-gym` (public)
- [x] Clone locally
- [x] Create virtual environment (`.venv`) and install dependencies
- [x] Create project directory structure (`env/`, `agents/rl/`, `agents/llm/`, `evaluation/`, `tests/`, `saves/`)
- [x] Add `__init__.py` to `env/`, `agents/`, `agents/rl/`, `agents/llm/`, `evaluation/`
- [x] Install GitHub Copilot in VS Code
- [x] Finalize `.gitignore` (ROM `*.gbc`, `runs/`, `.venv/`, `__pycache__/`, `saves/*.pkl`)
- [x] Write `requirements.txt` with pinned versions
- [x] First commit and push

---

### Phase 1 — PyBoy Exploration & RAM Reading

- [x] Open `notebooks/01_explore_pyboy.ipynb`
- [x] Read PyBoy documentation: https://github.com/Baekalfen/PyBoy
- [x] Run the game headless: `PyBoy("pokemon_rom.gbc", window="null")`
- [x] Advance frames: `pyboy.tick(1, render=False)`
- [x] Capture a screenshot: `pyboy.screen.ndarray`
- [x] Read a RAM address: `pyboy.memory[0xDCB8]` (player X position)
- [x] Run the game with SDL window, play manually, print RAM values while moving
- [x] Verify empirically: Player X (`0xDCB8`), Player Y (`0xDCB7`), Map ID (`0xDCB6`)
- [x] Verify: Badge bits (`0xD857`) — bit 0 should flip after beating Falkner
- [x] Verify: Battle flag (`0xD116`) — changes when entering/exiting a battle
- [x] Verify: Lead Pokemon HP (`0xDCFC`) and Max HP (`0xDCFE`)
- [x] Create initial save state after choosing Totodile in New Bark Town → `saves/initial_state.pkl`
- [x] Commit notebook with findings

---

### Phase 2 — Gymnasium Environment

- [x] Read Gymnasium documentation: https://gymnasium.farama.org/ ("Your first environment")
- [x] Write `env/actions.py` — `ACTIONS` list + `gymnasium.spaces.Discrete(8)`
- [x] Write `env/ram_reader.py` — `RAMReader` class, `read_all() -> dict`
- [x] Write `env/pyboy_wrapper.py` — `PyBoyWrapper`: `__init__`, `step(action)`, `reset()`, `capture_gif()`
- [ ] Write `env/pokemon_env.py` — `PokemonEnv(gymnasium.Env)`: `observation_space`, `action_space`, `step()`, `reset()`
- [ ] Test environment with random agent (1000 steps, no crash, reward printed)
- [ ] Write `tests/test_ram_reader.py` — test known values (badge count = 0 at start)
- [ ] Write `tests/test_env.py` — test reset shape, step reward, done flag
- [ ] Run `pytest tests/` — all pass
- [ ] Commit

---

### Phase 3 — RL Agent: Training Loop

- [ ] Read SB3 docs: https://stable-baselines3.readthedocs.io/
- [ ] Write `agents/rl/config.py` — all hyperparameters as constants
- [ ] Write `make_env(rank)` helper function (seeded env factory)
- [ ] Write `agents/rl/train.py` — `SubprocVecEnv` (4–8 envs) + PPO + `CheckpointCallback`
- [ ] Launch TensorBoard: `tensorboard --logdir ./runs/`
- [ ] Run training 100k steps — verify TensorBoard shows `ep_rew_mean` and `ep_len_mean`
- [ ] Debug if reward is always 0 or episode length always max (reward/RAM reader issue)
- [ ] Write `agents/rl/evaluate.py` — load model, run N episodes, print metrics
- [ ] Generate first GIF from a trained checkpoint
- [ ] Commit

---

### Phase 4 — Reward Engineering

- [ ] Add tile tracking to `PokemonEnv`: `self.visited_tiles = set()` tracking `(map_id, x, y)`
- [ ] Update `step()`: reward `new_tiles * 0.01`, penalize revisiting if agent loops
- [ ] If agent never leaves starting area: increase reward for map transition
- [ ] Optional: add image-based exploration bonus (hash of downsampled screen frames)
- [ ] Document all reward changes and rationale in notebook
- [ ] Run 500k–1M step training — TensorBoard shows exploration improvement
- [ ] Generate comparison GIF: early checkpoint vs later checkpoint
- [ ] Commit

---

### Phase 5 — LLM Agent: ReAct Loop

- [ ] Verify Ollama is running: `ollama serve` (background) + `ollama pull llama3.1:8b`
- [ ] Test Ollama from Python (OpenAI SDK with `base_url="http://localhost:11434/v1"`)
- [ ] Write `agents/llm/tools.py` — `TOOLS` list in OpenAI tool-calling format
- [ ] Write `agents/llm/prompts.py` — system prompt + `build_user_message(state_dict) -> str`
- [ ] Write `agents/llm/agent.py` — ReAct loop: read state → serialize → Ollama → execute tool → repeat
- [ ] Write `agents/llm/run.py` — entry point, saves full log (thought + action + state) to JSON
- [ ] Test manually for 5–10 runs — read Thought logs, adjust system prompt if needed
- [ ] Run 20 full evaluation runs and record badge rate
- [ ] Commit

---

### Phase 6 — Evaluation & Comparison

- [ ] Write `evaluation/metrics.py` — `compute_badge_rate`, `compute_mean_steps`, `compute_tile_coverage`
- [ ] Write `evaluation/benchmark.py` — run RL (100 episodes) + LLM (20 runs), save `results.json`
- [ ] Write `evaluation/visualize.py` — reward curve, heatmaps, comparison table, GIF generator
- [ ] Run full benchmark, generate all visualizations
- [ ] Create `notebooks/02_training_analysis.ipynb` with full analysis and commentary
- [ ] Commit

---

### Phase 7 — Polish & Publication

- [ ] Write complete `README.md` — architecture, installation, results with GIFs, learnings
- [ ] Write `Dockerfile` + `docker-compose.yml` (with NVIDIA runtime for GPU)
- [ ] Test Docker from scratch (clean path, no venv)
- [ ] Code cleanup: remove debug prints, add one-line docstrings to public functions
- [ ] Run `pytest tests/` final check — all pass
- [ ] Create GitHub Release v1.0 with best GIFs as release assets
- [ ] Publish blog post on spaghettibytes.blog
- [ ] Publish LinkedIn post
- [ ] Update FAANG roadmap: mark Pokemon Silver Gym as COMPLETE

---

## Key References

- [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator in Python
- [pret/pokegold](https://github.com/pret/pokegold) — Pokemon Silver full decompilation (RAM map)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) — RL environment standard
- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) — reference RL project
- [ReAct paper](https://arxiv.org/abs/2210.03629) — Yao et al. 2022
- [Ollama OpenAI compatibility](https://ollama.com/blog/openai-compatibility)

---

*Built by Fabio Grillo as part of his AI Engineer portfolio.*
