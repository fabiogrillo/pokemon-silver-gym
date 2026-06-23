# Pokemon Silver Gym — RL Agent vs AI Agent

A personal portfolio project exploring two approaches to playing **Pokemon Silver**: a
**Reinforcement Learning agent** (PPO) and a planned **text-based LLM agent** (local Ollama).
Both share one game environment and one win condition: **earning the Zephyr Badge** from Gym
Leader Falkner in Violet City.

The goal is to understand, hands-on, how RL training works, how an LLM agentic loop is built,
and how to compare the two quantitatively. Every line is written by hand (Copilot for
autocomplete only) to maximize learning.

> **Status**: work in progress. The RL agent (CnnPolicy + frame stack) is the active line;
> the LLM agent is planned. The RL line is currently on a **v2 re-baseline** (PWhiddy-style): a
> single generalist trained from the *egg-delivered* state to navigate New Bark → Violet City and
> beat Falkner — no backtracking — with an offline **map-visualization** overlay and a Dockerized
> playback demo.

---

## What this project demonstrates

- A custom **Gymnasium environment** built from a Game Boy ROM via PyBoy
- Reading **ROM RAM** (empirically verified addresses) for state extraction and reward shaping
- Training a **PPO agent** (Stable Baselines3) with parallel environments and TensorBoard
- **Map visualization** (PWhiddy-style): replay a checkpoint and overlay its trajectory + visitation
  heatmap on a stitched map of the New Bark → Violet corridor (PNG + animated GIF)
- **Dockerized playback**: run a pretrained agent from the egg-delivered state toward Falkner
- (Planned) a **ReAct loop** with tool-calling over the OpenAI-compatible Ollama API
- **Quantitative RL vs LLM comparison**: badge rate, steps/episode, exploration heatmaps

---

## Architecture

```
pokemon-silver-gym/
├── env/                       # Shared environment layer
│   ├── pyboy_wrapper.py       # PyBoy wrapper: step, reset, save/load state, GIF capture
│   ├── ram_reader.py          # RAM reader: position, HP, badges, battle, event flags
│   ├── pokemon_env_cnn.py     # Gymnasium env (CnnPolicy): Dict obs (RGB image + state vector)
│   ├── pokemon_env_mlp.py     # Gymnasium env (MlpPolicy) — legacy, superseded by the CNN line
│   ├── rewards.py             # Shared, RAM-driven reward function
│   ├── frontier_archive.py    # Go-Explore frontier reset (trajectory state-sharing)
│   └── actions.py             # Action space (8 GBC buttons)
│
├── agents/rl/
│   ├── config.py              # Hyperparameters + run config
│   ├── train_cnn.py           # PPO CnnPolicy training (SubprocVecEnv + TensorBoard)
│   ├── train_mlp.py           # PPO MlpPolicy training — legacy
│   ├── evaluate_cnn.py        # Eval over N episodes (per-episode JSONL, GIF, live --watch)
│   ├── make_gif.py            # Watchable gameplay GIF of a checkpoint
│   ├── map_layout.py          # Map-id → global canvas offsets (corridor stitching table)
│   ├── visualize_map.py       # Trajectory + heatmap overlay on the corridor map (PNG + GIF)
│   ├── play.py                # Dockerized playback entrypoint (egg-delivered → Falkner)
│   └── evaluate.py            # MLP eval — legacy
├── agents/llm/                # LLM ReAct agent (planned)
│
├── assets/maps/               # Background image (optional) for the trajectory overlay
├── Dockerfile                 # CPU-only image for the playback demo
├── tests/                     # RAM/event verification + smoke tests
├── saves/                     # Save states (egg_delivered_clean.state + others)
└── runs/                      # Checkpoints, TensorBoard logs, maps, run logs (gitignored)
```

**Data flow**:
```
PyBoy (GBC emulator) → RAMReader → Gymnasium Env → PPO agent
                                          ↓
                                 PyBoy.button(action) → next frame
```

---

## Win condition

Both agents start in New Bark Town with Totodile. **Goal**: reach Violet City and beat Gym
Leader Falkner for the **Zephyr Badge**. Tracked sub-milestones: tiles explored, the egg quest
(Mr. Pokemon → Elm, which opens the Route 30 story gate), trainer battles won, badge obtained.

---

## RL agent — PPO

- Algorithm: PPO (Stable Baselines3), **CnnPolicy** + `VecFrameStack(4)`. The earlier MlpPolicy
  line was superseded — full rationale and run history in `training_log.md`.
- Observation: Dict — downsampled 72×80 RGB screen + an 11-float state vector (HP, levels,
  battle, story flags) so the policy can read state pixels don't show.
- Reward: dense coordinate exploration + story events (egg, badge) + battle wins, all RAM-driven
  (`env/rewards.py`).
- Training: 12 parallel envs (SubprocVecEnv), GPU policy updates.
- **v2 re-baseline (current)**: one generalist from `saves/egg_delivered_clean.state` (the egg is
  already delivered, so the Route 30 gate is open and the backtracking sub-quest is skipped — the
  reward's carry/return terms are self-inert from this state). Objective: navigate Elm's Lab →
  Cherrygrove → Route 30/31 → Violet City → beat Falkner. Validate short, then scale to a long run.

## LLM agent — ReAct (planned)

- Pattern: ReAct (reason + act) over a local Ollama model via the OpenAI-compatible SDK.
- Tools: `press_button`, `get_game_state`, `wait_frames`.

---

## Comparison metrics

| Metric | RL Agent | LLM Agent |
|--------|----------|-----------|
| Badge rate | % over 100 eval episodes | % over 20 runs |
| Steps per episode | mean ± std | mean ± std |
| Unique tiles explored | heatmap | heatmap |
| Wall-clock per run | minutes | minutes |
| Tokens per run | N/A | Ollama token count |

---

## Setup

**Prerequisites**: Python 3.12+ (3.14 used), an NVIDIA GPU with CUDA (for RL training), and your
own legal Pokemon Silver ROM (`*.gbc`, not included).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp /path/to/pokemon_silver.gbc pokemon_rom.gbc
```

Train (CNN line) and watch metrics:

```bash
python -m agents.rl.train_cnn
tensorboard --logdir ./runs/          # http://localhost:6006
```

Evaluate a checkpoint (stochastic; `--watch` opens an SDL2 window):

```bash
python -m agents.rl.evaluate_cnn --model runs/checkpoints/<run>/<ckpt>.zip \
  --state saves/start.state --episodes 20 --log
```

Visualize where the agent went (trajectory + heatmap overlay → PNG + GIF):

```bash
python -m agents.rl.visualize_map --model runs/checkpoints/agent_076/<ckpt>.zip \
  --state saves/egg_delivered_clean.state --max-steps 8000 --out runs/maps/agent_076
# progression montage over all checkpoints of a run:
python -m agents.rl.visualize_map --all-checkpoints runs/checkpoints/agent_076 \
  --state saves/egg_delivered_clean.state --out runs/maps/agent_076
```

(If `assets/maps/johto_corridor.png` is absent, a labelled schematic background is generated — see
`docs/MAP_VISUALIZATION.md`.)

Dockerized playback demo (CPU-only; mount your ROM + a checkpoint):

```bash
docker build -t silver-falkner-agent .
docker run --rm \
  -v "$PWD/pokemon_rom.gbc:/app/pokemon_rom.gbc" \
  -v "$PWD/runs:/app/runs" \
  -e MODEL=/app/runs/checkpoints/agent_076/agent_076_final.zip \
  silver-falkner-agent --map
```

---

## Status

- [x] PyBoy wrapper + verified RAM reader (position, HP, badges, battle, event flags)
- [x] Gymnasium env (CNN + MLP lines) + RAM-driven reward shaping
- [x] PPO training pipeline (SubprocVecEnv + TensorBoard + checkpoints)
- [x] Evaluation tooling (per-episode JSONL, GIF, live `--watch`)
- [x] Map-visualization overlay (trajectory + heatmap, PNG + GIF) and Dockerized playback demo
- [~] RL v2 re-baseline: generalist from `egg_delivered_clean.state` → Falkner (validation run)
- [~] Reaching the Zephyr Badge from `start.state` — open research problem (see `training_log.md`)
- [ ] LLM ReAct agent
- [ ] Evaluation/comparison suite + writeup

---

## Key references

- [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator in Python
- [pret/pokegold](https://github.com/pret/pokegold) — Pokemon Gold/Silver decompilation (RAM map)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) — RL environment standard
- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) — reference RL project
- [ReAct paper](https://arxiv.org/abs/2210.03629) — Yao et al. 2022
- [Ollama OpenAI compatibility](https://ollama.com/blog/openai-compatibility)

---

*Built by Fabio Grillo as part of his AI Engineer portfolio.*
