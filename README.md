# Pokemon Silver Gym — RL Agent vs AI Agent

Two agents, one task: beat **Gym Leader Falkner** for the **Zephyr Badge** in **Pokémon Silver**.
One is a **Reinforcement Learning agent** (PPO, trained from pixels). The other is a **local
vision+text LLM agent** (ReAct over Ollama). They share the exact same game environment, so we can
compare *learning by trial-and-error* against *reasoning about the goal* — same metrics, same map.

The focused milestone is the **gym vertical slice**: start inside the Violet City Gym, climb to
Falkner, win the badge. The full New Bark → Violet *corridor* is kept as a stretch goal.
Everything is written by hand (Copilot for autocomplete only) to maximize learning.

> **Status**: ✅ **RL agent solved the gym slice — 100% badge rate** (`agent_087`, ~840 steps/episode,
> from `saves/violet_city_gym.state`). 🔬 The **LLM agent** (vision + ReAct + tool-calling) is built and
> tested; its finding is the crux of the comparison — **it wins battles but cannot navigate the gym to
> Falkner** (fixates / heads for the exit), where the RL policy scores 100%. The full corridor remains an
> open research problem (see `training_log.md`).

---

## 🎮 Try it (the RL agent, CPU-only)

Watch the trained agent walk into Falkner's gym and earn the Zephyr Badge — no GPU, no Python setup.
You only need **Docker** and your **own legal** Pokémon Silver ROM (it is *not* distributable, so it
is never bundled in the image — you mount it at runtime).

```bash
# 1. Pull the demo image (the pretrained agent_087 is baked in; CPU-only inference)
docker pull ghcr.io/fabiogrillo123/silver-falkner-agent:latest   # image name finalized at publish

# 2. Run it — mount YOUR rom as pokemon_rom.gbc
docker run --rm \
  -v "$PWD/pokemon_silver.gbc:/app/pokemon_rom.gbc" \
  ghcr.io/fabiogrillo123/silver-falkner-agent:latest
```

It loads `agent_087` from `saves/violet_city_gym.state`, climbs the gym, beats the two bird-keepers
and Falkner, and prints `🏅 ZEPHYR BADGE — Falkner defeated!` — typically within ~840 steps.

> Want a trajectory overlay or a gameplay GIF? Append `--map` or `--gif` and mount a writable
> `-v "$PWD/runs:/app/runs"`. To run a *different* checkpoint, set `-e MODEL=/app/runs/checkpoints/<run>/<ckpt>.zip`.

---

## What this project demonstrates

- A custom **Gymnasium environment** built from a Game Boy ROM via PyBoy
- Reading **ROM RAM** (empirically verified addresses) for state extraction and reward shaping
- Training a **PPO agent** (Stable Baselines3) with parallel environments and TensorBoard
- **Map visualization** (PWhiddy-style): replay a checkpoint and overlay its trajectory + visitation
  heatmap on a stitched map of the New Bark → Violet corridor (PNG + animated GIF)
- **Dockerized playback**: `docker pull` + run the pretrained agent (CPU-only) to watch it beat Falkner
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
- **Gym slice — SOLVED (`agent_087`, 100% badge)**: trained from `saves/violet_city_gym.state`
  (inside the gym, Totodile lv15). The decisive fix was *structural*, not reward-tuning: a
  `CONFINE_TO_GYM` flag ends the episode if the agent leaves the gym map, removing the
  "wander out and wild-grind" basin that capped earlier runs at ~40%. The full arc
  (10% → 40% → 100%) and the lesson are in `training_log.md` (agents 083–087).
- **Full corridor** (New Bark → Violet) remains an open exploration problem — the stretch goal.

## LLM agent — ReAct (local vision+text)

- Pattern: ReAct (reason + act) over a local Ollama model (`qwen3-vl:8b`) via the OpenAI-compatible SDK.
- Perception: RAM state as text **+ a screenshot** each turn. Tools: `move`, `press`, `get_state`,
  `wait_frames`, with a mode switch (movement macros overworld / single presses in battle).
- **Finding — fights but doesn't navigate:** on the gym slice the LLM reliably **wins battles**
  (advancing dialogue and attacking), but **cannot navigate the gym to Falkner**. It fixates at
  junctions and funnels toward the exit instead of climbing up. Getting it this far required fixing a
  button-edge bug (`settle` frames), an empty-response fallback, a save/restore **walkability probe**,
  an anti-fixation guardrail, and an RL-style **gym confinement** — yet even fully confined it bounces
  at the door rather than reaching Falkner. This is the crux of the comparison: the trained RL policy
  *internalizes the goal geometry* (100%); the local LLM *reasons in text* but has no spatial compass.
  Full debug arc in `docs/superpowers/plans/2026-06-29-llm-agent.md`.

---

## Comparison metrics

Gym vertical slice, from `saves/violet_city_gym.state`:

| Metric | RL Agent (`agent_087`) | LLM Agent (`qwen3-vl:8b`) |
|--------|------------------------|---------------------------|
| Badge rate | **100%** | **0%** (never reaches Falkner) |
| Battles won | full gym → badge | 1 (first bird-keeper), via mash-`a` |
| Failure mode | — | can't navigate: fixates / funnels to the exit |
| Steps per episode | ~840 | 500 cap (hits it wandering) |
| Wall-clock per run | seconds (CPU) | ~40 min (~6 s/step vision inference) |
| Tokens per run | N/A | ~1.3M / 500-step run |

Numbers are produced by `agents/llm/evaluate_llm.py` (LLM) and `agents/rl/evaluate_cnn.py` (RL),
joined by `agents/comparison.py`.

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

Evaluate / watch a checkpoint (`--watch` opens an SDL2 window; `--speed 2` = 2× so it's viewable):

```bash
python -m agents.rl.evaluate_cnn --model runs/checkpoints/agent_087/agent_087_final.zip \
  --state saves/violet_city_gym.state --episodes 10 --watch --speed 2 --log
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

Build the playback image yourself (CPU-only; the `agent_087` checkpoint is baked in — see the
**Try it** section above for the pre-built pull):

```bash
docker build -t silver-falkner-agent .
docker run --rm \
  -v "$PWD/pokemon_rom.gbc:/app/pokemon_rom.gbc" \
  silver-falkner-agent
```

---

## Status

- [x] PyBoy wrapper + verified RAM reader (position, HP, badges, battle, event flags)
- [x] Gymnasium env (CNN + MLP lines) + RAM-driven reward shaping
- [x] PPO training pipeline (SubprocVecEnv + TensorBoard + checkpoints)
- [x] Evaluation tooling (per-episode JSONL, GIF, live `--watch`)
- [x] Map-visualization overlay (trajectory + heatmap, PNG + GIF) and Dockerized playback demo
- [x] **RL gym slice solved** — `agent_087`, 100% badge from `violet_city_gym.state` (`training_log.md`)
- [x] **LLM agent** — vision + ReAct + tool-calling over Ollama (`qwen3-vl:8b`), 21 tests green
- [~] LLM gym slice: **wins battles but does not navigate to Falkner** — the RL-vs-LLM finding (documented)
- [x] Evaluation/comparison suite (RL vs LLM) — `evaluate_llm.py` + `comparison.py` (24 tests green)
- [ ] Final writeup / blog + video
- [ ] Full New Bark → Violet corridor — stretch goal (open research problem)

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
