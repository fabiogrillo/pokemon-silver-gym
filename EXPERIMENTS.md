# Experiments

A condensed log of what worked and what didn't while getting an agent to walk from New Bark Town to
the Violet City Gym and beat Falkner for the Zephyr Badge. The full blow-by-blow lives in the git
history; this is the distilled version.

The task decomposes into two hard parts that turned out to need very different treatment:

1. **Navigation** — cross six maps (New Bark → Route 29 → Cherrygrove → Route 30 → Route 31 → Violet
   City) through a story gate that only opens after an egg-delivery side-quest.
2. **The gym fight** — navigate the gym interior and win three battles (two trainers + Falkner).

Two agents were built: a PPO CNN reinforcement-learning agent, and a tool-calling vision-LLM agent
(`qwen3-vl:8b` via Ollama). The headline result is that RL solves the navigation the LLM cannot.

## Reinforcement learning

### Reward shaping

The single most important lesson was reward *scale discipline*. Early rewards mixed terms spanning
three orders of magnitude (+1000 story spikes next to +0.01 exploration bonuses). Under a reward
normalizer the running standard deviation is dominated by the rare spikes, which crushes the dense
exploration signal the agent actually navigates on. Keeping every term single-digit and applying one
global 0.1 scale fixed a whole class of "the agent just grinds in place" failures.

The second lesson was the *ratio* between exploration income and story/combat income. If touring a
new area can out-earn the story chain, the agent settles into a touring/grinding local optimum and
never progresses. Weighting events to dominate exploration by ~10–50× (following the ratio Peter
Whidden's Pokémon Red experiments use) was what made forward progress the profitable strategy.

Things that were tried and removed: a global wild-battle penalty (taught grass-avoidance, which
blocked the levelling needed for the gym), potential-based shaping with a negative for moving the
wrong way (the agent avoided the rewarded state entirely rather than eat the penalty), and large
one-shot map-milestone bonuses (destabilized the value function). The reward that stuck is monotonic
weighted event flags + saturating level reward (so grinding self-extinguishes) + a small dense
per-new-tile exploration bonus.

### The egg-delivery wall

The story gate on Route 30 opens only after an egg is picked up north of Cherrygrove and carried
~600 steps back to Elm's lab. This was the hardest single sub-skill: the backtrack crosses
already-visited tiles that pay no exploration reward, so a from-scratch policy never *experienced*
the delivery event and it never entered the gradient. Two ingredients cracked it:

- **A Go-Explore frontier archive** (`env/frontier_archive.py`): a fraction of environment resets
  restart from save-states harvested from the policy's *own* trajectory, shared across
  `SubprocVecEnv` workers through the filesystem. This manufactures the state diversity that 12
  parallel envs alone can't reach across the Route 29 bottleneck, without introducing any foreign
  save-state (which would create a visually-distinct "island" the CNN learns to segregate on —
  a failure mode that recurred throughout the project).
- **Directional return breadcrumbs**: small latched, positive-only map-entry bonuses paid only while
  carrying the undelivered egg, which factorize the long backtrack into learnable sub-goals.

### Navigation: solved

The corridor-navigation breakthrough came from three structural changes, layered over three cold/warm
runs:

- **Confine-to-corridor**: end the episode the instant the agent leaves the legal corridor. Earlier
  runs plateaued at Route 30 because a high exploration weight lured the policy into dead-end maps
  (Dark Cave, Sprout Tower), where episodes died earning nothing. Removing the *option* to wander
  off-path (rather than trying to tune the reward against it) was decisive — the same "remove the bad
  option from the state space" idea that solved the gym (below).
- **Earned episode budget**: start episodes begin with a short step cap that only grows when a new
  corridor waypoint is reached (from the Pokémon Red paper, arXiv:2502.19920). Aimless wandering is
  truncated fast; genuine progress buys room to continue.
- **The visited-coordinates observation**: a 48×48 crop of the tiles visited this episode, added as a
  separate Dict-observation key (not a stacked image channel, so battle/menu screens don't corrupt
  it). This was the key input. An earlier ablation had concluded this signal *hurt* — but that signal
  was being drawn **transposed**, because the RAM reader's `local_x`/`local_y` fields hold the Y/X
  coordinates swapped relative to their names. Drawn transposed, the crop never tracks the player's
  real movement axes; drawn correctly (de-transposed) it is exactly the indispensable input the paper
  reports. The earlier negative result was a coordinate bug, not evidence about the feature.

With all three, the final run (`agent_090`, cold, 100M steps planned) reached the Violet Gym in about
**95% of start-state episodes, sustained from 30M to 70M steps** — the first agent in the project to
solve full-corridor navigation from the New Bark start as a consolidated behavior.

Three caveats, all honest:

- **The badge is not reliably won by this policy.** Beating Falkner happened in only 6 of ~1500
  evaluation rollouts. The corridor reward optimizes navigation, not in-battle tactics.
- **Late-training collapse.** Past 75M steps `reach_gym` fell from ~0.9 to ~0.0 — textbook on-policy
  PPO drift: with navigation consolidated but the badge reward still unmet, the gradient kept pushing
  and degraded the working behavior. Periodic checkpointing makes this free to recover from; the best
  checkpoint is `agent_090_49999920` (50M steps), **not** the latest one.
- **The ~95% was a property of the *live* policy, not of any frozen checkpoint.** This one took a
  full forensic pass to find. Loading the 50M (or 30M) checkpoint and rolling it out offline — same
  env code, same wrapper stack, same save state, byte-identical observation spaces, even replaying
  it through SB3's own rollout collector with the learning rate pinned to 0 — produces a policy that
  wanders New Bark Town and never completes the corridor. Warm-starting a new run from the same zip
  confirms it from the other side: its true-start metrics begin near zero, then recover over a few
  million steps of fine-tuning. The training-time `reach_gym` was real, but it described a policy
  being continuously updated *while* it played (a 12k-step episode spans several PPO updates, each
  adapting to that very episode's fresh rollouts). The moment the weights are frozen, the behavior
  isn't there. On-policy "solved" metrics do not certify checkpoints — only offline rollouts of the
  frozen snapshot do.

### Fine-tuning toward the badge (agent_091)

The follow-up run warm-starts from `agent_090`'s 50M checkpoint with the learning rate halved, a
short entropy anneal, and three of the twelve envs moved to gym-interior save states (two gym-start,
one mid-Falkner-battle) so the remaining gap — the fight — gets direct gradient while eight
true-start envs keep the corridor reinforced. The frontier archive is seeded from `agent_090`'s own
814 harvested cells.

The fine-tune fixed the frozen-checkpoint problem within 20M steps: `agent_091`'s 20M snapshot,
rolled out cold from the New Bark start, walks the full corridor into the gym in ~5–6k steps per
episode — verified offline in both `DummyVecEnv` and `SubprocVecEnv` harnesses, which is exactly the
verification `agent_090` never passed.

### The gym fight: solved separately

Trained in isolation from a save-state *inside* the gym, the fight is solvable and stable. The same
structural trick applied: confine-to-gym (leaving the gym map ends the episode) took the badge rate
from a drifting ~40% to a reliable **100%**, because it removes the "wander out and grind wild
battles" basin that had capped every reward-tuning attempt. Best gym agent: `agent_087`, 100% badge,
~840 steps/episode, zero losses.

So both halves are solved — navigation as a ~95% behavior, the gym fight at 100% — just not welded
into one end-to-end policy. Composing the two is the natural next step.

## The LLM agent

The vision-LLM agent (`qwen3-vl:8b`, local via Ollama) reads the screen + a text state summary and
calls one tool per turn (`move` / `press` / `navigate_to` / `get_state` / `wait_frames`).
`navigate_to` runs A* over collision grids extracted from the game's map data.

Six attempts were run against the corridor task, iterating on a single failure mode:

1. The model would not pick strategic navigation targets — it perseverated on one coordinate. Fix: a
   **harness-owned leg checklist** where the harness (not the model) tracks the corridor as an
   ordered list of legs and each turn's prompt carries only the current leg's target. This made the
   model obedient — ~96% of overworld turns then issued `navigate_to` at the correct target.
2. With the model obedient, the bottleneck moved to the **executor** hitting dynamic sprites the
   static collision grid can't see. Each hardening pass (recover from a scripted NPC by engaging it →
   take a probe-verified greedy step when A* fails → face-and-press to clear a camping trainer)
   closed one interaction class, and a new sprite exposed the next.
3. Attempt 6 closed the sprite class wholesale: read the live NPC positions out of the game's own
   object table (`wObjectStructs`, from the pokegold disassembly, verified against the emulator) and
   feed them to A* as blocked tiles before every plan, so the executor routes around people it has
   never bumped into. It worked — the corridor that had taken hours of grinding fell in minutes
   (Route 29 + Cherrygrove crossed by turn ~183) — and then exposed the next class: the agent
   stepped through a doorway into a **gridless interior** (no collision grid is committed for
   houses) and spent 271 turns sliding along one row of a small room, misreading which map it was
   on, until the run was cut. Each fix genuinely raises the floor; the environment keeps having a
   next thing.

Best result across all six attempts: **Route 30 (2 of 6 maps)**. The LLM never reached Violet City,
the gym, or Falkner from the New Bark start. The failure is executor/environment coupling, not
reasoning: the model plans correctly but cannot execute reliably against an adversarial, unpredictable
environment one tool call at a time — exactly the kind of problem RL's trial-and-error solves.

Started from a save state inside the gym, the same agent beats both bird keepers (real battle wins,
picking moves) and then stalls against Falkner for ~850 turns with a level-4 Weedle on the field —
tactical reasoning holds up in the small, it's the long-horizon execution that never closes.

## Result

| | Best result from the New Bark start | Reaches Violet Gym | Beats Falkner |
|---|---|---|---|
| **RL** (`agent_091`, fine-tuned) | full corridor | yes, ~90% as a **frozen checkpoint** (38/41 offline episodes) | fine-tune in progress; solved at 100% in isolation by `agent_087` |
| **LLM** (`qwen3-vl:8b`, 6 attempts) | Route 30 (2/6 maps) | no | no (beats both bird keepers from a gym start, stalls on Falkner) |

## A note on reading the game's memory

Much of this project depended on reading the right bytes out of the emulator's RAM — party levels,
HP, the story event flags that gate progress, the player's map and coordinates. Those addresses were
found by cross-referencing the pokecrystal/pokegold disassembly with live memory scans while walking
the route, and Claude was a genuine help pinning them down. One quirk worth flagging for anyone
extending this: the RAM reader's `local_x`/`local_y` fields hold the coordinates **swapped** relative
to their names (WRAM stores Y before X). That's frozen in `env/ram_reader.py`; every geometry consumer
un-swaps at its own boundary (see `agents/rl/map_layout.ram_to_image_px`). Getting that wrong is what
produced the transposed visited-map ablation described above.
