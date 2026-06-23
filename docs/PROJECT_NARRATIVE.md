# Teaching an RL Agent to Beat a Pokémon Gym — From Pixels

> Reference script / outline for a YouTube video and blog post.
> Schematic on purpose: each section is a "beat" you can narrate over gameplay GIFs.
> Source of truth for details: `training_log.md` (full run-by-run lab notebook).

---

## 0. The hook (30s)

- The goal: a reinforcement-learning agent that plays **Pokémon Silver** *from raw pixels* — no scripting,
  no memory cheats for control — and earns the **first gym badge (Zephyr, Falkner)** starting from a fresh save.
- The honest spoiler: **it never beat the gym from the true start** — but the *story is the failures*. Over
  ~70 training runs we learned exactly *why* it's hard, and that's the interesting part.
- Tagline: "What does it actually take to teach a neural net to play a Game Boy game like a person?"

## 1. The setup (1 min)

- **Emulator**: PyBoy (Game Boy emulator) drives Pokémon Silver, controllable from Python.
- **Agent**: Stable-Baselines3 **PPO** with a **CnnPolicy** (NatureCNN) — it sees the **screen** (downsampled
  72×80 RGB, 4 frames stacked for motion) plus a small RAM-derived state vector (HP, level, in-battle, story
  flags). Reward/ground-truth read from RAM; *control* is pixels-only (like a human).
- **Reward**: dense per-new-tile exploration (move = progress) + big story events (get egg, deliver egg, beat
  trainers, badge). Scaled so events dominate exploration — the PokéRL (PWhiddy) recipe.
- **The task chain** the agent must discover: New Bark → Cherrygrove → Mr. Pokémon's house (**get the egg**) →
  **back south to Prof. Elm** (deliver) → this opens the Route 30 story gate → Route 31 → Violet City →
  **Violet Gym → beat Falkner**.

## 2. Act I — "It learns to walk" (1–2 min) [GIF: early agent vs agent_053]

- Early runs (MLP on a state vector, 18 of them): never generalize — the state vector lacks spatial info.
  → **Lesson 1: representation matters.** Switch to pixels (CnnPolicy).
- With pixels + dense exploration, the agent reliably learns the first leg:
  **start → Cherrygrove → pick up the egg** (consolidated to 100% by run *agent_053*).
- **GIF beats**: an early agent wandering aimlessly near New Bark, then agent_053 confidently walking the route
  and grabbing the egg. Visible learning.

## 3. Act II — Wall #1: "It won't carry the egg home" (2–3 min) [GIF: agent_053 stalling at Cherrygrove]

- To progress, the agent must carry the egg **back south** to Elm. It refuses.
- Why? A subtle, deep reason:
  - All the exploration reward it ever earned was *northward* (toward new tiles). It built a strong **northward
    habit**.
  - Carrying the egg south crosses **already-visited tiles that pay nothing** — a "barren corridor" — toward a
    reward that's far away.
- We threw the kitchen sink at it (each ruled out in the log):
  - **Reward shaping** (breadcrumbs, dense southward gradient): doesn't beat the habit; big rewards *destabilize*.
  - **Go-Explore / frontier reset** (restart from states on the agent's own trajectory): looked like it worked —
    metrics showed "delivery!" — but it was an **artifact** (the agent was being *reset near the goal*, not
    *navigating* there).
  - **A visual "I'm carrying the egg" marker** in the image (so the CNN can tell the two modes apart): the
    single shared network still collapses one skill while learning the other.
- **Lesson 2 (the big one): you cannot reward-shape your way past an *exploration* problem.** And **metrics can
  lie** — always verify behavior in evaluation, not just training curves.
- **Verdict**: from-start delivery is a *fundamental wall* for this method. We characterized it thoroughly, then
  **re-scoped**: study the gym fight directly, starting from a state where the egg is already delivered.

## 4. Act III — Wall #2: "The gym fight" (2 min) [GIF: agent_072 beating Falkner's Pidgeotto]

- From inside the gym (Totodile already strong enough), the agent *mostly wanders out to grind wild battles*
  instead of fighting Falkner — a **focus problem**, not a strength problem.
- Fix that worked: **start training from *inside* the Falkner battle** (forces the fight, no wandering). The
  agent learns to attack and **win → the badge fires (~50% at peak)**. The fight is *winnable and learnable*.
- Caveat: the fight policy is **unstable** (over-trains into bad moves and forgets) — the current open problem.
- **GIF beat**: the agent selecting attacks and KO-ing Falkner's Pidgeotto — the closest thing to "victory".

## 5. The lessons worth the video (2–3 min)

1. **Representation > cleverness.** Pixels beat a hand-built state vector; the network needs to *see* the world.
2. **Exploration is the real boss.** Sparse, distant rewards behind "barren" regions defeat reward shaping. The
   agent's *prior habits* dominate.
3. **Beware metrics that flatter you.** "Delivery solved!" in training was a reset artifact; evaluation from a
   true start told the truth. Always eval the actual deployment condition.
4. **Curriculum / anchor states segregate.** Training from foreign save-states builds skills that *don't
   transfer* to the real start — the network keys behavior to the visual scene.
5. **Decompose ruthlessly, and know when a wall is fundamental.** We split the task, isolated each sub-skill,
   and learned to *stop* throwing compute at a characterized wall (the discipline that's easy to lack).

## 5b. Re-baseline — "skip the backtracking, and *watch* it move" (1–2 min) [GIF: trajectory overlay]

- Following PWhiddy's own simplification (he started *after* Pokémon Red's early backtracking), we
  **re-baseline**: one **generalist** trained from `egg_delivered_clean.state` — the egg is already
  delivered, the Route 30 gate is already open — so the agent only has to *navigate forward* to Violet
  and fight Falkner. (The reward's carry/return terms are **self-inert** from this state, so no reward
  surgery is needed — just re-enable forward exploration.)
- **Seeing what it learned**: a Whidden-style **map overlay** (`agents/rl/visualize_map.py`) replays a
  checkpoint and draws its **trajectory** (colored by time) + **visitation heatmap** on a stitched map
  of the corridor. A per-run **progression montage** shows the path reaching further across checkpoints
  — the single most legible "it's learning" artifact for the video.
- **Lesson 3: pick the right starting line.** Cutting a *characterized* hard sub-problem (the
  backtrack) is not giving up — it's spending compute where the learning is visible.

## 6. Where it stands & what's next (1 min)

- **Achieved**: pixels-only navigation to the egg pickup (100%); proof the Falkner fight is winnable;
  an offline map-visualization overlay to *see* the agent's path; a Dockerized playback demo.
- **In progress**: the v2 generalist from the egg-delivered state → Falkner (validate short, then a
  multi-day run).
- **Not achieved**: the badge from a true start (the two walls compound).
- **Two honest paths forward** (pick one to discuss on camera):
  - **Push RL further**: stabilize the fight, then a *reverse curriculum* (extend the start point backward) to
    chain fight → gym → delivery. Higher effort, uncertain payoff, but a cleaner "we beat the gym" ending.
  - **Switch to an LLM agent** (the project's planned Phase 5): an LLM with tools/vision *reasons* about the
    quest instead of discovering it by trial-and-error — directly contrasting "learning from scratch (RL)" vs
    "reasoning with prior knowledge (LLM)". Likely the more compelling narrative and the stronger ending.

## 7. Closing line

- "RL didn't beat the game — but it showed us, precisely, *where the hard part is*: not the fighting, but the
  exploration. And that's the most honest thing a research project can deliver."

---

### Appendix — suggested on-screen artifacts
- GIFs (≤2x, via `agents/rl/make_gif.py`): early agent · agent_053 pickup · agent_050 longest route ·
  agent_072 Falkner KO.
- The **map-trajectory overlay** (`agents/rl/visualize_map.py`): per-checkpoint PNG/GIF and the
  per-run progression montage — the quest chain with the agent's actual path drawn on it (credit
  PWhiddy's coordinate-based exploration + pokerl-map-viz as the design reference).
- One training curve that *looked* like success (frontier "delivery") next to the eval that debunked it.
