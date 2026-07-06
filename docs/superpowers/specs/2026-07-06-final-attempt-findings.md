# Final Corridor Attempt — Research Findings (Phase C0)

**Date:** 2026-07-06 · **Task:** from `saves/egg_delivered_clean.state`, navigate New Bark → Route 29 →
Cherrygrove → Route 30/31 → Violet City, enter the gym, beat Falkner (Zephyr Badge).
**Budget:** 5 RL attempts + 5 LLM attempts. This doc ranks the candidate techniques and fixes the
attempt schedule with kill criteria. All claims grounded in `training_log.md`, the LLM plan outcome
(`docs/superpowers/plans/2026-06-29-llm-agent.md`), and the cited external sources.

---

## 1. Diagnosis — why the corridor is unsolved

**RL.** Across 87 agents, every corridor failure is the same shape: the *only* navigation driver is
exploration income, and whatever mis-calibration exists becomes a stable attractor the policy converges
into — Route-29 wild-grind (076), archive-bounded looping (077), Dark-Cave/Sprout-Tower new-map farming
(079's Route-30 plateau: 235 cave cells harvested while `nav/reach_route31` stayed 0.0 for ~83M steps).
Reward-side fixes consistently under-deliver: the agent_080/081 exploration whitelist removed the off-path
*income* but not the off-path *option* — the agent still physically enters caves/tower, burns episode
steps there earning nothing, and 081 ended up *worse* (stuck at Cherrygrove; whitelist reverted in 082).
The decisive counter-example is **agent_087**: the gym slice went 40% → **100% badge** not by another
reward term but by *removing the bad option from the state space* (`CONFINE_TO_GYM` terminates the episode
on leaving the gym map). The corridor equivalent has never been tried. Meanwhile agent_079 proved the
positive half of the recipe: bidirectional frontier (Go-Explore archive) + Violet curriculum anchors +
`EXPLORATION_SCALE=4.0` connected the corridor archive end-to-end and pushed the *start* policy past the
Phase-1 wall segment-by-segment (Cherrygrove 1.0 @ ~28M, Route-30 gate 1.0 @ ~37M, accelerating) before
the off-path lure stalled it. So the ingredients that advance the frontier are known; what is missing is
the structural pruning of dead-end options — exactly the 087 lesson, generalized.

**LLM.** The qwen3-vl:8b ReAct agent *wins battles* (mash-A works) but has **no spatial compass**: fully
confined and un-fixated it still funneled to the gym door 73 times and never climbed to Falkner. Two
compounding causes. (1) A latent data bug: `env/ram_reader.py` reads WRAM `wYCoord` into `local_x` and
`wXCoord` into `local_y` (WRAM order is Y-first; `agents/rl/map_layout.ram_to_image_px` is the canonical
un-swap). `agents/llm/perception.py` prints `({local_x}, {local_y})` as "(x, y)" — so every coordinate
the model ever reasoned about had its axes swapped, poisoning any "up decreases y" style inference.
(2) Even with clean coordinates, an 8B local VLM cannot do multi-step spatial planning from screenshots —
frontier labs hit the *same* wall: Claude 3.7 / Gemini 2.5 / o3 all needed a **pathfinding tool +
coordinate overlays + map memory** to navigate Pokémon at all, and navigation remained the weakest link
even then (see Sources). Gemini Plays Pokémon finished Crystal only with a fog-of-war tile map, a
navigate/pathfinder capability and persistent goal memory. Our agent has none of these: no map memory, no
coordinate goals, no pathfinding — only per-tile walkability probes. The fix is scaffolding (tools that
carry the spatial reasoning), not a better prompt.

---

## 2. RL — ranked candidate techniques (≤6)

**R1. CONFINE_TO_CORRIDOR — generalize the 087 structural fix. (top pick)**
What: new gated env flag (same pattern as `CONFINE_TO_GYM`, off by default): terminate the episode the
moment the agent enters a non-corridor map (complement of `CORRIDOR_WHITELIST` in `env/rewards.py`; keep
`ELM_LAB`/gatehouse interiors legal). Unlike the failed 080/081 income whitelist, this removes the
*option*, not just the reward — Dark Cave, Sprout Tower, school and houses stop being places where
episodes go to die. Why it beats the plateau: agent_079's diagnosis was precisely "each cave/tower floor
is a fresh +0.4 new-map lure"; with termination, entering one instantly ends the episode at a reward of
~0 marginal — PPO learns to avoid it within a few M steps, exactly as 087 learned to stay in the gym
(and 087's eval WITHOUT confinement stayed 100%, so the crutch generalizes away).
Cost: ~20 lines in `pokemon_env_cnn.py`/`config.py` (env change is additive+gated, checkpoints unaffected)
+ one 079-style run. Attempt: warm from `agent_079@130M` (already Route-30-capable), explore 4.0,
frontier + Violet anchors kept, 40–60M validation → extend to 200M+ if `nav/reach_route31` cracks.
Evidence: agent_087 arc (training_log.md); off-path-lure diagnosis (agent_079); PWhiddy V2's dense
coordinate exploration works when it can't be spent off-path — https://github.com/PWhiddy/PokemonRedExperiments.

**R2. Waypoint-staged episodes from intermediate saves + event-scaled episode budget.**
What: two structural episode tricks from the literature. (a) Curriculum envs reset from the corridor's
own intermediate saves — `saves/crossing.state`, `mid_route30.state`, `route31.state`,
`violet_city.state` — which are *on-corridor, same story flags* (egg delivered), so the visual-island
segregation that ruled out foreign-state curricula (agents 047–051) does not apply the same way: 078/079
already mixed Violet anchors productively for the archive. (b) Dynamic step budget per the Pokémon-Red
paper: start episodes at ~16k steps and grow the cap by a quantum per completed event/waypoint. Workers
desync, sample diversity rises, catastrophic forgetting drops — the paper calls the naive-navigation
reward + this budget indispensable to reach Cerulean. Why: it concentrates gradient on the *frontier
segment* (Route 30→31→Violet) instead of re-walking New Bark every episode.
Cost: config-only for (a); ~15 lines for (b). Attempt: combine with R1 in one run (both are structural,
non-conflicting). Evidence: https://arxiv.org/abs/2502.19920 (dynamic 10,240 + 2,048/event budget);
training_log agents 078–079.

**R3. Visited-coordinates observation channel — re-enabled and de-transposed.**
What: the Pokémon-Red paper's third input: a 48×48 **binary crop of visited coordinates centered on the
player**, encoded by its own Nature-CNN. Our 10e ablation removed the visited-mask channel as "semantic
noise", but the implementation had a real bug: `_visited_mask` uses `ram["local_x"]` as the screen-column
axis, and `local_x` is actually **wYCoord** — the mask was drawn transposed relative to the screen, i.e.
it genuinely *was* noise. Re-add it as a separate Dict-obs key (not a 4th image channel, so battle/menu
screens don't corrupt it), with axes routed through the `ram_to_image_px` un-swap convention.
Why: gives the CNN explicit "where the frontier is" — the paper found agents without the
visited-coordinate observation + nav reward fail all milestones. Cost: ~40 lines env-side (additive Dict
key ⇒ **cold start required**, old checkpoints won't load) + a cold 079-recipe run (~1–2 days).
Evidence: https://arxiv.org/abs/2502.19920 §II-C; boey lineage multi-channel map obs
https://github.com/CJBoey/PokemonRedExperiments1.

**R4. Frontier archive re-scored by waypoint ordinal (leading-edge sampling).**
What: `env/frontier_archive.py` exists and works; its `frontier_score` tiers are egg-quest-era
(carry > delivered > pre-egg) — meaningless for the corridor task where every state is "delivered".
Re-score cells by `max_waypoint` ordinal (gym 5 > Violet 4 > Route 31 3 > gate 2 > Cherrygrove 1) so
ε-greedy sampling concentrates resets at the leading edge instead of uniformly across the corridor.
Why: 079's archive connected end-to-end but start-policy consolidation lagged the archive by tens of M;
sharper reset pressure at the last-cracked segment shortens that lag. Cost: ~10 lines in
`frontier_score` + config. Attempt: fold into the R1 run (frontier already on in that recipe).
Evidence: Go-Explore-style state resets are also how pokemonred_puffer's "swarm" migrates env states
toward required paths — https://drubinstein.github.io/pokerl/, https://github.com/drubinstein/pokemonred_puffer.

**R5. Long run with the proven recipe (scale, last).**
What: the 079 recipe cracked segments at ~14M then ~9M each, *accelerating*, before the off-path stall;
agent_082's open question (131k-step episodes, warm 079) was never resolved because the project pivoted
to the gym slice. With R1+R2+R4 in place, commit the original 480M budget (~2 days on the RTX 5080).
Why: 4 segments remained past Route 30 (R31 → gatehouse → Violet → gym); at ~10–15M/segment that is
~40–60M of *productive* training — but only after the structural fixes; more steps alone reinforced the
grind (agent_076 lesson). Cost: GPU time only. Evidence: training_log 079/082; the Pokémon-Red paper's
400M-step curves (milestones keep unlocking late) — https://arxiv.org/abs/2502.19920.

**R6. Gym-arrival handoff (fallback, guaranteed partial win).**
What: if the corridor policy reaches the gym but flubs the fight, chain policies: corridor policy until
`map==GYM_MAP`, then swap in `agent_087_final` (100% gym badge) at inference. This is an eval-harness
composition, no training. Why: banked capability reuse; makes "badge from egg_delivered_clean" achievable
even if a single end-to-end policy falls short in 5 attempts. Cost: ~30 lines in an eval script. Honest
labelling required (composed system, not one policy). Evidence: agent_087 (100%), agent_085 arc.

---

## 3. LLM — ranked candidate techniques (≤6)

**L1. Spatial compass: `navigate_to(x, y | landmark)` tool backed by offline A*. (top pick)**
What: build an **offline walkability grid per corridor map** from the pret disassembly — no `env/`
changes: map dimensions + block layouts are `maps/*.blk` (W×H block ids), each block's 2×2 quadrant
collision comes from the tileset's paired `collision.bin` indexed by block id, and walkability per
collision id from `data/collision/collision_permissions.asm` (COLL_FLOOR walkable, WALLTILE not, ledges
directional). `agents/rl/map_layout.py` already ingests pokecrystal map-size/connection data for the
same 8 corridor maps and defines the coordinate un-swap. A* over that grid + warp/connection edges gives
a `navigate_to` tool that *executes* the button presses (reusing `move`'s loop, stopping on battle/map
change). The LLM then only chooses *where* to go; the tool does *how*. Why: this is the single
intervention every successful LLM-Pokémon system converged on — Claude Plays Pokémon's navigator tool,
Gemini Plays Pokémon's Pathfinder agent ("one-shot the Rocket Hideout B3F maze" after days stuck), and
GPP's Crystal completion. Our headline failure ("fights but can't navigate") is exactly what it removes.
Cost: the largest LLM-side item — a grid-extraction script (ROM files are text in the pret repo;
Silver's Johto overworld matches pokecrystal's for these maps, and `pret/pokegold` exists for exact
data) + an A* (~150 lines) + tool wiring; ~1 day. Verify the grid against the walkability probe on ~50
random tiles before trusting it. Evidence: https://blog.jcz.dev/the-making-of-gemini-plays-pokemon,
https://michaelyliu6.github.io/posts/claude-plays-pokemon/, https://github.com/pret/pokegold,
https://github.com/pret/pokecrystal/wiki/Add-a-new-tileset (block/collision format).

**L2. Fix the swapped coordinate labels (trivial, prerequisite to everything).**
What: in `agents/llm/perception.py`, report `({local_y}, {local_x})` as "(x, y)" (i.e. un-swap per the
`ram_to_image_px` convention) and state the axis convention in the prompt ("x grows east, y grows south;
'up' decreases y"). Audit `memory.py`/stuck-detection (internally consistent, can stay swapped) and any
direction hints. Why: every past coordinate inference the model made was on swapped axes; this is a
5-line fix that all other techniques depend on. Cost: minutes. Evidence: `env/ram_reader.py` lines 15–16
vs `agents/rl/map_layout.py:217–221`; coordinate-grounding was decisive in frontier-lab harnesses —
https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG/research-notes-running-claude-3-7-gemini-2-5-pro-and-o3-on.

**L3. Fog-of-war ASCII minimap + "reachable unseen tiles" in the prompt.**
What: from the same L1 collision grid, render a small ASCII crop (~15×11) around the player each
overworld turn: walls `#`, walked tiles `.`, unexplored walkable `?`, player `@`, plus a "reachable
unexplored tiles: (x,y)…" list and the current leg's target direction. Why: GPP's harness kept exactly
two spatial components as *essential* (fog-of-war tile memory, goals) while dropping visual overlays;
research-notes found coordinate/passability annotation dramatically cut wall-walking. It also gives the
model something better than a raw screenshot for its weak vision. Cost: ~100 lines once L1's grid
exists. Evidence: https://blog.jcz.dev/the-making-of-gemini-plays-pokemon,
https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG/research-notes-running-claude-3-7-gemini-2-5-pro-and-o3-on.

**L4. Macro-waypoint prompting (route as a landmark checklist).**
What: encode the corridor as an ordered list of legs with map + target coordinates (New Bark exit →
Route 29 west → Cherrygrove north → Route 30 gate → Route 31 → gatehouse → Violet → gym door → Falkner),
tracked in the harness (not by the model): current leg auto-advances on map change. The per-turn prompt
contains only the *current* leg goal ("you are on Route 29 at (x,y); reach the west exit around (0, 8);
call navigate_to"). Why: removes long-horizon planning from the model entirely — the known failure mode
of small models is plan retention, and GPP needed goals re-inserted after every context reset even for
Gemini 2.5 Pro. With L1+L4 the LLM's residual job is battles (which it already wins) + leg-local
decisions. This is fair for the comparison: the RL side got its route knowledge via reward waypoints.
Cost: ~50 lines + prompt edits in `agents/llm/config.py`. Evidence: GPP goal system (three-tier,
re-inserted) — https://blog.jcz.dev/the-making-of-gemini-plays-pokemon.

**L5. Persistent scratchpad memory + periodic summarization.**
What: extend `ShortTermMemory` with a harness-maintained note ("legs completed: …; battles won: …;
blocked directions at (x,y): …") + a rolling summary every ~30 turns, always prepended. Why: qwen3-vl's
effective context is small and each turn is currently near-stateless; GPP found periodic summarization +
goal persistence prevented loop-regression past long horizons. Lower priority than L1–L4 because the
macro-waypoint harness already externalizes most state. Cost: ~60 lines. Evidence:
https://blog.jcz.dev/the-making-of-gemini-plays-pokemon (context resets every ~100 turns; notepad).

**L6. Alternative local model (only if qwen3-vl still stalls with L1–L4).**
What: bench the already-pulled `gemma4:latest` (9.6 GB, vision + tool calling) against `qwen3-vl:8b`
(6.1 GB) on a scripted 20-turn corridor snippet: tool-call validity rate, empty-response rate, leg
completion. Notes: `llama3.2-vision:11b` (pulled) has **no tool support in Ollama**
(ollama/ollama#8345) — usable only with prompt-embedded JSON, an extra failure mode, so it is the last
resort; gemma-family tool calling in Ollama is workable but historically less native than qwen's
(community `gemma3-tools` variants exist for a reason). qwen3-vl remains the default: agentic
function-calling is an explicit design target for the Qwen-VL line. Cost: ~2h bench + config switch.
Evidence: https://github.com/ollama/ollama/issues/8345, https://ollama.com/library/gemma3,
https://medium.com/google-cloud/function-calling-with-gemma3-using-ollama-120194577fa6,
https://qwen.readthedocs.io/en/latest/framework/function_call.html.

---

## 4. Proposed attempt schedule (5 RL + 5 LLM)

Eval protocol (both paradigms): N≥10 episodes (RL) / N≥3 runs (LLM, ~40 min each) from
`saves/egg_delivered_clean.state`, report `badge_rate` + `max_waypoint` (0–5 per `WAYPOINT_ORDER`).

### RL

| # | Technique(s) | Config sketch | Kill criteria |
|---|---|---|---|
| RL-1 | **R1 + R4** (corridor confinement + waypoint-scored frontier) | warm `agent_079@130M`, explore 4.0, anchors+frontier on, 60M | `nav/reach_route31` still 0.0 at 40M → stop (confinement alone insufficient warm) |
| RL-2 | **RL-1 + R2** (staged saves + event-scaled episode budget) | same base; curriculum adds `crossing/route31/violet_city` envs; budget 16k + 16k/waypoint | no *new* segment (reach_violet_west==0) after 60M → stop |
| RL-3 | **R3 cold rebuild** (visited-coords obs, de-transposed) + R1+R2+R4 | COLD, full stack, 100M gate | reach_cherrygrove <0.5 at 30M (worse than 079's pace) → stop |
| RL-4 | **R5 scale-up** of the best of RL-1..3 | extend/warm best checkpoint to 200–480M | segment pace: no new waypoint ordinal consolidated in any 60M window → stop |
| RL-5 | **R6 handoff** (corridor policy → `agent_087_final` at gym door) | eval-harness only; use best corridor policy | if best policy's `reach_gym` = 0 → composed badge impossible → report corridor-best honestly |

Success gate: eval `badge_rate > 0` from egg_delivered_clean (any attempt) = task solved; else deliver
best `max_waypoint` + the characterization.

### LLM

| # | Technique(s) | Config sketch | Kill criteria |
|---|---|---|---|
| LLM-1 | **L2** only (coordinate fix), baseline re-run | corridor start, confinement off, 500 steps | expected to fail navigation; kill after 1 run if max_waypoint < 1 (calibration run) |
| LLM-2 | **L1 + L2** (A* `navigate_to` tool) | grid verified vs probe first; tools: navigate_to/press/get_state/wait | tool executes but agent never calls it sensibly for 2 runs → go to LLM-3 (harness picks goals) |
| LLM-3 | **L1 + L4** (macro-waypoint harness drives leg goals) | leg checklist in harness; prompt = current leg only | <2 legs completed per run across 2 runs → inspect: battle deaths → level/heal handling; else LLM-4 |
| LLM-4 | **+ L3 + L5** (ASCII minimap + scratchpad) on top of LLM-3 | full scaffold | max_waypoint ≤ LLM-3's after 2 runs → scaffold saturated → LLM-5 |
| LLM-5 | **L6** best scaffold × gemma4 (and prompt-JSON llama3.2-vision only if gemma4 regresses) | 20-turn bench first, then best model full run | bench tool-validity <70% → skip model; end of budget |

Success gate: badge in any run = headline upgrade ("LLM navigates *with a spatial compass*"); Falkner
reached but lost = report fight gap; still door-bouncing with full scaffold = the finding stands,
now with the strongest possible caveat coverage.

---

## 5. Sources

- PWhiddy, *Pokemon Red Experiments* (V2: coordinate-based exploration replaced KNN; reaches Cerulean) — https://github.com/PWhiddy/PokemonRedExperiments
- Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden, *Pokémon Red via Reinforcement Learning* (72×80 grayscale ×3 frames + **48×48 visited-coords binary crop** + state vector; R = event +2 / nav +0.005 per new coord / heal / saturated level reward; **dynamic episode budget 10,240 + 2,048 per event**; γ=0.997, 32 workers, 2048-step horizons, no entropy bonus; nav-reward ablation fails everything, ×10 nav reward over-explores; heal-reward exploitation; Cut unsolved) — https://arxiv.org/abs/2502.19920
- Rubinstein et al., *pokerl* book + `pokemonred_puffer` (beat the full game, <10M params; PRET + PyBoy introspection; swarm/state-migration lineage) — https://drubinstein.github.io/pokerl/ · https://github.com/drubinstein/pokemonred_puffer
- Boey fork (multi-channel map observation lineage past badge 2) — https://github.com/CJBoey/PokemonRedExperiments1
- pret disassemblies: Gold/Silver — https://github.com/pret/pokegold · block/collision format (`maps/*.blk`, per-block 2×2 `collision.bin`, `data/collision/collision_permissions.asm`) — https://github.com/pret/pokecrystal/wiki/Add-a-new-tileset · map editor showing the same data — https://github.com/Rangi42/polished-map
- *The Making of Gemini Plays Pokémon* (fog-of-war JSON tile map, Pathfinder agent, three-tier goals re-inserted after context resets, ~100-turn summarization, critique model; essentials-vs-removed table) — https://blog.jcz.dev/the-making-of-gemini-plays-pokemon · Crystal completion + Gemini 3 comparison — https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal
- Research notes: Claude 3.7 / Gemini 2.5 Pro / o3 on Pokémon (coordinate overlays + passability coloring cut wall-walking; models cannot self-generate usable maps; hallucination loops) — https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG/research-notes-running-claude-3-7-gemini-2-5-pro-and-o3-on · Claude-vs-Gemini navigator scope comparison — https://www.lesswrong.com/posts/7mqp8uRnnPdbBzJZE/is-gemini-now-better-than-claude-at-pokemon
- Claude Plays Pokémon harness overview (pathfinding tool executes optimal button sequences) — https://michaelyliu6.github.io/posts/claude-plays-pokemon/
- *PokéLLMon* (battles only: in-context reinforcement from battle feedback, knowledge-augmented generation, consistent action generation vs panic-switching; human-parity in battles — navigation out of scope, consistent with our "fights but can't navigate" split) — https://arxiv.org/abs/2402.01118
- Ollama local models: llama3.2-vision has no tool support — https://github.com/ollama/ollama/issues/8345 · gemma3/gemma-family vision + function calling — https://ollama.com/library/gemma3 · https://medium.com/google-cloud/function-calling-with-gemma3-using-ollama-120194577fa6 · Qwen-VL function-calling docs — https://qwen.readthedocs.io/en/latest/framework/function_call.html
