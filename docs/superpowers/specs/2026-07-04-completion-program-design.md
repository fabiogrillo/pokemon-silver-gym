# Pokémon Silver Gym — Completion Program (GIF → LinkedIn → Final Attempt → Refactor → Ship)

## Context

The core project is done: RL `agent_087` beats Falkner 100% from the gym slice; the LLM agent
(qwen3-vl:8b, ReAct) wins battles but cannot navigate to Falkner — the documented RL-vs-LLM finding
(24 tests green, eval + comparison suite committed). What remains is turning this into a public,
portfolio-grade artifact: a comparison GIF, a LinkedIn post, one last research-driven push on the
full corridor (New Bark → badge) for BOTH paradigms, a Whidden-style map visualization on the real
game map, a "human-made" repo refactor, and a dual-mode Docker demo.

**Decisions locked with Fabio (2026-07-04):**
- GIF = highlight montage (not synced runs); LLM thoughts shown shortened-for-legibility (real, only cut)
- Refactor = files + code style only, git history untouched; runs AFTER the final attempt
- Final attempt budget = **5 attempts per paradigm**, LLM local-only via Ollama (16 GB constraint)
- Docker = docker-compose with an Ollama sidecar service (turnkey `docker compose up`)
- Map-viz background = the REAL Johto map (extracted/stitched), precisely calibrated

**Phase order (approved deviation from the original list):** map-viz (D) moves BEFORE the final
attempt (C) because the trajectory overlay is the best debugging tool for the RL corridor attempts.
Refactor moves after C so the repo is cleaned once.

> A → B → D → C → E → F

---

## Phase A — Comparison GIF (highlight montage)

**Goal:** `assets/comparison.gif` (+ `.mp4` for LinkedIn, which compresses GIFs badly), a few
seconds long: left panel labeled **RL** (PPO), right panel labeled **LLM — qwen3-vl:8b**, right
panel with the (shortened, real) reasoning text below the screen.

**Build:**
1. **Record RL footage** — reuse the capture path of `agents/rl/make_gif.py` (`build_vec_env` +
   `agent_087_final.zip` from `violet_city_gym.state`), saving full-res frames + step metadata.
2. **Record LLM footage** — instrument the existing LLM loop (`agents/llm/run.py` `on_step`
   callback already exists) to also dump the screen frame per step alongside the JSONL thought
   trace. One recording run from `violet_city_gym.state` (~30-40 min at ~6 s/step; the stall
   happens after the first battle so a few hundred steps suffice).
3. **Select highlights** — 4 segments picked by step index from the traces: RL navigating up the
   gym, RL beating Falkner (badge moment), LLM winning its first battle, LLM bouncing at the door
   with its (real) thoughts visible. Segment indices live in a small config dict in the script.
4. **Compose** — new script `agents/make_comparison_gif.py` (PIL + imageio, both already deps):
   2x-upscaled GB screens (320×288 each) side by side, header bar with labels, caption strip under
   the right panel rendering the shortened thought for that segment; ~12–15 s total at 10–12 fps.

**Verify:** open the GIF, check readability of thought text at LinkedIn feed size (~half screen),
file size < 5 MB for the GIF (LinkedIn limit 8 MB), MP4 as primary upload.

## Phase B — LinkedIn post

English, no emoji, ~180–250 words, human voice. Per FAANG repo branding rules
(`PROJECT-INSTRUCTIONS.md`): technical showcase / learning journey ONLY — no job-search language
(Fabio is employed at Ermes). Content: joy for the milestone (RL 100% badge), the honest finding
(the LLM fights but has no spatial compass — RL 100% vs LLM 0% on the same slice), one concrete
technical nugget (e.g. the button-edge `settle` frames bug), and how Claude helped reverse-engineer
the Game Boy RAM bit layout (badge bit, battle flag, coordinates). Ends with the GIF + repo link.

Deliverable: post text in chat for Fabio to publish manually; also saved to the FAANG vault
(`Career/` or wherever Fabio keeps post drafts — confirm at execution).

## Phase D — Whidden-style map visualization (real map)

**Goal:** trajectory + heatmap overlays on the REAL New Bark → Violet map, quality good enough for
the blog post / YouTube video, and useful as a live debugging tool for Phase C.

**Build:**
1. Source the real map: stitched Johto overworld maps (spriters-resource / pokecrystal map data);
   assemble the corridor into `assets/maps/johto_corridor.png`.
2. Calibrate `agents/rl/map_layout.py`: per-map (bank, number) → pixel offset on the stitched
   canvas, verified with known landmarks (lab door, gym door, route gates). This is the precise
   calibration previously deferred.
3. `agents/rl/visualize_map.py` already renders PNG + reveal-GIF on top of `ASSET_BG` when present
   — it should mostly Just Work once the asset + offsets are right.

**Verify:** overlay of `agent_087` gym run + an egg-delivered corridor run lands on the correct
buildings/routes pixel-accurately.

## Phase C — Final attempt: 5 RL + 5 LLM attempts, egg-delivered → badge

**C0 — Research sweep (before any run):** study online sources (PWhiddy pokerl-map-viz + Pokemon
Red RL lineage, pokecrystal disassembly / DataCrystal RAM maps, PokéLLMon, Claude Plays Pokémon
writeups, YouTube RL-plays-Pokemon projects, relevant blog posts) + re-read `training_log.md`
(what the 87-agent arc already ruled out). Output: a short findings doc ranking candidate
techniques per paradigm, e.g.
- RL: goal-conditioned / map-id-aware observations, curriculum from intermediate save states
  (already have `crossing/route31/violet_city` states), exploration bonuses (RND-style), action
  masking, longer horizon training, reward shaping on corridor waypoints
- LLM: collision-map → ASCII minimap in the prompt, a `navigate_to(x,y)` pathfinding tool (A* on
  RAM collision data — the "spatial compass" it demonstrably lacks), persistent scratchpad memory,
  alternative local vision models fitting 16 GB, prompt/tool redesign

**C1/C2 — Attempts:** each attempt = hypothesis → change → run → eval → dated entry in
`training_log.md`. RL runs may be multi-day (GPU, unattended; monitored via TensorBoard + Phase D
overlays). LLM attempts are interactive. Hard stop: 5 per paradigm; if unsolved, record final
progress honestly and stop — that result is publishable either way (per the roadmap contingency).
This phase can run in /loop mode (self-paced monitoring) once a training run is launched.

## Phase E — Refactor to a lean, human-looking repo

Files + code only; git history untouched. Remove: `docs/superpowers/` (plans/specs), internal AI
artifacts, dead/legacy scripts (`agents/rl/train_mlp.py`, `evaluate.py` vs `evaluate_cnn.py`
duplication, stale notebooks), giant untracked logs stay untracked but `.gitignore` is tightened.
Rewrite over-verbose docstrings/comments into terse professional ones; normalize naming; keep the
tests. Target: a repo a strong engineer would plausibly have written by hand — small, sharp, no
scaffolding smell. (Tracked size is already only 5.5 MB; this is about content, not weight.)

## Phase F — README + Docker compose + ship

- README rewrite: project story, RL-vs-LLM results table (from `agents/comparison.py`), comparison
  GIF + map-overlay GIFs of BOTH agents, quickstart, docker instructions.
- `docker-compose.yml`: `app` service (existing image, `MODE=rl|llm` env switch selecting
  `agents.rl.play` vs the LLM runner) + `ollama` service (official image, pulls qwen3-vl:8b on
  first start, named volume for weights). ROM still mounted by the user (legal).
- Optional extras (nice-to-have, Fabio decides): GitHub social-preview image from the GIF, CI
  badge running the test suite, a GitHub Release attaching `agent_087_final.zip`.

## Cross-cutting (project rules)

- Chat Italian, all repo artifacts English.
- After each phase: sync README/docs + FAANG `Planning/Active-Roadmap.md` + current weekly log.
- Each phase gets its own implementation plan via superpowers:writing-plans (this program is too
  big for one plan); execution starts with Phase A.

## Verification (end-to-end)

- A: GIF/MP4 render, readable, size-compliant. B: post approved by Fabio. D: pixel-accurate
  overlays. C: 10 documented attempts (or success earlier); eval JSONL for every attempt. E: tests
  still green after cuts; repo reads clean. F: `docker compose up` runs both modes on a clean
  machine; README accurate.

## Next steps after approval

1. Save this design as the spec: `docs/superpowers/specs/2026-07-04-completion-program-design.md`
   (it will be removed in Phase E with the rest of docs/superpowers — it has served its purpose by
   then), commit.
2. Invoke superpowers:writing-plans for **Phase A** and start there.
