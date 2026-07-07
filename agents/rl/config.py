# RUN IDENTIFIER — bump for each new training run.
# Drives: checkpoint subdir (runs/checkpoints/{RUN_NAME}/),
#         checkpoint filename prefix ({RUN_NAME}_<step>_steps.zip),
#         TensorBoard log name (runs/{RUN_NAME}_<N>/).
#
# NAMING CONVENTION (since 2026-06-12): progressive "agent_NNN" (PWhiddy-style).
# The full historical mapping (PPO_N / PPO_CNN_N* → Agent NNN) lives at the top of
# training_log.md. The currently-live run still uses its legacy name (it was launched
# before the rename); the NEXT run is agent_046.
RUN_NAME = "agent_090"    # CORRIDOR FINAL ATTEMPT — RL-3 (R3 + R1 + softened R2 + R4), per
                          # docs/superpowers/specs/2026-07-06-final-attempt-findings.md §2-R3/§4 and
                          # docs/superpowers/plans/2026-07-07-rl3-visited-obs.md. RL-3 HYPOTHESIS: the
                          # PPO_CNN_10e ablation (see this file's historical 10e-era comments / findings
                          # §2) concluded the visited-coordinates observation was NOT the paper's
                          # (arXiv:2502.19920 §II-C) indispensable input — but that signal was drawn
                          # TRANSPOSED at the time (env/ram_reader.py's local_x/local_y hold wYCoord/
                          # wXCoord swapped vs TRUE x/y; see memory note "RAM reader x/y swapped"), so
                          # the crop's rows/cols didn't track the player's real movement axes. 10e's
                          # verdict was a BUG ARTIFACT, not evidence the signal itself hurts. `VISITED_OBS
                          # =True` re-adds it as a proper, DE-TRANSPOSED "visited" Dict-obs key (gated
                          # env feature landed commit 9ea5e9e, Task 1; regression-tested against a
                          # transposed reimplementation) — a genuinely NEW input vs every prior corridor
                          # run, hence COLD (obs-space change invalidates any warm checkpoint).
                          # R2 SOFTENING (agent_089 post-mortem, training_log.md "Agent 089 — RL-2
                          # verdict"): RL-2's kill criterion fired AND REGRESSED (`nav/reach_route31`
                          # still 0.0 at 29.3M, `nav/ep_max_waypoint` COLLAPSED 2.0→0.0 — agent_088 had
                          # already consolidated that far). Post-mortem hypotheses: (a) the 16,384-step
                          # base budget truncated start episodes before they could RE-EARN the
                          # consolidated wp-0→2 behavior (catastrophic-forgetting pressure instead of
                          # frontier focus); (b) halving pure-start envs (6/12 to staged saves +
                          # frontier absorption) cut the nav gradient share too far. This run softens
                          # BOTH per the post-mortem's own prescription: `DYN_BUDGET_BASE=32768` (2×,
                          # via the Task-1 override param — gentler truncation) and `CURRICULUM_STATES_CNN`
                          # restored to a pure-start MAJORITY (9/12 egg_delivered_clean, 1 each of
                          # crossing/route31/violet_city — down from 089's 6/12) so staged-save gradient
                          # is a supplement, not a co-equal split. R1/R4 UNCHANGED (CONFINE_TO_CORRIDOR
                          # =True, EXPLORATION_SCALE=4.0, FRONTIER_ENABLED=True, waypoint-scored).
                          # COLD (INIT_FROM_CHECKPOINT=None — visited_obs changes the obs space, no
                          # compatible checkpoint exists), 100M budget (more headroom than RL-1/2's 60M
                          # since this is a from-scratch structural change, not a warm continuation).
                          # KILL CRITERION (per Global Constraints / findings schedule): `nav/
                          # reach_cherrygrove < 0.5` at 30M → stop (a COLD run must at minimum re-clear
                          # the basic nav bar the two warm attempts took for granted).
# (prior 089) CORRIDOR FINAL ATTEMPT — RL-2 (RL-1 + R2), per
                          # docs/superpowers/specs/2026-07-06-final-attempt-findings.md §4 and
                          # docs/superpowers/plans/2026-07-06-rl2-staged-budget.md. RL-1 verdict
                          # (training_log.md "Agent 088 — RL-1 verdict"): R1 (CONFINE_TO_CORRIDOR) + R4
                          # (waypoint-scored frontier) cured the 079 off-path stall (Dark Cave farming
                          # is gone) and the START policy consolidated Cherrygrove + the Route-30 gate
                          # (wp 0→2 by 37M) — but plateaued there (`nav/reach_route31`=0.0, 40M kill,
                          # `nav/ep_max_waypoint` flat at 2 from 37M→41M) even though the frontier side
                          # already spans the WHOLE corridor to the gym (`front/ep_max_waypoint`=5.0).
                          # Hypothesis: the gap is start-POLICY CONSOLIDATION lagging the archive, not
                          # missing exploration. R2 attacks that directly with two structural, additive
                          # tricks: (a) `CURRICULUM_STATES_CNN` now dedicates curriculum slots to
                          # on-corridor intermediate saves (`crossing`/`route31`/`violet_city` — all
                          # egg-delivered, same story flags, per findings §2 R2) so the policy gets
                          # direct gradient AT the lagging Route-30→31→Violet segment every rollout,
                          # not just via the shared frontier archive. (b) `DYNAMIC_EPISODE_BUDGET=True`
                          # (env feature landed commit 2c6ee45, Task 1): start/curriculum episodes begin
                          # capped at DYN_BUDGET_BASE (16384 steps) and only earn a longer cap
                          # (min(MAX_STEPS, 16384*(1+waypoint_ordinal))) by reaching a NEW waypoint —
                          # concentrates the gradient on the frontier segment instead of re-walking New
                          # Bark every episode (Pokémon-Red paper trick, arXiv:2502.19920). WARM from
                          # `agent_088@39999936` (the RL-1 checkpoint at its kill point — Cherrygrove +
                          # Route-30-gate capable), 60M budget, R1/R4 flags UNCHANGED (CONFINE_TO_CORRIDOR
                          # =True, EXPLORATION_SCALE=4.0, FRONTIER_ENABLED=True).
                          # KILL CRITERIA (findings §4, RL-2 row + tightened early gate): no new segment
                          # (`nav/reach_violet`==0.0) after 60M → stop; ALSO stop early at 30M if
                          # `nav/reach_route31` is still 0.0 (tighter than RL-1's 40M gate — staged
                          # resets sit ON Route 31/Violet already, so a crack should show much sooner
                          # than a pure archive-mediated hand-off would).
# (prior 088) CORRIDOR FINAL ATTEMPT — RL-1 (R1 + R4), per
                          # docs/superpowers/specs/2026-07-06-final-attempt-findings.md §4. Hypothesis: the
                          # agent_079 recipe (bidirectional Go-Explore frontier + Violet curriculum anchors +
                          # EXPLORATION_SCALE=4.0) already connects the corridor archive end-to-end and pushes
                          # the START policy past the Phase-1 wall segment-by-segment (Cherrygrove 1.0 @28M,
                          # Route-30-gate 1.0 @37M) — but then PLATEAUS at Route 30 (`nav/reach_route31`=0 for
                          # ~83M) because EXPLORATION_SCALE=4.0 lures the policy into dead-end maps (Dark
                          # Cave/Sprout Tower: 235 cave cells harvested). R1 generalizes the agent_087 lesson
                          # (CONFINE_TO_GYM: 40%→100% badge by REMOVING the wander OPTION, not reward tweaks):
                          # new gated `CONFINE_TO_CORRIDOR` flag ends the episode the instant the agent leaves
                          # CORRIDOR_LEGAL, so off-path maps stop being places where episodes go to die. R4
                          # (frontier re-scored by waypoint ordinal, already landed — commits 25188af/c0dbebb)
                          # concentrates ε-greedy resets at the leading edge instead of uniformly. WARM from
                          # `agent_079@129999792` (Route-30-capable), 60M budget (RL-1's gate, not 079's
                          # own 480M). Every other knob below is REVERTED from agent_087's gym-slice state
                          # back to the agent_079 corridor recipe (recovered from training_log.md's agent_079
                          # entry + this file's own historical comments) — deltas documented inline and in
                          # training_log.md's Agent 088 entry.
                          # KILL CRITERION (findings §4, RL-1 row): `nav/reach_route31` still 0.0 at 40M →
                          # stop (confinement alone insufficient even warm-started).
# (prior 087) GYM TASK v5 — STRUCTURAL FIX: CONFINE_TO_GYM=True — leaving GYM_MAP ends the
                          # episode, so the agent CANNOT wander out to wild-grind (the stable basin that
                          # capped 083-086 at ~40%). Forces it to solve the gym. WARM from 085@5M (40%),
                          # pure 12×gym, engage reward kept, lr 5e-5, flat entropy 0.005, 5M. One new
                          # variable vs 086 (the confinement).
                          # (prior 086) STABILIZATION (lr 5e-5, flat entropy): best 30% — FAILED to beat 085's
                          # 40%. Drift is structural (wander-grind basin), not optimizer → hence the confinement.
                          # (prior 085) GYM TASK v3 — NEW one-shot "gym engage" reward. Result: 40% badge @5M
                          # (4× vs 083) but drifted after 5M. Best ckpt agent_085_4999980 (40%).
                          # (prior 084) GYM TASK v2 — MIXED 6×gym + 6×falkner_battle. Result: eval badge 0% (all
                          # ckpts) — worse than 083; reinforced battle-win but still wandered (no nav signal).
                          # (prior 083) GYM TASK: all 12 envs from saves/violet_city_gym.state → navigate to Falkner.
                          # EXPLORATION_SCALE=0 (anti-wander, per 071), FRONTIER off, WARM from 050@10M
                          # (gym-capable base). Result: transient train badge=1.0 (noisy) but eval=10% (wander).
                          # (prior 082) RL v2.6 VALIDATION: whitelist REVERTED (it failed — 081 stuck at Cherrygrove at
                          # 60M, worse than 079) + MAX_STEPS 65k→131k (2× longer episodes). Hypothesis: 079
                          # reaches Route 30 but RUNS OUT OF STEPS before Route 31; more steps/episode may let
                          # the Route-30 policy push through. WARM from 079@130M (same reward now, so no
                          # confound — only MAX_STEPS changed), 20M test. Watch nav/reach_route31.
# (prior 081) RL v2.5 LONG RUN (~2 days): off-path exploration whitelist, COLD. The 080 warm
                          # validation was inconclusive (only ~10M settled after warm re-adaptation, too short to
                          # crack a new segment) but the whitelist is mechanically sound (cave cells 235→89,
                          # Route 30 held). This is the clean test + real objective attempt: COLD generalist,
                          # corridor-only exploration, curriculum + bidirectional frontier, explore 4.0, 480M.
                          # WATCH nav/reach_route31: if it cracks, the corridor-restricted reward let it through;
                          # if it replateaus at Route 30 past ~50M, the (26,1)→(26,2) crossing is a specific wall.
# (prior) RL v2.4 VALIDATION: off-path exploration whitelist (no reward for caves / Sprout
                          # Tower / houses / side routes — only the corridor pays). agent_079 plateaued at
                          # Route 30 (reach_route31 = 0 for ~83M) because EXPLORATION_SCALE=4 lured it into
                          # Dark Cave & other dead-end maps (235 cave cells harvested). SHORT warm-start run
                          # (20M) from agent_079@130M to test whether removing the off-path lure lets it cross
                          # into Route 31 and beyond; if yes → relaunch the long run. (prior 079:)
                          # RL v2.3: v2.2 (curriculum + bidirectional frontier) + EXPLORATION_SCALE 1.0→4.0.
                          # agent_078 STALLED at 15M: nav/reach_cherrygrove flat 0, and the archive's MIDDLE
                          # (Cherrygrove 26_3, Route 30 26_1) stayed EMPTY — both fronts stalled (Route 29 at
                          # 40 cells, Route 31 at 22), visited_tiles never grew → the policy doesn't explore
                          # WEST, it loops/grinds a fixed zone. Fix: make a new-MAP crossing (+0.4 post at
                          # scale 4) out-earn the Route-29 wild-battle grind, pulling the policy across the
                          # bottleneck (Whidden's dense-exploration-as-primary-driver thesis). One variable
                          # changed vs 078 to isolate its effect; curriculum + frontier kept.
                          # (prior) RL v2.2: forward CURRICULUM (Violet anchors) + FRONTIER = bidirectional Go-Explore.
                          # agent_077 (frontier, NO seed) FAILED the same way: frontier can only reset into
                          # cells it has discovered, and it only ever discovered New Bark + east Route 29
                          # (rollout: westmost local_x=6, never Cherrygrove; archive stuck at 72 cells). The
                          # frontier can't CROSS the Route-29→Cherrygrove bottleneck unaided. Fix: seed forward
                          # diversity via curriculum anchors at Violet (the only post-delivery states we have):
                          # those envs explore SOUTH and harvest Route-31/gatehouse cells, start envs explore
                          # NORTH, and the shared frontier ratchets both fronts until the corridor connects.
                          # (prior) RL v2.1: GENERALIST + FRONTIER (Go-Explore) reset, scaled to a ~2-day run.
                          # agent_076 (cold, no frontier, 12 envs) CONFIRMED the Route-29 exploration wall:
                          # nav/reach_cherrygrove flat 0.0 through 8.7M — 12-env brute force locks into the
                          # wild-battle grind and never reaches Cherrygrove. Fix (per training_log agent_067):
                          # manufacture state diversity via the frontier archive so 12 envs get the explorative
                          # reach of many more — the "poor man's 40 envs". Same forward task (egg-delivered →
                          # Violet → Falkner), exploration ON, COLD. WATCH the ~15M gate: nav/reach_cherrygrove
                          # must rise (>=0.3-0.5) and nav/reach_route31 > 0, else abort and reseed/retune.

# ── (076, superseded) RL v2 RE-BASELINE (Whidden-style): one GENERALIST from egg-delivered, no frontier,
                          # 15M cold validation. Proved the Route-29 wall (above). Superseded by agent_077.

# ── (075, superseded) PHASE 2 Run 6: STABILIZE the Falkner fight (learn THEN commit). 072 (lr 1e-4/ent
                          # 0.02) learned the win (badge 0.49) then DRIFTED into bad moves; 074 (lr 5e-5/ent
                          # 0.01) too gentle to learn (stuck ~0.08). Fix: lr 7e-5 (intermediate) + ENTROPY
                          # SCHEDULE 0.02→0.005 over 8M (explore the battle-menu move early, then COMMIT so it
                          # stops drifting off the winning move) + CHECKPOINT_FREQ 1M (capture the peak; 072's
                          # 0.49 was missed by 5M-spaced saves). Warm 050@10M, 12×falkner_battle, expl cut.
                          # PROVED the isolated fight is solvable — badge_rate hit 0.49 — but then DESTABILIZED
                          # (badge→0, battles_lost 0.07→0.66; over-explored into bad moves). Fix: warm from
                          # 072@5M (the ~0.49-peak checkpoint) + LOWER lr 5e-5 + ent 0.01 → consolidate the
                          # winning move-policy stably instead of drifting off it. Same falkner_battle state,
                          # exploration cut. Should hold/climb badge_rate. (Not harmful chaining: same reward/
                          # state, just a gentler optimizer to stabilize.)

# ── (072, superseded) PHASE 2 Run 3: FORCE THE FIGHT. 070/071 showed the agent wanders out of the gym
                          # to wild-grind (cut-exploration insufficient). Fix: train from a save-state INSIDE
                          # the Falkner battle (saves/falkner_battle.state — created by running 050 until its
                          # opponent was Falkner's Pidgeotto lv9; Totodile lv15 @ full HP) → no wander option,
                          # the agent MUST select moves and win → badge. Warm 050@10M, lr 1e-4, ent 0.02
                          # (explore moves), exploration cut. Should consolidate FAST (lv15 vs lv9). If
                          # badge_rate → high, extend backward (reverse curriculum from earlier in the fight /
                          # the gym entrance). 12×falkner_battle.

# ── (071, superseded) PHASE 2 Run 2: CONSOLIDATE the Falkner fight (no wander-out). Run 1 (070) showed
                          # warm-050 is ~10% Falkner from the gym but wild-grinds (leaves the gym; 417 wins/20
                          # eps eval), and naive training ERODED it (badge 0.31→0.03). Fix: EXPLORATION_SCALE=0
                          # → new-tile/new-map pay nothing → no incentive to leave the gym → the FIGHT (gym
                          # damage + trainer + badge) is the only income → consolidate. Warm 050@10M, LOWER lr
                          # 5e-5 + ent 0.01 to PRESERVE the 10% base (not erode). 12×gym. Recommended lever.

# ── (070, superseded) PHASE 2 Run 1: beat FALKNER from the gym. Phase-1 from-start
                          # delivery is the project's characterized hard limit (11 runs); pivot to the gym
                          # fight from violet_city_gym.state (Totodile lv15 — already strong enough; egg
                          # delivered). The blocker (agent_050 eval) is FOCUS not strength: the agent wanders
                          # OUT of the gym to grind wild battles instead of fighting Falkner. Run 1: focused
                          # 12×gym training, warm 050 (beat Falkner ~6× in training), gym damage + trainer +
                          # badge rewards drive the fight. Watch front/badge_rate + gym_trainers + in-gym W/L;
                          # if it wanders out, run 2 adds anti-wander (cut exploration / penalize leaving gym).

# ── (069, superseded) CARRY SPECIALIST. The session proved "carry-nav" was always a
                          # frontier-reset artifact — no policy delivers from a standing carry state in eval,
                          # and shared-policy approaches collapse pickup. Fix: train a SINGLE-MODE specialist
                          # FROM carry states (8×mid_route30 Cherrygrove + 4×newbark_egg), FIXED-start so it
                          # MUST navigate the full Cherrygrove→Elm backtrack every episode (no near-lab
                          # artifact) → robust deliver-from-carry. Then (next step) hard-gate 053 (pickup) +
                          # this specialist by the egg flag → deliver from start. Warm 053 (map-nav base),
                          # NO frontier, NO marker (single carry mode), dense reward + breadcrumbs + delivery
                          # +30 teach south, ent 0.02 (explore past 053's N-habit), lr 1.5e-4.

# ── (068, superseded) SEEDED cold synthesis, REBALANCED 6→3 frontier. 067 (6/6 seeded) bootstrapped
                          # navigation (reach_cherrygrove 1.0, vs 066's 0.02 — seeding works!) and the start
                          # envs even reached route30_gate 0.85 (pickup about to develop), but then start nav
                          # COLLAPSED to 0: the 6 carry-mode (red-marker) frontier envs' southward gradient
                          # overwhelmed the northward (black-marker) pickup behavior — the marker didn't
                          # separate strongly enough at 6:6. Fix: 3/9 (proven 062 ratio) — keep the SEEDING
                          # (the nav-bootstrap key) but protect pickup with 9 start envs + only 3 carry envs.
                          # Else = 067: COLD, marker, dense, seeded from 065, lr 1.5e-4.

# ── (067, superseded) SEEDED cold synthesis. 066 (pure-start cold) hit the agent_019 wall: no state
                          # diversity → never learned directed navigation (reach_cherrygrove ~0.02 @26M,
                          # ep_rew plateaued). Fix: SEED the frontier archive with 065's 223 carry-state
                          # save-states (FRONTIER_SEED_FROM) so the frontier envs reset into the corridor
                          # from step 1 → state diversity that bootstraps map-navigation (the carry maps =
                          # the pickup-route maps, so it helps start envs too). Bumped frontier to 6/6 (a
                          # stuck cold needs diversity > pickup-reps; the marker keeps the heavier southward
                          # practice from collapsing pickup). Else = 066: COLD, marker, dense, lr 1.5e-4.

# ── (066, superseded by 067) COLD SYNTHESIS: egg-marker + dense southward reward. 065 PROVED carry-nav is
                          # learnable (dense reward → carry-depth 4) but pickup⟺carry-nav CANNOT coexist in
                          # one CNN with identical pre/post-pickup images (065 solved carry-nav, pickup
                          # collapsed to 0.02). The tension is REPRESENTATIONAL. Fix: COLD start + egg_marker
                          # (carrying=red/delivered=green/none=black corner) so the CNN SEES the mode and
                          # learns pickup (no-marker→go N) and carry-nav (red→go S, dense-rewarded) as
                          # SEPARATE modes from step 1 — no warm corruption (063's failure), no entrenched
                          # habit (065's). Cold ⇒ lr 1.5e-4 (8e-5 too slow from scratch). Keep dense reward,
                          # breadcrumbs, frontier 9/3 flat-tier terminate-on-delivery. THE synthesis of the session.

# ENV
ROM_PATH = "pokemon_rom.gbc"
STATE_PATH = "saves/start.state"  # Default single-env state (not used when CURRICULUM_STATES is set)
N_ENVS = 8  # Total parallel environments — must equal sum of counts in CURRICULUM_STATES

# Curriculum learning: (state_path, n_envs) pairs. Counts must sum to N_ENVS.
CURRICULUM_STATES = [
    ("saves/start.state",               2),  # learn the full path from scratch
    ("saves/mid_route30.state",         1),  # egg picked up, Cherrygrove area, Elm delivery ahead
    ("saves/route_31.state",            1),  # Route 31, test battle reward and map transition detection
    ("saves/before_elm_delivery.state", 2),  # naming done, Elm reward fires at first action
    ("saves/violet_city.state",         1),  # in front of Pokemon Center in Violet City, test map transition detection
    ("saves/violet_city_gym.state",     1),  # inside Violet City Gym, test battle reward and badge detection
]

# PPO - Principal Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2**13  # Number of steps to run in each environment per policy rollout
BATCH_SIZE = 64 # Minibatch size for updating the policy
N_EPOCHS = 10   # Number of epochs to update the policy
GAMMA = 0.999    # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
ENT_COEF = 0.08  # Entropy coefficient for exploration

# Training (MLP filone)
TOTAL_TIMESTEPS = 200_000_000 # Total number of timesteps to train on
CHECKPOINT_FREQ = 10_000_000    # Save a checkpoint every N timesteps
LOG_DIR = "runs/"            # Directory for TensorBoard logs and checkpoints
MODEL_DIR = "runs/checkpoints/"  # Directory to save model checkpoints

# ──────────────────────────────────────────────────────────────────────────
# CNN FILONE (PPO_19+) — used by train_cnn.py with pokemon_env_cnn.PokemonEnvCNN
# ──────────────────────────────────────────────────────────────────────────
N_ENVS_CNN = 12               # PyBoy CPU-bound (16 cores) — 12 envs improves throughput vs 8

# ── agent_047: EXPERIENCE INJECTION for the egg delivery (Go-Explore / puffer state-sharing style)
# After 6 reward-side attempts the delivery was never EXPERIENCED. 3 of 12 envs start from
# mid_route30.state (Cherrygrove, egg in hand, lab 2 transitions away) so delivery events enter the
# gradient within minutes. Differs from the ruled-out CNN_7 mixed curriculum: same quest corridor
# (same visual domain), and nav metrics are SPLIT by from_start (nav/* = start-state envs only,
# navret/* = return-leg envs) so regressions can't hide. (counts must sum to N_ENVS_CNN)
# agent_052: PURE START finishing run. Even at 8/4 the anchors' income captured the policy (probe:
# start envs farm Route 29 battles 99% of the time). All 12 envs on the real distribution; the
# quest value landscape (delivery, badge) is imprinted and decays slowly at lr 8e-5 — the bet is
# the start envs reach it before it fades. Win cap dropped to 2 in rewards.py.
# agent_054: still PURE START (no curriculum/anchors — those segregate, 047-051). The delivery is
# unblocked env-side instead: the pickup tile-reset re-arms ONLY the southern corridor (Route29→
# NewBark→Elm) so the return front's nearest income points south past spent Cherrygrove.
# agent_058: PURE START again — every anchor/curriculum run (047-051, 055-057) SEGREGATED the start
# policy (057: nav/egg_received pinned at 0 for 5M while the anchors delivered at ~1.0). The anchor
# approach is exhausted. 058 relies on the start policy's OWN exploration (entropy schedule) + the
# directional reset's southward income, warm from 053's pickup-100% base. No segregation possible.
CURRICULUM_STATES_CNN = [
    # agent_090: R2 SOFTENED (agent_089 post-mortem, training_log.md "Agent 089 — RL-2 verdict") —
    # 089's 6/12 pure-start split cut the nav gradient share too far AND its episodes never re-earned
    # the consolidated wp-0→2 behavior before truncating (`nav/ep_max_waypoint` collapsed 2.0→0.0).
    # Restored to a PURE-START MAJORITY (9/12 egg_delivered_clean.state, the nav/ success-gate
    # distribution) with only 1 env each on the 3 on-corridor staged saves (down from 089's 2 each) —
    # supplementary gradient at the lagging segment, no longer co-equal with the true start
    # distribution. All 3 staged saves are still egg-delivered/same story flags (not the foreign-state
    # curriculum segregation that ruled out 047-051). egg_delivered_clean is kept LAST so
    # FRONTIER_N_ENVS's dedicated last-4 ranks (8-11) land inside it — same archive-Go-Explore
    # mechanics as 088/089. Counts must sum to N_ENVS_CNN (12). is_start_env (env/pokemon_env_cnn.py)
    # only matches start.state/egg_delivered_clean.state, so the 3 staged slots log under front/ (not
    # nav/) by design — nav/ stays the clean, uncontaminated success gate.
    ("saves/crossing.state",            1),   # Route 29→Cherrygrove crossing — just past the old wp-2 plateau
    ("saves/route31.state",             1),   # Route 31 (post-gate) — direct practice AT the lagging segment
    ("saves/violet_city.state",         1),   # Violet end — explore south, harvest Route31/gym-approach cells
    ("saves/egg_delivered_clean.state", 9),   # true start (nav/ success gate); last 4 are the frontier envs
    # (prior 089) 2×crossing + 2×route31 + 2×violet_city + 6×egg_delivered_clean — REGRESSED (verdict above)
    # (prior 088) 3×violet_city + 2×violet_city_gym + 7×egg_delivered_clean (agent_079/078 bidirectional curriculum)
    # (prior 087/086/085/084/083 GYM TASK, superseded for this run) 12×violet_city_gym / mixed gym+falkner_battle
]

# ── agent_059: Go-Explore / frontier reset (start-continuous). A fraction of resets restart from a
# save-state SAMPLED from the shared archive (states harvested from the policy's OWN trajectory),
# instead of start.state. This is NOT an anchor/curriculum (which segregated, 047-051/055-057): the
# reset states are the policy's own visited cells (identical pixel distribution), and they are
# EPHEMERAL — continuously refreshed from the live policy, never a fixed foreign scene. nav/ metrics
# (true start episodes) stay the success gate; front/ metrics measure the frontier episodes.
# agent_066: stamp the egg-state marker into the obs image (carrying=red, delivered=green, none=black corner)
# so the CNN can SEE the pickup vs carry mode. Default-off elsewhere (the warm-marker run 063 failed); only the
# cold synthesis run uses it. Passed to the env via make_env.
EGG_MARKER         = False       # agent_069: OFF — the carry specialist is single-mode (always carrying), no
                                 # pickup/carry separation needed. (unchanged since; 079 also ran marker-off.)
EXPLORATION_SCALE  = 4.0         # agent_088: REVERTED to agent_079's value (was 0.0 for the 083-087 gym slice,
                                 # anti-wander per 071 — irrelevant here, there's no corridor to explore in the
                                 # gym task). At 4.0 a new-MAP crossing pays +0.4 post (> a capped battle win),
                                 # so progressing across map boundaries out-earns grinding — the driver that
                                 # connected the corridor archive end-to-end in 079. See CONFINE_TO_CORRIDOR
                                 # below for the fix to 079's failure mode (this scale luring off-path).
CONFINE_TO_GYM     = False       # agent_088: REVERTED — off for the corridor task (was True for 087's gym-only
                                 # slice). Leaving GYM_MAP must NOT end the episode; the gym is just the final
                                 # corridor waypoint here, not the whole task.
CONFINE_TO_CORRIDOR = True       # agent_088: R1 (final-attempt findings §2/§4, technique R1) — TURNED ON for
                                 # this run. Ends the episode if the agent leaves CORRIDOR_LEGAL (env/rewards.py)
                                 # — the structural fix for agent_079's off-path lure (Dark Cave / Sprout Tower
                                 # episodes died earning nothing, 235 cave cells harvested while
                                 # nav/reach_route31 stayed 0.0 for ~83M). Generalizes agent_087's
                                 # CONFINE_TO_GYM lesson (40%→100% badge via removing the OPTION, not reward
                                 # tweaks) to the corridor. Kill criterion: nav/reach_route31 still 0.0 at 40M.
# (prior) agent_079: BOOSTED 1.0→4.0 — at 1.0 the policy grinds Route 29 and never
                                 # explores west (078 stalled). At 4.0 a new-MAP crossing pays +0.4 post (> a
                                 # capped battle win), so progressing across the Route-29→Cherrygrove boundary
                                 # out-earns grinding (Whidden's dense-exploration-primary recipe). Dial back
                                 # later if it over-wanders past the gym. (1.0 = normal Phase-1 exploration.)
DYNAMIC_EPISODE_BUDGET = True    # agent_089: R2b FLIPPED ON, UNCHANGED for agent_090. "Earned episode budget"
                                 # (Pokémon-Red paper trick, arXiv:2502.19920) — start/curriculum episodes
                                 # begin capped at DYN_BUDGET_BASE and the cap only grows (up to MAX_STEPS)
                                 # when the episode reaches a NEW corridor waypoint. Frontier-origin
                                 # episodes are untouched (keep frontier_max_steps).
DYN_BUDGET_BASE    = 32768       # agent_090: R2-SOFTENING (agent_089 post-mortem, training_log.md
                                 # "Agent 089 — RL-2 verdict") — 089's 16384 base likely truncated start
                                 # episodes before they could RE-EARN the already-consolidated wp-0→2
                                 # behavior (`nav/ep_max_waypoint` collapsed 2.0→0.0 by 29.3M). Doubled to
                                 # 32768 via PokemonEnvCNN's `dyn_budget_base` override param (Task 1,
                                 # commit 9ea5e9e) — gentler truncation, gives episodes room to re-consolidate
                                 # before the earned-budget mechanic kicks in. (prior 16384, RL-2/agent_089)
VISITED_OBS        = True        # agent_090: R3 FLIPPED ON — gated "visited" Dict-obs key (48x48 uint8
                                 # crop of episode-visited tiles, de-transposed — see
                                 # env/pokemon_env_cnn.py's _visited_crop). Re-adds the signal the 10e
                                 # ablation removed while it was drawn TRANSPOSED (the ram x/y swap) — 10e's
                                 # verdict was a bug artifact, not evidence the signal itself hurts (see
                                 # RUN_NAME header). COLD-start-only change (obs-space change invalidates
                                 # any warm checkpoint trained without it — hence INIT_FROM_CHECKPOINT=None
                                 # below). (prior False, all runs through 089)
FRONTIER_ENABLED   = True        # agent_088: REVERTED — ON (was False for the bounded 083-087 gym task, which
                                 # has no corridor to explore). agent_079 ran the bidirectional Go-Explore
                                 # frontier (continued from 077/078) — reinstated here, now re-scored by
                                 # waypoint ordinal (R4, landed in frontier_archive.py commits 25188af/c0dbebb)
                                 # so ε-greedy resets concentrate at the leading edge instead of uniformly.
# (prior) agent_077: ON — Go-Explore reset manufactures state diversity to break the
                                 # Route-29 wall (12 envs alone can't; see training_log agent_067 seeding result).
FRONTIER_SEED_FROM = None  # agent_077: NO seed — agent_065's seeds are carry-mode states from a DIFFERENT task.
                           # (agent_088: unchanged — 079 also ran with no seed.)
                           # (prior) "runs/frontier_archive/agent_065"  # agent_067: seed the (cleared) archive with these
                                  # carry-state save-states at launch, to bootstrap the stuck cold start with
                                  # state diversity. None = no seeding (normal). Marker applies at obs-time, so
                                  # 065's marker-less save-states are compatible.
FRONTIER_N_ENVS    = 4            # agent_077: 4/12 frontier envs; the other 8 are pure-start so nav/ stays the
                                 # clean success gate. (prior) agent_068: back to 3/9 (067's 6/6 collapsed pickup).
                                  # Seeding (not env count) is what bootstraps cold nav, so 3 seeded frontier
                                  # envs still supply diversity while 9 start envs protect pickup. agent_060: #
                                  # of DEDICATED frontier envs (rest are pure start). 059
                                  # used p=0.5 on all 12 → southward gradient killed northward pickup
                                  # (nav/egg_received 1.0→0). 3/12 frontier keeps the start-task gradient
                                  # 3× the frontier gradient → pickup protected, archive still deepens.
FRONTIER_P         = 1.0          # reset prob FOR a dedicated frontier env (start envs are always p=0)
FRONTIER_MAX_STEPS = 8000         # agent_062: a FRONTIER episode ends when it delivers, OR is truncated at
                                  # this cap — so its gradient concentrates on the carry→deliver backtrack
                                  # instead of ~63k steps of off-task post-delivery wandering (which eroded
                                  # the start policy in 059-061). Start episodes still use the full MAX_STEPS.
                                  # ~8k steps is ample for the longest backtrack (north pocket → Elm).
FRONTIER_MAX_CELLS = 4000         # ~200KB/state → ≤ ~0.8GB on disk; frontier-weighted eviction
FRONTIER_CELL_K    = 4            # coord bucket size (tiles) for the cell key
FRONTIER_EPSILON   = 0.1          # uniform-sampling floor (diversity vs frontier exploitation)
FRONTIER_ROOT      = "runs/frontier_archive"   # per-run subdir <FRONTIER_ROOT>/<RUN_NAME>/ (cleared at start)

# CNN hyperparameters (Atari-style defaults adapted for Pokemon)
LEARNING_RATE_CNN = 7e-5      # agent_088: REVERTED to the 075/076-085 corridor-era value (was 5e-5, agent_086's
                              # gym-stabilization tweak — irrelevant here, that consolidated a drifting 40%
                              # gym peak). 079 ran with 7e-5 (083's entry: "Hyperparams = 075's gym-stabilize
                              # settings (lr 7e-5, ent schedule 0.02→0.005 over 8M)" — those settings were
                              # carried over unchanged from the 076-079 corridor runs into the gym pivot).
# (prior 086, gym-only) agent_086: LOWERED 7e-5→5e-5 to stabilize the 40% gym peak (085 drifted at 7e-5).
# (prior) agent_075: INTERMEDIATE lr — 1e-4 (072) learned but overshot into bad moves;
                              # 5e-5 (074) too gentle to learn. 7e-5 + the entropy schedule = learn then commit.
N_STEPS_CNN       = 2048      # smaller rollout (memory: 4 envs × 2048 × 72×80×12 ~ 450MB)
BATCH_SIZE_CNN    = 512       # bigger minibatch, exploits GPU
N_EPOCHS_CNN      = 4         # fewer epochs (more data per rollout)
ENT_COEF_CNN      = 0.02      # agent_088: REVERTED to the 075-085 schedule-start value (was 0.005 flat, agent_086's
                              # gym-only anti-drift tweak). ENT_COEF_CNN_END/ENT_ANNEAL_STEPS_CNN below are
                              # already at the 075-085 corridor-era values (0.005 / 8M) — unchanged.
# (prior 086, gym-only) agent_086: LOW + FLAT (start==end==0.005) — no exploration anneal, so the policy
                              # CONSOLIDATES the 40% winning behavior instead of exploring off it (085's drift).
# (prior) agent_075: SCHEDULE START. Explore the battle-menu move selection early (0.02),
                              # then anneal to 0.005 (commit to the winning move) over ENT_ANNEAL_STEPS — the fix
                              # for 072's "learned 0.49 then drifted into bad moves" instability. (prior note:)
                              # agent_059: entropy schedule DISABLED (start==end → constant 0.01).
                              # 058 proved the 0.03→0.01 ramp DE-LOCKS 053's pickup (oscillating 0↔1)
                              # without buying delivery — entropy adds noise around the dominant action,
                              # it can't synthesize the missing A-press. The frontier reset now supplies
                              # exploration via STATE DIVERSITY (restarting at the frontier), not action
                              # noise, so we hold 053's committed 0.01 and isolate the architecture as
                              # the single new variable. (If the A-press isn't discovered at archived lab
                              # cells by ~15M, that isolates "ent too low to sample A" → bump next run.)
ENT_COEF_CNN_END  = 0.005     # agent_075: commit low after exploring the move
ENT_ANNEAL_STEPS_CNN = 8_000_000   # agent_075: anneal 0.02→0.005 over the first 8M, then hold 0.005

TOTAL_TIMESTEPS_CNN = 100_000_000   # agent_090: RAISED for RL-3's COLD run (more headroom than RL-1/2's
                                    # 60M since this is a from-scratch structural obs-space change, not a
                                    # warm continuation of a prior policy). KILL CRITERION (Global
                                    # Constraints / findings schedule): `nav/reach_cherrygrove < 0.5` at
                                    # 30M → stop — a COLD run must at minimum re-clear the basic nav bar
                                    # the warm RL-1/RL-2 attempts took for granted.
# (prior 089) 60_000_000 — RL-2's budget (REGRESSED, kill criterion fired at 29.3M — see verdict above).
# (prior 088) RL-1's gate (findings §4) — NOT a revert to 079's own 480M budget (that's R5, a later
                                    # attempt); 40M kill gate + 20M headroom to confirm a clean read past
                                    # the gate. Extend toward 200-480M (R5) only if nav/reach_route31 cracks.
# (prior 086, gym-only) agent_086: 5M short stabilization — consolidate the 085@5M peak.
# (prior) agent_084: 10M — the fight consolidates fast from warm+falkner_battle force.
# (prior) agent_082: 20M warm test of longer episodes (MAX_STEPS 131k).
# (prior 081) ~2-day LONG run with the corridor-only exploration whitelist.
# (prior) agent_080: SHORT VALIDATION of the off-path whitelist (warm from 079@130M).
                                    # Watch nav/reach_route31 — if it cracks > 0, the cave-lure was the blocker →
                                    # relaunch long (480M). (prior 480_000_000)
                                    # agent_077: ~2-DAY run (@ ~2778 step/s). MONITOR the ~15M gate first and abort
                                    # if nav/reach_cherrygrove is still flat-zero. (prior 15_000_000)
                                    # agent_076: SHORT VALIDATION budget (~1.5h @ ~2700fps) — confirm the new
                                    # egg-delivered start + re-enabled exploration drive the forward journey
                                    # (rising nav/reach_cherrygrove, nav/reach_route31, non-zero exploration
                                    # reward, no instant deaths). If healthy, scale to a multi-day run by raising
                                    # this number ONLY (env + reward unchanged). Legacy gates below for reference:
                                    # (prior 150_000_000) PPO_CNN_10c budget (~15h @ ~2700fps). GO/NO-GO GATES (with 32k
                                    # episodes the CNN_5 benchmarks apply again — abort early if dead):
                                    #   10M: nav/reach_cherrygrove ≥ 0.6 AND nav/battles_won_mean > 0
                                    #        (the win reward must visibly counter hit-and-flee)
                                    #   30M: nav/egg_received_rate ≥ 0.1 rising; first deliveries expected
                                    #   50M: nav/egg_delivered_rate ≥ 0.2, nav/reach_route31 > 0.1
                                    #  100M: nav/badge_rate > 0 (else if reach_gym > 0.3: bump gym damage
                                    #        3→5 and badge 30→50, warm-restart from the 100M checkpoint)
CHECKPOINT_FREQ_CNN = 5_000_000     # agent_088: REVERTED to the 077-081 corridor-era value (was 1M, agent_082's
                                    # warm-test-only tweak — that experiment was abandoned/never resolved, see
                                    # env/pokemon_env_cnn.py's MAX_STEPS constant, also reverted for this run).
# (prior 082, abandoned) agent_082: 1M for the warm test. (prior 5M agent_081)
                                    # agent_077: 5M → ~96 checkpoints over 480M for the map-progression montage.
                                    # (prior 500_000) agent_076: 500k → several frames across the short validation run so the
                                    # map-visualization montage can show the journey extending over training.
                                    # (prior 1_000_000) agent_075: 5M→1M to CAPTURE the badge_rate peak (072's transient 0.49 fell
                                    # between 5M-spaced saves). in TIMESTEPS — train_cnn divides by N_ENVS for the SB3 callback
                                    # (CNN_5 bug: save_freq is counted in callback-calls = timesteps/n_envs,
                                    #  so the raw 5M×12=60M never fired → no intermediate checkpoints).

# Warm-start for the REVERSE CURRICULUM: each stage fine-tunes from the PREVIOUS stage's *_final.zip.
# Stage 1 starts from PPO_CNN_5_final (already knows New Bark → Cherrygrove → the gate area).
# UPDATE this line between stages (see the table above). Set to None for a cold start.
INIT_FROM_CHECKPOINT = None      # agent_090: COLD — RL-3's `VISITED_OBS=True` changes the observation
                              # space (new "visited" Dict-obs key), so no prior checkpoint is compatible
                              # (SB3's MultiInputPolicy would mismatch the obs dict). This is the point of
                              # RL-3: a genuinely new input requires a from-scratch policy to use it.
# (prior 089) "runs/checkpoints/agent_088/agent_088_39999936_steps.zip"   # WARM from 088@40M for RL-2 (REGRESSED).
# (prior 088) "runs/checkpoints/agent_079/agent_079_129999792_steps.zip"   # WARM from 079@130M
                              # (Route-30-capable — same checkpoint 080/082 warmed from) for RL-1.
                              # CONFINE_TO_CORRIDOR is the new variable added on top of this exact base.
# (prior 087, gym-only) "runs/checkpoints/agent_085/agent_085_4999980_steps.zip"   # agent_086: WARM from 085@5M —
                              # the 40%-badge peak. Stabilize (lower lr/entropy) to consolidate it, not drift off.
# (prior 085) "runs/checkpoints/agent_083/agent_083_4999980_steps.zip"   # WARM from 083@5M (eval badge 10%).
# (prior 083) "runs/checkpoints/agent_050/agent_050_9999984_steps.zip"   # WARM from 050@10M —
                              # the gym-capable base (beat Falkner ~6× in training; 072 reached badge_rate 0.49).
# (prior) "runs/checkpoints/agent_079/agent_079_129999792_steps.zip"   # agent_082: WARM from 079@130M
                              # (reaches Route 30, SAME reward as now) → clean test of MAX_STEPS only.
# (prior) None   # agent_081: COLD (clean test of the whitelist reward, no warm-start confound).
# (prior) "runs/checkpoints/agent_079/agent_079_129999792_steps.zip"   # agent_080: WARM from 079@130M
                              # (reaches Route 30) to quickly test whether the off-path whitelist unblocks Route 31.
                              # (prior None) agent_076: COLD generalist. The old specialists (050, 072-075) were trained on
                              # isolated states (Falkner battle, carry-mode) and would bias navigation; the v2
                              # generalist learns the whole journey from scratch. (prior warm-starts below.)
# (prior) "runs/checkpoints/agent_050/agent_050_9999984_steps.zip"  # agent_074: warm 050 (not a degraded 072 ckpt), low lr/ent from start to stabilize the fight.
                              # agent_073: warm from 072@5M (the ~0.49 badge_rate peak on the Falkner fight) +
                              # lower lr/ent to STABILIZE it (072 then drifted to 0). Same falkner_battle state.