# RUN IDENTIFIER — bump for each new training run.
# Drives: checkpoint subdir (runs/checkpoints/{RUN_NAME}/),
#         checkpoint filename prefix ({RUN_NAME}_<step>_steps.zip),
#         TensorBoard log name (runs/{RUN_NAME}_<N>/).
#
# NAMING CONVENTION (since 2026-06-12): progressive "agent_NNN" (PWhiddy-style).
# The full historical mapping (PPO_N / PPO_CNN_N* → Agent NNN) lives at the top of
# training_log.md. The currently-live run still uses its legacy name (it was launched
# before the rename); the NEXT run is agent_046.
RUN_NAME = "agent_076"    # RL v2 RE-BASELINE (Whidden-style): one GENERALIST trained from egg-delivered →
                          # navigate New Bark → Violet City → beat Falkner. Drops the isolated-state /
                          # backtracking design (the egg quest is already done in the start state, so the
                          # carry/return reward logic is self-inert) and re-enables forward exploration.
                          # COLD start (INIT_FROM_CHECKPOINT=None); single state egg_delivered_clean.state ×12.
                          # This is a SHORT VALIDATION run (15M) to confirm the new state + exploration drive
                          # the forward journey before scaling to a multi-day run (just bump TOTAL_TIMESTEPS_CNN).

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
    # agent_076 RL v2: ONE generalist from the egg-delivered state. The Route-30 north gate is already
    # open in this save, so the full forward path (Elm Lab → Route 29 → Cherrygrove → Route 30 → gate →
    # Route 31 → Violet gatehouse → Violet City → Gym → Falkner) is reachable. No curriculum, no
    # backtracking: exploration + forward waypoints + the gym fight are the income.
    ("saves/egg_delivered_clean.state", 12),
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
                                 # pickup/carry separation needed.
EXPLORATION_SCALE  = 1.0         # agent_076 (RL v2): RE-ENABLED — the generalist must NAVIGATE from Elm's lab to
                                 # the gym, so the dense per-new-tile / new-map exploration reward is the primary
                                 # driver again. (0.0 was the Phase-2 gym-only run that forbade leaving the gym.)
FRONTIER_ENABLED   = False       # agent_069: OFF — pure fixed-start carry training, no Go-Explore reset.
FRONTIER_SEED_FROM = "runs/frontier_archive/agent_065"  # agent_067: seed the (cleared) archive with these
                                  # carry-state save-states at launch, to bootstrap the stuck cold start with
                                  # state diversity. None = no seeding (normal). Marker applies at obs-time, so
                                  # 065's marker-less save-states are compatible.
FRONTIER_N_ENVS    = 3            # agent_068: back to 3/9 (067's 6/6 collapsed pickup via carry-dominance).
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
LEARNING_RATE_CNN = 7e-5      # agent_075: INTERMEDIATE lr — 1e-4 (072) learned but overshot into bad moves;
                              # 5e-5 (074) too gentle to learn. 7e-5 + the entropy schedule = learn then commit.
N_STEPS_CNN       = 2048      # smaller rollout (memory: 4 envs × 2048 × 72×80×12 ~ 450MB)
BATCH_SIZE_CNN    = 512       # bigger minibatch, exploits GPU
N_EPOCHS_CNN      = 4         # fewer epochs (more data per rollout)
ENT_COEF_CNN      = 0.02      # agent_075: SCHEDULE START. Explore the battle-menu move selection early (0.02),
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

TOTAL_TIMESTEPS_CNN = 15_000_000    # agent_076: SHORT VALIDATION budget (~1.5h @ ~2700fps) — confirm the new
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
CHECKPOINT_FREQ_CNN = 500_000       # agent_076: 500k → several frames across the short validation run so the
                                    # map-visualization montage can show the journey extending over training.
                                    # (prior 1_000_000) agent_075: 5M→1M to CAPTURE the badge_rate peak (072's transient 0.49 fell
                                    # between 5M-spaced saves). in TIMESTEPS — train_cnn divides by N_ENVS for the SB3 callback
                                    # (CNN_5 bug: save_freq is counted in callback-calls = timesteps/n_envs,
                                    #  so the raw 5M×12=60M never fired → no intermediate checkpoints).

# Warm-start for the REVERSE CURRICULUM: each stage fine-tunes from the PREVIOUS stage's *_final.zip.
# Stage 1 starts from PPO_CNN_5_final (already knows New Bark → Cherrygrove → the gate area).
# UPDATE this line between stages (see the table above). Set to None for a cold start.
INIT_FROM_CHECKPOINT = None   # agent_076: COLD generalist. The old specialists (050, 072-075) were trained on
                              # isolated states (Falkner battle, carry-mode) and would bias navigation; the v2
                              # generalist learns the whole journey from scratch. (prior warm-starts below.)
# (prior) "runs/checkpoints/agent_050/agent_050_9999984_steps.zip"  # agent_074: warm 050 (not a degraded 072 ckpt), low lr/ent from start to stabilize the fight.
                              # agent_073: warm from 072@5M (the ~0.49 badge_rate peak on the Falkner fight) +
                              # lower lr/ent to STABILIZE it (072 then drifted to 0). Same falkner_battle state.