# Training configuration for the CNN PPO agent (agents/rl/train_cnn.py).
# Current run: slide the heal reverse-curriculum one rung outward. agent_098 froze the heal:
# from the nurse-facing state the frozen checkpoint initiates the dialog, heals, walks to the gym
# and takes the badge 10/10 (~1.8k steps, 0 losses) — the first frozen heal->fight->badge chain in
# the project. From the Center door it's 3/10 and from the street 0; so the mastered mid-dialog
# rung retires and a low-HP Violet street state (harvested cell, lead lv 13 at 7/40 HP, only
# Falkner left) joins: street -> Center door -> nurse -> gym is now the chain under training.

# RUN_NAME drives every output path: checkpoints (runs/checkpoints/<RUN_NAME>/),
# checkpoint filenames (<RUN_NAME>_<step>_steps.zip) and TensorBoard logs (runs/<RUN_NAME>_<N>/).
RUN_NAME = "agent_099"

# ── Paths ────────────────────────────────────────────────────────────────────
ROM_PATH   = "pokemon_rom.gbc"
STATE_PATH = "saves/start.state"          # default single-env state (overridden by the curriculum below)
LOG_DIR    = "runs/"                       # TensorBoard logs
MODEL_DIR  = "runs/checkpoints/"           # checkpoint output

# ── Environments & curriculum ────────────────────────────────────────────────
N_ENVS_CNN = 12                            # PyBoy is CPU-bound; 12 workers on a 16-core box

# (state_path, n_envs) pairs; counts must sum to N_ENVS_CNN. The heal ladder (street -> door ->
# facing) keeps every rung's success reachable from the rung above it, with the mastered rung kept
# as a retention anchor; all low-HP states have 2 gym trainers beaten, so heal -> Falkner -> badge
# closes within a short horizon. The lv-15 gym env keeps the certified fight competence reinforced
# (that anchor held it at 9-10/10 through five runs). All states share the same story flags (post
# egg-delivery).
CURRICULUM_STATES_CNN = [
    ("saves/violet_street_lowhp.state",        1),  # Violet street, 7/40 HP: find the Center, heal, gym
    ("saves/center_falkner_lowhp_clean.state", 1),  # Center door, 7/40 HP: reach the nurse (3/10 frozen)
    ("saves/center_nurse_facing.state",        1),  # facing the nurse (10/10 frozen — retention rung)
    ("saves/violet_city_gym.state",            1),  # lv-15 gym anchor (fight-competence upkeep)
    ("saves/egg_delivered_clean.state",        8),  # true start; the last 4 are the frontier envs
]

# ── Episode / exploration structure ──────────────────────────────────────────
EGG_MARKER          = False    # stamp the egg-carry state into the observation (unused here)
EXPLORATION_SCALE   = 4.0      # weight on the new-tile/new-map exploration reward; high enough that
                               # crossing a map boundary out-earns grinding wild battles in place
CONFINE_TO_GYM      = False    # end the episode on leaving the gym map (used only for the gym-only task)
CONFINE_TO_CORRIDOR = True     # end the episode if the agent leaves the New Bark -> Violet corridor,
                               # so off-path maps stop being places where episodes stall
DYNAMIC_EPISODE_BUDGET = True  # start episodes with a small step cap that only grows when a new
                               # corridor waypoint is reached (earned-budget trick, arXiv:2502.19920)
DYN_BUDGET_BASE     = 32768    # base cap for the dynamic budget (grows up to the env's MAX_STEPS)
VISITED_OBS         = True     # add a 48x48 crop of this episode's visited tiles as a Dict-obs key
                               # (de-transposed; see env/pokemon_env_cnn.py:_visited_crop). Changing
                               # the observation space means this run must start cold (see below).

# ── Go-Explore frontier archive ──────────────────────────────────────────────
# A fraction of resets restart from save-states sampled from the policy's own trajectory, which
# manufactures the state diversity 12 envs alone can't reach across the Route 29 bottleneck.
FRONTIER_ENABLED   = True
FRONTIER_SEED_FROM = "runs/frontier_archive/agent_098"  # carries the Center-interior cells forward
FRONTIER_N_ENVS    = 4                          # dedicated frontier envs; the other 8 are pure-start
FRONTIER_P         = 1.0                         # reset probability for a frontier env (start envs are 0)
FRONTIER_MAX_STEPS = 8000                        # truncate a frontier episode past this many steps
FRONTIER_MAX_CELLS = 4000                        # archive size cap (~200KB/state -> ~0.8GB on disk)
FRONTIER_CELL_K    = 4                           # coordinate bucket size (tiles) for the cell key
FRONTIER_EPSILON   = 0.1                         # uniform-sampling floor vs frontier exploitation
FRONTIER_ROOT      = "runs/frontier_archive"     # per-run subdir <ROOT>/<RUN_NAME>/ (cleared at start)

# ── PPO hyperparameters (CNN policy) ─────────────────────────────────────────
GAMMA                = 0.999       # long-horizon discount: waypoints are thousands of steps apart
GAE_LAMBDA           = 0.95
LEARNING_RATE_CNN    = 3e-5        # halved vs the cold run: fine-tune, don't overwrite
N_STEPS_CNN          = 2048        # rollout length per env
BATCH_SIZE_CNN       = 512
N_EPOCHS_CNN         = 4
ENT_COEF_CNN         = 0.005       # resume at agent_091's annealed end value (its anneal finished)
ENT_COEF_CNN_END     = 0.005       # flat: no further anneal, the policy is consolidated
ENT_ANNEAL_STEPS_CNN = 1_000_000   # irrelevant with equal start/end values

# ── Run length & checkpoints ─────────────────────────────────────────────────
TOTAL_TIMESTEPS_CNN = 15_000_000   # one pass at densifying heal -> fight -> badge
CHECKPOINT_FREQ_CNN = 2_500_000    # in timesteps; train_cnn divides by N_ENVS for the SB3 callback
                                   # (tight: late-training collapse means the best checkpoint is
                                   # rarely the last — keep fine recovery points)

# Warm-start checkpoint: agent_098's final snapshot — heal frozen (nurse-facing 10/10 badge),
# Center door 3/10, gym lv15 9/10, nav 9/10 (certified 2026-07-15, runs/eval_logs/agent_098_*).
INIT_FROM_CHECKPOINT = "runs/checkpoints/agent_098/agent_098_final.zip"
