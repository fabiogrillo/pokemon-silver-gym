# Training configuration for the CNN PPO agent (agents/rl/train_cnn.py).
# Current run: close the level gap that blocks the end-to-end badge. agent_092's frozen checkpoint
# navigates 10/10 and wins the gym 10/10 from the lv-15 gym save state, but 0/10 end-to-end: the
# navigator arrives at lv 9-12 (it flees most corridor battles) and the fight was only ever trained
# from a lv-15 party. This run swaps the gym curriculum states for states harvested from agent_092's
# own true-start trajectories (lead lv 12, realistic HP), so the fight is trained in exactly the
# distribution the navigator produces.

# RUN_NAME drives every output path: checkpoints (runs/checkpoints/<RUN_NAME>/),
# checkpoint filenames (<RUN_NAME>_<step>_steps.zip) and TensorBoard logs (runs/<RUN_NAME>_<N>/).
RUN_NAME = "agent_093"

# ── Paths ────────────────────────────────────────────────────────────────────
ROM_PATH   = "pokemon_rom.gbc"
STATE_PATH = "saves/start.state"          # default single-env state (overridden by the curriculum below)
LOG_DIR    = "runs/"                       # TensorBoard logs
MODEL_DIR  = "runs/checkpoints/"           # checkpoint output

# ── Environments & curriculum ────────────────────────────────────────────────
N_ENVS_CNN = 12                            # PyBoy is CPU-bound; 12 workers on a 16-core box

# (state_path, n_envs) pairs; counts must sum to N_ENVS_CNN. The three gym states are cells
# harvested from agent_092's own frontier archive (true-start trajectories): lead lv 12, HP 29-47%,
# party as the navigator actually delivers it — NOT the hand-made lv-15 saves/violet_city_gym.state,
# whose level distribution the end-to-end pipeline never produces. All states share the same story
# flags (post egg-delivery), so they don't segregate the policy.
CURRICULUM_STATES_CNN = [
    ("saves/gym_entrance_lv12.state",   1),   # gym interior, no trainers beaten (lv 12, 47% HP)
    ("saves/gym_keeper1_lv12.state",    1),   # one bird keeper down (lv 12, 29% HP)
    ("saves/gym_falkner_lv12.state",    1),   # both keepers down, only Falkner left (lv 12, 47% HP)
    ("saves/egg_delivered_clean.state", 9),   # true start; the last 4 are the frontier envs
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
FRONTIER_SEED_FROM = "runs/frontier_archive/agent_092"  # bootstrap from the previous run's cells
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
TOTAL_TIMESTEPS_CNN = 15_000_000   # one focused pass at the level-matched gym fight
CHECKPOINT_FREQ_CNN = 2_500_000    # in timesteps; train_cnn divides by N_ENVS for the SB3 callback
                                   # (tight: late-training collapse means the best checkpoint is
                                   # rarely the last — keep fine recovery points)

# Warm-start checkpoint: agent_092's final snapshot (nav 10/10 + lv-15 gym fight 10/10 frozen;
# certified offline 2026-07-14 — see runs/eval_logs/agent_092_final_*.jsonl).
INIT_FROM_CHECKPOINT = "runs/checkpoints/agent_092/agent_092_final.zip"
