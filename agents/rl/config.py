# RUN IDENTIFIER — bump for each new training run.
# Drives: checkpoint subdir (runs/checkpoints/{RUN_NAME}/),
#         checkpoint filename prefix ({RUN_NAME}_<step>_steps.zip),
#         TensorBoard log name (runs/{RUN_NAME}_<N>/).
#
# Convention:
#   - PPO_N      → MLP filone (PPO_1 .. PPO_18, closed 2026-06-03)
#   - PPO_CNN_N  → CNN filone (restart from 1: PPO_CNN_1, PPO_CNN_2, ...)
RUN_NAME = "PPO_CNN_10"

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

# ── PPO_CNN_10: SINGLE-START redesign run ───────────────────────────────────────────────────────
# Event-dominant reward (egg +8/+12, per-trainer +5, badge +30, level saturation) + story/combat
# obs (egg bits, trainer counts, visited-mask channel). NO curriculum: mixed curricula segregate the
# policy (CNN_4/CNN_7 lessons) and the redesign makes the full quest learnable from start.state.
# (counts must sum to N_ENVS_CNN)
CURRICULUM_STATES_CNN = [
    ("saves/start.state", 12),  # full quest from New Bark Town — the ONLY start
]

# CNN hyperparameters (Atari-style defaults adapted for Pokemon)
LEARNING_RATE_CNN = 1.5e-4    # slightly more conservative than MLP
N_STEPS_CNN       = 2048      # smaller rollout (memory: 4 envs × 2048 × 72×80×12 ~ 450MB)
BATCH_SIZE_CNN    = 512       # bigger minibatch, exploits GPU
N_EPOCHS_CNN      = 4         # fewer epochs (more data per rollout)
ENT_COEF_CNN      = 0.03      # Atari-like exploration; CNN sees the path so less needed

TOTAL_TIMESTEPS_CNN = 150_000_000   # PPO_CNN_10 long-run budget (~18h @ ~2250fps). PWhiddy needed ~100M
                                    # for Brock. GO/NO-GO GATES (TensorBoard, abort early if dead):
                                    #   10M: nav/reach_cherrygrove ≥ 0.8 AND nav/egg_received_rate > 0 rising
                                    #   25M: nav/egg_delivered_rate ≥ 0.3, nav/reach_route31 > 0.1
                                    #        (if flat-zero at 40M: abort → delivery +12→+20)
                                    #   50M: nav/reach_violet ≥ 0.3, custom/in_battle ∈ [0.05, 0.3];
                                    #        offline eval 10 eps on the 50M checkpoint
                                    #  100M: nav/badge_rate > 0 (else if reach_gym > 0.3: bump gym damage
                                    #        3→5 and badge 30→50, warm-restart from the 100M checkpoint)
CHECKPOINT_FREQ_CNN = 5_000_000     # in TIMESTEPS — train_cnn divides by N_ENVS for the SB3 callback
                                    # (CNN_5 bug: save_freq is counted in callback-calls = timesteps/n_envs,
                                    #  so the raw 5M×12=60M never fired → no intermediate checkpoints).

# Warm-start for the REVERSE CURRICULUM: each stage fine-tunes from the PREVIOUS stage's *_final.zip.
# Stage 1 starts from PPO_CNN_5_final (already knows New Bark → Cherrygrove → the gate area).
# UPDATE this line between stages (see the table above). Set to None for a cold start.
INIT_FROM_CHECKPOINT = None   # PPO_CNN_10 = new obs shapes (image 72×80×4, vector 11) → incompatible
                              # with ALL previous checkpoints, COLD start. The 100M-gate warm-restart
                              # (reward-only tweak, same obs) CAN reuse PPO_CNN_10 checkpoints.