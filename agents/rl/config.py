# RUN IDENTIFIER — bump for each new training run.
# Drives: checkpoint subdir (runs/checkpoints/{RUN_NAME}/),
#         checkpoint filename prefix ({RUN_NAME}_<step>_steps.zip),
#         TensorBoard log name (runs/{RUN_NAME}_<N>/).
#
# Convention:
#   - PPO_N      → MLP filone (PPO_1 .. PPO_18, closed 2026-06-03)
#   - PPO_CNN_N  → CNN filone (restart from 1: PPO_CNN_1, PPO_CNN_2, ...)
RUN_NAME = "PPO_CNN_1"

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
N_ENVS_CNN = 4                # CNN forward is ~10× costlier than MLP, fewer envs

# CNN curriculum (counts must sum to N_ENVS_CNN). Suggested mix:
CURRICULUM_STATES_CNN = [
    ("saves/start.state",               2),  # main learning target
    ("saves/before_elm_delivery.state", 1),  # Elm reward immediato
    ("saves/violet_city_gym.state",     1),  # gym battle training
]

# CNN hyperparameters (Atari-style defaults adapted for Pokemon)
LEARNING_RATE_CNN = 2.5e-4    # slightly more conservative than MLP
N_STEPS_CNN       = 2048      # smaller rollout (memory: 4 envs × 2048 × 72×80×12 ~ 450MB)
BATCH_SIZE_CNN    = 256       # bigger minibatch, exploits GPU
N_EPOCHS_CNN      = 4         # fewer epochs (more data per rollout)
ENT_COEF_CNN      = 0.01      # Atari-like exploration; CNN sees the path so less needed

TOTAL_TIMESTEPS_CNN = 50_000_000   # first validation; ~30-60h on RTX 5080
CHECKPOINT_FREQ_CNN = 2_500_000    # every ~3-5h of training