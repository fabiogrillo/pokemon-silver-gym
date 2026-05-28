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
    ("saves/violet_city.state",         2),  # in front of Pokemon Center in Violet City, test map transition detection
    ("saves/violet_city_gym.state",     0),  # inside Violet City Gym, test battle reward and badge detection
]

# PPO - Principal Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 4096  # Number of steps to run in each environment per policy rollout
BATCH_SIZE = 64 # Minibatch size for updating the policy
N_EPOCHS = 10   # Number of epochs to update the policy
GAMMA = 0.999    # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
ENT_COEF = 0.08  # Entropy coefficient for exploration

# Training
TOTAL_TIMESTEPS = 100_000_000 # Total number of timesteps to train on
CHECKPOINT_FREQ = 12_500_000    # Save a checkpoint every N timesteps
LOG_DIR = "runs/"            # Directory for TensorBoard logs and checkpoints
MODEL_DIR = "runs/checkpoints/"  # Directory to save model checkpoints