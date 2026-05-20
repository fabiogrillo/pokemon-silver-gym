# ENV
ROM_PATH = "pokemon_rom.gbc"
STATE_PATH = "saves/totodile.state"
N_ENVS = 8 # Number of parallel environments for vectorized training

# PPO - Principal Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048  # Number of steps to run in each environment per policy rollout
BATCH_SIZE = 64 # Minibatch size for updating the policy
N_EPOCHS = 10   # Number of epochs to update the policy
GAMMA = 0.99    # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
ENT_COEF = 0.01  # Entropy coefficient for exploration

# Training
TOTAL_TIMESTEPS = 10_000_000 # Total number of timesteps to train on
CHECKPOINT_FREQ = 10_000 # Save a checkpoint every N timesteps
LOG_DIR = "runs/" # Directory for TensorBoard logs and checkpoints
MODEL_DIR = "runs/checkpoints/" # Directory to save model checkpoints