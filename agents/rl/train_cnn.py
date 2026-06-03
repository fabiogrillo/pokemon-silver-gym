"""
CnnPolicy training entry point — PPO_19 and beyond.

Differences from train_mlp.py:
  - Imports PokemonEnvCNN (screen-based observation, 72x80x3 uint8)
  - Uses VecFrameStack(n_stack=4) so policy sees motion across frames
  - VecTransposeImage moves channels-last (HWC) → channels-first (CHW) for PyTorch
  - "CnnPolicy" instead of "MlpPolicy" — SB3 picks NatureCNN automatically
  - device="cuda" (CNN forward must run on GPU)
  - VecNormalize: NORM_OBS=False (pixels are uint8, don't normalize them — SB3 scales /255 internally)
                  NORM_REWARD=True (same as MLP — running mean/std on reward)

CNN-specific config constants are read from config.py (LEARNING_RATE_CNN, N_STEPS_CNN, etc.).
TODO: define those constants in config.py before launching.
"""

import os
import sys
import numpy as np
import torch
from agents.rl import config
from env.pokemon_env_cnn import PokemonEnvCNN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecMonitor, VecNormalize, VecFrameStack, VecTransposeImage
)
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


def get_device():
    """Detect CUDA; abort on CPU since CnnPolicy is impractically slow there."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] CUDA detected → {name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    print("[device] CUDA NOT available. CnnPolicy on CPU is ~100× slower than on GPU.")
    print("[device] Aborting. Set CUDA_VISIBLE_DEVICES or install a CUDA-enabled torch.")
    sys.exit(1)


class InfoLoggerCallback(BaseCallback):
    """Logs mean values of custom info dict keys to TensorBoard at each rollout."""

    def __init__(self):
        super().__init__()
        self._keys = ["reward_exploration", "reward_events", "reward_penalties",
                      "visited_tiles", "hp_ratio", "in_battle"]
        self._buffers = {k: [] for k in self._keys}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for k in self._keys:
                if k in info:
                    self._buffers[k].append(info[k])
        return True

    def _on_rollout_end(self) -> None:
        for k, values in self._buffers.items():
            if values:
                self.logger.record(f"custom/{k}", np.mean(values))
        self._buffers = {k: [] for k in self._keys}


def make_env(rank, state_path):
    def _init():
        env = PokemonEnvCNN(
            config.ROM_PATH, state_path, headless=True,
            gif_prefix=config.RUN_NAME,  # ensures GIF filenames carry the run identity
        )
        env.reset(seed=rank)
        return env
    return _init


if __name__ == "__main__":
    device = get_device()
    # TODO: define a CNN-specific curriculum (or reuse config.CURRICULUM_STATES).
    # Note: with N_ENVS_CNN=4 the curriculum must sum to 4. Suggested:
    #   2×start + 1×before_elm_delivery + 1×violet_city_gym
    assert sum(n for _, n in config.CURRICULUM_STATES_CNN) == config.N_ENVS_CNN, \
        f"CURRICULUM_STATES_CNN counts must sum to N_ENVS_CNN ({config.N_ENVS_CNN})"

    env_fns = []
    rank = 0
    for state_path, n in config.CURRICULUM_STATES_CNN:
        for _ in range(n):
            env_fns.append(make_env(rank, state_path))
            rank += 1

    # ── Vec wrappers stack (order matters!)
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)           # (72,80,3) → (72,80,12)
    vec_env = VecTransposeImage(vec_env)                  # → (12,72,80) for PyTorch
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                           clip_obs=10.0, gamma=config.GAMMA)

    model = PPO(
        "CnnPolicy", vec_env, verbose=1,
        learning_rate=config.LEARNING_RATE_CNN,
        n_steps=config.N_STEPS_CNN,
        batch_size=config.BATCH_SIZE_CNN,
        n_epochs=config.N_EPOCHS_CNN,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        ent_coef=config.ENT_COEF_CNN,
        tensorboard_log=config.LOG_DIR,
        device=device,
    )

    checkpoint_dir = os.path.join(config.MODEL_DIR, config.RUN_NAME)
    callbacks = [
        CheckpointCallback(
            save_freq=config.CHECKPOINT_FREQ_CNN,
            save_path=checkpoint_dir,
            name_prefix=config.RUN_NAME,
        ),
        InfoLoggerCallback(),
    ]
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS_CNN,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=config.RUN_NAME,
    )

    model.save(os.path.join(checkpoint_dir, f"{config.RUN_NAME}_final"))
    print(f"Training completed and model saved as {config.RUN_NAME}_final.zip in {checkpoint_dir}")
