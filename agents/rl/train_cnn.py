"""
CnnPolicy training entry point — PPO_19 and beyond.

Differences from train_mlp.py:
  - Imports PokemonEnvCNN (Dict obs: image 72x80x4 uint8 [RGB + visited mask] + 11-float vector)
  - Uses VecFrameStack(n_stack=4) so policy sees motion across frames (image → 16 channels)
  - VecTransposeImage moves channels-last (HWC) → channels-first (CHW) for PyTorch
  - "CnnPolicy" instead of "MlpPolicy" — SB3 picks NatureCNN automatically
  - device="cuda" (CNN forward must run on GPU)
  - VecNormalize: NORM_OBS=False (pixels are uint8, don't normalize them — SB3 scales /255 internally)
                  NORM_REWARD=True (same as MLP — running mean/std on reward)

CNN-specific config constants are read from config.py (LEARNING_RATE_CNN, N_STEPS_CNN, etc.).
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
    """Logs custom metrics to TensorBoard at each rollout.

    Two kinds of metric:
      - per-STEP means (reward components, tiles, hp, battle): averaged over every step.
      - per-EPISODE navigation progress: recorded ONLY when an episode ends (done), so it is
        NOT confounded by how long the agent dwells in a map. `max_waypoint` is a monotonic
        ordinal 0..5 (0=start, 1=Cherrygrove, 2=Route30-gate, 3=Route31 POST-GATE, 4=Violet, 5=Gym).
        We log the mean furthest point reached AND the fraction of episodes that reached each
        waypoint. `nav/reach_route31` is the key signal: reaching post-gate Route 31 means the agent
        cleared the two-trainer STORY GATE (only possible after delivering the egg to Elm).

        NOTE (PPO_CNN_8): the reverse curriculum uses a SINGLE start-state per run, so we count
        ALL finished episodes — the metric cleanly reflects that stage's start. (The CNN_7
        from_start filter is no longer needed: there is no mixed curriculum to contaminate it.)
    """

    WAYPOINT_NAMES = ["cherrygrove", "route30_gate", "route31", "violet", "gym"]
    # Episode-end story/combat progress (PPO_CNN_10 go/no-go gate metrics):
    #   *_rate  = fraction of finished episodes where the bool flag was set
    #   *_mean  = mean count over finished episodes
    EP_RATE_KEYS  = ["egg_received", "egg_delivered", "zephyr"]
    EP_MEAN_KEYS  = ["route_trainers_beaten", "gym_trainers_beaten"]

    def __init__(self):
        super().__init__()
        self._keys = ["reward_exploration", "reward_events", "reward_penalties",
                      "visited_tiles", "hp_ratio", "in_battle"]
        self._buffers = {k: [] for k in self._keys}
        self._ep_waypoints = []  # furthest waypoint of each episode that finished this rollout
        self._ep_flags = {k: [] for k in self.EP_RATE_KEYS + self.EP_MEAN_KEYS}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for k in self._keys:
                if k in info:
                    self._buffers[k].append(info[k])
        # On episode end, SB3 keeps THIS step's info as infos[i] → its max_waypoint is the
        # episode's furthest point. Record it once per finished episode (un-dwell-confounded).
        # Single start-state per run → counting all episodes is clean.
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "max_waypoint" in info:
                self._ep_waypoints.append(info["max_waypoint"])
                for k in self._ep_flags:
                    if k in info:
                        self._ep_flags[k].append(float(info[k]))
        return True

    def _on_rollout_end(self) -> None:
        for k, values in self._buffers.items():
            if values:
                self.logger.record(f"custom/{k}", np.mean(values))
        self._buffers = {k: [] for k in self._keys}

        if self._ep_waypoints:
            wp = np.array(self._ep_waypoints)
            self.logger.record("nav/ep_max_waypoint", float(wp.mean()))
            for ordinal, name in enumerate(self.WAYPOINT_NAMES, start=1):
                self.logger.record(f"nav/reach_{name}", float((wp >= ordinal).mean()))
            for k in self.EP_RATE_KEYS:
                if self._ep_flags[k]:
                    name = "badge_rate" if k == "zephyr" else f"{k}_rate"
                    self.logger.record(f"nav/{name}", float(np.mean(self._ep_flags[k])))
            for k in self.EP_MEAN_KEYS:
                if self._ep_flags[k]:
                    self.logger.record(f"nav/{k.replace('_beaten', '')}_mean",
                                       float(np.mean(self._ep_flags[k])))
        self._ep_waypoints = []
        self._ep_flags = {k: [] for k in self.EP_RATE_KEYS + self.EP_MEAN_KEYS}


def make_env(rank, state_path):
    def _init():
        env = PokemonEnvCNN(
            config.ROM_PATH, state_path, headless=True,
            gif_dir=None,  # no GIFs during training (saves disk space and overhead)
            gif_prefix=config.RUN_NAME,  # ensures GIF filenames carry the run identity
        )
        env.reset(seed=rank)
        return env
    return _init


if __name__ == "__main__":
    device = get_device()
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
    vec_env = VecFrameStack(vec_env, n_stack=4)           # image (72,80,4) → (72,80,16)
    vec_env = VecTransposeImage(vec_env)                  # → (16,72,80) for PyTorch
    # norm_reward=False (PPO_CNN_5): the Phase-1 realignment already keeps every reward term
    # single-digit (×REWARD_SCALE), so reward normalization is unnecessary — and harmful before,
    # because the running-std was dominated by rare +1000 spikes, crushing the dense exploration
    # signal toward zero. Matches Whidden V2 (raw, well-scaled rewards).
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False,
                           clip_obs=10.0, gamma=config.GAMMA)

    # Warm-start (fine-tune) from a checkpoint if configured, else train a fresh policy.
    # PPO.load restores the saved policy + hyperparameters; we pass env + tensorboard_log so the
    # loaded model logs to a new run dir. learn(reset_num_timesteps=True) restarts the step counter.
    if config.INIT_FROM_CHECKPOINT and os.path.exists(config.INIT_FROM_CHECKPOINT):
        print(f"[init] Warm-starting (fine-tune) from {config.INIT_FROM_CHECKPOINT}")
        model = PPO.load(
            config.INIT_FROM_CHECKPOINT, env=vec_env, device=device,
            tensorboard_log=config.LOG_DIR,
        )
    else:
        if config.INIT_FROM_CHECKPOINT:
            print(f"[init] WARNING: {config.INIT_FROM_CHECKPOINT} not found — training from scratch")
        else:
            print("[init] Training from scratch (no warm-start checkpoint set)")
        model = PPO(
            "MultiInputPolicy", vec_env, verbose=1,   # Dict obs (image + state vector) → CombinedExtractor
            learning_rate=config.LEARNING_RATE_CNN,
            clip_range=0.1,
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
            # SB3 counts save_freq in callback-calls (= timesteps / n_envs), so divide to get the
            # intended interval in TIMESTEPS. (CNN_5 bug: raw 5M × 12 envs = 60M never fired.)
            save_freq=max(config.CHECKPOINT_FREQ_CNN // config.N_ENVS_CNN, 1),
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
