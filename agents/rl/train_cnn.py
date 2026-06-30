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
    """Log custom metrics to TensorBoard each rollout.

    - per-STEP means (reward components, tiles, hp, battle): averaged over every step.
    - per-EPISODE nav progress: recorded only on episode end, so it is NOT confounded by dwell
      time. `max_waypoint` is a monotonic ordinal 0..5 (0=start, 1=Cherrygrove, 2=Route30-gate,
      3=Route31 POST-GATE, 4=Violet, 5=Gym). Logs the mean furthest point AND the per-waypoint
      reach fraction. `nav/reach_route31` is the key signal: post-gate Route 31 means the
      two-trainer STORY GATE is cleared (only possible after the egg is delivered to Elm).
    """

    WAYPOINT_NAMES = ["cherrygrove", "route30_gate", "route31", "violet", "gym"]
    # Episode-end story/combat progress (PPO_CNN_10 go/no-go gate metrics):
    #   *_rate  = fraction of finished episodes where the bool flag was set
    #   *_mean  = mean count over finished episodes
    EP_RATE_KEYS  = ["egg_received", "egg_delivered", "zephyr"]
    EP_MEAN_KEYS  = ["route_trainers_beaten", "gym_trainers_beaten",
                     "battles_won", "battles_fled", "battles_lost", "return_progress"]

    def __init__(self):
        super().__init__()
        self._keys = ["reward_exploration", "reward_events", "reward_penalties",
                      "visited_tiles", "hp_ratio", "in_battle"]
        self._buffers = {k: [] for k in self._keys}
        # Episode-end stats split by origin: "nav" = true start.state episodes (the SUCCESS GATE),
        # "front" = Go-Explore frontier-reset episodes (agent_059), which begin deep along the
        # policy's own trajectory. Without the split, frontier episodes (which start past the
        # waypoints) would inflate the start-state metrics and hide regressions (the CNN_7 lesson).
        self._ep_waypoints = {"nav": [], "front": []}
        self._ep_flags = {p: {k: [] for k in self.EP_RATE_KEYS + self.EP_MEAN_KEYS}
                          for p in ("nav", "front")}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for k in self._keys:
                if k in info:
                    self._buffers[k].append(info[k])
        # On episode end, SB3 keeps THIS step's info as infos[i] → its max_waypoint is the
        # episode's furthest point. Record it once per finished episode (un-dwell-confounded).
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "max_waypoint" in info:
                prefix = "nav" if info.get("from_start", True) else "front"
                self._ep_waypoints[prefix].append(info["max_waypoint"])
                for k in self._ep_flags[prefix]:
                    if k in info:
                        self._ep_flags[prefix][k].append(float(info[k]))
        return True

    def _on_rollout_end(self) -> None:
        for k, values in self._buffers.items():
            if values:
                self.logger.record(f"custom/{k}", np.mean(values))
        self._buffers = {k: [] for k in self._keys}

        for prefix in ("nav", "front"):
            wps = self._ep_waypoints[prefix]
            if wps:
                wp = np.array(wps)
                self.logger.record(f"{prefix}/ep_max_waypoint", float(wp.mean()))
                for ordinal, name in enumerate(self.WAYPOINT_NAMES, start=1):
                    self.logger.record(f"{prefix}/reach_{name}", float((wp >= ordinal).mean()))
                for k in self.EP_RATE_KEYS:
                    if self._ep_flags[prefix][k]:
                        name = "badge_rate" if k == "zephyr" else f"{k}_rate"
                        self.logger.record(f"{prefix}/{name}",
                                           float(np.mean(self._ep_flags[prefix][k])))
                for k in self.EP_MEAN_KEYS:
                    if self._ep_flags[prefix][k]:
                        self.logger.record(f"{prefix}/{k.replace('_beaten', '')}_mean",
                                           float(np.mean(self._ep_flags[prefix][k])))
        self._ep_waypoints = {"nav": [], "front": []}
        self._ep_flags = {p: {k: [] for k in self.EP_RATE_KEYS + self.EP_MEAN_KEYS}
                          for p in ("nav", "front")}


class EntCoefScheduleCallback(BaseCallback):
    """Linearly anneal ent_coef `start`→`end` over `anneal_steps`, then hold `end`.

    agent_056 lever: a FIXED low ent_coef (0.01) can't learn new long-range behavior (delivery
    A-press, southward backtrack), but a FIXED high one (0.03, agent_052) never locks segment 1.
    The schedule explores high early, then commits low. SB3 reads model.ent_coef as a float each
    train() call, so the per-rollout mutation applies on the next update (lr supports schedules
    natively; ent_coef does not — hence this callback)."""

    def __init__(self, start, end, anneal_steps):
        super().__init__()
        self.start, self.end, self.anneal_steps = start, end, anneal_steps

    def _on_rollout_start(self) -> None:
        progress = min(1.0, self.num_timesteps / self.anneal_steps)
        self.model.ent_coef = self.start + (self.end - self.start) * progress
        self.logger.record("train/ent_coef_sched", self.model.ent_coef)

    def _on_step(self) -> bool:
        return True


def make_env(rank, state_path, frontier_root=None, p_frontier=0.0):
    def _init():
        env = PokemonEnvCNN(
            config.ROM_PATH, state_path, headless=True,
            gif_dir=None,  # no GIFs during training (saves disk space and overhead)
            gif_prefix=config.RUN_NAME,  # ensures GIF filenames carry the run identity
            # Every worker points its FrontierArchive at the SAME dir → frontier cells are shared
            # across SubprocVecEnv processes via the filesystem (no IPC). agent_060: p_frontier is
            # assigned PER ENV (dedicated frontier envs) — see __main__ — not a global flag.
            frontier_root=frontier_root,
            p_frontier=p_frontier if frontier_root else 0.0,
            frontier_max_cells=config.FRONTIER_MAX_CELLS,
            frontier_cell_k=config.FRONTIER_CELL_K,
            frontier_epsilon=config.FRONTIER_EPSILON,
            frontier_max_steps=getattr(config, "FRONTIER_MAX_STEPS", 2**16),
            egg_marker=getattr(config, "EGG_MARKER", False),  # agent_066: egg-state visual marker
            exploration_scale=getattr(config, "EXPLORATION_SCALE", 1.0),  # agent_071: 0 in Phase-2 gym runs
            confine_to_gym=getattr(config, "CONFINE_TO_GYM", False),  # agent_087: end episode on leaving GYM_MAP
        )
        env.reset(seed=rank)
        return env
    return _init


if __name__ == "__main__":
    device = get_device()
    assert sum(n for _, n in config.CURRICULUM_STATES_CNN) == config.N_ENVS_CNN, \
        f"CURRICULUM_STATES_CNN counts must sum to N_ENVS_CNN ({config.N_ENVS_CNN})"
    # Fail fast on a missing save state — PyBoyWrapper only WARNS and silently starts a fresh game,
    # which would train from the wrong scene (e.g. agent_076 needs saves/egg_delivered_clean.state,
    # created manually via `python tests/save_state.py egg_delivered_clean before_elm_delivery`).
    for state_path, _ in config.CURRICULUM_STATES_CNN:
        if not os.path.exists(state_path):
            sys.exit(f"[init] Missing save state: {state_path}\n"
                     f"       Create it first (see README / Part A) before training.")

    # agent_059: prepare the shared frontier archive dir (one per run). Clear it at startup so a run
    # never samples stale cells left by a DIFFERENT policy (those would be off-distribution).
    frontier_root = None
    if getattr(config, "FRONTIER_ENABLED", False):
        import shutil
        frontier_root = os.path.join(config.FRONTIER_ROOT, config.RUN_NAME)
        shutil.rmtree(frontier_root, ignore_errors=True)
        os.makedirs(frontier_root, exist_ok=True)
        # agent_067: optionally SEED the archive with another run's carry-state save-states, to bootstrap
        # a cold start that can't reach those states on its own (the pure-start-cold wall, 066).
        seed_from = getattr(config, "FRONTIER_SEED_FROM", None)
        if seed_from and os.path.isdir(seed_from):
            import glob as _glob
            seeded = 0
            for src in _glob.glob(os.path.join(seed_from, "*.state")):
                shutil.copy(src, frontier_root)
                seeded += 1
            print(f"[frontier] SEEDED {seeded} cells from {seed_from}")
        print(f"[frontier] Go-Explore reset ENABLED — shared archive at {frontier_root} "
              f"(frontier_envs={config.FRONTIER_N_ENVS}/{config.N_ENVS_CNN}, p={config.FRONTIER_P}, "
              f"max_cells={config.FRONTIER_MAX_CELLS}, k={config.FRONTIER_CELL_K})")

    # agent_060: DEDICATE the last FRONTIER_N_ENVS envs to frontier resets (p=FRONTIER_P); the rest
    # are PURE start (p=0). 059 (p=0.5 on ALL 12 envs) let the southward frontier gradient suppress
    # the NORTHWARD egg-pickup excursion → nav/egg_received cratered 1.0→0 (pickup is the foundation;
    # 0 pickup structurally caps delivery at 0). Keeping a majority of envs pure-start guarantees the
    # full start→pickup→deliver task keeps being reinforced at strength while a minority deepens the
    # archive. The archive is still SHARED, so the frontier practice transfers to the start envs.
    n_frontier = config.FRONTIER_N_ENVS if frontier_root else 0
    n_start = config.N_ENVS_CNN - n_frontier
    env_fns = []
    rank = 0
    for state_path, n in config.CURRICULUM_STATES_CNN:
        for _ in range(n):
            p = config.FRONTIER_P if rank >= n_start else 0.0   # last n_frontier envs are frontier
            env_fns.append(make_env(rank, state_path, frontier_root, p))
            rank += 1

    # ── Vec wrappers stack (order matters!)
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)           # image (72,80,4) → (72,80,16)
    vec_env = VecTransposeImage(vec_env)                  # → (16,72,80) for PyTorch
    # norm_reward=False (PPO_CNN_5): rewards are already single-digit (×REWARD_SCALE), so
    # normalization is unnecessary — and was harmful before, when rare +1000 spikes dominated the
    # running-std and crushed the dense exploration signal. Matches Whidden V2 (raw, well-scaled).
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
            # PPO.load restores the checkpoint's lr AND ent_coef — override both with the
            # config values so warm-starts can run with different optimizer settings.
            custom_objects={
                "learning_rate": config.LEARNING_RATE_CNN,
                "lr_schedule": (lambda _: config.LEARNING_RATE_CNN),
                "ent_coef": config.ENT_COEF_CNN,
            },
        )
        print(f"[init] learning_rate overridden to {config.LEARNING_RATE_CNN}, "
              f"ent_coef to {config.ENT_COEF_CNN}")
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
        # agent_056: entropy schedule (explore high → commit low). Overrides the static
        # ENT_COEF_CNN/custom_objects value from rollout 1 onward.
        EntCoefScheduleCallback(
            start=config.ENT_COEF_CNN,
            end=config.ENT_COEF_CNN_END,
            anneal_steps=config.ENT_ANNEAL_STEPS_CNN,
        ),
    ]
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS_CNN,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=config.RUN_NAME,
    )

    model.save(os.path.join(checkpoint_dir, f"{config.RUN_NAME}_final"))
    print(f"Training completed and model saved as {config.RUN_NAME}_final.zip in {checkpoint_dir}")
