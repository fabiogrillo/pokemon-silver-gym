import numpy as np
from agents.rl import config
from env.pokemon_env import PokemonEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


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


def make_env(rank):
    def _init():
        env = PokemonEnv(config.ROM_PATH, config.STATE_PATH, headless=True)
        env.reset(seed=rank)
        return env
    return _init


if __name__ == "__main__":
    vec_env = SubprocVecEnv([make_env(i) for i in range(config.N_ENVS)])
    vec_env = VecMonitor(vec_env)

    model = PPO("MlpPolicy", vec_env, verbose=1,
                learning_rate=config.LEARNING_RATE,
                n_steps=config.N_STEPS,
                batch_size=config.BATCH_SIZE,
                n_epochs=config.N_EPOCHS,
                gamma=config.GAMMA,
                gae_lambda=config.GAE_LAMBDA,
                ent_coef=config.ENT_COEF,
                tensorboard_log=config.LOG_DIR,
                device="cpu")

    callbacks = [
        CheckpointCallback(save_freq=config.CHECKPOINT_FREQ, save_path=config.MODEL_DIR, name_prefix="ppo_pokemon"),
        InfoLoggerCallback(),
    ]
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

    model.save(f"{config.MODEL_DIR}/ppo_pokemon_final")
    print("Training completed and model saved.")