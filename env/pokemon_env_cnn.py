import gymnasium as gym
import numpy as np
from .pyboy_wrapper import PyBoyWrapper
from .ram_reader import RAMReader
from .actions import ACTION_SPACE
from .rewards import (
    compute_reward, make_prev_state, make_reward_maxes,
    CHERRYGROVE, ROUTE_30_GATE, ROUTE_31, VIOLET_CITY, GYM_MAP,
)

MAX_STEPS = 2**15  # 32768 env-steps (~halved from 2**16): more episode resets → better
                   # credit assignment, still long enough to reach the gym (~14.6k steps to badge)

# Ordered route waypoints (ordinal = index + 1) — drives per-episode navigation progress logging
# to TensorBoard, independent of reward. 0 = start/New Bark … 5 = Violet Gym.
# Ordinal 3 (ROUTE_31, post-gate) = the agent has CLEARED the two-trainer story gate.
WAYPOINT_ORDER = [CHERRYGROVE, ROUTE_30_GATE, ROUTE_31, VIOLET_CITY, GYM_MAP]

class PokemonEnvCNN(gym.Env):
    """CNN-friendly Gymnasium environment for Pokemon Silver."""
    def __init__(self, rom_path, state_path, headless=True,
                 gif_dir="../runs/gifs/", render_mode=None,
                 gif_every_n_episodes=100, gif_prefix="episode"):
        """Initialize the Pokemon environment with the given ROM and state file.
        The environment uses PyBoy as the emulator backend and provides an RGB observation space for CNN input.
        """

        self.pyboy = PyBoyWrapper(rom_path, state_path, headless)
        self.ram_reader = RAMReader(self.pyboy.pyboy)
        self.render_mode = render_mode
        self.capture_gif = gif_dir is not None
        self.gif_dir = gif_dir
        self.gif_every = gif_every_n_episodes
        self.gif_prefix = gif_prefix    # passed as config.RUN_NAME from train_cnn.py
        self.gif_frames = []            # buffer for current episode's frames

        self.action_space = ACTION_SPACE
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(72,80,3), dtype=np.uint8) # RGB image from PyBoy's get_screen_ndarray()

        self.prev_state = {}
        self.reward_maxes = {}       # Per-episode running maxima (level/opponent/event rewards)
        self.visited_tiles = set()           # EPISODE-scoped tiles → small trail-following reward
        self.visited_tiles_lifetime = set()  # LIFETIME-scoped tiles → frontier-expansion reward (never reset)
        self.visited_maps  = set()   # Track visited (bank, map) pairs
        self.episode_maps = set()  # Track maps visited within the current episode for waypoint rewards

        self.steps = 0  # Step counter for episode length tracking
        self.episode_count = 0
        self.max_waypoint = 0  # Furthest route waypoint reached this episode (0..5)
        # True only for start.state envs — lets the nav metric measure the START-START frontier,
        # uncontaminated by curriculum envs that begin past the waypoints (PPO_CNN_7).
        self.is_start_env = str(state_path).endswith("start.state")

    def _get_obs(self, screen):
        """
        Convert the raw screen from PyBoy into the observation format for the agent.
        For CNN input, we can use the RGB values directly, possibly downsampled.
        """
        rgb = screen[:, :, :3]  # Drop alpha channel if present
        return rgb[::2, ::2].astype(np.uint8)  # Downsample to 72x80
    
    def step(self, action):
        screen = self.pyboy.step(action, n=16) # Advance the emulator by 16 frames (1/4 second at 60 FPS)

        # GIF capture: only on selected episodes, and every 3rd env-step to reduce GIF size
        # (1 env-step = 16 emulator ticks; sampling every 3 env-steps = ~5 GIF frames per second of gameplay)
        if self.capture_gif and self.episode_count % self.gif_every == 0 and self.steps % 3 == 0:
            self.gif_frames.append(screen[:, :, :3].copy())  # full frame, drop alpha

        ram_state = self.ram_reader.read_all()

        tile = (ram_state["map_bank"], ram_state["map_number"], ram_state["local_x"], ram_state["local_y"])
        new_tile = tile not in self.visited_tiles
        if new_tile:
            self.visited_tiles.add(tile)
        new_tile_lifetime = tile not in self.visited_tiles_lifetime
        if new_tile_lifetime:
            self.visited_tiles_lifetime.add(tile)

        # Track furthest route waypoint reached this episode (nav-progress logging, not reward)
        current_map = (ram_state["map_bank"], ram_state["map_number"])
        if current_map in WAYPOINT_ORDER:
            self.max_waypoint = max(self.max_waypoint, WAYPOINT_ORDER.index(current_map) + 1)

        # Compute the reward based on RAM state changes and exploration
        reward, reward_info = compute_reward(
            ram_state, self.prev_state, new_tile, self.visited_maps, self.episode_maps,
            self.reward_maxes, new_tile_lifetime,
        )

        terminated = ram_state['zephyr'] or (ram_state['hp_ratio'] <= 0 and ram_state['battle_type'] == 0)  # Episode ends if we win or lose

        info = {
            "reward_exploration": reward_info["exploration"],
            "reward_events": reward_info["events"],
            "reward_penalties": reward_info["penalties"],
            "visited_tiles": len(self.visited_tiles),
            "hp_ratio": ram_state["hp_ratio"],
            "map_number": ram_state["map_number"],
            "in_battle": int(ram_state["battle_type"] > 0),
            "zephyr": bool(ram_state["zephyr"]),
            "badge_count": ram_state["badge_count"],
            "max_waypoint": self.max_waypoint,
            "from_start": self.is_start_env,
        }

        self.prev_state = make_prev_state(ram_state) # Store only the relevant RAM values for reward edge detection
        self.steps += 1
        truncated = self.steps >= MAX_STEPS

        obs = self._get_obs(screen) # Return the processed RGB observation

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state defined by the ROM and state file. Returns the initial observation and info.
        """
        if self.gif_frames:
            path = f"{self.gif_dir}/{self.gif_prefix}_ep{self.episode_count:05d}.gif"
            self.pyboy.capture_gif(path, self.gif_frames)
        self.gif_frames = []  # Clear frames for the next episode

        screen = self.pyboy.reset()
        
        self.steps = 0
        self.episode_count += 1

        self.visited_tiles = set()
        self.episode_maps = set()
        self.max_waypoint = 0

        ram_state = self.ram_reader.read_all()
        self.prev_state = make_prev_state(ram_state)
        self.reward_maxes = make_reward_maxes(ram_state)

        init_tile = (ram_state["map_bank"], ram_state["map_number"], ram_state["local_x"], ram_state["local_y"])
        self.visited_tiles.add(init_tile)
        self.visited_tiles_lifetime.add(init_tile)  # lifetime set is NOT cleared on reset
        self.visited_maps.add((ram_state["map_bank"], ram_state["map_number"]))
        return self._get_obs(screen), {}

    def render(self):
        """
        Returns the current screen as an RGB array if render_mode == "rgb_array".
        In SDL2 mode PyBoy renders automatically to its own window; nothing to do here.
        """
        if self.render_mode == "rgb_array":
            screen = self.pyboy.pyboy.screen.ndarray
            return screen[:, :, :3]  # Drop alpha
        return None

    def close(self):
        """
        Clean up resources when the environment is closed.
        """
        self.pyboy.pyboy.stop()