import gymnasium as gym
import numpy as np
from .pyboy_wrapper import PyBoyWrapper
from .ram_reader import RAMReader
from .actions import ACTION_SPACE
from .rewards import (
    compute_reward, make_prev_state, make_reward_maxes,
    CHERRYGROVE, ROUTE_30_GATE, ROUTE_31, VIOLET_CITY, GYM_MAP,
    MR_POKEMON_BIT, ELM_BIT,
)

MAX_STEPS = 2**16  # 65536 env-steps (PPO_CNN_10): the optimal badge path is ~14.6k steps and now
                   # includes the egg backtrack + trainer battles + heals — 2**15 left only 2.2×
                   # slack for an imperfect policy; 2**16 ≈ 4.5× the optimal path.

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
        # Dict obs (PPO_CNN_10): screen image + VISITED-MASK 4th channel (PWhiddy mechanism — an
        # explicit frontier gradient at the map-door chokepoints) + RAM-derived state vector
        # (HP / level / battle / story flags), so the policy can condition on state it could never
        # read from pixels (e.g. the Route 30 fork looks IDENTICAL pre/post egg delivery).
        self.observation_space = gym.spaces.Dict({
            "image":  gym.spaces.Box(low=0, high=255, shape=(72, 80, 4), dtype=np.uint8),
            "vector": gym.spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
        })

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

    def _state_vector(self, ram):
        """RAM-derived self-state vector, all in [0,1]. Enemy fields are zeroed OUTSIDE battle because
        the enemy RAM (0xD0FC/0xD0FF) holds garbage there (same guard as the op_level reward)."""
        in_battle = 1.0 if ram["battle_type"] > 0 else 0.0
        return np.clip(np.array([
            ram["hp_ratio"],                                 # lead Pokemon HP fraction
            ram["lead_level"] / 100.0,                       # lead level
            ram["party_count"] / 6.0,                        # team size
            ram["badge_count"] / 8.0,                        # badges earned
            in_battle,                                       # currently in a battle?
            ram["enemy_hp_ratio"] * in_battle,               # enemy HP fraction (battle only)
            (ram["enemy_lead_level"] / 100.0) * in_battle,   # enemy level (battle only)
            1.0 if ram["flag_elm_mr_pokemon"] & MR_POKEMON_BIT else 0.0,  # egg received
            1.0 if ram["flag_elm_mr_pokemon"] & ELM_BIT else 0.0,         # egg delivered (gate open)
            ram["route_trainers_beaten"] / 4.0,              # Route 30/31 trainers beaten
            ram["gym_trainers_beaten"] / 2.0,                # gym trainers beaten
        ], dtype=np.float32), 0.0, 1.0)

    def _visited_mask(self, ram):
        """72x80 uint8 mask: 255 on the 8x8 px block of each VISIBLE metatile already visited this
        episode, 0 otherwise. The GB screen shows 10x9 metatiles with the player at metatile (4,4);
        at half-res one metatile = 8x8 px. Gives the CNN explicit 'where have I been' spatial memory —
        the frontier (dark area) is the direction the tile reward will pay."""
        mask = np.zeros((72, 80), dtype=np.uint8)
        bank, num = ram["map_bank"], ram["map_number"]
        px, py = ram["local_x"], ram["local_y"]
        for row, dy in enumerate(range(-4, 5)):      # 9 metatile rows
            for col, dx in enumerate(range(-4, 6)):  # 10 metatile cols
                if (bank, num, px + dx, py + dy) in self.visited_tiles:
                    mask[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = 255
        return mask

    def _get_obs(self, screen, ram_state):
        """Dict observation: downsampled RGB screen + visited-mask channel (→ CNN) + state vector (→ MLP)."""
        rgb = screen[:, :, :3]                          # drop alpha
        image = rgb[::2, ::2].astype(np.uint8)          # downsample to 72x80
        image = np.dstack([image, self._visited_mask(ram_state)])  # (72, 80, 4)
        return {"image": image, "vector": self._state_vector(ram_state)}
    
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
            # Story/combat progress flags — consumed by InfoLoggerCallback at episode end
            # to compute nav/egg_*_rate, nav/*_trainers_mean and nav/badge_rate.
            "egg_received": bool(ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT),
            "egg_delivered": bool(ram_state["flag_elm_mr_pokemon"] & ELM_BIT),
            "route_trainers_beaten": ram_state["route_trainers_beaten"],
            "gym_trainers_beaten": ram_state["gym_trainers_beaten"],
        }

        self.prev_state = make_prev_state(ram_state) # Store only the relevant RAM values for reward edge detection
        self.steps += 1
        truncated = self.steps >= MAX_STEPS

        obs = self._get_obs(screen, ram_state)  # Dict obs: screen image + self-state vector

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
        return self._get_obs(screen, ram_state), {}

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