import gymnasium as gym
import numpy as np
from .pyboy_wrapper import PyBoyWrapper
from .ram_reader import RAMReader
from .actions import ACTION_SPACE
from .rewards import compute_reward, make_prev_state

MAX_STEPS = 2**16  # Max steps per episode — must match config.py

class PokemonEnvMLP(gym.Env):
    """
    MLP-friendly Gymnasium environment for Pokemon Silver.
    Observation = 20-dim float vector built from RAM state (positions, flags, levels, HP).
    For pixel-based observation see env/pokemon_env_cnn.py — same reward, same actions.
    """
    def __init__(self, rom_path, state_path="../saves/totodile.state", headless=True, gif_dir="../runs/gifs/", render_mode=None):
        self.pyboy = PyBoyWrapper(rom_path, state_path, headless)
        self.ram_reader = RAMReader(self.pyboy.pyboy)
        self.render_mode = render_mode
        self.gif_dir = gif_dir
        self.episode_count = 0
        self.steps = 0  # Step counter for episode length tracking

        self.capture_gif = render_mode == "rgb_array"
        self.gif_frames = []  # Store frames for GIF creation if needed

        # Previous-frame RAM snapshot for reward edge detection (populated in reset())
        self.prev_state = {}
        self.visited_tiles = set()   # Track visited tiles for exploration reward
        self.visited_maps  = set()   # Track visited (bank, map) pairs for map transition reward (LIFETIME)
        self.episode_maps = set()  # Track maps visited within the current episode for waypoint rewards

        # Define action and observation spaces
        self.action_space = ACTION_SPACE
        
        self.observation_space = gym.spaces.Box(
            low = 0.0,
            high=1.0,
            shape=(20,),
            dtype=np.float32
        )

    def step(self, action):
        """
        Take an action in the environment, advance the emulator, and return the new state, reward, done, and info.
        """
        # Send the action to the emulator and advance it
        screen = self.pyboy.step(action)

        if self.capture_gif:
            self.gif_frames.append(screen.copy())
        
        # Read the new RAM state
        ram_state = self.ram_reader.read_all()

        tile = (ram_state["map_bank"], ram_state["map_number"], ram_state["local_x"], ram_state["local_y"])
        new_tile = tile not in self.visited_tiles
        if new_tile:
            self.visited_tiles.add(tile)

        # Compute the reward based on RAM state changes and exploration
        reward, reward_info = compute_reward(
            ram_state, self.prev_state, new_tile, self.visited_maps, self.episode_maps
        )

        # Example termination condition: episode ends if we win (zephyr badge) or lose (HP drops to 0 in overworld)
        terminated = ram_state["zephyr"] or (ram_state["hp_ratio"] <= 0 and ram_state["battle_type"] == 0)  # Episode ends if we win or lose

        # Return the observation, reward, done, and info
        obs = np.array([
            ram_state["map_bank"] / 255,   # [0]  map bank (group)
            ram_state["map_number"] / 255, # [1]  map number within group
            ram_state["local_x"] / 255,    # [2]  player X on current map
            ram_state["local_y"] / 255,    # [3]  player Y on current map
            ram_state["zephyr"],                                           # [4]  win condition (binary)
            ram_state["battle_type"] / 3,                                  # [5]  0=overworld 1=wild 2=trainer 3=gym
            ram_state["party_count"] / 6,                                  # [6]  number of Pokemon in party
            ram_state["hp_ratio"],                                         # [7]  lead Pokemon HP ratio
            float(not bool(ram_state["flag_rival_cherrygrove"] & 0x40)),  # [8]  1=rival beaten (0xD88E bit6 cleared)
            float(bool(ram_state["flag_elm_mr_pokemon"] & 0x40)),         # [9]  1=egg received from Mr. Pokemon
            float(bool(ram_state["flag_elm_mr_pokemon"] & 0x80)),         # [10] 1=egg delivered to Elm
            ram_state["lead_level"] / 100.0,                              # [11] lead Pokemon level (0xDA49)
            ram_state["enemy_lead_level"] / 100.0,                        # [12] enemy lead level (0xD0FC, 0 when not in battle)
            ram_state["enemy_hp_ratio"],                                   # [13] enemy HP ratio (0xD0FF/D101)
            min(len(self.visited_tiles) / 2**10, 1.0),                    # [14] unique tiles visited this episode
            ram_state["party_levels"][1] / 100.0,                         # [15] party slot 2 level (0=empty)
            ram_state["party_levels"][2] / 100.0,                         # [16] party slot 3 level
            ram_state["party_levels"][3] / 100.0,                         # [17] party slot 4 level
            ram_state["party_levels"][4] / 100.0,                         # [18] party slot 5 level
            ram_state["party_levels"][5] / 100.0,                         # [19] party slot 6 level
        ], dtype=np.float32)
        observation = obs  # or screen, or a combination of both
        info = {
            "reward_exploration": reward_info["exploration"],
            "reward_events":      reward_info["events"],
            "reward_penalties":   reward_info["penalties"],
            "visited_tiles":      len(self.visited_tiles),
            "hp_ratio":           ram_state["hp_ratio"],
            "map_number":         ram_state["map_number"],
            "in_battle":          int(ram_state["battle_type"] > 0),
        }

        self.prev_state = make_prev_state(ram_state)
        self.steps += 1
        truncated = self.steps >= MAX_STEPS
        return observation, reward, terminated, truncated, info
        

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state and return the initial observation.
        """
        self.pyboy.reset()

        self.episode_count += 1 
        self.steps = 0  # Reset step counter

        if self.gif_frames:
            self.pyboy.capture_gif(f"{self.gif_dir}/episode_{self.episode_count:04d}.gif", self.gif_frames)
            self.gif_frames = []  # Clear frames for the next episode

        self.visited_tiles = set()
        # self.visited_maps  = set()
        self.episode_maps = set()  # Reset episode-specific visited maps

        # Read initial RAM state — baseline so flags only trigger on change, not on non-zero
        ram_state = self.ram_reader.read_all()
        self.prev_state = make_prev_state(ram_state)
        self.visited_maps.add((ram_state["map_bank"], ram_state["map_number"]))
        
        obs = np.array([
            ram_state["map_bank"] / 255,   # [0]  map bank (group)
            ram_state["map_number"] / 255, # [1]  map number within group
            ram_state["local_x"] / 255,    # [2]  player X on current map
            ram_state["local_y"] / 255,    # [3]  player Y on current map
            ram_state["zephyr"],                                           # [4]  win condition (binary)
            ram_state["battle_type"] / 3,                                  # [5]  0=overworld 1=wild 2=trainer 3=gym
            ram_state["party_count"] / 6,                                  # [6]  number of Pokemon in party
            ram_state["hp_ratio"],                                         # [7]  lead Pokemon HP ratio
            float(not bool(ram_state["flag_rival_cherrygrove"] & 0x40)),  # [8]  1=rival beaten (0xD88E bit6 cleared)
            float(bool(ram_state["flag_elm_mr_pokemon"] & 0x40)),         # [9]  1=egg received from Mr. Pokemon
            float(bool(ram_state["flag_elm_mr_pokemon"] & 0x80)),         # [10] 1=egg delivered to Elm
            ram_state["lead_level"] / 100.0,                              # [11] lead Pokemon level (0xDA49)
            ram_state["enemy_lead_level"] / 100.0,                        # [12] enemy lead level (0xD0FC, 0 when not in battle)
            ram_state["enemy_hp_ratio"],                                   # [13] enemy HP ratio (0xD0FF/D101)
            min(len(self.visited_tiles) / 2**10, 1.0),                    # [14] unique tiles visited this episode
            ram_state["party_levels"][1] / 100.0,                         # [15] party slot 2 level (0=empty)
            ram_state["party_levels"][2] / 100.0,                         # [16] party slot 3 level
            ram_state["party_levels"][3] / 100.0,                         # [17] party slot 4 level
            ram_state["party_levels"][4] / 100.0,                         # [18] party slot 5 level
            ram_state["party_levels"][5] / 100.0,                         # [19] party slot 6 level
        ], dtype=np.float32)

        return obs, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self.pyboy.pyboy.screen.ndarray
        return None

    def close(self):
        """
        Clean up any resources used by the environment, such as closing the emulator instance.
        """
        self.pyboy.pyboy.stop()
