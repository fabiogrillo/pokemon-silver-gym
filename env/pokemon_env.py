import gymnasium as gym
import numpy as np
from .pyboy_wrapper import PyBoyWrapper
from .ram_reader import RAMReader
from .actions import ACTION_SPACE

MAX_STEPS = 2**14  # Max steps per episode to prevent infinite loops

class PokemonEnv(gym.Env):
    """
    A Custom OpenAI Gym environment for Pokemon Silver, using PyBoy as the emulator backend.
    The environment provides a discrete action space corresponding to the Game Boy buttons,
    and an observation space that can be defined as needed (e.g., RAM state, screen pixels).
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

        # Previous state tracking for reward calculation
        self.prev_party_count = 1  # Start with 1 Pokemon in party
        self.prev_flag_rival  = 0
        self.prev_flag_elm    = 0
        self.prev_flag_sprout2 = 0
        self.prev_flag_sprout3 = 0
        self.visited_tiles = set()  # Track visited tiles for exploration reward

        # Define action and observation spaces
        self.action_space = ACTION_SPACE
        
        self.observation_space = gym.spaces.Box(
            low = 0.0,
            high=1.0,
            shape=(13,),
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
        reward = self.compute_reward(ram_state, new_tile)  

        # Example termination condition: episode ends if we win (zephyr badge) or lose (HP drops to 0 in overworld)
        terminated = ram_state["zephyr"] or (ram_state["hp_ratio"] <= 0 and ram_state["battle_type"] == 0)  # Episode ends if we win or lose
        
        # Return the observation, reward, done, and info
        obs = np.array([
            ram_state["map_bank"] / 255,  # Normalize to [0,1]
            ram_state["map_number"] / 255,  # Normalize to [0,1]
            ram_state["local_x"] / 255,  # Normalize to [0,1]
            ram_state["local_y"] / 255,  # Normalize to [0,1]
            ram_state["zephyr"],  # Win condition not normalized since it's binary
            ram_state["battle_type"] / 3,  # 3 tpyes: wild, trainer, gym
            ram_state["party_count"] / 6,  # Max 6 Pokemon in party
            ram_state["hp_ratio"],
            ram_state["flag_rival_cherrygrove"] / 255,  # Normalize to [0,1]
            ram_state["flag_elm_mr_pokemon"] / 255,  # Normalize to [0,1]
            ram_state["flag_sprout_tower_2"] / 255,  # Normalize to [0,1]
            ram_state["flag_sprout_tower_3"] / 255,  # Normalize to [0,1]
            min(len(self.visited_tiles) / 2**10, 1.0)  # Normalize visited tiles by an estimated total of 512 unique tiles in the game
        ], dtype=np.float32)
        observation = obs  # or screen, or a combination of both
        info = {}

        self.prev_party_count = ram_state["party_count"]  # Start with 1 Pokemon in party
        self.prev_flag_rival  = ram_state["flag_rival_cherrygrove"]
        self.prev_flag_elm    = ram_state["flag_elm_mr_pokemon"]
        self.prev_flag_sprout2 = ram_state["flag_sprout_tower_2"]
        self.prev_flag_sprout3 = ram_state["flag_sprout_tower_3"]

        self.steps += 1
        truncated = self.steps >= MAX_STEPS
        return observation, reward, terminated, truncated, info
        

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        """
        self.pyboy.reset()

        self.episode_count += 1 
        self.steps = 0  # Reset step counter

        if self.gif_frames:
            self.pyboy.capture_gif(f"{self.gif_dir}/episode_{self.episode_count:04d}.gif", self.gif_frames)
            self.gif_frames = []  # Clear frames for the next episode

        self.visited_tiles = set()  # Track visited tiles for exploration reward

        # Read initial RAM state — must happen before prev_flag initialization
        ram_state = self.ram_reader.read_all()

        # Use actual RAM values as baseline so flags only trigger on change, not on non-zero
        self.prev_party_count  = 1
        self.prev_flag_rival   = ram_state["flag_rival_cherrygrove"]
        self.prev_flag_elm     = ram_state["flag_elm_mr_pokemon"]
        self.prev_flag_sprout2 = ram_state["flag_sprout_tower_2"]
        self.prev_flag_sprout3 = ram_state["flag_sprout_tower_3"]
        obs = np.array([
            ram_state["map_bank"] / 255,  # Normalize to [0,1]
            ram_state["map_number"] / 255,  # Normalize to [0,1]
            ram_state["local_x"] / 255,  # Normalize to [0,1]
            ram_state["local_y"] / 255,  # Normalize to [0,1]
            ram_state["zephyr"],  # Win condition not normalized since it's binary
            ram_state["battle_type"] / 3,  # 3 tpyes: wild, trainer, gym
            ram_state["party_count"] / 6,  # Max 6 Pokemon in party
            ram_state["hp_ratio"],
            ram_state["flag_rival_cherrygrove"] / 255,  # Normalize to [0,1]
            ram_state["flag_elm_mr_pokemon"] / 255,  # Normalize to [0,1]
            ram_state["flag_sprout_tower_2"] / 255,  # Normalize to [0,1]
            ram_state["flag_sprout_tower_3"] / 255,  # Normalize to [0,1]
            min(len(self.visited_tiles) / 2**10, 1.0)  # Normalize visited tiles by an estimated total of 512 unique tiles in the game
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

    def compute_reward(self, ram_state, new_tile):
        """
        Compute the reward based on changes in the RAM state. 
        """
        reward = 0.0
        
        # Reward for winning (getting the Zephyr badge)
        if ram_state["zephyr"]:
            reward += 1000.0  # Large reward for winning the game

        if ram_state["flag_rival_cherrygrove"] and not self.prev_flag_rival:
            reward += 200.0  # Reward for beating the rival in Cherrygrove

        if ram_state["flag_elm_mr_pokemon"] and not self.prev_flag_elm:
            reward += 200.0  # Reward for receiving the egg from Mr. Pokemon

        if ram_state["flag_sprout_tower_2"] and not self.prev_flag_sprout2:
            reward += 200.0  # Reward for entering Sprout Tower 2F
        
        if ram_state["flag_sprout_tower_3"] and not self.prev_flag_sprout3:
            reward += 200.0  # Reward for entering Sprout Tower 3F

        # Penalty for losing (HP drops to 0 in overworld)
        if ram_state["hp_ratio"] <= 0 and ram_state["battle_type"] == 0:
            reward -= 50.0  # Penalty for losing

        # Small reward for Catching a pokemon
        if self.prev_party_count < ram_state["party_count"]:
            reward += 50.0  # Reward for catching a new Pokemon

        # Small exploration reward for visiting new tiles
        if new_tile:
            reward += 1.0  # Reward for visiting a new tile

        # Punishment for not moving (staying on the same tile)
        if not new_tile:
            reward -= 0.05  # Small penalty for not exploring

        # Step penalty to encourage shorter solutions
        reward -= 0.01  # Small penalty per step to encourage efficiency

        return reward
