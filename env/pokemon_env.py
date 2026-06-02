import gymnasium as gym
import numpy as np
from .pyboy_wrapper import PyBoyWrapper
from .ram_reader import RAMReader
from .actions import ACTION_SPACE

MAX_STEPS = 2**16  # Max steps per episode — must match config.py

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
        self.prev_flag_violet_gym = 0
        self.visited_tiles = set()   # Track visited tiles for exploration reward
        self.visited_maps  = set()   # Track visited (bank, map) pairs for map transition reward
        self.episode_maps = set()  # Track maps visited within the current episode for exploration reward
        self.prev_map_bank   = 0
        self.prev_map_number = 0
        self.prev_hp_ratio   = 1.0
        self.prev_battle_type = 0 # Track battle state to detect transitions in/out of battle

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
        reward, reward_info = self.compute_reward(ram_state, new_tile)

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

        self.prev_party_count = ram_state["party_count"]
        self.prev_flag_rival      = ram_state["flag_rival_cherrygrove"]
        self.prev_flag_elm        = ram_state["flag_elm_mr_pokemon"]
        self.prev_flag_violet_gym = ram_state["flag_violet_gym"]
        self.prev_map_bank   = ram_state["map_bank"]
        self.prev_map_number = ram_state["map_number"]
        self.prev_battle_type = ram_state["battle_type"]
        self.prev_hp_ratio    = ram_state["hp_ratio"]
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

        # Read initial RAM state — must happen before prev_flag initialization
        ram_state = self.ram_reader.read_all()

        # Use actual RAM values as baseline so flags only trigger on change, not on non-zero
        self.prev_party_count     = ram_state["party_count"]
        self.prev_flag_rival      = ram_state["flag_rival_cherrygrove"]
        self.prev_flag_elm        = ram_state["flag_elm_mr_pokemon"]
        self.prev_flag_violet_gym = ram_state["flag_violet_gym"]
        self.prev_map_bank   = ram_state["map_bank"]
        self.prev_map_number = ram_state["map_number"]
        self.prev_battle_type = ram_state["battle_type"]
        self.prev_hp_ratio    = ram_state["hp_ratio"]
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

    def compute_reward(self, ram_state, new_tile):
        """
        Compute the reward based on changes in the RAM state.
        Returns (total_reward, breakdown_dict) for TensorBoard monitoring.
        """
        events = 0.0
        exploration = 0.0
        penalties = 0.0

        # Reward for winning (getting the Zephyr badge)
        if ram_state["zephyr"]:
            events += 1000.0

        # 0xD88E bit 6: falling edge (1->0) — EVENT_RIVAL_CHERRYGROVE_CITY (flag 1726), sprite visibility cleared after rival beaten
        RIVAL_BIT = 0x40
        if not (ram_state["flag_rival_cherrygrove"] & RIVAL_BIT) and (self.prev_flag_rival & RIVAL_BIT):
            events += 200.0

        # 0xD7BA bit 6 (0x40): rising edge — player picked up egg from Mr. Pokemon (flag 30)
        MR_POKEMON_BIT = 0x40
        if (ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT) and not (self.prev_flag_elm & MR_POKEMON_BIT):
            events += 100.0

        # 0xD7BA bit 7 (0x80): rising edge — egg delivered to Elm (flag 31)
        ELM_BIT = 0x80
        if (ram_state["flag_elm_mr_pokemon"] & ELM_BIT) and not (self.prev_flag_elm & ELM_BIT):
            events += 200.0

        # 0xD836 bit 4 (0x10): rising edge — Violet City Gym Trainer 1 beaten (flag #1020, empirically verified)
        TRAINER1_BIT = 0x10
        if (ram_state["flag_violet_gym"] & TRAINER1_BIT) and not (self.prev_flag_violet_gym & TRAINER1_BIT):
            events += 100.0

        # 0xD836 bit 3 (0x08): rising edge — Violet City Gym Trainer 2 beaten (flag #1019, empirically verified)
        TRAINER2_BIT = 0x08
        if (ram_state["flag_violet_gym"] & TRAINER2_BIT) and not (self.prev_flag_violet_gym & TRAINER2_BIT):
            events += 100.0

        # Gym-specific battle win: won a battle while inside Violet City Gym (map 10,7)
        # Unlike the general battle win reward (removed in PPO_10 — caused wild grinding near start),
        # this reward is map-constrained: only fires inside the gym → no local optima near start.state.
        # Fires at most 3×/episode: trainer1 + trainer2 + Falkner.
        if self.prev_battle_type > 0 and ram_state["battle_type"] == 0 and ram_state["hp_ratio"] > 0:
            if (ram_state["map_bank"], ram_state["map_number"]) == (10, 7):  # VIOLET_CITY_GYM
                events += 150.0
                
        # Small reward for catching a pokemon
        if self.prev_party_count < ram_state["party_count"]:
            events += 50.0

        # if HP raised significantly out of battle, it means we healed at a Pokemon Center or used a healing item — both good signs of progress
        if ram_state["hp_ratio"] - self.prev_hp_ratio > 0.4 and ram_state["battle_type"] == 0:
            events += 30.0

        # Reward for entering a map never visited before (one-shot per training lifetime)
        # map_bank = map group index from pret/pokegold map_constants.asm
        # (26,10)=Mr.Pokemon house, (3,2)=Sprout Tower 2F, (3,3)=Sprout Tower 3F — all empirically verified
        current_map = (ram_state["map_bank"], ram_state["map_number"])
        if current_map != (self.prev_map_bank, self.prev_map_number):
            if current_map not in self.visited_maps:
                exploration += 20.0
                if current_map == (26, 10):    # MR_POKEMON_HOUSE
                    events += 80.0
                elif current_map == (3, 2):    # SPROUT_TOWER_2F
                    events += 100.0
                elif current_map == (3, 3):    # SPROUT_TOWER_3F
                    events += 150.0
                # elif current_map == (26, 2):   # VIOLET_CITY_WEST
                #    events += 60.0
                # elif current_map == (10, 5):   # VIOLET_CITY_MAIN
                #    events += 80.0
                elif current_map == (10, 7):   # VIOLET_CITY_GYM
                    events += 400.0
                self.visited_maps.add(current_map)

            # Reward for visiting new tiles within the episode to encourage exploration (one-shot per tile per episode)
            if current_map not in self.episode_maps:
                if current_map == (26, 3):    # CHERRYGROVE / ROUTE_30_WEST (nuovo, ~150 step da start)
                    events += 25.0
                elif current_map == (26, 1):  # ROUTE_31
                    events += 50.0            # era +20
                elif current_map == (26, 2):  # VIOLET_CITY_WEST
                    events += 80.0            # era +30
                elif current_map == (10, 5):  # VIOLET_CITY_MAIN
                    events += 100.0           # era +40
                elif current_map == (10, 7):  # VIOLET_CITY_GYM (aggiunto, per-episode)
                    events += 200.0           # era assente (solo +400 one-shot)
                self.episode_maps.add(current_map)

        # Exploration reward for visiting new tiles
        if new_tile:
            exploration += 1.0
        
        # Stuck penalty: discourage revisiting tiles already seen this episode
        if not new_tile:
            penalties -= 0.003

        # Wild battle penalty (PPO_17): each step in a wild battle outside the gym costs -3.0
        # Wild battles (battle_type=1) deplete HP and prevent progress to the gym.
        # Map-constrained to NOT (10,7) so trainer/gym battles inside Violet City Gym are unaffected.
        if ram_state["battle_type"] == 1 and (ram_state["map_bank"], ram_state["map_number"]) != (10, 7):
            penalties -= 3.0

        # Penalty for losing (HP drops to 0 in overworld)
        if ram_state["hp_ratio"] <= 0 and ram_state["battle_type"] == 0:
            penalties -= 50.0

        # Step penalty to encourage shorter solutions
        penalties -= 0.001

        total = events + exploration + penalties
        return total, {"events": events, "exploration": exploration, "penalties": penalties}
