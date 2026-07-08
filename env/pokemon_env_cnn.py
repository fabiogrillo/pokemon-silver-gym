import gymnasium as gym
import numpy as np
import random
from .pyboy_wrapper import PyBoyWrapper
from .ram_reader import RAMReader
from .actions import ACTION_SPACE
from .rewards import (
    compute_reward, make_prev_state, make_reward_maxes,
    CHERRYGROVE, ROUTE_30_GATE, ROUTE_31, VIOLET_CITY, GYM_MAP,
    ROUTE_29, NEW_BARK, ELM_LAB,
    MR_POKEMON_BIT, ELM_BIT, CORRIDOR_LEGAL,
)
from .frontier_archive import FrontierArchive, cell_key, frontier_score

MAX_STEPS = 2**16  # 65536 — hard per-episode step cap (the dynamic budget grows up to this ceiling).

# Ordered route waypoints (ordinal = index + 1) — drives per-episode navigation progress logging
# to TensorBoard, independent of reward. 0 = start/New Bark … 5 = Violet Gym.
# Ordinal 3 (ROUTE_31, post-gate) = the agent has CLEARED the two-trainer story gate.
WAYPOINT_ORDER = [CHERRYGROVE, ROUTE_30_GATE, ROUTE_31, VIOLET_CITY, GYM_MAP]

DYN_BUDGET_BASE = 2**14  # 16384 — dynamic episode budget base cap (earned-episode-length trick,
                         # Pokémon-Red paper). Start/curriculum episodes begin capped at this short
                         # budget and only earn more steps (up to MAX_STEPS) by reaching new corridor
                         # waypoints, so aimless wandering is truncated fast while genuine progress is
                         # rewarded with room to keep going.

class PokemonEnvCNN(gym.Env):
    """CNN-friendly Gymnasium environment for Pokemon Silver."""
    def __init__(self, rom_path, state_path, headless=True,
                 gif_dir="../runs/gifs/", render_mode=None,
                 gif_every_n_episodes=100, gif_prefix="episode",
                 frontier_root=None, p_frontier=0.0, frontier_max_cells=4000,
                 frontier_cell_k=4, frontier_epsilon=0.1, frontier_max_steps=MAX_STEPS,
                 egg_marker=False, exploration_scale=1.0, confine_to_gym=False,
                 confine_to_corridor=False, dynamic_episode_budget=False,
                 visited_obs=False, dyn_budget_base=DYN_BUDGET_BASE):
        """PyBoy-backed Gymnasium env with Dict obs (RGB image + state vector) for CnnPolicy.

        Go-Explore frontier reset: if `frontier_root` is set, a fraction `p_frontier` of
        resets load a save-state sampled from the shared on-disk archive (states harvested from the
        policy's OWN trajectory) instead of `state_path`. All envs share the same archive dir, so a
        cell found by one worker seeds resets in every worker. No foreign save-state is introduced
        (reset states are the policy's own visited cells), so there is no visual island to segregate
        — the project's central failure mode is avoided.
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
        # Dict obs: RGB screen (-> CNN) + RAM-derived state vector (HP/level/battle/story flags), so
        # the policy can condition on state pixels don't show (e.g. the Route 30 fork looks identical
        # pre/post egg delivery). When `visited_obs` is on, a "visited" crop of this episode's tiles is
        # added as a SEPARATE Dict-obs key rather than a stacked image channel, which keeps battle/menu
        # screens from corrupting it (arXiv:2502.19920 §II-C). Gated (default OFF), so the obs space is
        # bit-identical to the default unless enabled. See `_visited_crop`.
        self.observation_space = gym.spaces.Dict({
            "image":  gym.spaces.Box(low=0, high=255, shape=(72, 80, 3), dtype=np.uint8),
            "vector": gym.spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
            **({"visited": gym.spaces.Box(low=0, high=1, shape=(48, 48), dtype=np.uint8)}
               if visited_obs else {}),
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
        # Battle outcome counters (per episode) — classified on the battle falling edge:
        # our hp 0 → lost · enemy hp 0 on the last in-battle frame → won · otherwise → fled
        self.battles_won = 0
        self.battles_fled = 0
        self.battles_lost = 0
        # True for the run's PRIMARY training state — lets the nav metric measure the true start
        # distribution, uncontaminated by curriculum envs that begin past the waypoints.
        # The generalist starts from egg_delivered_clean.state, so it counts as a start
        # env too (else every episode would log under front/ and nav/reach_* would stay empty).
        START_STATES = ("start.state", "egg_delivered_clean.state")
        self.is_start_env = str(state_path).endswith(START_STATES)
        self._egg_seen = False  # egg-pickup edge detector (set properly in reset())
        # Southward return-front tracker (telemetry only): max chain index reached while
        # carrying the undelivered egg. 0=never left the north pocket, 1=R30-north,
        # 2=Cherrygrove, 3=Route29, 4=NewBark, 5=Elm's lab.
        self.RETURN_ORDER = {(26, 1): 1, (26, 3): 2, (24, 3): 3, (24, 4): 4, (24, 5): 5}
        self.return_progress = 0

        # ── Go-Explore frontier reset ───────────────────────────────────────────────────────────
        self._rng = random.Random()
        self.p_frontier = p_frontier
        self.frontier_cell_k = frontier_cell_k
        self.frontier = (
            FrontierArchive(frontier_root, max_cells=frontier_max_cells,
                            epsilon=frontier_epsilon, rng=self._rng)
            if frontier_root else None
        )
        self._from_frontier = False  # True for episodes RESET from an archived cell (logged as front/)
        self.frontier_max_steps = frontier_max_steps  # frontier episodes use this shorter cap
        self._max_steps = MAX_STEPS  # per-episode truncation cap (set in reset() per origin)
        self.egg_marker = egg_marker  # egg-state image patch — off by default (keeps the obs clean)
        self.exploration_scale = exploration_scale  # weight on the exploration reward
        self.confine_to_gym = confine_to_gym  # end the episode if the agent leaves the gym map
        self.confine_to_corridor = confine_to_corridor  # end the episode if the agent leaves CORRIDOR_LEGAL,
                                              # so off-path maps stop being places where episodes stall
        self.dynamic_episode_budget = dynamic_episode_budget  # earned episode budget: start/curriculum
                                              # episodes begin capped at dyn_budget_base and grow the cap only
                                              # when the episode's max waypoint ordinal increases. Frontier
                                              # episodes are untouched (they keep frontier_max_steps).
        self.dyn_budget_base = dyn_budget_base  # base cap for dynamic_episode_budget (default DYN_BUDGET_BASE)
        self.visited_obs = visited_obs  # gate for the "visited" Dict-obs key (see _visited_crop). Off by
                                              # default, keeping the obs space bit-identical to the default.

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

    def _visited_crop(self, ram):
        """48x48 uint8 crop: 1 on tiles of the CURRENT map already visited THIS EPISODE, in TRUE
        (de-transposed) axes, player at center (24,24). Gated by `visited_obs`.

        env/ram_reader.py's local_x/local_y fields are swapped vs their names (local_x holds
        wYCoord, local_y holds wXCoord — frozen there, trained checkpoints depend on those RAM
        offsets/semantics). Getting this crop's axes right is exactly what makes the signal useful:
        drawn transposed it never tracks the player's real movement. Every geometry consumer must
        un-swap at its own boundary; canonical reference:
        agents/rl/map_layout.ram_to_image_px. Un-swap here too: true_x = ram['local_y'],
        true_y = ram['local_x'] — so walking true-EAST moves the mark along crop COLUMNS, never rows.
        `self.visited_tiles` entries are stored as (bank, num, ram_local_x, ram_local_y), i.e.
        (bank, num, true_y, true_x) — un-swap when reading them back out too.
        """
        crop = np.zeros((48, 48), dtype=np.uint8)
        bank, num = ram["map_bank"], ram["map_number"]
        true_x, true_y = ram["local_y"], ram["local_x"]  # un-swap (see docstring)
        for (t_bank, t_num, t_ram_x, t_ram_y) in self.visited_tiles:
            if t_bank != bank or t_num != num:            # same-map filter
                continue
            t_true_x, t_true_y = t_ram_y, t_ram_x          # un-swap the stored tuple too
            row = 24 + (t_true_y - true_y)
            col = 24 + (t_true_x - true_x)
            if 0 <= row < 48 and 0 <= col < 48:
                crop[row, col] = 1
        return crop

    # Optional egg-state visual marker: an 8x8 px corner patch stamped into the image encoding the egg
    # quest state (none / carrying / delivered), so the CNN can distinguish pre-pickup from carrying at
    # tiles where the screen looks identical either way. The egg bit in the state vector alone is a weak
    # signal; a colored patch is trivially CNN-detectable. Shape stays (72,80,3). Off by default.
    _EGG_MARKER = {"none": (0, 0, 0), "carrying": (255, 0, 0), "delivered": (0, 255, 0)}

    def _get_obs(self, screen, ram_state):
        """Dict observation: downsampled RGB screen (-> CNN, with the optional egg-state corner marker)
        + state vector."""
        rgb = screen[:, :, :3]                          # drop alpha
        image = rgb[::2, ::2].astype(np.uint8)          # downsample to 72x80 (this is a fresh copy)
        if self.egg_marker:                             # default OFF
            flags = ram_state["flag_elm_mr_pokemon"]
            if flags & ELM_BIT:
                state = "delivered"
            elif flags & MR_POKEMON_BIT:
                state = "carrying"
            else:
                state = "none"
            image[0:8, 0:8] = self._EGG_MARKER[state]   # stamp the egg-state patch (top-left corner)
        obs = {"image": image, "vector": self._state_vector(ram_state)}
        if self.visited_obs:
            obs["visited"] = self._visited_crop(ram_state)
        return obs
    
    def step(self, action):
        screen = self.pyboy.step(action, n=16) # Advance the emulator by 16 frames (1/4 second at 60 FPS)

        # GIF capture: only on selected episodes, and every 3rd env-step to reduce GIF size
        # (1 env-step = 16 emulator ticks; sampling every 3 env-steps = ~5 GIF frames per second of gameplay)
        if self.capture_gif and self.episode_count % self.gif_every == 0 and self.steps % 3 == 0:
            self.gif_frames.append(screen[:, :, :3].copy())  # full frame, drop alpha

        ram_state = self.ram_reader.read_all()

        # DIRECTIONAL tile-novelty reset on egg pickup. Re-arm ONLY the southern delivery corridor
        # (Route 29 -> New Bark -> Elm's lab); the north pocket, Route 30 gate, AND Cherrygrove stay
        # SPENT. So post-pickup the nearest fresh-tile income is ROUTE 29 -- a directional pull south,
        # the exact link where the return leg tends to stall. Leaving Cherrygrove re-armed instead let
        # the agent re-milk it and the return leg stalled. One-shot, latched on the egg flag, and
        # unfarmable (pre-pickup behavior is unchanged, so there is no pickup-avoidance).
        RETURN_CORRIDOR = {ROUTE_29, NEW_BARK, ELM_LAB}
        egg_now = bool(ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT)
        if egg_now and not self._egg_seen:
            self._egg_seen = True
            self.visited_tiles = {t for t in self.visited_tiles if (t[0], t[1]) not in RETURN_CORRIDOR}

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
            new_max_waypoint = max(self.max_waypoint, WAYPOINT_ORDER.index(current_map) + 1)
            # dynamic episode budget — grow the step cap the instant the episode reaches a NEW
            # waypoint (long episodes must be earned). Frontier-origin episodes keep their existing
            # (shorter) frontier_max_steps cap untouched — this only applies to start/curriculum episodes.
            if (self.dynamic_episode_budget and not self._from_frontier
                    and new_max_waypoint > self.max_waypoint):
                self._max_steps = min(MAX_STEPS, self.dyn_budget_base * (1 + new_max_waypoint))
            self.max_waypoint = new_max_waypoint

        # Track the southward return front while carrying the undelivered egg (telemetry only)
        if (egg_now and not (ram_state["flag_elm_mr_pokemon"] & ELM_BIT)
                and current_map in self.RETURN_ORDER):
            self.return_progress = max(self.return_progress, self.RETURN_ORDER[current_map])

        # ── Frontier harvest: snapshot promising states into the shared archive. Only
        # frontier-relevant cells (egg in hand/delivered, or inside the gym) are harvested — pre-egg
        # overworld is already practiced from start.state. The ~200KB save_state is taken LAZILY:
        # add() invokes the lambda only when the cell is new or has a higher score.
        if self.frontier is not None:
            egg_delivered = bool(ram_state["flag_elm_mr_pokemon"] & ELM_BIT)
            if egg_now or current_map == GYM_MAP:
                # frontier_score's max_waypoint must be THIS CELL's own waypoint ordinal at capture
                # time, NOT self.max_waypoint (the running EPISODE max). Otherwise an episode that
                # earlier reached Violet City would stamp that high ordinal onto a LATER, shallower
                # New Bark cell, and add()'s "higher score replaces" would overwrite and destroy the
                # shallow cell the corridor still needs practiced. Compute the cell's own ordinal
                # from current_map directly, mirroring the self.max_waypoint update above but WITHOUT
                # the running max().
                cell_waypoint = WAYPOINT_ORDER.index(current_map) + 1 if current_map in WAYPOINT_ORDER else 0
                score = frontier_score(egg_now, egg_delivered, self.return_progress,
                                       cell_waypoint, ram_state["gym_trainers_beaten"])
                key = cell_key(ram_state, k=self.frontier_cell_k)
                self.frontier.add(key, score, self.pyboy.save_state_bytes)

        # Battle outcome classification on the falling edge (prev in-battle → now overworld).
        # prev_state holds the LAST in-battle frame, so its enemy_hp_ratio distinguishes KO from flee.
        if self.prev_state.get("battle_type", 0) > 0 and ram_state["battle_type"] == 0:
            if ram_state["hp_ratio"] <= 0:
                self.battles_lost += 1
            elif self.prev_state.get("enemy_hp_ratio", 1.0) <= 0:
                self.battles_won += 1
            else:
                self.battles_fled += 1

        # Compute the reward based on RAM state changes and exploration
        reward, reward_info = compute_reward(
            ram_state, self.prev_state, new_tile, self.visited_maps, self.episode_maps,
            self.reward_maxes, new_tile_lifetime, exploration_scale=self.exploration_scale,
        )

        terminated = ram_state['zephyr'] or (ram_state['hp_ratio'] <= 0 and ram_state['battle_type'] == 0)  # Episode ends if we win or lose

        # confine-to-gym — leaving GYM_MAP ends the episode so the agent CANNOT wander out
        # to wild-grind (the stable basin that capped 083-086 at ~40% badge). Forces it to solve the gym.
        # Off by default (corridor task); enabled via config.CONFINE_TO_GYM for the gym slice.
        if self.confine_to_gym and current_map != GYM_MAP:
            terminated = True

        # confine-to-corridor: leaving CORRIDOR_LEGAL ends the episode, so the agent can't wander into
        # Dark Cave / Sprout Tower / other off-path maps and stall there. Off by default; both this and
        # confine_to_gym are independently usable.
        if self.confine_to_corridor and current_map not in CORRIDOR_LEGAL:
            terminated = True

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
            # from_start is FALSE for frontier-reset episodes so they log under front/ — the nav/
            # gate metrics then measure ONLY true start.state episodes, uncontaminated by episodes
            # that began deep along the policy's own trajectory.
            "from_start": self.is_start_env and not self._from_frontier,
            # Story/combat progress flags — consumed by InfoLoggerCallback at episode end
            # to compute nav/egg_*_rate, nav/*_trainers_mean and nav/badge_rate.
            "egg_received": bool(ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT),
            "egg_delivered": bool(ram_state["flag_elm_mr_pokemon"] & ELM_BIT),
            "route_trainers_beaten": ram_state["route_trainers_beaten"],
            "gym_trainers_beaten": ram_state["gym_trainers_beaten"],
            "battles_won": self.battles_won,
            "battles_fled": self.battles_fled,
            "battles_lost": self.battles_lost,
            "lead_level": ram_state["lead_level"],
            "return_progress": self.return_progress,
        }

        # a FRONTIER episode ends the moment it delivers the egg — its job (practicing the
        # carry→deliver backtrack) is done. Running on would add ~63k steps of off-task post-delivery
        # wandering, the aimless gradient that eroded the start policy into wandering in 059-061.
        # (Computed from the OLD prev_state, before it is overwritten below.)
        just_delivered = (bool(ram_state["flag_elm_mr_pokemon"] & ELM_BIT)
                          and not bool(self.prev_state.get("flag_elm_mr_pokemon", 0) & ELM_BIT))

        self.prev_state = make_prev_state(ram_state) # Store only the relevant RAM values for reward edge detection
        self.steps += 1
        truncated = self.steps >= self._max_steps or (self._from_frontier and just_delivered)

        obs = self._get_obs(screen, ram_state)  # Dict obs: screen image + self-state vector

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset to the loaded state (or a sampled frontier cell). Returns (obs, info)."""
        if self.gif_frames:
            path = f"{self.gif_dir}/{self.gif_prefix}_ep{self.episode_count:05d}.gif"
            self.pyboy.capture_gif(path, self.gif_frames)
        self.gif_frames = []  # Clear frames for the next episode

        # Go-Explore frontier reset: with prob p_frontier, restart from an archived cell
        # sampled from the policy's OWN trajectory (if the shared archive is non-empty). Otherwise the
        # normal start.state reset. Episodes from a frontier cell are tagged (_from_frontier) so they
        # log under front/, keeping the nav/ start-state gate metrics clean.
        state_bytes = None
        if self.frontier is not None and self._rng.random() < self.p_frontier:
            state_bytes = self.frontier.sample()
        if state_bytes is not None:
            screen = self.pyboy.reset_from_bytes(state_bytes)
            self._from_frontier = True
        else:
            screen = self.pyboy.reset()
            self._from_frontier = False
        # frontier episodes get the shorter cap (they end on delivery anyway); start
        # episodes keep the full MAX_STEPS for the whole start->pickup->deliver->gym trajectory.
        # dynamic_episode_budget overrides the start-episode cap to the earned-budget base
        # (grows in step() as new waypoints are reached); frontier episodes are unaffected.
        if self._from_frontier:
            self._max_steps = self.frontier_max_steps
        elif self.dynamic_episode_budget:
            self._max_steps = self.dyn_budget_base
        else:
            self._max_steps = MAX_STEPS

        self.steps = 0
        self.episode_count += 1

        self.visited_tiles = set()
        self.episode_maps = set()
        self.max_waypoint = 0
        self.battles_won = 0
        self.battles_fled = 0
        self.battles_lost = 0
        self.return_progress = 0

        ram_state = self.ram_reader.read_all()
        self.prev_state = make_prev_state(ram_state)
        self.reward_maxes = make_reward_maxes(ram_state)
        # Seed the egg-pickup edge detector from the loaded state, so savestates that already
        # hold the egg do NOT trigger the mid-episode tile-novelty reset.
        self._egg_seen = bool(ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT)

        init_tile = (ram_state["map_bank"], ram_state["map_number"], ram_state["local_x"], ram_state["local_y"])
        self.visited_tiles.add(init_tile)
        self.visited_tiles_lifetime.add(init_tile)  # lifetime set is NOT cleared on reset
        self.visited_maps.add((ram_state["map_bank"], ram_state["map_number"]))
        return self._get_obs(screen, ram_state), {}

    def render(self):
        """Return the screen as RGB if render_mode == 'rgb_array' (SDL2 renders to its own window)."""
        if self.render_mode == "rgb_array":
            screen = self.pyboy.pyboy.screen.ndarray
            return screen[:, :, :3]  # Drop alpha
        return None

    def close(self):
        """Stop the emulator."""
        self.pyboy.pyboy.stop()