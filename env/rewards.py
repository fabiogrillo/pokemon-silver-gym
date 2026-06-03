"""
Reward function for Pokemon Silver Gym — shared between MLP and CNN envs.

The reward signal depends only on RAM state changes + tile/map tracking,
NOT on the observation representation (pixels vs vector). Both envs use this.

State tracking strategy:
- prev_state: dict of previous-frame RAM values needed for edge detection
- visited_tiles: set of (bank, num, x, y) — episode-scoped (reset each ep)
- visited_maps: set of (bank, num) — LIFETIME-scoped (persists across episodes)
- episode_maps: set of (bank, num) — episode-scoped per-episode waypoint reward
"""

# ── Event flag bitmasks (verified empirically — see training_log.md)
RIVAL_BIT       = 0x40  # 0xD88E bit 6 — falls after rival beaten in Cherrygrove
MR_POKEMON_BIT  = 0x40  # 0xD7BA bit 6 — rises when egg received from Mr. Pokemon
ELM_BIT         = 0x80  # 0xD7BA bit 7 — rises when egg delivered to Elm
TRAINER1_BIT    = 0x10  # 0xD836 bit 4 — Violet City Gym Trainer 1 beaten
TRAINER2_BIT    = 0x08  # 0xD836 bit 3 — Violet City Gym Trainer 2 beaten

# ── Map IDs (bank, number) — verified via test_violet_city_detection.py
GYM_MAP             = (10, 7)   # Violet City Gym (Falkner)
MR_POKEMON_HOUSE    = (26, 10)
SPROUT_TOWER_2F     = (3, 2)
SPROUT_TOWER_3F     = (3, 3)
CHERRYGROVE_R30W    = (26, 3)
ROUTE_31            = (26, 1)
VIOLET_CITY_WEST    = (26, 2)
VIOLET_CITY_MAIN    = (10, 5)


def compute_reward(ram_state, prev_state, new_tile, visited_maps, episode_maps):
    """
    Compute reward + breakdown given current RAM state and tracking sets.

    Args:
        ram_state: dict from RAMReader.read_all() for current frame
        prev_state: dict with previous-frame values. Required keys:
            party_count, flag_rival_cherrygrove, flag_elm_mr_pokemon,
            flag_violet_gym, map_bank, map_number, battle_type, hp_ratio
        new_tile: bool — True if (bank, num, x, y) was never visited this episode
        visited_maps: set of (bank, num) — LIFETIME-scoped; mutated in place
        episode_maps: set of (bank, num) — EPISODE-scoped; mutated in place

    Returns:
        (total_reward, breakdown_dict) where breakdown has keys:
        "events", "exploration", "penalties"
    """
    events = 0.0
    exploration = 0.0
    penalties = 0.0

    # ── Win condition
    if ram_state["zephyr"]:
        events += 1000.0

    # ── Event flag transitions (edge detection)
    if not (ram_state["flag_rival_cherrygrove"] & RIVAL_BIT) and \
            (prev_state["flag_rival_cherrygrove"] & RIVAL_BIT):
        events += 200.0
    if (ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT) and \
            not (prev_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT):
        events += 100.0
    if (ram_state["flag_elm_mr_pokemon"] & ELM_BIT) and \
            not (prev_state["flag_elm_mr_pokemon"] & ELM_BIT):
        events += 200.0
    if (ram_state["flag_violet_gym"] & TRAINER1_BIT) and \
            not (prev_state["flag_violet_gym"] & TRAINER1_BIT):
        events += 100.0
    if (ram_state["flag_violet_gym"] & TRAINER2_BIT) and \
            not (prev_state["flag_violet_gym"] & TRAINER2_BIT):
        events += 100.0

    # ── Gym battle exit (map-constrained, safe vs grinding)
    if prev_state["battle_type"] > 0 and ram_state["battle_type"] == 0 and \
            ram_state["hp_ratio"] > 0:
        if (ram_state["map_bank"], ram_state["map_number"]) == GYM_MAP:
            events += 150.0

    # ── Caught a Pokemon
    if prev_state["party_count"] < ram_state["party_count"]:
        events += 50.0

    # ── Healed outside battle (Pokemon Center or item)
    if ram_state["hp_ratio"] - prev_state["hp_ratio"] > 0.4 and \
            ram_state["battle_type"] == 0:
        events += 30.0

    # ── Map transitions (one-shot lifetime + per-episode waypoints)
    current_map = (ram_state["map_bank"], ram_state["map_number"])
    if current_map != (prev_state["map_bank"], prev_state["map_number"]):
        # Lifetime one-shot bonuses
        if current_map not in visited_maps:
            exploration += 20.0
            if current_map == MR_POKEMON_HOUSE:
                events += 80.0
            elif current_map == SPROUT_TOWER_2F:
                events += 100.0
            elif current_map == SPROUT_TOWER_3F:
                events += 150.0
            elif current_map == GYM_MAP:
                events += 400.0
            visited_maps.add(current_map)

        # Per-episode waypoints (reset each episode)
        if current_map not in episode_maps:
            if current_map == CHERRYGROVE_R30W:
                events += 25.0
            elif current_map == ROUTE_31:
                events += 50.0
            elif current_map == VIOLET_CITY_WEST:
                events += 80.0
            elif current_map == VIOLET_CITY_MAIN:
                events += 100.0
            elif current_map == GYM_MAP:
                events += 200.0
            episode_maps.add(current_map)

    # ── Tile exploration
    if new_tile:
        exploration += 1.0

    # ── Stuck penalty: revisiting tiles already seen this episode
    if not new_tile:
        penalties -= 0.003

    # ── Wild battle penalty (PPO_18 calibration): −1.0/step outside gym
    if ram_state["battle_type"] == 1 and current_map != GYM_MAP:
        penalties -= 1.0

    # ── Death penalty: HP=0 in overworld
    if ram_state["hp_ratio"] <= 0 and ram_state["battle_type"] == 0:
        penalties -= 50.0

    # ── Step penalty
    penalties -= 0.001

    total = events + exploration + penalties
    return total, {"events": events, "exploration": exploration, "penalties": penalties}


def make_prev_state(ram_state):
    """
    Build the prev_state dict from a RAM snapshot.
    Used by envs in reset() (after first RAM read) and after each step().
    """
    return {
        "party_count":             ram_state["party_count"],
        "flag_rival_cherrygrove":  ram_state["flag_rival_cherrygrove"],
        "flag_elm_mr_pokemon":     ram_state["flag_elm_mr_pokemon"],
        "flag_violet_gym":         ram_state["flag_violet_gym"],
        "map_bank":                ram_state["map_bank"],
        "map_number":              ram_state["map_number"],
        "battle_type":             ram_state["battle_type"],
        "hp_ratio":                ram_state["hp_ratio"],
    }
