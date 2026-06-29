"""
Reward function for Pokemon Silver Gym — shared between MLP and CNN envs.

The reward signal depends only on RAM state changes + tile/map tracking,
NOT on the observation representation (pixels vs vector). Both envs use this.

── PPO_CNN_5 realignment (Whidden V2 recipe) ────────────────────────────────
After 22 runs with badge=0/10, the reward was REALIGNED to the recipe that
actually clears the first gym in PokemonRedExperiments V2:
  - Every term is kept the SAME ORDER OF MAGNITUDE (single digits), then the
    whole reward is multiplied by REWARD_SCALE (0.1). No more +1000 spikes that
    a reward normalizer's running-std drowns out (the "policy segregation" cause).
  - DENSE coordinate-exploration (+per new tile) is the PRIMARY driver.
  - LEVEL and OPPONENT-LEVEL rewards added — needed both as a progress signal and
    because Totodile must level up to beat Falkner (wild-battle penalty REMOVED).
  - Story progress is a MONOTONIC weighted event score (delta of running max), not big
    one-shot edge bonuses; rival is latched separately (its bit is not a clean latch).
  - PPO_CNN_8: egg quest weighted heavily (received +3, delivered +5) — delivery opens the
    two-trainer STORY GATE on Route 30 that blocked CNN_5/6/7. Trained via REVERSE CURRICULUM.
  - Per-episode mega-waypoints, wild-battle penalty, and the gym damage reward are
    GONE (the last is deferred to the Phase-4 battle-competence work).

── PPO_CNN_10 redesign (event-dominant reward) ──────────────────────────────
PPO_CNN_9_gymtest proved by direct arithmetic that exploration income structurally
dominated story/combat income (touring Violet ≈ 12 pre-scale vs full gym fight ≈ 16
discounted by death risk) — the root cause of EVERY avoidance/grinding local optimum.
Realigned to PWhiddy's actual ratio (events/badges dominate exploration ~10-50×):
  - Story chain raised: egg received +8, delivered +12, rival +3, badge +30.
  - NEW: per-trainer-beaten event rewards (+5 each) for Route 30/31 trainers and the
    two gym trainers — combat pays exactly where combat is mandatory.
  - Level cap hack (lead<=13) replaced by PWhiddy-style SATURATION of the summed
    party levels: full value up to 15, then 1/4 value — grinding self-extinguishes.
  - Sanity math (pre-scale/episode): wander ≈ 20 · grind ≈ 19 · avoid-battles ≈ 33 ·
    STORY-OPTIMAL ≈ 136. Locally at the gym: fight chain 55 vs tour-Violet 12.

State tracking strategy:
- prev_state:   previous-frame RAM values needed for edge detection (make_prev_state)
- reward_maxes: per-episode running maxima for level/opponent/event rewards (make_reward_maxes)
- visited_tiles:          set of (bank, num, x, y) — EPISODE-scoped → small "trail-following" reward
- visited_tiles_lifetime: set of (bank, num, x, y) — LIFETIME-scoped → dominant "frontier-expansion"
                          reward (PPO_CNN_6 fix: only genuinely-new territory pays, so re-treading the
                          known corridor can't satisfy the agent — the frontier keeps advancing)
- visited_maps:  set of (bank, num) — LIFETIME-scoped (persists across episodes)
- episode_maps:  set of (bank, num) — episode-scoped (kept for compatibility; no longer
                 carries reward in the realigned scheme)
"""

# ── Global reward scale (Whidden multiplies the whole step reward by 0.1)
REWARD_SCALE = 0.1

# ── Event flag bitmasks (verified empirically — see training_log.md)
RIVAL_BIT       = 0x40  # 0xD88E bit 6 — FALLS after rival beaten in Cherrygrove (not a clean latch)
MR_POKEMON_BIT  = 0x40  # 0xD7BA bit 6 — rises when egg received from Mr. Pokemon (latch)
ELM_BIT         = 0x80  # 0xD7BA bit 7 — rises when egg delivered to Elm (latch)
TRAINER1_BIT    = 0x10  # 0xD836 bit 4 — Violet City Gym Trainer 1 beaten (latch)
TRAINER2_BIT    = 0x08  # 0xD836 bit 3 — Violet City Gym Trainer 2 beaten (latch)

# ── Map IDs (bank, number) — RE-VERIFIED 2026-06-10 by walking the full route with save_state.py.
# The egg quest is the gate key: two trainers block Route 30 north (ROUTE_30_GATE) until the egg is
# delivered to Elm; only then can the agent pass (26,1)→(26,2) and continue toward Violet.
NEW_BARK         = (24, 4)   # New Bark Town (start)
ELM_LAB          = (24, 5)   # Prof. Elm's lab (egg delivery target)
ROUTE_29         = (24, 3)   # Route 29 (New Bark ↔ Cherrygrove)
CHERRYGROVE      = (26, 3)   # Cherrygrove City
MR_POKEMON_HOUSE = (26, 10)  # egg pickup (pre-gate, a detour off Route 30)
ROUTE_30_GATE    = (26, 1)   # Route 30 north — holds the two-trainer STORY GATE (where CNN_5/6/7 stalled)
ROUTE_31         = (26, 2)   # Route 31, POST-GATE (near Dark Cave) — reaching here = the gate is CLEARED
DARK_CAVE        = (3, 70)   # dead-end without Flash — off-path exploration sink
VIOLET_GATEHOUSE = (26, 11)  # gate house between Route 31 and Violet City
VIOLET_CITY      = (10, 5)   # Violet City main
GYM_MAP          = (10, 7)   # Violet City Gym (Falkner)

# agent_080: ONLY these on-corridor maps pay the EXPLORATION reward (new-tile + new-map). Off-path maps —
# Dark Cave & other caves (bank 3), Sprout Tower / Pokémon School, houses, side routes — pay NOTHING, so
# EXPLORATION_SCALE can't lure the agent into dead-end maps. Root cause of the agent_079 Route-31 plateau:
# each cave floor / tower floor was a fresh "new map" worth +0.4 (at scale 4), diverting it off the path
# (235 cave cells harvested!). Combat / level / heal / badge rewards are UNAFFECTED — the gym fight already
# works (front/badge_rate ~0.7), so the agent still does everything once it is on the corridor.
CORRIDOR_WHITELIST = {NEW_BARK, ROUTE_29, CHERRYGROVE, ROUTE_30_GATE, ROUTE_31,
                      VIOLET_GATEHOUSE, VIOLET_CITY, GYM_MAP}

# agent_065: the SOUTHERN DELIVERY CORRIDOR (Cherrygrove → Route29 → NewBark → Elm). Used for the dense
# per-tile southward-carry pull (see compute_reward) — the fix for the carry-NAVIGATION wall (ESCALATION #3).
SOUTH_CORRIDOR = {CHERRYGROVE, ROUTE_29, NEW_BARK, ELM_LAB}

# Per-episode forward waypoints — modest directional breadcrumbs that pull the policy through the
# map-door chokepoints (the flat per-tile reward gives no gradient toward a specific exit/door tile).
# LATCHED per episode (no cycling), small (no scale domination), and SAFE only because the reverse
# curriculum is single-start: a MIXED curriculum + per-episode bonuses caused the CNN_2..4 segregation.
WAYPOINT_REWARDS = {
    ROUTE_31:         2.0,   # crossed past the gate into Route 31
    VIOLET_GATEHOUSE: 2.0,   # entered the Route 31 ↔ Violet gatehouse (the building-door chokepoint)
    VIOLET_CITY:      2.0,   # reached Violet City
    GYM_MAP:          2.0,   # reached the gym
}

# Return-leg breadcrumbs (PPO_CNN_10b): per-episode latched map-entry bonuses, paid ONLY while
# carrying the undelivered egg. PPO_CNN_10 @41M: egg pickup oscillated at 3-5% but delivery stayed
# flat ZERO — the ~600-step backtrack crosses already-visited tiles paying nothing, so the delivery
# event was never even EXPERIENCED and could not enter the gradient. These factorize the backtrack
# into learnable sub-goals. The egg gating makes cycling impossible: they pay nothing before pickup
# and nothing after delivery.
# Return-leg breadcrumbs (PPO_CNN_10g): per-episode LATCHED map-entry bonuses paid ONLY while
# carrying the undelivered egg — the FULL southward chain, positives only.
# History: 10e parked in Mr.Pokemon's house post-pickup (reward-ghost attractor). 10f tried
# POTENTIAL-BASED shaping (south +1 / north −1) — unhackable within the carrying state, but the
# agent dodged the state itself: pickup rate collapsed 0.87→0.27 because its north-dwelling habit
# bled −0.1 per transition, and skipping the egg ended the bleed. Lesson: NEVER put negatives on
# a state the agent can simply avoid entering. Latched positives have no bleed (pickup stays
# strictly profitable), no cycling (latched), and pay on every southward first-entry.
# NOTE (agent_064, RULED OUT): scaling these into a STRONG southward staircase (2/5/12/20/30) FAILED — it
# destabilized the value fn (norm_reward=False; +30 spikes) and didn't cross the barren gap (sparse map-entry
# bonuses sit BEYOND the gap, no gradient ACROSS it). front/return_progress DECLINED, pickup eroded. The real
# wall (carry-NAVIGATION south vs an entrenched northward HABIT) is an exploration problem, not a reward-
# magnitude one — see training_log.md "ESCALATION #3". Reverted to the modest original values.
RETURN_BREADCRUMBS = {
    ROUTE_30_GATE: 1.0,  # out of the house pocket, heading south
    CHERRYGROVE:   2.0,  # back south through Route 30
    ROUTE_29:      3.0,  # the front reached Cherry then turned back — modest pull on the links
    NEW_BARK:      3.0,
    ELM_LAB:       2.0,  # at the delivery doorstep
}


def _event_score(ram_state):
    """Weighted, monotonic score of LATCHED story flags (safe for delta-of-max).

    The egg quest is the dominant pre-badge milestone — delivering the egg to Elm is what OPENS
    the two-trainer gate on Route 30, so it is weighted heavily enough to be worth the backtrack
    (Mr. Pokemon → back to Elm) that exploration reward alone would discourage.
    Excludes the rival flag (handled separately via a latched falling edge, because its bit rises
    again after leaving the area) and the badge (rewarded via zephyr).
    """
    s = 0.0
    if ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT:  # egg received from Mr. Pokemon
        s += 8.0
    if ram_state["flag_elm_mr_pokemon"] & ELM_BIT:         # egg delivered to Elm → OPENS THE GATE
        s += 30.0                                          # 12→20→30 (10k: thunderclap when it lands)
    # Per-trainer-beaten rewards: each win ≈ 250 tiles of income, exactly at the chokepoints
    # where combat is mandatory (the direct fix for the learned total-battle-avoidance of
    # crossing_wp / gymtest). Counts come from latched event flags → monotonic, delta-safe.
    s += 5.0 * ram_state.get("route_trainers_beaten", 0)   # Joey, Mikey, Don (R30), Wade (R31)
    s += 5.0 * ram_state.get("gym_trainers_beaten", 0)     # Bird Keepers Rod & Abe
    return s


def compute_reward(ram_state, prev_state, new_tile, visited_maps, episode_maps,
                   reward_maxes=None, new_tile_lifetime=False, exploration_scale=1.0):
    """
    Compute reward + breakdown given current RAM state and tracking sets.

    Args:
        ram_state:   dict from RAMReader.read_all() for current frame
        prev_state:  dict from make_prev_state() — previous-frame values for edge detection
        new_tile:    bool — True if (bank, num, x, y) was never visited THIS EPISODE
        visited_maps: set of (bank, num) — LIFETIME-scoped; mutated in place
        episode_maps: set of (bank, num) — EPISODE-scoped; mutated in place (compat only)
        reward_maxes: dict from make_reward_maxes() — per-episode running maxima; mutated
                      in place. If None (legacy MLP env), level/opponent/event rewards are
                      skipped and only exploration + heal + death + win remain.
        new_tile_lifetime: bool — True if (bank, num, x, y) was never visited in ANY episode
                      (LIFETIME-new). Drives the frontier-expansion reward. Defaults False so the
                      legacy MLP env (which doesn't track lifetime tiles) keeps working unchanged.

    Returns:
        (total_reward, breakdown_dict) with keys "events", "exploration", "penalties".
        All values are already multiplied by REWARD_SCALE (total == sum of breakdown).
    """
    events = 0.0
    exploration = 0.0
    penalties = 0.0

    # ── Win condition: must beat touring Violet City (~600 tiles = 12 pre-scale) BY ITSELF —
    # the gymtest agent rationally walked out of the gym when the badge paid only 10.
    if ram_state["zephyr"]:
        events += 30.0

    # ── Exploration: dense per-new-tile bonus (EPISODE-scoped) — the primary navigation driver.
    # PPO_CNN_7 reverted the CNN_6 lifetime-novelty experiment: under pure single-start it saturated
    # into a reward death-spiral (the frontier was never DISCOVERED, so the lifetime bonus never fired
    # there, while the re-tread reward was stripped away). CNN_7 fixes discovery via a small curriculum
    # instead. (`new_tile_lifetime` is still tracked by the env but unused here — kept for future use.)
    # agent_080: exploration pays ONLY on-corridor (CORRIDOR_WHITELIST) — entering caves / Sprout Tower /
    # houses / side routes earns nothing, so the agent isn't lured off the path to Falkner.
    current_map = (ram_state["map_bank"], ram_state["map_number"])
    on_corridor = current_map in CORRIDOR_WHITELIST
    if new_tile:                        # agent_082: whitelist REVERTED (it hurt — stuck at Cherrygrove);
        exploration += 0.02             # full exploration income is what powers path discovery.

    # ── Lifetime map milestone: small one-shot bonus the first time any map is entered

    # ── agent_065: DENSE SMALL southward-carry pull (the ESCALATION #3 fix for carry-NAVIGATION). +0.05 for
    # each NEW (episode-scoped) tile visited while CARRYING the undelivered egg in the southern delivery
    # corridor. This is a DENSE gradient that exists ACROSS the Cherrygrove→Route29 barren gap (vs 064's sparse
    # map-entry bonuses BEYOND it), and SMALL so it can't destabilize the value fn (vs 064's +30 spikes). While
    # carrying, going NORTH re-treads known tiles (+0); going SOUTH hits new corridor tiles (+0.02+0.05=0.07) →
    # a clear southward gradient for PPO to shift the entrenched northward habit. Positive-only + episode-latched
    # (no pickup-dodge, no cycling; a finite map can't be milked).
    carrying_egg_065 = ((ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT)
                        and not (ram_state["flag_elm_mr_pokemon"] & ELM_BIT))
    if new_tile and carrying_egg_065 and current_map in SOUTH_CORRIDOR:
        exploration += 0.05
    if current_map != (prev_state["map_bank"], prev_state["map_number"]):
        if current_map not in visited_maps:
            exploration += 1.0          # agent_082: whitelist reverted — all new maps pay again
            visited_maps.add(current_map)
        episode_maps.add(current_map)  # kept for compatibility
        # Per-episode directional breadcrumb (CNN_8): modest +2 the first time each forward waypoint is
        # reached THIS episode → a gradient through the map-door chokepoints the tile reward can't create.
        if (reward_maxes is not None and current_map in WAYPOINT_REWARDS
                and current_map not in reward_maxes["waypoints"]):
            events += WAYPOINT_REWARDS[current_map]
            reward_maxes["waypoints"].add(current_map)
        # Return-leg breadcrumb (PPO_CNN_10b): fires on map ENTRY, only while carrying the
        # undelivered egg, latched per episode — directional gradient for the Elm backtrack.
        carrying_egg = ((ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT)
                        and not (ram_state["flag_elm_mr_pokemon"] & ELM_BIT))
        if (reward_maxes is not None and current_map in RETURN_BREADCRUMBS
                and carrying_egg
                and current_map not in reward_maxes["return_waypoints"]):
            events += RETURN_BREADCRUMBS[current_map]
            reward_maxes["return_waypoints"].add(current_map)

    # ── Max-based progress rewards (need persistent per-episode running maxima)
    if reward_maxes is not None:
        # Level reward with SATURATION (replaces the lead<=13 cap hack): full value while the summed
        # party levels are <= 15, then 1/4 value beyond — 5→15 pays 10.0 (the levels needed for Falkner),
        # 15→40 grinding pays only 6.25 TOTAL, so grinding self-extinguishes instead of hitting a cliff.
        # (PWhiddy/puffer pattern: min(s, K + (s-K)/4).)
        level_sum = sum(ram_state["party_levels"])
        if level_sum > reward_maxes["level_sum"]:
            scaled_new = min(level_sum, 15.0 + (level_sum - 15.0) / 4.0)
            scaled_old = min(reward_maxes["level_sum"], 15.0 + (reward_maxes["level_sum"] - 15.0) / 4.0)
            events += 1.0 * (scaled_new - scaled_old)
            reward_maxes["level_sum"] = level_sum

        # Opponent-level reward: reward facing progressively stronger opponents.
        # GUARD: enemy RAM (0xD0FC) is garbage outside battle, so only read it in-battle
        # and clamp to a legal level range — a stale spike must not lock the running max.
        op_level = ram_state["enemy_lead_level"]
        if ram_state["battle_type"] > 0 and 0 < op_level <= 100 \
                and op_level > reward_maxes["op_level"]:
            events += 0.2 * (op_level - reward_maxes["op_level"])
            reward_maxes["op_level"] = op_level

        # Event reward: delta of a weighted, monotonic story-flag score. The egg quest dominates
        # (received +3, delivered +5 — delivery opens the two-trainer gate on Route 30 north).
        event_score = _event_score(ram_state)
        if event_score > reward_maxes["event_score"]:
            events += event_score - reward_maxes["event_score"]
            reward_maxes["event_score"] = event_score

        # Mr. Pokemon's house: per-episode LATCHED breadcrumb (+2) toward the egg pickup — but ONLY
        # while the egg is still needed. Once received, that house is a useless DEAD-END detour on the
        # RIGHT fork of Route 30; rewarding it lured the CNN_8 Stage-1 agent off the correct (LEFT) path
        # to Route 31, where it got stuck. So gate the bonus on "egg not yet received".
        if (current_map == MR_POKEMON_HOUSE and not reward_maxes["mrpokemon_done"]
                and not (ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT)):
            events += 2.0
            reward_maxes["mrpokemon_done"] = True

        # Rival: latched falling edge of RIVAL_BIT — pays at most once per episode
        # (the bit rises again on leaving the area, so we must latch it ourselves).
        # +3: a (near-)forced battle win on the optimal path — pro-combat signal.
        if (not reward_maxes["rival_done"]
                and not (ram_state["flag_rival_cherrygrove"] & RIVAL_BIT)
                and (prev_state["flag_rival_cherrygrove"] & RIVAL_BIT)):
            events += 3.0
            reward_maxes["rival_done"] = True

    # ── Battle WIN reward (PPO_CNN_10b eval finding): the agent learned hit-and-flee because
    # battles paid NOTHING (they only block tile income) — it chip-damaged itself to blackout.
    # +2.0 on the battle falling edge with the enemy VERIFIED at 0 HP (KO), nothing for fleeing.
    # No flee penalty: every battle penalty in history (PPO_17/18) taught grass avoidance instead.
    # Differs from the ruled-out flat +15 win reward (PPO_10): KO-verified, modest, and CAPPED at
    # 10 wins/episode (max 2.0 post-scale ≈ 1/6 of the story chain) so grinding can't dominate.
    # Win cap 10 → 5 → 2 (agent_052): the battle attractor kept capturing the start envs at the
    # first grass (probe @051: 99% of time on Route 29, 14 wins + 28 flees / 10k steps, egg never
    # taken). At cap 2 (0.4 post) battle income stops competing with the quest (~14 post) while the
    # battle COMPETENCE stays in the weights for the gym.
    if (reward_maxes is not None
            and prev_state["battle_type"] > 0 and ram_state["battle_type"] == 0
            and ram_state["hp_ratio"] > 0
            and prev_state["enemy_hp_ratio"] <= 0
            and reward_maxes["wins_rewarded"] < 2):
        events += 2.0
        reward_maxes["wins_rewarded"] += 1

    # ── Healed outside battle (encourages Pokemon Center use before the gym)
    if ram_state["hp_ratio"] - prev_state["hp_ratio"] > 0.4 and \
            ram_state["battle_type"] == 0:
        events += 2.0

    # ── Death penalty: HP=0 in overworld → episode ends. RAISED -1 → -8 (PPO_CNN_10d): at -1
    # (-0.1 post) death was nearly free and RESET the win cap, creating the 10c "kamikaze grinder"
    # loop — win 4-5 battles, die at 2.5k steps, respawn with a fresh win budget (ep_rew was RISING
    # on that cycle). At -8 (-0.8 post ≈ 40% of the max win budget) survival out-earns suicide:
    # +2.0 wins − 0.8 death < +2.0 wins + tiles + story chain alive. Also synergizes with the heal
    # reward (+2 on big HP recovery outside battle).
    if ram_state["hp_ratio"] <= 0 and ram_state["battle_type"] == 0:
        penalties -= 8.0

    # ── Gym damage reward (PPO_CNN_9 Phase-A gym validation): dense "deal damage to WIN" signal,
    # MAP-CONSTRAINED to Falkner's gym so it cannot reward wild grinding (the PPO_16 lesson). Rewards
    # reducing the enemy's HP (and KOs: a faint is a +1.0 HP drop) during a battle inside the gym.
    if current_map == GYM_MAP and ram_state["battle_type"] > 0 and prev_state["battle_type"] > 0:
        delta = prev_state["enemy_hp_ratio"] - ram_state["enemy_hp_ratio"]
        if delta > 0:
            events += 3.0 * delta

    # Removed for good (see training_log.md "Ruled out"): per-episode mega-waypoints, global wild-battle
    # penalty, flat step penalty, catch-pokemon bonus, big one-shot map milestones.

    events *= REWARD_SCALE
    # agent_071 (Phase 2): exploration_scale=0 zeros the new-tile/new-map (and Phase-1 dense) reward so the
    # agent has NO incentive to leave the gym for new tiles — the gym FIGHT (events) becomes the only income.
    exploration *= REWARD_SCALE * exploration_scale
    penalties *= REWARD_SCALE
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
        "enemy_hp_ratio":          ram_state["enemy_hp_ratio"],  # kept for Phase-4 damage reward
    }


def make_reward_maxes(ram_state):
    """
    Build the per-episode running-maxima dict from the initial RAM snapshot.
    Called by the env in reset() so the agent is NOT rewarded for its starting levels
    or for flags already set in the loaded save state. Mutated in place by compute_reward.
    """
    start_map = (ram_state["map_bank"], ram_state["map_number"])
    return {
        "level_sum":      sum(ram_state["party_levels"]),
        "op_level":       0,
        "event_score":    _event_score(ram_state),
        "rival_done":     False,
        "mrpokemon_done": False,
        "wins_rewarded":  0,
        # Pre-seed with the start map if it is itself a waypoint, so a stage that BEGINS on a waypoint
        # doesn't reward re-entering it (only genuinely NEW forward waypoints pay).
        "waypoints":      {start_map} if start_map in WAYPOINT_REWARDS else set(),
        # Same pre-seed for return breadcrumbs, relevant only if a save starts egg-in-hand on one
        # of these maps (e.g. before_elm_delivery.state inside the lab).
        "return_waypoints": (
            {start_map} if (start_map in RETURN_BREADCRUMBS
                            and (ram_state["flag_elm_mr_pokemon"] & MR_POKEMON_BIT)
                            and not (ram_state["flag_elm_mr_pokemon"] & ELM_BIT))
            else set()
        ),
    }
