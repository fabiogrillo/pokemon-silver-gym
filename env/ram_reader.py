class RAMReader:
    def __init__(self, pyboy_instance):
        self.pyboy = pyboy_instance
    
    def read_all(self):
        """Return all tracked RAM values as a dict. Called by the env step() and reset()."""
        # ── Global position (world coords — useful for debug, not used for tile tracking)
        x = self.pyboy.memory[0xD20D]
        y = self.pyboy.memory[0xD20E]

        # ── Tile exploration key: (map_bank, map_number, local_x, local_y)
        # map_bank + map_number together uniquely identify a map (map_number alone is not unique)
        map_bank   = self.pyboy.memory[0xDA00]
        map_number = self.pyboy.memory[0xDA01]
        local_x    = self.pyboy.memory[0xDA02]
        local_y    = self.pyboy.memory[0xDA03]

        # ── Badges: 0xD57C is a bitfield
        # bit 0 = Zephyr (Falkner), bit 1 = Hive, bit 2 = Plain, ... bit 7 = Rising
        badges      = self.pyboy.memory[0xD57C]
        badge_count = bin(badges).count('1')
        zephyr      = bool(badges & 0x01)  # our win condition

        # ── Battle type: 0=overworld | 1=wild | 2=trainer | 3=gym  (VERIFY empirically)
        battle_type = self.pyboy.memory[0xD116]

        # ── Party count
        party_count = self.pyboy.memory[0xDA22]

        # ── Lead Pokemon HP — two separate addresses:
        #   DA4C/DA4D = party slot 0 HP  (always valid, persists between battles)
        #   CB1C/CB1D = active combat HP (only meaningful while battle_type > 0)
        lead_hp     = (self.pyboy.memory[0xDA4C] << 8) | self.pyboy.memory[0xDA4D]
        lead_level  = self.pyboy.memory[0xDA49]  # Lead Pokemon level (slot 1)
        lead_max_hp = (self.pyboy.memory[0xDA4E] << 8) | self.pyboy.memory[0xDA4F]

        # ── Party Pokemon levels (all 6 slots)
        # Structure: wPartyMon1 base = 0xDA2A, level offset = +0x1F (empirically verified: 0xDA2A+0x1F=0xDA49=5 for Totodile at start)
        # Struct size = 0x30 bytes per slot (Gen 2 standard — VERIFY empirically with test_enemy_level.py)
        # Slots i >= party_count are zeroed to avoid stale RAM values from previous game sessions
        PARTY_LEVEL_BASE  = 0xDA49  # Slot 1 level — empirically verified ✓
        PARTY_STRUCT_SIZE = 0x30    # 48 bytes per party slot — calculated, needs verification
        party_levels = [
            self.pyboy.memory[PARTY_LEVEL_BASE + i * PARTY_STRUCT_SIZE] if i < party_count else 0
            for i in range(6)
        ]
        battle_hp   = (self.pyboy.memory[0xCB1C] << 8) | self.pyboy.memory[0xCB1D]
        hp_ratio    = lead_hp / lead_max_hp if lead_max_hp > 0 else 0.0

        # ── Event flags: raw bytes from the 0xD7B7–0xD8B6 "Flags in Game" block
        # Each byte holds 8 individual bit-flags. Watch which bit flips when you:
        #   - beat the rival in Cherrygrove     → flag_rival_cherrygrove
        #   - receive the egg from Mr. Pokemon  → flag_elm_mr_pokemon
        #   - enter Sprout Tower 2F / 3F        → flag_sprout_tower_2/3
        # 0xD88E bit 6 (0x40): starts as 1, cleared to 0 after beating rival on Route 29
        flag_rival_cherrygrove = self.pyboy.memory[0xD88E]
        # 0xD7BA bit 7 (0x80): set to 1 during Elm lab sequence (egg/police/pokeball)
        flag_elm_mr_pokemon    = self.pyboy.memory[0xD7BA]
        # 0xD836: Violet City Gym trainer beaten flags (empirically verified via test_enemy_level.py)
        # bit 4 (0x10): RISE after Trainer 1 beaten (flag #1020 = EVENT_BEAT_BIRD_KEEPER_ABE)
        # bit 3 (0x08): RISE after Trainer 2 beaten (flag #1019 = EVENT_BEAT_BIRD_KEEPER_ROD)
        flag_violet_gym        = self.pyboy.memory[0xD836]
        gym_trainers_beaten    = ((flag_violet_gym >> 3) & 1) + ((flag_violet_gym >> 4) & 1)

        # ── Route 30/31 trainer beaten flags (pret/pokegold event_flags.asm, formula
        # addr = 0xD7B7 + idx//8, bit = idx%8 — validated against 5 known anchors and
        # cross-checked 2026-06-11 by diffing egg_delivered.state vs crossing.state vs violet_city.state)
        # 0xD85E bit 0: EVENT_BEAT_BUG_CATCHER_DON   (#1336, Route 30)
        # 0xD85E bit 3: EVENT_BEAT_BUG_CATCHER_WADE  (#1339, Route 31)
        # 0xD86C bit 1: EVENT_BEAT_YOUNGSTER_JOEY    (#1449, Route 30)
        # 0xD86C bit 2: EVENT_BEAT_YOUNGSTER_MIKEY   (#1450, Route 30)
        flag_route_trainers_a = self.pyboy.memory[0xD85E]
        flag_route_trainers_b = self.pyboy.memory[0xD86C]
        route_trainers_beaten = (
            (flag_route_trainers_a & 1)
            + ((flag_route_trainers_a >> 3) & 1)
            + ((flag_route_trainers_b >> 1) & 1)
            + ((flag_route_trainers_b >> 2) & 1)
        )

        # ── Enemy Pokemon (valid only when battle_type > 0)
        # Source: DataCrystal — https://datacrystal.tcrf.net/wiki/Pokémon_Gold_and_Silver/RAM_map
        # $D0FC = Enemy Level
        # $D0FF/$D100 = Enemy current HP (2-byte big-endian)
        # $D101/$D102 = Enemy total HP   (2-byte big-endian)
        enemy_lead_level = self.pyboy.memory[0xD0FC]
        enemy_hp         = (self.pyboy.memory[0xD0FF] << 8) | self.pyboy.memory[0xD100]
        enemy_max_hp     = (self.pyboy.memory[0xD101] << 8) | self.pyboy.memory[0xD102]
        enemy_hp_ratio   = enemy_hp / enemy_max_hp if enemy_max_hp > 0 else 0.0

        return {
            "x": x,
            "y": y,
            "map_bank": map_bank,
            "map_number": map_number,
            "local_x": local_x,
            "local_y": local_y,
            "badges": badges,
            "badge_count": badge_count,
            "zephyr": zephyr,
            "battle_type": battle_type,
            "party_count": party_count,
            "lead_hp": lead_hp,
            "lead_max_hp": lead_max_hp,
            "battle_hp": battle_hp,
            "hp_ratio": hp_ratio,
            "flag_rival_cherrygrove": flag_rival_cherrygrove,
            "flag_elm_mr_pokemon": flag_elm_mr_pokemon,
            "flag_violet_gym": flag_violet_gym,
            "gym_trainers_beaten": gym_trainers_beaten,
            "route_trainers_beaten": route_trainers_beaten,
            "lead_level": lead_level,
            "party_levels": party_levels,
            "enemy_lead_level": enemy_lead_level,
            "enemy_hp": enemy_hp,
            "enemy_max_hp": enemy_max_hp,
            "enemy_hp_ratio": enemy_hp_ratio,
        }