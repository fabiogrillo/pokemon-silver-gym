class RAMReader:
    def __init__(self, pyboy_instance):
        self.pyboy = pyboy_instance
    
    def read_all(self):
        """
        Return all desired RAM values as a dictionary. 
        This is the main method that will be called by the environment's step() and reset() methods.
        """
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
        # Knowing the type lets the agent decide: run from wild battles to save steps,
        # fight trainer/gym battles for the large reward bonus.
        battle_type = self.pyboy.memory[0xD116]

        # ── Party count
        party_count = self.pyboy.memory[0xDA22]

        # ── Lead Pokemon HP — two separate addresses:
        #   DA4C/DA4D = party slot 0 HP  (always valid, persists between battles)
        #   CB1C/CB1D = active combat HP (only meaningful while battle_type > 0)
        lead_hp     = (self.pyboy.memory[0xDA4C] << 8) | self.pyboy.memory[0xDA4D]
        lead_max_hp = (self.pyboy.memory[0xDA4E] << 8) | self.pyboy.memory[0xDA4F]
        battle_hp   = (self.pyboy.memory[0xCB1C] << 8) | self.pyboy.memory[0xCB1D]
        hp_ratio    = lead_hp / lead_max_hp if lead_max_hp > 0 else 0.0

        # ── Event flags: raw bytes from the 0xD7B7–0xD8B6 "Flags in Game" block
        # Each byte holds 8 individual bit-flags. Watch which bit flips when you:
        #   - beat the rival in Cherrygrove     → flag_rival_cherrygrove
        #   - receive the egg from Mr. Pokemon  → flag_elm_mr_pokemon
        #   - enter Sprout Tower 2F / 3F        → flag_sprout_tower_2/3
        flag_rival_cherrygrove = self.pyboy.memory[0xD8CA]  # "Met rival in Cherrygrove"
        flag_elm_mr_pokemon    = self.pyboy.memory[0xD7BD]  # Elm quest / Mr. Pokemon discovery
        flag_sprout_tower_2    = self.pyboy.memory[0xD85C]  # Sprout Tower 2F
        flag_sprout_tower_3    = self.pyboy.memory[0xD85D]  # Sprout Tower 3F
        
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
            "flag_sprout_tower_2": flag_sprout_tower_2,
            "flag_sprout_tower_3": flag_sprout_tower_3,
        }