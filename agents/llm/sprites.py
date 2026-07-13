"""Live NPC sprite positions, read straight from the emulator's object structs.

The static collision grids (assets/collision/*.json) encode terrain only. Everything that
actually blocked the corridor runs — the scripted rival, wandering NPCs, camping trainers —
is a live sprite the grid cannot see, and the executor only discovered them by walking into
them (3 failed presses per tile). Reading the game's own object table makes every sprite a
known obstacle BEFORE planning, so A* routes around people the same way it routes around trees.

Addresses come from the pret/pokegold disassembly (ram/wram.asm + macros/ram.asm) and were
verified against this ROM live: `wObjectStructs` at 0xD1FD holds NUM_OBJECT_STRUCTS (13)
`object_struct` records of 0x28 bytes — the player is record 0, NPCs fill later slots, and an
unused slot has 0 in its Sprite byte. Each record's MapX/MapY (offsets 0x10/0x11) hold the
sprite's standing tile in TRUE (x=east, y=south) axes, offset by +4 for the map connection
border. Verified in saves/violet_city_gym.state: the player struct reads (9, 16) while the
RAM reader reports true (5, 12) — the same +4 on both axes; NO x/y swap here, unlike
env/ram_reader.py's player fields (see agents/rl/map_layout.ram_to_image_px).
"""

OBJECT_STRUCTS = 0xD1FD   # wObjectStructs (pokegold.sym)
OBJECT_LENGTH = 0x28      # one object_struct (wObject1Struct - wPlayerStruct)
NUM_OBJECT_STRUCTS = 13   # player + 12 NPCs (constants/map_object_constants.asm)
_SPRITE = 0x00            # 0 = empty slot
_MAP_X = 0x10             # standing tile, true axes, +4 border offset
_MAP_Y = 0x11
_COORD_OFFSET = 4


def live_npc_tiles(pyboy) -> frozenset:
    """TRUE (x, y) tiles currently occupied by live NPC sprites on the current map.

    The player's own struct (slot 0) is excluded — it would otherwise block every plan at its
    start tile. Coordinates can be transiently off-grid while a sprite walks (MapX/MapY update
    to the destination tile at step start); callers re-plan every press, so a stale tile at
    worst costs one detour step.
    """
    mem = pyboy.memory
    tiles = set()
    for i in range(1, NUM_OBJECT_STRUCTS):
        base = OBJECT_STRUCTS + i * OBJECT_LENGTH
        if mem[base + _SPRITE] == 0:
            continue
        tiles.add((mem[base + _MAP_X] - _COORD_OFFSET, mem[base + _MAP_Y] - _COORD_OFFSET))
    return frozenset(tiles)
