"""L1a: offline walkability-grid extractor for the egg-delivered -> Falkner corridor.

Run once (no vendored copy of the disassembly is kept -- this script fetches the handful of pret
source files it needs over HTTP at run time and throws them away):

    .venv/bin/python -m agents.llm.extract_collision            # (re)writes assets/collision/*.json
    .venv/bin/python -m agents.llm.extract_collision --verify   # ground-truth gate against the emulator

──────────────────────────────────────────────────────────────────────────────────────────────────
Data sources (raw.githubusercontent.com/pret/pokegold/master -- Silver's layout matches Gold's for
these maps, which pokegold disassembles just like pokecrystal):

  constants/map_constants.asm           map sizes IN BLOCKS ("map_const NAME, W, H"); the same
                                         numbers are already comment-documented (in TILES = 2x) in
                                         agents/rl/map_layout.py, cross-checked below.
  data/maps/maps.asm                    which TILESET_* each map header uses ("map Name, TILESET_X, ...").
  maps/<Name>.blk                       raw block-id bytes, row-major, W_blocks x H_blocks.
  data/tilesets/<tileset>_collision.asm "tilecoll A, B, C, D ; <block id>" -- 4 quadrant collision
                                         tokens per block. VERIFIED order (see _CLASSIFY docstring
                                         below): A=top-left, B=top-right, C=bottom-left, D=bottom-right.
  constants/collision_constants.asm     COLL_<TOKEN> = byte value; LAND_TILE/WATER_TILE/WALL_TILE/TALK
                                         bit values used by the permission table.
  data/collision/collision_permissions.asm  256-entry table (index = collision byte) -> LAND_TILE /
                                         WATER_TILE / WALL_TILE (optionally OR'd with TALK).

Walkability rule: LAND_TILE -> walkable; WATER_TILE, WALL_TILE -> not; ledges (collision byte in
the COLL_HOP_* range, hi nibble $A0 per HI_NYBBLE_LEDGES) -> still marked NOT walkable in the plain
`walkable` matrix (a ledge tile is never a legal place to stop), but ALSO recorded, with its
allowed hop direction(s), in a separate `ledges` map (v2; see below). `agents/llm/pathfind.py`'s A*
uses that side-channel to route a one-directional hop THROUGH a ledge tile instead of treating it
as a dead wall, without changing what "walkable" means for every other consumer of these grids.

Ledge directions (pokegold's engine/overworld/player_movement.asm `.TryJump`/`.ledge_table`, keyed
by `collision_byte & 7`): COLL_HOP_RIGHT/LEFT/UP/DOWN allow a hop only when facing that one
direction; COLL_HOP_DOWN_RIGHT/DOWN_LEFT/UP_RIGHT/UP_LEFT are corner tiles that allow a hop when
facing EITHER of the two named directions (the game ANDs the player's facing bitmask against the
tile's allowed-facings bitmask). This script derives the direction list straight from each token's
name (`"HOP_DOWN_LEFT"` -> `["down", "left"]`) rather than hand-maintaining a lookup table, so any
future HOP_* combination is picked up automatically.

Hop distance (empirically confirmed, 77/77 ledge/direction pairs across every extracted map, zero
exceptions): every ledge tile L has a non-walkable tile at `L + direction` and open ground only at
`L + 2*direction` -- i.e. the tile layout is always [foothold][ledge L][non-walkable buffer]
[landing], not [foothold][ledge L][landing]. This matches `.TryStep`/`.TryJump`'s two-phase design:
an ordinary step from the foothold onto L succeeds first (L's permission-table entry is LAND_TILE,
so nothing stops a plain walk onto it), and only the FOLLOWING step -- from L towards the buffer
tile -- fails collision (`.CheckLandPerms` on the buffer tile) and falls through to `.TryJump`,
which reads L's own collision byte (the tile the player is now standing on) to decide whether to
trigger STEP_LEDGE. The jump is sized to clear exactly that one buffer tile, landing on the far
side of it; the buffer tile's own collision is never re-checked once the jump is decided.
`agents/llm/pathfind.py`'s A* models this as a single hop edge from the foothold directly to
`L + 2*direction`, skipping L and the buffer tile as graph nodes entirely (matches the "ledge tile
stays walkable=0" convention below) -- but see agents/llm/tools._execute_navigate's ledge-escape
handling for what happens if the *live* player ends up standing exactly on L anyway (a real,
reachable position, confirmed directly against the emulator), which the offline grid alone can't
route from.

Grid convention: grid[y][x], TRUE axes (x east, y south) -- this matches the .blk file's own layout
(row-major by block-row=Y, block-col=X) directly; no un-swap needed here. The un-swap that matters
elsewhere (agents/rl/map_layout.ram_to_image_px) is only for the *live RAM* fields, which this
offline script never reads except in --verify, where it is applied explicitly.

Output JSON schema (v2, backward compatible): unchanged `bank`/`num`/`width`/`height`/`walkable`
keys, plus an optional `ledges` object mapping `"y,x"` (TRUE axes, same convention as `walkable`'s
indexing) -> a list of one or two direction strings ("up"/"down"/"left"/"right") the tile can be
hopped while facing. Only tiles that are ledges appear in the map; every other tile is implicitly
"not a ledge" whether or not `ledges` itself is present, so old readers that never look at
`ledges` see byte-identical `walkable` behavior to v1.
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.rl import map_layout as ml

BASE_URL = "https://raw.githubusercontent.com/pret/pokegold/master"

# (bank, num, pret map name, pret TILESET_* name (lowercased, matches data/tilesets/<x>_collision.asm),
#  pret map_constants.asm name, output filename)
MAPS = [
    (24, 4, "NewBarkTown",     "johto",             "NEW_BARK_TOWN",    "new_bark.json"),
    (24, 3, "Route29",        "johto",             "ROUTE_29",         "route_29.json"),
    (26, 3, "CherrygroveCity", "johto",            "CHERRYGROVE_CITY", "cherrygrove.json"),
    (26, 1, "Route30",        "johto",             "ROUTE_30",         "route_30.json"),
    (26, 2, "Route31",        "johto",             "ROUTE_31",         "route_31.json"),
    (10, 5, "VioletCity",     "johto",             "VIOLET_CITY",      "violet_city.json"),
    (10, 7, "VioletGym",      "elite_four_room",   "VIOLET_GYM",       "gym.json"),
]

OUT_DIR = "assets/collision"


def fetch_text(path: str) -> str:
    with urllib.request.urlopen(f"{BASE_URL}/{path}", timeout=20) as resp:
        return resp.read().decode("utf-8")


def fetch_bytes(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE_URL}/{path}", timeout=20) as resp:
        return resp.read()


def parse_collision_constants(text: str) -> dict:
    """constants/collision_constants.asm -> {NAME: int value}, e.g. {'LAND_TILE': 0, 'COLL_WALL': 7}."""
    consts = {}
    for m in re.finditer(r"^DEF\s+(\w+)\s+EQU\s+\$([0-9a-fA-F]+)", text, re.M):
        consts[m.group(1)] = int(m.group(2), 16)
    return consts


def parse_permission_table(text: str, consts: dict) -> list:
    """data/collision/collision_permissions.asm -> 256-entry list of ints (index = collision byte).

    Each `db EXPR ; comment` line's EXPR is one or more of LAND_TILE/WATER_TILE/WALL_TILE/TALK OR'd
    together; entries appear in byte order 0x00..0xff (asserted length $100 in the source).
    """
    lines = re.findall(r"^\s*db\s+([^;]+?)\s*;", text, re.M)
    assert len(lines) == 256, f"expected 256 permission entries, got {len(lines)}"
    table = []
    for expr in lines:
        val = 0
        for token in expr.split("|"):
            val |= consts[token.strip()]
        table.append(val)
    return table


def parse_quad_table(text: str) -> list:
    """data/tilesets/<x>_collision.asm -> list of (UL, UR, DL, DR) token tuples, indexed by block id."""
    rows = []
    for m in re.finditer(r"tilecoll\s+([^;]+);", text):
        toks = tuple(t.strip() for t in m.group(1).split(","))
        assert len(toks) == 4, toks
        rows.append(toks)
    return rows


def parse_map_block_size(text: str, const_name: str) -> tuple:
    """constants/map_constants.asm -> (width_blocks, height_blocks) for one `map_const NAME, W, H` line."""
    m = re.search(rf"^\s*map_const\s+{const_name}\s*,\s*(\d+)\s*,\s*(\d+)", text, re.M)
    assert m, f"map_const {const_name} not found"
    return int(m.group(1)), int(m.group(2))


# Ledges (COLL_HOP_*) sit at hi-nibble $A0 (HI_NYBBLE_LEDGES, constants/collision_constants.asm).
# Their permission-table entry is LAND_TILE (you CAN hop them), but a hop is one-directional (or
# two-directional for the diagonal-corner tokens), which the plain `walkable` matrix can't express
# -- classify_token() still marks them non-walkable there; ledge_hop_directions() below recovers
# the direction(s) separately for the `ledges` side-channel.
_LEDGE_HI_NIBBLE = 0xA0


def classify_token(token: str, coll_consts: dict, perm_table: list) -> int:
    """One quadrant's collision token ('FLOOR', 'WALL', 'HOP_DOWN', ...) -> 1 (walkable) or 0."""
    coll_id = coll_consts[f"COLL_{token}"]
    if (coll_id & 0xF0) == _LEDGE_HI_NIBBLE:
        return 0
    perm = perm_table[coll_id]
    return 1 if (perm & 0x0F) == 0x00 else 0  # low nibble 0x00 == LAND_TILE


def is_ledge_token(token: str, coll_consts: dict) -> bool:
    """True if `token`'s collision id falls in the COLL_HOP_* range (hi nibble $A0)."""
    coll_id = coll_consts[f"COLL_{token}"]
    return (coll_id & 0xF0) == _LEDGE_HI_NIBBLE


def ledge_hop_directions(token: str) -> list:
    """'HOP_DOWN_LEFT' -> ['down', 'left']; 'HOP_DOWN' -> ['down']; 'HOP_RIGHT' -> ['right'].

    Derived straight from the token's own name (matches pokegold's engine/overworld/
    player_movement.asm `.ledge_table`, which ANDs the player's facing bitmask against a
    per-token allowed-facings bitmask -- FACE_DOWN|FACE_LEFT for HOP_DOWN_LEFT, etc.) so any
    future HOP_* combination is picked up automatically without a hand-maintained table.
    """
    assert token.startswith("HOP_"), token
    return [part.lower() for part in token[len("HOP_"):].split("_")]


def build_grid(blk_bytes: bytes, w_blocks: int, h_blocks: int, quad_table: list,
                coll_consts: dict, perm_table: list) -> tuple:
    """Expand block-resolution .blk data + per-block quadrant collision into a tile-resolution
    walkability grid: grid[y][x], y in [0, 2*h_blocks), x in [0, 2*w_blocks); plus a
    `{(x, y): [direction, ...]}` map of ledge tiles found along the way (see module docstring)."""
    assert len(blk_bytes) == w_blocks * h_blocks, (len(blk_bytes), w_blocks, h_blocks)
    width, height = 2 * w_blocks, 2 * h_blocks
    grid = [[0] * width for _ in range(height)]
    ledges = {}
    for by in range(h_blocks):
        for bx in range(w_blocks):
            block_id = blk_bytes[by * w_blocks + bx]
            ul, ur, dl, dr = quad_table[block_id]
            for token, (tx, ty) in (
                (ul, (2 * bx, 2 * by)), (ur, (2 * bx + 1, 2 * by)),
                (dl, (2 * bx, 2 * by + 1)), (dr, (2 * bx + 1, 2 * by + 1)),
            ):
                grid[ty][tx] = classify_token(token, coll_consts, perm_table)
                if is_ledge_token(token, coll_consts):
                    ledges[(tx, ty)] = ledge_hop_directions(token)
    return grid, ledges


def extract_all():
    print("Fetching shared pret source files...")
    coll_consts = parse_collision_constants(fetch_text("constants/collision_constants.asm"))
    perm_table = parse_permission_table(fetch_text("data/collision/collision_permissions.asm"), coll_consts)
    map_consts_text = fetch_text("constants/map_constants.asm")

    quad_tables = {}  # tileset name -> quad table
    os.makedirs(OUT_DIR, exist_ok=True)

    for bank, num, map_name, tileset, const_name, out_name in MAPS:
        if tileset not in quad_tables:
            quad_tables[tileset] = parse_quad_table(fetch_text(f"data/tilesets/{tileset}_collision.asm"))
        quad_table = quad_tables[tileset]

        w_blocks, h_blocks = parse_map_block_size(map_consts_text, const_name)
        box = ml.MAP_INFO.get((bank, num))
        if box is not None:
            assert box.size == (2 * w_blocks, 2 * h_blocks), (
                f"{map_name}: map_constants.asm says {2*w_blocks}x{2*h_blocks} tiles, "
                f"but agents/rl/map_layout.MAP_INFO[({bank},{num})].size is {box.size}"
            )

        blk_bytes = fetch_bytes(f"maps/{map_name}.blk")
        grid, ledges = build_grid(blk_bytes, w_blocks, h_blocks, quad_table, coll_consts, perm_table)

        width, height = 2 * w_blocks, 2 * h_blocks
        flat = [c for row in grid for c in row]
        pct = 100.0 * sum(flat) / len(flat)
        print(f"  {map_name:18s} ({bank:2d},{num:2d}) {width:3d}x{height:<3d} tiles  "
              f"walkable {sum(flat):4d}/{len(flat):4d} ({pct:5.1f}%)  ledges {len(ledges):3d}  "
              f"-> {OUT_DIR}/{out_name}")

        ledges_out = {f"{y},{x}": dirs for (x, y), dirs in sorted(ledges.items())}
        out = {"bank": bank, "num": num, "width": width, "height": height, "walkable": grid,
               "ledges": ledges_out}
        with open(os.path.join(OUT_DIR, out_name), "w") as f:
            json.dump(out, f)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Step 4: ground-truth gate against the emulator.

_STATE_MAPS = [
    ("saves/violet_city.state", (10, 5), "violet_city.json"),
    ("saves/route31.state", (26, 2), "route_31.json"),
    ("saves/egg_delivered_clean.state", (24, 4), "new_bark.json"),
]

# probe direction -> (dx, dy) in TRUE (east, south) axes.
_DIR_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _grid_walkable(grid, width, height, x, y):
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    return bool(grid[y][x])


def verify():
    """Boot each of the 3 landmark save states, probe the emulator's actual 4-direction walkability
    at the player's live position (reusing agents.llm.agent.probe_walkable, save/restore, read-only),
    and compare against the committed grid's 4 neighbors at that position.

    THIS GATES THE WHOLE APPROACH: prints a 12-row mismatch table and returns True only if
    agreement is >= 11/12; a caller (main()) exits nonzero on failure so CI/the task can detect it.
    """
    # Local imports: keep the emulator/pyboy dependency out of the plain `extract_all()` path.
    from agents.llm.agent import probe_walkable
    from env.pyboy_wrapper import PyBoyWrapper
    from env.ram_reader import RAMReader

    rows = []
    for state_path, expect_key, grid_name in _STATE_MAPS:
        with open(os.path.join(OUT_DIR, grid_name)) as f:
            g = json.load(f)
        grid, width, height = g["walkable"], g["width"], g["height"]

        wrapper = PyBoyWrapper("pokemon_rom.gbc", state_path, headless=True)
        reader = RAMReader(wrapper.pyboy)
        s = reader.read_all()
        key = (s["map_bank"], s["map_number"])
        assert key == expect_key, f"{state_path}: expected map {expect_key}, got {key}"
        # env/ram_reader.py's local_x/local_y are swapped relative to their names (documented in
        # agents/rl/map_layout.py's ram_to_image_px) -- un-swap to get TRUE (x, y).
        true_x, true_y = s["local_y"], s["local_x"]

        open_dirs = set(probe_walkable(wrapper, reader, n=24, presses=3))
        wrapper.pyboy.stop(save=False)

        for direction, (dx, dy) in _DIR_DELTA.items():
            grid_says = _grid_walkable(grid, width, height, true_x + dx, true_y + dy)
            probe_says = direction in open_dirs
            rows.append((state_path, (true_x, true_y), direction, grid_says, probe_says,
                          grid_says == probe_says))

    agree = sum(1 for r in rows if r[-1])
    total = len(rows)
    print(f"\nGround-truth gate: {agree}/{total} agreement (need >= 11/12)\n")
    print(f"{'state':32s} {'pos':10s} {'dir':6s} {'grid':6s} {'probe':6s} match")
    for state_path, pos, direction, grid_says, probe_says, match in rows:
        mark = "OK" if match else "MISMATCH"
        print(f"{state_path:32s} {str(pos):10s} {direction:6s} {str(grid_says):6s} "
              f"{str(probe_says):6s} {mark}")
    return agree >= 11


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run the ground-truth gate only")
    args = parser.parse_args()

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)

    extract_all()


if __name__ == "__main__":
    main()
