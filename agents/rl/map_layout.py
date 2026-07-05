"""
Map stitching layout for the Whidden-style trajectory/heatmap overlay (see visualize_map.py).

The game only exposes LOCAL coordinates per map (map_bank, map_number, local_x, local_y from
env/ram_reader.py). To plot a single world-space trajectory across maps we need a table that places
each local map inside one global (tile) grid, which is then mapped onto the real stitched Johto
image at assets/maps/johto_full.png (7520x4320 px, TILE_PX=16 px/tile):

    global_tile = MAP_INFO[(bank, num)].offset + (local_x, local_y)
    image_px    = ANCHOR_PX + global_tile * TILE_PX

Scope: only the ~10 maps of the egg-delivered -> Falkner corridor are needed (this is the whole
objective of the RL v2 re-baseline).

─────────────────────────────────────────────────────────────────────────────
Fetched pokecrystal source data (raw.githubusercontent.com/pret/pokecrystal/master), used to derive
the EXACT (non-eyeballed) overworld offsets below.

1) Map sizes, in BLOCKS (1 block = 2x2 tiles) -- constants/map_constants.asm:
     map_const VIOLET_CITY,       20, 18   ; -> 40 x 36 tiles
     map_const ROUTE_29,          30,  9   ; -> 60 x 18 tiles
     map_const NEW_BARK_TOWN,     10,  9   ; -> 20 x 18 tiles
     map_const ROUTE_30,          10, 27   ; -> 20 x 54 tiles
     map_const ROUTE_31,          20,  9   ; -> 40 x 18 tiles
     map_const CHERRYGROVE_CITY,  20,  9   ; -> 40 x 18 tiles

2) Connections, offsets in BLOCKS -- data/maps/attributes.asm (map_attributes / connection macros):
     map_attributes NewBarkTown, NEW_BARK_TOWN, $05
       connection west, Route29, ROUTE_29, 0
     map_attributes CherrygroveCity, CHERRYGROVE_CITY, $35
       connection north, Route30, ROUTE_30, 5
       connection east, Route29, ROUTE_29, 0
     map_attributes VioletCity, VIOLET_CITY, $05
       connection east, Route31, ROUTE_31, 9
     map_attributes Route29, ROUTE_29, $05
       connection west, CherrygroveCity, CHERRYGROVE_CITY, 0
       connection east, NewBarkTown, NEW_BARK_TOWN, 0
     map_attributes Route30, ROUTE_30, $05
       connection north, Route31, ROUTE_31, -10
       connection south, CherrygroveCity, CHERRYGROVE_CITY, -5
     map_attributes Route31, ROUTE_31, $05
       connection south, Route30, ROUTE_30, 10
       connection west, VioletCity, VIOLET_CITY, -9

Connection math (standard GSC, converted blocks->tiles by x2), applied to place each map relative to
its already-placed parent P at (Px, Py) with size (Pw, Ph):
  north: N.x = Px + 2*offset ;  N.y = Py - N.h
  south: N.x = Px + 2*offset ;  N.y = Py + Ph
  west:  N.x = Px - N.w      ;  N.y = Py + 2*offset
  east:  N.x = Px + Pw       ;  N.y = Py + 2*offset

Derivation, seeding NEW_BARK_TOWN at (0, 0) and walking the corridor (each step cross-checked
against the *reverse* connection listed on the neighboring map -- they all agree, see below):

  NewBarkTown  (parent, seed)            -> (0, 0),      size (20, 18)
  "connection west, Route29, 0" on NewBarkTown:
    Route29.x = 0 - 60 = -60 ; Route29.y = 0 + 2*0 = 0    -> (-60, 0), size (60, 18)
    [cross-check "connection east, NewBarkTown, 0" on Route29:
     NewBarkTown.x = -60 + 60 = 0 ; .y = 0 + 0 = 0  -- matches]
  "connection west, CherrygroveCity, 0" on Route29:
    Cherrygrove.x = -60 - 40 = -100 ; Cherrygrove.y = 0 + 2*0 = 0  -> (-100, 0), size (40, 18)
    [cross-check "connection east, Route29, 0" on CherrygroveCity:
     Route29.x = -100 + 40 = -60 ; .y = 0 -- matches]
  "connection north, Route30, 5" on CherrygroveCity:
    Route30.x = -100 + 2*5 = -90 ; Route30.y = 0 - 54 = -54  -> (-90, -54), size (20, 54)
    [cross-check "connection south, CherrygroveCity, -5" on Route30:
     Cherrygrove.y = -54 + 54 = 0 ; .x = -90 + 2*(-5) = -100 -- matches]
  "connection north, Route31, -10" on Route30:
    Route31.x = -90 + 2*(-10) = -110 ; Route31.y = -54 - 18 = -72  -> (-110, -72), size (40, 18)
    [cross-check "connection south, Route30, 10" on Route31:
     Route30.y = -72 + 18 = -54 ; .x = -110 + 2*10 = -90 -- matches]
  "connection west, VioletCity, -9" on Route31:
    VioletCity.x = -110 - 40 = -150 ; VioletCity.y = -72 + 2*(-9) = -90  -> (-150, -90), size (40, 36)
    [cross-check "connection east, Route31, 9" on VioletCity:
     Route31.x = -150 + 40 = -110 ; .y = -90 + 2*9 = -72 -- matches]

Raw (pre-translation) offsets: NewBarkTown(0,0) Route29(-60,0) Cherrygrove(-100,0) Route30(-90,-54)
Route31(-110,-72) VioletCity(-150,-90). Minimum coordinate is (-150, -90) (VioletCity's top-left) --
translate every box by (+150, +90) so the whole corridor sits at non-negative tile coordinates
(required: `to_global_px` feeds pixel positions that must not go off-canvas):

  NewBarkTown  (0+150,   0+90) = (150, 90)
  Route29      (-60+150, 0+90) = (90,  90)
  CherrygroveCity (-100+150, 0+90) = (50, 90)
  Route30      (-90+150, -54+90) = (60, 36)
  Route31      (-110+150, -72+90) = (40, 18)
  VioletCity   (-150+150, -90+90) = (0, 0)

Sanity check against real Johto geography: VioletCity is north-west-most (0,0); Route31 sits just
south-east of it and its west edge touches Violet City's east edge; Route30 continues further
south-east from Route31 (their shared edge touches per test_route30_31_vertical_adjacency);
CherrygroveCity sits south of Route30's southern end; Route29 runs east from Cherrygrove to
NewBarkTown, which ends up furthest south-east -- matching the brief's expected topology.

Interiors (Elm's lab, the gym, the Route31/Violet gatehouse, Mr. Pokemon's house, Dark Cave) are not
overworld maps connected via the `connection` macro, so they are hand-placed as small `inset=True`
boxes tucked into a visually empty area near their parent overworld map. Their exact placement will
be fine-tuned against the real stitched image in Task 3 -- for now they only need to (a) be flagged
inset=True and (b) have non-negative offsets so they render on-canvas.
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field

# Pixels per in-game tile on the real stitched image (assets/maps/johto_full.png is 7520x4320 px).
TILE_PX = 16


@dataclass(frozen=True)
class MapBox:
    name: str
    offset: tuple[int, int]   # (gx, gy) top-left of this map on the global tile grid, in TILES
    size: tuple[int, int]     # (w, h) of this map, in TILES
    color: tuple[int, int, int] = field(default=(60, 70, 90))  # schematic fallback fill color
    inset: bool = False       # True for hand-placed interiors (not derived from connection data)


# (bank, number) -> MapBox. (bank, number) match the constants in env/rewards.py -- these keys are
# RAM-verified and must not change.
#
# Overworld offsets/sizes below are computed EXACTLY from the pokecrystal connection data fetched
# above (see the derivation block for the arithmetic). Interiors are hand-placed insets.
MAP_INFO: dict[tuple[int, int], MapBox] = {
    # ---- overworld corridor (exact, from connection data) ----
    (10, 5):  MapBox("Violet City",      offset=(0,  0),  size=(40, 36), color=(70, 100, 80)),
    (26, 2):  MapBox("Route 31",         offset=(40, 18), size=(40, 18), color=(80, 95, 70)),
    (26, 1):  MapBox("Route 30",         offset=(60, 36), size=(20, 54), color=(80, 95, 70)),
    (26, 3):  MapBox("Cherrygrove City", offset=(50, 90), size=(40, 18), color=(70, 100, 80)),
    (24, 3):  MapBox("Route 29",         offset=(90, 90), size=(60, 18), color=(80, 95, 70)),
    (24, 4):  MapBox("New Bark Town",    offset=(150, 90), size=(20, 18), color=(70, 100, 80)),

    # ---- interiors (hand-placed insets, tucked near their parent; refined in Task 3) ----
    (10, 7):  MapBox("Falkner's Gym",    offset=(14, 14), size=(8, 8),
                      color=(120, 80, 70), inset=True),   # inside Violet City's box (0,0)-(40,36)
    (26, 11): MapBox("Violet Gatehouse", offset=(42, 20), size=(6, 6),
                      color=(90, 90, 70), inset=True),    # near Route 31's west edge (Violet border)
    (26, 10): MapBox("Mr. Pokemon's",    offset=(64, 60), size=(8, 8),
                      color=(90, 90, 70), inset=True),    # along Route 30
    (24, 5):  MapBox("Elm's Lab",        offset=(154, 94), size=(8, 8),
                      color=(110, 90, 120), inset=True),  # inside New Bark Town's box
    (3, 70):  MapBox("Dark Cave",        offset=(200, 150), size=(10, 10),
                      color=(50, 50, 60), inset=True),    # off-path, placed clear of the corridor
}

# Ordered corridor (for legends / progression labels), south-start to gym.
CORRIDOR_ORDER = [(24, 5), (24, 4), (24, 3), (26, 3), (26, 1), (26, 2), (26, 11), (10, 5), (10, 7)]

# Image pixel of global tile (0, 0). Provisional (0, 0) -- measured for real in Task 3 once the
# corridor is calibrated against assets/maps/johto_full.png landmarks.
ANCHOR_PX: tuple[int, int] = (0, 0)  # measured in Task 3


def to_image_px(bank: int, num: int, local_x: int, local_y: int) -> tuple[int, int] | None:
    """Map a (bank, num, local_x, local_y) RAM reading to an image pixel on
    assets/maps/johto_full.png, or None if the map is not in the layout (caller should skip it)."""
    box = MAP_INFO.get((bank, num))
    if box is None:
        return None
    gx = ANCHOR_PX[0] + (box.offset[0] + local_x) * TILE_PX
    gy = ANCHOR_PX[1] + (box.offset[1] + local_y) * TILE_PX
    return gx, gy


# Back-compat alias for visualize_map.py, which calls this the "global px" mapping.
to_global_px = to_image_px


def corridor_bbox_px(pad_tiles: int = 4) -> tuple[int, int, int, int]:
    """Image-space bounding box (x0, y0, x1, y1) of the non-inset (overworld corridor) maps, padded
    by `pad_tiles` tiles on every side."""
    boxes = [b for b in MAP_INFO.values() if not b.inset]
    min_x = min(b.offset[0] for b in boxes) - pad_tiles
    min_y = min(b.offset[1] for b in boxes) - pad_tiles
    max_x = max(b.offset[0] + b.size[0] for b in boxes) + pad_tiles
    max_y = max(b.offset[1] + b.size[1] for b in boxes) + pad_tiles
    x0 = ANCHOR_PX[0] + min_x * TILE_PX
    y0 = ANCHOR_PX[1] + min_y * TILE_PX
    x1 = ANCHOR_PX[0] + max_x * TILE_PX
    y1 = ANCHOR_PX[1] + max_y * TILE_PX
    return x0, y0, x1, y1


def canvas_size() -> tuple[int, int]:
    """(width, height) in pixels of the real stitched Johto image (assets/maps/johto_full.png)."""
    return 7520, 4320
