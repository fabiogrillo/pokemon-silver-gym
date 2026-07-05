# Real-Map Trajectory Visualization (Whidden-style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trajectory + heatmap overlays rendered on the REAL stitched Johto map, pixel-accurate for the New Bark → Violet corridor, good enough for the blog/video and usable as the debugging tool for the Phase C corridor attempts.

**Architecture:** Restore the full stitched Johto map (7520×4320 @16 px/tile, recovered from Trash — deleted during the v2 reset). Replace the schematic placeholder offsets in `agents/rl/map_layout.py` with exact per-map tile offsets derived from the pokecrystal disassembly's map-connection data (PWhiddy's approach for Kanto), anchored to the image with one measured landmark. A landmark-marker harness renders labeled save-state positions onto the map so calibration is verified visually. `agents/rl/visualize_map.py` then renders onto a corridor crop of the real image instead of the schematic canvas.

**Tech Stack:** Python 3.12+ (`.venv`), PyBoy 2.7.1 (headless, save states), Pillow, imageio, numpy, pokecrystal disassembly data via raw.githubusercontent.com, pytest.

## Global Constraints

- ALL code/comments/commit messages in English. Chat with the user in Italian.
- The env layer (`env/`) MUST NOT change.
- Run everything with `.venv/bin/python` from the repo root; tests with `.venv/bin/python -m pytest <path> -v`.
- The consumer interface of `agents/rl/map_layout.py` used by `visualize_map.py` must keep working: `MAP_INFO: dict[(bank,num) -> MapBox]`, `to_global_px(bank,num,lx,ly) -> (px,py)|None`, `canvas_size() -> (w,h)`, `TILE_PX`, `CORRIDOR_ORDER`.
- Map identity: `(map_bank, map_number)` values already in `map_layout.py` are RAM-verified — (24,4) New Bark, (24,3) Route 29, (26,3) Cherrygrove, (26,1) Route 30, (26,2) Route 31, (10,5) Violet City, (10,7) gym, (24,5) Elm's lab, (26,11) gatehouse, (26,10) Mr. Pokemon's. Do NOT change them.
- The stitched image is the ground truth background; it is a community-stitched full-Johto map at 16 px/tile (GBC native). Source file recovered from `~/.local/share/Trash/files/johto_corridor.png` (original repo path `assets/maps/johto_corridor.png`, deleted 2026-06-29).
- Reveal-GIF output frames must stay ≤ ~900 px wide (crop + scale the corridor region; never emit 7520-px frames).
- Save states available as landmarks: `saves/newbark_egg.state`, `saves/egg_delivered_clean.state`, `saves/crossing.state`, `saves/mid_route30.state`, `saves/route31.state`, `saves/violet_city.state`, `saves/violet_city_gym.state`, `saves/gate.state`.

---

## File Structure

```
assets/maps/johto_full.png        # RESTORED full stitched Johto map (tracked, 513 KB)
assets/maps/README.md             # provenance + license note for the image
agents/rl/map_layout.py           # REWRITE: exact pokecrystal-derived tile offsets + image anchoring
agents/rl/visualize_map.py        # MODIFY: real-image background (corridor crop + scale), no naive resize
tests/test_map_layout.py          # NEW: unit tests for grid math and adjacency consistency
tests/verify_map_calibration.py   # NEW: landmark-marker harness (visual verification artifact)
runs/maps/…                       # rendered overlays (untracked)
assets/maps/corridor_agent087.png # final showcase overlay (tracked, produced in Task 4)
```

---

### Task 1: Restore the map asset + provenance note

**Files:**
- Create: `assets/maps/johto_full.png` (copied from Trash)
- Create: `assets/maps/README.md`

**Interfaces:**
- Produces: the background image at `assets/maps/johto_full.png`, size exactly (7520, 4320), palette mode.

- [ ] **Step 1: Restore the file**

```bash
mkdir -p assets/maps
cp ~/.local/share/Trash/files/johto_corridor.png assets/maps/johto_full.png
.venv/bin/python -c "from PIL import Image; im=Image.open('assets/maps/johto_full.png'); print(im.size, im.mode)"
```
Expected: `(7520, 4320) P`

- [ ] **Step 2: Write the provenance note**

`assets/maps/README.md`:

```markdown
# Map assets

- `johto_full.png` — community-stitched full Johto overworld map (Pokémon Gold/Silver),
  16 px per in-game tile (GBC native). Used ONLY as the background for the trajectory
  visualizations (`agents/rl/visualize_map.py`); no game assets are redistributed beyond
  this fan-made reference image. If you are the author and want attribution or removal,
  open an issue.
- `corridor_agent087.png` — generated overlay (trajectory + heatmap) produced by
  `visualize_map.py`; safe to regenerate at any time.
```

- [ ] **Step 3: Commit**

```bash
git add assets/maps/johto_full.png assets/maps/README.md
git commit -m "feat(viz): restore stitched Johto map as visualization background"
```

---

### Task 2: Exact corridor grid from pokecrystal connection data

**Files:**
- Modify: `agents/rl/map_layout.py` (full rewrite of the offset table; keep the public interface)
- Test: `tests/test_map_layout.py`

**Interfaces:**
- Consumes: pokecrystal raw files (fetch, do not vendor):
  - sizes (in BLOCKS; 1 block = 2×2 tiles): `curl -s https://raw.githubusercontent.com/pret/pokecrystal/master/constants/map_constants.asm | grep -E "NEW_BARK_TOWN|ROUTE_29|CHERRYGROVE_CITY|ROUTE_30|ROUTE_31|VIOLET_CITY"` — `map_const NAME, width, height`
  - connections (offsets in BLOCKS): `curl -s https://raw.githubusercontent.com/pret/pokecrystal/master/data/maps/maps.asm | grep -A 6 -E "map NewBarkTown|map Route29|map CherrygroveCity|map Route30|map Route31|map VioletCity"` — `connection <dir>, <Map>, <MAP_CONST>, <offset>`
- Produces (later tasks rely on these EXACT names):
  - `TILE_PX = 16` (image px per tile — replaces the schematic 6)
  - `MAP_INFO`, `CORRIDOR_ORDER` unchanged in shape; overworld entries get exact `offset` (in tiles, global grid) and exact `size` (in tiles = blocks×2)
  - `MapBox` gains field `inset: bool = False`; interiors (Elm's lab (24,5), gym (10,7), gatehouse (26,11), Mr. Pokemon's (26,10), Dark Cave (3,70)) keep hand-placed offsets with `inset=True`
  - `ANCHOR_PX: tuple[int, int]` — image pixel of global tile (0,0); set to a provisional `(0, 0)` in this task, measured for real in Task 3
  - `to_image_px(bank, num, lx, ly) -> tuple[int, int] | None` = `ANCHOR_PX + to_global_tile * TILE_PX` (None for unknown maps)
  - `to_global_px` kept as alias of `to_image_px` (visualize_map.py compatibility)
  - `corridor_bbox_px(pad_tiles: int = 4) -> tuple[int, int, int, int]` — image-space bounding box of the non-inset corridor maps

Connection math (standard GSC, all in blocks, converted ×2 to tiles): given parent P placed at `(Px, Py)` and `connection north, N, offset` on P: `N.x = Px + 2*offset`, `N.y = Py - N.h`; `south`: `N.y = Py + P.h`, `N.x = Px + 2*offset`; `west`: `N.x = Px - N.w`, `N.y = Py + 2*offset`; `east`: `N.x = Px + P.w`, `N.y = Py + 2*offset`. Seed `NEW_BARK_TOWN` at (0,0), walk the corridor connections, then translate everything so the minimum coordinate is (0,0).

- [ ] **Step 1: Fetch the disassembly numbers and record them**

Run the two curl commands above. Write the retrieved sizes/offsets as a comment block in `map_layout.py` (source of each number, e.g. `# ROUTE_29: 15x9 blocks (map_constants.asm); Route29 east -> NewBarkTown, offset 0 (maps.asm)`). Every offset in `MAP_INFO` must be justified by one of these lines — no eyeballed numbers for non-inset maps.

- [ ] **Step 2: Write the failing tests**

`tests/test_map_layout.py`:

```python
from agents.rl import map_layout as ml


OVERWORLD = [(24, 4), (24, 3), (26, 3), (26, 1), (26, 2), (10, 5)]


def test_tile_px_matches_image_scale():
    assert ml.TILE_PX == 16


def test_overworld_maps_not_inset_and_interiors_inset():
    for key in OVERWORLD:
        assert ml.MAP_INFO[key].inset is False
    for key in [(24, 5), (10, 7), (26, 11), (26, 10)]:
        assert ml.MAP_INFO[key].inset is True


def test_all_offsets_non_negative():
    for box in ml.MAP_INFO.values():
        assert box.offset[0] >= 0 and box.offset[1] >= 0


def test_route29_newbark_adjacency():
    """Route 29's east edge must touch New Bark's west edge (they are connected east/west)."""
    r29, nb = ml.MAP_INFO[(24, 3)], ml.MAP_INFO[(24, 4)]
    assert r29.offset[0] + r29.size[0] == nb.offset[0]


def test_route30_31_vertical_adjacency():
    """Route 31 sits directly north of Route 30 (south edge of 31 touches north edge of 30)."""
    r31, r30 = ml.MAP_INFO[(26, 2)], ml.MAP_INFO[(26, 1)]
    assert r31.offset[1] + r31.size[1] == r30.offset[1]


def test_to_image_px_uses_anchor_and_tile_px():
    box = ml.MAP_INFO[(24, 4)]
    px = ml.to_image_px(24, 4, 0, 0)
    assert px == (ml.ANCHOR_PX[0] + box.offset[0] * ml.TILE_PX,
                  ml.ANCHOR_PX[1] + box.offset[1] * ml.TILE_PX)


def test_unknown_map_returns_none():
    assert ml.to_image_px(99, 99, 0, 0) is None


def test_corridor_bbox_contains_all_overworld_maps():
    x0, y0, x1, y1 = ml.corridor_bbox_px()
    for key in OVERWORLD:
        box = ml.MAP_INFO[key]
        bx0 = ml.ANCHOR_PX[0] + box.offset[0] * ml.TILE_PX
        by0 = ml.ANCHOR_PX[1] + box.offset[1] * ml.TILE_PX
        assert x0 <= bx0 and y0 <= by0
        assert x1 >= bx0 + box.size[0] * ml.TILE_PX and y1 >= by0 + box.size[1] * ml.TILE_PX
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_map_layout.py -v`
Expected: FAIL (`TILE_PX == 6`, no `inset`, no `to_image_px`, no `corridor_bbox_px`).

- [ ] **Step 4: Rewrite map_layout.py**

Keep the module docstring's spirit but rewrite the content: `MapBox` with `inset` field; overworld offsets computed by hand from the fetched connection data (show the arithmetic in comments); interiors kept as insets placed in visually empty areas near their parent (their exact placement is refined in Task 3 against the real image); `ANCHOR_PX = (0, 0)` provisional with a `# measured in Task 3` note; implement `to_image_px`, `to_global_px = to_image_px`, `corridor_bbox_px`, keep `canvas_size()` returning the full image size `(7520, 4320)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_map_layout.py -v` → PASS (8 tests).

- [ ] **Step 6: Commit**

```bash
git add agents/rl/map_layout.py tests/test_map_layout.py
git commit -m "feat(viz): exact corridor tile grid from pokecrystal connection data"
```

---

### Task 3: Anchor measurement + landmark verification harness

**Files:**
- Create: `tests/verify_map_calibration.py`
- Modify: `agents/rl/map_layout.py` (set the real `ANCHOR_PX`; refine inset placements)

**Interfaces:**
- Consumes: `PyBoyWrapper` + `RAMReader` (existing, read-only usage), `map_layout.to_image_px`, `corridor_bbox_px`, the image.
- Produces: `runs/maps/calibration_check.png` — corridor crop with one labeled cross per landmark save state. THE deliverable of this task is that image being visually correct.

- [ ] **Step 1: Write the harness**

```python
"""Render each landmark save state's RAM position onto the real Johto map.

For every save state below: boot it headless, read (map_bank, map_number, local_x,
local_y), project via map_layout.to_image_px, and draw a labeled cross on a corridor
crop of assets/maps/johto_full.png. Inset maps are skipped (not on the overworld grid).

Run: .venv/bin/python tests/verify_map_calibration.py
Output: runs/maps/calibration_check.png — inspect it: every cross must sit where the
player actually stands in that state (New Bark for newbark_egg, the Route 30/31 area
for mid_route30/route31, Violet City for violet_city, ...).
"""

import os

from PIL import Image, ImageDraw

from agents.rl import map_layout as ml
from env.pyboy_wrapper import PyBoyWrapper
from env.ram_reader import RAMReader

STATES = [
    "saves/newbark_egg.state",
    "saves/egg_delivered_clean.state",
    "saves/crossing.state",
    "saves/mid_route30.state",
    "saves/route31.state",
    "saves/gate.state",
    "saves/violet_city.state",
]
ROM = "pokemon_rom.gbc"
OUT = "runs/maps/calibration_check.png"


def main():
    img = Image.open("assets/maps/johto_full.png").convert("RGB")
    draw = ImageDraw.Draw(img)
    for state in STATES:
        wrapper = PyBoyWrapper(ROM, state, headless=True)
        s = RAMReader(wrapper.pyboy).read_all()
        wrapper.pyboy.stop(save=False)
        key = (s["map_bank"], s["map_number"])
        label = f"{os.path.basename(state).removesuffix('.state')} {key} ({s['local_x']},{s['local_y']})"
        px = ml.to_image_px(*key, s["local_x"], s["local_y"])
        if px is None or ml.MAP_INFO.get(key, ml.MapBox('?', (0, 0), (0, 0))).inset:
            print(f"[skip] {label} (inset/unknown)")
            continue
        x, y = px
        draw.line([x - 24, y, x + 24, y], fill=(255, 0, 0), width=5)
        draw.line([x, y - 24, x, y + 24], fill=(255, 0, 0), width=5)
        draw.text((x + 28, y - 10), label, fill=(255, 0, 0))
        print(f"[mark] {label} -> {px}")
    x0, y0, x1, y1 = ml.corridor_bbox_px(pad_tiles=8)
    crop = img.crop((x0, y0, x1, y1))
    crop.thumbnail((1800, 1800))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    crop.save(OUT)
    print(f"[out] {OUT} (crop {x0},{y0},{x1},{y1})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Measure ANCHOR_PX**

Locate ONE unambiguous landmark: open `assets/maps/johto_full.png` (Read it as an image), find New Bark Town on the map (small town on the east side of Johto), and measure the image pixel of its top-left map corner. `ANCHOR_PX = that_pixel - MAP_INFO[(24,4)].offset * TILE_PX` (element-wise). Set it in `map_layout.py`. First run of the harness gives immediate feedback; expect 1-3 iterations of adjusting ANCHOR_PX by whole tiles.

- [ ] **Step 3: Iterate until visually correct**

Run: `.venv/bin/python tests/verify_map_calibration.py`, then Read `runs/maps/calibration_check.png` and check each cross against the geography (route31 cross on Route 31's strip, violet_city cross inside the city, crossing/mid_route30 along Route 30's vertical corridor, ...). Adjust `ANCHOR_PX` (global shift = anchor; single-map shift = that map's connection math is wrong, re-check Step 1 data of Task 2). Acceptance: every marker within ~1 tile (16 px) of its true spot.

- [ ] **Step 4: Re-run the unit tests**

Run: `.venv/bin/python -m pytest tests/test_map_layout.py -v` → still PASS (anchor change must not break grid tests).

- [ ] **Step 5: Commit**

```bash
git add agents/rl/map_layout.py tests/verify_map_calibration.py
git commit -m "feat(viz): anchor corridor grid to the real map, landmark-verified"
```

---

### Task 4: Renderer on the real image + showcase overlays

**Files:**
- Modify: `agents/rl/visualize_map.py`
- Create (artifact): `assets/maps/corridor_agent087.png` (+ overlays under `runs/maps/`, untracked)

**Interfaces:**
- Consumes: `map_layout` v2 (Task 2/3), existing `rollout_positions`, `heatmap_overlay`, `draw_trajectory`, `render` pipeline; checkpoints under `runs/checkpoints/`.
- Produces: `load_background()` returns the real image UNRESIZED (`Image.open(ASSET_BG).convert("RGB")` — the schematic fallback stays for when the asset is missing); `render(...)` gains corridor cropping + output scaling: after composing at native scale, crop to `ml.corridor_bbox_px(pad_tiles=8)` and scale so the width is ≤ 900 px (`OUT_MAX_W = 900`), for both the PNG and every reveal-GIF frame. `ASSET_BG` updated to `assets/maps/johto_full.png`. Inset maps: draw their `MapBox` rectangle + name on the composed image (so gym/lab trajectories remain visible as inset boxes), reusing the box-drawing loop currently in the schematic branch, but only for `inset=True` entries.

- [ ] **Step 1: Update visualize_map.py**

Concrete changes (keep everything else intact):
1. `ASSET_BG = os.path.join("assets", "maps", "johto_full.png")`.
2. `load_background()`: if the asset exists → `return Image.open(ASSET_BG).convert("RGB")` (NO resize). Schematic fallback unchanged.
3. After composing heatmap+trajectory in `render()` (and inside the GIF frame loop), apply:

```python
def finalize_frame(img: Image.Image) -> Image.Image:
    """Crop to the corridor and cap the output width (never emit native 7520-px frames)."""
    x0, y0, x1, y1 = ml.corridor_bbox_px(pad_tiles=8)
    img = img.crop((x0, y0, x1, y1))
    if img.width > OUT_MAX_W:
        img = img.resize((OUT_MAX_W, int(img.height * OUT_MAX_W / img.width)))
    return img
```

4. Draw inset boxes (name + rectangle at `offset*TILE_PX + ANCHOR_PX`) on the background copy once, before compositing, for `inset=True` maps only.
5. GIF memory guard: build reveal frames from the CROPPED+SCALED image, not the native one.

- [ ] **Step 2: Sanity-run the calibration harness once more**

Run: `.venv/bin/python tests/verify_map_calibration.py` → unchanged output (Task 4 must not move any marker).

- [ ] **Step 3: Render the showcase overlays**

```bash
# Corridor attempt from the egg-delivered start with the strongest available corridor checkpoint
ls runs/checkpoints/ | sort   # pick the newest corridor-capable agent (e.g. agent_081/082 v2 line)
.venv/bin/python -m agents.rl.visualize_map \
    --model runs/checkpoints/<BEST_CORRIDOR_CKPT>.zip \
    --state saves/egg_delivered_clean.state --max-steps 8000 \
    --out runs/maps/corridor_best

# Gym slice (agent_087) — trajectory lives in the gym inset
.venv/bin/python -m agents.rl.visualize_map \
    --model runs/checkpoints/agent_087/agent_087_final.zip \
    --state saves/violet_city_gym.state --max-steps 2000 \
    --out runs/maps/gym_agent087
```
`<BEST_CORRIDOR_CKPT>` is chosen at runtime by inspecting `runs/checkpoints/` (newest v2 agent) and, if unclear, `training_log.md`'s last corridor entries. Read both output PNGs and verify: trajectory hugs the actual routes (no lines through water/void), heatmap sits on walkable paths, reveal GIF ≤ 900 px wide.

- [ ] **Step 4: Promote the corridor overlay to assets**

```bash
cp runs/maps/corridor_best.png assets/maps/corridor_agent087.png   # rename to the actual agent id
git add assets/maps/ agents/rl/visualize_map.py
git commit -m "feat(viz): render trajectories on the real Johto map (corridor crop)"
```
(If the file is named for a different agent id, keep the name accurate — update `assets/maps/README.md` accordingly in the same commit.)

---

### Task 5: Phase sync

**Files:**
- Modify: `/home/fabio/Projects/FAANG-Job-Search/FAANG-Job-Search/Planning/Active-Roadmap.md`
- Modify: `/home/fabio/Projects/FAANG-Job-Search/FAANG-Job-Search/Planning/Weekly-Logs/2026-Q3/…` current weekly log (Week 28, Jul 6-12 — create is NOT allowed; if the Week-28 log does not exist yet, append to Week-27 instead; do not create future logs)

- [ ] **Step 1: Roadmap**

In the roadmap's Target Outcomes, update the RL v2 line's map-viz mention: `offline map-visualization overlay` → `map-visualization overlay on the real Johto map (pokecrystal-calibrated) ✅ 2026-07-05` (keep the rest of the line). Do NOT commit the vault (user-managed).

- [ ] **Step 2: Weekly log**

One line under the Pokémon section of the current weekly log: `- Map-viz calibrated on the real stitched Johto map (pokecrystal connection data); trajectory overlays ready for corridor debugging + blog.`

- [ ] **Step 3: No repo commit needed** (repo changes were committed in Task 4). Report the vault edits in the task report.

---

## Self-Review

**Spec coverage:** real map sourced (Task 1 — recovered, better than re-downloading), precise per-map offsets from pokecrystal connections (Task 2), landmark verification against known save states (Task 3 — the spec's "verified with known landmarks"), visualize_map renders on the real asset with usable output sizes (Task 4), sync rule (Task 5). Gap check vs spec Phase D: "overlay of agent_087 gym run + an egg-delivered corridor run" → Task 4 Step 3 covers both.

**Placeholder scan:** `<BEST_CORRIDOR_CKPT>` and `ANCHOR_PX` are runtime-discovered values with exact discovery procedures — intentional. No TBDs.

**Type consistency:** `to_image_px(bank,num,lx,ly) -> (px,py)|None` used in Tasks 2/3/4; `corridor_bbox_px(pad_tiles)` in 2/3/4; `inset` field in 2/3/4; `TILE_PX=16` and `ANCHOR_PX` shared. `to_global_px` alias keeps `visualize_map.positions_to_px` working before Task 4 touches it.

**Executor notes:** Tasks 2-4 need the implementer to Read PNG images (visual checks) — use a vision-capable implementer (sonnet). Task 3 is inherently iterative; budget 2-3 render-inspect cycles. Emulator boots in Task 3 take ~2 s per state, trivial. Task 4's corridor rollout takes minutes (8000 steps, CPU/GPU inference).
