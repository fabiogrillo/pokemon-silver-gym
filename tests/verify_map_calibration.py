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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        # NOTE: env/ram_reader.py's `local_x`/`local_y` fields are swapped relative to their
        # names. Per pokecrystal's ram/wram.asm, the four bytes right after wMapNumber are laid
        # out wYCoord THEN wXCoord (0xDA02 = Y, 0xDA03 = X), but ram_reader.py labels 0xDA02
        # "local_x" and 0xDA03 "local_y". Confirmed empirically too: every landmark below lands
        # exactly on its expected walkable tile only when the two values are swapped here; with
        # the as-labeled order, 3/7 states even compute a local coordinate that overflows past
        # their own map's tile bounds (impossible if it were really that axis). This is a
        # pre-existing bug in env/ram_reader.py (out of scope for this task -- env/ must not
        # change), so we compensate only here, for calibration accuracy.
        px = ml.to_image_px(*key, s["local_y"], s["local_x"])
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
