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
        # env/ram_reader.py's `local_x`/`local_y` fields are swapped relative to their names (see
        # map_layout.ram_to_image_px docstring); the un-swap compensation lives there, not here.
        px = ml.ram_to_image_px(*key, s["local_x"], s["local_y"])
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
