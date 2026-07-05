"""
Offline map visualization — replay a trained checkpoint and render WHERE the agent went, overlaid on
a stitched map of the New Bark → Violet (Falkner) corridor. Inspired by PWhiddy's pokerl-map-viz.

Two artifacts per run:
  - <out>.png : a static overlay = visitation HEATMAP (where it spent time) + the TRAJECTORY line of
                one episode, colored by time (blue = early, red = late).
  - <out>.gif : the same trajectory revealed step-by-step, so you can watch the path unfold.

It reuses the exact eval wrapper stack (agents/rl/evaluate_cnn.build_vec_env) and reads the agent's
(map_bank, map_number, local_x, local_y) from RAM each step, mapping to global canvas pixels via
agents/rl/map_layout. If assets/maps/johto_full.png exists it is used as the background (cropped to
the corridor + capped to OUT_MAX_W px wide for output); otherwise a labelled schematic canvas is
generated (so the tool runs before the real map asset is added).

Usage:
  python -m agents.rl.visualize_map \
      --model runs/checkpoints/agent_076/agent_076_5000000_steps.zip \
      --state saves/egg_delivered_clean.state --max-steps 8000 --out runs/maps/agent_076

  # progression montage: one PNG per checkpoint in a run directory + a combined GIF
  python -m agents.rl.visualize_map --all-checkpoints runs/checkpoints/agent_076 \
      --state saves/egg_delivered_clean.state --max-steps 8000 --out runs/maps/agent_076
"""

import argparse
import glob
import os
import re

import numpy as np
import imageio.v3 as iio
import torch
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from agents.rl.evaluate_cnn import build_vec_env
from agents.rl import map_layout as ml

ASSET_BG = os.path.join("assets", "maps", "johto_full.png")
OUT_MAX_W = 900  # never emit native (7520px-wide) frames — cap every PNG/GIF frame to this width


# ─────────────────────────────────────────────────────────────────────────────
# Rollout: collect the agent's per-step (bank, num, local_x, local_y) trail.
# ─────────────────────────────────────────────────────────────────────────────
def rollout_positions(model, state_path, max_steps, deterministic):
    """Run one episode and return the list of (bank, num, lx, ly) RAM readings (pre-step states)."""
    vec, env = build_vec_env(state_path, gif_dir=None, watch=False)
    obs = vec.reset()
    positions = []
    for _ in range(max_steps):
        ram = env.ram_reader.read_all()
        positions.append((ram["map_bank"], ram["map_number"], ram["local_x"], ram["local_y"]))
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, _ = vec.step(action)
        if dones[0]:
            break
    vec.close()
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Background: real stitched image if present, else a generated schematic.
# ─────────────────────────────────────────────────────────────────────────────
def _draw_inset_boxes(img: Image.Image) -> None:
    """Draw each inset=True MapBox (name + rectangle) onto `img` in place, anchored via
    ANCHOR_PX + offset*TILE_PX (i.e. to_image_px(bank, num, 0, 0) for the corner) so gym/lab
    trajectories remain visible as labeled boxes on the real map — these interiors have no art of
    their own on assets/maps/johto_full.png (see map_layout.py's inset derivation notes)."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for (bank, num), box in ml.MAP_INFO.items():
        if not box.inset:
            continue
        x0, y0 = ml.to_image_px(bank, num, 0, 0)
        x1, y1 = ml.to_image_px(bank, num, box.size[0], box.size[1])
        draw.rectangle([x0, y0, x1, y1], fill=box.color, outline=(140, 150, 170))
        draw.text((x0 + 3, y0 + 2), box.name, fill=(230, 235, 245), font=font)


def load_background() -> Image.Image:
    if os.path.exists(ASSET_BG):
        img = Image.open(ASSET_BG).convert("RGB")
        _draw_inset_boxes(img)
        return img
    # Schematic: dark canvas with one labelled rectangle per corridor map.
    w, h = ml.canvas_size()
    img = Image.new("RGB", (w, h), (24, 26, 32))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box in ml.MAP_INFO.values():
        x0 = box.offset[0] * ml.TILE_PX
        y0 = box.offset[1] * ml.TILE_PX
        x1 = (box.offset[0] + box.size[0]) * ml.TILE_PX
        y1 = (box.offset[1] + box.size[1]) * ml.TILE_PX
        draw.rectangle([x0, y0, x1, y1], fill=box.color, outline=(140, 150, 170))
        draw.text((x0 + 3, y0 + 2), box.name, fill=(230, 235, 245), font=font)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap overlay (where time was spent).
# ─────────────────────────────────────────────────────────────────────────────
def heatmap_overlay(points_px, size, radius=3) -> Image.Image:
    """Translucent red heatmap (RGBA) from a list of (px, py) points."""
    w, h = size
    heat = np.zeros((h, w), dtype=np.float32)
    for px, py in points_px:
        x0, x1 = max(0, px - radius), min(w, px + radius + 1)
        y0, y1 = max(0, py - radius), min(h, py + radius + 1)
        heat[y0:y1, x0:x1] += 1.0
    if heat.max() > 0:
        heat = np.log1p(heat)            # compress so heavily-trodden tiles don't wash everything out
        heat /= heat.max()
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (255 * heat).astype(np.uint8)            # red
    rgba[..., 1] = (180 * np.clip(heat * 1.5, 0, 1)).astype(np.uint8)  # toward yellow where hottest
    rgba[..., 3] = (200 * heat).astype(np.uint8)            # alpha
    return Image.fromarray(rgba, mode="RGBA")


def _time_color(t: float) -> tuple[int, int, int]:
    """Blue (early) → yellow (mid) → red (late) for trajectory coloring; t in [0,1]."""
    if t < 0.5:
        u = t / 0.5
        return (int(40 + u * 215), int(120 + u * 135), int(255 - u * 255))  # blue→yellow
    u = (t - 0.5) / 0.5
    return (255, int(255 - u * 255), 0)                                     # yellow→red


def draw_trajectory(base: Image.Image, points_px, width=2) -> Image.Image:
    """Draw the time-colored trajectory polyline (segments split on None/unknown maps)."""
    img = base.convert("RGBA")
    draw = ImageDraw.Draw(img)
    n = len(points_px)
    for i in range(1, n):
        a, b = points_px[i - 1], points_px[i]
        if a is None or b is None:
            continue
        draw.line([a, b], fill=_time_color(i / max(1, n - 1)), width=width)
    # Start (green) and end (white) markers
    drawn = [p for p in points_px if p is not None]
    if drawn:
        for p, c in ((drawn[0], (0, 255, 0)), (drawn[-1], (255, 255, 255))):
            draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=c, outline=(0, 0, 0))
    return img


def positions_to_px(positions):
    """Map raw RAM positions to canvas px, keeping None for maps outside the corridor layout.
    Goes through ram_to_image_px, which un-swaps env/ram_reader.py's local_x/local_y (see
    map_layout.ram_to_image_px docstring)."""
    return [ml.ram_to_image_px(*p) for p in positions]


def finalize_frame(img: Image.Image) -> Image.Image:
    """Crop to the corridor and cap the output width (never emit native 7520-px frames)."""
    x0, y0, x1, y1 = ml.corridor_bbox_px(pad_tiles=8)
    img = img.crop((x0, y0, x1, y1))
    if img.width > OUT_MAX_W:
        img = img.resize((OUT_MAX_W, int(img.height * OUT_MAX_W / img.width)))
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Render a static PNG + an animated GIF for a single trajectory.
# ─────────────────────────────────────────────────────────────────────────────
def render(positions, out_prefix, gif_stride=20, gif_fps=20, title=None):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    bg = load_background()
    size = bg.size
    px = positions_to_px(positions)
    known = [p for p in px if p is not None]

    # Static overlay = heatmap + full trajectory, composed at native scale then cropped/capped.
    composed = Image.alpha_composite(bg.convert("RGBA"), heatmap_overlay(known, size))
    composed = draw_trajectory(composed, px)
    composed = finalize_frame(composed)
    if title:
        ImageDraw.Draw(composed).text((6, composed.height - 14), title, fill=(255, 255, 255))
    png_path = f"{out_prefix}.png"
    composed.convert("RGB").save(png_path)

    # Animated reveal — built from the cropped+scaled frame, never the native-resolution one.
    frames = []
    for k in range(gif_stride, len(px) + gif_stride, gif_stride):
        frame = finalize_frame(draw_trajectory(bg.convert("RGBA"), px[:k])).convert("RGB")
        frames.append(np.asarray(frame))
    gif_path = f"{out_prefix}.gif"
    if frames:
        iio.imwrite(gif_path, np.stack(frames), plugin="pillow", loop=0,
                    duration=int(1000 / gif_fps))

    reached = {ml.MAP_INFO[(p[0], p[1])].name for p in positions if (p[0], p[1]) in ml.MAP_INFO}
    print(f"[viz] {png_path} + {gif_path} | {len(positions)} steps | maps: {', '.join(sorted(reached))}")
    return png_path


def _ckpt_step(path):
    m = re.search(r"_(\d+)_steps", os.path.basename(path))
    return int(m.group(1)) if m else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to a single .zip checkpoint")
    p.add_argument("--all-checkpoints", help="Directory of checkpoints — render one PNG per ckpt + a montage GIF")
    p.add_argument("--state", default="saves/egg_delivered_clean.state")
    p.add_argument("--out", required=True, help="Output path prefix (no extension)")
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[viz] device={device} state={a.state} max_steps={a.max_steps}")

    if a.all_checkpoints:
        ckpts = sorted(glob.glob(os.path.join(a.all_checkpoints, "*_steps.zip")), key=_ckpt_step)
        if not ckpts:
            raise SystemExit(f"No *_steps.zip checkpoints under {a.all_checkpoints}")
        montage = []
        for ckpt in ckpts:
            step = _ckpt_step(ckpt)
            model = PPO.load(ckpt, device=device)
            positions = rollout_positions(model, a.state, a.max_steps, a.deterministic)
            png = render(positions, f"{a.out}_{step:09d}", title=f"step {step:,}")
            montage.append(np.asarray(Image.open(png).convert("RGB")))
        iio.imwrite(f"{a.out}_progression.gif", np.stack(montage), plugin="pillow", loop=0, duration=900)
        print(f"[viz] progression montage → {a.out}_progression.gif ({len(ckpts)} checkpoints)")
    else:
        if not a.model:
            raise SystemExit("Provide --model <ckpt.zip> or --all-checkpoints <dir>")
        model = PPO.load(a.model, device=device)
        positions = rollout_positions(model, a.state, a.max_steps, a.deterministic)
        render(positions, a.out, title=os.path.basename(a.model))


if __name__ == "__main__":
    main()
