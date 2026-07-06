import base64
import io
import numpy as np
from PIL import Image


def encode_screenshot(frame) -> str:
    arr = np.asarray(frame)
    if arr.shape[-1] == 4:        # RGBA → RGB (PyBoy screen has alpha)
        arr = arr[:, :, :3]
    img = Image.fromarray(arr.astype("uint8"), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_user_content(state: dict, frame, memory_note: str, send_image: bool) -> list:
    text = format_state_text(state)
    if memory_note:
        text += "\n\n" + memory_note
    content = [{"type": "text", "text": text}]
    if send_image:
        b64 = encode_screenshot(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return content

# Corridor waypoints keyed by (map_bank, map_number); no axis swap involved here (map ids, not
# coordinates) — see format_state_text below for the coordinate un-swap.
WAYPOINT_MAPS: list[tuple[tuple[int, int], int]] = [
    ((26, 3), 1),  # Cherrygrove City
    ((26, 1), 2),  # Route 30
    ((26, 2), 3),  # Route 31
    ((10, 5), 4),  # Violet City
    ((10, 7), 5),  # Violet City Gym
]
_WAYPOINT_LOOKUP = dict(WAYPOINT_MAPS)


def waypoint_ordinal(bank: int, num: int) -> int:
    """Corridor-waypoint ordinal for a map id (New Bark/Route 29 and unknown maps -> 0)."""
    return _WAYPOINT_LOOKUP.get((bank, num), 0)


def format_state_text(state: dict) -> str:
    # ram local_x/local_y are swapped (wYCoord/wXCoord); perception must report TRUE (x, y).
    # Un-swap here, mirroring agents/rl/map_layout.ram_to_image_px.
    true_x, true_y = state["local_y"], state["local_x"]
    lines = [
        f"Map {state['map_bank']}-{state['map_number']} at ({true_x}, {true_y})",
        f"Badges: {state['badge_count']} (Zephyr: {'yes' if state['zephyr'] else 'no'})",
        f"Lead Pokémon: L{state['lead_level']} HP {state['lead_hp']}/{state['lead_max_hp']}",
        f"Trainers beaten — route: {state['route_trainers_beaten']}, gym: {state['gym_trainers_beaten']}",
    ]
    if state["battle_type"] == 0:
        lines.insert(0, "Mode: Overworld")
    else:
        kind = {1: "wild", 2: "trainer", 3: "gym"}.get(state["battle_type"], "unknown")
        lines.insert(0, f"Mode: Battle ({kind})")
        lines.append(
            f"Enemy: L{state['enemy_lead_level']} HP {state['enemy_hp']}/{state['enemy_max_hp']}"
        )
    return "\n".join(lines)