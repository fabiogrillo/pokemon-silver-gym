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

def format_state_text(state: dict) -> str:
    lines = [
        f"Map {state['map_bank']}-{state['map_number']} at ({state['local_x']}, {state['local_y']})",
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