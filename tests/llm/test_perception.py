from agents.llm.perception import format_state_text
import base64
import numpy as np
from agents.llm.perception import encode_screenshot, build_user_content


def test_encode_screenshot_returns_png_base64():
    frame = np.zeros((144, 160, 4), dtype=np.uint8)  # RGBA like PyBoy
    b64 = encode_screenshot(frame)
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_build_user_content_with_image():
    frame = np.zeros((144, 160, 4), dtype=np.uint8)
    content = build_user_content(_base_state(), frame, "note: stuck", send_image=True)
    kinds = [c["type"] for c in content]
    assert "text" in kinds and "image_url" in kinds
    text = next(c["text"] for c in content if c["type"] == "text")
    assert "Map 24-5" in text and "stuck" in text


def test_build_user_content_text_only():
    frame = np.zeros((144, 160, 4), dtype=np.uint8)
    content = build_user_content(_base_state(), frame, "", send_image=False)
    assert all(c["type"] == "text" for c in content)

def _base_state(**over):
    s = {
        "map_bank": 24, "map_number": 5, "local_x": 4, "local_y": 6,
        "battle_type": 0, "party_count": 1, "lead_hp": 19, "lead_max_hp": 19,
        "lead_level": 5, "badge_count": 0, "zephyr": False,
        "enemy_lead_level": 0, "enemy_hp": 0, "enemy_max_hp": 0,
        "gym_trainers_beaten": 0, "route_trainers_beaten": 0,
    }
    s.update(over)
    return s


def test_overworld_text_has_position_and_no_battle():
    txt = format_state_text(_base_state())
    assert "Map 24-5" in txt
    assert "(4, 6)" in txt
    assert "Overworld" in txt
    assert "Battle" not in txt


def test_battle_text_includes_enemy():
    txt = format_state_text(_base_state(battle_type=2, enemy_lead_level=9,
                                        enemy_hp=12, enemy_max_hp=20))
    assert "Battle" in txt
    assert "enemy" in txt.lower()
    assert "9" in txt
