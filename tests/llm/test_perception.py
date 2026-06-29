from agents.llm.perception import format_state_text


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