from agents.llm.memory import ShortTermMemory


def _pos(x, y, mb=24, mn=5, bt=0):
    return {"local_x": x, "local_y": y, "map_bank": mb, "map_number": mn, "battle_type": bt}


def test_not_stuck_when_moving():
    m = ShortTermMemory(window=10, stuck_window=4, stuck_radius=1)
    for i in range(4):
        m.record(_pos(i, 0), "t", "move", {"direction": "right", "steps": 1}, "ok")
    assert m.is_stuck() is False


def test_stuck_when_oscillating_in_place():
    m = ShortTermMemory(window=10, stuck_window=4, stuck_radius=1)
    for x in [5, 6, 5, 6]:
        m.record(_pos(x, 8), "t", "move", {"direction": "left", "steps": 1}, "ok")
    assert m.is_stuck() is True


def test_render_note_warns_when_stuck():
    m = ShortTermMemory(window=10, stuck_window=4, stuck_radius=1)
    for x in [5, 6, 5, 6]:
        m.record(_pos(x, 8), "t", "move", {"direction": "left", "steps": 1}, "ok")
    note = m.render_note()
    # the stuck warning no longer uses the word "stuck"; it now steers toward the walkable
    # directions (or pressing 'a' at a trainer) — assert the warning fires and gives direction.
    assert "warning" in note.lower()
    assert "not moved" in note.lower()