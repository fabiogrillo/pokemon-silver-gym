from agents.llm.perception import waypoint_ordinal


def test_waypoint_ordinals():
    assert waypoint_ordinal(24, 4) == 0   # New Bark
    assert waypoint_ordinal(24, 3) == 0   # Route 29
    assert waypoint_ordinal(26, 3) == 1   # Cherrygrove
    assert waypoint_ordinal(26, 1) == 2   # Route 30
    assert waypoint_ordinal(26, 2) == 3   # Route 31
    assert waypoint_ordinal(10, 5) == 4   # Violet City
    assert waypoint_ordinal(10, 7) == 5   # gym
    assert waypoint_ordinal(3, 70) == 0   # unknown maps -> 0
