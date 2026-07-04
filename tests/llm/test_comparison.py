from agents.comparison import summarize


def test_summarize_badge_rate_and_means():
    recs = [
        {"badge": True, "steps": 100, "tiles": 50, "battles_won": 2, "tokens": 1000, "wall_clock_s": 30},
        {"badge": False, "steps": 200, "tiles": 80, "battles_won": 0, "tokens": 3000, "wall_clock_s": 60},
    ]
    s = summarize(recs)
    assert s["badge_rate"] == 0.5
    assert s["mean_steps"] == 150
    assert s["mean_tiles"] == 65
    assert s["mean_battles_won"] == 1.0


def test_summarize_empty_is_safe():
    s = summarize([])
    assert s["badge_rate"] == 0.0
    assert s["mean_steps"] == 0.0


def test_summarize_tolerates_missing_optional_fields():
    # RL eval JSONL has no tokens/wall_clock_s — those must default to 0, not KeyError.
    s = summarize([{"badge": False, "steps": 840, "tiles": 120, "battles_won": 3}])
    assert s["mean_tokens"] == 0.0
    assert s["mean_wall_clock_s"] == 0.0
    assert s["mean_steps"] == 840
