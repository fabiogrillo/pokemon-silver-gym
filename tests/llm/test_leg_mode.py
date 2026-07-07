"""Unit tests for LLM-3's leg_mode: the harness (agents/llm/legs.py's LegTracker), not the model,
picks the current strategic target. `ReActAgent.run` appends `LegTracker.goal_note(...)` to every
overworld turn's memory note (after the walkable-directions line) when `cfg.leg_mode` is True (the
default), and reports how many corridor legs were completed in the run summary.
"""
from agents.llm import legs
from agents.llm.config import LLMConfig
from agents.llm.agent import ReActAgent


class SpyStubClient:
    """Offline stand-in for OllamaClient: always presses 'a', and records every messages list it
    is called with so tests can inspect the exact text sent to the model."""
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append(messages)
        return {"thought": "stub", "tool_name": "press", "args": {"button": "a"}, "tokens": 0}


def _user_texts(spy):
    """The text portion of every user-turn content list across all recorded calls."""
    texts = []
    for messages in spy.calls:
        content = messages[1]["content"]
        texts.append(content[0]["text"] if isinstance(content, list) else content)
    return texts


def test_overworld_note_carries_the_new_bark_leg_target_when_leg_mode_enabled(emulator):
    wrapper, reader = emulator
    cfg = LLMConfig()
    cfg.max_steps = 2
    cfg.leg_mode = True
    agent = ReActAgent(cfg)
    spy = SpyStubClient()
    agent.client = spy

    agent.run(wrapper, reader)

    new_bark_leg = legs.LEGS[0]
    tx, ty = new_bark_leg.target
    texts = _user_texts(spy)
    assert texts, "expected at least one recorded call"
    assert all(f"Leg '{new_bark_leg.name}'" in t for t in texts)
    assert all(str(tx) in t and str(ty) in t for t in texts)
    assert all(f"navigate_to({tx}, {ty})" in t for t in texts)


def test_leg_note_is_appended_after_the_walkable_directions_line(emulator):
    wrapper, reader = emulator
    cfg = LLMConfig()
    cfg.max_steps = 1
    cfg.leg_mode = True
    agent = ReActAgent(cfg)
    spy = SpyStubClient()
    agent.client = spy

    agent.run(wrapper, reader)

    text = _user_texts(spy)[0]
    walkable_idx = text.index("Walkable directions from here")
    leg_idx = text.index("Leg '")
    assert leg_idx > walkable_idx


def test_leg_mode_disabled_omits_the_leg_goal_note(emulator):
    wrapper, reader = emulator
    cfg = LLMConfig()
    cfg.max_steps = 2
    cfg.leg_mode = False
    agent = ReActAgent(cfg)
    spy = SpyStubClient()
    agent.client = spy

    agent.run(wrapper, reader)

    new_bark_leg = legs.LEGS[0]
    texts = _user_texts(spy)
    assert texts, "expected at least one recorded call"
    assert all(f"Leg '{new_bark_leg.name}'" not in t for t in texts)


def test_summary_reports_legs_completed(emulator):
    wrapper, reader = emulator
    cfg = LLMConfig()
    cfg.max_steps = 2
    agent = ReActAgent(cfg)
    agent.client = SpyStubClient()

    summary = agent.run(wrapper, reader)

    assert "legs_completed" in summary
    assert isinstance(summary["legs_completed"], int)
    # boot state (saves/egg_delivered_clean.state) is on New Bark Town, leg 0's map. Pressing 'a'
    # in place for 2 steps never leaves that map, so LegTracker.completed_count counts exactly leg
    # 0 as visited (consecutively from the start) and stops at leg 1, whose map was never visited.
    assert summary["legs_completed"] == 1


def test_leg_mode_defaults_to_true():
    assert LLMConfig().leg_mode is True
