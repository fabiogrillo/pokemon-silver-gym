from agents.llm.config import LLMConfig
from agents.llm.agent import ReActAgent


class StubClient:
    """Offline stand-in for OllamaClient: always presses `a`."""
    def chat(self, messages, tools):
        return {"thought": "stub", "tool_name": "press", "args": {"button": "a"}, "tokens": 0}


class EmptyStubClient:
    """Offline stand-in for OllamaClient that always returns an empty (no tool_call) reply —
    simulates qwen3-vl:8b ending a turn with reasoning only, even after llm_client's retry."""
    def chat(self, messages, tools):
        return {"thought": "", "tool_name": None, "args": {}, "tokens": 0}


def test_on_step_receives_frame(emulator):
    wrapper, reader = emulator
    cfg = LLMConfig()
    cfg.max_steps = 2
    agent = ReActAgent(cfg)
    agent.client = StubClient()  # no network

    frames = []

    def on_step(step, state, out, obs, frame):
        frames.append(frame)

    agent.run(wrapper, reader, on_step=on_step)
    assert len(frames) == 2
    assert frames[0].shape[0] == 144 and frames[0].shape[1] == 160  # GB screen


def test_empty_reply_in_overworld_falls_back_to_wait_not_press_a(emulator):
    """An empty (tool_name None) reply while NOT in battle must fall back to a harmless
    wait_frames, never a press of 'a' — pressing 'a' facing an NPC self-feeds their textbox
    open/close loop forever (see runs/llm_logs/run_1783362673.jsonl)."""
    wrapper, reader = emulator
    cfg = LLMConfig()
    cfg.max_steps = 1
    agent = ReActAgent(cfg)
    agent.client = EmptyStubClient()  # no network; always empty

    observations = []

    def on_step(step, state, out, obs, frame):
        observations.append((state["battle_type"], obs))

    agent.run(wrapper, reader, on_step=on_step)
    assert len(observations) == 1
    battle_type, obs = observations[0]
    assert battle_type == 0  # boot state is overworld, not battle
    assert "wait" in obs["note"]
    assert "press a" not in obs["note"]
