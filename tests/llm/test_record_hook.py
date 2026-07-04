from agents.llm.config import LLMConfig
from agents.llm.agent import ReActAgent


class StubClient:
    """Offline stand-in for OllamaClient: always presses `a`."""
    def chat(self, messages, tools):
        return {"thought": "stub", "tool_name": "press", "args": {"button": "a"}, "tokens": 0}


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
