"""Unit tests for the confine_to_home_map gate in agents/llm/agent.py.

The gym-slice harness used to unconditionally undo any map change (snapshot reload) and filter
map-exit directions out of the walkable-directions probe. That behavior must now be gated behind
`cfg.confine_to_home_map` (default False, for the corridor task) — a map change should be a
normal, un-undone outcome when the flag is off, and still be undone when the flag is explicitly
turned on for the gym slice.

We stub out `probe_walkable` (it does its own save/load_state cycling that is orthogonal to the
confinement seam under test) and `execute_tool` (so we can deterministically flip the RAM-reported
map on the "after action" read, without needing a real map-exit walk) so the only remaining
`wrapper.pyboy.load_state` calls are from the confinement undo path itself.
"""
import agents.llm.agent as agent_mod
from agents.llm.config import LLMConfig
from agents.llm.agent import ReActAgent


class StubClient:
    """Offline stand-in for OllamaClient: always presses `a`."""
    def chat(self, messages, tools):
        return {"thought": "stub", "tool_name": "press", "args": {"button": "a"}, "tokens": 0}


def _rig(wrapper, reader, monkeypatch):
    """Common test rig: stub probe_walkable/execute_tool, flip the reported map after the first
    action, and spy on wrapper.pyboy.load_state. Returns the load_state call counter (a dict)."""
    real_read = reader.read_all
    home = real_read()
    new_map_number = home["map_number"] + 1
    moved = {"flag": False}

    def fake_read():
        s = real_read()
        if moved["flag"]:
            s["map_bank"], s["map_number"] = home["map_bank"], new_map_number
        return s
    monkeypatch.setattr(reader, "read_all", fake_read)

    # probe_walkable does its own save/load_state churn every step, unrelated to the confinement
    # seam under test; stub it so the only load_state calls we observe are the undo path's.
    monkeypatch.setattr(agent_mod, "probe_walkable", lambda *a, **kw: ["up"])

    real_execute_tool = agent_mod.execute_tool

    def fake_execute_tool(name, args, wrapper, reader, cfg):
        obs = real_execute_tool(name, args, wrapper, reader, cfg)
        moved["flag"] = True  # simulate: this action carried us onto a new map
        return obs
    monkeypatch.setattr(agent_mod, "execute_tool", fake_execute_tool)

    # PyBoy is a C-extension type (its `load_state` attribute is read-only, even at the class
    # level), so we can't monkeypatch it directly. Instead, swap `wrapper.pyboy` (a plain instance
    # attribute on our own PyBoyWrapper) for a thin proxy that counts `load_state` calls and
    # forwards everything else untouched.
    load_state_calls = {"n": 0}
    real_pyboy = wrapper.pyboy

    class _LoadStateSpy:
        def load_state(self, *a, **kw):
            load_state_calls["n"] += 1
            return real_pyboy.load_state(*a, **kw)

        def __getattr__(self, attr):
            return getattr(real_pyboy, attr)
    monkeypatch.setattr(wrapper, "pyboy", _LoadStateSpy())

    return load_state_calls, new_map_number, home["map_bank"]


def test_map_change_not_undone_when_not_confined(emulator, monkeypatch):
    wrapper, reader = emulator
    load_state_calls, new_map_number, bank = _rig(wrapper, reader, monkeypatch)

    cfg = LLMConfig()
    cfg.max_steps = 2
    cfg.confine_to_home_map = False
    agent = ReActAgent(cfg)
    agent.client = StubClient()

    seen_maps = []
    notes = []

    def on_step(step, state, out, obs, frame):
        seen_maps.append((state["map_bank"], state["map_number"]))
        notes.append(obs["note"])

    agent.run(wrapper, reader, on_step=on_step)

    # the confinement/undo path never fires, so pyboy.load_state is never called
    assert load_state_calls["n"] == 0
    # the note is never annotated with the gym-confinement message
    assert all("cannot leave the gym" not in n for n in notes)
    # step 1's observation reflects the NEW map (the transition was not reloaded away)
    assert seen_maps[1] == (bank, new_map_number)


def test_map_change_undone_when_confined(emulator, monkeypatch):
    wrapper, reader = emulator
    load_state_calls, new_map_number, bank = _rig(wrapper, reader, monkeypatch)

    cfg = LLMConfig()
    cfg.max_steps = 2
    cfg.confine_to_home_map = True
    agent = ReActAgent(cfg)
    agent.client = StubClient()

    notes = []

    def on_step(step, state, out, obs, frame):
        notes.append(obs["note"])

    agent.run(wrapper, reader, on_step=on_step)

    # the undo path fires and reloads a snapshot
    assert load_state_calls["n"] > 0
    assert any("cannot leave the gym" in n for n in notes)
