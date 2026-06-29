"""Run one LLM-agent episode and write a JSONL trace.

Run: python -m agents.llm.run
Requires: ollama serve + qwen3-vl:8b, ROM + save state.
"""
import json
import os
import time
from .config import LLMConfig
from .agent import ReActAgent
from env.pyboy_wrapper import PyBoyWrapper
from env.ram_reader import RAMReader


def main():
    cfg = LLMConfig()
    os.makedirs(cfg.log_dir, exist_ok=True)
    path = os.path.join(cfg.log_dir, f"run_{int(time.time())}.jsonl")
    wrapper = PyBoyWrapper(cfg.rom_path, cfg.state_path, headless=True)
    reader = RAMReader(wrapper.pyboy)

    with open(path, "w", buffering=1) as f:
        def on_step(step, state, out, obs):
            f.write(json.dumps({
                "step": step, "map": [state["map_bank"], state["map_number"]],
                "pos": [state["local_x"], state["local_y"]], "battle": state["battle_type"],
                "thought": out["thought"][:300], "tool": out["tool_name"], "args": out["args"],
                "obs": obs["note"], "tokens": out["tokens"],
            }) + "\n")

        t0 = time.time()
        summary = ReActAgent(cfg).run(wrapper, reader, on_step=on_step)
        summary["wall_clock_s"] = round(time.time() - t0, 1)
        f.write(json.dumps({"summary": summary}) + "\n")
    wrapper.pyboy.stop(save=False)
    print("SUMMARY:", summary)
    print("Trace:", path)


if __name__ == "__main__":
    main()