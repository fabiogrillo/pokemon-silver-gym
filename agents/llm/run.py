"""Run one LLM-agent episode and write a JSONL trace.

Run: python -m agents.llm.run
Requires: ollama serve + qwen3-vl:8b, ROM + save state.
"""
import json
import os
import time
import argparse
from .config import LLMConfig
from .agent import ReActAgent
from env.pyboy_wrapper import PyBoyWrapper
from env.ram_reader import RAMReader


def main():
    cfg = LLMConfig()
    os.makedirs(cfg.log_dir, exist_ok=True)
    path = os.path.join(cfg.log_dir, f"run_{int(time.time())}.jsonl")
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="open PyBoy window + print reasonings")
    args = ap.parse_args()
    
    wrapper = PyBoyWrapper(cfg.rom_path, cfg.state_path, headless=not args.watch)
    reader = RAMReader(wrapper.pyboy)

    with open(path, "w", buffering=1) as f:
        def on_step(step, state, out, obs):
            if args.watch:
                thought = out["thought"] or "(no reasoning text)"
                print(f"[{step}] {thought}\n -> {out['tool_name']}({out['args']}) | "
                      f"{obs['note']} map {state['map_bank']}-{state['map_number']} "
                      f"@({state['local_x']},{state['local_y']}) "
                      f"battle={state['battle_type']} tokens={out['tokens']}")
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