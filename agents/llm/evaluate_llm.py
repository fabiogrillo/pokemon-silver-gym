"""Evaluate the LLM agent over N episodes (mirrors agents/rl/evaluate_cnn.py fields).

Each episode is a full ReAct run from cfg.state_path; one JSONL line per episode is written to
runs/eval_logs/llm_<ts>.jsonl with the fields the RL eval also reports, so agents/comparison.py can
join the two. An aggregate summary (badge rate + means) is printed at the end.

Run:
    python -m agents.llm.evaluate_llm --episodes 5
    python -m agents.llm.evaluate_llm --episodes 3 --state saves/violet_city_gym.state

NOTE: an LLM episode is SLOW — ~40 min at ~6 s/step of vision inference on qwen3-vl:8b — so keep N
small. Requires `ollama serve` with the model pulled, plus the ROM + save state.
"""
import argparse
import json
import os
import time

from .config import LLMConfig
from .agent import ReActAgent
from env.pyboy_wrapper import PyBoyWrapper
from env.ram_reader import RAMReader


def evaluate(episodes: int, state_path: str | None = None, log_path: str | None = None) -> list:
    cfg = LLMConfig()
    if state_path:
        cfg.state_path = state_path

    os.makedirs("runs/eval_logs", exist_ok=True)
    if log_path is None:
        log_path = f"runs/eval_logs/llm_{int(time.time())}.jsonl"

    print(f"[eval-llm] model={cfg.model} state={os.path.basename(cfg.state_path)} "
          f"episodes={episodes} max_steps={cfg.max_steps}")
    print(f"[eval-llm] log: {log_path}\n")

    results = []
    agent = ReActAgent(cfg)  # reuses one Ollama client across episodes
    with open(log_path, "w", buffering=1) as f:
        for ep in range(1, episodes + 1):
            wrapper = PyBoyWrapper(cfg.rom_path, cfg.state_path, headless=True)
            reader = RAMReader(wrapper.pyboy)
            t0 = time.time()
            summary = agent.run(wrapper, reader)
            wrapper.pyboy.stop(save=False)

            summary.update({
                "episode": ep,
                "wall_clock_s": round(time.time() - t0, 1),
                "model": cfg.model,
                "state": os.path.basename(cfg.state_path),
            })
            results.append(summary)
            f.write(json.dumps(summary) + "\n")
            print(f"ep {ep:2d}/{episodes} | badge={summary['badge']} "
                  f"battles_won={summary['battles_won']} steps={summary['steps']} "
                  f"tiles={summary['tiles']} tokens={summary['tokens']} "
                  f"stopped={summary['stopped']} wall={summary['wall_clock_s']}s")

    _print_aggregate(results, log_path)
    return results


def _print_aggregate(results: list, log_path: str) -> None:
    n = len(results) or 1
    badge_rate = sum(1 for r in results if r["badge"]) / n
    mean = lambda k: sum(r[k] for r in results) / n
    print(f"\nSummary over {len(results)} episodes:")
    print(f"  Badge rate:        {badge_rate * 100:.1f}%  "
          f"({sum(1 for r in results if r['badge'])}/{len(results)})")
    print(f"  Mean battles won:  {mean('battles_won'):.2f}")
    print(f"  Mean steps:        {mean('steps'):.0f}")
    print(f"  Mean tiles:        {mean('tiles'):.0f}")
    print(f"  Mean tokens:       {mean('tokens'):.0f}")
    print(f"  Mean wall-clock:   {mean('wall_clock_s'):.0f}s")
    print(f"  Eval log:          {log_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the LLM agent over N episodes.")
    ap.add_argument("--episodes", type=int, default=5,
                    help="number of episodes (LLM runs are slow — keep small)")
    ap.add_argument("--state", default=None, help="override cfg.state_path")
    ap.add_argument("--log", default=None, help="override the JSONL output path")
    args = ap.parse_args()
    evaluate(args.episodes, args.state, args.log)


if __name__ == "__main__":
    main()
