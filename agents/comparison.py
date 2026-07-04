"""Join RL + LLM eval JSONL logs into one comparison table.

Both eval harnesses (agents/rl/evaluate_cnn.py --log, agents/llm/evaluate_llm.py) write one JSON
object per episode with at least `badge`, `steps`, `tiles`, `battles_won` (the LLM log adds `tokens`
and `wall_clock_s`; the RL log omits those, so they default to 0). This aggregates each into one row.

Run:
    python -m agents.comparison --rl runs/eval_logs/<rl>.jsonl --llm runs/eval_logs/<llm>.jsonl
"""
import argparse
import json


COLUMNS = ["n", "badge_rate", "mean_steps", "mean_tiles", "mean_battles_won",
           "mean_tokens", "mean_wall_clock_s"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def summarize(records: list) -> dict:
    n = len(records)
    return {
        "n": n,
        "badge_rate": (sum(1 for r in records if r.get("badge")) / n) if n else 0.0,
        "mean_steps": _mean([r["steps"] for r in records]),
        "mean_tiles": _mean([r["tiles"] for r in records]),
        "mean_battles_won": _mean([r.get("battles_won", 0) for r in records]),
        "mean_tokens": _mean([r.get("tokens", 0) for r in records]),
        "mean_wall_clock_s": _mean([r.get("wall_clock_s", 0) for r in records]),
    }


def _load(path):
    """Load per-episode records from a JSONL eval log (skips any {"summary": ...} trace line)."""
    if not path:
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "summary" in obj:  # run.py-style trace footer, not an episode
                continue
            records.append(obj)
    return records


def _fmt(value):
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def render_table(rows: dict) -> str:
    """rows: {agent_label: summary_dict}. Returns a Markdown table string."""
    header = "| Agent | " + " | ".join(COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    lines = [header, sep]
    for agent, s in rows.items():
        lines.append("| " + agent + " | " + " | ".join(_fmt(s[c]) for c in COLUMNS) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="RL vs LLM comparison table from eval JSONL logs.")
    ap.add_argument("--rl", help="RL eval JSONL (agents/rl/evaluate_cnn.py --log)")
    ap.add_argument("--llm", help="LLM eval JSONL (agents/llm/evaluate_llm.py)")
    args = ap.parse_args()

    rows = {}
    if args.rl:
        rows["RL (CnnPolicy)"] = summarize(_load(args.rl))
    if args.llm:
        rows["LLM (qwen3-vl:8b)"] = summarize(_load(args.llm))
    if not rows:
        ap.error("provide --rl and/or --llm")
    print(render_table(rows))


if __name__ == "__main__":
    main()
