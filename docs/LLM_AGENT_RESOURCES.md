# LLM Agent — Learning Resources (by topic & task)

You're starting close to zero on LLMs/agents — that's fine. This doc lists **what to
study, in order, mapped to each build task**, and ends with **how it all connects** into
a working agent. Keep it tight: study the foundations, then build.

**Resource tags:** 📄 article · 🎥 video · 📑 paper · 🛠️ docs · ⏱ rough time.

---

## ▶ How it all fits (read this first)

To build the Pokémon LLM agent you chain these pieces — each topic below maps to a step:

```
[1] what an LLM is  →  [2] what an "agent" is  →  [3] the ReAct loop + tool calling
        →  [4] run a model locally (Ollama)  →  [5] wire 3 tools to the Pokémon env
        →  [6] evaluate like the RL agent  →  [7] compare RL vs LLM
```

The agent is just a **loop**: read game state → the model *thinks* → it calls a *tool*
(a button press) → you run it → repeat. Everything else is plumbing around that loop.

---

## 1. Foundations — what an LLM actually is  ⏱ ~1.5h

| | Resource | What you get |
|---|---|---|
| 🎥 | [3Blue1Brown — Transformers, visually](https://www.3blue1brown.com/lessons/gpt/) + [Attention](https://www.3blue1brown.com/lessons/attention/) | Visual intuition for how a transformer turns tokens into predictions. |
| 📄 | [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | The classic picture-by-picture explainer of the architecture. |
| 🎥 | [Karpathy — Intro to LLMs (1h)](https://www.youtube.com/watch?v=zjkBMFhNj_g) | The big picture: pretraining, fine-tuning, what LLMs can/can't do. No math needed. |

*Goal:* understand that an LLM predicts text, can follow instructions, and can be asked to output a structured action — which is what makes tool-calling possible.

## 2. Foundations — what an "agentic" LLM is  ⏱ ~1h

| | Resource | What you get |
|---|---|---|
| 📄 | [Lil'Log — LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | The reference map: planning, memory, tool use. Read **Tool Use** + **Memory** closely. |
| 📄 | [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | Practical: the difference between *workflows* and *agents*, and when to keep it simple. |

## 3. The agent pattern — ReAct + tool calling  ⏱ ~1h

| | Resource | What you get |
|---|---|---|
| 📑 | [ReAct paper (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) | The core idea we use: interleave a *Thought* with an *Action*. Read intro + the ALFWorld example (a game-like agent). |
| 📄 | Lil'Log §Tool Use (above) | How a model decides *which* tool to call and with what arguments. |

*Goal:* be able to describe the loop "Thought → Action(tool) → Observation → repeat" in your own words.

## 4. Running it locally — Ollama tool use  ⏱ ~45min

| | Resource | What you get |
|---|---|---|
| 🛠️ | [Ollama — OpenAI compatibility](https://ollama.com/blog/openai-compatibility) | Use the standard `openai` Python client against `http://localhost:11434/v1/`. |
| 🛠️ | [Ollama — Tool support](https://ollama.com/blog/tool-support) | How you pass `tools=[…]` JSON schemas and read back `tool_calls`. |

*Reality check:* small local models are uneven at tool calling — **keep ≤3 tools**,
validate/repair the JSON before executing, and pick a tool-capable model (you have
qwen3.x / gemma; Llama 3.1 and Mistral Nemo are also good).

## 5. Prior art — LLMs playing Pokémon  ⏱ ~1h

| | Resource | What you get |
|---|---|---|
| 📑 | [PokéLLMon (arXiv 2402.01118)](https://arxiv.org/abs/2402.01118) · [project](https://poke-llm-on.github.io/) | Three transferable tricks: in-context feedback, knowledge-augmented prompts, consistent actions. Battle-only but the prompting transfers. |
| 🎥 | [Claude Plays Pokémon — talk](https://www.youtube.com/watch?v=CXhYDOvgpuU) | What breaks over long horizons (memory, getting stuck) in a full game-playing agent. |

---

## 6. Build tasks — resource → code map

| Task | File | Lean on |
|------|------|--------|
| Game state → text | `env/ram_reader.py` (`to_prompt()`) | §1 (why text beats pixels for an LLM), PokéLLMon prompt design |
| 3 tools: `press_button`, `get_game_state`, `wait_frames` | `agents/llm/tools.py` | §4 Ollama tool support |
| ReAct loop | `agents/llm/agent.py` | §3 ReAct + Lil'Log Tool Use |
| Runner + logging of Thoughts | `agents/llm/run.py` | §3 (log every Thought — it's your debugger + blog gold) |
| Model/config | `agents/llm/config.py` | §4 (model choice, max steps, token budget) |

## 7. Evaluation  ⏱ build, not read

Mirror `agents/rl/evaluate_cnn.py` → `agents/llm/evaluate_llm.py`: same episodes from
`saves/violet_city_gym.state`, same per-episode JSONL fields (`badge`, `steps`, `tiles`,
`battles_won`, wall-clock, **tokens**). Define success *before* running so the comparison
is honest.

## 8. Comparison — RL vs LLM

Reuse the JSONL logs in `runs/eval_logs/`; `agents/comparison.py` joins RL + LLM runs into
one table (badge rate, mean steps, tile coverage, tokens, wall-clock). **The story:** RL
*discovers by trial-and-error* (and hit an exploration wall); the LLM *reasons about the
goal*. That contrast is the spine of the blog post.

## 9. (optional) Fine-tuning — background only

Not needed for this agent (it's prompt-based). If curious later: 📄 Hugging Face LoRA/PEFT
intro. Skip until the prompt-based agent works end-to-end.

---

## ✅ Reading order (the minimum path)
1. §1 3B1B + Illustrated Transformer (intuition)
2. §2 Lil'Log Tool Use + Anthropic agents
3. §3 ReAct paper (intro + ALFWorld)
4. §4 Ollama tool support
5. §5 PokéLLMon §3
→ then write `to_prompt()` and `tools.py`, and the loop in `agent.py`.

**Synthesis:** once §1–§5 click, the build is small — three tools, one loop, one eval
script that mirrors the RL one. You already have the environment, the save state, and the
metrics; the LLM agent just plugs reasoning into the same socket the RL policy used.
