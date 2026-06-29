from dataclasses import dataclass

GOAL = (
    "You are playing Pokémon Silver. Starting from Elm's Lab area (the egg is already "
    "delivered), navigate through Cherrygrove City, Route 30/31, into Violet City, enter "
    "the Gym, and defeat Gym Leader Falkner to earn the Zephyr Badge. Make steady progress "
    "toward Violet City; do not wander backward."
)

SYSTEM_PROMPT = (
    "You are an agent that plays Pokémon Silver by calling tools. On every turn you receive "
    "the current game state as text and a screenshot. Think briefly about the best next action, "
    "then call exactly ONE tool. In the overworld, prefer the `move` tool to travel several "
    "tiles at once toward your goal. In battle, use `press` one button at a time. Use `press(\"a\")` "
    "to talk to people, confirm menus, and advance dialogue. If you seem stuck, change direction. "
    "Always call a tool; never answer with prose only.\n\n" + GOAL
)

@dataclass
class LLMConfig:
    """
    Configuration class for LLM agents.
    """
    model: str = "qwen3-vl:8b", 
    base_url: str = "https://localhost:11434/v1",
    api_key: str = "ollama", 
    rom_path: str = "pokemon_rom.gbc",
    state_path: str = "saves?egg_delivered_clean.state",
    max_step: int = 500,
    token_budget: int = 400_000,
    temperature: float = 0.73,
    request_timeout: int = 120,
    frames_per_press: int = 24,
    move_max_steps: int = 10,
    stuck_window: int = 8,
    stuck_radius: int = 1,
    send_image: bool = True,
    log_dir: str = "runs/llm_logs"