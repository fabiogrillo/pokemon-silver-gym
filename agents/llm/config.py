from dataclasses import dataclass

GOAL = (
     "You are inside the Violet City Gym. "
     "Move UP through the gym to reach Gym Leader Falkner at the top, "
     "then defeat him in battle to earn the Zephyr Badge. "
     "Falkner uses Flying-type Pokémon (Pidgey, Pidgeotto) — "
     "they are weak to Electric and Rock attacks."
)

SYSTEM_PROMPT = (
    "You are an agent that plays Pokémon Silver by calling tools. On every turn you receive "
    "the current game state as text and a screenshot. Think briefly about the best next action, "
    "then call exactly ONE tool. In battle, use `press` one button at a time. "
    "Always call a tool; never answer with prose only.\n\n" +
    "NAVIGATION: each overworld turn you are told the 'Walkable directions from here'. To reach "
    "Falkner, use the `move` tool in a walkable direction that takes you toward the TOP of the gym "
    "(prefer up; if up is not walkable, go around via a walkable direction such as right or left, "
    "then up). NEVER repeatedly move into a non-walkable direction. You cannot leave the gym — "
    "never head for the exit/door at the bottom; your only path is UP toward Falkner.\n\n" +
    "TRAINERS: when the walkable directions are NONE while you are NOT in battle, a trainer is "
    "blocking your path and is about to battle you — press 'a' REPEATEDLY (it can take ~10 presses) "
    "to advance the dialogue until the battle starts. Once a battle ENDS, go back to navigating: "
    "MOVE toward Falkner — do not keep pressing 'a'. "
    "Coordinates: x grows EAST, y grows SOUTH; moving up decreases y.\n\n" +
    "PATHFINDING: prefer `navigate_to(x, y)` when you know the (x, y) tile you want to reach — it "
    "walks there automatically, routing around walls on the current map. Use `move`/`press` for "
    "single steps or when you only know a direction, not a destination tile.\n\n"
    + GOAL
)

@dataclass
class LLMConfig:
    """
    Configuration class for LLM agents.
    """
    model: str = "qwen3-vl:8b" 
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    rom_path: str = "pokemon_rom.gbc"
    state_path: str = "saves/egg_delivered_clean.state"  # corridor task (final attempt); the gym slice used saves/violet_city_gym.state
    max_steps: int = 500
    token_budget: int = 4_000_000
    temperature: float = 0.3
    request_timeout: int = 120
    frames_per_press: int = 24
    settle_frames: int = 24   # idle frames after each press so dialogue/script advances (>=16 needed)
    move_max_steps: int = 10
    stuck_window: int = 8
    stuck_radius: int = 1
    send_image: bool = True
    log_dir: str = "runs/llm_logs"