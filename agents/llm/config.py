from dataclasses import dataclass

GOAL = (
     "You start near New Bark Town. Your objective is to travel the overworld corridor to "
     "Violet City and defeat Gym Leader Falkner: New Bark Town -> Route 29 -> Cherrygrove City -> "
     "Route 30 -> Route 31 -> Violet City -> enter the Violet City Gym -> move up through the gym "
     "to reach Falkner at the top, then defeat him in battle to earn the Zephyr Badge. "
     "Falkner uses Flying-type Pokémon (Pidgey, Pidgeotto) — "
     "they are weak to Electric and Rock attacks."
)

SYSTEM_PROMPT = (
    "You are an agent that plays Pokémon Silver by calling tools. On every turn you receive "
    "the current game state as text and a screenshot. Think briefly about the best next action, "
    "then call exactly ONE tool. In battle, use `press` one button at a time. "
    "Always call a tool; never answer with prose only.\n\n" +
    "NAVIGATION: each overworld turn you are told the 'Walkable directions from here'. Head "
    "generally toward Violet City (north of New Bark Town, through Cherrygrove and the connecting "
    "routes), then into the gym and up toward Falkner. NEVER repeatedly move into a non-walkable "
    "direction. Leaving one map for the next (e.g. through a route edge, town gate, or door) is "
    "expected and correct — that is how you make progress toward Violet City.\n\n" +
    "TRAINERS: when the walkable directions are NONE while you are NOT in battle, a trainer is "
    "blocking your path and is about to battle you — press 'a' REPEATEDLY (it can take ~10 presses) "
    "to advance the dialogue until the battle starts. Once a battle ENDS, go back to navigating: "
    "MOVE toward your destination — do not keep pressing 'a'. "
    "Coordinates: x grows EAST, y grows SOUTH; moving up decreases y.\n\n" +
    "PATHFINDING: prefer `navigate_to(x, y)` when you know the (x, y) tile you want to reach on the "
    "CURRENT map — it walks there automatically, routing around walls. Its target must be a "
    "walkable tile on the CURRENT map, so it cannot aim at a tile on the next map over. Use "
    "`move`/`press` for single steps, or to walk onto a map exit (a route edge, town gate, or door "
    "at the edge of the current map) — once you cross it you are on the next map, which is normal "
    "and means you made progress toward Violet City; re-orient and keep going (with `navigate_to` "
    "or `move`) from there.\n\n"
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
    confine_to_home_map: bool = False  # gym-slice harness guardrail (June arc); MUST be False for the corridor task