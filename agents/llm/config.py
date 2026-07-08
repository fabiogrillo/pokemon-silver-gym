from dataclasses import dataclass

# Kept short deliberately: qwen3-vl:8b always thinks and the longer this prompt is, the more it
# thinks before answering, which raises the empty-response rate (thinking-only turns with no
# tool call). See agents/llm/llm_client.py's retry-on-empty for the other half of that fix.
GOAL = (
    "Goal: reach Violet City (New Bark Town -> Route 29 -> Cherrygrove -> Route 30 -> Route 31 "
    "-> Violet City), enter the Gym, climb to Falkner, defeat him for the Zephyr Badge "
    "(Flying-type; weak to Electric/Rock)."
)

SYSTEM_PROMPT = (
    "You play Pokemon Silver via tool calls: read the text state and screenshot, think "
    "briefly, call exactly ONE tool -- never prose only. In battle press one button at a "
    "time.\n\n"
    "Overworld: use 'Walkable directions from here'; head toward the goal, never repeat a "
    "non-walkable direction. Crossing a map edge (route/gate/door) is expected progress. If "
    "none are walkable, a trainer/NPC blocks you -- press 'a' until battle starts or dialogue "
    "clears, then resume moving. Coordinates: x=east, y=south, up decreases y. Prefer "
    "`navigate_to(x, y)` for a known walkable tile on the current map, else `move`/`press`.\n\n"
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
    state_path: str = "saves/egg_delivered_clean.state"  # start of the New Bark -> Violet corridor
    max_steps: int = 1000  # enough headroom that the cap measures how far the agent gets, not
                           # how quickly it stalls
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
    confine_to_home_map: bool = False  # keep the agent on its starting map; must be False for the corridor
    leg_mode: bool = True  # harness-owned goals via agents/llm/legs.LegTracker (each turn's note names
                            # the current corridor leg's target). False = the model picks its own
                            # navigate_to targets.