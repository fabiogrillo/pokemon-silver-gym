def format_state_text(state: dict) -> str:
    lines = [
        f"Map {state['map_bank']}-{state['map_number']} at ({state['local_x']}, {state['local_y']})",
        f"Badges: {state['badge_count']} (Zephyr: {'yes' if state['zephyr'] else 'no'})",
        f"Lead Pokémon: L{state['lead_level']} HP {state['lead_hp']}/{state['lead_max_hp']}",
        f"Trainers beaten — route: {state['route_trainers_beaten']}, gym: {state['gym_trainers_beaten']}",
    ]
    if state["battle_type"] == 0:
        lines.insert(0, "Mode: Overworld")
    else:
        kind = {1: "wild", 2: "trainer", 3: "gym"}.get(state["battle_type"], "unknown")
        lines.insert(0, f"Mode: Battle ({kind})")
        lines.append(
            f"Enemy: L{state['enemy_lead_level']} HP {state['enemy_hp']}/{state['enemy_max_hp']}"
        )
    return "\n".join(lines)