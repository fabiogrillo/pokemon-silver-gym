from collections import deque


class ShortTermMemory:
    def __init__(self, window: int, stuck_window: int, stuck_radius: int):
        self.window = window
        self.stuck_window = stuck_window
        self.stuck_radius = stuck_radius
        self.history = deque(maxlen=window)
        self.positions = deque(maxlen=stuck_window)

    def record(self, state, thought, tool_name, args, note):
        self.history.append((tool_name, args, note))
        self.positions.append(
            (state["map_bank"], state["map_number"], state["local_x"], state["local_y"])
        )

    def is_stuck(self) -> bool:
        if len(self.positions) < self.stuck_window:
            return False
        maps = {(mb, mn) for mb, mn, _, _ in self.positions}
        if len(maps) > 1:
            return False
        xs = [x for _, _, x, _ in self.positions]
        ys = [y for _, _, _, y in self.positions]
        return (max(xs) - min(xs) <= self.stuck_radius
                and max(ys) - min(ys) <= self.stuck_radius)

    def render_note(self) -> str:
        recent = ", ".join(
            f"{name}({args.get('direction') or args.get('button') or ''})"
            for name, args, _ in list(self.history)[-self.window:]
        )
        note = f"Recent actions: {recent}" if recent else ""
        if self.is_stuck():
            note += ("\nWARNING: you have not moved for several turns. Follow the 'Walkable "
                     "directions' below: MOVE in one of them (toward the top of the gym). If they "
                     "are NONE, a trainer is blocking you — press 'a' repeatedly to start the battle.")
        return note