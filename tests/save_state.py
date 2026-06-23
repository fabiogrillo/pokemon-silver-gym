"""
Run the game at 3x speed in an SDL2 window. Prints the current map (bank, number)
and local coords on EVERY map change, so you always know where you are while walking.
Close the window with X to save the state AT THE CURRENT POSITION and exit
(there is no "save" key — closing the window saves).

Usage:
    python tests/save_state.py <name>           (load from start.state)
    python tests/save_state.py <name> <load>    (load from saves/<load>.state)

Examples:
    python tests/save_state.py mid_route30
    python tests/save_state.py egg_delivered before_elm_delivery
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboy import PyBoy
from env.ram_reader import RAMReader

SAVE_NAME = sys.argv[1] if len(sys.argv) > 1 else "new"
LOAD_NAME = sys.argv[2] if len(sys.argv) > 2 else "start"
LOAD_PATH = f"saves/{LOAD_NAME}.state"
SAVE_PATH = f"saves/{SAVE_NAME}.state"

SAMPLE_EVERY = 10    # read RAM every N ticks
STATUS_EVERY = 180   # ~1s at 3x: periodically reprint position even without a map change

pyboy = PyBoy("pokemon_rom.gbc", window="SDL2", sound=False)
with open(LOAD_PATH, "rb") as f:
    pyboy.load_state(f)

pyboy.set_emulation_speed(3)
reader = RAMReader(pyboy)

ram = reader.read_all()
prev_bank, prev_num = ram["map_bank"], ram["map_number"]
print(f"Game started at 3x. Loaded from '{LOAD_PATH}'.")
print(f"Initial position: map=({prev_bank},{prev_num}) local=({ram['local_x']},{ram['local_y']})")
print(f"Close the window with X to save to '{SAVE_PATH}' (CURRENT position).\n")

tick = 0
# pyboy.tick(1) returns False when the user closes the window with X
while pyboy.tick(1):
    tick += 1
    if tick % SAMPLE_EVERY != 0:
        continue

    ram = reader.read_all()
    bank, num = ram["map_bank"], ram["map_number"]
    lx, ly = ram["local_x"], ram["local_y"]

    if bank != prev_bank or num != prev_num:
        print(f"MAP CHANGED → ({bank},{num})  local=({lx},{ly})   [was: ({prev_bank},{prev_num})]")
        prev_bank, prev_num = bank, num
    elif tick % STATUS_EVERY == 0:
        print(f"   ...in map=({bank},{num}) local=({lx},{ly})")

os.makedirs("saves", exist_ok=True)
with open(SAVE_PATH, "wb") as f:
    pyboy.save_state(f)

final = reader.read_all()
pyboy.stop()
print(f"\nState saved to: {SAVE_PATH}")
print(f"Saved position: map=({final['map_bank']},{final['map_number']}) "
      f"local=({final['local_x']},{final['local_y']})")
