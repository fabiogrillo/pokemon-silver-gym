"""
Load a state, take a PRE-event snapshot, watch for the battle to end, take a POST-event
snapshot automatically, then print the diff.

Usage:
    python tests/ram_scan.py checkpoint
"""
import sys
import numpy as np
from pyboy import PyBoy

SAVE_NAME = sys.argv[1] if len(sys.argv) > 1 else "checkpoint"
LOAD_PATH = f"saves/{SAVE_NAME}.state"

RAM_START = 0xC000
RAM_END   = 0xE000

def snapshot(pb):
    return np.array([pb.memory[a] for a in range(RAM_START, RAM_END)], dtype=np.uint8)

def diff(before, after, addr_filter=None, top_n=60):
    changed = np.where(before != after)[0]
    if addr_filter:
        lo, hi = addr_filter
        changed = changed[(changed + RAM_START >= lo) & (changed + RAM_START <= hi)]
    print(f"\n{len(changed)} bytes changed" + (f" in 0x{addr_filter[0]:04X}-0x{addr_filter[1]:04X}" if addr_filter else "") + ":")
    for idx in changed[:top_n]:
        addr = RAM_START + idx
        b, a = before[idx], after[idx]
        new_bits = a & ~b & 0xFF
        print(f"  0x{addr:04X}:  {b:3d} (0b{b:08b})  ->  {a:3d} (0b{a:08b})"
              + (f"  [new bits: 0b{new_bits:08b}]" if new_bits else ""))
    if len(changed) > top_n:
        print(f"  ... and {len(changed) - top_n} more")

pyboy = PyBoy("pokemon_rom.gbc", window="SDL2", sound=False)
with open(LOAD_PATH, "rb") as f:
    pyboy.load_state(f)
pyboy.set_emulation_speed(3)

snap_before = snapshot(pyboy)
print(f"State '{LOAD_PATH}' loaded. PRE snapshot taken.")
print("Fight the battle. The diff fires automatically when it ends.")

in_battle  = False
snap_after = None

while pyboy.tick(1):
    battle_type = pyboy.memory[0xD116]
    if battle_type > 0:
        in_battle = True
    if in_battle and battle_type == 0 and snap_after is None:
        snap_after = snapshot(pyboy)
        print("\nBattle ended! POST snapshot taken. You can close the window with X.")
        # keep running so the user sees the result in-game

if snap_after is None:
    snap_after = snapshot(pyboy)

pyboy.stop()

print("\n" + "="*60)
print("FULL DIFF (all RAM regions):")
diff(snap_before, snap_after)

print("\n" + "="*60)
print("FILTERED DIFF — event flags region (0xD7B7-0xD8B6):")
diff(snap_before, snap_after, addr_filter=(0xD7B7, 0xD8B6))
