"""
Flag monitor starting from totodile.state.

Tracks:
  0xD88E bit 6  — rival flag (expected: falls 1→0 when rival battle ends)
  0xD7BA bit 6  — Mr. Pokemon egg pickup (expected: rises 0→1 when egg received)
  0xD7BA bit 7  — Elm delivery (expected: rises 0→1 when egg delivered)

Output:
  - Status line every 2 seconds of gameplay (current raw values)
  - Full snapshot at every battle start and end (auto-detected via 0xD116)
  - Immediate alert whenever any tracked flag byte changes

Usage:
    python tests/test_event_detection.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboy import PyBoy
from env.ram_reader import RAMReader

STATE_PATH   = "saves/start.state"
STATUS_EVERY = 120   # ticks between periodic status lines (~2 sec at 60 fps)
SAMPLE_EVERY = 10    # read RAM every N ticks (reduces CPU usage)
LOG_PATH     = "runs/event_detection.log"

os.makedirs("runs", exist_ok=True)
_logfile = open(LOG_PATH, "w", buffering=1)  # line-buffered: writes are immediate

def log(msg=""):
    print(msg)
    _logfile.write(msg + "\n")

pyboy = PyBoy("pokemon_rom.gbc", window="SDL2", sound=False)
with open(STATE_PATH, "rb") as f:
    pyboy.load_state(f)
pyboy.set_emulation_speed(2)

reader = RAMReader(pyboy)


def fmt_flags(ram):
    rival = ram["flag_rival_cherrygrove"]
    elm   = ram["flag_elm_mr_pokemon"]
    return (
        f"  0xD88E = {rival:#04x}  ({rival:08b})  "
        f"bit6(rival)  = {'1  ← rival NOT yet beaten' if rival & 0x40 else '0  ← rival beaten'}\n"
        f"  0xD7BA = {elm:#04x}  ({elm:08b})  "
        f"bit6(egg_pickup) = {'1' if elm & 0x40 else '0'}  "
        f"bit7(elm_delivery) = {'1' if elm & 0x80 else '0'}"
    )


ram = reader.read_all()
log("\n" + "="*65)
log(f"Loaded: {STATE_PATH}")
log("Initial flag state:")
log(fmt_flags(ram))
log("="*65)
log("Play the game normally. Status prints every ~2 sec.")
log("Automatic snapshots fire at every battle start/end.\n")

prev_rival  = ram["flag_rival_cherrygrove"]
prev_elm    = ram["flag_elm_mr_pokemon"]
prev_battle = ram["battle_type"]
total_reward = 0.0
tick = 0

while pyboy.tick(1):
    tick += 1

    if tick % SAMPLE_EVERY != 0:
        continue

    ram    = reader.read_all()
    rival  = ram["flag_rival_cherrygrove"]
    elm    = ram["flag_elm_mr_pokemon"]
    battle = ram["battle_type"]

    # ── Periodic status line ─────────────────────────────────────────────────
    if tick % STATUS_EVERY == 0:
        log(
            f"[t={tick:8d}]  map=({ram['map_bank']:3d},{ram['map_number']:3d})  "
            f"0xD88E={rival:#04x}({rival:08b}) b6={'1' if rival & 0x40 else '0'}  |  "
            f"0xD7BA={elm:#04x}({elm:08b}) b6={'1' if elm & 0x40 else '0'} b7={'1' if elm & 0x80 else '0'}  |  "
            f"battle={battle}"
        )

    # ── Battle start snapshot ────────────────────────────────────────────────
    if battle > 0 and prev_battle == 0:
        log(f"\n{'>'*10} BATTLE STARTED (type={battle})  tick={tick} {'<'*10}")
        log(fmt_flags(ram))
        log()

    # ── Battle end snapshot ──────────────────────────────────────────────────
    if battle == 0 and prev_battle > 0:
        log(f"\n{'>'*10} BATTLE ENDED  tick={tick} {'<'*10}")
        log(fmt_flags(ram))
        log()

    # ── Rival flag change alert ──────────────────────────────────────────────
    if rival != prev_rival:
        log(f"\n[t={tick:8d}]  *** 0xD88E CHANGED ***")
        log(f"  before: {prev_rival:#04x} ({prev_rival:08b})")
        log(f"  after:  {rival:#04x} ({rival:08b})")
        if not (rival & 0x40) and (prev_rival & 0x40):
            log("  bit6 FELL 1→0 — RIVAL EVENT DETECTED → +200 reward")
            total_reward += 200.0
        elif (rival & 0x40) and not (prev_rival & 0x40):
            log("  bit6 ROSE 0→1 — NOTE: bit rose back after falling")

    # ── Mr. Pokemon / Elm flag change alert ──────────────────────────────────
    if elm != prev_elm:
        log(f"\n[t={tick:8d}]  *** 0xD7BA CHANGED ***")
        log(f"  before: {prev_elm:#04x} ({prev_elm:08b})")
        log(f"  after:  {elm:#04x} ({elm:08b})")
        if (elm & 0x40) and not (prev_elm & 0x40):
            log("  bit6 ROSE 0→1 — EGG PICKED UP FROM MR. POKEMON → +100 reward")
            total_reward += 100.0
        if (elm & 0x80) and not (prev_elm & 0x80):
            log("  bit7 ROSE 0→1 — EGG DELIVERED TO ELM → +200 reward")
            total_reward += 200.0

    prev_rival  = rival
    prev_elm    = elm
    prev_battle = battle

pyboy.stop()
log(f"\nSession ended. Total simulated reward: {total_reward:+.1f}")
_logfile.close()
print(f"\nLog completo salvato in: {LOG_PATH}")
