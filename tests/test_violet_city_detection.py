"""
Violet City event detection test.

Tracks:
  0xD57C bit 0   — Zephyr Badge (rises 0→1 when Falkner beaten)
  0xDA00/0xDA01  — map_bank / map_number (printed on every map transition)
  0xD7B7–0xD8B6  — full event flag block (alerts on any byte change)

Also keeps the existing rival / Mr.Pokemon / Elm tracking from the
previous detection test, so everything is visible in one log.

Usage:
    python tests/test_violet_city_detection.py
    python tests/test_violet_city_detection.py saves/mid_route30.state
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboy import PyBoy
from env.ram_reader import RAMReader

STATE_PATH   = sys.argv[1] if len(sys.argv) > 1 else "saves/mid_route30.state"
STATUS_EVERY = 120    # ticks between periodic status lines (~2 sec at 60 fps)
SAMPLE_EVERY = 10     # read RAM every N ticks
LOG_PATH     = "runs/violet_city_detection.log"

# ── Event flag block: 0xD7B7 – 0xD8B6 (256 bytes) ───────────────────────────
FLAG_START = 0xD7B7
FLAG_END   = 0xD8B6  # inclusive

os.makedirs("runs", exist_ok=True)
_logfile = open(LOG_PATH, "w", buffering=1)

def log(msg=""):
    print(msg)
    _logfile.write(msg + "\n")

pyboy = PyBoy("pokemon_rom.gbc", window="SDL2", sound=False)
with open(STATE_PATH, "rb") as f:
    pyboy.load_state(f)
pyboy.set_emulation_speed(5)

reader = RAMReader(pyboy)


def read_flag_block():
    return bytes(pyboy.memory[addr] for addr in range(FLAG_START, FLAG_END + 1))


def fmt_map(bank, number, x, y):
    return f"map=({bank},{number}) local=({x},{y})"


ram = reader.read_all()
log("=" * 70)
log(f"Loaded: {STATE_PATH}")
log(f"Start: {fmt_map(ram['map_bank'], ram['map_number'], ram['local_x'], ram['local_y'])}")
log(f"Badges byte: {ram['badges']:#04x} ({ram['badges']:08b})  zephyr={'YES' if ram['zephyr'] else 'no'}")
log("=" * 70)
log("Play normally. Status every ~2 sec. Alerts fire on flag changes.")
log("MAP TRANSITION events are printed in full on every map change.\n")

prev_map_bank   = ram["map_bank"]
prev_map_number = ram["map_number"]
prev_badges     = ram["badges"]
prev_rival      = ram["flag_rival_cherrygrove"]
prev_elm        = ram["flag_elm_mr_pokemon"]
prev_flag_block = read_flag_block()

total_reward = 0.0
tick = 0

while pyboy.tick(1):
    tick += 1

    if tick % SAMPLE_EVERY != 0:
        continue

    ram   = reader.read_all()
    bank  = ram["map_bank"]
    num   = ram["map_number"]
    lx    = ram["local_x"]
    ly    = ram["local_y"]
    bdg   = ram["badges"]
    rival = ram["flag_rival_cherrygrove"]
    elm   = ram["flag_elm_mr_pokemon"]

    # ── Periodic status ──────────────────────────────────────────────────────
    if tick % STATUS_EVERY == 0:
        log(
            f"[t={tick:8d}]  {fmt_map(bank, num, lx, ly)}"
            f"  badges={bdg:#04x}({bdg:08b})"
            f"  zephyr={'YES' if ram['zephyr'] else 'no'}"
            f"  rival=0xD88E({rival:08b})"
            f"  elm=0xD7BA({elm:08b})"
        )

    # ── Map transition ───────────────────────────────────────────────────────
    if bank != prev_map_bank or num != prev_map_number:
        log(f"\n{'#'*10} MAP TRANSITION  tick={tick} {'#'*10}")
        log(f"  prev: ({prev_map_bank},{prev_map_number})")
        log(f"  now:  ({bank},{num})  local=({lx},{ly})")
        log()

    # ── Badge change ─────────────────────────────────────────────────────────
    if bdg != prev_badges:
        log(f"\n{'!'*10} BADGE BYTE CHANGED  tick={tick} {'!'*10}")
        log(f"  before: {prev_badges:#04x} ({prev_badges:08b})")
        log(f"  after:  {bdg:#04x} ({bdg:08b})")
        if (bdg & 0x01) and not (prev_badges & 0x01):
            log("  bit0 ROSE 0→1 — ZEPHYR BADGE OBTAINED — Falkner beaten! → +1000 reward")
            log(f"  Map at badge time: ({bank},{num}) local=({lx},{ly})")
            total_reward += 1000.0
        log()

    # ── Rival flag change ────────────────────────────────────────────────────
    if rival != prev_rival:
        log(f"\n[t={tick:8d}]  *** 0xD88E CHANGED ***")
        log(f"  before: {prev_rival:#04x} ({prev_rival:08b})")
        log(f"  after:  {rival:#04x} ({rival:08b})")
        if not (rival & 0x40) and (prev_rival & 0x40):
            log("  bit6 FELL 1→0 — RIVAL beaten → +200 reward")
            total_reward += 200.0

    # ── Elm/Mr.Pokemon flag change ───────────────────────────────────────────
    if elm != prev_elm:
        log(f"\n[t={tick:8d}]  *** 0xD7BA CHANGED ***")
        log(f"  before: {prev_elm:#04x} ({prev_elm:08b})")
        log(f"  after:  {elm:#04x} ({elm:08b})")
        if (elm & 0x40) and not (prev_elm & 0x40):
            log("  bit6 ROSE 0→1 — EGG PICKED UP from Mr.Pokemon → +100 reward")
            total_reward += 100.0
        if (elm & 0x80) and not (prev_elm & 0x80):
            log("  bit7 ROSE 0→1 — EGG DELIVERED to Elm → +200 reward")
            total_reward += 200.0

    # ── Full event flag block scan ───────────────────────────────────────────
    curr_flag_block = read_flag_block()
    if curr_flag_block != prev_flag_block:
        changed_bytes = [
            i for i in range(len(curr_flag_block))
            if curr_flag_block[i] != prev_flag_block[i]
        ]
        log(f"\n[t={tick:8d}]  EVENT FLAG BLOCK CHANGED ({len(changed_bytes)} byte(s))")
        log(f"  current map: ({bank},{num}) local=({lx},{ly})")
        for i in changed_bytes:
            addr = FLAG_START + i
            old  = prev_flag_block[i]
            new  = curr_flag_block[i]
            diff = old ^ new
            log(f"  0x{addr:04X}: {old:#04x}({old:08b}) → {new:#04x}({new:08b})  changed_bits={diff:08b}")
        log()
        prev_flag_block = curr_flag_block

    prev_map_bank   = bank
    prev_map_number = num
    prev_badges     = bdg
    prev_rival      = rival
    prev_elm        = elm

pyboy.stop()
log(f"\nSession ended. Total simulated reward: {total_reward:+.1f}")
_logfile.close()
print(f"\nLog saved to: {LOG_PATH}")
