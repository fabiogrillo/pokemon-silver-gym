"""
Full monitor: party levels, enemy stats, and detection of ALL event-flag changes
(0xD7B7–0xD8B6). Play in the SDL2 window yourself; every flag change is printed and
logged to tests/debug.log with the exact RAM address, flag number, and direction
(RISE/FALL). This is the most reliable way to discover which addresses map to which
in-game events.

Usage:
    python tests/test_enemy_level.py [state_name]
    state_name: save file without the .state extension.
                Default: violet_city_gym (to test battle/enemy stats).
                Use "start" to test event flags from the beginning.

Examples:
    python tests/test_enemy_level.py                       # violet_city_gym.state
    python tests/test_enemy_level.py start                 # start.state (flag events)
    python tests/test_enemy_level.py before_elm_delivery   # before_elm_delivery.state

Party-level checks:
  - From start.state: party_levels = [5, 0, 0, 0, 0, 0]  (Totodile lv5, slots 2-6 empty)
  - After catching one: party_levels = [5, X, 0, 0, 0, 0]  (slot 2 populates)
  - Random/wrong values → struct size 0x30 is incorrect

Event-flag checks:
  - Enter battle → flags change (battle_type/battle_flags)
  - Beat the rival → 0xD88E bit6 FALL (known: flag #1726)
  - Receive egg from Mr. Pokemon → 0xD7BA bit6 RISE (flag #30)
  - Deliver egg to Elm → 0xD7BA bit7 RISE (flag #31)
  - Beat a gym trainer → NEW flag to discover
  - Receive Pokédex → NEW flag to discover
"""
import sys
from pyboy import PyBoy
from env.ram_reader import RAMReader

ROM_PATH   = "pokemon_rom.gbc"
STATE_NAME = sys.argv[1] if len(sys.argv) > 1 else "violet_city_gym"
STATE_PATH = f"saves/{STATE_NAME}.state"
LOG_PATH   = "tests/debug.log"

# ── Event flag memory range
FLAG_BASE = 0xD7B7  # Start of wEventFlags area
FLAG_LEN  = 256     # 256 bytes = 2048 possible event flags (0xD7B7–0xD8B6)

pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)
with open(STATE_PATH, "rb") as f:
    pyboy.load_state(f)

ram_reader = RAMReader(pyboy)
pyboy.set_emulation_speed(2)

def emit(line):
    """Print to terminal and write to log file simultaneously."""
    print(line)
    log.write(line + "\n")
    log.flush()

print(f"Loaded: {STATE_PATH}")
print(f"Play in the SDL2 window. Log → {LOG_PATH}")
print()

HEADER = (
    f"{'Step':>6}  {'map(b,n)':>9}  {'btl':>3}  "
    f"{'lead':>4}  {'enemy':>5}  {'e_hp%':>5}  {'hp%':>5}  "
    f"{'party_levels':^29}  note"
)
SEP = "-" * 110

step        = 0
prev_battle = 0
prev_enemy  = 0

with open(LOG_PATH, "w") as log:
    emit(f"=== Session: {STATE_PATH} ===")
    emit(HEADER)
    emit(SEP)

    # Snapshot initial event flags
    prev_flags = [pyboy.memory[FLAG_BASE + i] for i in range(FLAG_LEN)]

    while pyboy.tick(1):
        ram     = ram_reader.read_all()
        battle  = ram["battle_type"]
        lead_lv = ram["lead_level"]
        enemy_lv     = ram["enemy_lead_level"]
        enemy_hp_pct = ram["enemy_hp_ratio"]
        our_hp_pct   = ram["hp_ratio"]
        party_lvs    = ram["party_levels"]  # list[6]
        cur_map      = (ram["map_bank"], ram["map_number"])

        # ── Event flag change detection: compare each byte to previous frame
        curr_flags = [pyboy.memory[FLAG_BASE + i] for i in range(FLAG_LEN)]
        for i, (old, new) in enumerate(zip(prev_flags, curr_flags)):
            if old != new:
                addr        = FLAG_BASE + i
                changed_bits = old ^ new
                for bit in range(8):
                    if changed_bits & (1 << bit):
                        flag_num  = i * 8 + bit
                        direction = "RISE" if (new & (1 << bit)) else "FALL"
                        emit(
                            f"  *** FLAG {direction}: "
                            f"addr=0x{addr:04X} bit={bit} flag=#{flag_num:4d} | "
                            f"0b{old:08b} → 0b{new:08b} | "
                            f"step={step} map={cur_map}"
                        )
        prev_flags = curr_flags

        # ── Periodic battle/enemy status (only when state changes or every 60 steps in battle)
        changed  = (battle != prev_battle) or (enemy_lv != prev_enemy and battle > 0)
        periodic = (battle > 0) and (step % 60 == 0)

        if changed or periodic:
            note = ""
            if battle > 0 and prev_battle == 0:            note = "<<< BATTLE START"
            if battle == 0 and prev_battle > 0:            note = "<<< BATTLE END"
            if enemy_lv != prev_enemy and battle > 0:      note += " (enemy_lv changed)"

            lvs_str = " ".join(f"{lv:3d}" for lv in party_lvs)
            map_str = f"({cur_map[0]:2d},{cur_map[1]:2d})"
            emit(
                f"{step:>6}  {map_str:>9}  {battle:>3}  "
                f"{lead_lv:>4}  {enemy_lv:>5}  {enemy_hp_pct:>5.2f}  {our_hp_pct:>5.2f}  "
                f"[{lvs_str}]  {note}"
            )

        prev_battle = battle
        prev_enemy  = enemy_lv
        step       += 1

pyboy.stop()
emit(SEP)
emit("Done.")
