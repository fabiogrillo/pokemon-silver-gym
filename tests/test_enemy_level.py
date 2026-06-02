"""
Monitor completo: party levels, enemy stats, e rilevamento di TUTTI i cambiamenti
nei flag di evento (0xD7B7–0xD8B6).

Usa la finestra SDL2 per giocare tu stesso: ogni volta che un flag cambia,
viene stampato a terminale E loggato in tests/debug.log con:
  - indirizzo RAM esatto
  - numero del flag
  - direzione (RISE/FALL)

Questo è il modo più affidabile per scoprire quali indirizzi corrispondono
a quali eventi in-game.

Uso:
    python tests/test_enemy_level.py [state_name]

    state_name: nome del file di salvataggio senza estensione .state
                Default: violet_city_gym (per testare battle/enemy stats)
                Usa "start" per testare i flag di evento dall'inizio

Esempi:
    python tests/test_enemy_level.py                       # carica violet_city_gym.state
    python tests/test_enemy_level.py start                 # carica start.state (flag events)
    python tests/test_enemy_level.py before_elm_delivery   # carica before_elm_delivery.state

Cosa verificare per i party levels:
  - Da start.state: party_levels = [5, 0, 0, 0, 0, 0]  (Totodile lv5, slot 2-6 vuoti)
  - Se catturi un Pokemon: party_levels = [5, X, 0, 0, 0, 0]  (slot 2 si popola)
  - Se i valori sono casuali/sbagliati → struct size 0x30 non è corretto

Cosa verificare per i flag di evento:
  - Entra in battaglia → vedi flag cambiare (battle_type/battle_flags)
  - Batti il rivale → vedi 0xD88E bit6 FALL (già noto: flag #1726)
  - Ricevi uovo da Mr.Pokemon → vedi 0xD7BA bit6 RISE (flag #30)
  - Consegna uovo a Elm → vedi 0xD7BA bit7 RISE (flag #31)
  - Batti un trainer in palestra → NUOVO flag da scoprire!
  - Ricevi running shoes / Pokédex → NUOVO flag da scoprire!
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
