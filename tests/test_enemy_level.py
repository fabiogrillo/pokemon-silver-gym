"""
Verifica enemy_lead_level (0xD0FC) e enemy_hp_ratio tramite RAMReader.
Carica da violet_city_gym.state. Gioca tu nella finestra SDL2 ed entra in battaglia.

Uso:
    python tests/test_enemy_level.py

Output: terminale + tests/debug.log (entrambi aggiornati in tempo reale).

Cosa aspettarsi:
  - Fuori battaglia:  battle=0  enemy_lv=0  enemy_hp%=0.00
  - In battaglia:     battle=2  enemy_lv=7  enemy_hp%=1.00 (scende man mano che fai danno)
  - enemy_lv cambia quando il trainer manda fuori il secondo Pokemon
"""
from pyboy import PyBoy
from env.ram_reader import RAMReader

ROM_PATH   = "pokemon_rom.gbc"
STATE_PATH = "saves/violet_city_gym.state"
LOG_PATH   = "tests/debug.log"

pyboy      = PyBoy(ROM_PATH, window="SDL2", sound=False)
with open(STATE_PATH, "rb") as f:
    pyboy.load_state(f)

ram_reader = RAMReader(pyboy)
pyboy.set_emulation_speed(2)

HEADER = f"{'Step':>6}  {'battle':>6}  {'lead_lv':>7}  {'enemy_lv':>8}  {'enemy_hp%':>9}  {'our_hp%':>7}  note"

def emit(line):
    """Stampa su terminale e scrivi su file contemporaneamente."""
    print(line)
    log.write(line + "\n")
    log.flush()

print(f"Pronto. Entra in battaglia. Log → {LOG_PATH}")

step        = 0
prev_battle = 0
prev_enemy  = 0

with open(LOG_PATH, "w") as log:
    emit(HEADER)
    while pyboy.tick(1):
        ram          = ram_reader.read_all()
        battle       = ram["battle_type"]
        lead_lv      = ram["lead_level"]
        enemy_lv     = ram["enemy_lead_level"]
        enemy_hp_pct = ram["enemy_hp_ratio"]
        our_hp_pct   = ram["hp_ratio"]

        changed  = (battle != prev_battle) or (enemy_lv != prev_enemy and battle > 0)
        periodic = battle > 0 and step % 60 == 0

        if changed or periodic:
            note = ""
            if battle > 0 and prev_battle == 0: note = "<<< BATTLE START"
            if battle == 0 and prev_battle > 0: note = "<<< BATTLE END"
            if enemy_lv != prev_enemy and battle > 0: note += " (enemy_lv changed)"
            emit(f"{step:>6}  {battle:>6}  {lead_lv:>7}  {enemy_lv:>8}  {enemy_hp_pct:>9.2f}  {our_hp_pct:>7.2f}  {note}")

        prev_battle = battle
        prev_enemy  = enemy_lv
        step       += 1

pyboy.stop()
emit("Done.")
