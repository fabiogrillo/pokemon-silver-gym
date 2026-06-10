"""
Lancia il gioco a 3x velocità con finestra SDL2.
Stampa la mappa corrente (bank, number) e le coordinate locali a OGNI cambio mappa,
così sai sempre dove sei mentre cammini e decidi dove salvare.
Chiudi la finestra con il tasto X per salvare lo state (nella posizione CORRENTE) e uscire.

NB: non c'è un tasto "salva" — chiudendo la finestra si salva la posizione corrente.

Uso:
    python tests/save_state.py <nome>           (carica da start.state)
    python tests/save_state.py <nome> <load>    (carica da saves/<load>.state)

Esempi:
    python tests/save_state.py mid_route30
    python tests/save_state.py egg_delivered before_elm_delivery
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboy import PyBoy
from env.ram_reader import RAMReader

SAVE_NAME = sys.argv[1] if len(sys.argv) > 1 else "nuovo"
LOAD_NAME = sys.argv[2] if len(sys.argv) > 2 else "start"
LOAD_PATH = f"saves/{LOAD_NAME}.state"
SAVE_PATH = f"saves/{SAVE_NAME}.state"

SAMPLE_EVERY = 10    # leggi la RAM ogni N tick (basta e avanza)
STATUS_EVERY = 180   # ~1 sec a 3x: ogni tanto ristampa la posizione anche senza cambio mappa

pyboy = PyBoy("pokemon_rom.gbc", window="SDL2", sound=False)
with open(LOAD_PATH, "rb") as f:
    pyboy.load_state(f)

pyboy.set_emulation_speed(3)
reader = RAMReader(pyboy)

ram = reader.read_all()
prev_bank, prev_num = ram["map_bank"], ram["map_number"]
print(f"Gioco avviato a 3x. Caricato da '{LOAD_PATH}'.")
print(f"Posizione iniziale: mappa=({prev_bank},{prev_num}) local=({ram['local_x']},{ram['local_y']})")
print(f"Chiudi la finestra con X per salvare in '{SAVE_PATH}' (posizione CORRENTE).\n")

tick = 0
# pyboy.tick(1) restituisce False quando l'utente chiude la finestra con X
while pyboy.tick(1):
    tick += 1
    if tick % SAMPLE_EVERY != 0:
        continue

    ram = reader.read_all()
    bank, num = ram["map_bank"], ram["map_number"]
    lx, ly = ram["local_x"], ram["local_y"]

    if bank != prev_bank or num != prev_num:
        print(f"MAPPA CAMBIATA → ({bank},{num})  local=({lx},{ly})   [prima: ({prev_bank},{prev_num})]")
        prev_bank, prev_num = bank, num
    elif tick % STATUS_EVERY == 0:
        print(f"   ...sei in mappa=({bank},{num}) local=({lx},{ly})")

os.makedirs("saves", exist_ok=True)
with open(SAVE_PATH, "wb") as f:
    pyboy.save_state(f)

final = reader.read_all()
pyboy.stop()
print(f"\nState salvato in: {SAVE_PATH}")
print(f"Posizione salvata: mappa=({final['map_bank']},{final['map_number']}) "
      f"local=({final['local_x']},{final['local_y']})")
