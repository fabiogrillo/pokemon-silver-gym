"""
Lancia il gioco a 3x velocità con finestra SDL2.
Chiudi la finestra con il tasto X per salvare lo state e uscire.

Uso:
    python tests/save_state.py before_rival
    python tests/save_state.py before_elm
    python tests/save_state.py before_sprout2
"""
import sys
import os
from pyboy import PyBoy

SAVE_NAME = sys.argv[1] if len(sys.argv) > 1 else "checkpoint"
LOAD_PATH = "saves/before_elm.state"
SAVE_PATH = f"saves/{SAVE_NAME}.state"

pyboy = PyBoy("pokemon_rom.gbc", window="SDL2", sound=False)
with open(LOAD_PATH, "rb") as f:
    pyboy.load_state(f)

pyboy.set_emulation_speed(3)
print(f"Gioco avviato a 3x. Chiudi la finestra con X per salvare in '{SAVE_PATH}'.")

# tick() restituisce False quando l'utente chiude la finestra con X
while pyboy.tick(1):
    pass

os.makedirs("saves", exist_ok=True)
with open(SAVE_PATH, "wb") as f:
    pyboy.save_state(f)
pyboy.stop()
print(f"State salvato in: {SAVE_PATH}")
