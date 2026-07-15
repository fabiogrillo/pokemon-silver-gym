from pyboy import PyBoy
from .actions import ACTIONS
import io
import imageio.v3 as iio
import numpy as np

class PyBoyWrapper:
    def __init__(self, rom_path, state_path="../saves/totodile.state", headless=True):
        self.state_path = state_path
        self.pyboy = PyBoy(rom_path, window="null" if headless else "SDL2", sound=False)
        # Load the save state; if it is missing the game starts fresh.
        try:
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)
            self._release_all_buttons()
            print(f"Loaded state from {state_path}")
        except Exception as e:
            print(f"Failed to load state from {state_path}: {e}")
            print("Insert a valid path for the save file.")

    def _release_all_buttons(self):
        """Clear the joypad after loading a state. Save states serialize the live joypad, and
        frontier states are harvested mid-press, so a loaded state can carry a stuck direction:
        the game then auto-walks on its own and eats subsequent inputs (found 2026-07-15 — a
        Center state walked the player 4 tiles right and soft-locked input until A/B mashing)."""
        for name in ACTIONS:
            self.pyboy.button_release(name)

    def step(self, action, n=24, settle=0):
        """Press the button for `action`, advance `n` frames, release, then idle `settle` frames.

        settle>0 leaves the button released for `settle` frames so consecutive presses produce
        distinct button-down edges, which dialogue/trainer-script advancement requires. Default 0
        keeps the RL caller's timing unchanged.
        """
        self.pyboy.button_press(ACTIONS[action])
        self.pyboy.tick(count=n)
        self.pyboy.button_release(ACTIONS[action])
        if settle:
            self.pyboy.tick(count=settle)
        return self.pyboy.screen.ndarray

    def reset(self):
        """Reload the initial state and stabilize the emulator (120 ticks = 2s @ 60fps)."""
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
        self._release_all_buttons()
        self.pyboy.tick(count=120)
        return self.pyboy.screen.ndarray

    def save_state_bytes(self):
        """Serialize full emulator state to bytes (for the Go-Explore frontier harvest).
        ~200KB/state; load_state round-trips RAM exactly via BytesIO."""
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        return buf.getvalue()

    def reset_from_bytes(self, data):
        """Reset to a frontier save-state from the policy's own trajectory,
        instead of the fixed state file. Same stabilize-tick as reset()."""
        self.pyboy.load_state(io.BytesIO(data))
        self._release_all_buttons()
        self.pyboy.tick(count=120)
        return self.pyboy.screen.ndarray

    def capture_gif(self, save_path, frames, fps=12):
        """Save a list of episode frames (ndarrays) as a GIF via ImageIO."""
        rgb_frames = [f[:, :, :3] for f in frames]  # drop alpha channel if present
        iio.imwrite(save_path, np.stack(rgb_frames), plugin="pillow",
                    loop=0, duration=1000 // fps)  # duration in ms per frame
