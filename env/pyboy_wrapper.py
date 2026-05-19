from pyboy import PyBoy
from .actions import ACTIONS
import imageio.v3 as iio
import numpy as np

class PyBoyWrapper:
    def __init__(self, rom_path, state_path="../saves/totodile.state", headless=True):
        """
        Initialize the PyBoy emulator and RAM reader.
        """
        self.state_path = state_path
        self.pyboy = PyBoy(rom_path, window="null" if headless else "SDL2", sound=False)

        # Load state in state_path if it exists, otherwise start a new game and save the initial state.
        try:
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)
            print(f"Loaded state from {state_path}")
        except Exception as e:
            print(f"Failed to load state from {state_path}: {e}")
            print("Insert a valid path for the save file.")

    def step(self, action, n=24):
        """
        Take and int action and converts it to a button press in PyBoy, 
        then advance the emulator by one frame.
        """
        self.pyboy.button_press(ACTIONS[action])
        self.pyboy.tick(count=n)
        self.pyboy.button_release(ACTIONS[action])

        return self.pyboy.screen.ndarray

    def reset(self):
        """
        Reset the game at initial state for new episode.
        Load the initial state and stabilize the emuletor by ticking
        and returns the screen array.
        """
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
            
        self.pyboy.tick(count=120)  # Stabilize the emulator for 2 seconds at 60 FPS

        return self.pyboy.screen.ndarray

    def capture_gif(self, save_path, frames, fps=12):
        """
        Takes a list of ndarrays in one episode and saves them
        as a gif with ImageIO.
        """
        rgb_frames = [f[:,:,:3] for f in frames]  # Drop alpha channel if present


        iio.imwrite(
            save_path,
            np.stack(rgb_frames),
            plugin="pillow",
            loop=0,
            duration=1000//fps
        )  # duration in ms per frame

        return
    
