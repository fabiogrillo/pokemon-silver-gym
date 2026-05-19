import gymnasium

# Ordered list of GBC button names — index = action integer output by the RL agent.
# pyboy_instance.button(ACTIONS[action_int]) is how this gets used in pyboy_wrapper.py.
ACTIONS = ["up", "down", "left", "right", "a", "b", "start", "select"]

ACTION_SPACE = gymnasium.spaces.Discrete(len(ACTIONS))