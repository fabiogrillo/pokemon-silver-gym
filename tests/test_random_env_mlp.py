from env.pokemon_env_mlp import PokemonEnvMLP

ROM_PATH = "pokemon_rom.gbc"
STATE_PATH = "saves/violet_city_gym.state"

env = PokemonEnvMLP(ROM_PATH, STATE_PATH, headless=False)
obs, info = env.reset()
print(f"Reset OK - obs shape: {obs.shape}")

total_reward = 0.0
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 250 == 0:
        ram = env.ram_reader.read_all()
        print(f"Step {step:4d} | x={ram['local_x']:3d} y={ram['local_y']:3d} | Reward: {reward:+.2f} | Total: {total_reward:+.2f} | battle={ram['battle_type']} | Terminated: {terminated} | Lead level: {ram['lead_level']} | HP ratio: {ram['hp_ratio']:.2f} | Enemy lead level: {ram['enemy_lead_level']}")

    if terminated or truncated:
        print(f"Episode ended at step {step} with total reward {total_reward:.2f}")
        obs, info = env.reset()
        total_reward = 0.0

env.close()
print("Test completed successfully.")
