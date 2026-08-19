import stable_retro
import time

env = stable_retro.make("SonicAndKnuckles3-Genesis-v0", render_mode="human")
obs = env.reset()

try:
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.016)
        if terminated or truncated:
            obs = env.reset()
except KeyboardInterrupt:
    print("Closing...")

env.close()
