import stable_retro
import pygame
import time

pygame.init()
screen = pygame.display.set_mode((320, 224))
pygame.display.set_caption("Sonic - Keyboard Control")

env = stable_retro.make("SonicAndKnuckles3-Genesis-v0", render_mode="human")
obs = env.reset()

prev_special_stage = 0

try:
    while True:
        # [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]
        action = [0] * 12

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: action[0] = 1   # B - jump
        if keys[pygame.K_UP]:    action[4] = 1   # UP
        if keys[pygame.K_DOWN]:  action[5] = 1   # DOWN
        if keys[pygame.K_LEFT]:  action[6] = 1   # LEFT
        if keys[pygame.K_RIGHT]: action[7] = 1   # RIGHT

        obs, reward, terminated, truncated, info = env.step(action)

        if info['special_stage'] != prev_special_stage:
            print(f"special_stage changed: {prev_special_stage} -> {info['special_stage']}")
            print(f"Full info: {info}")
            prev_special_stage = info['special_stage']

        if terminated or truncated:
            obs = env.reset()
            prev_special_stage = 0

        time.sleep(0.016)

except KeyboardInterrupt:
    print("Closing...")

pygame.quit()
env.close()
