import numpy as np
import pygame
import stable_retro
from coordinator import SonicCoordinator

SCALE = 3

pygame.init()
screen = pygame.display.set_mode((320 * SCALE, 224 * SCALE))
pygame.display.set_caption("Sonic RL Agent")
clock = pygame.time.Clock()

env = SonicCoordinator(ss_model=None, render_mode="rgb_array")
obs, info = env.reset()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    raw_frame = env.base_env.render()

    if raw_frame is not None:
        surface = pygame.surfarray.make_surface(raw_frame.swapaxes(0, 1))
        scaled = pygame.transform.scale(surface, (320 * SCALE, 224 * SCALE))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

    if terminated or truncated:
        obs, info = env.reset()

    clock.tick(60)

pygame.quit()
env.close()
