import stable_retro
import pygame
import time

pygame.init()
screen = pygame.display.set_mode((320, 224))
pygame.display.set_caption("Sonic - Special Stage Watch")

env = stable_retro.make("SonicAndKnuckles3-Genesis-v0", render_mode="human")
obs = env.reset()

in_special = False
log = open("special_stage_log.txt", "w")

try:
    while True:
        action = [0] * 12

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: action[0] = 1
        if keys[pygame.K_UP]:    action[4] = 1
        if keys[pygame.K_DOWN]:  action[5] = 1
        if keys[pygame.K_LEFT]:  action[6] = 1
        if keys[pygame.K_RIGHT]: action[7] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        if info['special_stage'] == 1 and not in_special:
            in_special = True
            log.write("=== ENTERED SPECIAL STAGE ===\n")
            log.flush()

        if in_special and info['special_stage'] == 1:
            log.write(f"{info}\n")
            log.flush()

        if info['special_stage'] == 0 and in_special:
            in_special = False
            log.write("=== EXITED SPECIAL STAGE ===\n")
            log.flush()

        if terminated or truncated:
            obs = env.reset()

        time.sleep(0.016)

except KeyboardInterrupt:
    print("Closing...")

log.close()
pygame.quit()
env.close()
