#!/usr/bin/env python3
"""
trace_reload.py -- frame-by-frame RAM trace through a death/reload.

Instead of diffing two far-apart snapshots (which drowns the trigger in the
object-table rebuild), this captures RAM EVERY frame once you press 1, for
~180 frames, then reports for each changed control-region byte the FIRST frame
it changed. The reload TRIGGER changes early and briefly; the object churn
changes later. Sort by first-change-frame to surface the trigger.

Controls: play, press 1 to start a ~180-frame trace, then trigger the death.
The trace auto-stops and prints. R=reset Q=quit.
"""
import pygame, stable_retro
import numpy as np

GAME='SonicAndKnuckles3-Genesis-v0'; SCALE=3
HELD={pygame.K_z:0,pygame.K_a:1,pygame.K_RETURN:3,pygame.K_UP:4,
      pygame.K_DOWN:5,pygame.K_LEFT:6,pygame.K_RIGHT:7,pygame.K_x:8}

def ram(env): return np.frombuffer(bytes(env.get_ram()),dtype=np.uint8).copy()

def main():
    pygame.init()
    screen=pygame.display.set_mode((320*SCALE,224*SCALE))
    pygame.display.set_caption("trace_reload: 1=start trace then die  R=reset Q=quit")
    font=pygame.font.SysFont('monospace',18,bold=True); clock=pygame.time.Clock()
    env=stable_retro.make(GAME,use_restricted_actions=stable_retro.Actions.FILTERED,
                          render_mode='rgb_array'); env.reset()
    tracing=False; frames=[]; base=None; msg="Play. Press 1, THEN die within ~3s."
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_q: running=False
                elif e.key==pygame.K_r: env.reset(); tracing=False; frames=[]; msg="reset"
                elif e.key==pygame.K_1:
                    base=ram(env); frames=[]; tracing=True
                    msg="TRACING ~180 frames... trigger the death NOW"; print(msg)
        keys=pygame.key.get_pressed(); buttons=[0]*12
        for k,idx in HELD.items():
            if keys[k]: buttons[idx]=1
        env.step(buttons)
        if tracing:
            frames.append(ram(env))
            if len(frames)>=180:
                tracing=False
                # first-change frame per control-region byte
                first={}
                for fi,fr in enumerate(frames):
                    ch=np.nonzero(fr!=base)[0]
                    for i in ch:
                        if 0xF000<=i<=0xFFFF and i not in first:
                            first[i]=(fi,int(base[i]),int(fr[i]))
                # sort by earliest change; show the first movers
                items=sorted(first.items(), key=lambda kv: kv[1][0])
                print(f"\n{len(items)} control bytes changed. EARLIEST movers (likely trigger):")
                for addr,(fi,b,a) in items[:40]:
                    print(f"  frame {fi:3d}  ${addr:04X}: {b:3d} -> {a:3d}")
                msg=f"trace done: {len(items)} bytes, earliest at frame {items[0][1][0] if items else '-'}"
                print(msg)
        raw=env.render()
        if raw is not None and not isinstance(raw,bool):
            surf=pygame.surfarray.make_surface(raw.swapaxes(0,1))
            screen.blit(pygame.transform.scale(surf,(320*SCALE,224*SCALE)),(0,0))
            try:
                z=env.data.lookup_value('zone'); a=env.data.lookup_value('act')
                screen.blit(font.render(f"zone:{z} act:{a}",True,(255,255,0)),(8,8))
            except Exception: pass
            screen.blit(font.render(msg[:70],True,(0,255,128)),(8,28))
        pygame.display.flip(); clock.tick(60)
    pygame.quit(); env.close()

if __name__=='__main__': main()
