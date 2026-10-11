"""Make states/doomsday_start.state: Doomsday Zone, as nobody's playthrough reaches it
(Tyler's skipped every giant ring). Replays his full playthrough to the start of DEZ3
(frame 412,127), grants all 14 emeralds (chaos count $FFB0 = 7, super count $FFB1 = 7,
the per-stage array $FFB2-$FFB8 = 3 "super won"), and plays his DEZ3 inputs on: they clear
DEZ3 on his exact frame (418,034, 7:43.38: he had 26 rings, too few to transform), and the
game then loads Doomsday ($0C) instead of the ending (2026-10-11). The state is saved on the
first frame of Doomsday's load; its act DDZ1 starts 77 frames later, Hyper Sonic, 50 rings.
CPU, ~5 min. Record it with:  python3 record_play.py --state states/doomsday_start.state"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(sys.path[0])
os.environ.setdefault('MG_LOG', os.devnull)
import coordinator as C, demos
DEZ3_START = 412127
actions, meta = demos.load_play('demos/play_20261010-202923.npz')
table = meta['action_table']
emu = C.make_retro()
emu.data.set_variable('super_count', {'address': 0xFFFFB1, 'type': '|u1'})
for i in range(7):
    emu.data.set_variable(f'em_arr{i}', {'address': 0xFFFFB2 + i, 'type': '|u1'})
emu.reset(); emu.em.set_state(demos.play_start_state(meta))
for a in actions[:DEZ3_START]:
    C.emu_frame(emu, table[a])
emu.data.set_value('chaos_emeralds', 7)
emu.data.set_value('super_count', 7)
for i in range(7):
    emu.data.set_value(f'em_arr{i}', 3)
split = C.ActSplitter()
for i in range(DEZ3_START, len(actions)):
    before = emu.em.get_state()
    _, info = C.emu_frame(emu, table[actions[i]])
    split.feed(i, info)
    if int(info['zone']) == 0x0C:
        out = os.path.join('states', 'doomsday_start.state')
        open(out, 'wb').write(before if int(info['mg_game_mode']) == C.GM_LEVEL else emu.em.get_state())
        print(f'wrote {out} at frame {i}; DEZ3 cleared at {split.acts[-1]["clear"] if split.acts else None} '
              f'({split.acts[-1]["time"] if split.acts else None})')
        break
else:
    sys.exit('Doomsday never loaded')
