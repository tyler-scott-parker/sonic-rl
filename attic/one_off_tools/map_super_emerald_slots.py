"""Which array slot ($FFB2+i) is which pedestal: set slot i to 3 (collected) and the rest to 2
(gray), arrive, and see which emerald object differs. Also: the emerald objects' code/position."""
import os, sys
sys.path.insert(0, '/home/tyler/Projects/sonic-rl'); os.chdir('/home/tyler/Projects/sonic-rl')
os.environ['MG_LOG'] = os.devnull
import numpy as np, coordinator as C
from PIL import Image
SCR = 'attic/one_off_tools'
emu = C.make_retro()
for i in range(7):
    emu.data.set_variable(f'em_arr{i}', {'address': 0xFFFFB2 + i, 'type': '|u1'})
M1 = open('states/handoff/mhz1_ring7_minus1.state', 'rb').read()
def word(a, i): return (int(a[i]) << 8) | int(a[i + 1])
def arrive(vals):
    emu.reset(); emu.em.set_state(M1)
    for i, v in enumerate(vals):
        emu.data.set_value(f'em_arr{i}', v)
    for f in range(160):
        raw, info = C.emu_frame(emu, C.ACTION_BUTTONS[0])
    r = np.frombuffer(bytes(emu.get_ram()), dtype=np.uint8).reshape(-1, 2)[:, ::-1].reshape(-1)
    objs = {}
    for slot in range(0xB000, 0xCFCC, 0x4A):
        code = (word(r, slot) << 16) | word(r, slot + 2)
        if code:
            objs.setdefault(code, []).append((word(r, slot + 0x10), word(r, slot + 0x14), int(r[slot + 0x22]), int(r[slot + 0x2C])))
    return raw, objs
raw, base = arrive([2] * 7)
known = {0x090e1c, 0x090700, 0x018b3e, 0x010a94, 0x01365c, 0x006c2c, 0x090b22, 0x0455cc, 0x090ddc, 0x0160d2, 0x01952a}
print('objects with all gray (new codes vs array=1 arrival):')
for code, lst in base.items():
    if code not in known:
        print(f'  {code:#08x} x{len(lst)}: {sorted(lst)}')
for i in range(7):
    vals = [2] * 7; vals[i] = 3
    raw, objs = arrive(vals)
    diff = {c: sorted(set(objs.get(c, [])) ^ set(base.get(c, []))) for c in set(objs) | set(base)}
    diff = {f'{c:#08x}': d for c, d in diff.items() if d and c not in (0x010a94, 0x01365c, 0x0160d2, 0x01952a, 0x090e1c)}
    print(f'slot {i} collected -> differs: {diff}')
    if i == 0:
        Image.fromarray(raw).resize((640, 448), Image.NEAREST).save(f'{SCR}/slot0_collected.png')
