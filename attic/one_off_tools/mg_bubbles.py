#!/usr/bin/env python3
# mg_bubbles.py (2026-10-12; Tyler: "Are air bubbles visible in ram?"): HCZ1/HCZ2 underwater refills and rising objects.
"""Where do air bubbles come from in RAM / the ROM? Replays Tyler's whole-game recording through HCZ1 (and
HCZ2), finds every frame his air refilled while still underwater (he breathed a bubble), and lists the
object slots near Sonic then (code pointer, position) and the ROM layout objects near those spots."""
import collections, os, struct, sys
sys.path.insert(0, '/home/tyler/Projects/sonic-rl'); os.chdir('/home/tyler/Projects/sonic-rl')
os.environ.update(MG_ENV_VERSION='2', CUDA_VISIBLE_DEVICES='', MG_LOG=os.devnull)
import numpy as np
import coordinator as C, demos

actions, meta = demos.load_play('demos/play_20261010-202923.npz')
table = meta['action_table']
base = C.make_retro(); base.reset(); base.em.set_state(demos.play_start_state(meta))
split, swap, cur = C.ActSplitter(), None, None
events, prev_air, prev_under = [], None, False
codes_near = collections.Counter()
want = {'HCZ1', 'HCZ2'}
done_acts = set()


def word(r, a): return int(r[a]) << 8 | int(r[a + 1])
def long(r, a): return (word(r, a) << 16) | word(r, a + 2)


for i, a in enumerate(actions):
    raw, info = C.emu_frame(base, table[a])
    ev = split.feed(i, info)
    if ev == 'start':
        cur = split.cur['name']
    elif ev == 'clear':
        done_acts.add(cur); cur = None
        if want <= done_acts:
            break
    if cur not in want:
        continue
    r = np.frombuffer(bytes(base.get_ram()), dtype=np.uint8)
    if swap is None:
        x = int(info['x'])
        if x >> 8 == x & 0xFF:
            continue
        swap = word(r, 0xB010) != x
    ram = r.reshape(-1, 2)[:, ::-1].reshape(-1) if swap else r
    under = bool(ram[0xB02A] & 0x40)
    air = int(ram[0xB02C])
    sx, sy = word(ram, 0xB010), word(ram, 0xB014)
    if under and prev_under and prev_air is not None and air > prev_air:
        near = []
        for b in range(0xB000 + 0x4A * 3, 0xCFB6, 0x4A):          # skip the player slots
            code = long(ram, b)
            if not code:
                continue
            ox, oy = word(ram, b + 0x10), word(ram, b + 0x14)
            if abs(ox - sx) < 96 and abs(oy - sy) < 96:
                near.append((hex(code), ox, oy))
                codes_near[hex(code)] += 1
        events.append((cur, i, prev_air, air, sx, sy, near))
    prev_air, prev_under = air, under

print(f'{len(events)} underwater air refills (bubbles breathed):')
for e in events:
    print(f'  {e[0]} frame {e[1]}: air {e[2]} -> {e[3]} at ({e[4]},{e[5]}); objects within 96 px: {e[6][:6]}')
print('object code pointers near those moments:', codes_near.most_common(8))

# ROM layout objects near the bubble spots (HCZ = zone 1, internal acts 0/1)
rom = open(os.path.join(os.path.dirname(C.stable_retro.__file__), 'data', 'stable', C.GAME, 'rom.md'), 'rb').read()
ids = collections.Counter()
for z, act in ((1, 0), (1, 1)):
    p = struct.unpack('>I', rom[C.RING_TABLE + 4 * (2 * z + act):C.RING_TABLE + 4 * (2 * z + act) + 4])[0]
    objs = []
    while True:
        x, yf, oid, sub = struct.unpack('>HHBB', rom[p:p + 6])
        if x == 0xFFFF: break
        objs.append((x, yf & 0xFFF, oid, sub)); p += 6
    for e in events:
        if (e[0] == 'HCZ1') != (act == 0):
            continue
        for x, y, oid, sub in objs:
            if abs(x - e[4]) < 160 and 0 <= y - e[5] < 400:          # below or at him (bubbles rise)
                ids[(hex(oid), sub)] += 1
print('ROM layout objects at or below the bubble spots (id, subtype): count', ids.most_common(10))

# --- rising objects in HCZ1 (bubbles): replay again, track object slots every frame underwater ---
base.reset(); base.em.set_state(demos.play_start_state(meta))
split, cur = C.ActSplitter(), None
tracks = {}                      # slot -> [code, first (x, y), last y, frames rising]
spawns = collections.Counter()
for i, a in enumerate(actions):
    raw, info = C.emu_frame(base, table[a])
    ev = split.feed(i, info)
    if ev == 'start':
        cur = split.cur['name']
    elif ev == 'clear':
        if cur == 'HCZ1':
            break
        cur = None
    if cur != 'HCZ1':
        continue
    r = np.frombuffer(bytes(base.get_ram()), dtype=np.uint8)
    ram = r.reshape(-1, 2)[:, ::-1].reshape(-1) if swap else r
    for b in range(0xB000 + 0x4A * 3, 0xCFB6, 0x4A):
        code = long(ram, b)
        ox, oy = word(ram, b + 0x10), word(ram, b + 0x14)
        t = tracks.get(b)
        if not code or t is None or t[0] != code or abs(ox - t[1][0]) > 24:
            if t and t[3] >= 40:
                spawns[(hex(t[0]), t[1][0] // 16 * 16, t[1][1] // 16 * 16)] += 1
            tracks[b] = [code, (ox, oy), oy, 0] if code else None
            continue
        if oy < t[2]:
            t[3] += 1
        t[2] = oy
print('rising objects in HCZ1 (code, first seen x, y), times seen:', spawns.most_common(12))
codes = collections.Counter(k[0] for k in spawns.elements())
print('by code:', codes.most_common(5))
z, act = 1, 0
p = struct.unpack('>I', rom[C.RING_TABLE + 4 * (2 * z + act):C.RING_TABLE + 4 * (2 * z + act) + 4])[0]
objs = []
while True:
    x, yf, oid, sub = struct.unpack('>HHBB', rom[p:p + 6])
    if x == 0xFFFF: break
    objs.append((x, yf & 0xFFF, oid, sub)); p += 6
near_ids = collections.Counter()
for (code, sx, sy), n in spawns.items():
    for x, y, oid, sub in objs:
        if abs(x - sx) <= 32 and abs(y - sy) <= 48:
            near_ids[(hex(oid), code)] += n
print('ROM objects at the rising objects\' first sightings (id, code): count', near_ids.most_common(8))
print('HCZ1 layout: count of each candidate id:', {i: sum(1 for o in objs if hex(o[2]) == i) for i, _ in [k for k, _ in near_ids.most_common(4)]})
