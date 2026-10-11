# Sonic 3 & Knuckles Special Stage RL — Full Project Context
*Last updated: 2026-10-10 ~16:50 — PAUSED by Tyler, nothing training. d7 parked at 975,008 steps (main test passed). Env v7 = uncommitted prototype: 11/14 recordings pass, stage 4 = checker bug (fixed, rerun pending), stages 6/10 = Up timing; a turn-queueing env problem needs Tyler's call before any v7 training. Branch `corner-decisions` (see "Current state")*

---

# ⚑ START HERE — orientation for a fresh chat

**Read this block first.** This doc lives in the Project knowledge base; you are
probably reading it cold. Below is what you need to be useful immediately without
re-asking Tyler things he's already answered.

### What this project is
Training an RL agent (RecurrentPPO / sb3-contrib, stable_retro) to play the Blue
Spheres special stages in Sonic 3 & Knuckles. Long arc: one agent that plays all
14 special stages. Runs locally on Tyler's RTX 2070 Super.
Second track (2026-09-30): the **base game** — acts of S3&K, learned from a
recording of Tyler's own playthrough and improved on (`coordinator.py`).

### How we work together (important — don't get this wrong)
- **Two channels, and they differ.** In the *web chat* you have NO filesystem
  access: deliver code via `present_files`, he downloads and moves it into
  `~/Projects/sonic-rl/` himself. In **Claude Code** you are running *on* his
  machine in `~/Projects/sonic-rl` with real tools — read and edit files
  directly, and run things in `venv`. Everything below about file swaps applies
  to the web-chat channel. (The repo moved from `~/sonic-rl` on 2026-08-25;
  every path in the code is now relative to the repo.)
- **After any file swap, confirm it landed** before launching:
  `grep -n "<new symbol>" ~/Projects/sonic-rl/<file>.py`. This has bitten
  repeatedly — running stale code and then debugging a bug that was already fixed.
- **For small edits, prefer an in-place heredoc patch** he can paste into the
  terminal over a full-file download. Much less friction, no file-move step.
- **File attachments from Tyler often arrive EMPTY** in this interface. If an
  upload has no content, ask him to paste the text directly into chat. Plain-text
  paste has been 100% reliable; attachments have not.
- He is technically strong and wants **honest tradeoffs, not agreement.** Push
  back when a plan is expensive or uninterpretable; propose the cheap test first.
  That discipline has repeatedly saved days of wasted compute.

### Current state (2026-09-29)
- **Nothing is training right now.** A logic-error review on 2026-09-29 found
  bugs in the env that shaped EVERY run so far (specialist, v1–v5) — see
  "What Changed 2026-09-29" right below this block. They are fixed as
  **env v2** on git branch **`env-v2-fixes`** (not yet merged to master).
- **v6 is defined but not started:** all 14 boards, v4's hyperparameters
  (512 LSTM, lr 3e-4) on env v2, so v6 vs v4 isolates the env fixes. Launch:
  `python3 smoke_stages.py v6` (must say ALL 14 READ CORRECTLY), then
  `nohup python3 train_ss_multistage.py v6 > logs/ss_multistage_v6.log 2>&1 &`.
- **Do not resume v5.** It trained on env v1; it is superseded by v6. v5 at
  19.5M: mean 2.6% of board, spread 0.8–4.8%, 0 wins.
- Every run pins `SS_ENV_VERSION` in its preset: v1–v5 and the specialist stay
  on env v1 (bit-identical to the old env, so old checkpoints still evaluate
  and watch correctly); v6 onward uses env v2.
- The project was consolidated on 2026-08-19 (see "Project layout") and is
  under **git**: branch or commit, never `_v6.py` copies.
- Analyzer: **`ms_analyze.py --run v6`** (the `--run` shortcut resolves the
  per-run log, so it can no longer read the ghost of a previous run).
- **Learning from Tyler's play** (branch **`demo-learning`**, stacked on
  `env-v2-fixes`): record a run with `python3 record_demo.py 1`, then
  `python3 train_ss_multistage.py d1`. See "Learning from Tyler's play" below.
- **Base game (2026-09-30): rebuilt; run `aiz1` PAUSED at 5.0M steps** (start
  point 82%) on Tyler's recording `demos/play_20260930-153946.npz` (AIZ1 in
  1:25.43, clean). Resume: `nohup python3 train_main.py aiz1 --resume >>
  logs/main_aiz1.log 2>&1 &`. He stopped 0:08 into AIZ2 inside a hidden
  giant-ring room — `record_play.py --resume ...` + 3 Backspace presses gets
  back before the tree. See "Base game rebuild 2026-09-30".
- **2026-10-09 21:53: Tyler chose to go straight to d5 and OK'd its
  criteria as written** (Pre-Committed Criteria, now AGREED). d5 launched
  fresh, alone on the machine, into `logs/ss_multistage_d5.log`. Progress:
  `demo_progress.py --run d5`. Gate at 40M frames (~5 h alone); probe
  (`PROBE_RUN=d5 probe_steering_v4.py 100 0.56`) when the start point first
  reaches 56%.
- **Overnight (Tyler away, 2026-10-09 ~22:10): Claude watches d5 and has
  Tyler's OK to start the fallback run** if d5 is stopped by one of its
  rules: **d4** (γ 0.95) if d5 never got below 56%, **d6** (= d4 with γ 0.99)
  if it did. Criteria: Pre-Committed Criteria, "Overnight fallback". Watch log
  `logs/d5_watch.log` (git-ignored).
- **2026-10-10 05:04: d5 STOPPED by its stall stop; d6 LAUNCHED (fallback).**
  d5's start point sat at **21% (ckpt 13) from 31.36M to 46.72M frames**
  (15.36M ≥ the agreed 15M) → `touch STOP_d5`, clean save at **2,113,600
  steps (~46.7M frames)**. Its full path: 97%→72% (8.09M), stuck at 72% 8.6M
  frames, 72%→24% (24.56M), 24% 5.2M frames, 23%/21% (31.36M), then stuck.
  Fresh starts: 0 clears in ~46.7M frames, mean ~80-88/102 late, best 100
  (main test — first fresh clear within 100M — left open, not failed). It
  got below 56%, so the fallback is **d6** (d4 with γ 0.99), launched fresh
  at 05:04 into `logs/ss_multistage_d6.log`; baseline d5. A read-only probe
  from 21% on d5's final save is the stall's diagnosis (tag `d5_v4_21`); the
  56% probe's model copy is kept as `attic/one_off_tools/d5_v4_56_model.*`.
  Resume d5 later with `--resume >>` if wanted — its save is intact.
- **2026-10-10 09:15-10:25: Tyler recorded stages 5-10** (d7 training
  alongside), all flawless clears, every one reproduced by the recorder:
  stage5_20261010-091532 (58, 4,148 frames), stage6_20261010-092841 (128,
  6,829), stage7_20261010-095111 (60, 5,339), stage8_20261010-100029 (64,
  3,227), stage9_20261010-102154 (155, 2,988), stage10_20261010-102523 (37,
  2,558). Then 10:39-10:50: stage11_20261010-103921 (67, 4,013),
  stage12_20261010-104326 (72, 2,702), stage13_20261010-104726 (54, 4,116),
  stage14_20261010-105034 (71, 3,427) — ALL 14 BOARDS RECORDED, all flawless.
  11-14: longest sphere-or-ring gap 555 / 361 / 921 (s13's last sphere) /
  368; bumper hits 1 / 0 / 0 / 4 (all → backwards); B jumps 0 / 3 / 9 / 6.
  What they exposed:
  1. **Stall rule vs long runs → env v6 (Tyler chose both parts).** v3+'s 600
     frames without a sphere ended his stage-5 clear at frame 2,448 (1,116- and
     657-frame ring-lined runs across the board). v6 = v5 + a ring pickup
     restarts the clock (longest sphere-or-ring gap: s1 257, s2 248, s3 253, s4
     393, s5 379, s8 318 [753 spheres-only], s9 91) + replays of his recordings
     skip the rule (s6 1,159 and s7 738 with neither while hunting the LAST
     spheres; the agent still gets 600). At s5's long gaps a red-free route
     (walk + 2-cell jumps, a lower bound) was 19-21 cells vs his 47-77: route
     choice, not necessity. Verified: v4/v5 seeded trajectories byte-identical
     to e5e27a1 (stages 1 and 5); stage 5 as an EPISODE on v6 clears at
     STALL_FRAMES 380 and stalls at 378 (its 379 gap), v5 still stalls at
     2,448; v6 random-agent episodes still stall; smoke_corners d3/d4 pass.
  2. **smoke_corners read spring launches as jumps** (fixed): $E432 = $80 is a
     B jump (always with B), $81 a spring launch (never). Stage 8 is the first
     route over springs (4; stage 9 has 6).
  3. **OPEN — bumper bounce → running BACKWARDS.** Every bumper hit in his
     recordings (6: s5 x1, s6 x2, s9 x1, s10 x2; s1-s4 have none) flips speed
     negative when the lock releases, and he runs backwards until he presses
     Up (speed winds -4096 → 0 → +4096 over ~16 frames; 5 of 6, s9's came 27
     frames before the clear). Up during the lock doesn't prevent it. The env
     has no Up action, and the header's "runs forward again on his own" does
     not match any of these. smoke_corners reads the reversal as "straight"
     (heading doesn't change). Breaks s5 (dies at 18 spheres) and s6 (52).
  4. **OPEN — jump timing.** A v4-v6 jump fires at the decision point (~half
     a cell before the centre). His 52 jumps take off anywhere from -124 to
     +116 units from the centre (26 before, 26 at/after). Breaks s9: at
     (13,27) he collected the blue, then jumped; the env takes off first,
     skips it and lands on red at (11,27) (dies at 129).
  v6 smoke (scratch copy of smoke_corners on d4's setup, one demo, CHK_ENV=6):
  s1, s2, s3, s4, s8 ALL PASS; s5, s6 fail (3), s9 fails (4); s7, s10 not
  yet run. Nothing has trained on v6; d4 (v5, stages 1-4) is untouched.
- **2026-10-10 23:21: d9 RUNNING overnight (Claude in charge of the box until morning, Tyler's words: "I trust your judgement and decision making").** Before it: (1) **d8 PASSED** (median clear 3,089 vs bar 3,202, 82% clears) and **the control d8c collapsed** at ~16.5M frames after being faster than d8 without the bonus — read under "Run d8" RESULTS; (2) d9's first launch refused stage 6 — a bug in the pre-launch replay check (stall rule), fixed cde6b65, d9 launched fresh. **Main game:** Tyler recorded the WHOLE game in one session (`demos/play_20261010-202923.npz`, 27 acts); act names fixed (LRZ3 / HPZ1 / DEZ3 / LBZ2b), MG env v2's route matcher fixed (ties by the clock), and `check_play_acts.py` now shows his own inputs clear every act on his frame — section "Tyler's full playthrough" under the base game. Watch: `logs/d9_watch.log`.
- **2026-10-10 ~20:15: BACKUPS + GitHub.** All four sonic-rl branches are now pushed to GitHub (SSH key `~/.ssh/id_ed25519_github`; before this nothing past master was on any remote). What git ignores (models, *.pkl, checkpoints, ss_log*.jsonl, logs/, states/, roms/) is snapshotted daily at 05:00 to `/mnt/games/backups/sonic-rl/<date_time>/` (`~/.local/bin/backup-projects` — now 15 folders incl. chipsim, barony, chess/stockfish, erenshor-ai and Claude's memory — systemd user timer; hard-linked rsync snapshots, newest 14 kept, venv/ excluded). Verified on setup: file-for-file match except logs being written, model restore byte-identical, unchanged files shared between snapshots.
- **2026-10-10 ~20:45: WHERE d7 LOSES TIME vs Tyler — measured, CPU only** (`attic/one_off_tools/route_vs_tyler.py d7 60`, d7's save run stochastically from 60 fresh starts of stage 1 + 5 deterministic; his recording replayed through the same env; log/json beside it). 51/60 cleared. Median clear 3,193 vs his 2,925 (+268). **Turns 70 vs 55: each turn is exactly 15 frames of rotating in place (his 825 still frames = 55 x 15; d7's 1,050 = 70 x 15), so the 15 extra turns are 225 frames = ~84% of the gap.** Cells travelled 151 vs 146 (+5); outline trick identical (4 multi-sphere frames, 16 spheres); bumper hits 0/0; d7 jumps 16 times (436 frames airborne) vs his 0, and airborne travel costs no time. By quarter: +230 frames at 25% of spheres, +168 at 50%, +298 at 75%, +268 at the clear, so most of the loss is in the opening. Why nothing fixed it: the per-cell discount doesn't charge for turns (a turn adds no decision); d8's bonus prices one at ~105 (15 x 7). **Read for d8: does its turn count fall?** (rerun this script with `d8` once it has trained). speed_progress.py now prints bootstrap 95% CIs: d7's last 300 = median clear 3,179-3,234, clear rate 62.3-73.0%.
- **2026-10-10 ~20:25: control d8c STOPPED — it harmed Tyler's gameplay** (The Witcher 3), ~15 min after launch, despite being pinned to cores 6-7 at nice 19: the GPU (policy inference + updates, 508 MB VRAM) is shared, and pinning/nice only limit CPU. Clean save at **19,000 steps**. **Rule from now on: no training while Tyler games**; CPU-only checks on <= 2 cores at nice 19 are fine. Resume d8 / d8c with `--resume >>` their logs when he's not playing. v7 committed (075aace).
- **2026-10-10 ~20:00: v7 (+ up-now + turn_here) VERIFIED — ALL 14 recordings PASS, 0 failures; v4-v6 untouched.** Still UNCOMMITTED (Tyler's rule: report first); backup patch refreshed. Logs: `attic/one_off_tools/v7_check_2026-10-10_final/` (one per stage + `isolation_v4-v6.log`). Frames vs his: s1 2,925=2,925, s2 3,783/3,786, s3 3,944=, s4 4,183/4,185, s5 4,145/4,148, **s6 6,823/6,829 (was 6,834, 5 slow)**, s7 5,339=, s8 3,227=, s9 2,988=, **s10 2,560/2,558 (was 2,563)**, s11 4,006/4,013, s12 2,702=, s13 4,116=, **s14 3,423/3,427 (was 3,435)**. Only s10 is still over his time (2 frames, inside the 2-per-Up allowance). smoke_corners picked up-now for 182 of 224 Up trials on s6, 30/31 on s10, 84/211 on s14, 0/25 on s5. Caveat: "neither Up timing reached your next corner" 42 times (s6) / 31 (s14) across start points; every start point still reproduced his finish (cell alignment). No "restore not faithful" warnings. `check_v7_isolation.py 4 5 6`: v4 (with and without the compass), v5, v6 obs + rewards identical to git HEAD over 1,200 seeded steps each. Ran pinned to 2-3 cores at nice 19 while Tyler played. **Next: Tyler's OK to commit v7; then define the multi-stage v7 run; d8 (paused 21,800 steps) and its control are queued.**
- **2026-10-10 ~18:45: EVERY SpecialStageEnv opened a window — FIXED (eda22d4).** `stable_retro.make()` defaults to `render_mode='human'`, which opens a pyglet window and redraws it on every step and reset; special_stage_env never passed render_mode (coordinator.py passes None). So each training run had 8 live windows, and each check script popped one that stole focus from Tyler's game (he reported it while playing The Witcher 3). Now `render_mode='rgb_array'`: no window, `render()` still returns the frame (watch_ss.py draws it itself), and emulator state doesn't depend on it. Verified: a new env loads neither pyglet nor stable_retro.rendering, viewer None. **To read on d8's resume: steps/s vs its 95 before** — every earlier run paid for drawing those windows.
- **2026-10-10 ~18:20: v7 recording check FINAL (checker: lookahead on turns, not onto jumps): all 14 of his clears reproduce.** 12 PASS (s4 now passes: 115/115, 2 frames sooner). s6 and s10 clear every sphere but 5 frames slow on 2 Ups each (allowance 2 per Up) — settled by the Up-timing measurement below. No other failure. Logs (one per stage, latest run) in `attic/one_off_tools/v7_check_2026-10-10_final/`; backup patch refreshed (still uncommitted).
- **2026-10-10 17:37: d8 PAUSED at Tyler's request** (he wanted the v7 reruns done before using the machine for games). Clean save at **21,800 steps** (~0.5M frames; 12 clears in its first 17 fresh starts). Resume with the `--resume >>` command below; nothing else changes. The control (d7 resumed, no bonus) runs AFTER d8 (Tyler's choice).
- **2026-10-10 ~17:50: run d8 LAUNCHED** (Tyler: "Go ahead and build d8"). d8 = d7 + a clear-time bonus (`SS_SPEED_BONUS=7`), starting from d7's save; criteria under Pre-Committed Criteria, "Run d8". Log `logs/ss_multistage_d8.log`, episodes `ss_log_d8.jsonl`; stop `touch STOP_d8`; resume `source venv/bin/activate && nohup python3 train_ss_multistage.py d8 --resume >> logs/ss_multistage_d8.log 2>&1 &`. The v7 prototype was stashed while building d8 (git stash "v7 prototype 2026-10-10") so d8 is committed on clean code, then popped back into the working tree (still uncommitted). v7 rerun on the lookahead checker (same afternoon): stage 4 now PASSES (115/115, 2 frames sooner); 8, 9, 12 and 14 still pass. Stage 6 REGRESSED (died at 127/128 in 6,743 frames; 42 jumps where neither timing reached his next corner): the lookahead plays his next choice at its nominal timing, and on stage 6 that choice is usually another hop. To fix: skip the lookahead (or try both timings) when his next choice is a jump.
- **PAUSED by Tyler 2026-10-10 ~16:50 — nothing training; v7 still an
  UNCOMMITTED prototype, report NOT yet given to Tyler.** That afternoon:
  the full 14-recording v7 check ran (logs kept in
  `attic/one_off_tools/v7_check_2026-10-10/`; working tree == backup patch,
  refreshed). **11 of 14 PASS** (1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14; s1,
  s7, s12, s13 frame-exact, s2/s5/s11 3-7 frames sooner, s14 8 frames slow
  on 4 Ups = exactly at the allowance). **Stages 6 and 10 clear every sphere
  but run 5 frames slow on 2 Ups** (allowance 2 per Up = 4): stage 6 now
  clears 128/128 (before the landing decisions it died at 127). Traced on
  s10 (`v7_drift.py`): the first Up costs ~4 frames (he holds Up through the
  bounce lock; the env presses once past the cell's centre), the second ~1;
  so the "2 per Up" allowance was set too tight. Open choice for Tyler:
  accept (allowance 4 per Up, with this as the reason) or add a second,
  immediate Up timing. **Stage 4 FAILED on a CHECKER bug, not a missing
  move** (`v7_dbg_follow.py`): at (21,23) jump and late-jump both land in
  (19,23); the checker took late-jump (landing time closest to his) but
  that landing is past (19,23)'s turn window, so his right turn queued and
  fired at (18,23). Fixed in smoke_corners (uncommitted): best_jump now
  prefers the timing from which his NEXT choice reaches the corner after
  it, then landing time. The rerun of the 10 jump stages on that fix was
  STOPPED by the pause before any finished — rerun them first (`make_smoke_
  one_demo.py`, then stages 4-10, 12-14 with CHK_ENV=7).
  **An ENV problem found while tracing stage 4 — must be decided before any
  v7 training** (`attic/one_off_tools/measure_v7_turn_outcome.py`, random
  agent, 25 eps x 14 stages; MV=6 for the baseline). When the agent picks a
  turn, does it fire in that step, or is it still queued when the next
  decision arrives (fires a cell late)?
  | decision kind | v7 decision points | turns fired in-step | queued (a cell late) |
  |---|---|---|---|
  | normal | 1,782 | 87% | 13% (v6: same 13%) |
  | landing (v7) | 1,508 | 38% | **62%** |
  | backwards restart (v7) | 201 | 0% | **100%** |
  | forwards after Up (v7) | 100 | 61% | 39% |
  Why it matters: the obs has no position within the cell (dropped for
  generalization), so at a landing the agent cannot tell whether "right"
  turns here or a cell later; ss_next_corner (in the obs) shows a queued
  turn only afterwards, and the next cell then offers a SECOND decision with
  that turn already queued. The 13% on normal decisions is PRE-EXISTING
  (v4-v6, so d3/d5/d7 had it): a decision is offered on the frame Sonic
  ENTERS a bumper cell (checked: stage 1 (12,4), stage 2 (25,14) are code
  3), and the push-back moves him into the previous cell the next frame,
  which counts as a new decision. Directions, none built: (a) never stop
  for a decision while a turn is queued (needs an exception running
  backwards: the game doesn't turn then, so the step would run to
  CORNER_MAX_FRAMES and Up would be impossible); (b) one obs bit "a turn
  now fires in this cell" (window still ahead); (c) no decision on a bumper
  cell's entry frame. Ask Tyler before building any of them.
- **PAUSED by Tyler 2026-10-10 ~13:40 — nothing training** (he wanted the
  machine for games). d7 stopped cleanly at **975,008 steps (22.70M frames)**;
  start point 1% (ckpt 0) since 13.17M — the curriculum walked all the way
  back; last 300 fresh starts: **203 clears**, mean 93.7/102. The 15M stall
  stop is moot at ckpt 0 (nothing left to move); the 50M gate read is still
  formally open (bar long met). What next for d7 is Tyler's call. Resume:
  `source venv/bin/activate && nohup python3 train_ss_multistage.py d7
  --resume >> logs/ss_multistage_d7.log 2>&1 &`.
  **Env v7 is an UNCOMMITTED PROTOTYPE in the working tree** (Tyler: prototype
  and report before any commit or training; backup patch
  attic/v7_prototype_2026-10-10.patch, untracked). v7 = v6 + `up` (pressed
  once past the cell's centre, as he does) + `late-jump` (B once past the
  centre) + extra decision points: running backwards, the first move after
  standing still (bounce lock / corner rotation); the first move forwards
  after Up; and every landing (jump or spring). Findings behind it, all
  measured on his recordings / in the emulator: every bumper hit leaves Sonic
  running BACKWARDS until Up (Up ignored while rotating, 1 frame enough when
  moving, nothing going forwards); a sphere is collected on ENTERING its
  cell; a jump covers ~512 units at any speed (32/26/22/19 frames at
  4096/5120/6144/7168 — v4's "32 frames" was base speed); B tapped in the air
  queues a hop on landing (18 of 20 stage-6 jumps, 1-3 frames grounded).
  smoke_corners v7 reading (also uncommitted): Up turn-round ONSETS (once
  each, on a decision's own frame too); jumps early/late, and a take-off on
  entering the next cell = straight here + jump there; at each jump it TRIES
  jump / late-jump / straight-then-jump from a snapshot and keeps the one
  reaching your next corner in the frames you took (snapshot yardstick: no
  "restore not faithful" warnings seen); v7 allows 2 frames late per Up.
  Verified on this prototype: v4/v5/v6 seeded trajectories byte-identical to
  HEAD (stages 1, 5, 10); smoke_corners d3 + d4 ALL PASS. Last full v7 run
  BEFORE the landing-decision change: 12 of 14 pass; stage 10 clears 5
  frames slow (2 Ups — he holds Up through a lock, the env presses on the
  first move); stage 6 died at 127/128 on chained hops (→ landing decisions).
  After adding landing decisions the checker had to pick jump timing by
  landing TIME, not cell (else it never switched; stage 4 regressed) — that
  version is written but its stage 4 / 6 tests were stopped by the pause.
  **Next on resume:** rerun all 14 on v7 — `python3 attic/one_off_tools/
  make_smoke_one_demo.py` (regenerates from smoke_corners.py), then per
  recording `CHK_ENV=7 python3 attic/one_off_tools/smoke_one_demo.py <stage>
  <demo basename>` — then write the v7 report for Tyler before committing.
- **2026-10-10 10:40: d7 PASSED ITS MAIN TEST** — first fresh-start clear of
  stage 1 at 10.3M frames (102/102 in 5,064 frames), a second soon after in
  3,471 frames (Tyler's demo: 2,925). Bar: within 100M frames. d5: none in
  46.7M; no earlier run ever cleared stage 1 from the start. At 10.05M frames
  its start point was 41% vs d5's 72% at equal frames (21 ckpts ahead). Still
  open: the 21% probe and the 50M gate. Watch continues.
- **2026-10-10 09:07: d7 RESUMED by Claude** (Tyler: "please resume the sonic
  rl project"); same `--resume >>` command, PID 151406, banner ram_feats=13.
  Launch inside `venv` (`source venv/bin/activate`): the first attempt used
  the bare pyenv python3 and died at `import torch` (noted in the log). Watch
  as for d5: 30-min summaries in `logs/d7_watch.log`; events = process death,
  first reach of 21% (ckpt <= 13) + the save after it (probe due), first fresh
  clear, 50M-frame gate, 15M stall stop. At 1.29M frames: 90%, level with d5.
- **PAUSED by Tyler 2026-10-10 08:01 — nothing training** (he's using the
  machine for games). d7 stopped cleanly at **49,200 steps (~1.1M frames,
  start point 91%)**. Resume: `nohup python3 train_ss_multistage.py d7
  --resume >> logs/ss_multistage_d7.log 2>&1 &` (append, not `>`). Tyler
  wants Claude to watch d7 when it resumes, as with d5 overnight: the
  21% probe, gate at 50M frames, stall stop, first fresh clear.
- **2026-10-10 07:53: d7 LAUNCHED** (d5 + compass to the nearest blue; Tyler's
  choice of straight-line compass on d5's setup). Criteria AGREED before
  launch: Pre-Committed Criteria, "Run d7". Log `logs/ss_multistage_d7.log`;
  progress `attic/one_off_tools/watch_demo_run.py d7 d5`.
- **2026-10-10 07:35: d6 STOPPED by Tyler** (clean save at 614,400 steps,
  ~16.5M frames), to try a pointer toward the nearest sphere next. Nothing
  training. d6 at the stop: start points s1 74% / s2 64% / s3 79% / s4 73%,
  0 fresh clears (bests 81/102, 102/127, 96/118, 82/115); stage 1 was 6
  ckpts AHEAD of d5 at equal stage-1 frames (74% vs 83% at 5.2M) — an early
  read, not the agreed 50M-frame interference read, which it never reached.
- **d5's 21% stall, DIAGNOSED 2026-10-10 05:30 (probe on its final save,
  tag `d5_v4_21`; caveat: `probe_steering_v4.py 100 0.21` starts its band at
  ckpt 12 = 19%, one point behind training's ckpt 13).** Analyzer yardsticks
  held (choice = move 100% of 15,292; 99% paired). **0 of 100 reach the
  target; 88 stalls, 12 red.** (25,21) is no longer the problem (collected
  in 94, entering from (25,20) 93). The leftovers sit on the far side of the
  board, columns 5-6 rows 12-17 (e.g. (15,5)/(16,5)/(17,5) in 11 stalls). In
  the stall window it wanders (distinct cells/intersection 0.79, vs d3's 0.06
  pinballing) at chance (53% vs 51% on a shortest route), nearest blue a
  median 22 intersections away.
  **The leftover blues are OUT OF VIEW when it stalls — MEASURED** with
  `attic/one_off_tools/stall_view.py` (same rule as probe_visibility.py:
  toroidal Chebyshev distance <= PATCH_R 7). Yardstick: while collecting a
  blue is in view 98-99% of frames. Stall windows, median share of the window
  with a blue in view: **d5 at 21%: 0%** (18 of 88 windows have it >= half
  the time; nearest blue median 12 cells); d5 at 56%: 0% (8 cells); d3 at
  56%: 0% (10 cells — the sealed-in staircase is ~10 rows from (25,21)).
  This REOPENS for the v4 runs what the d2 probe refuted on v3 (there a blue
  was in view ~95% of the window): "Don't propose a wider patch /
  sphere-direction feature" was written from d2 and does not hold for d3/d5
  stalls. Not acted on — Tyler's call.
- **2026-10-10 ~00:45: d5 PASSED d3's 56% wall** (ckpt 33 at 19.88M frames;
  d3 sat at ckpt 34 from 11.62M to 17.31M and never left). The pointer only
  moves back, so the 40M-frame gate's bar ("below 56%") is already met; d5
  keeps running. Its path: 97%→72% at d3's pace (+2 to +4 ckpts behind), then
  **stuck at 72% (ckpt 47) for 8.6M frames** (8.09M→16.69M; 0 of 377
  episodes from ckpt 47 reached the target of 31, 81% ended by the stall
  rule), then 72%→36% in ~4M frames (36% at 20.66M). Frames:
  `attic/one_off_tools/watch_demo_run.py d5 d3` (frames, pointer = running min
  of `demo_start`, d3 at equal stage-1 frames; reproduces d3's 17.31M / 56%
  since 11.62M exactly).
- **d5's mechanism check FAILED as written, while the outcome improved.**
  Probe on the first save after 56% (00:49:50; by then the pointer was at
  **36%**, so this measures a model 12 points past the wall):
  `probe_steering_d5_v4_56.log`, steer_*_d5_v4_56.json; analyzer yardsticks
  held (choice = move made 100% of 7,671; 99% paired). From 56%:
  **30 clears / 55 stalls / 15 red** (d3: 6 / 90 / 4). But at (26,21) with
  (25,21) still blue, **0 of 49 first visits step onto (25,21)** (pass needed
  left as the majority): arriving from the west it jumps east 21 of 26 (as
  d3, 65 of 65); arriving from the east (new in d5, 23 of 49) it goes
  straight 23 of 23 where right would take it. Instead d5 **comes back for
  (25,21) along row 25** — entered from (25,20) 31, (25,22) 16, (24,21) 4
  (d3: (24,21) 20, (25,20) 6, (25,22) 3); collected in 51 of 100 episodes
  (d3: 29); stalls with only (25,21) left 11 of 55 (d3: 46 of 90). So γ 0.99
  got it past the wall by a different plan than predicted: it still skips
  (25,21) at (26,21) but no longer seals itself in before returning.
  Tools: `attic/one_off_tools/pocket_choice.py` (first choice at (26,21);
  reproduces d3's 74 of 87 / P 0.998), `pocket_next.py` (next cell actually
  entered), `pocket_fate.py` (how/whether (25,21) gets collected).
- **d3 is PARKED, not resumed:** saved at **821,424 steps / 17.31M frames**,
  start point **56%** since 11.62M frames, fresh starts mean 84.6/102, best
  99, 0 clears. It never reached its 25M-frame gate or its 26.62M stall stop;
  Tyler skipped the remaining ~70 min because the 56% wall was already
  measured (below). It passed the gate's bar (past 78%) long before 25M.
- **2026-10-09 20:16: d3 RESUMED** (marker in its log). Frames counted exactly
  from `ss_log_d3.jsonl` (`steps` − demo_frac × 2925): **11.84M** at resume
  (the ~12.4M below was steps × 23), so the 25M-frame gate is ~22:00.
- **Why d3 stalls at 56% — MEASURED 2026-10-09 (on the 537,624-step save,
  before the gate): a mistake whose penalty lands ~100 corners later, at
  γ = 0.95 per corner.** `attic/one_off_tools/probe_steering_v4.py [eps]
  [start frac]` + `steering_analyze_v4.py <tag>` (v3 probe's scoring copied
  unchanged; each decision paired with the intersection it chose for).
  Yardsticks held: 99% of 9,894 decisions paired; straight/left/right choice
  = the move made at 5,296 of 5,298; turn byte 807/807; pickups on blue.
  - **The d2 mechanism is fixed.** Where only straight was right, while
    collecting: straight **64%** (d2 on v3: 6-9%), jump 34%. Turns due: 72%.
    Airborne 39% of frames (d2 46%), 24% from 78%.
  - **From 78%: 90/100 clears. From 56%: 6 clears, 90 stalls, 4 red.** 78
    of the 90 stalls are in ONE place: the zigzag staircase of blues between
    bumper columns 24 and 27 (rows 12-19, cols 25-26; exits only top and
    bottom). Once all of it is red, nothing in it can be turned on, so Sonic
    pinballs between the bumpers and hops over red until the stall rule.
  - Chain: at **(26,21)** (Tyler takes it at 62% of his run) the policy
    jumps east with **P = 0.998** (74 of 87 episodes) instead of turning
    left onto blue (25,21), one step away — the rest of the time it takes a
    1-step blue 99%. **53 of 90 stalls end with exactly that one sphere
    left.** Later, at (15,26), it jumps over bumper (14,26) (97/100
    episodes), stranding (15,25)/(14,25)/(13,25); it goes back for them,
    which seals it in the red staircase. From 78% that "pocket last" plan
    works (16 of 21 jumpers cleared) because the pocket IS the last of the
    board; from 56% (25,21) is still out there.
  - **The penalty for the (26,21) jump arrives a median 101 decisions later
    (82-120): 0.95^101 = 0.006.** A sphere is worth the same whenever it is
    collected, so skipping it costs nothing until then: the error is
    invisible to d3's credit assignment. (At γ 0.99: 0.36.) d4 has γ 0.95 too.
  - **The reward clip does NOT bind here — MEASURED 2026-10-09 22:10** (this
    replaces an earlier line here saying it "caps that penalty as well").
    d3's saved VecNormalize return std is **28,313** (the ~710 in the 09-29
    note was the old env's), so a stall/red at 48 spheres (−98,500 raw, the
    median for d3's stalls near 56%) normalizes to **−3.5**, a sphere to
    +0.07: all inside `clip_reward=10`. Not a reason for d3's wall, and not
    a useful next experiment for it.
- **d5 = d3 with γ 0.99, nothing else changed — DEFINED, NOT STARTED**
  (Tyler's go 2026-10-09; `smoke_corners.py d5` ALL CHECKS PASS). Plan: let
  d3 reach its 25M-frame gate (re-run the v4 probe on that save) and, if
  still at 56%, its stall stop at **26.62M frames** (56% since 11.62M); then
  `touch STOP_d3` and launch fresh:
  `nohup python3 train_ss_multistage.py d5 > logs/ss_multistage_d5.log 2>&1 &`.
  Its criteria are in Pre-Committed Criteria (AGREED 21:53; d3 was not run
  to its gate/stall stop after all — see the top of "Current state").
  d4 (γ 0.95) waits on d5's answer.
- **PAUSED by Tyler 2026-10-09 ~19:30 — nothing is training.** Pick up here:
  - **d3** (env v4, stage 1) saved at **537,616 steps (~12.4M frames)**: start
    point **56%** of Tyler's run (d2 stalled at 78% after 22.8M frames);
    fresh starts mean **80.8/102**, best **98**, **0 clears** yet. Resume:
    `nohup python3 train_ss_multistage.py d3 --resume >> logs/ss_multistage_d3.log 2>&1 &`
    (~94 steps/s ≈ 2,200 frames/s alone). Its pre-committed checks (see
    Pre-Committed Criteria): gate at **25M frames** (start point past 78% —
    already met; re-run the steering probe adapted to v4 there), stall stop at
    15M frames without a move, first fresh clear within 100M frames.
  - **d4** (env v5, stages 1-4, one demo each, 5-7 held out) is defined and
    verified (`smoke_corners.py d4` passes), NOT STARTED — waits on d3's gate
    and Tyler's go.
  - Tyler's demos: stage1 (Sep 30), stage2/3/4 (2026-10-09), all clears.
  - Branch `corner-decisions` (on top of base-game ← demo-learning ←
    env-v2-fixes): nothing merged or pushed. d2 (v3) remains resumable.
- **`d2` STOPPED by Tyler 2026-10-09 ~16:45 at 6,355,232 steps** to give d3
  the machine (graceful stop, resumable with `--resume >>`). Start point 78%
  (79%→78% at 5.71M after ~2.3M steps at 79%). In frames: 97%→78% took 22.8M
  frames; early points ~0.1M frames each, later 1-9M (78% alone: 9.3M).
- Before that: **`d2` resumed 2026-10-09 15:08 at 5,148,624 steps** (~340
  steps/s, second `=== resumed 2026-10-09` marker in the log). Before that it
  was paused by Tyler 13:58 that day after a 12:28 resume at 4,050,816 steps.
  At 5.2M: start point still 79%; fresh-start bucket mean 26.2, new best 75
  (was 69), 0 clears. Stop: `touch STOP_d2`.
  At 4.89M (13:38): start point **still 79%**; since the resume 0/316 matches
  from 79%, 9.5% from 83%, overall falling 2.4%→1.7%; fresh starts mean
  21.8→26.8 spheres, best 63, 0 clears. 53% of demo-start episodes ended by
  the stall rule, 35% on own-trail red.
- **"It can't see the last spheres" — TESTED 2026-10-09 and REFUTED.**
  `attic/one_off_tools/probe_visibility.py` played the 13:41 d2 save 100× from
  the 79% start (CPU, own log, no pool captures) and replayed Tyler's run as
  the yardstick; `probe_analyze.py` summarizes. Yardsticks held: the patch
  from the agent's own obs and from RAM agreed on all 39,115 steps; blue cells
  = the game's counter every step; one step before a pickup the clear-path
  distance was 1 in 1,661 of 1,773 (the rest = springs/loops). Result: 56
  stalls, 38 red, 6 clears. In **all 56 stalls a blue sphere was in the 15×15
  view (95% of the 600-frame window, median) AND reachable without crossing
  red**; median 3 blues left, nearest **3 cells** when the stall clock started,
  and the agent passed within 1 cell. Tyler's own run from 79% always had a
  blue within 5 cells (median 1) and never needed a jump. So a wider patch or a
  direction-to-nearest-sphere feature would NOT fix this — it sees them and
  doesn't steer onto them. Not yet measured: what it does in those 10 s
  (turn presses, heading, lane vs the target's lane).
- **Why it doesn't steer onto them — MEASURED 2026-10-09: on the ground it
  can't go straight, so it jumps to go straight, and in the air it can't turn
  or collect.** `attic/one_off_tools/probe_steering.py` played the 15:08 d2
  save (5.15M) 100× from 79% (CPU, raw per-frame pose + per-decision action
  probabilities + every board change); `steering_analyze.py` scores every
  intersection's move against a heading-aware shortest route to the nearest
  blue (no reversing, no red/bumpers, no jumps). Yardsticks held: Tyler's moves
  are on a shortest route at 96% of his 145 intersections (a random legal move:
  58%), 100% when a blue was 1 step away; the turn inferred from cells matched
  the queued-corner byte on 859/859 turns; 548/548 pickups landed on a blue
  cell. Results (58 stalls, 30 red, 12 clears):
  - **Air:** Sonic is airborne ($E432 bit 7; value 1 = a buffered jump, still
    on the ground) on 46% of frames (53% in stall windows); Tyler on 0% from
    79%. 64% of intersections are crossed in the air (74% in stalls). **Turns
    don't happen in the air:** a turn was pressed before 85–91% of airborne
    intersections and only 7–8% turned (on the ground: 92%).
  - **Ground:** it turns at ~90% of intersections (Tyler: 44%). Where ONLY
    straight was right it went straight 6% (collecting) / 9% (stalled); the
    policy's chance of pressing no turn in the whole window is median 0.00 /
    0.07 (P(turn) ~0.5 per decision, ~3 decisions per cell). A turn needs one
    press in the window; straight needs none in all three — so the policy never
    learned ground straights and uses a jump (one press, ~24 frames of straight)
    instead. P(start a jump) is 0.12 per ground decision and 0.39 in the air
    (chaining hops).
  - **Stragglers:** in 31 of 58 stall windows the closest approach was a blue
    1 step away — and in 26 of those 31 Sonic was airborne at that
    intersection (73 airborne vs 15 grounded dist-1 moments in all). Grounded
    moves in stall windows are at chance (57% vs 53%; collecting 75% vs 55%).
    Direct fly-overs of a blue: only 14, in 17% of windows — it mostly passes
    BESIDE the straggler mid-hop, not over it.
  - Same disease as d1 ("never held a straight line"): env v3 cut the window
    from ~12 choices per cell to ~3, which left the asymmetry intact.
  Candidate fix (a NEW run, Tyler's call — not started): **one decision per
  intersection** (straight / left / right / jump, taken at each corner's centre
  for the next one), so straight costs one choice like a turn; Tyler's demo is
  145 such decisions. Needs a gamma rescale (per cell, not per 4 frames) and an
  emulator check that a one-frame tap after a centre always queues the next
  corner. Raw outputs (`steer_*.json`, 33 MB, and the probe's model copy
  `d2.zip/.pkl`) are untracked in `attic/one_off_tools/`.
- **Env v4 + run `d3` — BUILT 2026-10-09 (Tyler said go), branch
  `corner-decisions`.** `SS_ENV_VERSION=4`: one step = one corner, actions
  straight / left / right / jump; the env taps the button once and plays on
  to the next decision point (grounded, moving, just entered a cell not yet
  decided, still before its centre along his ACTUAL motion). Same obs, reward,
  stall rule as v3. Emulator facts it rests on (env v4 header has them all):
  a queued turn fires in a 32-unit window just past a corner's centre, then
  16 frames of rotation; a tap anywhere after the previous window queues it;
  jumps take off on the tap frame and last 32 frames; a turn tapped in the air
  fires after landing, but a jump tapped with/after a turn CANCELS it (v3's
  jump+turn actions were plain jumps); Up does nothing while running and skips
  the ~99-frame start wait (fresh v4 starts tap it, as Tyler did); a bumper
  pushes Sonic back ~a cell and he runs into it again until a turn fires, with
  his motion running AGAINST heading x speed sign (why decisions use actual
  motion — the first version used heading and got stuck 240 frames).
  Verified: `python3 smoke_corners.py d3` — Tyler's run is 145 corners
  (straight 90, right 31, left 24, jump 0) and replays through v4's own step
  frame-exact (2925 frames, 102, clear); all 65 v4 start points begin on his
  decision points and his choices from each reproduce his finish. v1/v2/v3
  play bit-identical to before (seeded random play, with and without demos —
  NB raw emulator state bytes differ run to run even for identical code, so
  never hash them). Random play: ~21 frames/step, no step hits the 240-frame
  cap; sub-8-frame steps that don't end the episode are all bumper bounces.
  `d3` = d2's config on env v4 with gamma 0.95 per corner (~the same ~20-cell
  horizon). A step is ~16-32 frames vs d2's 4 — **compare d2 and d3 in
  frames**: d2's pre-committed 50M steps = 200M frames ≈ 9M d3 steps (Tyler's
  call whether that is the d3 budget). `watch_ss.py d3` draws every frame.
- **Tyler's stage-2 demo, recorded 2026-10-09 17:02:**
  `demos/stage2_20261009-170251.npz` — flawless clear, 127/127, 3,786 frames,
  0 rewinds, no jumps. NOT used by d3 (d3 is pinned to the stage-1 demo); it
  is for a later multi-stage run. As v4 corners: 213 (straight 158, right 35,
  left 20), same route, 127, clear, in 3,783 frames — 3 fewer because a few of
  his turns were tapped on the last frame of a corner's window (fires one
  frame late) while v4 queues before the window.
- **Tyler's stage-3 demo, 2026-10-09 17:09:** `demos/stage3_20261009-170931.npz`
  — flawless clear, 118/118, 3,944 frames, 0 rewinds. Converting it to v4
  corners exposed a **v4 bug, fixed the same day**: the game sometimes skips a
  frame's logic (a lag frame — nothing moves, input ignored), and it lands
  right after Sonic crosses into a new cell, i.e. right after a decision point
  (2 lag frames in 10,655 of Tyler's frames, BOTH right after a decision). v4
  tapped for one frame, so a left at stage 3's frame 3339 was dropped. v4 now
  holds the button until the game takes it (corner byte / jump byte), at most
  `CORNER_TAP_MAX` = 3 frames. After the fix: stage 1 frame-exact (smoke
  passes), stage 2 unchanged (3,783 frames), stage 3 frame-exact (224 corners,
  118, clear); 4,000 random v4 steps identical to before the fix. d3 ran its
  first 193,400 steps on the one-frame tap (Tyler's stage-1 run has no lag
  frames, so the effect was ~nil) and was resumed on the fix at 193,400
  steps (~4.5M frames), marked "=== resumed ... lag-frame fix" in its log.
- **Tyler's stage-4 demo, 2026-10-09 17:26:** `demos/stage4_20261009-172620.npz`
  — clear, 115/115, 4,185 frames, 1 rewind; the first demo with jumps. It
  exposed a **v4 expressiveness gap: turn-then-jump.** Jump pressed DURING a
  corner's rotation is buffered ($E432 = 1, the "j=1" seen in d2's probe) and
  fires the instant the turn ends — launching straight out of the corner. v4's
  corner gets one choice (turn OR jump) and its next jump chance is half a
  cell later. Tyler used it twice at the end of stage 4. Tested in scratch:
  with two extra actions (left+jump, right+jump: tap the turn, press B once
  the rotation starts, until the jump byte is set) ALL FOUR demos reproduce
  their clears (stage 4: 115/115 in 4,183 frames; corners: straight 121, left
  50, right 32, jump 3, right+jump 2). Also needed for checking demos with
  jumps: align human and v4 decisions BY CELL — a jump taken at a different
  frame lands at a different phase, so v4 can get a decision on a cell the
  human passed without one (answer: straight). d3 is unaffected: stage 1 has
  no jumps. Proposed for the multi-stage run: env v5 = v4 + the two
  turn-jump actions (a new action space, so not for d3).
- **Env v5 BUILT 2026-10-09 (Tyler said go):** `SS_ENV_VERSION=5` = v4 + actions
  `left+jump` / `right+jump` (tap the turn until taken, wait for the rotation
  to start, press B until the game buffers it). v4/v3 stay bit-identical
  (seeded random play, with and without demos, vs the committed code). Random
  v5 play: every turn+jump not cut short by a bumper turned and then jumped
  (643/643). Run **`d4`** defined, NOT STARTED: stages 1-4, one pinned demo
  each, env v5, d3's hyperparameters; stages 5-7 held out for the zero-shot
  test. Launch only after d3's gate and Tyler's go.
- **Turn-window fix (v4 AND v5), found building v5:** the game's turn window
  is where `(position & 0xFF) < 32` along the travel — the 32 units AFTER a
  corner's centre moving up an axis (S, E), the 32 units BEFORE it moving
  down (N, W) (tap sweeps in both directions). v4's decision rule only asked
  "before the centre", so a cell reached by a landing or bounce 0-31 units
  above a corner while moving N/W could be offered too late, and the turn
  fired one corner late. The press comes on the NEXT frame, so moving down an
  axis the cell needs at least one frame's travel above the centre (16 units
  at base speed: Tyler's stage-4 landing 16 above, then a right turn, is
  fine; random play's landing 8 above was not); otherwise it is skipped
  (straight). A first fix demanded 32 units and wrongly skipped Tyler's turn.
  Random v4 play: 3 of 2,787 choices (0.11%) were late under the old rule.
- **Jump needs a fresh press (v4 AND v5):** with B still held from the frame
  before, pressing it does nothing. Found at stage 4's start point 40 (Tyler
  held B through a landing in his hop-hop; the env's jump did nothing and
  Sonic ran into red). A press whose button was down on the previous frame
  now gets one released frame first. Only that 1 of 329 demo start points
  began mid-press (stage 1's 65: none); in play it can follow a very short
  bumper-bounce step.
- `smoke_corners.py` now matches human and env corners BY CELL, names moves
  an env lacks, and checks every demo's own start points. Final state:
  `smoke_corners.py d3` (v4) and `d4` (v5) ALL CHECKS PASS — all four demos
  replay to their clears from a fresh start and from all 329 start points.
  v3 still bit-identical. v4 changed on purpose (window rule, fresh press).
- Earlier: **paused by Tyler 2026-10-01 17:33.** `d2` stopped
  cleanly at **4,050,816 steps**, start point **79%** (ptr 77). That day's
  session (resumed 14:25 at 2.06M): 82% held ~1.0M steps (1.69M→2.72M), then
  81% at 2.72M, 80% at 3.36M, 79% at 3.40M and still 79% at 4.05M — pace slowed
  from ~75K steps per point (97→82%) to ~650K. Match rate from the 81–84%
  starts was 3–7% vs the 25% (8 of the last 32) needed to move. Fresh starts:
  mean spheres 21.6 in the last bucket (best 69 overall), 0 clears. At ~650K
  per point, 79%→0% is ~50M more steps, right at the pre-committed cutoff;
  check again at ~6–7M before calling it a stall. ~176 steps/s while Vanguard
  Galaxy held the GPU at 82%.
- Before that: `d2` (special stage 1
  from Tyler's flawless clear, `demos/stage1_20260930-165817.npz`, on **env
  v3**: 4-frame choices + the stall rule) was resumed 2026-09-30 21:10 at 303K
  and stopped cleanly at **2,060,008 steps**. The first start point FELL at
  ~614K steps (the milestone d1 never reached); start point now **82%**
  (ptr 80). Pace: 97→88% in ~150K steps, then ~1 point per 150K. The last
  bucket (start 84%) matched only 2% vs 10–13% before — watch for a stall
  there. Fresh starts: mean spheres 1.9 → 17.4, best 41/102, 0 clears.
  Resume: `nohup python3 train_ss_multistage.py d2 --resume >>
  logs/ss_multistage_d2.log 2>&1 &`, read with `python3 demo_progress.py
  --run d2`. Speed fell 270 → 160 steps/s while a game was using the GPU.
  See "Env v3 — d1's failure" below.
- **The test of the demo approach is pre-committed** — see "Pre-Committed
  Criteria": each run's first fresh-start clear, within 50M steps.

### What to expect / what to watch
14 boards split episodes 14 ways, so per-stage numbers come SLOWLY.
**Don't over-read early results** — wait for 20,000+ episodes, ideally 40,000,
before trend arrows mean anything. Key metrics: `ratio(min/max)` (near 1.0 =
uniform competence across boards), `stages covered: N/14`, and
`X climbing / Y flat / Z regressing`.

### Solved problems — DO NOT re-derive these
1. **Win detection** — board-clear latch (`blue_spheres_remaining == 0`), not the
   laggy emerald counter. Completion reward now actually pays out.
2. **Warp to any zone** — set zone/act → game mode `0xFFF600` = `0x0C` →
   pit-death (`y` = 60000). Semi-manual form is the reliable one.
3. **Grant 7 Chaos Emeralds** — count at `0xFFFFB0` = 7 **AND** the per-stage
   array `0xFFFFB2`–`0xFFFFB8` each = 1. **Count alone does NOT work.**
4. **Addressing** — full `0xFFFFxx` via `env.data.set_variable(...)` then
   `set_value`/`lookup_value`. **Never** `env.get_ram()[addr & 0xFFFF]` — that
   raw indexing reads wrong bytes and caused multiple false trails.
5. **All 14 stages captured** and backed up in `states/backup_captures/`.
   No further capture work needed.
6. **Board layout is `$F100–$F4FF`** (32x32, row-major). A position's cell is
   `row=((y+$80)>>8)%32, col=((x+$80)>>8)%32` — ROUNDED, because a sphere is
   collected as Sonic crosses its half-cell boundary. The old "buffer $F000,
   row=(y+8)%32" was only right for board rows 0–23 (see 2026-09-29).
7. **`$E433` is the exit-animation timer**, not a fade flag: it starts on the
   frame Sonic touches red, or ~347 frames after a board clear (the frame the
   emerald is awarded). `special_stage` only drops to 0 ~650 (death) / ~1000
   (clear) frames later.
8. **Cell code 5 = yellow spring** (safe; launches Sonic over several cells;
   `$E432` reads `$81` during a spring launch, `$80` during a normal jump).
9. **`ss_next_corner`**: +4 = left turn queued, −4 = right (data.json had it
   backwards). Heading `$E426`: `$00`=N `$40`=W `$80`=S `$C0`=E; left raises it.

### The open research question
The v2 5-board run **memorized the training set** (zero-shot on held-out boards
= 0.09). Position-drop forces generalization at 2 boards but not at 5. Tyler
decided memorize-all-14 is an acceptable *product* — the game has exactly 14 fixed
stages, so a perfect memorizer is behaviorally identical to a generalist here —
hence no holdout in v3. **But the generalization question remains open and
interesting.** Untested hypothesis: a *smaller* network (less capacity to
memorize) might force a general skill. If a proper generalization test is ever
wanted, S&K's standalone Blue Spheres mode procedurally generates effectively
unlimited boards — a far stronger test set than holding out 2 of 14.

### Likely next steps
- Record a stage-1 demo and run `d1`; read `demo_progress.py --run d1`. The
  question: fresh-start clears of stage 1 (the specialist never got one in
  326M steps), and play that beats the demo.
- Periodically read `ms_analyze.py`; judge whether all 14 climb together.
- If boards stall near the floor → board-specific difficulty; investigate what
  they share (v2 had this with stage 7).
- Possible follow-ups: smaller-network capacity experiment; handoff integration
  with the main-game agent; eventually a main-game (non-special-stage) generalist —
  a much harder problem Tyler has been thinking ahead about.

### ⧉ OBSERVATION AUDIT — items 1 & 2 SHIPPED in v4; item 3 answered 2026-09-29
**Item 3 answer:** the grid did NOT encode jumpers distinctly. Springs (code 5)
fell into ch1 alongside red, so every spring looked like a death cell (292
spring cells on boards 8–14, 171 on stage 9). Env v2 gives them ch4.

Evidence found 2026-07-26 that the observation stack carries fat. Do these ONE
AT A TIME on a fresh run, measuring each — never bundle (confounds attribution),
never mid-experiment (changing obs shape breaks resume + confounds the cool).

**STATUS: v4 shipped items 1 and 2 TOGETHER** (dead RAM features dropped AND
pixels dropped AND patch widened, in one change) — which violates the one-at-a-
time rule above and means a v4-vs-v3 difference cannot be attributed to any
single one of them. Worth knowing when reading the v4/v5 numbers. Item 3 (jumper
lookahead) is untouched. All three are now one-line config changes
(`SS_USE_PIXELS`, `SS_PATCH_R`) rather than a forked env file, so a clean
one-variable ablation is cheap to run properly.

The observation is a 3-stream dict (line ~486 of the env):
`{'pixels': 84x84x3, 'ram': 15 features, 'grid': 11x11x4 heading-rotated patch}`.

1. **4 DEAD RAM features (confirmed).** RAM features 2,3,4,5 = `sx,cx,sy,cy` =
   sin/cos of X,Y position. Under `SS_DROP_POSITION=1` the env hardcodes them to
   `0.0` (env line ~462). So on the v3 run they are **constant zeros** — 4 dead
   inputs the net must learn to ignore. Trivial fix on next fresh run: drop them,
   set `NUM_RAM_FEATURES = 11`. Small but clean win.
2. **Pixels vs. grid patch (THE big suspect — ablate).** Both are spatial views
   of the same board. Pixels are ~21k input values (84x84x3), dwarfing everything.
   If the grid patch already gives what the agent needs spatially, pixels may be a
   large, expensive, mostly-redundant input eating capacity — directly relevant to
   the shallow-plateau/capacity concern. Test: does the agent learn as well or
   better on **grid-patch-only** (drop pixels)? Highest-value ablation. If pixels
   are redundant, removing them frees real capacity.
3. **Jumper/bumper perception is REACTIVE, not spatial.** RAM features 12,13,14 =
   `bumper_lock, bumper_hit, jumping` — these are STATE flags ("am I hitting/
   jumping NOW"), not lookahead ("a jumper is N cells ahead"). So the agent isn't
   blind to jumpers, but only learns about one *after committing*. If jumpers prove
   a wall (watch per-board plateaus on bounce-heavy boards): first check whether
   the **grid patch already encodes jumper cells distinctly** (then spatial
   perception exists, net just needs to use it); only if not, add a lookahead
   feature. This reframes the earlier "add jumper perception" idea — the fix is
   making the EXISTING signal anticipatory, not adding perception from scratch.

Principle throughout: remove/add features in response to demonstrated need, not
anticipation. Each unmotivated input is unpaid-down complexity that costs
capacity AND interpretability. Ablation (drop a component, measure) is the only
honest way to know what's load-bearing — intuition can't tell you.


### Knowledge base contents (keep current!)
`sonic_rl_context.md` (this file) · `special_stage_env.py` · `data.json`.
**These go stale** — when the env or context changes materially, re-upload, or a
fresh chat will confidently reason from an old version.
(`COMMANDS_v3.md` is retired: the launch/stop/resume commands now live in the
docstring of `train_ss_multistage.py`, which cannot drift away from the code.)

---

## Base game rebuild 2026-09-30 — learn from Tyler's playthrough, then beat it

**Goal (Tyler's words):** get the base-game coordinator set up properly; use a
recording of his own playthrough as the agent's starting point; have it
improve on his play from there.

### What was wrong with the old main game (each checked in the emulator)
1. **`coordinator.py` could not run.** `SS_STREAK_MULTIPLIER` and
   `STUCK_SECONDS` were undefined — the stuck check raises NameError on the
   first frame Sonic doesn't move, i.e. the first step — and every path was
   `~/sonic-rl`. So the v10 model was trained by a version of the file that no
   longer exists, by a trainer that no longer exists.
2. **The act-clear trigger read an unnamed byte.** data.json's
   `level_end_bonus` ($FAA8) is `_unkFAA8` in the disassembly (set by some
   bosses' end sequences only). It never marks a signpost clear.
3. **Death was seen late, and episodes were 3 lives long.** It keyed on the
   lives counter, which drops ~80 frames after the fatal frame; the level
   reloads ~290 frames after. The scenario's `done` is lives == 0, so every
   episode ran through all three lives.
4. **Progress was normalized by `screen_x_end`** = `Camera_max_X_pos`, the
   camera's *current* right limit ($6000 in AIZ1 until an event locks it), not
   the level's end.
5. **Two more mislabels in data.json:** `shield` ($FE47) is
   `Saved_status_secondary`, the starpost copy; `special_stage` ($FE48) is
   `Special_bonus_entry_flag` — 1 = special stage, **2 = bonus stage**, so the
   old `== 1` check missed bonus stages. (The SS env's use of it is fine.)
6. **No spindash rev:** the old action 8 was B+**UP** (button index 4) under a
   "B + down (spin dash charge)" comment. Launch speed 2048 unrevved vs 2944
   after 6 revs.
7. **v10** (`sonic_ppo_v10_emerald_priority.zip`, PPO CNN, 4x84x84, 9 actions,
   210M steps): 3 of 3 stochastic 20k-frame episodes stall at x≈6765 — the
   first ledge of AIZ1 — and never die. The ledge is cleared by a plain full
   jump (tested), so the action-table bug is NOT the cause. Kept as a record;
   not worth warm-starting.

### RAM facts (skdisasm constants resolved to addresses; checked in the emulator)
The disassembly (github.com/sonicretro/skdisasm) names every RAM variable, and
its `loc_XXXXX` labels are ROM addresses of this exact ROM (`Obj_Sonic` =
$10A94, matching Sonic's code pointer). A parser over `sonic3k.constants.asm`
reproduces every address we already trusted ($B000, $EE78, $F600, $FE10,
$FE20, $FFB0), so the rest can be read off it.

| what | address | name | notes |
|---|---|---|---|
| player routine | $B005 | Player_1+routine | 2 control, 4 hurt, **6 dying, 8 restarting, $C drowned** |
| player status | $B02A | +status | bit 1 in air, bit 2 rolling, bit 5 pushing |
| spindash | $B03D | +spin_dash_flag | 1 while revving |
| game mode | $F600 | Game_mode | $0C level, $8C loading, $34 special stage, $48 SS results |
| timer running | $FE1E | Update_HUD_timer | **0 = timer stopped** (results, capsule, death fall, SS entry) |
| in-game time | $FE23/24/25 | Timer min/sec/frame | holds the act's time at the clear |
| stage entry | $FE48 | Special_bonus_entry_flag | 1 special, 2 bonus; cleared on return |
| boss fight | $F7AA | Boss_flag | minibosses too |
| title-card act | $EE4F | Apparent_act | internal `act` ($FE11) can differ: AIZ switches to act 2 mid-act with x −$2F00 |
| signpost | $FAA6 | Signpost_addr | Obj_EndSign = ROM $837B2; needs bit 0 of Slotted_object_bits ($FA9A) |

**Act end, measured** (a real signpost spawned into AIZ1): lands, $40 frames,
then on the first frame Sonic is on the ground the results object starts,
`Update_HUD_timer` → 0 and the timer freezes on the act time; controls lock;
the tally runs ~1000 frames; `Apparent_act` → 1; the act-2 timer restarts from
0:00. So **"the timer stops with Sonic alive in the level" is the act clear.**
Act 2 ends at the capsule, whose opening clears the same flag per the
disassembly (`loc_7403A`) — not yet seen in the emulator; the first recording
through an act 2 will show it (`ActSplitter` would report no clear). The rule's
one known false positive: a cutscene that stops the timer mid-act (the
disassembly has one in Hidden Palace, the Knuckles grab) reads as a clear —
check each new act's recorded clear frame/time against what you saw.

### The design (`coordinator.py`, `record_play.py`, `train_main.py`)
- **Episode = one act**, from a start point along a recording, ending on the
  **clear** (terminated, +600 frames), **death** (routine 6/8/$C, on that
  frame; terminated, −1800 frames ≈ a starpost restart), **special/bonus
  stage entry** (truncated — neutral; the in-game timer doesn't run there and
  the SS agent will take these over later), **stuck** — 600 frames (10 s)
  with no new progress along the route, boss fights excepted (truncated;
  added after aiz1's first stall, below), or **timeout** at 2x the
  recording's time from that start + 20 s (truncated).
- **Reward = frames faster than the recording, along its own route.**
  `RouteProgress` maps Sonic's position to the nearest point of the recorded
  route (same internal zone/act, within 96 px, searched −300/+600 demo frames
  around the last match; after 8 misses the whole remaining route). Each step
  pays the new best demo frame reached (a ratchet) minus the frames the step
  took. The undiscounted return of a clear is therefore exactly
  `(recording's frames to the clear − agent's frames) + 600` — "improve on
  Tyler" is the objective, not a proxy for it. A route he didn't take is free
  to win: off the route progress holds, and jumps when the agent rejoins.
  Boss arenas get the whole route stacked on one screen, so while the boss
  flag is up progress holds at his fight start and jumps to his fight end when
  it drops: a faster kill is worth exactly the frames it saves.
- **Backward curriculum** (Salimans & Chen 2018, as in the SS `d1` design): a
  start point every 60 demo frames; the pointer starts at the end of each act
  and moves one back when ≥25% of the last 16–32 episodes from it cleared
  within 1.25x his time + 1 s. 25% of episodes always start at the act's
  beginning (the honest measure).
- **Actions (9, held 4 frames):** noop, left, right, down, up, jump,
  jump+left, jump+right, jump+down. Down+left/right is left out on purpose:
  `SonicKnux_Roll` ignores down while left/right is held (tested: down+right
  at speed doesn't roll). The recorder maps held keys the same way the game
  reads them, except while revving a spindash, where down keeps the charge.
- **Observation:** 4 stacked 84x84 grayscale frames, each the max of its
  step's last two emulator frames. Pixels only for now — add features on
  demonstrated need.
- **Recordings** (`demos/play_<stamp>.npz`): inputs only, plus the action
  table itself (not just a hash — the replay always uses the table it was
  recorded with), the start state by name + SHA1, and every act's start/clear
  frame and time, which the trainer re-verifies by replay before launching.
  One file is a whole playthrough across sessions (`--resume`, instant via a
  local cache). Rewind points thin to 1 per 10 s beyond the last minute.
- Integration states are **Sonic & Tails** (AIZ1, HCZ1, MGZ1) — except
  AngelIslandZone.Act2, which is Sonic alone on a title card. A playthrough
  from AIZ1 makes its own act-2 start.

### Verified
- `smoke_main.py` (7 checks, ~7 s): RAM map at the AIZ1 start; death seen on
  the fatal frame with lives still 3 (they drop 80 frames later); the
  down+left/right premise; a scripted mini-act (AIZ1 to a test signpost at
  x=7100, `MG_TEST_GOAL_X`) clears on the timer-stop frame and replays; **from
  all 11 start points, the demo's own inputs clear on its exact frame and
  return exactly +600** (start states, observation stacks and reward
  accounting agree); standing still is cut as `stuck` after ~600 frames;
  the recorder round-trips with a rewind, a cached resume, and a rewind past
  the point the resumed session started from.
- Beating the script scores as it should: jumping the ledge sooner cleared
  25 / 89 frames faster → returns 625 / 689.
- **Special stage inside an act** (failing stage 1 hands-off from
  `curriculum_stage_1.state`, which was entered from AIZ1 at 0:19): 551
  frames in the stage, results (gm $48; timer flag cleared at 612), reload
  (gm $8C from 1107; entry flag cleared at 1210 with Sonic put back at the
  ring, x 7112 y 1216, timer restored), level + timer at 1260. No frame meets
  the clear condition, and only the ~50 loading frames after the flag clears
  count as act time. Ring *entry* is not tested in the emulator (teleporting
  onto the ring crashed the game to the title screen); per the disassembly
  (`loc_90926`) the flag and game mode are set on the same frame, before the
  timer is touched. A pit death mid-act and its reload also leave the act
  open (no false clear).
- **Tyler's first recording** (`play_20260930-153946`): AIZ1 1:25.43, 0
  deaths, 0 rewinds. In S3&K AIZ1 the timer stops at the **miniboss kill**
  (frame 5060); the signpost only spawns 184 frames later, results after
  that — so the clear is the kill, which is when the act time is locked in.
  Boss flag: 2459–3253 (the jungle-burning flyover, cleared by the internal
  act switch at 3253 with x 12249 → 217) and 4476–5244 (the miniboss). From
  every sampled start point his own inputs clear on his frame; route progress
  never leads him and lags ≤ 40 frames outside fights.
- **aiz1's first stall (4.3M steps): the start point sat at 83% for 3.5M
  steps.** Cause: a wall just before the miniboss (x ≈ 3830, internal act 1)
  that is climbed by going BACK — up-left onto a ledge (y 838), then up-right
  over the top (y 740 → 766). Tyler's route there includes a failed direct
  jump (he fell back onto the rock pile, hesitated ~1 s, then took the ledge);
  that's harmless for the reward (it happens on the rock pile where the ledge
  route starts). The agent reached the ledge in sampled play but not the
  second jump. Entropy was NOT collapsed (−1.94 of a max 2.2). The real waste:
  **84% of all frames trained on were 47-second timeouts** (median 2800
  frames), mostly pressed against that wall; clears were 4%. Fix: `stuck`
  truncation after STUCK_FRAMES = 600 frames with no new route progress
  (boss fights excepted; the demo's own inputs never lag its route by > 40
  frames). Resumed at 4,346,400 steps with the pointer kept; PAUSED at
  4,998,008 steps (pointer 69 = 82%, one past the wall) so the special-stage
  run d1 gets the machine. Resume: `nohup python3 train_main.py aiz1 --resume
  >> logs/main_aiz1.log 2>&1 &`.
- Recorder bugs found on first real use, fixed: a cached `--resume` kept no
  rewind points (Backspace couldn't go back past where the session ended),
  and loading a state leaves the emulator's screen black until a frame runs,
  so the recorder showed black after every rewind and at startup. Fixed by
  caching the rewind points and drawing a one-frame peek (state restored;
  determinism checked). `record_demo.py` (special stages) had the same black
  screen — fixed too.
- Recording advice given: record the time run WITHOUT giant rings or starpost
  bonus stars. The reward is time, so a ring detour only lowers the bar (the
  agent skips it); emeralds need their own reward decision and, ideally, the
  env playing the stage through instead of truncating (an agent following a
  with-rings route into the ring gets cut short, and the curriculum counts it
  as a miss).
- **Smoke run** (`train_main.py smoke`, the mini-act, 9 min, 536K steps): the
  start point walked 92% → 0% by 287K steps. From the act's start, clear rate
  19.5% → 100% and the median margin −1270 → **+20 frames** over the log; it
  beat the script in 67% of clears in the last sixth, best +115 frames (1.9 s).
  So the whole loop — record, replay, curriculum, reward — produces an agent
  that out-plays its recording.

- **`audit_play.py` (2026-10-10, for Tyler's full-game recording):** run after each recording session. It replays the playthrough (must reproduce every act) and prints per act: start/clear frame, own time, the in-game reading at the clear, the in-game time elapsed since the act began, deaths, stages and boss fights. It flags: an act that never cleared with another after it; clears under 15 s; an act 2 cleared with no boss fight before it (except the known LRZ2 / SOZ2; every clear is the boss's last hit, see below); a clear >30 s after the last boss fight (except SSZ1's Mecha Sonic); an act resumed after a cutscene timer stop other than the known LBZ2b; acts out of the game's order; own count vs in-game elapsed off by >2 s with no death or stage. On play_20260930-153946: nothing flagged. AIZ1's start state already has 0:01.15 on the clock, and own 1:24.33 vs elapsed 1:24.18 is 9 lag frames.

### Game sound in the recorder (2026-10-10)
Tyler asked whether audio could work while he plays; it never had. stable-retro does expose it: `em.get_audio()` gives each frame's samples (int16 stereo, 735-736 per frame) and `em.get_audio_rate()` = 44,100. Checked on AIZ1: sound on 240 of 240 frames. `game_audio.py` streams them through SDL (pygame._sdl2 AudioDevice, a callback reading a short buffer: primed at ~2 frames, capped at 0.1 s) to the system's default output. pygame 2.6.1 won't take None for "default", so the device is looked up from `pactl get-default-sink`; it was the HDMI output on 2026-10-10, and the headset is "Built-in Audio Analog Stereo". record_play.py: sound plays at 100% speed only (quiet in slow motion, paused or rewinding), M toggles it, `--no-sound` starts muted, and a missing device just means no sound. At 100% the sound card paces the loop (pygame's Clock.tick(60) ran ~61 fps, overfilling the buffer and dropping a sliver about once a second): silent test 60.15 fps, buffer peak 67 ms, 0 dropped. Output only: recordings and replays unchanged; smoke_main.py passes. record_demo.py (special stages) can use the same module.

### MG env v2 (2026-10-10) — his play is a guide, not a rail
**Tyler, 2026-10-10:** "My play should stay as a guide. I don't want to hinder innovation. I'd like the agent to be rewarded for clearing the level quickly as well." The reward already pays clear time: per step, route progress at his pace minus frames, so a clear's undiscounted return is exactly (his frames − the agent's) + 600, and each frame saved is worth a full frame (vs the 600 clear bonus / 1,800 death penalty; much stronger than the SS bonus). No extra speed term was added. Two things tied the agent to his route; `MG_ENV_VERSION=2` (coordinator.py; default 1 = the 09-30 env, which smoke and aiz1 keep) changes them:
1. **Stuck rule:** v1 cut any episode after 600 frames without new progress ALONG HIS ROUTE, so an alternate route leaving his for >10 s was cut. v2 also counts new ground (a 128x128-px block not yet visited this episode) as progress. Standing still or being pinned against a wall is still cut.
2. **Off-route credit timing:** v1 charged −1 per frame off his route at once and paid the progress only on rejoining / at the clear, beyond gamma 0.995's ~13 s horizon. v2 books off-route frames (outside boss fights) and settles them on the first step back on the route, together with the progress rejoining pays, or at the episode's end. The net lands as one number. Every episode's undiscounted return is unchanged.
**Verified (`check_mg_v2.py`, ALL CHECKS PASS):** v1 = git HEAD on every trajectory (seeded random play from 8 start points of the real AIZ1, hold-left and stand-still from 6, the synthetic mini-act's invariant); on v2 the mini-act's own inputs still clear on its frame with return exactly 600 from all 11 start points; on v2 every one of 31 episodes' returns equals the v1 accounting of the same steps (2,993 off-route steps paid 0 until settled); standing still is still cut (616-768 frames; two start points kill a standing Sonic at 555 / 197 frames on every version); hold-left lasts >= v1 from every start (e.g. 1,256 vs 660 frames); random play survives longer before `stuck` (e.g. 3,144 vs 1,912 frames). `smoke_main.py` ALL CHECKS PASSED (v1 default). train_main.py: runs take `env_version` (default 1); new main-game runs should set 2. **aiz1 pinned to play_20260930-153946** (it used 'play_*', which would have pulled Tyler's coming full-game recording into a resume).

### Tyler's full playthrough (2026-10-10) — every act, one session
**`demos/play_20261010-202923.npz`**: AIZ1 to the ending in one sitting with game sound (~20:35-22:31), 424,851 frames (1:58:01), 13 rewinds, no special stages or bonus stages (giant rings skipped, as advised for a time run), no deaths left in the timeline. **27 cleared acts**, act time 105:41.90 by the recording's own count. The recorder's replay check passed. The file as recorded is commit 77b6432 (pushed); only its act NAMES were rewritten afterwards (below).
**Act names were wrong for the late game (fixed 7a1ad4c; screenshots checked):** internal zone $16 act 0 = Lava Reef's boss arena (the timer carries on from LRZ2) → **LRZ3**; $16 act 1 = Hidden Palace → **HPZ1** (was HPZ2); $17 act 0 = the final Death Egg boss (timer carries on from DEZ2) → **DEZ3**; both used to be 'HPZ1'. $0D = the ending → END. **LBZ2's timer stops at its boss and runs again for the Big Arm fight** after the Death Egg launch cutscene, which made a second 'LBZ2'; `coordinator.part_name` names a same-act restart right after a clear **LBZ2b**. Training keys acts by name, so these would have merged different levels. `attic/one_off_tools/relabel_play_acts.py` rewrote the names in the file's metadata after checking the replay matches every act in everything but the name (inputs untouched).
**Every act clear is the boss's last hit** — act 1 (miniboss) and act 2 alike, seen in screenshots at every clear: the signpost and the capsule come after, on a stopped clock (the header used to say "results after the signpost (act 1) or the capsule (act 2)"). So an act's time is exactly what the agent is scored on.
**Boss fights the game never raises Boss_flag for:** MGZ1's and SOZ1's minibosses, the SOZ2 boss, SSZ1's Mecha Sonic, DEZ3 (LRZ2 has no boss). There only route progress keeps `stuck` away (flagged fights hold progress and pause `stuck`).
**Acts (own frames):** AIZ1 5,098 · AIZ2 9,866 · HCZ1 9,154 · HCZ2 16,829 · MGZ1 13,815 · MGZ2 19,140 · CNZ1 10,445 · CNZ2 22,257 · ICZ1 16,514 · ICZ2 10,611 · LBZ1 17,307 · LBZ2 17,184 · LBZ2b 1,275 · MHZ1 9,032 · MHZ2 11,549 · FBZ1 18,561 · FBZ2 17,442 · SOZ1 21,636 · SOZ2 23,667 · LRZ1 20,123 · LRZ2 16,321 · LRZ3 8,448 · HPZ1 3,470 · SSZ1 18,792 · DEZ1 14,169 · DEZ2 21,902 · DEZ3 5,907. `audit_play.py`: nothing flagged.
**Training cost:** all 27 acts build segments with unique keys: 6,339 start points (one per second), ~1 GB RSS and ~6.5 min of replay per worker on first reset (8 workers ≈ 8 GB; the box has 31 GB). Per-act runs (MG_ACTS) build only their acts.
**`check_play_acts.py <play.npz> [n]` — can the env train on every act?** Drives the real env with HIS per-frame inputs (the emulator call swapped for his recorded input, so the 4-frame action repeat doesn't interfere) from n evenly spaced start points per act; each episode must clear on his frame with return exactly CLEAR_BONUS. Negative control: his inputs one frame late → death / stuck within seconds. **On env v2 as first built: 9 of 162 cut as `stuck`** — AIZ2 x4 (the bombing run: the screen locks ~20 s while the ship flies over and he dodges in place), MHZ1 x1, SOZ1 x4. Where the demo stays in one place, many route points tie for nearest and argmin took the earliest, freezing progress. **Fix (4a2686f, in v2 since v2 had never trained): among route points within ROUTE_TIE = 16 px of the nearest, take the one closest to the last match + the frames since.** After: **ALL ACTS PASS** (162 episodes, 1.15M frames); check_mg_v2 ALL PASS (v1 identical to HEAD). Denser rerun, 20 start points per act: **ALL ACTS PASS** (540 episodes, 3.82M frames, 46 min at nice 15 beside d9). Logs: attic/one_off_tools/check_play_acts_{before,after}_tiefix.log, ..._after_tiefix_n20.log. Run it on every new recording.

### Deliberately open
- Special stages are neutral truncations. The coordinator's eventual job is to
  hand giant rings to the SS agent (emeralds as a reward is a decision for
  then: it trades against act time).
- The death penalty (1800 frames) sets the risk appetite: a shortcut is worth
  it when it saves more than P(death) x 30 s. Revisit once there's play to read.
- `train_main.py` has its own copies of the SS trainer's small callbacks
  (GracefulStop, AutoSave, Prune): the SS trainer parses args at import and
  can't be imported. Extract a shared module if a third trainer appears.

---

## What Changed 2026-09-29 — logic-error review → env v2 (branch `env-v2-fixes`)

A full read of the code plus emulator probes against the 14 captured states.
Every number below was measured, not inferred. Items 2–6 shaped the training
signal of every run so far: the specialist and v1–v5 results above were all
produced on a partly blind, late-terminating env.

1. **The repo move broke every path, silently.** The project moved
   `~/sonic-rl` → `~/Projects/sonic-rl` (2026-08-25) and ~20 hardcoded paths did
   not follow. A missing stage file fell through to the game's default start
   (Angel Island Act 1): 11-frame "deaths" at −2500 and no episode log (its
   directory didn't exist). `train_ss_multistage.py v5 --resume` would have
   trained on that and autosaved over v5. **Fixed:** paths are repo-relative, a
   missing state file raises, the trainer refuses to launch without every stage
   file, and a failing log write warns.
2. **The grid read the wrong memory for 8 of 32 rows.** The layout is at
   `$F100–$F4FF` (its blue count equals the game's counter on all 14 boards).
   The env read `$F000–$F3FF` with `row=(y+8)%32` — right for board rows 0–23,
   but rows 24–31 wrapped onto always-zero RAM. 284 of 1,328 blue spheres (39%
   on stage 1) and whole red walls were invisible, and boards 2, 10, 11, 14
   start Sonic inside the blind rows. With pixels and position dropped, v4/v5
   had no other way to see them. (Hidden fraction vs v5 %board correlates only
   +0.11, so how much of the plateau this explains is for v6 to show.)
3. **Deaths registered ~11 s late.** The episode ended when `special_stage`
   dropped, 651–678 frames after red contact (~1000 after a clear). **~44% of all
   v4/v5 env steps were dead animation**, and the death penalty reached the fatal
   action discounted by γ^675: 0.13 at γ=0.997 and **0.001 at 0.99**, so during
   the specialist's long γ=0.99 phase the explicit death penalty was effectively
   invisible. That is probably the real reason the 0.99→0.997 change helped
   (~100× more visible penalty), more than the lookahead story.
4. **Springs looked exactly like red** (code 5 in ch1). See audit item 3.
5. **The emerald bonus never paid.** The emerald count rises ~347 frames after
   the clear; v1 compared it with the previous frame's count only on the final
   frame, ~660 frames later. All logged wins say `cleared`.
6. **The patch centre floored instead of rounding.** Pickup happens at the
   half-cell boundary (x = `..80`), so the centre sat a cell behind on S/E
   headings and a cell ahead on N/W for half of every cell — breaking the "same
   situation → same patch" invariance the pixel drop relied on. The heading
   quadrant had the same flaw (a right turn rotated the patch at its start, a
   left turn at its end).
7. Smaller: `_maybe_capture` assumed 102 spheres (the 37-sphere board filed four
   junk snapshots on its first step, even with restarts off, and snapshots
   carried no stage); feature 1 `speed/4096` saturated at base speed (max is
   8192); feature 4 `total_progress` is exactly `1 − feature 0`;
   `eval_transfer --max-steps 6000` cut off 45% of the specialist's 60+-sphere
   runs; `ms_analyze.py --chunks 8 …` took 8 as the window; rolling checkpoints
   were every 800K steps, not the documented 100K.

**What env v2 changes** (`SS_ENV_VERSION=2`, the default): grid from `$F100`;
rounded cell and heading quadrant; 5-channel grid (ch4 = spring); the episode
ends on the red-contact frame or the clear frame (a clear pays completion + time
bonus + its emerald: chaos for stages 1–7, super for 8–14); speed feature /8192;
`total_progress` dropped (10 RAM features). **`SS_ENV_VERSION=1` reproduces the
old env bit-for-bit** (checked over 8,690 steps under both the v5 and the
specialist presets: identical observations, rewards, dones and logs), and every
pre-v6 preset pins it.

**Verified:** `smoke_stages.py v6` now also checks that the board's blue count
equals the game's counter and that the episode ends on the exact exit frame (all
14 pass). On every pickup the sphere was directly ahead in the patch one frame
earlier, on all four headings. One frame before a death the patch shows the red
cell ahead (env v1 showed empty floor). A 4-minute v6 training smoke ran clean
(median episode 380 steps, vs 1,102 for v5's first 104 episodes).

**Still open — deliberately not in v6 (reward design, its own experiment):**
`VecNormalize(clip_reward=10)` with the saved return std (~710) caps any raw
reward above ~7,100. The death penalty `2500 + 2000n` passes that at n ≥ 3 (31% of
v5 deaths; about a fifth of v5 episodes came out net positive), and in env v2 the
clear step (last sphere + completion + emerald) is capped too. So the "net −2500
death" invariant is not what PPO optimizes — and the invariant itself makes every
death's undiscounted return identical no matter how many spheres were collected.
Decide it on purpose: raise or remove the clip, pre-scale the rewards, or accept a
fixed death penalty.

---

## Learning from Tyler's play (demos) — branch `demo-learning`

**Goal:** the agent learns from Tyler's recorded runs AND improves on them, so it
never copies his inputs. It only starts episodes from states his runs passed
through; the game's reward decides what good play is. His routing gets reused
where it works, his mistakes don't get learned, and it can find better routes.

**Recording:** `python3 record_demo.py <stage>` (50% speed by default; `--speed 25`
for hard boards; opens paused — any move key starts, since Sonic is already
running in the captured state). Arrows turn/advance, Z/X/C/Space jump, **Backspace rewinds ~2 s**
(hold to keep going), P pause, 1–4 set speed, Enter saves, R restarts, Esc quits;
a gamepad works too. He plays with exactly the agent's 7 actions, so every frame is
a move the agent has. Slow motion and rewind don't change what's recorded: the demo
is one clean full-speed input sequence. A run that dies is still useful. Saving
replays the inputs through the training env and confirms they reproduce the run.
Demos land in `demos/stageN_<timestamp>.npz` — inputs + metadata only, with the
start state pinned by SHA1 (no ROM data, so they're safe to commit; re-capturing
a stage makes its old demos refuse to load rather than replay wrong).

**How training uses them — the backward curriculum (Salimans & Chen 2018):** a
share of episodes (`demo_prob`) start from a checkpoint along a demo (one every
30 frames, lazily rebuilt in each worker by replaying the demo — nothing on
disk). The start point begins near the END of the demo and moves one checkpoint
toward the stage start whenever ≥25% of the last 16–32 episodes from it collected
at least as many spheres as the demo did from there (`DemoCurriculumCallback`).
At 0% the agent plays the stage from the start, measured against the whole run.
Verified: from any start point, replaying the demo's own remaining inputs
reproduces its ending to the frame, so episodes really start on the demo's states.

**Running it:** `python3 train_ss_multistage.py d1` — stage 1, demos `stage1_*`,
75% of episodes from demos, env v2, v6's hyperparameters. It refuses to launch
until a stage-1 demo exists, and re-verifies each demo before training. Progress:
`python3 demo_progress.py --run d1` (start point, match rate and sphere margin
vs. the demo, then fresh-start play vs. the whole run). The curriculum state
persists in `sonic_ss_multistage_d1_demos.json`, so `--resume` keeps it. More
boards: record them and add a RUNS entry with their `demos` patterns.

**Improving beyond the demo:** the bar to move on is only "match", but the reward
keeps paying for more spheres and faster clears; the margin column shows where the
agent beats the run. For boards Tyler can't finish, a run that dies still works —
episodes that start near his death point push the agent past it. In a 4-minute
smoke test on a scripted 8-sphere demo, the start point walked 93% → 78% and the
agent out-collected the demo from each start (margin +0.9 to +2.9).

**Env v3 — d1's failure (2026-09-30).** d1 (env v2, per-frame choices) ran
3.36M steps and never matched Tyler from its first start point (98/102, 4
to go; his finish = two straight cells, then two single right taps).
Measured: (1) every episode took the 99th sphere and then hit red — the
policy (entropy −1.69 of a max −1.95) pressed a turn on 53% of frames, and
one tap anywhere in a cell queues the turn for the next corner, so a
straight cell needed ~12 turn-free choices in a row; (2) it learned to hop
around the red board to postpone dying (median 752 frames alive after the
start point, where he needed 75) — a far-off death costs less after
discounting. **Env v3** (`SS_ENV_VERSION=3`): the agent chooses every 4
frames (FRAME_SKIP; demos still replay frame by frame), and 600 frames
without a sphere ends the episode as a death (STALL_FRAMES; his longest gap
was 257). Checked: v1 and v2 are bit-identical to the committed env over
4,003 / 6,007 seeded steps (trajectories and logs); under v3 his demo
replays and his own inputs from the last start point clear; the stall rule
fires at exactly 600 frames with red's penalty; all 14 boards read (straight
runs on boards 5 and 11 meet no sphere in 10 s and end by the rule). **d2** =
d1 on env v3, gamma 0.988 per 4-frame step (= d1's 0.997 per frame, the
same ~5.5 s horizon). `watch_ss.py` plays v3 at real speed.

**First real demo (2026-09-30):** `demos/stage1_20260930-165817.npz` — Tyler
cleared stage 1 first try: 102/102 spheres in 2925 frames (48.75 s), 0
rewinds, not a single jump (91% of frames he let Sonic run, turning the
rest). `d1` launched on it; its first start point is 98 collected / 4 to go,
where early episodes reliably take the 99th sphere and then hit red. The
curriculum state lives in `sonic_ss_multistage_d1_demos.json` (run output,
not committed). Pre-committed test: "Pre-Committed Criteria".

---

## Project Overview

Training a dedicated RL agent to complete Blue Spheres special stages in Sonic 3 & Knuckles using stable_retro. Immediate goal: complete Stage 1 (collect all 102 blue spheres) for handoff integration with the main game agent. Longer-term goal: generalize to all 14 special stages (the 7 Chaos Emerald stages from Sonic 3 + the 7 Super Emerald stages from S&K).

**Current status (2026-07-27):** The story so far, newest first.
- **v4 — observation redesign, LAUNCHED & training (fresh from random).** Tests whether a leaner, capacity-efficient observation gets 14 boards deeper than v3's pixel-heavy one. Changes vs v3: (1) **pixel stream DROPPED** — the 84x84x3 frame (~21k values, 99.99% of input volume) removed; (2) **grid patch WIDENED** PATCH_R 5→7 (11x11 → 15x15, clears the 5-cell jumper distance, keeps heading-rotation); (3) **4 dead RAM features removed** (sx,cx,sy,cy were constant 0 under DROP_POSITION) → NUM_RAM_FEATURES 15→11. Files: `special_stage_env_v4.py`, `train_ss_multistage_v4.py`, `smoke_v4.py`. Log: `ss_log_v4.jsonl`. Baseline to beat: v3's ~3.5% plateau.
- **v3 — 14-board run, DONE (capacity-suspected plateau).** Trained all 14, fresh, 512-LSTM. Plateaued SHALLOW at ~3.0-3.5% mean %board. LR-cooled 3e-4→1e-4 (clip_fraction 0.41→0.25, small gain 3.0→3.5, re-plateaued as clip crept back up). Did NOT do the 2nd cool — Tyler's call: cooling is a squeeze, not a fix for a shallow plateau; the small cool-gain pointed at a capacity/representation ceiling, not an optimization one. → rebuilt as v4.
- **v2 — 5-board run, DONE (memorization).** Zero-shot on held-out 5,6 = 0.09 → memorized the training set. Position-drop forces generalization at 2 boards, not 5.
- **Specialist (`sonic_ss_model_v2`, PAUSED — control).** Single-board, ~326M steps, got deep (96/102) via curriculum + LR cooling. Memorized its board. The baseline that makes the generalization findings legible.

### Why drop pixels — the conceptual rationale (not just "redundant")
Pixel vision is **representationally unstable**: when Sonic turns, the camera rotates and the ENTIRE pixel field transforms — the same board configuration produces a totally different pixel array per viewing angle. The network must then spend capacity learning rotational/positional *invariance* from scratch (re-deriving "a sphere is a sphere regardless of facing"). The **heading-rotated grid patch eliminates this by construction**: the same local configuration always produces the same patch regardless of absolute heading — invariance is built INTO the representation, not learned. So dropping pixels removes not just redundant volume but a whole *category of learning burden* (invariance-learning) that was competing for the capacity the 14 boards need. "Grid vision is concrete because it's invariant; the same input always means the same thing, which pixels can't guarantee once the camera moves." This is the core reason to expect v4's leaner obs to fit 14 boards deeper.

- **Specialist run (`sonic_ss_model_v2`, ~326M steps, PAUSED — baseline/control):** the single-stage Stage-1 agent. PROVED a single stage is winnable (fresh best 96/102 after the win-fix; `Wins from restarts` 0→3). Memorized its board. Preserved as control.
- **Step 1 — 2-stage probe (`sonic_ss_multistage_v1`, DONE):** dropping absolute position encoding lifted stage-1↔stage-2 transfer from **~0.06 → ~0.75**. Position encoding = the memorization handle, at 2 boards.
- **Step 2 — 5-stage run (`sonic_ss_multistage_v2`, DONE — negative result):** trained 1,2,3,4,7 / held out 5,6. Among-trained boards converged nicely (~0.76 ratio, S1 best ~24%), BUT **zero-shot on held-out 5,6 = 0.09 → the policy MEMORIZED the 5-board set, did not generalize.** Position-drop is insufficient at 5 boards; the network has capacity to memorize the set. Untested cheaper hypothesis: a smaller network might force generalization.
- **Back-seven capture: UNBLOCKED this session.** The warp + emerald-array recipes (see 2026-07-19 section) let us reach Mushroom Hill and grant true 7-emerald state; confirmed the Super Emerald special stages load. Decision: pursue all-14 (memorize-all-14 is an acceptable product for a fixed finite game). Next: recapture lost stage 2, capture the back-seven, run a 14-board trainer (train ~12 / hold out ~2).

---

## What Changed This Session (2026-07-19) — back-seven capture UNLOCKED (warp + emerald recipes)

Goal: capture the back-seven Super Emerald special stages to enable a 14-stage generalist run. Blocker: those stages require reaching Mushroom Hill Zone with 7 collected Chaos Emeralds, then entering a rainbow ring → Hidden Palace. Both sub-problems SOLVED this session via memory pokes. (Also: Step-2 5-stage result came back — see below.)

### CRITICAL addressing gotcha (cost us hours)
`data.json` addresses are FULL Genesis addresses `0xFFFFxx` (work-RAM base `0xFF0000`), e.g. `chaos_emeralds` = `0xFFFFB0`, `zone` = `0xFFFE10`, `act` = `0xFFFE11`, game mode = `0xFFF600`.
- **`env.data.set_variable('name', {'address':0xFFFFB0, 'type':'|u1'})` then `set_value`/`lookup_value`** is the ONLY reliable poke/read interface. Register any address this way.
- **Do NOT** index `env.get_ram()[addr & 0xFFFF]` and trust it — that raw indexing read garbage/wrong bytes all session and sent us down false trails (`$F600=0`, bogus emerald neighborhood). Always go through `set_variable`.

### WARP TO ANY ZONE (clean full load) — SOLVED
Poking `zone` alone is inert (game only reads it on a level *load*). Pit-death alone reloads the *current* level, skipping the VRAM art DMA → corrupted/garbled tiles. The working recipe forces a real load:
1. Set `zone` + `act`.
2. Set game mode `0xFFF600` = `0x0C` (commits the zone as current level; Sonic spawns corrupted-but-alive).
3. Trigger a pit-death: set `y` (`0xFFB014`) = 60000 → the reload now does a FULL load (art included) → clean level.
- **Reliability quirk:** the fully-scripted version (mode-poke + scripted death in one process) is flaky — Sonic sometimes never spawns. The RELIABLE method is semi-manual: script step 1-2 and save a "commit" state (`states/mh_commit.state`), then in the capture tool LOAD it, run into a hazard to die, and respawn clean. One manual death per zone; then savestate that clean state as a permanent waypoint.
- Zone IDs (from SCHG doc): 00 AIZ, 01 HCZ, 02 MGZ, 03 CNZ, 04 FBZ, 05 ICZ, 06 LBZ, **07 Mushroom Hill**, 08 SOZ, 09 LRZ, 0A SSZ, 0B DEZ, 0C Doomsday, **16 Hidden Palace, 17 Hidden Palace (entered from giant ring)**.

### GRANT ALL 7 CHAOS EMERALDS (true collection state) — SOLVED
The count byte alone is NOT enough. Hidden Palace pedestals read a per-stage collection ARRAY, not the count. Poking only `chaos_emeralds=7` opened the door (cutscene/transition fire on the count) but left pedestals EMPTY (array was zero → zero real emeralds to display). The true state needs BOTH:
- `chaos_emeralds` (`0xFFFFB0`) = 7
- **`0xFFFFB2`–`0xFFFFB8` array, each byte = 1** ("chaos emerald collected"). Values: 0=not done, 1=chaos collected, 2=super grayed, 3=super activated.
- (`0xFFFFB1` = super_emeralds count; `0xFFFFBB` = 0 runs S3 special stages / 1 runs S&K special stages; `0xFFFFBA` = "exchanged" flag that disables super transform.)
- **CONFIRMED WORKING:** set count=7 + array all-1, warp to Mushroom Hill, enter rainbow ring → Knuckles cutscene played, all 7 gray Super Emeralds spawned on pedestals, and a Super Emerald special stage loaded correctly. Back-seven capture is now unblocked.

### Source that cracked it
The SCHG (Sonic Community Hacking Guide) "RAM Editing" page — Main System Memory Locations. It names `$FFB0` (chaos count), `$FFB1` (super count), `$FFB2-$FFB8` (the finished-special-stage array), `$FFBA`, `$FFBB`. Also the Save-slot format: `$06-$07` word = 2-bits-per-emerald collection state in SRAM (the persistent copy). Level ID table gives all zone/act hex including Hidden Palace = 16/17.

### New capture tooling this session
- `capture_stages.py` gained: `G` (grant 7 chaos emeralds — NOTE: count only, does NOT set the $FFB2-B8 array; for full state use the script recipe), `V` (warp to zone=target index via pit-death), plus earlier `T`/`W`. **FOOTGUN:** the `S` save key writes `curriculum_stage_<target>.state` — it silently OVERWROTE `curriculum_stage_2.state` when the index sat at 2. Habit: bump index to 20+ for any non-capture work. Stage 2 (127-sphere board) was lost, needs recapture; other 6 captures backed up to `states/backup_captures/`.
- Discovery/warp scripts (in outputs): `zone_probe.py`, `find_reload_trigger.py`, `trace_reload.py`, `diff_load_paths.py`, `diff_art_region.py`, `check_stages.py`.

### Step-2 (5-stage) RESULT — memorization, not generalization
The v2 run (train 1,2,3,4,7 / hold out 5,6) trained up: four boards (1,2,3,4) converged to a tight ~15-24% band (S1 best ~24%), S7 the laggard climbing slowly (~6%). Among-trained transfer ratio looked healthy (~0.75-0.87). **BUT `zeroshot_test.py` on held-out 5,6 = 0.09** — the policy MEMORIZED the training set, did not generalize to unseen boards. Same signature as the specialist. Position-drop forces generalization at 2 boards but NOT at 5 (network has enough capacity to memorize the set). Open question: does a SMALLER network (less capacity to memorize) force generalization? That's a cheaper next experiment than the 14-board run — worth testing before committing to it. But decision was to push for 14 anyway (memorize-all-14 is a legitimate product: one agent, all fixed stages, no observable difference from a true generalist for this finite game).

---

## What Changed the Session Before That (2026-07-02) — Step 2: capture all 7, scale to 5 stages

### Captured the full front-seven (Chaos Emerald stages)
Using `capture_stages.py` booted from `curriculum_stage_3.state` (rings-entered counter baked in, so no full replay) + the fail-forward loop (die on each stage → land in main game → walk to next giant ring → counter advances → capture). Added an **auto-reload-on-death toggle (`T`, default OFF)** so death falls through to the main game instead of snapping back — the front-seven do NOT require winning (dying still advances the ring counter). Added a **`blue_spheres` HUD line** to confirm each capture is a distinct board. `check_stages.py` verified all 7 distinct (unique RAM fingerprints + blue counts):

| stage | blue | | stage | blue |
|---|---|---|---|---|
| 1 | 102 | | 5 | 58 |
| 2 | 127 | | 6 | 128 |
| 3 | 118 | | 7 | 60 |
| 4 | 115 | | | |

NOTE: sphere count ≠ difficulty. Small boards (5,7) may demand MORE precise navigation. Web search confirmed the real difficulty driver is the **speed ramp** (speed rises ~every 25s; late-stage high-speed navigation is the wall), and that difficulty rankings online are for the *back-seven Super Emerald* stages (not yet captured), not these.

### Back-seven require a different capture strategy (NOT yet done)
The back-seven (Super Emerald stages) only unlock after collecting all 7 Chaos Emeralds — meaning you must **genuinely CLEAR the front-seven, not fail them.** Fail-forward can't reach them. Deferred until the architecture is proven worth the manual-clear grind. Possible shortcut to test: poke `chaos_emeralds`=7 and see if the Super-Emerald rings become reachable without legit clears. Efficiency tip when doing it for real: savestate after each clear (main-game state at N emeralds) so a botched stage costs one retry, not the whole run.

### Step 2 launched — and a contamination bug caught + fixed
`train_ss_multistage_v2.py` (derived from v1 via sed): train on 1,2,3,4,7 / hold out 5,6, fresh from random, 50M nominal but open-ended. **First launch was contaminated:** a stale `sonic_ss_multistage_v2.zip` on disk (from a partial earlier attempt) caused the trainer to RESUME instead of start fresh — the per-stage curve showed stages 1,2 healthy then COLLAPSING when 3,4,7 entered the mix (looked like catastrophic interference; was actually a resumed 1M-step partial). Fixed by moving the stale checkpoint aside (`contaminated_v2_attempt/`) so `os.path.exists` returns False → fresh launch. **Footgun compounded:** sed replaced checkpoint *filenames* but left the log print strings saying "v1" — the misleading `"Resuming multistage v1"` message sent debugging down the wrong path. Fixed the strings. **Lesson: always confirm the startup line says `Starting FRESH ... (random weights)` before trusting a "fresh" run, and read the per-1000 `--curve`, not the cumulative ratio — the cumulative averaged over the dead run's history and hid the collapse.**

### Step 2 clean-run result so far (~16k+ fresh episodes into the clean run)
Generalization HOLDING at 5 boards. Per-stage avg (as % of each board), trending up every chunk:
- **S1 ~13% (17 recent), S2 ~8% (14), S3 ~8% (13)** — climbing together and CONVERGING (the generalization signature; memorization would diverge). S3 closed the gap to S2 across the last few reads.
- **S4 ~7% — PLATEAUED at ~7.5 across ~8 consecutive per-1000 chunks** while others climbed past it. Led early, now 2nd-hardest.
- **S7 ~2-4% — the floor**, crawling (0.6→2.4 across the run so far). Slowest by far. Matches the "small board = precise navigation = hard" hypothesis.
- Transfer ratio ~0.76 (same as the 2-stage run — generalization scaling cleanly for the cooperating boards).

**Read:** core test passing (3/5 boards prove generalization decisively). Secondary finding emerging: stages 4 and 7 resist — likely a shared board property the local-patch obs doesn't capture well. That "which boards resist and what they share" is the most interesting open question and would motivate Step 3 / the Option-B encoding decision. Depth is low everywhere (best 13%) but ON PACE for random-weights/5-board/position-blind — the specialist needed ~12.5k eps to hit avg 30 on ONE board WITH position.

### Specialist vs generalist framing (important for reading eventual results)
The generalist will almost certainly play each individual board WORSE than the 96-sphere specialist — that's the specialist/generalist tradeoff, not failure. "Stronger" = breadth/transfer, NOT peak depth on one memorized board. Success metric is the transfer ratio staying high as boards are added + the held-out zero-shot number, NOT raw per-board depth. A 14-stage agent at 60% each >> a 1-stage agent at 94%.

### New tools this session
- `check_stages.py` — load each captured state, report blue count + RAM fingerprint, flag duplicates.
- `train_ss_multistage_v2.py` — 5-stage trainer (train 1,2,3,4,7 / holdout 5,6; new ckpt names; log strings fixed to say v2).
- `zeroshot_test.py` — evaluates the HELD-OUT stages 5,6 vs the trained stages (true unseen-board transfer). DEFERRED until trained boards firm up (leaders in the 20s+).
- `capture_stages.py` — added `T` auto-reload toggle + `blue_spheres` HUD.

---

## What Changed the Session Before (2026-06-30) — win-fix → transfer probe → multistage validation

Three findings, in order. Each gated the next.

### 1. The win-detection bug (found + fixed) — completion reward was dark the entire run
`_compute_reward` decided win-vs-death **solely by whether the emerald counter incremented**, inside the `special==0` (left-stage) termination block. But a board-clear leaves the stage the same way a death does, and the emerald award LAGS several frames behind the board emptying (the fly-off animation) — and may not credit at all on warped-in restart states. So **every board-clear was misclassified as a death** (`death_at_sphere=N`, −2500 penalty), and the completion reward (`SS_COMPLETION_BASE + time_bonus`) **never paid out once across the entire ~300M-step run.** `success` was true in 0 of ~96k episodes.

**Fix (stage-agnostic, restart-safe):** key the win on the **board emptying** (`blue_spheres_remaining == 0`) instead of the laggy emerald. Latch `self._board_cleared = True` when `blue_spheres == 0 and self._spheres_collected > 0`; win condition became `if self._board_cleared or emerald_gained:`. Verified deterministically (`verify_win_fix.py`: synthetic 2→0 clear with emerald held flat → success=True, completion reward paid). Post-fix on the specialist: fresh best 93→96, `Wins from restarts` 0→3, critic re-fit cleanly (value_loss spikes to ~0.11 and recover, explained_variance stayed 0.97-0.99).

### 2. Zero-shot transfer probe — the specialist MEMORIZED Stage 1
`transfer_probe.py`: load the specialist, run fresh episodes (no training, CPU) on each stage. Result: **Stage 1 avg 34 / Stage 2 avg 2.2 / Stage 3 avg 3.0** — a ~15x gap, all stage-2/3 episodes dying by sphere 2-8. Confirmed NOT a probe artifact (diagnostic showed stage 2 reads blue=127, stage 3 blue=118, grids have real content). Conclusion: the Stage-1 policy learned a **memorized route**, not a general skill — and even red-avoidance didn't transfer, because avoidance was FUSED into the memorized route via the absolute position encoding rather than learned as a standalone reactive skill.

### 3. Multistage architecture probe — dropping position FORCES generalization (VALIDATED)
Hypothesis: the sin/cos absolute position encoding (ram features 2-5, full-resolution `ss_x`/`ss_y` fingerprint) is the memorization handle. Test (Option A, the clean diagnostic): zero those features, train fresh-from-scratch on stages 1+2 mixed 50/50, restart curriculum OFF.

**Result at ~16M steps (last 5000 labeled episodes):**
- Stage 1: avg 31.6 (31% of 102), Stage 2: avg 24.7 (19% of 127)
- **Transfer ratio ~0.75-0.78** (vs specialist's ~0.06). Both stages climbing together over the `--curve` trend.

Interpretation: **position was the memorization handle.** Removing it produced a >10x jump in cross-stage transfer and a policy that reads the local grid patch and applies it to both boards. Caveats: (a) absolute depth is much lower than the specialist (generalist plays shallower — expected tradeoff); (b) stage 2 lags stage 1 (~19% vs 31%), a small real gap to watch (harder layout? longer board? mild residual specialization?). Neither undercuts the verdict.

### Env changes made this session (all in `special_stage_env.py`)
- **Win fix:** `_board_cleared` latch + win condition (see #1).
- **Position drop:** `SS_DROP_POSITION=1` zeroes ram features 2-5, keeps vector length 15 (obs space unchanged, so it's a config flag not an arch change — but it BREAKS warm-start from v2, so multistage is fresh-from-random).
- **Stage sampling:** `SS_STAGES=1,2` makes fresh resets `random.choice` between stage state files.
- **Per-stage normalization:** features 0 and 8 now use `self._stage_total` (captured from RAM at reset) instead of hardcoded 102 — required so stage 2's 127-sphere board isn't mis-normalized.
- **Per-stage logging:** `_log_episode` now emits `stage` (parsed from the loaded state filename) and `stage_total`. NOTE the ordering footgun that bit us: `_cur_stage` must be defaulted at the TOP of `reset()` and set during stage-pick — do NOT init it in `_reset_state()`, which runs AFTER the stage-pick and would clobber it back to 0.

### The disciplined staircase (committed plan — validate one variable at a time)
- **Step 1 (DONE, passed):** position-drop, 2 stages, short run → does transfer improve? YES (0.06→0.75).
- **Step 2 (NEXT):** scale to 4-5 stages (capture 2-3 more Angel Island ones, hold 1 out as zero-shot test), medium run → does generalization HOLD across more boards? Two stages could be a fluke.
- **Step 3 (if Step 2 holds):** full run, decide target 7 vs 14. Back-7 (Super Emerald) stages require a full playthrough to Mushroom Hill w/ 7 emeralds to capture — expensive, deferred until the architecture is proven.
- **Open design question with evidence behind it:** keep position fully dropped (max generalization, shallow play) vs Option B (coarse/region position added back, trying to recover depth while keeping transfer). 0.75 ratio suggests full-drop works; ride it into Step 2 as-is, add coarse position only if depth becomes the blocker.

---

## What Changed the Prior Session (2026-06-27) — the "push for the win" arc

The prior run was stuck: frontier frozen at 94, fresh-avg eroding, entropy floored. The earlier read had been "both parametric levers (lr, gamma) tested, horizon ruled out, this is a pure exploration-frontier problem." Reading the actual code overturned that. Three levers, only one of which had really been pulled:

1. **lr — already done, was the real prior win.** Cooled 3e-4 → 1e-4 → **5e-5**. The 5e-5 cut (applied via `custom_objects`) produced a clean, durable step-up: fresh-start average jumped ~35 → ~50 around ep 43,200 and held. This is the change that got the run to a stable high-80s/low-90s frontier in the first place.

2. **gamma — NEVER ACTUALLY APPLIED until this session.** The earlier gamma change (0.99 → 0.997) was edited in the *constructor*, which does not run on a resume — `RecurrentPPO.load()` restores gamma from the pickle, and `custom_objects` only carried `learning_rate`. Confirmed by reading it off the model: `gamma 0.99`. So the past "horizon ruled out" conclusion was unsound; horizon was never tested. gamma=0.99 → effective horizon ≈ 1/(1−γ) ≈ 100 steps ≈ **~1 sphere of lookahead**, far too myopic for endgame route-planning through a board that's mostly red trail. Now correctly set to **0.997** (≈333 steps, 3–4 spheres) via both `custom_objects` AND an explicit `model.gamma = RESUME_GAMMA` assignment, with gamma added to the resume print line so it's verifiable from the live log. Verified live: `gamma 0.997`.

3. **curriculum (restart pool) — the real fix for the wall.** See the dedicated section below. Short version: the pool was seeding restarts at avg depth **52**, nowhere near the 88–94 wall, for two compounding reasons — (a) uniform `random.choice(files)` over a pool dominated by well-stocked shallow buckets, and (b) `CAPTURE_AT` topped out at 90, so the final 12 cells were never snapshotted and the Go-Explore ratchet had no top rung. Fixed both. After the surgery the frontier ratcheted **94 → 96 → 98 → 99 → 101** over ~3,000 restart episodes.

**Result signature (confirms the mechanism):** as the frontier climbed, restart `gained_avg` collapsed from ~13 to ~1.6–2.1. Low gain + climbing frontier = restarts now land *at* the wall, make a small push, bank the new deep state, and relaunch from it. That's the ratchet working exactly as the sim predicted.

**Things deliberately NOT done (and why):**
- No more lr/gamma/penalty fiddling — parametric levers are ruled out for *this* wall (it's an exploration-frontier problem, now correctly targeted by the curriculum).
- **Red channel was considered and rejected.** Earlier claim that "the agent can't see red" was WRONG and was corrected. In `_grid_patch`, red (code 1) and transitional (code 10) fall through the `~known` catch-all into channel 1 ("unknown"), so the agent *does* have a danger signal — it's just mislabeled and conflated with transitional. A dedicated red channel is a marginal disambiguation, not a fix for blindness, and it would break clean resume (GRID_CHANNELS 4→5). The 37% trail-death rate is therefore more likely a control/reaction-distance problem at high speed than a perception one. If a perception lever is ever justified, the better experiment is a **wider patch** (more lookahead cells at Speed 4–8), not a new channel — and only if the frontier genuinely plateaus. *[2026-09-29: two facts undercut this reasoning — springs (code 5) also landed in ch1, and env v1 read board rows 24–31 from the wrong memory, so red there was invisible and deaths there could never be classified as trail deaths (the 37% was measured blind). Both fixed in env v2.]*
- **Bouncer-penalty experiment (VERY EARLY, pre-grid-patch, minimal RAM) — superseded, do NOT re-add.** Long ago, before the grid patch existed and before we had much RAM data, we experimented with giving a **negative reward when the agent hit a bouncer sphere**. Theory: if the policy is memorizing the whole board as a route, one bouncer bump scrambles that memorized map, and any blue/red spheres grabbed after the bump become noise that harms the memorization — so penalize bouncers to discourage the disruption. This was an early, indirect swing at memorization *before* we understood the mechanism. **Superseded:** Step 1 later traced memorization to the **absolute position encoding**, which the multistage runs drop directly (`SS_DROP_POSITION=1`) — the root-cause fix, vs. the bouncer penalty's symptom-level one. The blind-agent conditions that made bouncers so catastrophic (no local layout view) also no longer hold now that the grid patch feeds a heading-rotated local patch. **Live version of the concern still valid:** bouncers (yellow/white) are a plausible *memorization-fragility point* on specific boards — a bounce is a high-variance transition that's hard to memorize a route through. Right response is NOT to re-add the penalty; it's to watch for per-board plateaus (esp. bouncer-heavy boards) and, if one stalls, prefer a wider perception patch / curriculum concentration over a reward hack. (Note: `speed_level` logging traces back to this era of hazard/speed diagnosis.)
- **WHY the bouncer penalty actively BACKFIRED (the concrete failure, not just "it didn't help"):** bumper corridors exist across many maps. When Sonic enters one he **oscillates rapidly back and forth between the bumpers**, and a per-bump negative reward **stacks a penalty on every bounce** — so one entry into a corridor produced a *cascade* of tens of penalties from essentially a single event Sonic couldn't easily control. Two failure effects: (1) reward-magnitude explosion uncorrelated with any single decision (penalty ∝ corridor length / bounce-loop duration = noise, not signal — wrecks credit assignment); (2) the safest learnable response became "avoid the whole bumper region," which on dense boards means refusing to traverse areas needed to collect spheres. The hack meant to improve bumper *handling* instead taught bumper *avoidance* of large playfield regions.
- **General lesson (design principle):** **per-EVENT penalties are dangerous when the event can repeat rapidly from a single situation** (bumper oscillation, wall-grinding, any bounce loop). A per-contact penalty becomes an uncontrolled cascade. Prefer penalizing the **outcome** (death, net progress lost) over the **contact** — which is what the current reward structure already does by leaning on the death penalty rather than hazard-contact penalties. Keep it that way.

---

## Prior-Session Specialist Training State (2026-06-27 snapshot, ~262M)
*Superseded — the specialist later reached ~326M with the win-fix (fresh best 96, 3 restart wins) and is now PAUSED as baseline. Table kept for the mid-run picture.*

| Metric | Value |
|--------|-------|
| Total steps | ~262.7M |
| learning_rate | 5e-5 |
| gamma | 0.997 (applied this session; was 0.99 the entire prior run) |
| Frontier (deepest total ever) | **101 / 102** |
| Fresh-start best | 93 (was 91 for the whole prior run) |
| Fresh-start avg (recent 500) | ~38–45 |
| Wins | 0 |
| entropy_loss | ~-1.1 to -1.35 (floored, intentionally not fought) |
| explained_variance | 0.98–0.999 (critic healthy; re-fit cleanly after gamma change) |
| approx_kl | 0.05–0.09 (slightly warm but stable) |
| Trail-death rate | 37.0% (flat through everything — untouched lever) |
| Speed levels seen | now up to 7–8 (didn't exist pre-curriculum; deep endgame at max speed) |
| fps | ~440–540, 8 envs |

Frontier climb is **decelerating** at the top (gained_avg ~1.6–2.1, three buckets hovering at 101). This is expected — the last cell is the hardest in the stage (max speed, board ~entirely red trail, zero-margin final sphere). NOT yet a confirmed plateau; needs 10–15M more before that call.

---

## Pre-Committed Criteria / Open Decisions

Written down *before* the data, to read the same way later:

### Run mg1 (aiz1 on MG env v2) — PROPOSED by Claude 2026-10-11 overnight; NOT agreed, NOT started
**Why:** aiz1 (MG env v1, Tyler's 09-30 AIZ1) walked 98% → 84% in 344K steps, then sat at 83% for 3.7M steps, reached 82% (start point 69) at 4.66M once `stuck` existed, and was paused at 5.0M. **Probe on its final save (2026-10-11, CPU, `attic/one_off_tools/probe_aiz1_wall.py 69 20`): from start point 69, 20 of 20 stochastic episodes end `stuck`** at demo time ~4,425 of 5,060, the agent at x 3,765-3,830, y 900-1,190, under a ledge with wall spikes; its furthest x 3,842-3,893. Tyler's route there (demo t 4,260-4,455): at (3,830, 942) he goes back UP-LEFT to (3,730, 821), then up to (3,753, 757) and right along y ~760. Going up-left earns no route progress, and v1 cuts after 600 frames without it, so the agent never gets the time to find the climb. MG env v2 counts new ground (a 128-px block not visited this episode) against `stuck`; up-left is new ground. (v2 also settles off-route time on return and breaks route ties by the clock.)
**Setup (RUNS 'mg1'):** aiz1 exactly — same recording (pinned), AIZ1 only, lr 2.5e-4, PPO settings, demo_prob 0.75, 50M-step budget, fresh weights — except `env_version=2`. Alone on the box ~510 steps/s (aiz1's rate): 5M steps ≈ 2.7 h.
**Proposed criteria:**
- **Wall gate: start point below 82% (past start point 69) within 5M steps** (aiz1 needed 4.0M steps from 83% to 82% and never passed it).
- **Main test: a fresh-start clear of AIZ1 within 50M steps** (aiz1's agreed test, unchanged).
- **Stall stop:** start point unmoved for 3M steps.
- **Clear time** vs his 5,060 frames: reported, not a bar.
- Caveat: v2 changes three things at once. If mg1 passes the wall and that matters for the full-game run, a follow-up can isolate the new-ground rule.
**Launch-ready (checked 2026-10-11 ~01:40, CPU, no repo files written):** `check_play_acts.py demos/play_20260930-153946.npz 20` under v2 (with the tie-break): his AIZ1 clears on his exact frame with return = CLEAR_BONUS from all 20 start points (`attic/one_off_tools/check_play_acts_mg1_demo_n20.log`). A random-action dry run of the env exactly as `configure_env('mg1')` sets it (v2, AIZ1, demo_prob 0.75; 40 episodes, log to scratch): obs 84x84x4, 9 actions; 25 clear (from the last start point), 9 stuck (fresh starts), 6 timeouts (the miniboss at start point 83, cut at 2 x 80 + 1,200 frames as designed). Launch: `nohup python3 train_main.py mg1 > logs/main_mg1.log 2>&1 &`, then check it says FRESH.
**After mg1:** the full recording (27 acts, `play_20261010-202923`) — e.g. one zone (AIZ1 + AIZ2) first, then all acts; ~1 GB RAM per worker for all 27 acts.

### Run d9 (all 14 boards on env v7) — criteria AGREED with Tyler 2026-10-10 ~21:05 ("All of those rules sound good to me. And let's go with your recommendations as well"); LAUNCHED 2026-10-10 23:21 by Claude (Tyler gave Claude the box overnight)
**Launch notes:** the first launch (23:19) REFUSED: the trainer's pre-launch `demos.replay_check` cut stage 6 at 127/128 spheres (frame 5,961). Cause: replay_check never set `env._replaying`, which since env v6 is what exempts a recording from the 600-frame stall rule (training's own `_build_demos` sets it); stage 6 waits >600 frames for its last sphere, and no earlier launch had replayed a stage with such a gap. Fixed (cde6b65); all 14 then reproduced their clears under d9's settings, and d9 launched fresh at 23:21 (`logs/ss_multistage_d9.log`, ~85 steps/s at first). Overnight watch: `attic/one_off_tools/d9_status.py d9 --state logs/d9_watch_state.json` every 10 min into `logs/d9_watch.log` (per board: frames, start point, fresh clears, best; stall-stop and gate bookkeeping).
**Setup (RUNS 'd9'):** stages 1-14, one recording each (all flawless, pinned by name), env v7 (up / up-now, late-jump, landing decisions, turn_here), d7's recipe: compass (SS_POINTER), gamma 0.99 per decision, LSTM 512, lr 3e-4, demo_prob 0.75, backward curriculum per recording, NUM_ENVS 8. Fresh weights (v7's 14 ram feats and 9 actions don't fit d7's network). No stages held out: Tyler's goal is all 14 boards, and the special stages are exactly those 14, so learning those boards IS the goal; generalization beyond them isn't needed. Checked: the 14 demo names resolve to stages 1-14 in order.
**Choices — Tyler took both recommendations (no speed bonus in d9, speed phase after; all 14 boards at once):**
1. **Speed bonus: NOT in d9; a speed phase afterwards** (d9's save → a d10 with SS_SPEED_BONUS, as d7 → d8), once d8 vs d8c shows whether the bonus works and what it costs in clears. One variable at a time; the bonus pays nothing until a board clears anyway. Alternative: include it from the start (k=7). Caveat for either: 7/frame is ~0.3% of a stage-1 clear per 100 frames but ~0.9% on stage 10 (37 spheres), so its strength varies by board.
2. **All 14 at once** (as above). Alternative: a smaller set first (d6 ran 4 boards: start points 64-79% at ~4.1M frames each, 0 fresh clears at 16.5M total).
**Criteria (AGREED; frames counted as for d4/d6: per stage from ss_log_d9.jsonl, steps - demo_frac x that demo's length; fresh = steps):**
- **Gate at 70M total frames (~5M per board):** every board's start point has moved off its recording's end, every board is below 90%, and the median board is below 75% (d6 reached 64-79% at ~4.1M frames per board). If not, 14 boards are too thin a mix: fall back to a smaller set.
- **Interference read (report, not a stop):** stage 1's start point at equal stage-1 frames vs d7 (single board, same recipe); within 5 checkpoints = none.
- **Main test: a fresh-start clear on ALL 14 boards within 280M total frames** (~20M per board, 2x d7's 10.3M to its first stage-1 clear). Also report how many boards have one at 140M (d7's pace per board).
- **Stall stop:** no board's start point moves for 15M total frames (the rule from the d4/d6 overnight fallback).
- **Clear time:** reported per board against his recordings as a reference, never a bar (Tyler: his play is a starting guide).
**Before launch:** `smoke_corners.py d9` — **PASSED 2026-10-10** (ALL CHECKS PASS, all 14 recordings, every start point, d9's own config incl. the compass; log logs/smoke_corners_d9.log; jump timings tried 5,695 / 0 unreached, Up timings 560 / 142 unreached but every start point reproduced his finish) — must pass (14 recordings, all start points; ~2.5 CPU-hours; the per-recording v7 check already passed on d4's setup, and d9 differs only in the compass obs). Wall time is unknown until measured: the window fix (eda22d4) changed speed. Guess 3,000+ frames/s on 8 cores → gate ~6-7 h, main-test budget ~26 h.

### Env v7 open questions: Up timing and turn queueing, settled by MEASUREMENT — AGREED with Tyler 2026-10-10 ~18:00, before measuring
Tyler: "I'll trust good data more than anything." These rules are fixed before any of the measurements run. Script: `attic/one_off_tools/measure_v7_decisions.py` (env v7 prototype, d7's obs preset + v7, run at nice 19 while he plays).
- **Up timing.** States: every back-restart decision (running backwards after a bounce) in his 14 recordings, plus random-agent ones across all 14 stages. From each, Up-now (press at once, as he does) vs Up-late (env v7: once past the cell's centre), then every corner option (straight / left / right / jump). Record frames, and the outcomes reachable (cell + heading at the next decision after that).
  - Up-now loses no outcome Up-late reaches in >= 99% of bounces → **replace** Up-late with Up-now (no new action).
  - Up-now loses outcomes in > 1% of bounces but saves frames → **offer both** (a 9th action).
  - Up-now saves < 1 frame on average → **keep** Up-late, and set smoke_corners' per-Up allowance from the measured worst case.
- **Turn queueing, inferability.** Random-agent decisions on v7; target = the "window still ahead" bit (it predicts in-step vs a-cell-late at 99.3% / 97.9% away from bumpers). Features = what the agent already has: the ram obs + its own last action (+ the centre cell of the patch, i.e. "am I on a bumper"). Classifiers: a small MLP and a lookup table, scored on held-out episodes.
  - >= 98% held-out accuracy → **no fix** (its memory can learn it).
  - < 98% → **add the obs bit** (dynamics unchanged; the 14 recordings replay as now).
- **Pointless decisions.** At bumper-entry decisions and at decisions offered while a turn is already queued: try every action from the same snapshot (snapshot yardstick: the same action twice gives identical results).
  - No action ever changes the outcome vs straight → **remove** those decision points, then re-verify all 14 recordings.
  - Some action does → they're real choices; **keep** them.

- **RESULTS (2026-10-10 ~18:35; `measure_v7_decisions.log` / `_out.json`, 15 min; snapshot yardstick 20/20 identical):**
  - **Up timing → OFFER BOTH.** 314 bounces (14 his, 300 random). Up-now loses an outcome Up-late reaches in 13 (4.1%, over the 1% bar); where both reach the same one it saves mean 1.75 frames (median 2, range -4..+3, n=1,163). On 4 of his 14 bounces Up-now was slower AND lost a corner (s6, s9, s11, s14), so neither dominates. Built as action 9, `up-now`.
  - **Turn queueing → ADD THE OBS BIT.** 7,647 random-play decisions, 700 episodes (1,585 held out); bit=1 share 67.1%. Held-out accuracy from the obs it already had plus its last action: MLP 90.9%, lookup table 80.8% (bar 98%). By kind: normal 98.5%, start 100%, back-restart 87.2%, landing 81.7%, after-Up 72.2%. Built as `turn_here`, the last ram feature (v7 only; 14 feats on d7's preset).
  - **Pointless decisions → KEEP.** Every action, then straight at the next decision: at bumper-entry decisions some action ends elsewhere than straight in 198/200 (99%); with a turn queued 197/200 (98%). A turn chosen at a bumper's entry is queued and fires after the bounce: a real choice.
  - Built into the v7 prototype right after (still uncommitted); smoke_corners picks his Up timing like his jump timing. Verification running at nice 19 on cores 5-7 while Tyler plays: all 14 recordings + `check_v7_isolation.py 4 5 6` (v4-v6 obs and rewards identical to git HEAD).

### Run d8 (d7 + clear-time bonus) — agreed with Tyler 2026-10-10 (~17:40, "Go ahead and build d8", replying to this proposal); speed bar corrected before launch
**Tyler's goal (2026-10-10): optimize for faster clear times; rings don't matter; his play is a starting guide, not the target** ("I'm far from the most skilled. I'm sure there are routes way better than mine"). Before d8 the reward paid almost nothing for speed: −0.1/frame and a clear time bonus of at most 300, against +2,000 per sphere (500 frames faster ≈ 58; invisible after VecNormalize, return std ~28K). Only the per-corner discount (γ 0.99) favoured shorter routes, and it counts cells, not frames.
d8 differs from d7 ONLY by `SS_SPEED_BONUS=7`: a clear also pays 7 × (SS_MAX_STEPS 18,000 − frames), so each frame saved is worth 7 and 100 frames ≈ 0.3% of a stage-1 clear. It is absolute: his time is not in the reward. It is paid only on a clear, so it is inert until the agent clears, and it can't reward dying early. (Deaths net −2,500 however long they take; a per-frame cost would make long failures cost more.) Rings stay out of the reward. d8 starts from d7's save (new `init` field in RUNS: weights, VecNormalize and curriculum pointer, at 975,008 steps, ckpt 0); its frame count and log start at zero. Verified by `check_speed_bonus.py` (ALL CHECKS PASS): with the flag off, the same 3,670 steps under git HEAD's env give identical rewards and observations. With it on, the observations and trajectory are identical and the reward differs only on the 2 clearing steps (his replay and his corners through step(), both at 2,925 frames), by exactly 105,525. `smoke_corners.py d8` ALL CHECKS PASS. Frames counted as for d7.
- **d7's baseline, measured from ss_log_d7.jsonl (last 300 fresh starts):** clear rate 67.7%, median clear 3,202 frames (p25 3,128, p75 3,335), best 3,014. Last 1,000: 49.5%, median 3,310 — **d7 was still improving when paused.** The proposal to Tyler said "below 3,471 (d7's)"; that was one early clear, not d7's level, so the bar below uses the measured median (stricter).
- **Speed (main test):** median frames of d8's fresh-start clears (last 300 fresh starts) **below 3,202 within 20M frames** of d8's own training.
- **Reliability guard:** fresh-start clear rate (last 300) **≥ 50%**. Below that, the bonus is too strong (it trades clears for speed).
- **Reference, not a bar:** Tyler's 2,925; d7's best 3,014. Report where d8 lands relative to both.
- **Attribution — the CONTROL is run `d8c`** (Tyler decided at ~17:35 to run it after d8, then at ~20:05 to run it NOW, while he plays). d8c = d7 with no change, started from d7's save exactly as d8 is (`init='d7'`), frames from 0, so the two compare at EQUAL frames and d7's own save stays untouched. Launched pinned to cores 6-7 at nice 19 (OMP threads 2) while he games; same NUM_ENVS 8, so only wall-clock speed differs. **Read (PROPOSED by Claude, not yet agreed):** at 20M frames each, compare median fresh-clear frames (last 300 fresh starts) with bootstrap 95% CIs. The bonus gets the credit only if d8's median is below d8c's with non-overlapping CIs; the reliability guard (>= 50% clears) applies to both. Original note: d7 alone was still getting faster, so d8 beating 3,202 doesn't by itself credit the bonus. A control would be d7 resumed (no bonus) for the same frames: side by side on the box (each ~half speed), or after. Without one, the read is only "faster than d7 at its pause".
- **RESULTS at 20M frames each (2026-10-10 23:16-23:18; both stopped cleanly; `logs/d8_vs_d8c_20M.txt`, `speed_progress.py d8|d8c`):**
  - **d8 PASSES its agreed test:** last 300 fresh starts: median clear **3,089 frames** (95% CI 3,089-3,098) vs the 3,202 bar; clear rate **82.3%** (78.0-86.7%) vs the 50% guard (d7: 67.7%). Best 3,012 (his 2,925; d7's best 3,014). By 2M bucket the median fell steadily: 3,368 / 3,446 / 3,258 / 3,276 / 3,227 / 3,206 / 3,124 / 3,119 / 3,089 / 3,097; clear rate 62% → 77-91% (one dip to 24% at 8-10M).
  - **The control d8c COLLAPSED at ~16.5M frames.** Until then it got faster WITHOUT the bonus, faster than d8: median 3,230 / 3,224 / 3,206 / 3,206 / 3,183 / 3,352 / **3,036 / 3,016** (12-16M; best 2,884), while its clear rate fell 70-77% → 52%. Then one update (22:33-22:34, iteration 81: value loss 0.022 → 0.486, explained variance 0.94 → 0.15, approx_kl 0.05 → 0.10) took fresh clears from 25/25 to 0/25 and demo-start success to 1/300; the next updates restored 90+ spheres but no clears. Last 300 fresh: 1 clear (0.3%). d8 beside it was unaffected (same box, same code, same time). No checkpoint before the collapse survives (first rolling checkpoint is at 800K steps; the collapse was at ~717K).
  - **Read (Claude; the proposed rule was never agreed, and its control broke):** the rule as written credits the bonus (d8c fails the guard; its one clear is 3,382), but that compares against a collapsed run. Before the collapse the control was FASTER than d8 at equal frames, so the speed gain is NOT attributable to the bonus: plain continued training from d7 sped up too. What the bonus run shows is reliability: d8 stayed at 77-91% clears where the control slid to 52% and then fell apart. One seed each, so the collapse may be chance. For the planned speed phase after d9 (d10): the evidence mildly favours keeping the bonus, for reliability rather than speed; Tyler's call.
  - **Where d8 spends its time (route_vs_tyler.py d8 60 on the final save, CPU; log/json in attic/one_off_tools/):** 57/60 stochastic fresh starts cleared (d7: 51/60). Median clear 3,158 (d7 3,193, his 2,925; this probe samples actions, so it runs slower than the training log's 3,089). **Turns 62 (d7 70, his 55): the bonus cut 8 turns = 120 frames of rotating in place**, but d8 travels 157 cells (d7 151, his 146) and jumps 24 times (d7 16, his 0). Remaining gap to him ≈ 7 extra turns (~105 frames) + ~11 extra cells (~150 frames at his ~14 frames per cell). Outline trick identical (16 spheres), 0 bumper hits. By quarter: 966 / 1,573 / 2,550 / 3,158 vs his 826 / 1,495 / 2,272 / 2,925.
- **Collapse lesson for long runs (d9):** a single PPO update can destroy a good policy (lr 3e-4 constant, clip 0.1, ~0.05 approx_kl per update is common here). Rolling checkpoints every 800K steps, last 5 kept, are the only way back; watch for sudden clear-rate drops.

### Run d7 (d5 + compass to the nearest blue) — AGREED with Tyler 2026-10-10 07:52, before launch
d7 differs from d5 ONLY by `SS_POINTER=1`: 3 features appended to the ram
vector — unit vector (forward, right) from Sonic's cell to the nearest blue
cell, straight line on the wrapping board, in the patch's heading frame, and
distance/16. Tyler chose the straight-line compass over a red-free route
pointer and d5 (stage 1) as the base. Why: d5's 21% stalls had every
leftover blue outside the 15x15 view (stall_view.py), 71 of 88 still
reachable red-free, route median 27 cells (stall_reach.py).
Verified by `check_pointer.py` (ALL CHECKS PASS): flag off = git HEAD's env
(ram, grid, rewards identical over a seeded 600-step trajectory); flag on
changes only the 3 appended features; on all 488 observations whose nearest
blue was inside the patch, the patch has a blue exactly where the pointer
says (all 4 headings, all 8 direction signs); never a nearer patch blue.
`smoke_corners.py d7` ALL CHECKS PASS. Frames counted as for d3/d5.
- **Gate at 50M frames:** start point **below 21%** (ckpt <= 12) — further
  than d5 ever got (d5 reached 21% at 31.36M and sat there 15.4M frames). If
  not → the compass isn't the fix (or not alone).
- **Mechanism, when the start point first reaches 21% (ckpt <= 13), on the
  next save:** `PROBE_RUN=d7 probe_steering_v4.py 100 0.21` (same start band
  as d5's probe). **Pass = >= 25 of 100 reach the target** (the curriculum's
  own advance bar; d5: 0 of 100). Also read `stall_view.py`: blue in view in
  stall windows (d5: median 0%).
- **Pace (read, not a stop rule):** frames to 56% and to 21% vs d5's 18.80M
  and 31.36M.
- **Stall stop:** start point unmoved for 15M frames.
- **Main test:** first fresh-start clear of stage 1 within **100M frames**
  (d5: none in 46.7M).
- **RESULTS (2026-10-10, Claude watching; frames counted as for d5 — the same
  code reproduces d5's 18.80M / 31.36M):**
  - **Main test: PASSED at 10.3M frames** (10:40, 102/102 in 5,064 frames; a
    second at 3,471; Tyler's demo 2,925). By 12.68M: 19 clears in the last 300
    fresh starts, mean 68.8/102.
  - **Gate bar met early:** ckpt 12 (below 21%) at 11.72M frames; 12% (ckpt 7)
    at 12.59M. The formal read at 50M frames is still to come.
  - **Pace:** 56% at 9.31M (d5 18.80M), 21% at 11.33M (d5 31.36M).
  - **Mechanism probe at 21%: FAILS AS WRITTEN, narrowly — 22 of 100 reach
    the target (bar >= 25; d5 0).** Save 10:53:19, 500K steps, band [12, 16]
    (ckpt 12 = 19%, one behind training's 13 — d5's caveat). 22 cleared, 41
    stalled, 37 red (d5: 0 / 88 / 12). The curriculum's own 25% bar was met
    on the very next stretch: 21% -> 19% at 11.72M, 0.39M frames after.
    stall_view: blue in view median 13% of a stall window (d5 0%), 10 of 41
    windows >= half (yardstick 96% while collecting). Stall windows: route
    distance 25 -> 8 -> 16 (closer in 25 of 41), but with the nearest blue 9+
    cells away its moves are on a shortest route 51% vs 56% chance — the
    compass gets it near, not onto. Tag d7_v4_21 (steer_*_d7_v4_21.json).

### Overnight fallback d4 / d6 — AGREED with Tyler 2026-10-09 ~22:10, before either launches
**LAUNCHED: d6, 2026-10-10 05:04** (d5 got below 56%, then its stall stop
fired). Baseline d5. Smoke: `smoke_corners.py d6` ALL CHECKS PASS 22:58.

Only if d5 is stopped by one of its rules overnight (gate, stall stop); Claude
launches it then without waiting. **Which:** d5 never got below 56% → **d4**
(γ 0.95), baseline **d3**. d5 got below 56%, then a rule stopped it → **d6**
(d4 with γ 0.99, nothing else changed), baseline **d5**. Either way the
difference from its baseline is training four boards (stages 1-4, one demo
each, env v5) instead of one; stages 5-7 stay held out. `smoke_corners.py`
must pass for the run first. Frames counted as for d3, per stage from each
episode's `stage`.
- **Interference read at 50M frames** (~6 h alone): stage 1's start point vs
  the baseline's at the same number of STAGE-1 frames. Within 5 start points
  (or ahead) = four boards don't slow stage 1; further behind = interference.
- **Walls:** a stage whose start point is unmoved for 15M of its OWN frames
  is walled — probe it (probe_steering_v4 needs adapting to other stages)
  before naming a cause.
- **Stall stop:** no stage's start point moves for 15M total frames.
- **Main test:** first fresh-start clear of any of stages 1-4 within 100M
  frames (record stage, frames, time).
- **Held-out 5-7:** zero-shot test (`eval_transfer.py`) only after a fresh
  clear — not overnight.

### Run d5 (d3 with γ 0.99 per corner) — AGREED with Tyler 2026-10-09 21:53, before launch (proposed earlier that evening)
**RESULTS so far (2026-10-10 ~01:00):** gate bar met early (below 56% at
19.88M frames, pointer can't move forward); mechanism check **FAILED as
written** (left onto (25,21) 0 of 49 — it got past by returning along row 25;
see "Current state"). **GATE PASSED at 40.11M frames (03:59): start point
21% (ckpt 13)**. Pace read: 97%→56% took 18.80M frames vs d3's 11.62M
(1.6x — inside the 2-3x allowance), incl. 8.6M stuck at 72%. Stall stop
and main test (fresh clear within 100M) still open; 0 fresh clears at 40M
(fresh mean ~85/102, best 98).
**STALL STOP fired 2026-10-10 05:04** (21% unmoved 31.36M→46.72M frames);
stopped cleanly at 2,113,600 steps. Main test left open (0 fresh clears at
~46.7M of its 100M; best 100/102).

d5 differs from d3 ONLY in γ (0.95 → 0.99), so a difference is γ's. The
question: does a ~100-corner horizon get past d3's 56% wall (the (26,21)
jump whose penalty lands ~101 corners later)? Frames counted as for d3.
- **Gate at 40M frames:** start point **below 56%** (further back than d3
  ever got; d3 reached 56% at 11.6M frames and sat there). The margin allows
  γ 0.99 to learn the early points 2-3x slower than d3. If not → γ isn't the
  fix (or not alone): stop and rethink.
- **Mechanism, when the start point first reaches 56% (or at the gate):**
  `PROBE_RUN=d5 probe_steering_v4.py 100 0.56`. At (26,21) d3 jumped with
  P 0.998 and 53 of 90 stalls ended with (25,21) the only sphere left. Pass
  = the left turn onto (25,21) is the majority choice there.
- **Pace (read, not a stop rule):** frames per start point 97%→56% vs d3's
  11.6M total — γ 0.99's cost on the easy part.
- **Stall stop:** start point unmoved for 15M frames, as d3.
- **Main test:** first fresh-start clear of stage 1 within **100M frames**,
  as d3. Same attribution caveat (demos + fixed env together).

### Run d3 (env v4, one choice per corner) — agreed with Tyler 2026-10-09, at ~1M frames
**Budgets are in FRAMES.** The 50M-step test below was written when a step was
1 frame (d1), i.e. ~50M frames, a sixth of the specialist's 326M. d2's 4-frame
steps silently made it 200M frames, and d3's ~23-frame steps would make it
>1B. Read in frames, d2 had spent half of it (25M) and was 22% of the way
back, slowing: heading for a miss. Pace data now exists: d2 averaged ~1.2M
frames per start point (22.8M frames for 97%→78%), the last point 9.3M.
- **Gate at 25M frames** (~1.1M d3 steps, ~3 h alone): the start point must
  be past **78%** (where d2 was at equal frames). If not, the action change
  didn't move the wall: stop and rethink. At the gate, re-run the steering
  probe (adapted to v4) for the mechanism: does it go straight on the ground
  where only straight is right? (d2: 6-9%.)
- **Stall stop:** the start point unmoved for **15M frames** (~2 h) → stop and
  diagnose, don't wait (d2's slowest single point: 9.3M).
- **Main test:** first **fresh-start clear of stage 1 within 100M frames**
  (~4.3M d3 steps, ~13 h of machine time alone). From d2's 1.2M frames/point ×
  d3's 65 start points ≈ 78M, plus margin; under a third of the specialist's
  326M frames, which never cleared from a fresh start. Record the frame count
  of the first fresh clear and frames per start point.
- Attribution caveat as below: a pass credits demos + the fixed env together.
- Frames = sum over episodes of frames played (`ss_log_d3.jsonl`: `steps`
  minus the demo start frame); ~23 frames per d3 step, ~2,200 frames/s alone.

### Does learning from Tyler's recordings beat learning from scratch? (2026-09-30, before results)
The baselines it has to beat, both learned from scratch:
- **Base game — v10** (PPO CNN, ~210M steps): stalls at AIZ1's first ledge
  (x≈6765) in 3 of 3 runs. It never cleared AIZ1.
- **Special stage 1 — the specialist** (~326M steps): never a fresh-start
  clear (best fresh 96/102; its 3 wins were all from restart states).

The tests. A *fresh start* means the act's or stage's own start state, not a
start point along a recording:
- **aiz1 passes** when it clears AIZ1 from a fresh start (`main_progress.py
  --run aiz1`, "from start" clear rate > 0). Stronger bar: beating Tyler's
  1:25.43 from there (margin > 0).
- **d1 passes** when it clears stage 1 from a fresh start
  (`demo_progress.py --run d1`, "fresh starts" clears > 0). Stronger bar:
  more of them than the specialist's restart wins (3).
- **Decisively better** = both pass within **50M steps each** (a sixth of
  the specialist's budget, a quarter of v10's). Record the step count of each
  run's first fresh clear.

How to read the other outcomes:
- **d1's start point never leaves 97%** → the backward curriculum's first
  step is too steep in Blue Spheres (the end of a run is the hardest part to
  steer). That's a finding about the curriculum, not about demos.
- **aiz1's start point walks back but fresh starts never clear** → look at
  where fresh episodes die. So far that's x 7168–7423, just past the first
  ledge, in a part the curriculum hadn't reached yet.
- **Attribution is not clean.** v10 and the specialist also ran on buggy envs
  (base game: late death, wrong clear signal; special stage: board rows
  24–31 invisible, deaths ~11 s late). A pass says "demos + fixed env" beat
  "scratch + buggy env". To credit the demos alone: special stage → a
  stage-1 run on env v2 without demos (not yet defined); base game → no
  control exists yet (its progress is measured along the recording, so a
  no-demo run needs an x-progress reward first).

State when written: aiz1 paused at 5.0M steps (start point 82%, just past
the wall before the miniboss); d1 running alone at ~417 steps/s (start
point 97%, 0 matches yet).

- **First win** will come from a **restart** episode (`success: true`) → that's the *scaffolded* win + emerald. Proves the endgame is solvable.
- **Handoff-real solve** = a **fresh-start** completion (the main game drops the agent in at sphere 0). This lags the scaffolded win — a cold run must survive 0→90 (rare) *and* close 90→102. Fresh ceiling is 93, so there's real distance between "won from a restart" and "can win from zero."
- **If the frontier sits genuinely flat at 101 for 10–15M more steps** → THEN the patch-width experiment (wider lookahead, breaks clean resume) becomes evidence-justified. Not before.
- **Do NOT touch the run while mid-ratchet.** Breaking clean resume torches the banked near-finish states, which are the most expensive states in the pool to regenerate.
- Watch: explained_variance dip-and-recover (the gamma signal — already observed and recovered), deep pool buckets (`s92`/`s94`/`s97`) populating, frontier moving off 101.

---

## Architecture — PPO (Active)

**Framework:** RecurrentPPO + MultiInputLstmPolicy (sb3-contrib)
**LSTM:** lstm_hidden_size=512, n_lstm_layers=2, shared_lstm=False, enable_critic_lstm=True
**n_steps=1024, batch_size=256, n_epochs=4, clip_range=0.1, gae_lambda=0.95**
**LSTM hidden size is per-run:** 512 through v4, **1024 in v5** (the capacity test).

**Observations (Dict) — shape is per-run, set by `SS_ENV_VERSION` / `SS_USE_PIXELS` /
`SS_PATCH_R` / `SS_DROP_POSITION`. Current default (env v2, v6) is
`{ram: 10, grid: 15×15×5}` — NO pixels. v4/v5 (env v1) are `{ram: 11, grid: 15×15×4}`.**
The pre-v4 shape, described below, is what the specialist still uses (on env v1):
- `pixels` — 84×84 RGB (NatureCNN). **Dropped from v4 on** (see "Why drop pixels").
- `ram` — 11 stage-agnostic features (10 in env v2, which drops `total_progress`
  = 1 − feature 0); +4 sin/cos absolute-position features at indices **2-5**
  when `SS_DROP_POSITION=0` (specialist only) → 15 total.
  Position keeps that historical slot deliberately: a reordered vector would let
  an old checkpoint load with matching shapes but permuted inputs — a silent
  failure, unlike a shape mismatch.
- `grid` — local heading-rotated patch, PATCH_SIZE=11 pre-v4 / **15 from v4**,
  GRID_CHANNELS=4 (env v1) / **5 (env v2)**. Env v1 reads it from the wrong
  memory for board rows 24–31 and floors the centre; see 2026-09-29.
  - ch0 = blue (code 2)
  - ch1 = "unknown" = `~known` — **red (1) and transitional (10) land here**
    (and in env v1 also springs, code 5); functionally a danger channel
  - ch2 = bumper (code 3)
  - ch3 = ring (code 4)
  - ch4 = spring (code 5) — env v2 only

**Action space:** Discrete (all button combinations)

### Cell codes (special-stage grid)
`0=empty  1=red  2=blue  3=bumper  4=ring  5=spring  10=transitional(→red)`

### Reward Structure
- Per step: small time penalty
- Blue sphere: large positive (≈ +2000-scale)
- Death: scaled on *this-episode* collections so net death ≈ **−2500 regardless of depth** (the net-negative-death invariant). Because the penalty scales on this-episode spheres only and restarts grant zero reward for handed-in spheres, a restart-from-88 that dies at 94 gets the same clean "collect more before dying" gradient as a fresh death. **Caveats found 2026-09-29:** env v1 paid it 651–678 frames after contact (discounted to 0.13 at γ=0.997); env v2 pays it on contact. And the `clip_reward=10` in VecNormalize caps it once n ≥ 3, so the invariant is not what PPO actually sees — open, see "What Changed 2026-09-29". (That was with the old runs' return std ~710. On d3 the std is 28,313 and nothing is clipped — measured 2026-10-09, see "Current state".)
- Chaos emerald / completion: large bonus + time bonus. Env v1 never paid the emerald part (see 2026-09-29); env v2 pays completion + time bonus + the stage's emerald on the clear frame.

---

## Curriculum System (self Go-Explore restart pool) — v4

**Mechanism:** on `reset()`, with probability `RESTART_PROB=0.25`, restore a saved emulator savestate from the pool instead of starting fresh. ~78% of episodes are honest from-scratch starts (forgetting guard). On a deep run, `_maybe_capture` snapshots the emulator at sphere-count thresholds; restart episodes that push past the frontier bank the new deeper states automatically (the ratchet runs through restart episodes).

**v4 changes (this session, both in `special_stage_env.py`):**
1. **Extended capture ladder:** `CAPTURE_AT = [30, 40, 50, 60, 70, 80, 84, 88, 90, 92, 94, 97, 100]` — adds rungs past 90 so the final cells finally get snapshotted (gives the ratchet a top).
2. **Depth-weighted restart sampling:** replaced uniform `random.choice(files)` with `_weighted_pick`, sampling a bucket by `RESTART_DEPTH_WEIGHTS` (skewed deep) then a uniform file within it. Renormalizes over buckets that actually have files, so empty deep buckets cost nothing and the realized mean depth self-advances as the agent banks deeper states.
   ```python
   RESTART_DEPTH_WEIGHTS = {30:1,40:1,50:1,60:1,70:2,80:4,84:6,88:9,90:11,92:13,94:14,97:14,100:14}
   ```
3. `RESTART_PROB` left at 0.25 (forgetting guard + restart volume already ample at ~10 eps/sec).

**DELETED 2026-08-19:** `states/curriculum_pool/` (515 MB) was removed in the
consolidation pass. It was Stage-1-only and unused by every multistage run
(`SS_RESTART_PROB=0` throughout). The mechanism below is intact in the env — a
future curriculum run just rebuilds the pool from scratch.

**Why it works (Monte-Carlo confirmed before running):** on a pool shaped like the real one (shallow buckets saturated, deep ones sparse), uniform sampling gives mean restart depth ~55.8 (matches the real ~52); depth-weighting pulls it to ~81 on day one (renormalizing onto the deepest populated buckets) and climbs toward ~90 as deep buckets fill. `POOL_PER_BUCKET=40`, `POOL_DIR=states/curriculum_pool` (in the repo).

**Restart-pool stats trajectory:** avg restart depth 52 → 57 (climbing); frontier 94 (frozen 13k+ eps) → 96 → 98 → 99 → 101.

---

## Entropy

Entropy is floored (~-1.1 to -1.35) and we are **deliberately not fighting it.** A global `ent_coef` bump would inject noise across the whole trajectory, degrading the mastered early game to buy exploration that's mostly wasted on solved cells. The curriculum relocates *where* the agent spends episodes (at the wall) instead — so even a low-entropy policy gets its attempts concentrated where they're needed. The entropy floor is a non-issue under the restart approach.

---

## Verification Lessons / Footguns (learned this session)

- **Verify hyperparameters off the LIVE process, not a disk file.** A `RecurrentPPO.load('model')` readback reads the *on-disk* pickle, which lags the running process until the next AutoSave. A gamma readback printed `0.99` mid-boot only because the new process hadn't re-saved yet; the live model already had 0.997 (proven by the resume print line showing the matching lr). Fix applied: gamma is now in the resume print line, so `grep Resuming logs/ss_trainer.log` shows the live value.
- **`custom_objects` applies on resume; constructor edits do NOT.** Editing `gamma=` in the constructor is a no-op on a resumed run. Use `custom_objects` (and/or explicit attribute assignment after load).
- **`env.data.set_value()` silently no-ops on unknown keys.** It printed "set ... worked" while writing to a name not in `data.json` (wrote nowhere useful). Always confirm the write landed by reading back.
- **A stale on-disk checkpoint makes a "fresh" trainer silently RESUME.** `if os.path.exists(f'{ZIP}.zip'): load` is correct logic but bites when a partial run left a checkpoint behind. Before any fresh run, confirm no `{ZIP}.zip`/`.pkl` exists (move them aside), and **check the startup line says `Starting FRESH ... (random weights)`, not `Resuming`.** (2026-07-02: v2 resumed a 1M-step partial, looked like catastrophic interference in the curve.)
- **sed-deriving a script leaves stale print strings.** Replacing `v1→v2` in filenames doesn't touch log messages like `"Resuming multistage v1"` — which then actively misleads debugging. After a sed-derive, grep the print strings too.
- **The `--curve` (per-1000) view is truth; the cumulative ratio can lie.** `stage_split.py`'s top-line ratio averages over the whole window — during a config change (or a resumed/contaminated run) it reports the ghost of earlier episodes and hides a current collapse. Always read the per-1000 trend rows.
- **All runs shared one `ss_log.jsonl` → analyzers read the GHOST of previous runs.** Bit us twice (v2 launch, v3 launch). FIXED: env now honors `SS_LOG` env var, so each run gets its own log (the trainer sets `ss_log_<run>.jsonl` in the repo). Use `ms_analyze.py --run <name>`.
- **`_cur_stage` ordering footgun (fixed):** init it at the TOP of `reset()` and set during stage-pick — do NOT init in `_reset_state()`, which runs AFTER the stage-pick and clobbers it back to 0. (Symptom: log showed `stage:0` with correct `stage_total`.)

### Found in the 2026-08-19 consolidation pass
- **The sed-derive footgun had already fired, undetected.** `train_ss_multistage_v5.py`'s docstring instructed you to launch `train_ss_multistage_v3.py` and to `pkill -f train_ss_multistage_v3.py`. Following its own instructions would have trained the v3 network on the v3 env while writing to v5's log. The doc listed the footgun; the file still had it. **Root cause: no version control.** Fixed by `git init` + one parameterized trainer.
- **A "config flag" that only zeroes a feature still costs you.** `SS_DROP_POSITION=1` kept the 4 position slots as constant zeros to preserve the obs shape. That convenience shipped 4 dead inputs into v1/v2/v3 and cost those checkpoints their resumability once v4 removed them properly. If a flag makes a feature meaningless, remove the feature.
- **Two probes that silently ran unnormalized.** `transfer_probe.py` and `zeroshot_test.py` both wrapped their `VecNormalize.load()` in a bare `except: pass`. It was ALWAYS failing — stable_retro allows one emulator per process, and the load built a second one. Harmless here only because every trainer sets `norm_obs=False`. `eval_transfer.py` now reads the pickle directly and says which case applies.
- **`transfer_probe.py` set `env.state_path` but not `STAGE_FILES`.** With `SS_STAGES` set, the env would still random-pick a stage on reset, so "run 60 episodes on stage 2" could silently sample other boards. `eval_transfer.py` pins both.
- **There is no main-game trainer on disk.** `train.py` was a stale copy of the SS specialist trainer. The main-game model and `coordinator.py` survive; the training script does not.

### Found in the 2026-09-29 review
- **A "verified fact" can be half-verified.** The `$F000, row=(y+8)%32` grid mapping was checked where Sonic happened to be (board rows 0–23); the wrap for rows 24–31 was never tested, and for months a quarter of every board was invisible. Verify a mapping over its WHOLE range, against an independent count — `smoke_stages.py` now asserts the grid's blue count equals the game's counter.
- **Silent fallbacks turn a broken setup into a "healthy" run.** A missing state file loaded Angel Island; a failing log write was swallowed. Both are loud now. After the repo move, `~/sonic-rl` paths failed exactly this way.
- **Check when the episode actually ends, not just that it does.** Waiting for `special_stage` to drop added ~670 dead frames per death — 44% of all v4/v5 samples, and a γ^675 discount on the death penalty.
- **Old checkpoints stay on the env they were trained on.** `SS_ENV_VERSION=1` reproduces the pre-fix env bit-for-bit; every pre-v6 preset pins it. Feeding a v1 checkpoint the fixed grid would load fine and silently change its inputs.
- **`coordinator.py` (main game) cannot run as-is:** `SS_STREAK_MULTIPLIER` and `STUCK_SECONDS` are referenced but never defined, its special-stage failure penalty waits on a `terminated` that the scenario only raises at lives == 0, and its `CURRICULUM_STATES[2]` now points at special stage 2. *[Rebuilt 2026-09-30 — see "Base game rebuild 2026-09-30".]*

---

## Multi-Stage Groundwork (generalization plan)

**The tension:** the machinery that cracked Stage 1 — sin/cos absolute position encoding + LSTM stage-region memory — is layout-*memorizing*, which is perfect for one stage and partly counterproductive for generalization. Past a point, more Stage-1 training deepens the overfit you'll then have to overcome.

**Plan (gated on the win):**
1. Gate on the **scaffolded (restart) 102 win**, NOT the fresh-start win (which bottlenecks on a clean 0→90, a tail event). Snapshot that model as a known-good Stage-1 solver to fall back to.
2. **Zero-shot transfer probe first:** load the solved model, point env at Stage 2, run with NO training, read fresh-start average. Near zero → memorization, architecture problem (lean on local patch, add stage-ID input). 20–40 → local perception generalizes, multi-stage plan is cheap.
3. **Drip-feed, NOT all-at-once.** Add one stage at a time as a minority share (Stage1 ~70% / Stage2 ~30%), always keeping prior stages represented (cross-task version of the 78% fresh-start guard). Dumping 13 unfamiliar stages = catastrophic forgetting (the NNUE-run ghost). Drip-feed also lets you *measure per-stage transfer*.
4. **Hold a few stages out entirely** as a zero-shot test set — the portfolio-piece generalization experiment.
5. `reset(stage=N)` would weighted-sample which `curriculum_stage_N.state` to warp into, exactly like the restart pool picks depth. You assign the stage label by filename index at load time (see below — there's no RAM stage-number byte to read).

### Special-Stage Selection Mechanism (reverse-engineered this session)

From the disassembly/wiki research + direct RAM probing:
- The game **cycles through stages in fixed order on each Giant-Ring ENTRY, advancing regardless of win/fail.** The selector is a hidden "rings-entered / next-SS-index" counter, **NOT** emerald count.
- `special_stage` ($FFFE48, `|i1`) reads **0 in the main game, 1 in ANY special stage** — it's a FLAG, not a stage-number label. (Initially mis-identified as the stage number; a 3-stage diff disproved it — reads 1 for stages 1/2/3.)
- `chaos_emeralds` ($FFB0): writable but **not the selector** (poking it didn't change the layout). `super_emeralds` ($FFB1).
- **No clean stage-number byte exists in RAM** — `find_stage_byte.py` EXACT filter found 0 hits across 3 stages. Stage identity lives in a large layout data block (~$A800+ region), not a scalar. So: **label stages by filename index at load time, feed N to the policy as a one-hot from the loader** (no RAM read).
- A candidate 0-indexed counter showed up in the LINEAR hunt ($F152-ish region, reading [0,1,2] across 3 stages) — possible "next-SS index" but unconfirmed; needs a 4th stage to distinguish from coincidence.
- **Debug mode:** PAR code `FFFFE2:0101`. A one-shot poke doesn't stick — PAR re-applies every frame; replicating it needs a frame-by-frame hold in the step loop. Debug level-select *can* warp to a chosen special stage directly, BUT requires the level-select menu, which the integration cannot reach (boots mid-Angel-Island, not the title screen). **Net: no clean spawn-any-stage shortcut on this setup.** Capturing stages = playing the game and savestating, which is fine.
- The back 7 (Super Emerald) stages need a full playthrough to Mushroom Hill with 7 Chaos Emeralds — a real grind, deferred.

### Captured Stages — ALL 14 COMPLETE (verified 2026-07-23)

All 14 special stages captured + confirmed distinct via `check_stages.py`: **14 unique RAM fingerprints, 14 distinct blue counts, all `special=1`.** Each has `.state` + `.ram.bin` in `states/`. Backups in `states/backup_captures/`.

| Stage | Spheres | Set | Note |
|-------|---------|-----|------|
| 1 | 102 | front-7 (Chaos) | original |
| 2 | 127 | front-7 | **recaptured 07-23** after accidental overwrite |
| 3 | 118 | front-7 | |
| 4 | 115 | front-7 | Step-2: plateaued ~7.5 |
| 5 | 58  | front-7 | Step-2 held-out |
| 6 | 128 | front-7 | Step-2 held-out |
| 7 | 60  | front-7 | Step-2: the floor, ~2.4 |
| 8 | 64  | BACK-7 (Super Emerald) | |
| 9 | 155 | BACK-7 | **largest board** |
| 10 | 37 | BACK-7 | **smallest board** |
| 11 | 67 | BACK-7 | |
| 12 | 72 | BACK-7 | |
| 13 | 54 | BACK-7 | |
| 14 | 71 | BACK-7 | |

Capture phase COMPLETE. Note the wide size range (37–155 spheres): per-stage normalization (`_stage_total`, captured at reset) is essential, and episode lengths will vary a lot across boards — factor that into per-stage result reading.

**How the back-7 were captured:** grant true emerald state (count=7 + `$FFFFB2-B8` array all=1) → warp to Mushroom Hill (zone `$07`) → die once for clean load → enter rainbow ring in the secret room → Knuckles cutscene → Hidden Palace with 7 gray pedestals → jump each pedestal → Super Emerald special stage → `S` to capture. Hidden Palace is also directly warpable (zone `$16`, or `$17` when entered from a giant ring).

**Important:** sphere counts differ per stage (102/127/118) — the "all stages have 102" assumption was wrong; that's a Stage-1-specific number. Transfer probe needs only 2 stages, so 3 is already enough; the rest waits for the win.

**Capture method:** fresh boot always starts Angel Island with progress 0, so the *first* Giant Ring serves stage 1. To get stage N you enter and *fail* (don't press S) through the earlier stages to advance the cycle, then capture the target. Wait for the sphere count to settle before pressing S (a mid-load capture once read a bad/transient count).

---

## Project layout (after the 2026-08-19 consolidation)

The project used **filenames as version control**: `special_stage_env{,_v4,
.pre_win_fix,.pre_multistage,.pre_stagelog}.py` and
`train_ss_multistage{,_v2,_v3,_v4,_v5}.py`, each sed-derived from the last.
That had already produced a live bug — **v5's own docstring told you to launch
`train_ss_multistage_v3.py` and to `pkill -f train_ss_multistage_v3.py`**, i.e.
run the wrong network against the wrong env while writing to v5's log. The doc
already listed "sed-deriving a script leaves stale print strings" as a footgun;
the fix is to stop sed-deriving scripts.

**The repo is now under git.** Never create a `_v6` or `.pre_x` file again —
branch or commit instead.

### Active files
| file | what it is |
|---|---|
| `special_stage_env.py` | THE env. Env-var driven: `SS_ENV_VERSION` (1 = the pre-2026-09-29 env, bit-identical, for old checkpoints; 2 = fixed, default), `SS_USE_PIXELS`, `SS_PATCH_R`, `SS_DROP_POSITION`. Absorbed `special_stage_env_v4.py`. |
| `train_ss_multistage.py` | THE multistage trainer. Takes a **run name**; `RUNS` holds every run's stages, obs preset, hyperparameters and result. |
| `train_ss.py` | The single-board specialist trainer (`sonic_ss_model_v2`, PAUSED, control). Pins the specialist obs preset itself. |
| `smoke_stages.py` | Pre-launch check that the env reads each stage. Asserts against the **live** obs shape, so it covers every preset. Absorbed `smoke_all14.py` + `smoke_v4.py`. |
| `eval_transfer.py` | Transfer / zero-shot evaluation. Absorbed `transfer_probe.py` + `zeroshot_test.py`. |
| `ms_analyze.py` | THE multistage analyzer. `--run v5` resolves the per-run log. |
| `analyze_ss.py`, `ss_brief.py` | Specialist (single-board) analyzer + its terminal-friendly wrapper. |
| `capture_stages.py`, `check_stages.py` | Stage capture + distinctness verification. Capture phase is COMPLETE; kept for re-capture. |
| `watch_ss.py` | Watch a checkpoint play. Takes a run name (or `specialist`) so the preset always matches. |
| `watchdog.sh <run>` | Keep a run alive across crashes. Was `watchdog_v3.sh`. |
| `coordinator.py` | THE base-game env (rebuilt 2026-09-30): one act per episode from a recorded playthrough; reward = frames faster than the recording. Env-var driven (`MG_*`), set by the trainer. |
| `record_play.py` | Record a playthrough (slow-mo, rewind, `--resume`); saves `demos/play_<stamp>.npz` and proves it replays. |
| `train_main.py` | THE base-game trainer (PPO CNN + backward curriculum per act). `RUNS` is importable (`configure_env`). |
| `smoke_main.py` | Base-game pre-launch checks + the synthetic mini-act for `train_main.py smoke`. |
| `main_progress.py`, `watch_main.py` | Read a base-game run against the recording; watch the agent (or `--you`, the recording) play an act. |
| `data.json` | retro integration RAM map. |

### Model artifacts
- `sonic_ss_model_v2.zip` + `sonic_ss_vecnormalize_v2.pkl` — **specialist baseline/control, ~326M steps, PRESERVE.**
- `sonic_ss_multistage_v{1,2,3,4,5}.zip` + `_vecnormalize.pkl` — one final checkpoint per multistage run. **v1/v2/v3 are NOT resumable** against the current env (they used the old 15-feature vector with 4 dead zeros); they are kept as records.
- `checkpoints_ss_multistage_<run>/` — rolling checkpoints, kept only for live or paused runs (d2, and `checkpoints_main_aiz1/`). Deleted 2026-10-09 (1.3 GB): v5's and d1's rolling checkpoints and `checkpoints_main_smoke/`; their final saves (v5 19.53M, d1 3.36M) are the top-level `.zip`s.
- `sonic_ppo_v10_emerald_priority.zip` — the OLD main-game model (~210M steps), a record only: stalls at AIZ1's first ledge (2026-09-30), and its env/trainer are gone. `attic/main_game/` has its watchers.
- `main_<run>.zip` + `_vecnormalize.pkl` + `main_<run>_curriculum.json`, `checkpoints_main_<run>/`, `mg_log_<run>.jsonl` — base-game runs.
- `ss_log_v{3,4,5}.jsonl` — per-run episode logs. `ss_log.jsonl` is the old SHARED log (specialist + v1/v2 blended; totals are meaningless, read recent windows only).
- `states/curriculum_stage_{1..14}.state` — all 14 captured entry states, backups in `states/backup_captures/`.

### `attic/`
Retired, not deleted. `superseded/` = the old per-version env/trainer copies and
the scripts folded into the consolidated ones. `one_off_tools/` = tools for
problems now SOLVED and written down above (win-fix appliers/verifiers, the RAM
address hunts, the warp/reload discovery scripts). `main_game/` = the map-overlay
tooling and its assets.

**Note found during the pass:** there is **no main-game trainer on disk.** The
old `train.py` is a stale copy of the SS specialist trainer (its docstring even
says "train_ss.py") pointing at the superseded `sonic_ss_model`. The main-game
model `sonic_ppo_v10_emerald_priority.zip` and `coordinator.py` survive, but the
script that trained them does not. Resuming the main game needs that rebuilt.

### Deleted in the pass (5.5 GB)
Rolling checkpoints of finished runs (their final state is preserved as the
top-level `.zip`), `archive_v1/`, `contaminated_v2_attempt/`, the v1 specialist
model, the `.pre_win_fix` backups, and `states/curriculum_pool/` (the Stage-1-only
Go-Explore pool, unused by any multistage run).

---

## Resume / Key Commands

Commands live in the trainer's docstring too, where they cannot drift from the code.

### Multistage runs
```bash
cd ~/Projects/sonic-rl && source venv/bin/activate

python3 train_ss_multistage.py --list          # every run: config, env version, result, checkpoint

# launch v6 (env v2, v4's hyperparameters) — smoke it first
python3 smoke_stages.py v6
nohup python3 train_ss_multistage.py v6 > logs/ss_multistage_v6.log 2>&1 &
grep -E "FRESH|Resuming|preset" logs/ss_multistage_v6.log   # MUST say "Starting FRESH", preset env=v2

# resume it later
nohup python3 train_ss_multistage.py v6 --resume >> logs/ss_multistage_v6.log 2>&1 &

# env v4 (one choice per corner): verify against the demo, then launch d3
python3 smoke_corners.py d3                    # must end "ALL CHECKS PASS" (~4 min)
nohup python3 train_ss_multistage.py d3 > logs/ss_multistage_d3.log 2>&1 &
python3 demo_progress.py --run d3

# a NEW experiment: add an entry to RUNS (pin SS_ENV_VERSION in its obs preset)

# --resume is REQUIRED whenever a checkpoint exists, so a "fresh" launch can
# never silently resume a partial (the bug that faked catastrophic interference
# on the v2 launch). Launching without it refuses, loudly.

# stop gracefully (loses nothing):
touch ~/Projects/sonic-rl/STOP_v6

# keep it alive across crashes:
nohup ./watchdog.sh v6 > logs/watchdog_v6.log 2>&1 &
```

### Before launching — verify the env reads every board
```bash
python3 smoke_stages.py v6              # all 14 under that run's obs preset
python3 smoke_stages.py v6 --stages 1,2 # quick check
```

### Reading results
```bash
python3 ms_analyze.py 20000 --run v6    # one row per board, sorted hardest-first
python3 ms_analyze.py 40000 --run v6 --chunks 8
# Watch: ratio(min/max) near 1.0 = uniform competence; "N climbing / M flat";
# stages covered. Don't over-read before ~20,000 episodes.
```

### Transfer / zero-shot (the generalization question)
```bash
python3 eval_transfer.py v6 --holdout 5,6 -n 80        # zero-shot (needs a run trained WITHOUT 5,6)
python3 eval_transfer.py --model sonic_ss_model_v2 \
        --preset specialist --stages 1,2,3             # the specialist probe (env v1)
# transfer ratio: 0.06 (specialist, memorized) -> 0.75 (2-stage) -> 0.09 (5-stage holdout)
# (all measured on env v1 with a 6000-step cap; the cap is now the env's own 18000)
```

### Specialist run (PAUSED — baseline/control, preserve sonic_ss_model_v2)
```bash
nohup python3 train_ss.py > logs/ss_trainer.log 2>&1 &
grep -m1 "Resuming" logs/ss_trainer.log     # shows the LIVE lr AND gamma
python3 ss_brief.py                          # condensed specialist analyzer
```

### Watch it play
```bash
python3 watch_ss.py v6           # or: v5 / specialist (each runs on its own env version)
```

### Learning from your play
```bash
python3 record_demo.py 1               # play stage 1 (slow-mo + rewind), Enter saves to demos/
python3 train_ss_multistage.py d1      # backward curriculum from your stage-1 run(s)
python3 demo_progress.py --run d1      # catching up with you? beating you?
```

### Base game — learn from your playthrough, then beat it
```bash
python3 smoke_main.py                      # 7 checks, ~7 s; must say ALL CHECKS PASSED
python3 record_play.py                     # play from Angel Island 1 (F5 saves, Enter saves+quits)
python3 record_play.py --resume demos/play_<stamp>.npz   # keep going later
python3 train_main.py --list
nohup python3 train_main.py aiz1 > logs/main_aiz1.log 2>&1 &
grep -E "FRESH|Resuming|demo check" logs/main_aiz1.log
python3 main_progress.py --run aiz1        # start point, match rate, frames vs you
python3 watch_main.py aiz1                 # the agent from the act's start
python3 watch_main.py aiz1 --you           # your recording, same start
touch ~/Projects/sonic-rl/STOP_main_aiz1   # graceful stop
```

### Stage capture (complete — kept for re-capture)
```bash
python3 capture_stages.py        # play to a stage, tap S; [/] sets index
python3 check_stages.py          # verify all 14 distinct
```

---

## Key Learnings

1. **The wall at 94 was a curriculum-targeting failure, not an exploration ceiling.** The restart pool was seeding at depth 52 (uniform sampling over shallow-dominated buckets + no captures above 90), so it never actually practiced the wall. Targeting it (deep captures + depth-weighted sampling) broke a wall frozen for the entire prior run within ~3,000 restart episodes.

2. **Verify, don't assume, that an intervention applied.** The gamma change was believed-done for an entire prior run but never actually took (constructor edit on a resumed model = no-op). One unverified `sed` cost a wrong "horizon ruled out" conclusion. Always read the hyperparameter off the live model.

3. **gamma=0.99 was too myopic for endgame planning** (~1-sphere lookahead). 0.997 (~3–4 spheres) is appropriate for threading the final cells through a red-trail-saturated board. *[2026-09-29 reinterpretation: env v1 paid the death penalty ~670 frames after contact, so at γ=0.99 it was discounted to 0.1% — effectively invisible — and at 0.997 to 13%. The change mostly made death visible.]*

4. **The agent CAN see red** (red/transitional → grid channel 1) — *except in board rows 24–31, which env v1 read from the wrong memory; and springs showed up as red too. Both fixed in env v2 (2026-09-29).* Earlier "it's blind" claim was wrong. Trail deaths (37%, flat) are more likely control/reaction-distance at high speed than perception. The candidate perception lever, if ever needed, is a wider patch, not a red channel — and both break clean resume, so neither fires without a confirmed plateau.

5. **Two LSTM layers + sin/cos absolute position encoding was the right backbone for a SINGLE stage — but the position encoding is the MEMORIZATION HANDLE, and it must be dropped for multi-stage.** VALIDATED 2026-06-30: the specialist's cross-stage transfer ratio was ~0.06 (pure memorization); zeroing ram features 2-5 (`SS_DROP_POSITION=1`) lifted it to ~0.75 (generalizing). The full-resolution absolute `ss_x`/`ss_y` fingerprint let the LSTM map position→action per-board instead of reading the stage-agnostic local grid patch. For any multi-stage run: position dropped (or coarsened — Option B, untested).

6. **Restart-from-frontier (Go-Explore) makes the entropy floor irrelevant** by relocating where episodes are spent, rather than injecting global exploration noise.

7. **Special-stage selection has no clean RAM hook** on this setup — cycle-by-entry counter, menu-gated warps unreachable from a mid-level boot, debug flag needs frame-by-frame holding. Capturing stages = playing + savestating. Stages are labeled by filename index at load time.

8. **Checkpoint resume preserves VecNormalize + policy perfectly** — no warmup on resume.

---

## DreamerV3 Attempt (abandoned — reference)

World-model approach abandoned on the 2070 Super (8GB). World model used ~6.8GB, leaving no room for `imag_horizon > 15`; with horizon 15, sphere collections (every ~85–120 steps) almost never appear in imagined rollouts, so the actor learned to die quickly to minimize imagined negative reward. At 695K steps avg was 2.9 and declining. DreamerV3 needs 16GB+ for this task at scale. Key fixes that worked along the way (for reference if revisited on better hardware): lazy emulator init for stable_retro's one-emulator-per-process restriction, 64×64 frame resize, 256 DiscDist buckets (so zero doesn't sit on a bucket boundary), dropped time penalty, dense per-sphere bonus, reward scaling /2000.

**Takeaway:** PPO + LSTM + position encoding + self-curriculum is genuinely competitive on this task and is the chosen architecture. The curriculum (Go-Explore restart pool) is what addresses PPO's sample-efficiency gap on the sparse deep-endgame region.
