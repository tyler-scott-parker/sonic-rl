# Sonic 3 & Knuckles Special Stage RL — Full Project Context
*Last updated: 2026-10-16 ~10:20 (see CURRENT STATE right below). Older header: 2026-10-10 ~16:50 — PAUSED by Tyler, nothing training. d7 parked at 975,008 steps (main test passed). Env v7 = uncommitted prototype: 11/14 recordings pass, stage 4 = checker bug (fixed, rerun pending), stages 6/10 = Up timing; a turn-queueing env problem needs Tyler's call before any v7 training. Branch `corner-decisions` (see "Current state")*

---

## CURRENT STATE (2026-10-16 ~10:20) — read this first; the dated history below is a log

**Box:** mg2 RUNNING, restarted 2026-10-17 ~10:02 from 86,957,008 with **fight_lead=600 + front_ahead=1500** (d876b7d; live-checked: boss share AIZ2 145-148 / HCZ2 225-227 / MGZ2 272-274, HCZ2 front-share starts 3-30) on top of imitation 'fights' (0c28ac8, MGZ2 read PARTIAL). Claude in charge while Tyler is at work ("I'm heading to work so I'll leave you in charge for a bit. I'll check up on my breaks"; same rules as overnight). Supervisor + watch v2 (/tmp/claude-1000/overnight_watch2.sh) running; pre-committed reads by /tmp/claude-1000/mgz2/lead_ahead_read.sh at +5M (>= 91,957,008, ~13:00) and +10M (~16:00): hcz2_chase_eval k13/k9, boss_timeline HCZ2 224, fresh stats. Waiting on Tyler: MGZ2's ground-phase hits (next MGZ2 step). Pausing mg2: stop watch + supervisor, then touch STOP_main_mg2 after steps appear.

**d10** (special stages, 14 boards, `train_ss_multistage.py d10`): FINISHED 2026-10-16 21:52 at **10,080,272** (board 14's first fresh clear). Board 14 = three skills (corridor survival, the turn east at (13,3), the jump at (13,8)); see the 11:40-12:50 bullet. 13/14 boards have a fresh clear; board 14 doesn't. Its wall (~46-47% of the route) is a MAZE maneuver: bumper bounce in a one-cell corridor, overshoot the junction (13,3) and press Up, turn right, run east, jump at (13,8) over red onto a spring. Learnable with v7 moves (smoke_corners_board14 ALL PASS); the agent just fails it. Live settings: `book_pin={14: board14_2921_96f1dc25}` (the route switch was NOT the cause, but harmless), `SS_STAGE_WEIGHTS='14:6'` (from next resume). **Tyler wants d10 finished before moving on.**
- Resume (only when he's not gaming): `source venv/bin/activate && setsid nohup python3 train_ss_multistage.py d10 --resume >> logs/ss_multistage_d10.log 2>&1 < /dev/null &`; harvest loop `setsid nohup taskset -c 6,7 nice -n 19 python3 -u route_book.py harvest --every 30 >> logs/route_book_harvest.log 2>&1 < /dev/null &`; then `setsid nohup python3 attic/one_off_tools/ss_to_mg_handover.py d10 mg2 14 > logs/ss_to_mg_handover.out 2>&1 < /dev/null &` (on the first fresh board-14 clear it stops d10 + harvest, resumes mg2, starts the supervisor; notes in logs/overnight_20261015.log). Status: `python3 attic/one_off_tools/d9_status.py d10`.

**mg2** (main game, AIZ1-MGZ2, `train_main.py mg2`): RUNNING since 2026-10-16 21:52 (resumed from **64,193,984** by the handover). All settings load from RUNS: boss share 25%, ring share 15% (clip starts for rings 1, 2, 3, 6, 11, 13), front share 15%, time balance, AIZ1 weight 1.0 (it was FORGETTING at 0.3), fight bar 2.5x, imitation 'front', fight_codes (MGZ1's unflagged boss), passive route book `routes_main/` (main_book.py; harvest every 30 min in the watch). Supervisor: `attic/one_off_tools/mg_supervise.sh mg2 720` after the first steps. Status: `attic/one_off_tools/mg_status.py mg2`.
- Where it stands: AIZ1 clears from the start (best 3,986 vs his 5,060) but was forgetting; AIZ2 pointer 143 (out of its fight), HCZ2 225 and MGZ2 280 (in fights), MGZ1 188 (his route doubles back into a gap); fresh-start frontiers AIZ2 ~7-11%, HCZ1 24%, HCZ2 3.5% (blocking one jump at the curve's top gets 24/24 up; then the pit needs a SPIN DASH, he does it twice), MGZ1 28% (ledge staircase, then his route goes LEFT on a carrier at (5296,1060); the agent presses right), MGZ2 14%. Map of every act's stuck spots: logs/wall_atlas (attic/one_off_tools/wall_atlas.py). HCZ1 drowning down to 2.5% (no bubble compass needed).

**Rings / emeralds:** only 7 Chaos Emeralds before MHZ: options = rings 4, 5 (found unaided) + clips 1, 2, 3, 6, 11, 13. Super Emerald clips (for when those zones train): MHZ 23, 24, 25, 29; FBZ 32; LRZ 36, 38 x2, 39. Don't ask Tyler for CNZ clips ("they aren't hyper rings"). Tools: `ring_clips.py` (record), logs/ring_maps/v2 (maps), `main_book.py`, `chain_verify.py`.

**Start-up cache:** trainers load start points from `start_cache/` (start_cache.py; check_start_cache.py); a code/settings change rebuilds once in the trainer. **Footguns:** emulator snapshot bytes differ between processes for the same game state (compare by emulation); `pgrep -f X` matches your own shell (use `ps -eo pid,args | grep "[X]..."`); a STOP file touched during a trainer's start-up is deleted, not obeyed; `ring_clips.py --export` must never rewrite start files (clips check sha1); one emulator per process; importing train_ss_multistage runs its argparse (exec the head instead); a running bash script keeps its parsed loop (restart to apply edits); check per-try outcomes, not curriculum summary lines, before naming a cause.

**Open for Tyler:** old datasets bc_v1-v3 (~9.6 GB) keep/delete; Heroic's `default` prefix (906 MB, Elite's) keep/delete.


2026-09-30: Tyler asked to get the base-game coordinator set up properly, using a recording of his own playthrough as the agent's starting point and having it improve on his play. Done as a rebuild: `coordinator.py` (env), `record_play.py` (recorder), `train_main.py` (PPO + backward curriculum; `RUNS` importable), `smoke_main.py`, `main_progress.py`, `watch_main.py`; old watchers moved to `attic/main_game/`. Committed 2026-09-30 on branch **`base-game`** (b1cef03..3afe56e, incl. the stuck cutoff, Tyler's stage-1 demo, the pre-committed test, SS env v3 and run d2), stacked on demo-learning → env-v2-fixes; none of the three merged to master or pushed.

Facts, design and evidence live in the repo's context doc, section "Base game rebuild 2026-09-30" — read that, don't re-derive. Key points: act clear = the in-game timer stops with Sonic alive in the level; death = player routine 6/8/$C on that frame; reward = frames faster than his recording along its own route (+600 clear, −1800 death); data.json mislabels level_end_bonus/shield/special_stage. The old v10 model is a dead end (stalls at AIZ1's first ledge).

Smoke run on a scripted mini-act: curriculum walked 92%→0%, then the agent beat the script (median +45 frames, 71% of clears, best +118) in ~10 min.

**Why:** a fresh session could otherwise re-audit the old coordinator, trust data.json's labels, or look for the work on the wrong branch.
**PAUSED by Tyler 2026-10-09 13:58; nothing is training.** d2 ran 12:28–13:58 that day (4.05M→5,148,624 steps, stopped cleanly via `touch STOP_d2`; log appended after a "=== resumed 2026-10-09" marker). Resume with the same `train_ss_multistage.py d2 --resume >> logs/ss_multistage_d2.log` command. Before that: paused by Tyler 2026-10-01 17:33. d2 stopped cleanly at 4,050,816 steps (resumed 14:25 at 2.06M). At 4.03M: start point 79%; pace slowed from ~150K steps/point to ~650K (82% held 1.0M steps, then broke); match rates 3–7% vs the 25% needed (8 of last 32); fresh-start mean 21.6, best 69, 0 clears; ~176 steps/s while Vanguard Galaxy used the GPU. Earlier that day Tyler had paused everything. d2 was resumed 2026-09-30 21:10 and stopped cleanly at 2,060,008 steps: the first start point fell at ~614K (milestone d1 never hit), now 82%; fresh-start best 41/102, 0 clears; last bucket's match rate dipped to 2%. Next when resumed: `train_ss_multistage.py d2 --resume >> logs/ss_multistage_d2.log` (append, not `>`); check at ~6–7M whether 79% is a real stall. Footgun: a `pgrep -f "train_ss_multistage.py d2"` wait loop matches its own shell and never exits — match the log instead. After that: `train_main.py aiz1 --resume`; Tyler's AIZ2 recording is unfinished. The context doc's "Current state" has the commands.
**How to apply:** check `git branch` in ~/Projects/sonic-rl first (work is on base-game). Tyler recorded AIZ1 (demos/play_20260930-153946.npz, 1:25.43) and run `aiz1` was launched 2026-09-30 (nohup, logs/main_aiz1.log); read it with `main_progress.py --run aiz1` before advising. He stopped 0:08 into AIZ2 inside a hidden giant-ring room (resume + 3 Backspace presses gets back before the tree). In AIZ1 the timer stops at the miniboss kill, not the signpost. aiz1 stalled at 83% (a wall climbed by going back up-left; 84% of frames were 47 s timeouts) → added `stuck` truncation (600 frames without route progress), resumed at 4.35M steps (committed 2dd3a0b); PAUSED by Tyler at 5.0M steps (pointer 69 = 82%) to give d1 the machine — resume with `train_main.py aiz1 --resume`. Judge the demo approach by the PRE-COMMITTED test in the context doc ("Pre-Committed Criteria": first fresh-start clear of AIZ1 / stage 1 within 50M steps each; attribution caveat = old baselines ran on buggy envs). Run d1 (SS stage 1, env v2) on Tyler's flawless stage-1 demo FAILED its first start point (per-frame turns + hop-to-survive stall) → built SS env v3 (4-frame choices, 600 frames without a sphere = death; v1/v2 verified bit-identical) and launched d2 alone on the machine. Still unverified in the emulator: act 2's capsule clear and mid-act timer-stopping cutscenes. Special stages are neutral truncations until the SS agent handoff is built. Relates to [[sonic-rl-env-bugs]] and [[measure-before-blaming-code]].

(Full detail: the dated bullets under "Current state" below; newest last.)

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
- **2026-10-12 ~00:30: boards 3 and 9 (the last two on his times) look near-optimal; route_polish.py (new).** (1) **Board 9:** d10's fresh clears are all 3,008 frames, ONE route: replayed side by side (scratchpad b9_compare.py) the agent follows Tyler's route cell for cell and frame for frame through sphere 153 (frame 2,884), then takes the last two in the other order (+20). His ending is boxed in: the last blues (29,21)/(29,22) sit inside a ring of bumpers; the only way in is a jump over the red at (29,27) onto the spring at (29,26), which throws Sonic to the empty (29,20), the bumper at (29,19) sends him back east over both. (2) **Board 3:** d10's median fresh clear 5,496 vs his 3,944; route_search (every start point, ~15 sweeps) cleared it 60-80 times per sweep, never faster. His ending closes a loop round a 3x4 blue block (top side already red) with the minimum two turns. (3) **A turn costs ~16 frames:** Sonic stops while the heading turns 90 degrees (heading 128 -> 64 over 16 frames at a fixed position). (4) **route_polish.py:** breadth-first over every input on every frame from a book frame, identical game states merged (RAM minus the input-history ring $E500-$E5FF and the controller bytes; a mid-cell direction press buffers a turn at $E42A and is kept), beam 3000 with per-(cell, heading, jump) variety. Sanity check: it searches up to the book's own time and finds it. **From 3,830 on board 3 and 2,880 on board 9 it finds exactly 3,944 and 2,988: nothing faster in those finishes.** Running overnight from 3,600 (board 3) and 2,700 (board 9). The second route search (boards 3, 9) was stopped to free its core; search A (3, 9, 14) continues. **Overnight (~02:30-04:48):** with a purely greedy beam those longer runs lost the book's path and answered nothing (both ended boxed in with 2 spheres left), so half the beam now follows the book (its own path always, then states exactly on its track, most frames ahead first; ff93e82). Rerun: **from 2,700 on board 9 and 3,600 on board 3 (288 and 344 frames out, beam 3000, ~4,500 distinct states per frame) it finds exactly 2,988 and 3,944: nothing faster.** route_search A finished its 20 sweeps (01:29) without beating either. Running from 04:50: board 9 from 2,400, board 3 from 3,200.
- **2026-10-12 ~06:30: FINISH LINE CHANGED by Tyler.** Told boards 3 and 9 are still on his times, he said: "That's ok. All we need is that final clear", i.e. d10's first fresh clear of board 14, the one board without one (at 72.6M frames: 13/14, board 14 best 69/71 from fresh, start point ~52%). Beating his time on all 14 is no longer required; the book already gives a verified route (= a guaranteed emerald) on all 14. When board 14 gets its fresh clear, the main game starts (mg1, his "i want to start knocking that out"). The board 3/9 searches from 2,400 / 3,200 were stopped unfinished.
- **2026-10-12 ~07:35: MAIN GAME STARTED, d10 PAUSED.** Tyler: "can we come back to board 14 overnight? i want to start working on the main game. would you be able to put a main game run on my screen so i can follow along?" d10 stopped cleanly at 4,172,816 steps (board 14: start point 48%, best 69/71 from fresh); resume it overnight with `--resume`. mg1 launched FRESH (`nohup python3 train_main.py mg1 > logs/main_mg1.log 2>&1 &`); criteria still Claude's proposal ("Run mg1"). Viewer on Tyler's screen: `python3 watch_main.py mg1 --live` (new: reloads the save before each episode, starts at the curriculum's current start point, F = act start), log logs/watch_mg1.log. Watch: scratchpad mg_watch.sh → logs/mg1_watch.log via attic/one_off_tools/mg_status.py. At 139k steps AIZ1 was already at start point 70 (83%), next to aiz1's wall at 69.
- **2026-10-12 ~07:50-08:25 (Tyler gaming; mg1 STOPPED at 975,000 steps, CPU only):** (1) **mg1 at the wall** (`probe_aiz1_wall.py 69 30 2 mg1`): 28 stuck + 2 timeouts of 30; reached the ledge 14/30, over the top 0/30 (early: the gate is 5M). (2) **Board 14 from fresh** (d10's save, 40 sampled runs, scratchpad b14_fails.py): all 40 died, spheres 7-45 of 71 (11 at 7); the 4x4 block at rows 2-5, cols 15-18 always left. The status table's "best 69/71" is a rare high; a fresh clear is far off. (3) **Tyler's concerns** ("i used items to rush my way across dangerous areas ... fire power up during the eggman fight in the lava ... star power ups ... areas in levels that loop until a condition is met. plus, the barrel in carnival night"): `attic/one_off_tools/mg_hazards.py` (item stretches from $FFB02B, revisited 128-px blocks, Up/Down rhythm) and `mg_items_needed.py` (each star / fire-shield stretch replayed with the item taken away). Without the item his inputs DIE in LRZ1, LRZ2, LRZ3 (fire shields; LRZ3 = the lava boss), FBZ1 and DEZ1 (stars), get STUCK in AIZ1 (star), take 1 hit in MGZ2 and LBZ2 (stars); not needed: HCZ2/CNZ2 stars, MGZ2/DEZ2 fire shields. Start points inherit his item; behind the monitor the agent must break it. Shields carry between acts (AIZ2 and HCZ2 begin with his bubble shield). Up/Down rhythm: CNZ2 x ~6,000 (14 switches over 23 s, holds 41-51 frames) and x ~11,500; MHZ1 x 4,414 (holds 17-23): all far above the 4-frame action repeat. Most-revisited blocks are boss arenas (AIZ2, LRZ1, LRZ3, DEZ3, ICZ1's end); the loops Tyler means are not identified yet (asked him).
- **2026-10-12 ~08:45: the two loops Tyler named, tested (`attic/one_off_tools/mg_loops.py`).** Both acts wrap vertically at y 2,048. **ICZ1** (pit): his route loops 4 times at x ~25,450 (wraps at demo t 10,231 / 10,782 / 11,688 / 12,449), escape = climbing out at ~12,825-13,275. **SOZ2** (crossing sand slides): 3 wraps at x ~4,130 (2,997 / 3,583 / 4,745), escape = jump+left across at ~4,855-5,260. RouteProgress (window -300/+600 demo frames, ties by clock) matches the agent's lap n to his lap n, so his laps are paid as he ran them and escaping early is paid as a shortcut (reacquire). **Missing the escape:** SOZ2 from his last lap holding Down: +308 demo frames of credit while sliding past his escape spot, then nothing, `stuck` after one more lap, return -1,068. ICZ1 from his lap 3 replaying his lap-3 inputs over and over: lap 1 exact (return 0), the extra lap is matched to his lap 4, ~+230 demo frames while passing his escape ledge, then nothing, `stuck` at 2,160 frames, return -751. So: no payment for looping past his lap count, a cut ~10 s after new ground runs out, and one small false credit (~4-5 s of his time) for passing the escape spot without escaping. Not fixed: the escape still pays far more.
- **2026-10-12 ~08:50: mg1 RESUMED** (Tyler: "you can have the box for now") from 975,000 steps. Start point moved 69 -> 68 at ~1.04M (a window of 25% matches at 69), but it is NOT past the wall: probe at 1,275,008 steps from start point 69: ledge 26/30 (was 14/30 at 975k), over the top 1/30 (1 clear); at 1.37M the last 500 episodes were 477 stuck, 1 clear. So the proposed wall gate ("still at start point >= 69 at 5M") would pass on a lucky window: read the climb rate (probe_aiz1_wall.py 69 30 2 mg1) instead. Viewer on Tyler's screen: `watch_main.py mg1 --live --fresh --best 20` (Tyler: "can you have the watcher show me its best runs from the start?"): 20 sampled tries off-screen, shows the best (replay checked identical).
- **2026-10-12 ~09:10: mg1 SPECIAL STAGES ON** (Tyler: "i'm curious to see if the agent will run into the giant ring that's near the very start of the stage", then "can we have special stages turned on?"). Before: all 61 giant-ring exits were fresh starts at AIZ1's ring (7112, 1216), ~5% of fresh starts, 12-17% of his route. train_main RUNS now take `ss` ('truncate' default, 'pilot'); mg1 resumed with ss='pilot' at 2,165,608 steps: the book's board-1 route plays it (guaranteed emerald), refund makes a won stage cost 0 net (check_ss_pilot: cost 4,234 frames, paid back), the episode goes on from the ring. check_ss_pilot.py ALL CHECKS PASS (logs/check_ss_pilot_20261012.log). Viewer: hooks the emulator's step so stages are drawn and heard; G / --ring shows a try that entered a ring.
- **2026-10-12 ~09:40-10:00: phone stream + footage queue.** `watch_main.py mg1 --live --fresh --best 20 --ring --stream 8790` (stream_out.py: HLS via ffmpeg on the tailnet address, page with switch buttons; Tyler watches on his iPhone); watch_tries.py finds best-of-20 runs in its own process, 2 queued, so something always plays. **mg1 stall ALERT** at 4.31M: start point 68 unmoved 3.3M steps, 486/500 stuck at the wall. Cause found in the curriculum: starts are drawn from [pointer, pointer + 4] and every one of them counted toward moving the pointer, so easier starts past the climb moved it while the climb itself stayed ~1/30. **Tyler: "Let's go with your suggestion"** (~10:05): mg1 stopped at 4,459,616 steps, pointer set back to 72 (demo t 4,320, on the rock pile just before the up-left climb; old file logs/main_mg1_curriculum_before_ptr72.json), and RUNS['mg1'] curr_at_pointer=True, curr_min_n=32: only episodes starting AT the pointer count, a full window of 32 with >= 25% matched before it moves. Simulated: the old rule moved 72 -> 70 on 40 easy later starts alone; the new rule holds, then moves on 8/32 at the pointer. Resumed. mg_status.py shows "at the start point itself: k of the last 32".
- **2026-10-12 ~10:15-10:50: giant rings and emeralds.** Tyler: "I'm wondering if we can implement a system that rewards the agent for getting to its farthest location quickly. The main problem is that I don't want to discourage it from finding big rings", then "In a perfect world, I'd like the agent to get all 14 emeralds and beat the secret final stage ... The agent will need to find some on its own". The route reward already pays frames ahead of him at its farthest point; the gap was the ring TRIP (AIZ1's ring is 241 px off his route: mg1's stage episodes had median 1,240 off-route frames vs 296). **MG_SS_DETOUR=refund** (coordinator): when an emerald is won, the frames from the last route progress before the ring and from the stage until the next progress are forgiven (each <= DETOUR_CAP 1,200). **check_ss_detour.py ALL CHECKS PASS**: same game under both rules; control return identical (-1,377); ring try -2,175 -> -827 = 368 + 980 forgiven. mg1 resumed at 5,476,224 steps with ss_detour='refund', emerald_bonus=300 (5 s per emerald on top). **Giant ring map** (`attic/one_off_tools/giant_rings.py`, ROM object layouts: pointer table $1E3D98, object id $85; giant_rings.json): **77 rings** (AIZ 4, HCZ 4, MGZ 11, CNZ 11, FBZ 4, ICZ 5, LBZ 8, MHZ 11, SOZ 11, LRZ 8; none in SSZ/DEZ/DDZ); within 96 px of his route 2 (AIZ2 x 14,008; LRZ2 x 8,512), within 400 px 46, farther 29. 14 emeralds = 7 Chaos (S3 zones + FBZ before 7) + 7 Super via Hidden Palace from S&K zones. **AIZ1 wall, exact hurdle** (at-pointer data, ~2,400 episodes since the reset): start 72 (rock pile) 363 eps, 357 stuck, 0 matched; start 73 (one second on, on the ledge) 45% clear, 20% matched; 74-76 ~70% clear. The up-left jump from the rock pile onto the ledge is the whole wall. Idea if it doesn't come: a start point between 72 and 73 (half a second) as a stepping stone.
- **2026-10-12 ~11:00: AIZ1 wall objects** (Tyler: "I think there is normally a ledge there that collapses. If the ledge breaks before you clear the area, then you may need to move far enough away so that the ledge despawns and respawns"; also Launch Base: Tails rode a spinning elevator early and he had to leave and re-enter). ROM layout (AIZ internal act 2): object $04 (collapsing platform) at (3696, 872) and, flipped, at (3857, 807), both on his climb; spikes $08 at (3664, 922) and (3889, 856); rocks $05 under the rock pile. `attic/one_off_tools/mg_wall_objects.py` (object slots in RAM by position): his inputs from start point 72 break the ledge platform 104 frames in and the top one at 160, each just after he leaves it. mg1 (5.78M) from 72, 20 sampled tries, all stuck: 13 with both platforms intact, 2 where the ledge collapsed under it (fell to y ~1,170-1,195), 5 where a platform vanished only after it ran 400+ px back left (unloaded off-screen, not broken); 2 tries reached the top level (y ~760) and still failed. So broken platforms are a minor cause; the jumps are the problem. Respawn-distance test inconclusive (Sonic came back to different places). Risk noted: leave-and-return tricks the agent triggers off his timing must be improvised over visited ground, which `stuck` cuts after 10 s.
- **2026-10-12 ~11:25: stepping stones up AIZ1's climb** (Claude recommended adding them now rather than waiting, since 72 had 0 matches in 363 starts over ~1.3M steps and aiz1 sat at this wall 3.7M steps; Tyler: "Go ahead"). coordinator MG_EXTRA_STARTS ("ACT:t,t;..."), RUNS 'extra_starts': extra start points at demo frames 4,335 / 4,350 / 4,365 (take-off, mid-jump, on the ledge), regular ones unchanged (start points 0-72 keep their numbers; new 73-75; old 73+ now 76+; 87 in all). **check_extra_starts.py ALL CHECKS PASS**: regular start points identical; his inputs from 72-76 clear on his frame with return 600. mg1 stopped at 5,995,432 steps, pointer set to 75 (old file logs/main_mg1_curriculum_before_extras.json), resumed. Phone stream restarted so its env has the same numbering. If this works: automatic stepping stones wherever the at-pointer rate stays ~0 while the next start point is fine. **First read (~12:10):** start point 75 (t 4,365) is Sonic standing ON the collapsing ledge, (3,745, 839): in every try (and his run) the ledge breaks 60 frames after that start (his: 56, when he is already up at (3,834, 767)). mg1 at 6.5M from 75: 240 starts, 0 of the last 32 matched; `mg_wall_objects.py mg1 16 75`: 15 stuck, 1 slow clear; Sonic peaks at y ~740 (higher than his 766) but does not get right onto the top before the ledge drops. One second on (t 4,380, mid second jump) it matched 20%. The crux is the second jump's take-off within that second.
- **2026-10-12 ~12:40: can the agent's controls do his jumps?** Tyler: "I think the amount of time that it got stuck before is what's making me feel skeptical. This game is riddled with precise jumps across all many of its stages. This will be far from the only one we'll have to deal with." `attic/one_off_tools/quantized_his.py` (his per-frame inputs rounded to the 9 actions x 4 frames, all 4 window alignments): from AIZ1 start points 70, 72, 75, 76, 80 the rounded replay clears in 3 of 4 alignments, within ~30 frames of him (72: 771 / 740 / 770 / timeout; his 740). From 30 and 60 it drifts and sticks (open-loop: once a jump lands a frame late his later inputs don't fit), which says nothing about the controls. So the climb is a discovery problem, not an action-space limit. **Proposed (not started): imitation first** — a dataset from the whole-game recording (every frame: the agent's observation + his rounded action, ~425k samples, all alignments), supervised pretraining of the policy (and the value head on his discounted returns), then PPO with the curriculum + a decaying imitation loss; stepping stones / slow-clear rule as backups. First test: the pretrained network alone from AIZ1's rock pile, and how far it gets from each act's start.
- **2026-10-12 ~10:50-11:45: imitation first, built and tested** (Tyler: "This sounds great!"; he gamed ~10:50-11:25: mg1 STOPPED at 6,874,240 steps, phone stream stopped, CPU-only work on cores 6-7 at nice 19). `build_bc_dataset.py` -> datasets/bc_v1 (git-ignored, 3.2 GB): every frame of his three recordings, the env's own process_frame image + his next 4 frames rounded to the 9 actions: **406,960 samples**, 29 acts (labels: right 34%, noop 29%, left 17%, down 8%, jump+right 6%, rest <= 2.5%; 86% of samples all 4 frames agree). `train_bc.py bc1 --init mg1 --epochs 3` (2 CPU threads, 9 min): held-out accuracy 11.2% (mg1 as is) -> 52.3% (plateau from ~0.3 epochs; jumps 0-20%: their timing is ambiguous frame by frame). bc2 (`--soft 4 --jump-weight 3 --epochs 1`): 48.5%. `eval_bc.py` (no RL, AIZ1 climb, 10 sampled tries): from 75 (the ledge) mg1 0/10, bc1 4/10 near his pace (735-841 vs 695), bc2 3/10; from 76 mg1 8/10 (slow), bc1 7/10, bc2 6/10 (one 655 < his 680); from 72 all 0/10; from each act's start (mg2's six) all get stuck 1-20% in (copying alone compounds errors). **Run mg1bc STARTED 11:27** (Tyler: "you can have the box"): mg1's config exactly, init='bc1', start point 75 copied from mg1. **Within 82k steps it matched 8 of 32 at 75 and moved to 74** (mg1: 0 of 32 there in ~0.9M). **PAST THE WALL in 17 minutes:** 75 -> 74 at 57k steps, -> 73 at 295k, -> 72 (the rock pile) at 725k, -> 71 at 1.10M (11:44); clears averaging 470-520 frames slower than him, inside the match margin. mg1 + aiz1 had 0 matches from the rock pile in > 7M steps. Imitation pretraining is what broke it (same config, same start point, same stepping stones; only the starting network differs).
- **2026-10-12 ~12:00-12:20: second spot + drowning.** mg1bc walked to start point 68 (t 4,080) by ~2.0M steps, then the stepping-stones flag: 299 starts at 68, 259 stuck, 4 clears, 0 of 32 matched; from 69 (t 4,140) 77% clear. Stuck at x_max 3,264 in 247 of 259: his route reaches (3,267, 1,011) at 4,090, loops up and back left to (3,145, 937) and drops back at 4,140, through a path swapper (object $02, subtype $91) at (3200, 960); presumably the swap puts him on the collision layer where the way right is open. Stepping stones added at 4,095 / 4,110 / 4,125 (mg1bc stopped at 2,822,200, pointer -> 71 = t 4,125, old file logs/main_mg1bc_curriculum_before_loop_stones.json; 90 start points; check_extra_starts.py mg1bc ALL CHECKS PASS). Within ~100k steps: 7 of 29 matched at 71. **Drowning** (Tyler: "should we measure drowning as it's own type of death?"): already a death (routine $C); now logged as cause 'drown' (episode log 'cause'; mg_status counts it). `attic/one_off_tools/mg_air.py` on his whole-game run: underwater AIZ2 9 s, HCZ1 94 s (longest stretch 36 s, lowest air 9: into the countdown), HCZ2 74 s (lowest 19), CNZ2 55 s (lowest 17), LBZ2 84 s (lowest 9: countdown); nowhere else. The agent can't know its air (4 frames, no sound) until the digits show: **proposed, not built:** draw an air bar into a corner of its 84x84 view (same network shape, so bc/mg weights still load; dataset rebuild + bc retrain ~15 min), before mg2 (HCZ1/HCZ2).
- **2026-10-12 ~12:20-12:35: AIR BAR BUILT** (Tyler: "let's add the air bar"). coordinator: RAM var mg_air (Player_1 + $2C), `obs_frame` = process_frame + (MG_AIR_BAR=1) a 2 x 30 px strip in the top-right corner of each 84x84 frame, one white pixel per second of air left (30 out of water); used at all three places frames are made (start points, steps, after a special stage); RUNS 'air_bar' (default off: mg1/mg1bc unchanged). datasets/bc_v2 = the 406,960 samples with the bar (Hydrocity 1 bar lengths 10-30, full 80% of the time; checked by eye on upscaled frames: short at air 10, longer at 20, full on the title card). **bc3** = mg1bc's network (3.42M steps; its held-out imitation accuracy had fallen to 13.9% under RL) + one epoch on bc_v2: 50.3%. Water test, `eval_bc.py` HCZ1 from 11 start points x 5 sampled tries: bc1 (no bar) drowned 7 of 55, bc3 (bar) 8 of 55; both mostly stuck. Imitation alone doesn't read the bar; its value should show under RL (drowning costs the death penalty). mg2's draft now: init='bc3', air_bar=True, AIZ1's six stepping stones (copy mg1bc's curriculum at launch).
- **2026-10-12 ~12:35-13:24: mg1bc PAUSED for Tyler's gaming** at 4,095,808 steps, start point 70 (t 4,110, mid-jump of his loop-back: 2 of 32 matched; it crosses the path swapper fine, then hesitates and drifts left). **Imitation during RL BUILT** (Tyler: "yes please. go ahead and build that"; CPU only, cores 6-7, nice 19): train_main `ImitationCallback` (RUNS 'imitation': dict(data=..., weight 1.0, decay_steps 3M, steps 8, batch 256, behind 120, ahead 600)): after each PPO update (at the next rollout's start), 8 gradient steps on the weighted log-likelihood of his action, half of each batch from his frames around each act's current start point, half from the run's acts (its own recordings only); own Adam; the weight scales the STEP SIZE (Adam undoes loss scaling: the first version multiplied the loss, which would barely have faded) and falls linearly to 0 over 3M steps from when it starts; the value head isn't trained by it. **check_imitation.py mg1bc ALL CHECKS PASS**: 5,060 of his AIZ1 frames; frontier t 3,991-4,710 around start point 70; 12 updates: loss 1.91 -> 0.34, accuracy 37% -> 88%; step size halved at half the decay; no steps at 0; value head untouched; 1,024 steps inside PPO's real loop with imitation/* logged. Enabled in RUNS for mg1bc (bc_v1, from its next resume) and mg2's draft (bc_v2). **Resumed ~13:24 with imitation: off start point 70 -> 69 at 4,198,208 steps, ~100k steps (~4 min) after the resume** (before: 338+ starts at 70, 2 of 32 matched); at 69, 5 of its first 16 starts matched. (Times in this doc for the air bar, the pause and the build were first written 20-60 min ahead of the clock; corrected from commit times.)
- **2026-10-12 ~13:35: AIZ1 CLEARED FROM THE VERY START — FASTER THAN TYLER.** mg1bc passed the loop-back (69 -> 68 at 4.27M with 44% matched; 68 -> 67 at 4.35M with 41%, where it had had 0 of 32) and by ~4.5M steps its fresh starts (25% of episodes, played from the act's start) cleared AIZ1 4 times: 1:59.71 (7,115 frames), 1:19.18 (4,682), 1:14.96 (4,431), **1:14.65 (4,408 frames) vs his 1:25.43 (5,060): 10.8 s faster** (game clock; each ends at the miniboss, x ~4,600; no special stage in them). mg1's agreed-in-spirit main test (a fresh-start clear of AIZ1 within 50M steps) met at ~11.4M steps in all (mg1 6.9M -> bc1 -> mg1bc 4.5M). What did it: imitation (bc1 init + the fading pull aimed at the stuck start point) + the backward curriculum + stepping stones at the two hidden spots.
- **2026-10-11 ~21:55: Tyler gave Claude the box for the night; d10 resumed (1,153,008 steps).** Book at 12/14: board 14 fell to route_search on d10's demo_mix save (3,161 vs his 3,427; that save cleared board 14 from 16-22 of 83 start points per sweep vs 3 before); boards 2 (3,304) and 6 (4,294) improved too. Left: boards 3 and 9. **Tyler: start the main game (mg1) if the book completes tonight** (see Run mg1).
- **2026-10-11 ~20:45: d10 PAUSED (Tyler: "let's pause on this for now", then "i'm going to play the witcher. you can do any cpu work that you need").** d10 stopped gracefully at 1,153,008 steps (demo_mix on since 1,103,800); resume with `nohup python3 train_ss_multistage.py d10 --resume >> logs/ss_multistage_d10.log 2>&1 &` only once Tyler is done gaming. CPU only meanwhile, pinned to cores 6-7 at nice 19: `route_search.py 3,9,14 --sweeps 20` (d10's 1,153,008 save; logs/route_search_A.log) and the harvest loop (logs/route_book_harvest.log). Book still 11/14 (left 3, 9, 14); route_search had 10 sweeps on 3 and 9 with many clears but none faster than his.
- **2026-10-11 ~20:26: d10 resumed with `demo_mix=0.3` (Tyler: "Yes" to the SS_DEMO_MIX proposal, see Run d10).** Stopped gracefully at 1,103,800 steps, resumed with `--resume`; the banner says "demo mix: 30%". Book at 20:25: 11/14 faster than Tyler (left 3, 9, 14); boards 1 (2,869), 8 (2,527), 10 (2,079), 13 (3,531) improved since 17:45 (route_search B and d10). The old d10 process predated the splice-label fix, so board 10 and 13 book entries (and history/board13_3558) said "Tyler's frames to there" for agent-route splices: relabelled (source chain via the parent route; inputs untouched, meta 'relabelled'). The curriculum copy name hashes the file, so d10 re-adopted those two identical routes.
- **2026-10-11 ~17:00 (Tyler away ~2 h, "you have full control"):** (1) **The live book switch WORKS in d10:** at 16:47 the harvest's new board-2 (3,545) and board-6 (4,428) routes were adopted mid-run, start points carried over at frame 58 / 2,302. (2) **smoke_corners d10, first run: 8 FAILED — a test artifact, not d10:** it read the LIVE book, and the 16:05 harvest replaced boards 4, 5, 6, 7, 8, 11 while it ran (its env had built start points from the old routes; exactly those boards failed, board 2 changed too but had been checked before the swap; d10 itself trains from content-named copies and can't see this). smoke_corners now freezes the book at its start. (3) Board 7's route also showed a corner smoke_corners read as **'up, then jump' — a game mechanic the agent found: a JUMP out of a backward run turns Sonic round mid-air** (frame 3,137 jump, 3,141 speed −5,120 → +5,120, no Up for 119 frames). smoke_corners counted every backward→forward flip as Up; now only with Up pressed within 2 frames (all 10 of Tyler's flips are on the Up frame itself). **Rerun with both fixes: ALL CHECKS PASS** (all 14 book routes as of 16:47, 63 min; logs/smoke_corners_d10.log). (4) **`route_search.py`** (new): the agent from EVERY start point of each book route (training's curriculum only starts near each board's pointer, so late stretches stopped being tried); clears that beat the book are saved by the env's own capture, the harvest promotes them. Running at nice 19: A = boards 1, 3, 9, 12, 14 (10 sweeps, logs/route_search_A.log), B = the other nine (5 sweeps, _B.log). Board 9 trial: 43 clears from 74 start points, none under 2,988. **It works: first sweeps took board 1 under Tyler (2,882 vs 2,925, the agent from frame 2,656 of his route) and board 12 (2,516, then 2,491 vs 2,702), plus 2, 5, 6, 7, 8, 11 further → 17:45: faster than Tyler on 11/14 (left: 3, 9, 14).** Labels: candidates from processes with SS_LOG=/dev/null said 'null' and every splice said "Tyler's frames to there" even on an agent route; the env now records the parent route's own source (`SS_ROUTE_TAG` names the finder) and the affected book entries were relabelled (inputs untouched; 'relabelled' in their meta). (5) special_stage_env: a worker re-reads the book's frames when its demos switch (next resume).
- **2026-10-11 ~16:35: d10 LAUNCHED (Tyler: "Let's go ahead and start it").** d9 stopped first at 5,059,832 steps (~96.5M frames; 10/14 boards fresh-cleared, main test unread by Tyler's choice). d10 started from d9's save: all 14 book routes passed the trainer's replay check, every board's start point carried over at d9's frame, first update clean (GPU ~65%, 0.8 GB). Started before `smoke_corners.py d10` finished (~2.5 CPU-h, still running beside it; same env as d9, whose smoke passed; result → logs/smoke_corners_d10.log). Watch: `attic/one_off_tools/d9_status.py d10` (now handles d10: book routes, Tyler's times from his recordings, the book line, the 50M-frames-without-improvement alert; gate alert d9-only); harvest loop still running beside it.
- **2026-10-11 ~15:50: run d10 DEFINED and checked, NOT started (Tyler: "Yes that sounds great" to starting d10's curriculum along the route book's routes).** d10 = d9's save + SS_SPEED_BONUS=7 + `demos='book'`: the backward curriculum runs along the book's current best route per board and switches live when the harvest promotes a faster one. `check_book_curriculum.py d10` ALL PASS, `check_route_book.py` ALL PASS after the env change; smoke_corners d10 in logs/smoke_corners_d10.log. Launch timing is Tyler's call (d9's agreed main test vs switching now) — section "Run d10" under Pre-Committed Criteria. **Book at 16:05: faster than Tyler on 9/14** — 2 3,697; 4 **3,495 vs 4,185**; 5 2,626; 6 4,556; 7 **3,676 vs 5,339**; 8 **2,666 vs 3,227**; 10 2,094; 11 2,694; 13 3,651 (left: 1, 3, 9, 12, 14). The main-game check now also requires the game to AWARD the emerald (count up within 3,000 frames of the stage's end): `route_book.py verify` 14/14.
- **2026-10-11 ~15:30: THE ROUTE BOOK — every special stage in the main game is now a guaranteed emerald (`route_book.py`; Tyler: "My idea was to have 100% fresh start completion so that the main game agent is guaranteed an emerald each time" → Claude proposed the book → "The route book is an excellent idea!").** The game is deterministic from the handover, so a stored winning route per board (one action per frame from the captured state) clears from a giant ring every time; a sampled policy can't be proven 100%. **All 14 boards verified from the main game** (boards 2-7 newly reachable: `$FE16` = Current_special_stage, the next chaos board is $FE16 + 1). The book started as Tyler's 14 recordings; **training now saves every clear faster than the book** (special_stage_env → routes/candidates/, fresh starts and demo starts spliced onto his frames), and `route_book.py harvest --every 30` checks and promotes them (running beside d9, logs/route_book_harvest.log). **First harvest: the book beats Tyler on 5/14 boards — 2 3,718 vs 3,786; 5 2,703 vs 4,148; 6 4,657 vs 6,829; 10 2,094 vs 2,558; 11 2,806 vs 4,013.** The loop's first pass (15:32) added **board 13: 3,651 vs 4,116** — a board the agent has never fresh-cleared (his first 1,655 frames, then the agent's): **6/14**. The coordinator replays the book by default (MG_SS_ROUTES=1); choose_super takes the shortest route first. d9 was stopped and resumed at 4,653,824 steps (~89M frames) to pick up the capture. Section "Special stages inside the main game" → "The route book"; finish line rewritten under "Pre-Committed Criteria".
- **2026-10-11 ~09:15: main-game special-stage handoff BUILT and checked (`ss_pilot.py`, coordinator MG_SS=pilot, `check_ss_pilot.py` ALL PASS)**, incl. Hidden Palace's Super Emeralds — section "Special stages inside the main game". d9 STOPPED 08:44 at 66.8M frames (3,481,800 steps; Tyler gaming; resume with `--resume >>`), 9/14 boards fresh-cleared, gate (70M) not yet reached.
- **2026-10-11 ~08:30, Tyler's finish line for the special stages: "fresh start clears on all 14 boards and beating my times on every board", before moving on to the main game** ("I'd like to have the special stages more or less perfected before we move onto the main game"). His recorded times are now the bar to beat for this phase (still not a ceiling). mg1 stays parked. Meanwhile Claude builds the main game's special-stage handoff (Tyler: "Can we set up the logic for the main game special stage handling while we wait?").
- **2026-10-11 07:04: d9 at 55.4M frames, no alerts all night; 8 of 14 boards have a fresh clear, board 10 and board 6 already faster than his recordings; stage 1 is 37 checkpoints ahead of d7 (transfer, not interference). Gate (70M) ≈ 09:10. Overnight side work (all committed): mg1 launch-ready + mechanism check, smoke_main on v1/v2, full-game env cost, d8c collapse anatomy, record_demo sound — see "Run d9" overnight progress and "Run mg1".**
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
Tyler asked whether audio could work while he plays; it never had. stable-retro does expose it: `em.get_audio()` gives each frame's samples (int16 stereo, 735-736 per frame) and `em.get_audio_rate()` = 44,100. Checked on AIZ1: sound on 240 of 240 frames. `game_audio.py` streams them through SDL (pygame._sdl2 AudioDevice, a callback reading a short buffer: primed at ~2 frames, capped at 0.1 s) to the system's default output. pygame 2.6.1 won't take None for "default", so the device is looked up from `pactl get-default-sink`; it was the HDMI output on 2026-10-10, and the headset is "Built-in Audio Analog Stereo". record_play.py: sound plays at 100% speed only (quiet in slow motion, paused or rewinding), M toggles it, `--no-sound` starts muted, and a missing device just means no sound. At 100% the sound card paces the loop (pygame's Clock.tick(60) ran ~61 fps, overfilling the buffer and dropping a sliver about once a second): silent test 60.15 fps, buffer peak 67 ms, 0 dropped. Output only: recordings and replays unchanged; smoke_main.py passes. **record_demo.py (special stages) has it too since 2026-10-11** (Claude, overnight): same rules (100% speed only, M toggles, `--no-sound`, a missing device = silent). Tested headless by driving its real loop with scripted keys under SDL's dummy video/audio drivers, with the sound module stood in by a call counter (the dummy driver allows one open device and pygame's mixer takes it): one push per played frame with sound on at 100%, none muted, at 50% or paused, the device closed on exit. Not yet heard on real speakers (it was the middle of the night); record_play.py proves the module itself.

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
**`check_play_acts.py <play.npz> [n]` — can the env train on every act?** Drives the real env with HIS per-frame inputs (the emulator call swapped for his recorded input, so the 4-frame action repeat doesn't interfere) from n evenly spaced start points per act; each episode must clear on his frame with return exactly CLEAR_BONUS. Negative control: his inputs one frame late → death / stuck within seconds. **On env v2 as first built: 9 of 162 cut as `stuck`** — AIZ2 x4 (the bombing run: the screen locks ~20 s while the ship flies over and he dodges in place), MHZ1 x1, SOZ1 x4. Where the demo stays in one place, many route points tie for nearest and argmin took the earliest, freezing progress. **Fix (4a2686f, in v2 since v2 had never trained): among route points within ROUTE_TIE = 16 px of the nearest, take the one closest to the last match + the frames since.** After: **ALL ACTS PASS** (162 episodes, 1.15M frames); check_mg_v2 ALL PASS (v1 identical to HEAD). Denser rerun, 20 start points per act: **ALL ACTS PASS** (540 episodes, 3.82M frames, 46 min at nice 15 beside d9). Logs: attic/one_off_tools/check_play_acts_{before,after}_tiefix.log, ..._after_tiefix_n20.log. Run it on every new recording. `smoke_main.py` after the fix: ALL CHECKS PASSED on MG_ENV_VERSION=1 and on 2 (2026-10-11 01:55, all 7 checks incl. every recorded playthrough).

### Special stages inside the main game: `ss_pilot.py` (built 2026-10-11 while d9 trained; Tyler: "Can we set up the logic for the main game special stage handling while we wait?")
`coordinator.py` with **MG_SS=pilot** hands a giant ring to a trained special-stage agent (MG_SS_RUN, default d9; MG_SS_MODEL for a specific .zip); default MG_SS=truncate is the old behaviour (the episode ends, neutral). The agent plays through the SAME SpecialStageEnv it trained on, attached to the main game's emulator (new: `SpecialStageEnv(base_env=...)`, `reset(options={'state': bytes})`; inert by default — all 14 recordings still replay, smoke_main passes), with its run's settings read from train_ss_multistage.RUNS by `ast` (importing the trainer would start a run). Afterwards: no input while the game shows the results and puts Sonic back at the ring, then the episode goes on (one long step; frame stack rebuilt from the reload).
**Measured on real ring entries** (fixtures cut from Tyler's playthrough, private states repo `handoff/`; finders in attic/one_off_tools/: `scan_giant_rings.py` lists the 7 giant rings the game loads near his route — AIZ2, HCZ1, MHZ1 x2, LRZ1 x2, LRZ2 — `find_giant_ring_entry.py`, `map_super_emerald_slots.py`):
- **AIZ1's ring (7112,1216), no emeralds → board 1.** Game mode $34 and entry flag 1 on the entry frame; the board loads 133 frames later (then identical to curriculum_stage_1.state); Up first taken at entry+157 (Sonic sets off alone at +349). The sphere counter holds the level's leftovers until the board loads (read 5,120 in one run, 0 in another), so readiness is found by trying Up each frame from a saved state (taken = it changes Sonic's speed vs no input) and undoing it.
- **The speed-up timer runs from the board's loading, not the run's start** (feature 3, ss_speed_timer/1800): the captured states read e.g. 1,694 on board 1 vs 1,787 at the earliest start. Started 93 frames early, Tyler's stage-1 inputs died at sphere 66. So the pilot waits until the timer reads what the board's captured state reads (11-110 frames, inside the stage, act clock stopped). Then **Tyler's stage-1 inputs clear 102/102 in exactly his 2,925 frames, and d9's deterministic agent sees identical observations and makes identical choices for all 1,670 decisions** vs the captured state.
- **No stall rule inside the game** (the 600-frame "death" is a training device; `env._replaying` turns it off with the training log and pool snapshots). SS_MAX_STEPS still ends the agent's turn; then the pilot lets go with a tap of left every 90 frames (board 11 run straight with no input never ended in 4,000 frames) until the game ends the stage.
- **Return:** results (gm $48), reload, Sonic back at the ring, act clock restored; ~1,000 frames after a death, of which ~50 are outside the stage.
- **Hidden Palace (Tyler's note: "hyper emerald rings warp you to a separate stage first").** MHZ1's ring at (8768,1088) with all 7 Chaos Emeralds (count 7 + $FFB2-B8): entry flag **2**, zone **$17**, a level load; the shrine runs 83 frames later, act clock stopped, Sonic under the Master Emerald. Pedestals are EMPTY with the array at 1 and hold gray Super Emeralds at 2 (3 = won). Gray emerald = object code $0908DE, subtype = its slot; won = $09089E. **Touching one starts ITS board, free choice:** slot→board 0→11, 1→10, 2→12, 3→9, 4→13, 5→8, 6→14 (boards 8-14 = our captures, named by layout). Afterwards the game returns to the ACT at the ring, not to the shrine. The pilot walks under the chosen gray emerald (below) and jumps, steering to it. **Fixed 2026-10-11 ~12:30:** the first controller only checked that SOME stage started; the two high emeralds (slots 3, 4 = boards 9, 13) sit on pedestals solid from below, so it bounced off and touched boards 8 / 14 instead. Now SHRINE_PLAN takes off ~48 px outward, jumps outward 30 frames, then steers onto the top (found by search: offsets 40-64 px all work from 4 arrival timings). Verified: every board 8-14 loads the right board from 4 arrival timings.
- **Determinism (2026-10-11):** Tyler's recordings replayed from the main game's handover clear IDENTICALLY (same frame count) on ALL 14 boards: 1 (AIZ1's ring), 2-7 (AIZ1's ring with `$FE16` = board - 1 and the earlier boards' Chaos Emeralds held: **$FE16 = Current_special_stage, values 0-6 load boards 1-7, measured**), 8-14 (Hidden Palace). So a stored winning input sequence per board wins every time in the main game. Flag 2 outside zone $17 = a bonus stage: still truncated.
- **THE ROUTE BOOK (`route_book.py`, built 2026-10-11 ~15:00; Tyler: "The route book is an excellent idea!").** `routes/board{NN}.npz` = the fastest verified route per board, in the demo format (one action per frame from the captured state; inputs only, no ROM data); replaced routes go to `routes/history/`. The pilot replays the board's route instead of running the agent (`SSPilot(routes=True)`, coordinator MG_SS_ROUTES=1 default; the stage record says `route`; should a route ever end with the stage still on, the agent plays on and the record says `route_broke`). `choose_super` treats a board with a route as a sure win in its frames (shortest first; with Tyler's routes board 10). **Harvest:** special_stage_env (ROUTE_DIR, default routes/candidates/, SS_ROUTES='' turns it off) records each frame's action index, and on a clear faster than the book's route for its board (each worker reads the book once, then beats its own best) saves the route: a fresh start as played; a demo start = Tyler's frames up to the start point + the episode's (so beating his last stretch is a faster whole route — most first-harvest gains were these); pool restarts can't be routes (unknown prefix). Recording changes nothing the agent sees or is paid. `route_book.py harvest` takes the fastest candidates per board (up to 5) and promotes the first that clears on exactly its last frame from the captured state AND from the main game (`main_fixture`); failures go to candidates/rejected/, slower ones are deleted. `route_book.py` (status), `seed` (from his recordings: all 14 passed both checks), `verify` (every route, both checks). **`check_route_book.py`: ALL CHECKS PASS** — a training-mode fresh clear (board 10, 2,094 frames) and a demo-start clear (2,320 of Tyler's frames + the agent's) are each saved and replay to a clear on exactly their last frame; a clear no faster than the book is not saved. `check_ss_pilot.py` check 6: the coordinator plays AIZ1's ring and Hidden Palace (auto → board 10) by the book.
- **Accounting — DECIDED by Tyler 2026-10-11 ~09:30: "Winning emeralds should give a reward. That reward should offset the time penalty for entering the ring."** MG_SS_TIME=**refund** (default): every frame from the ring until the act clock runs again is charged (a cleared board 1 ≈ 4,200) and paid back in full when an emerald is won. A won stage nets 0, a lost one its whole time, so with MG_EMERALD_BONUS=0 (extra frames per emerald, on top) rings can only break even and the agent will take them only where the SS agent nearly always wins — consistent with perfecting the SS agent first; a bonus makes emeralds worth chasing (told Tyler; his call). Also =real (never paid back) and =game (only the ~50 reload frames outside the stage). Stage frames don't run the episode's timeout clock. Logged per episode as `stages` (board, outcome, spheres, emeralds, frames, cost, paid, shrine slot).
- **Which Super Emerald — Tyler: "Maybe the agent can optimize this through normal gameplay?"** MG_SS_SUPER=**auto** (default; or a list of boards): `ss_pilot.choose_super` ranks boards by Super Emeralds won per frame spent, p / (p f_win + (1-p) f_loss), with p drawn from Beta(1+wins, 1+losses) (Thompson sampling) over the SS run's last 100 fresh starts per board plus every stage the main game has logged, so it keeps trying and settles on what pays. (Ranking by the reward's own expectation was tried first and rejected: under refund a win only breaks even, so it chose board 14 — never won, shortest losses — 79% of the time.) On d9's record: board 10 first (92%), then 11; after 40 simulated game wins on 11 and slow losses on 10 it switches to 11 every time. The main-game agent's own policy doesn't make this choice: shrine visits are far too rare to learn from.
- **`check_ss_pilot.py [model.zip]` — ALL CHECKS PASS (2026-10-11, d9's 08:35 save, ~3 min CPU):** Tyler's inputs from the handover; identical agent view; a main-game AIZ1 episode through the ring (back at (7112,1216) with the clock on, episode continues, refund / real / game cost and payback, bonus, logged); Hidden Palace preferring board 8, board 11, and auto (his inputs clear them, Super Emerald won, back in MHZ1 at the ring); MG_SS unset still truncates. d9's own agent on these: board 1 died at 95/102, board 8 at 63/64 (the 08:35 save; boards 1 and 8 had no fresh clears yet).
- **Doomsday Zone** (Tyler 2026-10-11: the true final level, after DEZ3 with all the emeralds — all 14 as Sonic): no recording (he skipped every giant ring). **Start state MADE (`states/doomsday_start.state`, private states repo; `attic/one_off_tools/make_doomsday_start.py`, ~5 min CPU):** his playthrough replayed to DEZ3's start (frame 412,127), all 14 emeralds granted ($FFB0 = 7, $FFB1 = 7, $FFB2-B8 = 3), his DEZ3 inputs on: they clear DEZ3 on his exact frame (418,034, 7:43.38; he had 26 rings, too few to transform) and the game loads Doomsday ($0C) at 420,418 instead of the ending (his END2 began 420,488). The state is that first loading frame; act DDZ1 starts 77 frames in, Hyper Sonic, 50 rings, clock running. Regenerated twice: raw bytes differ (known: never compare state bytes) but RAM and screen are identical every 100th frame over 1,200 frames of the same inputs. **Recorder:** `coordinator.integration_state` now also takes a `.state` file path in the repo (raw em.get_state() bytes), so `python3 record_play.py --state states/doomsday_start.state` records it, and replays/training load it from the recording's meta like any start (sha1-checked). Headless test: 900 frames recorded, saved, replay check passes, act DDZ1 found; smoke_main passes.
- **Tyler RECORDED Doomsday 2026-10-11 ~10:00 (`demos/play_20261011-095800.npz`, 23,910 frames, no rewinds, from states/doomsday_start.state; his first attempt failed, the second beat it; d9 ran at nice 10 meanwhile at his OK).** Doomsday NEVER STOPS THE ACT CLOCK: final hit at frame 21,220 (score +1,000, clock 5:52.36), white-out from 21,421, the game moves to the ending zone ($0D) at 21,463 (clock 5:56.41) with the timer still running. So the recorder saw no clear. **New rule (ActSplitter and the env's own clear check): an act in zone $0C clears on the first frame of zone $0D**; no other recording changes (full playthrough and AIZ1 recording replay-check identical; smoke_main ALL PASS incl. all 4 recordings). The file was committed as recorded (e8989ea), then its act list rewritten by `attic/one_off_tools/resplit_play_acts.py` (inputs untouched): **DDZ1 77-21,463, 5:56.41**, then END2 (the ending, unfinished, as in the full playthrough). audit_play: nothing flagged. **check_play_acts n=20: ALL ACTS PASS** (his inputs clear on his frame with return = CLEAR_BONUS from all 20 start points; log attic/one_off_tools/check_play_acts_doomsday_n20.log). Watch item for training it: Boss_flag is on for the WHOLE act (one fight, demo t 0-21,386), so route progress never pays inside it and `stuck` is paused throughout: only time, deaths (rings run out) and the clear shape it; the backward curriculum starts near the end. If that proves too thin, the boss's hits (+1,000 score each) are a candidate signal.

### Deliberately open
- Special stages: by default still neutral truncations; MG_SS=pilot hands them to the SS agent (section above). Emeralds as a reward (MG_EMERALD_BONUS) and game vs real time (MG_SS_TIME) are Tyler's call: they trade against act time.
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

### Special-stage finish line, rewritten for the route book (2026-10-11 ~15:30)
**Tyler (2026-10-11 ~13:00):** "My idea was to have 100% fresh start completion so that the main game agent is guaranteed an emerald each time." No sampled policy can be shown to clear 100% (100/100 still allows ~3% failures), but a replayed route can: the stage is deterministic from the handover. Claude proposed the route book; **Tyler: "The route book is an excellent idea!"** With it, 100% is met by construction (14/14 routes, each checked from the main game), and the finish line becomes: **the book's route beats Tyler's recording on all 14 boards** (`python3 route_book.py` prints it). Training's job is now to find faster clears, from fresh starts or from his demo start points; reliability no longer matters for the main game. 2026-10-11 15:32: 6/14 (2, 5, 6, 10, 11, 13); 16:05: 9/14 (+ 4, 7, 8). The proposal below (80% reliability, median CI) is superseded; d10's speed bonus still fits the new goal and still needs Tyler's OK. A natural d10 change to discuss: start its backward curriculum from the BOOK's routes instead of only his recordings, so the agent works on beating the current best route's last stretch.

### Run d10 (speed phase along the route book) — DEFINED 2026-10-11 ~15:45; book curriculum AGREED with Tyler ("Yes that sounds great"); LAUNCHED ~16:35 ("Let's go ahead and start it")
**What:** d9's save (`init='d9'`: weights, VecNormalize, curriculum) + `SS_SPEED_BONUS=7` (a clear pays 7 x (18,000 - frames); d8's term) + **`demos='book'`**: a share of episodes (75%) start along the route book's current route for each board instead of Tyler's recording. Goal: the book beats Tyler on all 14 boards (`python3 route_book.py`). Everything else as d9 (all 14 boards, env v7, compass, gamma 0.99 per decision).
**Mechanics (train_ss_multistage.py `book_demos` + `DemoCurriculumCallback`, special_stage_env `set_demo_files`):** each book route is copied to `routes/curriculum/board{NN}_{frames}_{sha8}.npz` (git-ignored) so a demo id names one exact route. A demo new to the curriculum takes over its board's start point **at the same frame** (first start point at or after it): a spliced route shares the older route's frames up to the splice, so there it is the very state the agent had mastered; past the splice it is the agent's own run. Every 5 minutes the callback checks the book; on a change every worker switches at its NEXT reset (no episode is disturbed; unchanged demos are reused, so a one-board change costs one ~5 s replay per worker), and once worker 0 has the new set the callback carries the start point over and pushes it. The curriculum file now records each start point's stage and frame. Route capture (the harvest) works on book-route starts the same way (prefix = the book route's frames to the start point).
**Checks:** `check_book_curriculum.py d10` — ALL PASS (2026-10-11): all 14 book routes build as demos (90 s per worker at startup); start points carried from d9 by frame (identical on the 8 boards whose book route is still his; on the spliced boards 5, 6, 10, 11, 13 the carried start is exactly d9's frame, inside the shared prefix); a book change rebuilds 1 demo and reuses 13, adoption waits for the rebuild; set_demo_files mid-episode leaves the episode identical. `check_route_book.py` ALL PASS after the env change. `smoke_corners.py d10` (now reads the book for demos='book').
**Launch:** `nohup python3 train_ss_multistage.py d10 >> logs/ss_multistage_d10.log 2>&1 &` (must say "Starting d10 from d9"); the harvest loop keeps running beside it. Before launching, stop d9 (`touch STOP_d9`), so d10 inherits its latest save. **Launched 2026-10-11 ~16:35** at Tyler's word, d9 stopped at 5,059,832 steps (~96.5M frames) without its main test read. Resume after a stop: `nohup python3 train_ss_multistage.py d10 --resume >> logs/ss_multistage_d10.log 2>&1 &`.
**Late stretches go unpractised — measured 2026-10-11 ~19:00 (d10's 17:23 save, training rules, 3 tries per start point from the pointer to the end, by tenth of the route):** board 4 (pointer 49%): 50-60% 2/18 … 70-80% 25/33, 90-100% 24/24; board 6 (50%): 19/36 … 30/33; board 7 (66%): 2/12, 7/27, 16/24, 25/27 — healthy, success rising toward the end. **Board 14 (69%): 0/6, 0/30 (70-80%), 4/27, 8/27; board 13 (44%): 1/15, 5/36, 3/33, 0/30 (70-80%), 6/24, 12/21 — dead zones.** Board 14's last stretch (Tyler's 524 frames from sphere 68 to 69) needs a precise sequence: up column 27, jump the single red at (17,27) onto the spring at (16,27), fly north, bounce off the bumper at (9,27), collect (11,27)/(12,27) running backwards, turn round, east along row 13 to (13,30); the straight-line compass points up column 27 but the agent wanders the bumper box or dies at the red (from frames 3,003-3,255: 0/48 clears, from 3,302 on most clear). d9's pointer walked back through all of this (≥25% success at each point), so the skill was had and has faded: the curriculum starts only in [pointer, pointer + 4]. Board 13's 70-80% is the d9 agent's own lucky splice, never practised. **APPLIED 2026-10-11 ~20:26 (Tyler: "Yes"): `SS_DEMO_MIX=0.3` — RUNS['d10'] `demo_mix=0.3` (kept out of `obs`, so the pilot, route_search and smoke_corners never see it; the last two also pin it to 0 when they start at an exact point); d10 stopped at 1,103,800 steps and resumed** — 30% of demo starts drawn from ANY start point between the pointer and the end (special_stage_env, default 0 = exactly the old draw, same random stream, checked; 0.3 → 128/400 starts beyond the band). The curriculum ignores those episodes (outside its band). Side effect wanted: training itself keeps trying to beat the late stretches = more splice candidates.
**Stop rule (proposed):** read `route_book.py` every ~25M frames; stop when the book beats Tyler on all 14, or when 50M frames pass without a book improvement.

### (superseded) Tyler's special-stage finish line + run d10 (speed phase) — PROPOSED by Claude 2026-10-11 ~11:50; NOT agreed, NOT started
**Tyler (2026-10-11):** "I'd like to have the special stages more or less perfected before we move onto the main game" … "yes I want fresh start clears on all 14 boards and beating my times on every board". Where it stands (d9's gate snapshot, eval_finish_line.py, game rules): 9/14 boards cleared, median beats him on 3 (6, 10; 5 by the median).
**Proposed finish line, measured by `eval_finish_line.py` on a frozen snapshot, 100 fresh starts per board, the game's rules (no stall rule, sampled actions):**
- **Beats his time:** the board's median clear frames is below his recording's frames, and the bootstrap 95% interval's upper end is too (so it's not a lucky sample).
- **Clears reliably:** at least 80 of 100 fresh starts clear (a lost stage costs the main game 4,000-8,000 frames under the refund rule, so a coin-flip board isn't "perfected").
- **On all 14 boards** at once, in one snapshot.
**Run d10 (proposed):** d9's save + `SS_SPEED_BONUS=7` (d8's term: a clear pays 7 x (18,000 - frames), so every frame saved is worth 7), everything else as d9 (all 14 boards, his recordings as curriculum, v7). Start it when d9 passes its main test (a fresh clear on all 14 boards) or reaches its 280M-frame budget, whichever is first, so the speed term never competes with a board that has never cleared. Evidence for the bonus: d8 kept 77-91% clears and cut 8 turns per run where its control slid to 52% and then collapsed; the speed gain itself wasn't attributable (1 seed each). Checkpoint the finish line every ~25M frames; stop d10 when the finish line passes, or when 50M frames pass without a board newly meeting it.
**Open choices for Tyler:** the 80% reliability bar (higher = longer); whether "beats" needs the interval (proposed) or just the median.

### Run mg2 (six acts) — criteria AGREED 2026-10-12 ~13:50 (Tyler: "let's do it!"), STARTED ~13:50
**Launch (2026-10-12 ~13:50):** mg1bc stopped at 4,835,616 steps (AIZ1 start point 63; 12 fresh clears, 8 of its last 100 fresh starts; best 1:14.65). Init = **bc4** (mg1bc's latest network + one epoch on datasets/bc_v2, held-out 49.3%; mg1bc as it was: 28.6%), not bc3 (made from mg1bc at 3.42M, before the loop-back and the fresh clears). mg1bc's curriculum file copied (AIZ1 at 63; the other five acts at their ends). Imitation (bc_v2) on; air bar on. No concurrent control (Claude's recommendation). Everything else as drafted below.
**Ring camping found and fixed (2026-10-12 ~14:00; Tyler: "sonic appears to be camping on the spot where the big ring used to be"):** every stage episode so far (mg1bc 58, mg2 24) ended `stuck` at the ring with trip_back = 600 frames forgiven (exactly the stuck window). AIZ1's ring sits in a pocket: after the stage, holding right stops at x 7,157 and left at 7,066; jump+right gets out and back to his route (progress 625 -> 848). The frame-by-frame refund of the way back made standing there free. Now the way back is booked as usual and refunded only when route progress resumes (each leg <= 1,200 frames); camping is charged. check_ss_detour.py (now also a try taken over after the stage with jump+right): control identical; ring-then-camp refunds only the trip there (+368); ring-then-jump-out refunds 368 + 196 = +564. mg2 resumed with it at 470,800 steps, then stopped at 480,208 for Tyler's gaming.
**While Tyler gamed (~14:05-14:30; CPU only, cores 6-7, nice 19):** logs/cpu_checks_1405: check_play_acts.py on his whole game through mg2's env settings (MG_AIR_BAR=1, MG_SS=pilot, refund, bonus 300; v2), 6 start points per act: **ALL ACTS PASS** (162 episodes, 1,146,924 frames); smoke_main ALL CHECKS PASSED; check_ss_pilot ALL CHECKS PASS; check_imitation mg2 (bc_v2) ALL CHECKS PASS. **Ring compass BUILT** (Tyler: "What about big rings that are off my route? I think most big rings must be off my route" -> "Let's try the compass"): of 77 rings, the 7 nearest to his route before MHZ (Chaos) are 78-227 px off it and the 7 nearest from MHZ on (Super) 81-243 px: within about a screen, but off-screen from his path (the screen reaches 112 px above/below Sonic). coordinator MG_RING_COMPASS / RUNS ring_compass: an 11x11 box under the air bar, one white dot toward the nearest giant ring of Sonic's zone and internal act (ROM object layouts, id $85) not used this episode; distance squeezed by a square root (edge from ~620 px, COMPASS_RANGE 768); a stage entered within 160 px of a ring uses it. check_compass.py ALL CHECKS PASS (77 rings; a real try into AIZ1's ring: 454 steps, direction and distance right every time; the ring drops out after its stage). datasets/bc_v3 = his frames with the air bar AND the compass. ImitationCallback now keeps its fade start in main_<run>_imitation.json (a resume continues the fade; new data restarts it): before, every gaming pause restarted it at full strength. check_imitation.py mg2 bc_v3 ALL CHECKS PASS (six acts' frontiers; fade kept across a resume). **mg2 on its next resume: ring_compass=True, imitation data bc_v3.**
**Automatic stepping stones BUILT (~14:30-14:45, CPU):** `mg_stones.py RUN` (stalled = the stepping-stones flag's rule: >= 200 starts at the pointer since it arrived, < 8 of the last 32 matched): stones at the quarter points of the gap to the next start point (>= 16 frames), its middle (8-15), none below 8 (the imitation pull's job); written to main_<run>_stones.json, which train_main.extra_starts() merges with RUNS 'extra_starts' into MG_EXTRA_STARTS; the act's pointer moves onto the last new stone. The trainer writes every act's start-point times to main_<run>_starts.json at start. coordinator now MERGES extras named DEMO:ACT and ACT (it used to take the DEMO:ACT list alone, hiding the other). `attic/one_off_tools/mg_supervise.sh RUN`: mg_watch.sh + on the flag: stop the run, mg_stones.py, resume, exit to report (stopping first: the trainer rewrites the curriculum file on any pointer move). Tested on mg1bc (state backed up and restored): --force AIZ1 at start point 63 put stones at t 3,795/3,810/3,825, pointer 63 -> 66; check_extra_starts.py mg1bc with them + the hand stones ALL CHECKS PASS (his inputs clear from 63-67, 71-75, 78-82).
**AIZ1 from the start, 20 sampled tries each (eval_bc.py, MG_SS=truncate):** mg1bc 5/20 clears (4,306 / 4,367 / 4,926 / 5,068 / 5,753 frames; failures at 10-72%, mostly ~45%); **bc4 0/20** (stuck at 6-17%); bc5 (mg1bc + one copying epoch on bc_v3 with AIZ1 LEFT OUT) 0/20 with the bar + compass and 0/20 without (6-17%). One copying pass anywhere in the shared network pulls AIZ1 back to the plain copy's level (bc1 also stuck ~8-17% from the start). So mg2's start from bc4 cost the AIZ1 skill; mg2's first 480k steps: AIZ1 0/102 fresh clears. **Proposed (Tyler away gaming): restart mg2 from mg1bc itself (no upfront copying pass), the fading imitation pull teaching the five new acts during training; watch AIZ1's fresh-clear rate for the same damage arriving slowly.**
**mg2 RESTARTED ~16:25 from mg1bc** (Tyler: "you can have the box for a bit", after the proposal above): the bc4 attempt's files moved to logs/archive_mg2_from_bc4/; init='mg1bc', AIZ1 at start point 63 (mg1bc's curriculum file), the other five acts at their ends; imitation bc_v3 (bar + compass), air bar and compass on. Supervised by `mg_supervise.sh mg2` (automatic stepping stones); window `watch_main.py mg2 --live --fresh --best 20 --ring`. Stopped at 761,000 steps for Tyler's gaming (~16:45): AIZ1 fresh starts 0/139 clears. **CPU test, AIZ1 from the start, 20 sampled tries each:** mg1bc without markers 5 clears (stuck early 0-19%: 3; later 30-79%: 11; 1 ring); **mg1bc with the air bar + compass 2 clears** (stuck early: 10; later: 5; 1 ring; 2 deaths, 1 drowned); mg2 @761k 0 clears (stuck early 8, later 11, 1 death). So the drop is mostly the new markers (mg1bc never saw them: a distribution shift, early-act failures 3 -> 10), not lost skill; mg2's early AIZ1 trains only through fresh starts (25% of its AIZ1 episodes) and the imitation frames. Keep going; if AIZ1's fresh clears don't return within a few million steps, suspect the six-act pull and give AIZ1 a smaller share of it. Resumed ~16:55 (Tyler: "you can have it back"); **first fresh AIZ1 clear in mg2 at ~1.2M steps** (5,770 frames; 3 in its last 100 fresh starts), so the marker shift is passing. At 1.21M: AIZ2 43 special stages / 43 emeralds; **HCZ1 40 drownings in its last 500 episodes**; new acts 3-9 start points back each.
- **2026-10-12 ~18:15: air bubbles** (Tyler: "Are air bubbles visible in ram? This might be a direction we can go in if drowning becomes an issue. I have a feeling that it will be"). `attic/one_off_tools/mg_bubbles.py` on his whole-game run: his ONLY underwater refill in HCZ1/HCZ2 was the bubble-shield monitor (ROM object $01 subtype 7 at (5,342, 2,123)); he never breathed a bubble there (surfaced; air down to 9 s twice). Rising objects in his HCZ1 footage: code 0x30834, 274 sightings, all first seen on ROM object **$38** (27 in HCZ1's layout) = the bubble makers, so their positions come from the ROM like giant rings (seen at one cluster only, x ~13,350, y ~2,020: confirm $38 in LBZ2/CNZ2 before relying on it). Proposed: an air compass (own box) to the nearest air source underwater: bubble makers ($38), bubble-shield monitors ($01/7), the surface. Asked Tyler: build now, or first see if the air bar alone lowers HCZ1's drownings.
- **2026-10-12 ~19:20: mg2 COLLAPSED when its imitation pull reached 0, rolled back.** The supervisor stopped on a stall alert (AIZ2 at 160 for 3.3M total steps); the per-act clear rates by 1,000 episodes showed every act healthy to ~9,000 episodes, then **AIZ1 41% -> 5% -> 0%** between episodes 10,000 and 11,000, HCZ2 23% -> 1%, MGZ2 25% -> 9%; HCZ1 drownings 80 of 500. Policy entropy (max ln 9 = 2.2) sat at ~0.5-0.6 while the pull's weight was > 0.3 (0.5M-2.3M steps), then rose 0.75 -> 0.93 -> 1.25 -> 1.8 as the weight went to 0 at 3M: PPO's entropy bonus (ent_coef 0.01) flattens the policy once nothing holds it (mg1bc never reached the end of its fade; the runs without imitation sat near 1.9). Fix: ImitationCallback `floor` (mg2: 0.3, a permanent small copying term); mg2 stopped at 4,630,408 and **rolled back to its 2,261,000-step checkpoint** (later checkpoints + the collapsed save in logs/archive_mg2_collapse/; pre-collapse 2.26M/2.76M also in checkpoints_main_mg2_keep/); resumed at weight 0.3. mg_status.py: **ALERT entropy** above 1.2 (it fires on the collapsed run at 1.84), and the step-based stall alert scaled by the number of acts (the stepping-stones flag covers stalls per act). mg_log_mg2.jsonl keeps the collapsed branch's episodes (by time: ~18:40-19:20).
- **2026-10-12 ~19:45: PAUSED for the week (Tyler at his usage limit: "Can I have my box back? ... I'm now further than I've ever been and I can't wait to pick this back again").** mg2 stopped cleanly at **2,583,608 steps** (resumed from the 2,261,000 rollback with the floor; entropy back to 0.47). Bubble compass: Tyler: "I think we should wait until we have more data before committing to a bubble compass" (judge on HCZ1's drowning rate after the rollback; ~81 of 500 before, mixed with the collapsed branch). **TO RESUME:** `source venv/bin/activate && nohup python3 train_main.py mg2 --resume >> logs/main_mg2.log 2>&1 &` (~7 min of recording checks first), then `attic/one_off_tools/mg_supervise.sh mg2 720` (stepping stones by itself; stops on alerts incl. entropy > 1.2) and Tyler's window `python3 watch_main.py mg2 --live --fresh --best 20 --ring` (phone: `--stream 8790`). Watch: entropy stays < 1.2; AIZ1's fresh clears come back; HCZ1 drownings; the 10M gate (each new act >= 20 start points back: they were 3-9 back at 2.58M). Parked: d10 (board 14's fresh clear), mg1bc (the AIZ1 comparison: stopped at 4,835,616, 12 fresh clears, best 1:14.65 vs his 1:25.43).
- **2026-10-14 ~19:15: mg2 RESUMED** (Tyler: "let's get back to work on my sonic rl project") from 2,583,608 steps, supervised (`mg_supervise.sh mg2 720`). mg_status.py fix: the rollback had left the stall clocks in the future (AIZ1 since 3.39M, MGZ1 since 4.54M), silencing those acts' stall alerts; now clamped to the current step.
- **2026-10-14 ~19:30: focus overlay** (Tyler: "this agent has vision, right? is it possible to draw lines in the watcher so i can see what the agent is focusing on?"). `focus.py` + `watch_main.py --focus` (V cycles off / decision / value / both): perturbation saliency (Greydanus 2018): blur one patch of the 84x84 view at a time; orange = how far that moves its 9 action scores, cyan = how far it moves the critic's value; outlined at half the running-average peak; an inset shows the agent's actual 84x84 view (air bar + compass). GPU: 12 ms a pass, every agent step; CPU fallback trails by ~180 ms (~100 px at speed). Measured choices: action-score distance, not KL (KL blank on 75% of steps, peak varied 366x; scores 4x); scale = average of recent peaks, not a decaying max (that hid typical moments). What it shows on mg2 @2.6-3.3M in AIZ1: the HUD (RINGS, TIME, sometimes lives) lights up in most frames on both maps; Tails often; the critic on the Rhinobot + spikes next to Sonic. HUD caveat: a blurred HUD is out-of-distribution (always crisp in training), so it may be a reaction, not use; ring count (a hit kills at 0) and clock (pace vs Tyler) are plausible real reasons. Test that would settle it: change the ring count / timer in RAM (in-distribution) and see if the action or value moves. Not run. Checked on the way: the obs is exactly max(last 2 frames) of the step (no lag in the env).
- **2026-10-14 ~19:50: the HUD test — it reads the ring count and the clock's SECONDS** (Tyler: "yes let's test that"). `attic/one_off_tools/hud_reading_test.py mg2` (log: logs/hud_reading_test_mg2.log), mg2 @3.58M, 12 start points x 6 acts from his recordings: copy A plays 24 agent steps, copy B gets one RAM edit then A's exact buttons; compared on the last 4 steps; kept only if Sonic ends in the same place and the pictures differ only inside the HUD. Scale: the agent's own step-to-step change = KL median 0.095, |dV| median 0.119. Results (KL median / % of decisions that change / mean dV): none 0 / 0% / 0 (sanity); **rings -> 0: 0.035 / 16% / -0.17** (value drops in 5 of 6 acts; shifts from jump+right toward left/noop: more careful, sensible: a hit at 0 rings kills; may be the flashing red RINGS label rather than the digits); rings + 50: 0.004 / 7% / +0.01; **timer + 20 s: 0.042 / 14% / -0.09**; timer + 1:00: 0.0006 / 3% / 0.00; controls: score + 5000: 0.0005 / 3% / -0.05; lives + 2: 0 / 0.3% / 0 (so the lives box in the focus overlay was the blur artifact). Reading: the clock is NOT read as pace (minutes don't matter); the seconds digits are a pattern it uses, most likely a shortcut for 'where am I' (episodes start from his recording, where the clock at each spot is his time). Strongest in the newer acts (timer+20s KL: HCZ1 0.083, HCZ2 0.070, MGZ2 0.066), weakest in AIZ1 (0.004), the act it knows best. Risk: as it gets faster than him, the clock at each spot reads differently from training. Option for a future run (Tyler's call, not done): blank the HUD's TIME/SCORE (and lives) in the observation, keep the ring status as a clean indicator like the air bar.
- **2026-10-14 ~19:55: watcher act switching** (Tyler: "i'd also like the ability to swap the watcher between stages if we're doing multistage training"). watch_main.py builds every act of the run at launch (mg2: ~75 s, ~800 MB; the window shows progress, the build runs on a thread); A / Shift+A = next / previous act, the panel shows e.g. 'HCZ1 3/6 (A)'; the phone stream has a Next act button. watch_tries.py also builds every act, takes the act from flags.json, and drops a batch as soon as any switch changes (tested in a scratch dir: HCZ1 run, switch, 'dropped', MGZ1 run).
- **2026-10-14 ~20:30: the boss hit counter FOUND** (Tyler asked "are boss hits rewarded? ... the agent pays essentially no attention to enemies ... may be an issue for boss fights"; answer: no, a fight pays the demo's fight time as one lump at the kill, -1/frame meanwhile; then "go ahead and find the boss hit counter"). Context that made it matter: 5 of mg2's 6 curricula sat inside their act-ending boss fights (AIZ2 4 s before his kill, HCZ1 13 s, HCZ2 9 s, MGZ2 13 s; MGZ1's miniboss raises no Boss_flag, so its arena cuts episodes as `stuck`: 147 of 400); from those start points clears were rare or slow (AIZ2 21% of episodes, median 11 s slower than him; HCZ2 9%; MGZ2 1 of 16). **Counter = collision_property, +$29 of the boss's object** (S3K's touch code decrements it instead of destroying an enemy that has one). `attic/one_off_tools/find_boss_hits.py` replayed his whole-game recording (all 27 acts) and found it for 24 fights: mg2's AIZ1 6, AIZ2 8, HCZ1 6, HCZ2 8, MGZ1 6 (no flag), MGZ2 8, each last step on exactly the frame his clock stopped; hits >= 32 frames apart (the flash). Exceptions (later zones): CNZ1's miniboss counts at +$45 (4); FBZ1, HPZ1, DEZ2 never set collision_flags (a child object takes the hits); SOZ1/DEZ1 no counter found. **Runtime rule** (`attic/one_off_tools/check_boss_hits.py`, class BossHits, every agent step, no slot known in advance): a hit = an occupied object slot (2..109) whose +$29 dropped by exactly 1, whose count since it appeared peaked at 5..16 (bosses 6/8/9/14; 3-hit badniks 3; Robotnik's unkillable drill car / AIZ1 fire-scene object 255/96), touchable (+$28 nonzero) within the last 64 frames (a hit boss flashes untouchable, so 'touchable at the last step' missed ~1/3 of hits), >= 16 frames since that slot's last hit. Whole game: 157 of 183 counter steps found, 166 firings in fight windows (the extras: SSZ1's second Mecha fight 8, CNZ1 1, likely real hits on other objects); mg2's six acts EXACT, no false firing; outside acts only the ending's final boss. Simpler rules: enemy-type +$28 at the last step 123/183; any -1 step 183/183 but 183 false firings (an AIZ2 timer at +$29). NEXT (Tyler's go-ahead needed for the reward parts): a boss bar in the obs (hits left), each hit paid up front with the same total taken off the fight's lump (fight total unchanged), a hit resets the stuck clock (MGZ1).
- **2026-10-14 ~20:55: boss bar + hit pay + stuck reset LIVE in mg2** (Tyler: "let's add 2 and 3 as well"). coordinator "Boss hits": BossTracker (the checked rule; its per-slot memory rides with every start point, so a mid-fight start knows '5 of 8 left'); MG_BOSS_BAR (RUNS boss_bar): while a boss is tracked, a 2x32 strip left of the air bar, 2 px per hit, white = left, gray = taken (nothing drawn outside fights); MG_BOSS_PAY (RUNS boss_pay, mg2: 120 frames): paid on the hit, taken back when the fight's lump arrives or the episode ends, so no episode's return changes; either flag makes a hit count as progress for `stuck`. Log field boss_hits; mg_status prints them. `check_boss.py` ALL CHECKS PASS: his recording 6/8/6/8/6/8 hits (last within 2 frames of each clear, none elsewhere); mid-fight starts know their boss; 24 episodes of the mg2 agent with pay on vs off on the same actions: identical observations, outcomes and returns, rewards differ only on hit/settle steps, every clear returns exactly his frames - agent frames + 600; MGZ1 start 214 lasted 2,080 frames with tracking vs cut as stuck at 1,568 without. **Pre-existing v2 bug found and fixed on the way:** an off-route step ignored its route gain, but a fight's lump (progress jumping to the demo's fight end as Boss_flag drops) can land on an off-route step, and was dropped (AIZ1 start 48: +113 instead of +486, the 373-frame fire-scene lump). Now paid; check_mg_v2.py still ALL CHECKS PASS. Affects mg2 from this resume (AIZ1's fire scene, MGZ2's two drill-car windows). Imitation: datasets/bc_v4 (bc_v3 + boss bar; same 406,960 samples; bar verified on his AIZ2 frames), its fade carried over from bc_v3 (main_mg2_imitation.json: both 0) so the pull stays at the 0.3 floor. mg2 stopped at 6,241,416 (entropy 0.62) and resumed with all of it; first 100 episodes: 211 boss hits across all six acts. Watch: the boss-fight start points (AIZ2 159, HCZ2 268, MGZ2 305, MGZ1 219) start moving back; MGZ1's stuck share falls.
- **2026-10-15 ~04:56 (Claude, overnight): MGZ2 stall alert, NO stones, supervisor re-armed.** Night so far (boss features from 6.24M): AIZ1 35 -> **0 (its start point reached; fresh clears 61 of the last 100, best 3,994 frames vs his 5,060)**, AIZ2 159 -> 154, HCZ1 129 -> 124, HCZ2 263 -> 234, MGZ1 215 -> 207, MGZ2 297 -> 285 (there since 00:23); entropy 0.53-0.63 throughout; 36.6M steps at 04:56 (~1,150 steps/s with the window closed). The supervisor exited on `ALERT stall: MGZ2 start point 285 unmoved for 18,427,904 steps`. 699 episodes have started AT 285 (3 of the last 32 matched), but **stones would not help**: 285, 286 and 287 look alike over their last 60 starts (matched 8 / 7 / 5, deaths 36 / 29 / 31, median x_max 15,784 for all three, 1-4 boss hits each): this is the MGZ2 fight being hard throughout, not a hurdle between two start points (the pointer got past 286-287 on lucky windows). mg_stones.py --dry --force MGZ2 would put stones at t 17,115/17,130/17,145, inside that fight. Left alone; supervisor re-armed (`mg_supervise.sh mg2 720`, output logs/mg_supervise_mg2_0505.out; the stall is marked reported). **Tool bug found, NOT fixed (for Tyler):** the stepping-stones flag (mg_status.py and mg_stones.py) counts starts at the pointer since the curriculum file's mtime, but the trainer rewrites that file whenever ANY act's pointer moves, so in a six-act run the count keeps resetting (it showed '3 started there' vs 699 real) and the flag can hardly fire; the step-based stall alert (3M x 6 acts) is the only backstop. A fix would count from the first episode at that start point after the act's last episode at an earlier one (pointers only move back), but fixing it would also make it add stones in boss fights like this one, so it needs a 'not inside a fight' check too.
- **2026-10-15 ~06:00-07:00: emeralds — the bonus now pays on getting back (MG_EMERALD_PAY=return, 1,200 frames) LIVE in mg2 from 43,483,624 steps.** Tyler: "What about the goal of collecting all the emeralds?" Read (mg_log_mg2): the stages are solved (1,176 entered, 1,176 won, all board 1: his recordings hold no emeralds at any act start, so training never plays another board), the rings are not: AIZ1 fresh starts pass its ring 99% of the time but took it 8% early -> <1% (5 of the last ~700); AIZ2 fresh up to 69% -> 3%; **AIZ1 fresh starts that took the ring cleared 28% vs 63% (median return -258 vs +366), HCZ1 (its ring 111 px off his route, found as the curriculum reached it) 7 of 30 vs 71%**: after the ring Sonic gets stuck (AIZ1's pocket), and the 5 s bonus paid AT the stage didn't cover the clears lost; the imitation pull (his route took no rings) pulls the same way. Rings used: AIZ1's, AIZ2's at 6,584 and HCZ1's at 12,608, of the 19 in mg2's six acts (the rest are where the curricula haven't reached, or 1,000-1,700 px off his route). Claude proposed: pay the bonus at the clear and raise it to 20 s; **Tyler: "Let's try your recommendation"**. Built as **paid on getting back** instead of at the clear: gamma 0.995 per step is a ~13 s horizon and AIZ1's ring is ~75 s before its clear, so a clear-time bonus is worth <1% at the ring; paying it on the first step of new route progress after the stage (or at a clear) keeps the point (ring then stuck earns nothing extra; ring then clear always beats clear without) and lands within the horizon; an emerald kept after a later death is the game's own rule. coordinator MG_EMERALD_PAY ('stage' default = the old rule; 'return'), stage records 'bonus' / 'bonus_paid_at', train_main RUNS emerald_pay; mg2: emerald_bonus 300 -> 1200, emerald_pay='return'. **check_emerald_pay.py mg2 ALL CHECKS PASS** (logs/check_emerald_pay_mg2.log; check_ss_detour's AIZ1 tries + a fresh clear by mg2's save, replayed under both rules, bonus 1,200): no-stage episodes identical step by step; ring_back same return, the bonus moved from the stage step to 50 steps later as route progress rose 625 -> 730; ring (as played) and ring_stuck (no input after the stage) never got back: -1,200, nothing paid; a clear with a bonus owed pays it. mg_status prints 'got back after X of the Y paid on return'. Watch: fresh-start ring rates (AIZ1, AIZ2), got-back share, AIZ1 fresh clear rate (61 of 100 before), entropy. **Doomsday correction:** Claude said there was no Doomsday recording (from coordinator's stale docstring); Tyler: "I believe we already have a doomsday recording": yes, demos/play_20261011-095800.npz (DDZ1, 5:56.41, 2026-10-11); docstring fixed.
- **2026-10-15 ~07:15-07:45: stepping-stones flag FIXED, first automatic stones in mg2** (Tyler: "let's fix the stepping stone bug"; commit 1162a18). Bug: mg_status.py and mg_stones.py counted starts AT a pointer since the curriculum file's last write, which ANY act's move rewrites, so in a six-act run the flag hardly fired (AIZ2 counted 8 of its real 503 starts at 154, HCZ1 1 of 558). Fix: `mg_stones.since_arrival` (one rule for both tools): the episodes at p after the last one that began below p (a rollback), from a per-act `floor` in the watch state (mg_status's look before a backward move; *now* for a forward move, since stones renumber the start points). Guard (Tyler chose it): stones only when start point p + 1 matches him at the 8-of-32 bar the pointer moves on, else the status prints "stalled but hard throughout" (no alert, no restart). Evidence per start since arrival: AIZ2 154 6% matched vs 155 30% (deaths 215 -> 94); HCZ1 124 1% vs 125 26% (stuck 78% -> 5%); MGZ2 284-288 all 5-16% (hard throughout, as found overnight). Second bug: mg_supervise.sh exports `CUDA_VISIBLE_DEVICES=` for its tools and the stones restart inherited it (device='auto' -> the run would have resumed on the CPU); the restart now unsets it. Stones added 07:37 at 46.45M steps: AIZ2 t 9255/9270/9285 (pointer 154 -> 157), HCZ1 t 7455/7470/7485 (124 -> 127); restart on the GPU verified, window relaunched (detached), supervisor re-armed 12 h. Watch: AIZ2 and HCZ1 walking back through their stones to 154/124 and past.
- **2026-10-15 ~07:55: PAUSED** (Tyler: "let's put a pause on all this for now"). mg2 stopped cleanly at **46,687,640 steps**, about 240k steps after the stones went in; supervisor and window closed. To resume, use the recipe in the 2026-10-12 PAUSED bullet; the window command now adds `--focus`. Stones load from main_mg2_stones.json. Watch on resume: AIZ2 157 -> 154 and HCZ1 127 -> 124 through their stones, the MGZ2 "hard throughout" line at 200 starts, and the emerald return-pay ring rates (too early at the pause).
- **2026-10-15 ~09:55: mg2 RESUMED** (Tyler: "please resume with sonic rl training. i don't need the watcher opened") from 46,687,640 steps, supervised (`mg_supervise.sh mg2 720`), NO window. Tyler had Vanguard Galaxy open and chose "train anyway while I play" (GPU shared; ~870 steps/s vs ~1,150 alone).
- **2026-10-15 ~10:55-11:05: BOSS SHARE live in mg2 from 50,257,048 steps** (Tyler: "is there a way we can train on boss fights only?", then "ok lets go with the boss share inside mg2"; commit 29024fb). **Correction to earlier notes:** the "boss-fight start points (AIZ2 159, HCZ2 268, MGZ2 305, MGZ1 219)" in the 10-14 bullets were where the pointers were parked, NOT where the fights begin. His fights as start points (check_boss_share.py, from the routes' Boss_flag windows): AIZ1 41-54 + 81-89, AIZ2 144-166, HCZ1 131-155, HCZ2 225-279, MGZ1 none (no Boss_flag; its fight's first hit ~218), MGZ2 18-24 + 213-217 + 268-318. So at 49.5M AIZ2 (157), HCZ2 (231) and MGZ2 (284) were still INSIDE their end-of-act fights; HCZ1 (125) and MGZ1 (189) were before theirs. Evidence (last 12,000 episodes): AIZ1, cleared from the start, took **25% of the frames**, AIZ2 7%; the stuck acts already trained mostly in-fight (HCZ2 89% of its frames from starts in the fight, MGZ2 73%, AIZ2 45%); HCZ1 and MGZ1 failures mostly reached the boss (253 of 445, 325 of 550). In-fight outcomes: AIZ2 clear 82% / death 17%; HCZ1 84 / 14; HCZ2 33 / **44** / timeout 23; MGZ2 45 / 32 / timeout 22. Built: coordinator `MG_BOSS_SHARE` (share of resets drawn for the acts whose pointer sits inside one of his fights, start point uniform from the fight's first to the pointer's band, even split between those acts; an act drops out when its pointer leaves the fight) and `MG_ACT_WEIGHTS` (act draw for the other resets); RUNS mg2 boss_share=0.25, act_weights AIZ1 0.3. Expected starts: AIZ1 4.2%, AIZ2/HCZ2/MGZ2 22.5% each, HCZ1/MGZ1 14.2%. Log lines carry `boss_share`; `mg_stones.since_arrival` ignores them (they start below the pointer on purpose, which used to mean a rollback); mg_status keeps 'last 500' to non-boss-share episodes (comparable with before) and prints a 'boss share' line. **check_boss_share.py mg2 ALL CHECKS PASS** (logs/check_boss_share_mg2.log). Watch: AIZ2/HCZ2/MGZ2 pointers leaving their fights, in-fight death and timeout rates, AIZ1 fresh clears (61-72% of 100 before; it now gets ~6% of the frames + the imitation pull).
- **2026-10-15: hit penalty DISCUSSED, not built.** Tyler: "maybe we need to add some type of punishment for taking damage. my fear is that sonic will get stuck in certain areas and refuse to progress because the damage punishment is too severe". Claude's view: a ring-losing hit costs nothing now but sets up a death beyond the ~13 s horizon, so a small hit charge could help; the risk is real because a boss must be approached, and HCZ2/MGZ2 already time out ~22%. Every frame already costs reward, so a charge of X frames buys at most ~X frames of waiting; size it from data (how much a hit raises P(death) x 1,800) and treat rising timeouts as the alarm. First step proposed (awaiting Tyler): log hits taken per episode (Player_1 routine 4 = hurt), logging only.
- **2026-10-15 ~11:10-11:35: hit logging, then the RING SHARE, live in mg2 from 51,367,256 steps** (commits 951ee2f, 41de2e9). (1) Rings: Tyler proposed rewarding rings instead of punishing hits; Claude offered (1) pay per pickup (changes the goal; scatter-and-regrab farming) vs **(2) rings worth something while held** (pickup pays, a hit's ring loss costs the same, capped; can't be farmed, bounded by rings held): **Tyler chose 2.** Claude had said Sonic comes back from a giant ring with 0 rings; **Tyler: "are you sure he comes back from the giant ring at 0? i thought it saved your ring count" -- he was right** (tested: ring counter set to 25 before AIZ1's giant ring, 25 after the stage), so no special-stage exception is needed. Step 1 only, live now: each episode logs the hits Sonic survives, `hurt: [[frame, rings held, rings left], ...]` (Player_1 routine 4; check_hurt_log.py ALL PASS, incl. a shield hit keeping 91 rings); mg_status prints 'hits survived'. The ring value itself is NOT built: size it from these logs first (how much losing the rings raises P(death)), cap it, timeouts as the alarm. (2) **New goal, Tyler: "i think the goal is to encourage sonic to grab all 14 giant rings. speed can come afterwards"** (and, on recording a ring run: "ouch, i was hoping to not have to play through the whole game again"). Found: 14 of the 19 giant rings in mg2's acts are within ~360 px of his route (AIZ1 2, AIZ2 2, HCZ1 1, HCZ2 2, MGZ1 4, MGZ2 3; giant rings per zone game-wide: AIZ 4, HCZ 4, MGZ 11, CNZ 11, FBZ 4, ICZ 5, LBZ 8, MHZ 11, SOZ 11, LRZ 8); what worked against rings: the curriculum's time bar counted the stage's time and the trip (every ring run failed it), the imitation pull (his recordings take none), and they are off his route. **Tyler: "let's try it out"** to: `MG_RING_SHARE` 0.15 (start at the start point at or before where his route passes nearest each ring within 400 px, or the one before; rings evenly; the compass points the way), `MG_BAR_FORGIVE` (the bar leaves out a won stage's cost and its refunded trip legs; log field `forgiven`), mg2 emerald bonus 1,200 -> **3,600** frames (still paid on getting back). Logs: `ring_share` (the target ring key), `giant_rings` (rings used). mg_status: per ring 'tries, took it, got back'. check_ring_share.py + check_boss_share.py ALL PASS (logs/check_ring_share_mg2.log). Expected starts: boss 25%, ring 15%, the rest by act weights. Fallback agreed in principle: a ring not found after a few million steps -> Tyler records only that detour (10-30 s from a save state), never the whole game. Watch: per-ring take rates, entropy (0.57 -> ~0.73 after the boss share), AIZ1 fresh clears, timeouts.
- **2026-10-15 ~11:50-12:10: TIME BALANCE live in mg2 from 52,341,264 steps** (Tyler: "is there a reason why training time isn't more evenly distributed?", then "yes, balance it by time"; commit 1e9bb40). Found: the shares drew EPISODES, and episodes differ ~3x in length (HCZ2/MGZ2 fight episodes ~1,100 steps, AIZ2's ~290), so in the 15 min after the ring share HCZ2 took 30% of the frames, MGZ2 28%, AIZ2 (the act furthest behind) 10%. `MG_BALANCE_TIME` (RUNS balance_time): each option drawn in proportion to weight / its recent episode length (each worker starts from the log's last 8 MB, mean of the last 200 episodes per draw, then an average over its own episodes, alpha 0.05), each source in proportion to share / its mean length: so boss 25%, ring 15%, rest 60% are shares of STEPS, each split evenly in steps (AIZ1 at weight 0.3). Expected (check_balance_time.py from the log's lengths): AIZ1 6%, AIZ2 22%, HCZ1 12%, HCZ2 22%, MGZ1 16%, MGZ2 23%. mg_status prints 'training time, last 3,000 episodes' by act. check_balance_time.py + check_boss_share.py + check_ring_share.py ALL PASS (the last two test the episode draws with the balance off).
- **2026-10-15 ~12:30: PAUSED** (Tyler: "let's put a pause on training for now"). mg2 stopped cleanly at **53,033,872 steps**; supervisor stopped; nothing running. First 10 min of the time balance (777 episodes): AIZ1 5%, AIZ2 22%, HCZ1 11%, HCZ2 21%, MGZ1 17%, MGZ2 24% of the steps (boss 23%, ring 14%), as predicted; entropy 0.75. **TO RESUME:** `source venv/bin/activate && setsid nohup python3 train_main.py mg2 --resume >> logs/main_mg2.log 2>&1 < /dev/null &` (~7 min of recording checks), then `setsid nohup attic/one_off_tools/mg_supervise.sh mg2 720 > logs/mg_supervise_mg2_$(date +%H%M).out 2>&1 < /dev/null &`; window only if Tyler wants it (`python3 watch_main.py mg2 --live --fresh --best 20 --ring --focus`). Boss share, ring share, 3,600 emerald bonus, forgiving bar, time balance and hit logging all load from RUNS. Watch: mg_status's per-ring 'tries, took it, got back' lines, AIZ2/HCZ2/MGZ2 pointers leaving their fights, entropy (< 1.2). Open: size the ring value (option 2) from the 'hurt' logs; fallback detour clips for rings never found.
- **2026-10-15 ~12:48: RESUMED** (Tyler: "please resume sonic rl training") from 53,033,872 with the recipe above + supervisor (720 min); window opened at his request. **~13:20-14:00: the three fights holding pointers are held by SPEED, not by losing** (Tyler: "it's possible that we're facing similar issues across all boss fights"). Careful: the boss share renumbered start points at 11:15 (AIZ2 156 = 596 of his frames before the last hit since then, 506 before; compare by `target`). Training log, at each pointer: AIZ2 156 clears 68-83% but kept pace 24% (11:05-12:10) -> 17% -> 5% (after the resume), clears 1.7x -> 2.3x his; HCZ2 229 (his 3,089) 6-21%, ~1.9x; MGZ2 284 (his 2,100) 6-18%, ~1.7-1.9x, ~10% already before the boss share. **`attic/one_off_tools/boss_timeline.py RUN ACT K ckpt.zip...`** (new): plays the fight from start point K per checkpoint (training env, sampled actions), times every boss hit and hit taken, with his recorded inputs through the same env first (yardstick: his replay clears on his frame, hits at the scan's frames). Checkpoints 50.76M-53.53M copied to checkpoints_main_mg2_keep (rolling dir keeps 5). Raw: logs/boss_timeline_{AIZ2_156,HCZ2_229,MGZ2_284}.json. Findings (24 episodes per checkpoint, so noisy): time is lost in whole boss cycles, not gradually. **AIZ2:** he lands hits 4-8 every ~32 frames from ~45 px away (taking a hit at 580 and finishing 16 frames later); the agent matches hits 1-4 (53.53M lands 3-4 faster than him) then a 350-650-frame wait; 64-84% of these waits follow a hit taken: knocked ~160 px off, it waits on the ground ~350-460 frames while the boss crosses to x 19,080. Kept pace by checkpoint 50.76/51.26/51.87/52.84/53.53M: 6/7/5/4/1 of 24, deaths 7/7/5/4/11. **MGZ2:** waits mostly WITHOUT a hit (20-26%); 50.76M = 7 clears/14 deaths at median 1,608 (faster than his 2,100), later = 13-15 clears/5-9 deaths at median 3,945 (a fixed cycle): it traded speed for survival. **HCZ2:** his own fight has long natural gaps (692, 1,096, 708); the tool's kept pace 8/24, 2/24, 9/24 vs the training log's 11% recently: no clear story. Long waits mostly start at full rings (not low-ring caution). Proposed to Tyler, DECLINED ("it's probably a non issue then. let's let it work"; bar unchanged, training left running): with the goal 'rings first, speed later', let start points inside a boss fight count any clear (or within ~2.5x) toward the pointer bar, so the curriculum leaves the fights and reaches the earlier act sections (AIZ2's spring at ~21% where fresh starts stall); the per-frame reward still pays for speed.
- **2026-10-15 ~14:20: PAUSED** (Tyler: "let's pause on training for now"). mg2 stopped cleanly at **54,723,480 steps**; supervisor and window closed; nothing running, GPU free. Resume with the ~12:30 recipe above (resume, then supervisor; window only if asked: Tyler had it open this session). Settings unchanged since 12:10 (boss bar NOT loosened, Tyler's call).
- **2026-10-15 14:07: RESUMED** (Tyler: "please resume sonic rl training") with the ~12:30 recipe; recording checks passed, first steps 54,727,576 on the GPU; supervisor `mg_supervise.sh mg2 720` armed (logs/mg_supervise_mg2_1407.out). Phone stream running for Tyler at the gym: `watch_main.py mg2 --live --fresh --best 20 --ring --focus --stream 8790` (http://100.64.197.99:8790/, tailnet only, logs/stream_mg2_1413.out). **Phone stream caught up with the window** (Tyler: "Does the mobile trainer have all the same features as the desktop version?" then "yes" to adding the gaps): new Previous act and Focus buttons (Focus shows the mode and cycles off / decision / value / both; V now goes through the same request as the button, done on the main loop's next frame); the page's state can carry a mode name (stream_out.shown). Tested on the live stream: Focus cycled all four and back; Previous act AIZ1 1/6 -> MGZ2 6/6, Next act back to AIZ1 (frames grabbed from the HLS). Still window-only by design: speed 1-4, P pause, R restart (the stream runs at real time, a few seconds behind); M = the phone's own sound. **14:40 act feedback** (Tyler: "I pressed it multiple times and accidentally skipped around": the video runs ~5-8 s behind, so a tap showed nothing): the page now has an act row, Previous / act name / Next; the name (watch_main.act_label: the act on screen plus presses not yet done) updates on the tap itself, in yellow with an arrow and 'Switching: the video catches up in a few seconds' for 8 s; buttons flash on a tap. Tested: rapid Next x3 / Previous x4 / Next read AIZ2, HCZ1, HCZ2, HCZ1, AIZ2, AIZ1, MGZ2, AIZ1 tap by tap; page rendered headless at phone width. Committed with this bullet; stream restarted 14:40 (logs/stream_mg2_1440.out).
- **2026-10-15 ~15:00: HCZ2's opening, measured** (Tyler, watching fresh starts on the phone: "The start of hydro city act 2 seems to be a big problem"). Last 300 fresh HCZ2 starts: progress median 3.4%, p90 4.9%, max 10% (other acts' medians 11-28%, AIZ1 clears); 77% stuck at the first curve (x ~1500-1560, y ~2090), 23% die just past it. His opening: stands at (306, 2092) ~2 s, spin dash (Down, B+Down twice) at frames 361-391, rolls the corridor at 7-9 px/frame holding Right, rolls up the curve at 1540-1660. (1) At x ~405 only a roll gets through (Right held from 306: x max 405); the agent HAS this: every try spin-dashes, half reach the curve as early as him. (2) The curve needs Sonic ROLLING: from his frame 500 (rolling, x 1127) his inputs go up (y 1849), nothing pressed goes up, one jump then Right (lands running, same speed) does NOT (y 2019), brake then Right does not. The agent hops in the corridor (1-138 jump presses between its dash and the curve; 7 of 8 tries reach the curve mid-jump), and a landing ends the roll. 3 of 8 got up. (3) Just past the curve a wall slides left and crushes Sonic in a narrow gap (frames: gap open 80 frames before, closed at death; deaths have rings and no hit); he waits for it twice, frames ~720-790 at (1830, 1900) and ~900-930 at (1944, 1772). Not an env bug (his inputs replay through). Why training doesn't fix it: HCZ2's start point is at 81%, the opening is ~228 start points away, and fresh starts end there by the stuck rule. Tools: attic/one_off_tools/hcz2_opening_roll.py, hcz2_opening_death.py. Proposed to Tyler: starts before the pointer (offered 13:20 for AIZ2), HCZ2's opening as the case for it.
- **2026-10-15 ~15:30: FRONT SHARE built** (Tyler: "Yes, it's probably a good use of time" to starts before the pointer). Uniform starts below the pointer would have given HCZ2's opening ~10 of its 227 start points, so they aim instead: 15% of the steps (RUNS front_share=0.15, MG_FRONT_SHARE) start in the 10 s of his route up to each act's fresh-start frontier (the median of its last 32 fresh starts' reach, coordinator.fresh_frontier, re-read from the log's tail every 60 s so practice follows the frontier); start points from 1, below the pointer; an act drops out while its frontier is at/past its pointer (AIZ1). Logged front_share; out of the at-pointer rule (starts below it), mg_stones.since_arrival, mg_status's fresh/last counts; time balance treats it as a fourth source (boss 25 / ring 15 / front 15 / main 45, so at-pointer starts fall from ~45% to ~34% of steps). Ranges at build (real log): AIZ2 9..18 (t 540-1080), HCZ1 28..37, HCZ2 1..9 (t 60-540: the corridor before his spin dash and the roll up to the curve), MGZ1 55..64, MGZ2 41..50. mg_status prints each act's frontier and the front share's 'got 1 s past it' count. check_front_share.py (new) ALL PASS before the refresh went from every 50 resets to every 60 s (50 resets made check_balance_time's 1.2M draws re-read the log ~25k times); all four share checks rerun after (nice 19, cores 6-7): check_front_share, check_balance_time (front 15.0% of the steps, boss 25.1, ring 15.0, main 45.0; by act now AIZ1 5, AIZ2 22, HCZ1 13, HCZ2 22, MGZ1 16, MGZ2 23%), check_ring_share, check_boss_share ALL PASS. Training PAUSED at **57,297,688 steps** (Tyler asked to prioritise the checks, then went to play Witcher 3: no GPU work while he plays); phone stream stopped. **TO RESUME** (after he's done gaming): the ~12:30 recipe (train_main.py mg2 --resume, then mg_supervise.sh mg2 720); the front share loads from RUNS. Claude's expectations (not agreed criteria): HCZ2's frontier past the curve (> ~700 frames, 4.2%) within ~3M steps of the resume; the other frontiers move forward; pointers move slower (fewer at-pointer starts); entropy stays < 1.2.
- **2026-10-15 ~16:00: other problem areas, from the log** (Tyler: "are there any other problem areas that we should look at while we're at it?"; attic/one_off_tools/mg_problem_areas.py, last 300 fresh / 500 at-pointer per act). Fresh starts stop at a tight spot in every unfinished act (AIZ2 37% of non-clears in one 256-px cell, HCZ1 62% (half its deaths drownings), MGZ1 36%, MGZ2 35%, HCZ2 60%): the front share covers these. Giant rings: 11 of 14 never taken after ~6M steps of ring share (all of MGZ1's and MGZ2's at 0 over 60-100 tries each): not yet looked at. Boss fights at the AIZ2/HCZ2/MGZ2 pointers: as before (Tyler's call: let them run). **MGZ1's pointer (188, t 11,280) probed** (attic/one_off_tools/mgz1_pointer_probe.py, CPU, 8 tries): two spots. (A, the pointer's wall) his route doubles back: he rolls right over a ~40-px gap in the ground at x ~10,950 (skims it at speed), stops at the pillar (x 11,118), spends ~200 frames on the rainbow spinning top (Down / B+Down), then goes back LEFT, drops into the gap and goes left and down through the area below to (8,450, 3,300) before coming back right along y ~3,700. The agent reaches the pillar and holds Right, or drops in and holds Right below. By start point (recent): 188 8 of 8 stuck there; 189 (just before the top) 77% stuck, success 23/204; 190-192 (on the top, his turn back already started) ~45% clear, success ~28%. The automatic stones can't help: the guard needs 189 to work and it doesn't, while 190 does (the guard looks one start point ahead). Too early to act (only 8 starts at 188); if it stalls, a hand stone between 189 and 190 or a two-ahead guard is the tool. (B) the act's end boss (twin-drill machine, rocks from the ceiling), at x 11,800-12,070, y ~3,770: ~27% of starts 190-192 end there `stuck` after 2-3 hits (600 frames without a hit); his fight ~700 frames with one ~210-frame wait. Not looked into further.
- **2026-10-15 ~16:30-17:00: MGZ1's boss and the giant rings, probed** (Tyler, playing Witcher: "Can you dig into spot b while I play? Feel free to do more probing"; all CPU, nice 19, cores 6-7). (1) **MGZ1's end boss is not a fight to the env**: route fights come from Boss_flag ($FFF7AA) and MGZ1's twin-drill boss never sets it (fights [], 0 of 230 start points flagged); the other five acts' end fights are flagged. From start point 218 (his arena entry, t 13,080; attic/one_off_tools/mgz1_boss_probe.py): his 6 hits at 48/88/136/556/684/735 (a 420-frame ceiling phase after hit 3); 16 tries: 15 clear (573-1,748 frames, 4 within the bar), 1 stuck (600 frames without a hit in that phase). With no fight, walking the arena matches his route: most tries had 690-734 of the 735 frames of progress by hit 3-4 (the clock still charges each frame), and `stuck` can fire mid-fight (in a real fight it can't): ~27% of starts 190-192 end that way after ~3 hits. Tyler: other bosses may never set the flag (the Knuckles fight), and bosses also appear in levels as invulnerable props (MGZ1's drill boss hovers over the level at t ~2,957: ring-probe screenshot). So a fix must key on objects that take hits in his run (object code), not 'a boss object present'; and the flag already misfires on a scene: AIZ1's first flagged stretch (t 2,459-3,253) is the fire scene. A full-game scan (every recording; hits without the 5..16 count window; flagged / not) was running at this writing. (2) **Giant rings** (attic/one_off_tools/ring_reach_probe.py, 4 tries from each ring-share start for the 12 rings not regularly taken): in 43 of 44 tries the closest the agent ever got was its first frame; it never turns toward them. The rings sit 196-364 px off his route, mostly below (AIZ1 576 +212, MGZ1 x4 +252..+311, MGZ2 5440 +356) or above (AIZ2 6584 -176, HCZ2 14272 -302, MGZ2 2112 -260), off-screen (224 px tall), under floors / behind walls (HCZ2 9920: behind a brick wall while Sonic rides a water tube). The nearest point on his route is not a way in: these need another path, likely branching earlier. The open 'detour clips' idea (Tyler records route -> ring) fits; his call.
- **2026-10-15 ~17:30: full-game boss scan + RING CLIPS tool** (Tyler: "There may be other bosses that never trigger the flag. I'm thinking of the knuckles fight"; "bosses appear in the level itself ... invulnerable and act as props"; "Detour clips sounds excellent, as long as I can resume near those areas and don't have to replay the whole level"). (1) Scan of every recording (attic/one_off_tools/boss_scan.py: BossTracker's hit rule without the 5..16 count window; hits grouped per act and object code): real bosses NOT flagged: MGZ1 end ($00088572, count 6), SOZ2 end ($0007764E, 8; no flag anywhere in SOZ1/SOZ2), SSZ1's last fight ($0007B288, 8; its first two are flagged), DEZ3 both ($00080542 count 8; $00080D30 count 3, 12 hits). Flagged with no countable hits: HPZ1 (the Knuckles fight: flagged, but the hit counter can't see his hits, so no boss bar / hit pay there), FBZ1, DEZ2. Flagged stretches that are scenes/props (no hits): AIZ1 t 2,387-3,184 (fire scene), MGZ2 t 1,046-1,455 and 12,723-13,050, FBZ2 t 10,186-12,916: the env treats these as fights (progress held, stuck off). Count-3 objects ($0003438E in MGZ1/MGZ2) are 3-hit badniks, not bosses. A props follow-up (attic/one_off_tools/boss_presence.py: where each boss object code is present, touchable or not) was running at the commit. (2) **ring_clips.py** (new): `--export` writes, for each of the 14 rings in mg2's acts, his saved state at the ring-share start point and 5/10/15/20/30/45/60 s earlier (gzipped, states/ring_clips/, 12 MB, index.json; coordinator.integration_state now reads .state.gz); `ring_clips.py` lists them (his act time there, the agent's ring-share record; rings 4 and 5 it finds itself); `ring_clips.py N [--back S]` runs record_play.py from that state, then replays the clip, finds the giant ring it went into (Sonic within 96 px on the frame before the stage) and files it as demos/clips/ringNN_<ACT>_<x>_<y>_<stamp>.npz with meta['clip']. Tested without a person: the agent's own ring-4 frames (logged from the env) replayed through PlayRecorder from the export went into ring 4 at frame 161 (27 px) and filed as such; a Right-held clip from ring 6 filed as 'did NOT go into'. The GUI itself wasn't opened (Tyler was gaming). Next: Tyler records clips for the 12 rings; then the env uses them (a clip's start points as a mini-route into the ring for the ring share): not built.
- **2026-10-15 ~16:05-16:50: training resumed with the front share; Tyler recorded ring clips** ("i want to try creating some giant ring recordings. i closed the witcher so you can have the box"). mg2 resumed from 57,297,688 (first steps 57,301,784; trainer at nice 5 while he recorded), supervisor re-armed. Props follow-up (attic/one_off_tools/boss_presence.py): each unflagged boss's object code (MGZ1 end, SOZ2 end, SSZ1 last) appears ONLY in its real fight in his runs (MGZ1's hovering drill is another object), so a fight keyed on those codes is prop-safe; DEZ3's count-3 'boss' is dozens of short-lived parts. Not built. Clips that went in (demos/clips/): rings 1 (6.8 s), 2 (11.8 s), 3 (16.7 s), 6 (from --back 15; 21.6 s), 11 (from ring 9's start, ~2 min of play, entered ring 11: filed under the ring entered), 13 (7.0 s). Misses kept local, not committed: ring 6 (--back 20), ring 8. Skipped or closed: 7 (Tyler: needs "a massive amount of speed"), 8, 9, 10, 12, 14. ring_clips.py changes: later starts (--back -5..-20, files 'aheadNN'); the export never rewrites an existing start file (a rebuild's emulator state differs in BYTES though not in play, and every clip checks its start's sha1: one re-export orphaned all recorded clips until the files were restored from the states repo); a clip is filed under whichever listed ring it went into (meta clip.started_for); the list counts only clips that went in. Maps around rings 6-14 (attic/one_off_tools/ring_map.py: Sonic held on a grid while the game's camera scrolls, screens stitched, his route drawn) in logs/ring_maps/ (not in git); Tyler: "these maps are just a little hard to read". **Tyler: "we can only go up to 7 emeralds until we hit mushroom hill zone"**: a whole-game run needs only the 7 cheapest rings in AIZ..LBZ (past the 7th, a ring gives nothing until MHZ), so the agent needs ring options, not all 14; with rings 4/5 (found unaided) + these 6 clips it has 8. He knows more rings in CNZ. Next: clip starts in the ring share.
- **2026-10-15 ~17:00-17:30: RING CLIPS in the ring share; Carnival Night + Mushroom Hill rings recordable** (Tyler: "yes" to building the clip training; "i think i know where a few more rings are in carnival night"; "maybe we should focus on mushroom hill"). (1) coordinator MG_RING_CLIPS (RUNS ring_clips='demos/clips', train_main.ring_clip_files: the clips whose meta clip.entered_at is set, so a resume picks up new ones): a ring-share reset for a ring with a clip starts at a point along it, drawn evenly over its points (every CLIP_EVERY=30 frames of its last CLIP_SPAN=1800 up to CLIP_BEFORE=15 before he went in); each point's route progress t is what RouteProgress measured replaying the clip there (ring 11's clip is 2 min: t 7,696-9,428, not its start's 2,940); the build raises if a clip's replay doesn't enter a stage on the recorded frame. Time-balance keys unchanged (still ('ring', zone, act, x, y)); the log carries ring_clip [clip, point, of]; mg_result 'clip' makes ActCurriculumCallback skip it (a clip's 'start' is his route start point the clip began from, which can equal a pointer); mg_status prints per ring 'from his clip: tries, took it'. check_ring_clips.py (new) ALL PASS: 6 clips, 182 points, every point replays into its ring on the recorded frame, t never falls, one step of no input pays 0, 4,016 of 60k draws on clips and none of a clip ring's on his route, all 182 points drawn, the curriculum skips a clip result even with start == its pointer (and counts it without the mark). check_ring_share.py now tests with clips off (env._clips = {}); check_front_share's frontier-moves test sat exactly on a start point's frame (float rounding put the frontier a hair below): moved off it. All four share checks PASS. Training restarted with clips at **59,089,096** (nice 5 while Tyler records). (2) ring_clips.py --export --acts: rings 15-22 = CNZ1 x3, CNZ2 x5; 23-29 = MHZ1 x4, MHZ2 x3 (old numbers and files untouched, verified by md5). MHZ clips (for the super emeralds after MHZ, once MHZ is in training): 23 (2.3 s), 24 (2.7 s), 25 (10.3 s) went in; 26 skipped by Tyler.
- **2026-10-15 ~17:30-17:55: more clips; the Super Emerald set** (Tyler: "we can't do carnival night because they aren't hyper rings"). ring_clips.py exports added FBZ (rings 30-32) and LRZ (33-39; LRZ has 8 giant rings in the ROM, the 8th (5184, 2880) is >400 px from his route). Clips that went in: MHZ 29 (from ring 28's start), FBZ 32 (55.4 s), LRZ 36 (22.4 s), 38 (2.1 s; and again from ring 37's start, 57 s), 39 (8.5 s). Closed without saving: 26, 27 (skipped), 30 (one miss kept local), 31, 33, 34 (x2), 35, 37. Emerald accounting: the 7 Chaos Emeralds come from S3 zones (AIZ..LBZ), already covered by 8 options (rings 4, 5 found unaided + clips 1, 2, 3, 6, 11, 13), so no CNZ clips; Super Emeralds (Hyper Sonic) come from S&K-zone rings: clips for MHZ 23, 24, 25, 29, FBZ 32, LRZ 36, 38, 39 = 8 options for 7. The S&K clips wait for those zones to join training (his states hold 0 emeralds, so in training they'd give chaos stages; super stages only in a chained run). mg2 trains on with clip starts for rings 1, 2, 3, 6, 11, 13 from 59,089,096 (trainer at nice 5 since Tyler recorded; a restart puts it back at 0).
- **2026-10-15 ~18:00-18:30: progress read, then FIGHT BAR 2.5x + imitation at the frontiers** (Tyler: "have boss fights improved? what about that fresh start in hydrocity?", then "yes, and maybe we should loosen the boss fights"). Read (/tmp progress_check, boss-share episodes by window 10:00-14:07 / 14:07-16:05 / front share 16:05-17:11 / clips 17:11-): AIZ2 clears 63 -> 70 -> 75 -> 74%, 1.77x -> 1.66x his time (better); HCZ2 noisy 36/22/39/26%, deaths 46-62% (flat); MGZ2 clears ~31-35% while deaths 41 -> 26% turned into timeouts 24 -> 44% (survives, doesn't finish). No pointer moved all day until HCZ2 228 -> 227 ~18:15. HCZ2 fresh starts: median reach 3.4% in every window, past the curve 18-27%, past the sliding wall 0; front-share practice there (260 episodes) got past the curve only 21-33%: still hopping; Claude's ~3M-step expectation not met. AIZ2's fresh median slipped 11.5% -> 6.8% after the front share went on (~200 tries: maybe noise, watch). Built: coordinator MG_FIGHT_SLACK (RUNS fight_slack=1.5): an episode that starts at a start point with his Boss_flag matches him within 2.5x his frames + 60 (the 13:20 proposal, declined then; the condition 'only if stuck millions more steps' was met: ~6M); check_fight_bar.py ALL PASS (AIZ2 155, HCZ2 227, MGZ2 284 just under/over 2.5x; a non-fight start keeps 1.25x; slack 0 restores it). train_main ImitationCallback cfg 'front' (RUNS mg2 imitation front=True): the focused half of each imitation batch also takes his frames in each act's front-share window (FRONT_BACK up to coordinator.fresh_frontier, re-read every 60 s) for acts whose frontier is short of the pointer: 4,201 pointer frames + 2,971 frontier frames (AIZ2, HCZ1, HCZ2, MGZ1, MGZ2); HCZ2's window = his t 0-570 (his corridor frames after the spin dash, t 395-540: 0 jump presses). load_data() split out of _on_training_start (no model needed). check_imitation_front.py ALL PASS; check_imitation.py mg2 datasets/bc_v4 ALL PASS (end to end, temp state: main_mg2_imitation.json untouched). Restarted from **60,383,504** at nice 0. Watch: AIZ2/HCZ2/MGZ2 pointers leaving their fights (AIZ2 clears ~74% at ~1.7x, so 8 of 32 should come fast); HCZ2 fresh starts past the curve; AIZ2 fresh reach recovering; entropy.
- **2026-10-15 ~18:25: PAUSED** (Tyler: "i'd like to pause on this after that"; he then played Witcher 3). mg2 stopped cleanly at **60,510,712 steps** (only ~130k steps ran on the fight bar + imitation front, so nothing to read yet); supervisor stopped; nothing running. Footgun: a STOP file touched while the trainer is still starting up (recording checks, act + clip builds, ~7-10 min) is deleted at start-up, not obeyed: touch it again once total_timesteps lines appear. **TO RESUME** (only when Tyler isn't gaming): `source venv/bin/activate && setsid nohup python3 train_main.py mg2 --resume >> logs/main_mg2.log 2>&1 < /dev/null &`, then after the first total_timesteps line `setsid nohup attic/one_off_tools/mg_supervise.sh mg2 720 > logs/mg_supervise_mg2_$(date +%H%M).out 2>&1 < /dev/null &`. Everything loads from RUNS (front share, ring clips, fight_slack, imitation front). First reads after the resume: the AIZ2/HCZ2/MGZ2 pointers leaving their fights, HCZ2 fresh starts past the curve, AIZ2 fresh reach, entropy.
- **2026-10-15 ~20:35-22:45: resumed; Claude in charge overnight; AIZ1 weight 0.3 -> 1.0** (Tyler: "please continue", "I'd like to put you in charge of the box tonight", "Make the change"). Resumed 20:35 from 60,510,712. Read at 64.0M (~3.5M steps of fight bar + imitation front): AIZ2 pointer 155 -> 143, out of its fight (87%); HCZ2 228 -> 225; MGZ2 284 -> 280; ring clips work (takes from clip starts, all 0 before: ring 1 37%, ring 3 30%, ring 13 61%, ring 2 7%, ring 6 7%, ring 11 5%); AIZ2 fresh 6.8% -> 11.5% (recovered); HCZ2 fresh still 3.4% (front-share practice gets past the curve 22 of the last 100). **AIZ1 forgetting:** main-episode clears 60-74% (10-14 night, 10-15 morning) -> 46% at 14:00 -> 14-20% from 15:00, stuck at his climb (his route t ~2,220-2,460, x ~10,500-11,600) and the burnt part's wall (act 1 x ~3,000-4,600); its main episodes fell from ~1,000/h to 20-60/h once boss share (25%), ring share and the time balance went in (weight 0.3 -> ~4% of the steps). A compass theory (ring-1 clips teaching it to turn back for ring 1, the compass pointing 3,600 px behind) was checked and REFUTED: measured in time order no try turns back more than ~580 px after its furthest point (a first measure that ignored order counted his own route's loop 10,212 -> 9,470 and looked like a 2-3k px turn-back). act_weights AIZ1 0.3 -> 1.0 (its main share ~2.5% -> ~7.5% of the steps); restarted from **64,168,520**. Overnight watch: /tmp/claude-1000/overnight_watch.sh (30-min status to logs/overnight_20261015.log; exits if trainer or supervisor stops). Watch: AIZ1 clears recovering; the fight pointers; HCZ2 fresh.
- **2026-10-15 22:35 -> 10-16 08:00: overnight d10 for board 14; board 14's route PINNED** (Tyler: "Would you be able to finish work on that final special stage and then come back to the main game after it finishes?"; morning: "Probe first, then decide", then "Pin route, keep d10 going"). mg2 stopped at 64,193,984; d10 resumed from 4,172,816 with the route-book harvest loop (cores 6-7, nice 19); attic/one_off_tools/ss_to_mg_handover.py (detached) waits for d10's first fresh clear of board 14 (ss log line: stage 14, no 'demo', success), then STOP_d10, stops the harvest loop, resumes mg2 + supervisor. Overnight (+64M frames to 144M, 7.45M steps, no crash or alert): boards 4 and 7 now clear from the start (7's start point 43% -> 1.1%, fresh clears 2 -> 88), board 13 fresh clears 6 -> 19, book +923 frames (12/14 faster than Tyler). Board 14: 0 fresh clears in 881, best 69/71 (64 since midnight). Cause found: at ~5.08M steps the harvest promoted a 32-frame-faster board-14 route (board14_2889_a4b6a62c) and the curriculum carried the start point by frame (1,351) onto it, 47% into a route the agent never practised (it goes up a cell and turns back down the column there); on the old route (board14_2921_96f1dc25) the start point had been moving 55% -> 52% -> 51% -> 50% ...; after the switch it never moved (2.4M steps). Probe (/tmp board14_probe.py, 40 tries from start points 26-30): 0 cleared, spheres median 2 (best 35 of the 40 needed), actions erratic (jump / up-now spam). Its 'leaves the book after N cells' count was off by one (agent's cell after its first move vs the book's cell at the start) and is not relied on. Board 14 also gets the fewest frames (2.7%): boards are drawn evenly by episode and its episodes start halfway. Fix: train_ss_multistage book_demos(stages, pins) + RUNS d10 book_pin={14: 'board14_2921_96f1dc25'}: board 14 trains on the old route whatever the book holds (the book keeps 2,889 for the main game's pilot). d10 restarted from 7,487,624 at 07:55: 'episodes start 46% of the way in (carried over from board14_2889_a4b6a62c at frame 1351)'; handover re-armed (from byte 22,006,226). At the old route's pace (~9% per ~0.9M steps) the start point needs ~13 h of d10 to reach the board's start; a fresh clear may come sooner (best fresh 69/71).
- **2026-10-16 ~08:15: CORRECTION to the bullet above: the route switch was NOT the cause.** Per-try results in ss_log_d10 on the OLD route (board14_2921) at 46-49%, before the switch: success 3% (46%, n 36), 1% (47%, n 72), 10% (48%, n 51), 0% (49%, n 45), median 0-3 spheres; the curriculum's '[demo] ... matched or beat it from 47%' lines (31-56%) are band/window rates, not per-start success, and were misread as progress. The pre-switch checkpoint (4,972,816, kept in checkpoints_ss_multistage_d10_keep/) cleared 0 of 30 from the pinned start-point band (old route, today's vecnormalize). Since the pin: 0 of 100 at 46-49%. So board 14 has ONE hard spot at ~46-47% on both routes; the pointer had just reached it when the harvest switched routes. The pin is harmless but not a fix. Tyler: "Dig first, d10 keeps running": checking with a board-14-only copy of smoke_corners.py whether that spot can be followed with v7 decisions from those start points at all (a start state past a turn/jump window would make it unwinnable).
- **2026-10-16 ~08:45: board 14's spot is a MAZE maneuver, learnable but not learned** (Tyler, gaming: "if i remember correctly, i think board 14 was a maze"). d10 paused for his gaming at **7,815,432** (handover, harvest loop and watch stopped). attic/one_off_tools/board14_maze_probe.py (RAM position: x at $E422, y at $E424, checked against a one-frame step) + board14_route_moves.py: at the stuck start points Sonic is in a one-cell corridor walled with red (rows 9-14, col 3); the old route (board14_2921) bounces off the bumper at (9,3), takes the blues at (11,3) and (12,3), overshoots the junction (13,3) to (14,3), presses Up back to it, turns Right, runs east 5 cells and JUMPS at (13,8) over the red at (13,9) onto the spring at (13,10) (frames 1,347-1,560). The earlier probe's deaths cluster at (13,9) (no or mistimed jump) and (13,2) (turned west into red). The straight-line compass is right in the corridor (start points 26-28) but points west across the wrap through red at 29-30. attic/one_off_tools/smoke_corners_board14.py (smoke_corners limited to board 14 + one route): ALL CHECKS PASS for both routes (2921: all 68 start points; 2889: all 64) begin on a decision point and the route's v7 choices reproduce its finish: NOT an env limit; a precise multi-step skill board 14 rarely practises (2.7% of d10's frames). Proposed to Tyler (for after his gaming): more of d10's episodes on board 14, and stepping stones through the sequence (junction / mid-run / just before the jump).
- **2026-10-16 ~09:00-09:20 (times in these 10-16 bullets corrected from git commit times; they had been written 2-3 h late): CPU jobs while Tyler games** (he picked all four from a list; then "You can turn those both on and commit and post everything": d10 obs SS_STAGE_WEIGHTS='14:6' and mg2 fight_codes=True, each from its next start). (1) **HCZ1 drowning**: drownings fell from 13.3% of HCZ1 episodes (10-12 evening) -> 8.6 -> 6.8 -> 4.2 -> 3.8 -> 2.5% (10-15 evening), 58% -> 22% of its deaths; left: fresh starts 9.3% (mostly after getting stuck in the early underwater stretch, x ~4,100-5,100) and the miniboss area (x ~13,800). Read: no bubble compass needed now. (2) **Board-14 fix prepared**: special_stage_env SS_STAGE_WEIGHTS ('14:6' = board 14 drawn 6x as often as each other board, fresh and demo starts alike; unset = the old even draw); check_stage_weights.py ALL PASS (board 14 32.2% of demo starts, 29.9% of fresh, expected 31.6%). Not in RUNS d10 yet (add obs SS_STAGE_WEIGHTS='14:6' on Tyler's OK). Stepping stones judged not useful: start points are already every 2 decisions and 26-30 all fail. (3) **Fights by object code prepared**: coordinator MG_FIGHT_CODES (RUNS 'fight_codes'; off): a boss object with >= 3 hits in his run makes a fight of its presence stretch where no flag stretch overlaps (BossTracker.hit_slots, table_codes, slot_code; _finish_seg; _boss_on at run time). check_fight_codes.py ALL PASS: MGZ1 gains (13068, 13815) keyed to $00088572, start points 218-229 boss-flagged, every other act unchanged, the hovering-drill prop (start point 49, t 2,940) not a fight, his inputs through the fight stay in it every step and clear at 100%. (4) **Readable ring maps**: attic/one_off_tools/ring_map2.py (box around ring + his nearest point padded a screen, held Sonic/Tails masked out, route with direction arrows, ring labelled, his clip path in green, edge HUD scraps cropped, screens where the camera didn't reach the held point dropped): logs/ring_maps/v2/ring06..14.png (not in git).
- **2026-10-16 ~09:20-09:45: more CPU jobs** (Tyler: "Are there any other light CPU tasks"; picked all four). (1) **HCZ2 after the imitation 'front' change** (64,193,984 save; attic/one_off_tools/hcz2_opening_roll.py, 20 tries + a 12-try end probe): the corridor HOPPING IS GONE (18 of 20 pressed no jump between dash and curve, all reach the curve rolling at frame ~552 vs his 541; at 57M it was 1-138 presses, 7 of 8 mid-jump), but only 4 of 12 got up the curve (all 4 then died at the sliding wall); 8 stall at its foot (x max ~1,588) though rolling: whatever stops them now happens on the curve itself. Same success rate as before (3/8): the habit changed, the result not yet. (2) **chain_verify.py** (new): replays his full-game recording as the book would play it (book route where the book has one for that act of that recording, else his frames; bridges = his frames), checking each act clears on its route's last frame and each bridge starts the next act on time; reports the first drift. --his: ALL 27 ACTS AND BRIDGES REPLAY (418,035 frames). Test books: his AIZ2 frames as a book file pass and carry on; the same minus 10 frames: 'DRIFT in AIZ2 ... no event at step 9856'. (3) **Ring routes in the main book**: a fresh start that took a giant ring, won an emerald and cleared saves its per-step agent actions (coordinator _book_steps / _book_ring_candidate; the pilot replays stages deterministically from the special-stage route book, so raw frames aren't needed: its handover probes with state save/restore); main_book.py harvest replays the steps through the env from the same start (clear on the last step, same frames, rings, emeralds) and files the fastest per ring set in index 'rings'. check_main_book_rings.py ALL PASS: HCZ1 ring 5 from its ring-share start (start point 118) on the first try: 1 emerald, 6,613 frames (act 2,446), replay identical, kept out of the book (not from the act's start); a copy minus 3 steps rejected. check_main_book.py ALL PASS again (plain act routes unchanged). (4) **Disk** (no deletions): / at 91% (45 GB free), /mnt/games 834 GB free. Candidates: caches ~13 GB (pip 9.2, winetricks 2.2, Brave 1.7), Witcher camtest captures 12 GB, sonic-rl datasets bc_v1-v3 ~9.6 GB (mg2 uses bc_v4), Elite Dangerous 54 GB and Anomaly ~44 GB could move to /mnt/games; keep Witcher 3 and Last Epoch on the SSD.
- **2026-10-16 ~09:45-10:20: three CPU digs** (no training; Tyler gaming). (1) **HCZ2 curve** (64,193,984 save, /tmp probe, 16 tries from the opening): all roll into the curve at 6-6.75 px/frame like him; 3 of 16 get up. The failures climb to x ~1,582 / y ~2,058, slow to ~+2.5 and then press JUMP (+right) near the top (frames ~572-576), leave the slope at ~+0.2 and stall at y ~2,013-2,023, rolling back down (-2.5 to -3); the successes keep holding Right with no jump and leave the top on the slope's own momentum (x 1,587 -> 1,598, reaching y ~1,900). He held Right from the spin dash at frame 391 and was airborne at the top (1,604, 2,010) by frame 580. So the last HCZ2 opening failure is one habit: a jump pressed near the top of the curve. (2) **Board 14 junction** (d10 7,815,432): from start point 30 (at (14,3)) the agent presses Up correctly, then turns LEFT at (13,3), runs west along row 13 across the wrap and dies at (12,27) (4 of 30 end there); the route turns RIGHT (east). From 29 it tries east with left+jump; from 26-28 it spams up/jump and often runs north out of the corridor. The core error is the turn direction after the Up turnaround; the 14:6 board weight (more practice) is the planned answer. (3) **Wall atlas**: attic/one_off_tools/wall_atlas.py OUT_DIR [--hours 8] [--acts ...] (CPU only): for each mg2 act, the top clusters of where failed tries END (episode log x, y; red death, orange stuck, cyan out of time) over his route, with counts, start points and fresh-start share; a mosaic background, or screens from his own play in boss arenas. A try's place on his route = its start point's time + progress x target (progress is relative to the start point and stays 0 in fights; AIZ1's coordinates restart after the fire, so position alone misplaces it). logs/wall_atlas/*.png + index.txt (not in git). Last 8 h of mg2 (10-15 ~14:30-22:30): AIZ1 stuck at the hollow-tree spiral (47%, 15%) and the loop just before it (45%, 11%), boss deaths 13% (AIZ1 is only 386 failures since 14:34: ring-clip starts 57 and 7 dominate; its fresh clear rate fell 71% -> 0-25% at 8-15 fresh tries an hour, the extra failures are stuck, not boss deaths); AIZ2 stuck at 21% (climb by the waterfall, 25%), boss 19%, start points 66-67 stuck at 42% (10%); HCZ1 stuck at 25% (38%, mostly fresh), end fight 19% (half out of time), 79% stuck (6%); HCZ2 the curve + sliding wall 46% (507 stuck = rolled back, 181 crushed), end boss 40%; MGZ1 stuck at 28% where his route doubles back (31%), end boss 11%; MGZ2 end boss 34% (half out of time), stuck at 16% (17%) and at 39% from start points 114-115 only (8%).
- **2026-10-16 ~10:50-11:30: START-UP CACHE** (Tyler: "is there a way we can speed up start up time?", then "Go ahead and build the cache"). Measured first (2 shared cores): mg2's start-up was a one-core replay of his WHOLE 424,851-frame recording in the trainer (~4 min) plus a rebuild of the 1,242 start points in each of the 8 workers (~1.8 min); d10's was 14 replay_checks (~1 min) plus every worker replaying all 14 routes (~1.3 min). Observed: ~6 min (mg2) and ~4 min (d10) from launch to first episode. Built `start_cache.py`: a pickle per build in `start_cache/` (gitignored, excluded from the daily backup, pruned after 7 days unused), keyed by the sha1 of everything the build reads (recording/clip/route bytes, coordinator.py or special_stage_env.py + demos.py source, every MG_ (minus MG_LOG) or SS_ (minus SS_DEMOS) setting, the acts, the whole game integration incl. ROM, stable-retro/numpy/cv2/Python versions); any change = new key = rebuild, never a stale read. Only the trainer's start-up writes: mg2 `coordinator.prebuild()` (replaces the whole-recording check: builds once in the main process, refuses if an act lands on other frames than recorded or never comes up; a cached key = inputs that passed before); d10 replay_check + `special_stage_env.prebuild_demos()` only for demos not cached (per demo, so a promoted route rebuilds alone; the stage layout its replay recorded in `_original_grids` is cached with it). Workers and tools only read, and build as before on a miss (a live route switch in d10 still replays in the workers). check_start_cache.py mg2 / d10: ALL PASS (cached 0.1 s vs fresh 105 s / 78 s; same start points apart from snapshot bytes; same emulation, screen + RAM over 240 seeded frames, from 143 / 94 snapshots; same 3,000 seeded steps incl. 338 d10 episodes). FOOTGUN found on the way: Genesis Plus GX snapshot BYTES differ between any two processes even for identical game state: ~145 KB of real state sits in a fixed 1 MB buffer (0xFD000) whose tail is leftover heap memory, and the YM2612 channels hold host pointers that loading recomputes; compare snapshots by emulation, never by bytes. The mg2 cache is 216 MB (the garbage tail compresses badly), d10's 14 files ~70 MB. Expected start-up now: well under a minute with the cache (imports, model load, worker spawn), ~2 min after a code change (one build instead of nine). The running d10 still has the old code; the handover's mg2 start will be the first real one (key checked: the handover inherits no MG_/SS_ settings).
- **2026-10-16 ~11:40-12:50: four CPU digs while Tyler gamed** (he picked all four; models: mg2 64,193,984, d10 8,184,240; scripts in attic/one_off_tools). (1) **HCZ2 curve fix test** (hcz2_curve_fix.py, 24 paired tries, same seeds with/without jump blocked at x 1500-1640, y >= 1990): up the curve 8/24 as is -> **24/24** with the one jump press blocked; every one then dies at ~4.8%. **hcz2_after_curve.py**: past the curve Sonic lands in a pit (x ~1820-1920, y ~1900); he steps left, crouches and SPIN-DASHES up its steep wall (twice: ~(1831,1900) then ~(1944,1772)); the agent hops/rolls around the pit and the moving wall crushes it (ends ~(1888-1920, 1824-1856)). So HCZ2's opening = two habits: no jump at the curve's top, spin dash out of the pit. (2) **Board 14 turn test** (board14_turn_fix.py, 30 paired tries from the band 26-30, the route's 'right' forced at (13,3) after (14,3)): only 11/30 get back to the junction at all (19 die in the one-cell corridor, rows 10-12 col 3, mostly from 26-28); of those 11 the policy picks left 6, left+jump 3, jump 1, up-now 1; forcing right doesn't clear any (4 of 11 then die on the red at (13,9): no jump at (13,8)); some left picks collect 30-33 spheres westward but none reaches the route's count. So board 14 = three skills (corridor survival, the turn, the (13,8) jump), not one turn; the 14:6 weight stays the plan. (3) **AIZ1 hollow tree** (spot_dig.py AIZ1 34 12, box x 10400-11800 y 600-1200, his t 1900-2700): from start point 34 (t 2,040) 6/12 clear, 2 stuck at the spiral's foot (11,610, 1,116), 4 deaths (2 at the end boss); he spin-dashes at ~10,643 and holds DOWN (rolling) through the loop and all the way up the spiral; the agent mixes down/right/left. (4) **MGZ1 doubling-back** (spot_dig.py MGZ1 45 12, box x 4400-5300 y 900-1550): 11/12 stuck. He spin-dashes at (4843,1469), hops up a ledge staircase with jump+right to (5296,1060), then goes LEFT and is carried back west through the air (~300 frames, x 5296 -> 4684, y ~980-1060), on to x ~3800: the route doubles back. The agent: 2/12 reach (5301,1060) and press right into the wall there; 9 stall on the staircase ledges (x 4800-5200, y 1100-1450).
- **2026-10-16 ~21:40-22:00: FORGETTING CHECK** (Tyler: "Do we have any protections against catastrophic forgetting?", then "yes" to an alarm). `attic/one_off_tools/forget_check.py RUN [--state F] | --backtest [H]`: per act (mg) / board (ss), fresh starts only (fixed target), last 50 vs the 150 before; ALERT when the clear rate (if it was >= 15%) or the median reach (mg progress, ss spheres/total) falls to <= half with z >= 3; --state reports once, re-arms at >= 75% of the old level. **Backtest:** mg2 7 alerts, all real: the 10-12 18:58 collapse (rolled back), AIZ2's 10-15 16:58 slip, and **AIZ1's forgetting at 10-15 17:58 (clears 59% -> 24%), ~4.5 h before Claude found it by hand**; d9 1, d8c 1 (its known collapse). **d10 24 alerts in ~44 h, real but mostly transient:** boards' fresh clear rates swing hard (board 9 0/220 for 6 h on 10-15/16 night, then 63-81%; board 12 74/98 -> 17/57 after 13:00, back to 48% by 21:45; board 4 ~45% -> ~20% since 13:00). Boards 4 and 12 dipped once board 14 got 6x (others' tries fell ~40%), but swings predate that. Harmless to the goal: the special stages in the main game are flown by the route book's banked routes (14/14), not the d10 policy. Wired into Claude's watch v2 (/tmp/claude-1000/overnight_watch2.sh): alerts LOGGED (not exits: nothing changes in training overnight without Tyler) + the oldest rolling checkpoint kept (forget_<time>/); mg2 also keeps its newest rolling checkpoint every 3 h (ladder_20261016/; its 5 rolling span only ~1.5 h, d10's ~10 h). Full pre-overnight mg2 state copied to checkpoints_main_mg2_keep/pre_overnight_20261016_64193984/. State seeded: mg2 AIZ1 alert (52% -> 14%, the pre-pause forgetting) marked known; it re-arms when AIZ1 is back to >= 39% (= the AIZ1-weight-1.0 read).
- **2026-10-17 03:14-03:25 (overnight, Claude in charge): stepping stones fired.** mg2_supervise added stones for MGZ1 start point 188 (6/32 matched, 298 starts) as designed, restarted mg2 (74.7M, ~473 steps/s) and exited; re-armed. **BUG (not fixed, for Tyler): mg_stones.py doesn't skip start point 0** (mg_status's alert rule does): it also gave AIZ1, parked at 0 (5,266 starts, 6/32 matched = beating his time from the very start), stones at t 15/30/45 and moved its pointer 0 -> 3 (t 45). Harmless (45 frames of a 5,060-frame act; fresh starts unchanged); not reverted overnight. Forgetting check: **AIZ2 fresh reach 17% -> 6% (z 4.0)**: bimodal (stuck 4-7% or 11-21%); share past 10%: 1/10 at the pause -> 65-69% 23:00-00:59 -> 52% at 02h -> 6/21 at 03h (slide after a gain); kept main_mg2_72193984 (forget_1017_0322/), ladder has 70,693,984 (~01:16). AIZ1 fresh clears 25-32% (14% at the pause; not yet the >= 39% recovery bar).
- **2026-10-17 07:51 overnight read (mg2 64.93M at 22:16 -> 83.26M; no crashes, entropy 0.73 -> 0.68, ~470-500 steps/s):** **AIZ1 RECOVERED** (weight 1.0 works): fresh clears 14% at the pause -> 38-40% (bar was >= 39%), fresh frontier 47% -> 82%, pointer back at 0 (walked back over the ptr-0 stones itself); **main book AIZ1 5,060 -> 5,028 -> 4,092 frames** (verified by replay). AIZ2 143 -> 139 (out of its fight, boss share clears 376/500), fresh frontier 4.1% -> 7.0%, the 03:22 reach alert faded (z 1.3). **HCZ1 125 -> 112** (81.4% -> 73.4%, the biggest mover). HCZ2 225 -> 224 (stuck, 3/32 at the pointer; fresh 3.5% flat = the curve habit). MGZ1 188 -> 189 after the stones (one stone, 15 frames, from the old wall). MGZ2 280 -> 270, now 'stalled but hard throughout' in its end-boss fight (271 matched 8% of 610) while taking 38% of training time. Fresh frontiers otherwise flat (HCZ1 25.5%, MGZ1 29%, MGZ2 13.4%); still no fresh clear outside AIZ1.
- **2026-10-17 ~08:00-08:40: MGZ2 boss DUG** (Tyler: "fix the stones bug and dig into MGZ2"; stones fix 9ec9915). mg2 at ~83M, pointer 270 (his fight = start points 268-318; 270 = 2,940 of his frames, 8 boss hits). **Held by LOSING now, not speed:** since 21:52, from 270 clear 11% / death 53% / timeout 35%; clears take 1.63x his time (inside the 2.5x bar); clears ~50% from 278-283 (inside the flight). Deaths = two hits: the first takes all ~11 rings, the sky arena has none, the next kills. **The fight (screenshots logs/boss_frames/MGZ2_270/, boss_snapshots.py):** phase 1 on the ground by the pillar, Eggman's drill pod hovers, he lands 4 hits in frames 180-424; phase 2 in the sky from ~640: TAILS CARRIES SONIC, the boss sweeps horizontally at y ~1,808 and dives to the ground; he hovers at sweep height (up 21% of frames, noop 40%) and taps jump only to drop onto it (9 presses in 2,300 frames, all at hits 5-8). **The agent (83M, boss_timeline 24 eps: 7 clear / 8 death / 9 timeout; 64.19M was 0 / 8 / 16):** phase 1 FASTER than him (hit 5 at ~436 vs his 1,072) but 11 of 24 tries take a hit there (jumping into the pod's underside) -> 0 rings; phase 2: jump in 27% of frames, ~38 fresh presses per 1,000 frames (his 3.9), 61% of flight frames at the bottom (y 1,952, on the ground). **Emulator check (mgz2_carry_branch.py, his carried state at 1,760):** noop holds y ~1,820, up climbs to 1,740, down dips and recovers, ONE JUMP = Sonic lets go, lands on the ground (1,952) and Tails re-grabs him ~180 frames later. So its jump habit keeps dropping it off Tails; from the ground it can only hit the boss on its rare dives, which are also what hit it. **Why imitation hasn't fixed it:** bc_v4 has his 2,299 flight frames (action mix like his play: noop 43, left 22, right 13, up 19, jump 3%), but the focused half covers the pointer window only (behind 120 / ahead 600 = the hover phase); flight = 2.9% of the six acts' samples in the uniform half. **Proposed (not built):** when a pointer sits inside one of his fights, the focused imitation window covers the whole fight (MGZ2: his frames 268-318), like the 'front' change that fixed HCZ2's hopping; read via boss_timeline at 270 after ~2-3M steps.
- **2026-10-17 ~08:50-09:15: IMITATION 'fights' built** (Tyler: "yes, go ahead" to the MGZ2 proposal). coordinator demo_info() now also returns each act's fights (route['fights'], what boss_ranges uses); train_main ImitationCallback cfg 'fights' (RUNS mg2 imitation fights=True): while an act's pointer t0 lies in one of his fights, its focused window is the whole fight (union with -120..+600). Fights per act (start points): AIZ1 44-57 + 84-92, AIZ2 144-166, HCZ1 131-155, HCZ2 225-279, MGZ1 221-232 (fight codes), MGZ2 18-24 + 213-217 + **268-318** (t 16,042-19,140). Now only MGZ2 holds a pointer in a fight: +2,378 of his frames; focused half by act before -> after: MGZ2 18% -> 39%, AIZ2/HCZ1/HCZ2/MGZ1 18% -> 14%, AIZ1 8% -> 6%; his flight frames ~1.5% -> ~12% of each imitation batch. **check_imitation_fights.py mg2 ALL CHECKS PASS** (logs/check_imitation_fights_mg2.log; incl. a pointer moved into AIZ1's fight brings it in, one outside any fight = unchanged). **Pre-committed read (before the restart):** boss_timeline.py mg2 MGZ2 270 --eps 24 on a checkpoint 2-3M steps after the restart vs 83,045,392 (7 clear / 8 death / 9 timeout; flight jump presses median ~38 per 1,000 frames; 61% of flight frames at y >= 1,952, via mgz2_flight_actions.py): WORKS if flight presses < 15 per 1,000 AND ground time < 30% AND >= 12/24 clear; partial if the presses drop but clears don't (then the phase-1 hits are next); no effect if presses stay > 30.
- **2026-10-17 ~08:30-08:45: HCZ2 dug** (Tyler: "maybe we should do something about hcz2"). Pointer 224 = 1 s BEFORE his end fight (225-279, t 13,482-16,829). Since 21:52: from 224 clear 12% / death 62% / timeout 26%; clears 1.66x his 3,389 frames, so only 7/23 within the 1.25x bar that applies there (start point has no Boss_flag; all 23 within 2.5x): matched 4%. 225-226 clear 13-15%, 227-228 28-32%. **Three gaps, all from 'pointer just before a fight':** the 1.25x bar; HCZ2 DROPPED OUT of the boss share (boss_ranges needs the pointer inside the fight; its boss-share count frozen at 1,262 since last night); the new imitation 'fights' window doesn't cover it. MGZ2 hits the same at 267. **boss_timeline HCZ2 224 (84.05M, x24): 1 clear / 12 death / 11 timeout; first hit median 4,988 vs his 420.** The boss spawns when Sonic reaches x ~16,543 (his frame 168); 11/24 tries trigger it by frame 200, 13 dawdle left of it (272 frames to never; 2 never). Even with the boss on time: 7/11 die, 1 clears (hits taken under the boss, dy -64..-192: bombs/spout; he took 3 hits himself). Screenshots logs/boss_frames/HCZ2_224/. Proposed: one rule 'the act's fight = the fight its pointer is in OR the next one starting within ~10 s after it' for boss share, fight bar and imitation 'fights'; switch on after the MGZ2 'fights' read (re-adding HCZ2 to the boss share halves MGZ2's share and would muddle that read).
- **2026-10-17 ~08:45-09:00: FIGHT LEAD built, NOT switched on** (Tyler: "yes, build it"). coordinator MG_FIGHT_LEAD (RUNS 'fight_lead', frames; 0 = off) + fight_at(fights, t): his fight that t is inside or that starts within the lead after t; used by boss_ranges (a pointer in a lead-in gets that fight: lo = its first start point, hi = lo + DEMO_BAND), the bar's _slack (fight slack up to the lead before a fight) and the imitation 'fights' window. **check_fight_lead.py mg2 600 ALL CHECKS PASS** (logs/check_fight_lead_mg2.log): lead 0 = the old rules at all 1,248 start points (fight test, boss_ranges with the pointer at each, slack); lead 600: 201 in-fight pointers unchanged, 99 lead-in pointers get their fight, 948 none; lead-ins per fight listed (HCZ2 215-224, MGZ2 258-267, AIZ2 134-143...). With today's pointers at 600: boss share MGZ2 -> AIZ2 144-148 + HCZ2 225-229 + MGZ2 268-274 (AIZ2 at 138 is 6 start points before its fight: temporary, it clears 11/12 there); imitation +AIZ2 and HCZ2's fights (6,579 -> 10,354 focused frames, front off); HCZ2 224's 23 clears: 7 match at 1.25x, all 23 at the fight bar. check_fight_bar, check_boss_share, check_imitation_fights still ALL PASS. **Switch-on = RUNS mg2 fight_lead=600 + restart, right after the MGZ2 'fights' read (~09:55)** so that read isn't muddled.
- **2026-10-17 ~09:00-09:15: HCZ2 OPENING = a 24-s WALL CHASE** (Tyler: "let's work on the opening"). Training log, last 300 fresh tries: 7% end before the curve, 80% ON the curve (stuck), 11% in the basin after it (ALL deaths), 2% just past it (deaths, x ~2,015-2,045); same as on 10-16. Curve probe at 84.05M (hcz2_curve_fix.py 24): as is 2/24 up; jump blocked 9/24 (was 24/24 at 64M): it now avoids the curve other ways too. Screenshots logs/boss_frames/HCZ2_0/: he rolls up the curve (~frame 560), drops into a basin, spin-dashes up its quarter pipe (curled at (1,856,1,904) at 800, (1,918,1,682) at 840), spin-dashes up a second (yellow) pipe at ~(1,944,1,772), on. **attic/one_off_tools/hcz2_pit_branch.py** (his state, scripted 4-frame actions): air stays 30 (not drowning); every scripted branch from his frame 780 is crushed at fixed frames (~1,031 in the basin, ~1,175 on the ledge) whatever the input = a moving wall; REACTIVE=1 (spin dash when grounded+slow in the basin / on the ledge, cooldown) gets out of the basin AND up the yellow pipe (x 2,197 by ~1,100, his timing) then stalls at x 2,197 and is crushed at 1,360 (he's at 2,353 by 1,100): the agent's action set CAN do it. CHASE=1 (stand still from his state at frame F): crushed for every F from 560 to 1,880 (x 1,532 -> 3,016), only stuck from 2,000 on (x 3,061): **the chase = his frames ~560-2,000, a timed run of maneuvers.** Reading: climbing the curve = entering the chase = certain death (-1,800) for a policy that can't run it, while stalling on the curve = 'stuck' truncation (no penalty): learned avoidance (curve stalls 63% -> 80% while basin deaths 18% -> 11%). Proposed: practice the chase from his states (front share also AHEAD of the frontier: [frontier - 600, frontier + 1,500]) + the imitation front window the same; criteria to agree.
- **2026-10-17 ~09:20-09:45: FRONT AHEAD built, NOT switched on** (Tyler: "yes, build it"). coordinator MG_FRONT_AHEAD (RUNS 'front_ahead', frames; 0 = off): front_ranges' hi = last start point at or before frontier + ahead (below the pointer); train_main front_windows the same. **check_front_ahead.py mg2 1500 ALL CHECKS PASS** (logs/check_front_ahead_mg2.log): ahead 0 = old ranges for 1,234 (act, frontier) pairs and the old imitation windows; at 1,500 today: AIZ2 3..12 -> 3..37, HCZ1 29..38 -> 29..63, **HCZ2 1..9 -> 1..34 (24 of 34 start points inside the chase)**, MGZ1 56..65 -> 56..90, MGZ2 37..46 -> 37..71; imitation front frames 2,979 -> 10,479, incl. all 1,441 of his chase frames (right 47%, noop 20%, left 18%, down 7%, jump+down 6%). **Third chase obstacle found:** at x 2,197 a wall of green breakable blocks: he spin-dashes through it (frames 1,048-1,076; logs/boss_frames/HCZ2_0/tyler_sheet_x2197.png); so the chase takes >= 3 spin dashes (basin pipe, yellow pipe, blocks). **attic/one_off_tools/hcz2_chase_eval.py** CKPT --k K --n 24. **Baseline (85.55M):** from k 9 (t 540, before the curve) curve 11 / basin 10 / yellow pipe 3 / beyond 0 / past the chase 0; from k 13 (t 780, in the basin) basin 1 / **yellow pipe 23** (it gets out of the basin from his state: hold-right and noop both die there, so it likely spin-dashes already) / beyond the blocks 0. **Pre-committed read (before switch-on), at ~5M and ~10M steps after it:** WORKS if from k 13 >= 12/24 get past the blocks (x >= 2,197... i.e. stage 'rest of the chase' or further) AND from k 9 >= 4/24 get past the chase (x >= 3,061) AND HCZ2 fresh tries past the curve (x_max >= 1,650) >= 30% of the last 300 (baseline 13%); no harm = AIZ2/HCZ1/MGZ1/MGZ2 fresh frontiers (7/25.5/29/13.4%) not down > 3 points and AIZ1 fresh clears >= 30%. 0/24 past the blocks from k 13 at 10M = front ahead not enough (next: discuss a spin-dash aid). Switch-on together with fight_lead=600 right after the MGZ2 'fights' read. Side effect: the imitation focused half grows (front windows x3.5 + the lead fights), so MGZ2's flight share of it falls from 39% to ~10-15%.
- **2026-10-17 ~10:00: MGZ2 'fights' READ = PARTIAL** (pre-committed rule, bullet '~08:50-09:15'). Checkpoint 86,550,000 (2.5M steps after the restart), boss_timeline MGZ2 270 x24 + mgz2_flight_actions.py: flight fresh jump presses per 1,000 frames median ~37 -> ~20 (bar < 15: not met); flight frames on the ground 61% -> 42% (bar < 30%: not met); jump share of flight frames 27% -> 15%; clears 7/24 -> 3/24 (bar >= 12: not met; 24 tries are noisy); clear frames median 2,337 vs his 2,940. Presses halved, clears didn't rise = 'partial': the phase-1 (ground) hits are next: hit taken there 11/24 -> 10/24, 7 of the 12 deaths had one. Left on; the ground-phase hits wait for Tyler. Raw: logs/boss_timeline_MGZ2_270_fights_86550000.json, logs/mgz2_flight_actions_86550000.out.
- **2026-10-17 ~10:05: criteria for FIGHT LEAD (written before switch-on; Tyler away, approved the build + switch-on plan):** at ~5M and ~10M steps after it, boss_timeline HCZ2 224 x24 vs 84,050,000's 1 clear / 12 death / 11 timeout and boss-by-frame-200 11/24: WORKS if clears >= 6/24 AND the boss triggered by frame 200 in >= 18/24; plus HCZ2's pointer off 224 (<= 223) within 10M steps. Report MGZ2's at-pointer rate and AIZ2's pointer (boss share now split 3 ways).
- **2026-10-17 ~10:15: CORRECTION (Tyler) to the MGZ2 bullets above: there is NO GROUND in the boss's flight phase.** "Sonic falls off camera and tails picks him up. It doesn't matter where Sonic falls. He could be all the way on the other side of the screen from tails and tails will still grab him." So y ~1,952 is the bottom of the screen, where a dropped Sonic sits out of view until Tails flies over and re-grabs him, not a floor he stands on. Read every 'on the ground' in the ~08:00-08:40 and ~10:00 bullets as 'fallen off the bottom, waiting for Tails' (same measure: flight frames at y >= 1,952: 61% at 83M, 42% at 86.55M). Unchanged: a jump press while carried = Sonic lets go; it is never fatal, it costs time (mgz2_carry_branch.py: back in Tails' grip ~180 frames after the press); the agent's late hits come when the boss swoops down to the bottom where Sonic waits, which is also where it gets hit. The 'fights' fix (fewer random presses) targets the same thing.
- **2026-10-17 ~10:25: MGZ2 flight: HOLDING UP MATTERS** (Tyler: "I held up during my playthrough because I thought it helped me gain air. However, I don't think it matters. I think tails will float you to the middle of the screen regardless of what you press, but I'm not quite sure"). attic/one_off_tools/mgz2_up_test.py (his fight from 270, deterministic): his inputs = clear at 2,940, all 8 hits, every flight jump press from y 1,731-1,811; **his 'up' presses -> noop for the whole flight = only the 4 ground-phase hits, death at 3,720; his presses now start from y 1,872-1,952 and miss**; 'up' -> noop from 2,700 = hits 1-6, the last two miss, death. Carry heights (mgz2_carry_branch.py): noop ~1,820 (= the boss's sweep line ~1,808: a drop lands beside/below it), up ~1,740 (cap), down dips then returns. So the flight needs height (up) THEN a drop onto the sweep: the agent's 'up' share of flight frames 14% (83M) -> 11% (86.55M) vs his 21%: a second flight habit besides the jump spam. His flight frames in bc_v4 carry it (up 19%).
- **2026-10-17 12:45 +5M READ (checkpoint 91,957,008; fight_lead + front_ahead since 86,957,008), interim; the final read is +10M.** FRONT AHEAD (HCZ2 chase): from k 13 (basin) yellow pipe 15 / **past the blocks 9** (was 23 / 0; bar >= 12/24: not yet); from k 9 (before the curve) curve **1** / basin 8 / yellow pipe 14 / past the blocks 1 / past the chase 0 (was 11 / 10 / 3 / 0 / 0; bar >= 4/24 past the chase: not met); HCZ2 fresh tries past the curve 12% of the last 300 (was 13%; bar >= 30%: not met; fresh frontier still 3.5%): from his rolling state at k 9 it now climbs the curve (stalls 11 -> 1), from its own fresh starts it doesn't (it arrives slower than him?). HCZ2 front share last 100: past it 87, death 73 (practising the chase, dying in it). FIGHT LEAD (HCZ2 224, boss_timeline x24): **clear 5** / death 13 / timeout 6 (was 1 / 12 / 11; bar >= 6/24: almost); **boss triggered by frame 200 in 21/24** (was 11; bar >= 18: MET); first hits median 612/656 (his 420; was 4,988); pointer still 224 (5 of the last 32 matched under the fight bar). No harm: AIZ1 fresh clears 36% (>= 30 ok); frontiers AIZ2 7 -> 11.5%, HCZ1 25.5 -> 24.6, MGZ1 29 -> 28.5, MGZ2 13.4 -> 12.7 (all within 3 points); pointers AIZ2 138 -> 131, HCZ1 112 -> 107, MGZ1 189 -> 187, MGZ2 270 -> 269, HCZ2 224; entropy 0.61. Watch: forgetting ALERT 12:36 AIZ2 fresh reach 21% -> 7% (z 4.1; the same bimodal swing as overnight; frontier 11.5% at 12:50; kept main_mg2_89457008). Main book AIZ1 4,092 -> 4,034.
- **2026-10-17 15:20 +10M READ = NOT MET on both (final read; checkpoint 96,957,008).** FRONT AHEAD: from k 13 (basin) past the blocks 7/24 (0 -> 9 at +5M -> 7; bar 12); from k 9 curve 1 / basin 7 / yellow pipe 11 / past the blocks 5 / past the chase 0 (bar 4); HCZ2 fresh past the curve 15% (13 -> 12 -> 15; bar 30%), fresh frontier still 3.5%. It learned pieces of the chase from his states (out of the basin, sometimes through the blocks) but never the whole 24 s, and fresh starts still stall on the curve (from his rolling state at k 9 it climbs; from its own starts it doesn't: likely arrives slower than him, he spin-dashes in the corridor ~t 395). The '0/24 past the blocks at 10M' clause doesn't fire (7/24). FIGHT LEAD (HCZ2 224): clear 5/24 (1 -> 5 -> 5; bar 6), boss triggered by frame 200 **8/24** (11 -> 21 -> 8: the walk-in dawdle came back), clear frames median 6,326 (1.87x his); pointer 224 -> **223** (met). No harm: AIZ1 fresh clears 33%; fresh frontiers AIZ2 7 -> 21.3%, HCZ1 25.5 -> 24.6, MGZ1 29 -> 28.5, MGZ2 13.4 -> 15.4; pointers AIZ2 138 -> 130, HCZ1 112 -> 104, MGZ1 189 -> 185, MGZ2 270 -> 267 (now in its lead-in, kept in the boss share by fight_lead as designed); entropy 0.78. Note AIZ2's main episodes at 130 (before its fight) are stuck 309/500 (a new spot at ~79%). Both settings LEFT ON (no further changes without Tyler). Next options for Tyler: HCZ2 fresh-start corridor speed dig (CPU); a spin-dash aid (the chase needs >= 3); MGZ2 ground-phase hits + climbing before drops.
- **2026-10-17 ~15:35-16:10: HCZ2 corridor dug = NOT a speed problem; the curve stall is ONE jump reflex kept alive by avoidance** (Tyler: "Let's start with option 1"; tools attic/one_off_tools/hcz2_corridor.py, hcz2_dash_revs.py, hcz2_curve_trace.py, hcz2_curve_obs.py; checkpoint 96,957,008). (1) His corridor: lands from a shaft at t ~280 (305, 2,092), spin-dashes at t 369-380 (2 jump+down presses), rolls off at +8.5 px/frame, +6.8 at the curve's foot (t 548), straight up. (2) The agent from his state at k 0/3/5/6 (before his dash): 0/48 up the curve, though it rolls 12/12 at +6.5 at the foot (its own dash: crouch + ~3 revs); from k 7/8 (after his dash): 24/24 up at +6.8. (3) Scripted from his standing state at k 6 (hcz2_dash_revs.py): ANY spin dash (1-4 presses; +6.0 to +7.8 at the foot) then hold right goes UP; hold right from a stand stops at x 405 (needs a roll). So speed is not it. (4) hcz2_curve_trace.py: from k 6 every try presses **jump+right at (1,582-1,583, 2,038-2,058)** on the curve and falls back (6/6); from k 7 it presses right there (6/6, the next step lands 5 px higher). hcz2_curve_obs.py: P(any jump) at that step 0.96 (k 6) vs 0.00 (k 7); swapping the HUD rows changes it little (0.83-0.95): a sharp reflex at one spot, hit or missed by the 4-frame step phase. (5) The critic: rolling to the curve -1.6..-1.7; over it (k 7) -1.9 then -2.0..-2.6 into the basin/chase; after the jump and fallback (k 6) ~-1.8 at the foot. The agent rates the chase below stalling: stall = 'stuck' truncation (bootstrapped, no penalty), chase = likely crush (-1,800). Fix levers (for Tyler): make the chase survivable faster (more HCZ2 chase practice and/or a spin-dash aid), or remove the free exit ('stuck' as costly).
- **2026-10-17 ~16:20: FRONT WEIGHTS built + going on** (Tyler: "yes, give HCZ2 more chase practice"). coordinator MG_FRONT_WEIGHTS (RUNS 'front_weights', DEMO:ACT:w; default 1) weighs an act's draws within the front share (both draw paths); RUNS mg2 front_weights={HCZ2: 3}, front_share 0.15 -> 0.20. **check_front_weights.py mg2 ALL CHECKS PASS** (logs/check_front_weights_mg2.log; the env's own _draw_by_time, 200k draws, today's pointers/frontiers/lengths): old settings = front 3.0% per act; new = HCZ2 front 3.0% -> **8.6%** of steps, AIZ2/HCZ1/MGZ1/MGZ2 2.8-2.9% (was 3.0), boss 25.1 and ring 15.1 unchanged, main 44.8 -> 39.9%. **Pre-committed read (same bars as the front-ahead read, from the restart):** at +5M and +10M steps, hcz2_chase_eval: WORKS if from k 13 >= 12/24 past the blocks (at 96.96M: 7) AND from k 9 >= 4/24 past the chase (0) AND HCZ2 fresh past the curve >= 30% (15%); no harm = other acts' fresh frontiers (AIZ2 21.3, HCZ1 24.6, MGZ1 28.5, MGZ2 15.4%) not down > 3 points, AIZ1 fresh clears >= 30%; pointers reported (main practice is 11% smaller). Also re-run hcz2_curve_obs.py: the critic's value over the curve vs stalled at the foot (96.96M: ~-2.0..-2.6 vs ~-1.8).
- **2026-10-17 18:17 BASELINE for front weights (102,780,216, i.e. 15.8M steps of front_ahead, before the weights):** front_ahead kept working after its 10M read: from k 13 past the blocks **10/24** (96.96M: 7), from k 9 past the blocks 7 (5), past the chase 0; **HCZ2 fresh past the curve 28%** of the last 300 (15%; bar 30), fresh through the blocks 10 (1); **P(jump) at the curve-top spot from k 6: 0.21** (0.96); the critic no longer drops over the curve (k 7: before ~-1.6, over it -1.44..-1.9; the k 6 run climbed too). The avoidance is fading as the chase gets survivable. The front-weights read compares against these numbers (logs/read_front_weights_baseline_102780216.out).
- **2026-10-17 20:47 FRONT WEIGHTS +5M READ (checkpoint 107,780,216) vs the 102.78M baseline:** from k 13 past the blocks **20/24** (10; bar 12: MET); from k 9 past the blocks 18/24 (7), past the chase 0 (bar 4: not yet); **HCZ2 fresh past the curve 46%** (28; bar 30: MET), fresh past the blocks 27 of 300 (10), **first fresh try through the whole chase (1)**; HCZ2 fresh frontier 3.5 -> 5.7%. HCZ2 fight from 224: **13/24 clear** (5), boss on time 21/24; **HCZ2 pointer 223 -> 220**. No harm: AIZ1 fresh clears 38%; frontiers AIZ2 21.3 (=), HCZ1 24.5, MGZ1 31.4 (up), MGZ2 16.0 (up); pointers HCZ1 104 -> 95, MGZ1 185 -> 183, MGZ2 267, **AIZ2 stuck at 130 since ~15:00** (0 of its last 32 there matched, main episodes 389/500 'stuck': a new wall at ~79%, predates this change; morning item). Next chase wall: tries past the blocks end at x ~2,556-2,560 (k 13: 9 of 24; most fresh ones) and ~2,666-2,741 (9); he goes back LEFT at ~2,530 and drops (y 2,013 at his frame 1,520) before heading right; front-share starts deeper in reach 3,664-3,701 (one 7,531).
- **2026-10-17 23:18 FRONT WEIGHTS +10M READ (checkpoint 112,780,216) = 2 of 3 bars MET:** from k 13 past the blocks 15/24 (bar 12: MET), past the chase 1; from k 9 past the blocks 19/24, **past the chase 1/24** (bar 4: not met); **HCZ2 fresh past the curve 65%** (bar 30: MET), past the blocks 50 of 300, past the chase 2. HCZ2 fight from 224: 11/24 clear, boss on time 17/24; pointer 220 -> 218. No harm: AIZ1 fresh clears 29% (bar 30, borderline): hourly 28-49% all day, forget_check AIZ1 clears 35 -> 24% (z 1.5), reach 88 -> 79% (z 2.2): not significant; its practice steady (~110-125 episodes/h, unchanged by the 18:20 change); frontiers HCZ1 24.5, MGZ1 31.9, MGZ2 16.2 (up), AIZ2 21.2 (=); pointers HCZ1 95 -> 92, MGZ1 183 -> 180, MGZ2 267, AIZ2 130 (stuck since ~15:00). Verdict: the chase is being learned (curve 15 -> 65% in ~16M steps since front_ahead + weights); the chase's second half (x ~2,560 drop, ~2,700) is the open wall. Left on.
- **2026-10-17 ~23:30-23:55: AIZ2's stall at 130 dug (Claude in charge overnight; CPU only, no change made).** Since ~15:00 (pointer 131 -> 130) its tries from 130-134: stuck 75% / death 15% / clear 9% (1,503; before 15:00 from 131-138: stuck 39 / death 31 / clear 29%); stuck ends at x 18,176-18,687, y ~384-511. Screens (logs/boss_frames/AIZ2_130/: tyler_sheet.png, stuck_sheet.png; attic/one_off_tools/aiz2_stuck_probe.py): from 130 the first ~700 frames are identical in every try (his x exactly: a scripted run through the burning jungle under Eggman's bomber), then he steps left (~18,003, frame 700), crosses the rope bridge (18,555 at 800) into the boss arena (18,800 at 900; boss in ~1,000). The agent (112.78M, 11/12 stuck) jitters left/right/jump between x 18,200 and 18,690 at the bridge's start and never crosses. **Not avoidance:** aiz2_bridge_value.py: critic ~-1.3..-1.45 before the bridge, -1.37 -> -0.76 in the arena (it rates crossing BETTER). A behavior gap at the bridge: 130 is ~830 frames before his fight (t 8,629), just outside fight_lead 600 (AIZ2's lead-in = 134-143), so since 15:00 AIZ2 has no boss share and its imitation window stops at +600 (his bridge crossing ~t 8,600 is at the edge). For Tyler: fight_lead 600 -> 900 would put 130 back in (boss share, 2.5x bar, imitation of his walk + fight); or look at why the bridge walk is lost.
- **2026-10-18 ~00:30: FIGHT LEAD 600 -> 900 (15 s)** (Tyler: "You can move that lead to 15 seconds to help AIZ2"). check_fight_lead.py mg2 900 ALL CHECKS PASS (logs/check_fight_lead_mg2_900.log): lead 0 = old rules; at 900: 201 in-fight pointers unchanged, 144 lead-in pointers, 903 none; AIZ2's lead-in 129-143 (covers its pointer 130); boss share with today's pointers AIZ2 144-148 + HCZ2 225-229 + MGZ2 268-272 (3-way); imitation focus + AIZ2's fight from 130's window. AIZ2 at 130 so far: 483 tries, 52 clears, 12 match at 1.25x, all 52 at the fight bar. **Pre-committed read at +5M steps:** WORKS if AIZ2's pointer leaves 130 (<= 129) AND aiz2_stuck_probe.py from 130 has <= 6/12 stuck (112.78M: 11/12); no harm = HCZ2 fight from 224 >= 8/24 clear (112.78M: 11), HCZ2 fresh past the curve >= 50% (65%), MGZ2 boss-share clears not below ~110 of 500 (~123-126), AIZ1 fresh clears >= 30%.
- **2026-10-18 07:17 +5M READ of fight_lead 900 (checkpoint 126,510,024) + the AIZ1/AIZ2 'collapse' explained.** AIZ2 pointer 130 -> **127** (bar <= 129: MET); aiz2_stuck_probe from 130: 8/12 stuck, 3 clear (11/12 at 112.78M; bar <= 6: not met). HCZ2: fresh past the curve **96%**, past the blocks 109 of 300, **past the whole chase 12**; from k 13 past the chase 6/24, from k 9 3/24; HCZ2 pointer 218 -> 214; HCZ2 fight from 224 5/24 clear (bar >= 8: NOT met; boss share now 3-way). HCZ1 92 -> 87, MGZ1 180 -> 176, MGZ2 267. Entropy 0.41 (lowest yet). **AIZ1 fresh clears 7%, frontier 15%; AIZ2 frontier 4%: NOT forgetting, a route change** (act_fresh_eval.py AIZ1 x12 with MG_SS truncate: 107.78M / 112.78M: 4 / 6 clears, x median 12,344; **118.28M / 126.51M: 12/12 enter the giant ring** at x ~7,100). Training log, AIZ1 fresh: 18-22h 6% played a special stage, 37% clear; since 02:00 **90%** play one (pilot wins it), 230 of 263 'got back' (emerald pay 3,600 paid), then 220 end **stuck at x ~7,170-7,940, y ~1,024-1,279**; AIZ2 fresh 96-98% take its ring all along, then stuck (now mostly x ~6,400). The decline started ~23:00 (AIZ1 reach 0.88 -> 0.46), deepened at 02:00; before the lead change. The forgetting alarm FIRED (00:21 AIZ2 reach 21 -> 4%, 03:29 AIZ1 reach 45 -> 15%; checkpoints kept forget_1018_0021 = 112.78M, forget_1018_0329 = 118.28M) but the watch only logs alerts (designed so for d10's swings), so Claude didn't see it for ~7 h; Claude also read the 23:18 AIZ1 dip as noise (looked at clears, not reach). Reading: the emerald pays 3,600 on getting back, the clear far less, and 'stuck' after the bonus costs nothing: the same free-exit pattern as HCZ2's curve. A rollback would not fix it (same incentive). Kept: 107.78, 108.28, 112.78, 118.28, 120.28, 121.51, 126.01, 126.51M. For Tyler: the 'stuck' cost question, and/or practice of the way back after AIZ1/AIZ2's rings.
- **2026-10-18 ~08:30: SPLIT EMERALD BONUS built + going on** (Tyler: "Let's split the bonus. That seems like a strong idea", after Claude's read that AIZ1 has 18 rings (median), takes 0 hits and reaches x ~8,100 after the stage, then wanders ~1,500 frames until 'stuck': incentive, not skill). coordinator MG_EMERALD_PAY=split (+ MG_EMERALD_SPLIT, MG_EMERALD_HOLD; RUNS emerald_pay/emerald_split/emerald_hold): SPLIT (0.5) of the bonus on getting back as =return did, the rest once route progress since the stage reaches HOLD (1,800 of his frames = 30 s of his route) or at a clear; never if the episode ends first; stage records get 'bonus2_paid_at'. **check_emerald_split.py mg2 ALL CHECKS PASS** (logs/check_emerald_split_mg2.log; the check_emerald_pay tries under =return vs =split HOLD 120 / huge): no-stage tries and never-back tries identical step by step; ring_back pays half at the return step, the rest 11 steps later at progress 123 (HOLD 120; return = =return's) or never (HOLD huge; return -600 = half of the test's 1,200); clear_owed pays both. RUNS mg2: emerald_pay 'split', emerald_split 0.5, emerald_hold 1800 (bonus stays 3,600). **Pre-committed read at +5M steps:** WORKS if AIZ1 fresh tries that played a stage clear >= 25% (since 02:00: 27/263 = 10%) AND AIZ1 fresh clears overall >= 20% (7%) AND AIZ2 fresh frontier >= 10% (4%); report the second-half pay rate (bonus2_paid_at) for AIZ1/AIZ2 stages; rings kept = AIZ1 fresh tries taking the ring >= 50% (90%; if it collapses toward the old 6%, the split killed the ring pull); no harm = HCZ2 fresh past the curve >= 80% (96%), other frontiers not down > 3 points.
- **2026-10-18 10:36 SPLIT BONUS +5M READ (132,121,992) = WORKS (all three bars).** AIZ1 fresh since 07:40 (172): took a stage 90% (rings kept), **clears overall 23%** (7%; bar 20), **clears after a stage 25%** (10%; bar 25, just), first half paid 150, second half 78; AIZ1 frontier 15 -> 79.5%. AIZ2 fresh (134): stage 96%, clears 0, second half paid only 6 of 121, **frontier 4 -> 16.9%** (bar 10). No harm: HCZ2 fresh past the curve 92% (bar 80), past the chase 3 (of 71); frontiers HCZ1 24.5, MGZ1 31.2, MGZ2 15.3, HCZ2 5.7 (none down > 1 point). Pointers HCZ1 87 -> 85, HCZ2 214 -> 212, MGZ1 176 -> 174, AIZ2 127, MGZ2 267. Main book: **AIZ1 ring route 8,767 frames incl. the stage = act time 4,590 vs his 5,060** (the emerald AND faster than him). Watch: entropy 0.41 -> 0.99 (alarm at 1.2). Left on.
(Drafted by Claude 2026-10-12 ~11:50, revised ~12:35.)
**Tyler 2026-10-12:** "Is there any merit in multistage training?", then "Go ahead and draft it".
**Why:** (1) one hard spot stalls a one-act run (mg1: the whole box on AIZ1's climb); with several acts the others keep moving; (2) transfer: d9 (14 boards at once) ran 37 checkpoints ahead of d7 on stage 1 at equal stage-1 frames, no interference; (3) the goal (14 emeralds, Doomsday) needs one agent for the whole game.
**Setup (RUNS 'mg2'):** acts named DEMO:ACT (new): AIZ1 from play_20260930-153946 exactly as mg1 (87 start points incl. the stepping stones 4,335/4,350/4,365, keyed 'play_20260930-153946:AIZ1') + AIZ2, HCZ1, HCZ2, MGZ1, MGZ2 from play_20261010-202923 (164/153/280/230/319 start points; 1,233 in all). `init='mg1'` (new): mg1's weights and VecNormalize at launch, its curriculum file copied (AIZ1's start point carries over; the other acts start at their ends), steps from 0. Otherwise as mg1 now: env v2, lr 2.5e-4, demo_prob 0.75, ss='pilot', ss_detour='refund', emerald_bonus=300, at-pointer curriculum, plus `curr_successes=8` (new): move once 8 of the last <= 32 starts AT the pointer matched him (the same 25% bar where it's marginal; an easy start point moves after 8 instead of 32). Simulated: easy point moves after 8 starts; later-band successes never move it; held at 7, moved on the 8th; a 3% point never moved in 2,000 runs of 300 starts. Budget 100M steps (~2.3 days alone on the box at ~500 steps/s).
**Proposed criteria:**
- **Gate at 10M steps:** each of the five new acts has moved its start point back >= 20 start points (~20 s of his route). One that hasn't: look at that zone (env/observation trouble there) before going on.
- **AIZ1 sharing read (report, not a stop):** AIZ1's at-pointer match rate at its start point in mg2 vs mg1's last 32 starts at the same start point.
- **Stepping-stones flag (not a stop):** an act's start point with >= 200 starts there since it arrived, unmoved (`mg_status.py` ALERT "stepping stones"): add extra start points there (by hand now; automatic later).
- **Main test:** a fresh-start clear in >= 3 of the 6 acts within 100M steps; all six reported.
- **Stop:** env errors, or the gate failing in >= 3 of the five new acts.
**Pace estimate:** ~1/6 of episodes per act; at mg1's ~2,500 episodes per 1M steps (fewer as start points move back and episodes lengthen), 75% demo starts and a fifth of those at the pointer, ~60 at-pointer starts per act per 1M steps: <= 8 moves per 1M at an easy stretch, 2-4 at a marginal one. MGZ2 (319) and HCZ2 (280) could need 40-150M steps: hence 3 of 6 in the main test.
**Open for Tyler:** (a) a concurrent control (mg1 continuing on AIZ1 alone) for a clean interference read: halves both runs' speed; Claude recommends no. (b) Launch timing: Claude recommends after mg1 shows whether AIZ1's stepping stones work (mg2 inherits them).
**Revised 2026-10-12 ~12:45 (after mg1bc broke AIZ1's wall with imitation, and Tyler: "let's add the air bar"):** init='bc3' (mg1bc's network at 3.42M steps + one epoch of imitation on datasets/bc_v2, the frames with the air bar), air_bar=True, AIZ1's six stepping stones as in mg1bc (4,095/4,110/4,125/4,335/4,350/4,365); at launch copy mg1bc's curriculum file so AIZ1 starts where mg1bc is. Criteria unchanged.
**Checked 2026-10-12:** RUNS loads; `check_extra_starts.py mg2` (the six acts build; AIZ1's start points identical to mg1's with the extras; his inputs from 72-76 clear on his frame); the curriculum rule simulated. No mg2 files written; not launched.

### Run mg1 (aiz1 on MG env v2) — PROPOSED by Claude 2026-10-11 overnight; NOT agreed, NOT started
**Tyler 2026-10-11 ~21:55: "if the book is complete tonight then i'd want to go ahead and start on the main game. i'm sure there will be many issues that we need to fix along the way and i want to start knocking that out."** Plan: when the book beats him on all 14 (`route_book.py status`), run `route_book.py verify`, stop d10 (its job done), the route searches and the harvest, then launch mg1 (the main game's first run; its criteria below are still Claude's proposal for Tyler to confirm). Re-checked 2026-10-11 ~22:05 under the current code: `check_play_acts.py demos/play_20260930-153946.npz 20` ALL ACTS PASS (20 start points, his exact frame); the random-action dry run of `configure_env('mg1')`: obs 84x84x4, 9 actions, 40 episodes = 21 clear / 13 stuck / 6 timeout. Watch: `attic/one_off_tools/mg_status.py mg1 --state logs/mg1_watch_state.json` (start point, outcomes, fresh clears, giant-ring exits; ALERT on stall or the wall gate).
**Why:** aiz1 (MG env v1, Tyler's 09-30 AIZ1) walked 98% → 84% in 344K steps, then sat at 83% for 3.7M steps, reached 82% (start point 69) at 4.66M once `stuck` existed, and was paused at 5.0M. **Probe on its final save (2026-10-11, CPU, `attic/one_off_tools/probe_aiz1_wall.py 69 20`): from start point 69, 20 of 20 stochastic episodes end `stuck`** at demo time ~4,425 of 5,060, the agent at x 3,765-3,830, y 900-1,190, under a ledge with wall spikes; its furthest x 3,842-3,893. Tyler's route there (demo t 4,260-4,455): at (3,830, 942) he goes back UP-LEFT to (3,730, 821), then up to (3,753, 757) and right along y ~760. Going up-left earns no route progress, and v1 cuts after 600 frames without it, so the agent never gets the time to find the climb. MG env v2 counts new ground (a 128-px block not visited this episode) against `stuck`; up-left is new ground. (v2 also settles off-route time on return and breaks route ties by the clock.)
**Setup (RUNS 'mg1'):** aiz1 exactly — same recording (pinned), AIZ1 only, lr 2.5e-4, PPO settings, demo_prob 0.75, 50M-step budget, fresh weights — except `env_version=2`. Alone on the box ~510 steps/s (aiz1's rate): 5M steps ≈ 2.7 h.
**Proposed criteria:**
- **Wall gate: start point below 82% (past start point 69) within 5M steps** (aiz1 needed 4.0M steps from 83% to 82% and never passed it).
- **Main test: a fresh-start clear of AIZ1 within 50M steps** (aiz1's agreed test, unchanged).
- **Stall stop:** start point unmoved for 3M steps.
- **Clear time** vs his 5,060 frames: reported, not a bar.
- Caveat: v2 changes three things at once. If mg1 passes the wall and that matters for the full-game run, a follow-up can isolate the new-ground rule.
**Launch-ready (checked 2026-10-11 ~01:40, CPU, no repo files written):** `check_play_acts.py demos/play_20260930-153946.npz 20` under v2 (with the tie-break): his AIZ1 clears on his exact frame with return = CLEAR_BONUS from all 20 start points (`attic/one_off_tools/check_play_acts_mg1_demo_n20.log`). A random-action dry run of the env exactly as `configure_env('mg1')` sets it (v2, AIZ1, demo_prob 0.75; 40 episodes, log to scratch): obs 84x84x4, 9 actions; 25 clear (from the last start point), 9 stuck (fresh starts), 6 timeouts (the miniboss at start point 83, cut at 2 x 80 + 1,200 frames as designed). Launch: `nohup python3 train_main.py mg1 > logs/main_mg1.log 2>&1 &`, then check it says FRESH.
**Mechanism check (2026-10-11 ~01:40, CPU; `probe_aiz1_wall.py 69 20 {1,2}`, rows in `attic/one_off_tools/aiz1_wall/`):** aiz1's own final save (trained on v1) from start point 69, 20 stochastic episodes on each env. **v1: cut after median 950 frames (567-2,044), 18 stuck + 2 deaths. v2: cut after median 1,644 frames (1,064-2,480), 20 stuck.** Both reach the same route point (demo t ~4,425-4,429, under the ledge). So v2 does what it was built for here: ~1.7x more time at the wall while the agent covers new ground. This v1-trained policy doesn't use that time to climb, which it never learned to. Whether a policy trained on v2 learns the climb is exactly mg1's question; this check doesn't answer it.
**Watch item, giant rings:** entering a special stage ends an episode as `stage`, a truncation, so PPO bootstraps the value of a special-stage screen it never trains on; an exit the value net liked could become a way to end bad episodes early. aiz1's log (6,158 episodes): 49 `stage` (0.8%), almost all fresh starts reaching the ring at x ~7,088, 3.1% in its first tenth and 0.2-0.3% by the end, so it was never exploited. Count `stage` per act in mg1 and the full-game runs; if it grows, end the episode as a termination (no bootstrap) instead.
**After mg1:** the full recording (27 acts, `play_20261010-202923`) — e.g. one zone (AIZ1 + AIZ2) first, then all acts; ~1 GB RAM per worker for all 27 acts.
**Measured cost (2026-10-11 ~01:45, v2, one worker, CPU core 7 at nice 19 beside d9; random actions for 60 s):** AIZ1 + AIZ2 = 249 start points, built in 14 s, 135 MB per worker. **All 27 acts = 6,339 start points, built in 336 s, 930 MB per worker** → 8 workers ≈ 7.5 GB RAM (box: 31 GB) and a ~6-minute startup, since each worker builds its own segments. Env speed is the same either way, ~390 steps/s (1,560 frames/s) per core, so the env is not the bottleneck (aiz1 trained at ~510 steps/s across 8 workers). If restarts become frequent, cache the built segments to disk once (saves the 6 minutes per launch); not needed for a single run.

### Run d9 (all 14 boards on env v7) — criteria AGREED with Tyler 2026-10-10 ~21:05 ("All of those rules sound good to me. And let's go with your recommendations as well"); LAUNCHED 2026-10-10 23:21 by Claude (Tyler gave Claude the box overnight)
**Launch notes:** the first launch (23:19) REFUSED: the trainer's pre-launch `demos.replay_check` cut stage 6 at 127/128 spheres (frame 5,961). Cause: replay_check never set `env._replaying`, which since env v6 is what exempts a recording from the 600-frame stall rule (training's own `_build_demos` sets it); stage 6 waits >600 frames for its last sphere, and no earlier launch had replayed a stage with such a gap. Fixed (cde6b65); all 14 then reproduced their clears under d9's settings, and d9 launched fresh at 23:21 (`logs/ss_multistage_d9.log`, ~85 steps/s at first). Overnight watch: `attic/one_off_tools/d9_status.py d9 --state logs/d9_watch_state.json` every 10 min into `logs/d9_watch.log` (per board: frames, start point, fresh clears, best; stall-stop and gate bookkeeping).
**Overnight progress (Claude's watch, `logs/d9_watch.log`, chart `logs/d9_progress.png` via `attic/one_off_tools/plot_demo_run.py d9`):** no ALERT all night (stall, collapse or gate). Decision rate steady ~106/s; counted frames ~7.5M/h early, ~6.9M/h from 02:30 (fewer frames per decision) → gate (70M) ≈ 09:10. | time | M frames | start-point median | boards with a fresh clear |: 00:33 9.2 / 86%; 01:32 16.9 / 74.6% / 1 (first: stage 3 at 15.74M, 7,164 frames vs his 3,944); 02:32 24.2 / 67% / 4; 04:03 34.6 / 59% / 5; 05:33 45.1 / 50.8% / 7; **07:04 55.4 / 38.6% / 8** (boards 2, 3, 5, 6, 8, 9, 10, 11; all 14 off their recording's end, max 74%, min 1.1%). **Fresh clears faster than his recording (median, 07:04):** board 10 2,098 vs 2,558 (51% of its last ~100 fresh starts clear), board 6 5,880 vs 6,829; board 5 4,157 vs 4,148 (tied); board 9 3,031 vs 2,988 (2 clears). Slower so far: 2 (4,678 vs 3,786), 3 (6,394 vs 3,944), 8 (5,886 vs 3,227, 1 clear), 11 (4,844 vs 4,013). **Interference read (stage 1 vs d7 at equal stage-1 frames):** 9M run frames: 12 checkpoints ahead; 17M: 10; 45M: 23 (50% vs 83%); 55M: **37 ahead (24% vs 80%)**: strong positive transfer, the opposite of interference.
**GATE PASSED (2026-10-11 ~11:12, 70.97M frames, 3,701,240 steps; the run was stopped twice that morning for Tyler's gaming/recording and resumed with --resume):** every board off its recording's end (14/14), every board below 90% (max 69.0%, board 14), median start point 27.6% (< 75%). Fresh clears on 10/14 boards (none yet on 4, 7, 13, 14). Median fresh clear (training log, with the stall rule) vs his: 1 4,507/2,925; 2 4,613/3,786; 3 5,901/3,944; 5 **3,368/4,148**; 6 **5,815/6,829**; 8 6,356/3,227; 9 3,008/2,988; 10 **2,094/2,558**; 11 4,951/4,013; 12 6,098/2,702 (1 clear). **Interference read:** stage 1 at 22.9% vs d7's 74% at the same 6.50M stage-1 frames, 34 checkpoints ahead (transfer). d9 continues toward the main test (a fresh clear on all 14 within 280M). **Finish-line evaluation of the gate snapshot** (`eval_finish_line.py d9 -n 40`, the game's rules = no stall rule, sampled actions; logs/finish_line_d9_gate.{log,json}): clears per 40 / median clear vs his: 1 11 / 7,046 vs 2,925; 2 24 / 4,353 vs 3,786; 3 **34** / 7,685 vs 3,944; 4 0; 5 20 / **3,967 vs 4,148 (median only, CI 3,085-5,152)**; 6 13 / **5,647 vs 6,829 (CI 5,328-5,914)**; 7 0; 8 0; 9 19 / 3,008 vs 2,988 (every clear exactly 3,008: one route, 20 frames slower); 10 11 / **2,443 vs 2,558 (CI 2,305-2,551)**; 11 **31** / 4,588 vs 4,013; 12 1 / 7,494 vs 2,702; 13 0; 14 0. **9/14 boards cleared at least once; the median beats him on 3/14 (6 and 10 with the whole CI below his time, 5 on the median).** Without the stall rule many clears are slow (wandering that training would have cut: board 3's median 7,685 vs 5,283 in training), and clear rates are higher than training's (training's last 100 fresh: 1 4%, 3 17%, 9 22%, 11 30%). **Watch item:** board 10's training clear rate fell from 51% of its last ~100 fresh starts at 07:04 to 20% at the gate (eval 28%), its clears still fast (median 2,094).
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
  - **Anatomy (Claude, 2026-10-11 ~01:35, from logs/ss_multistage_d8c.log + ss_log_d8c.jsonl):** the update at 682,552 steps had approx_kl 0.099, the session's largest (its 102 updates: typically 0.01-0.04, two others at 0.070-0.074 that did no harm) and clip fraction 0.146. Episodes 22:00-22:33: 72-74% clears, 26-28% deaths, 13-15% stalls. 22:34-22:50: 0-1% clears, ~100% deaths, **59-68% on its own trail** (a red sphere it made), 25-32% stalls, ~55-60 spheres. 22:50-23:17: still 0%, but now **74-78% stalls at ~90 of 102 spheres**, with explained variance 0.98-1.00 and value loss 0.006-0.03: the value net predicts the new outcome perfectly, a stable bad habit rather than noise, and entropy doubled (0.11 → 0.22). PPO did not repair it in 2,000+ episodes.
  - **Candidate guard for future runs (not d9 mid-run; Tyler's call): `target_kl`.** SB3 / sb3-contrib stop an update's epochs once a minibatch's approx_kl exceeds 1.5 x target_kl. Calibrate per run: fine-tuning d8c ran at 0.01-0.04, so target_kl 0.05 (cut at 0.075) would have touched only the collapse update. A fresh run is noisier: d9 at 0.9M steps has median 0.034, p90 0.06, max 0.075 (clip fraction 0.32-0.37, entropy ~0.95), so 0.05 would cut d9's routine updates; it would need ~0.08 there.
  - **Watch:** `attic/one_off_tools/d9_status.py` raises `ALERT collapse?` when demo-start success over the last 400 episodes falls below a quarter of the 2,000 before (on d8c it fires right after the collapse; on d8 once, at its 8-10M dip, which recovered).

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
| `route_book.py` | The route book: the fastest verified route per special-stage board (`routes/`), guaranteed emeralds in the main game; status / seed / harvest [--every N] / verify. `check_route_book.py` checks training's route capture. |
| `route_search.py` | The agent from every start point of each book route (CPU, nice 19 beside training); faster clears land in routes/candidates/ for the harvest. |
| `route_polish.py` | Frame-exact search over every input from a book frame to the board's end (merged identical states, beam); a faster finish lands in routes/candidates/. Finds the book's own time as its sanity check. |
| `check_book_curriculum.py` | Pre-launch check for `demos='book'` runs (d10): book routes as demos, start points carried over by frame, live book switch. |
| `ss_pilot.py` | Plays a special stage inside a main-game episode (route book first, else the agent); Hidden Palace's Super Emeralds; `check_ss_pilot.py`. |
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
