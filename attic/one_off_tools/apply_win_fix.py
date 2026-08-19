#!/usr/bin/env python3
"""
apply_win_fix.py -- apply the board-clear success fix to special_stage_env.py

Makes three edits so a board clear (blue_spheres_remaining -> 0) is detected
as a WIN that pays the completion reward, instead of being misclassified as a
death because the emerald counter lagged.

Safe:
  - writes a timestamped backup first
  - matches exact anchor strings; aborts if any anchor is missing or ambiguous
  - idempotent: refuses to double-apply (detects the marker)
  - prints a unified diff of what it changed

Run:  python3 apply_win_fix.py
Then: python3 verify_win_fix.py   (should print >>> PASS)
"""
import sys, os, time, difflib

PATH = "special_stage_env.py"

if not os.path.exists(PATH):
    sys.exit(f"ERROR: {PATH} not found. Run this from ~/sonic-rl/")

with open(PATH) as f:
    src = f.read()

if "_board_cleared" in src:
    sys.exit("Looks already patched (_board_cleared present). Aborting to avoid double-apply.")

original = src

# ---- Edit 1: initialize the latch each episode, right after _reset_state() ----
anchor1 = "        self._reset_state()\n"
if src.count(anchor1) != 1:
    sys.exit(f"ERROR: expected exactly one '{anchor1.strip()}' (found {src.count(anchor1)}). Aborting.")
src = src.replace(
    anchor1,
    "        self._reset_state()\n"
    "        self._board_cleared = False\n",
    1,
)

# ---- Edit 2: latch the clear right after prev_blue_spheres is stored ----
anchor2 = "        self.prev_blue_spheres = blue_spheres\n"
if src.count(anchor2) != 1:
    sys.exit(f"ERROR: expected exactly one prev_blue_spheres assignment (found {src.count(anchor2)}). Aborting.")
src = src.replace(
    anchor2,
    "        self.prev_blue_spheres = blue_spheres\n"
    "\n"
    "        # Board cleared = remaining hit zero after collecting >=1 this\n"
    "        # episode. Stage-agnostic, restart-safe win signal: the emerald\n"
    "        # award lags via the fly-off animation and may not credit on\n"
    "        # warped-in restart states.\n"
    "        if blue_spheres == 0 and self._spheres_collected > 0:\n"
    "            self._board_cleared = True\n",
    1,
)

# ---- Edit 3: win on board-clear OR emerald (not emerald alone) ----
anchor3 = (
    "            if emerald_gained:\n"
    "                time_bonus = max(0.0, (SS_MAX_STEPS - self.steps)\n"
    "                                 / SS_MAX_STEPS) * SS_TIME_BONUS_MAX\n"
    "                reward += SS_COMPLETION_BASE + time_bonus\n"
    "                self._log_episode(success=True, emerald_type=emerald_type)\n"
)
if src.count(anchor3) != 1:
    sys.exit("ERROR: could not find the emerald win-branch exactly as expected "
             f"(found {src.count(anchor3)}). The file may have drifted; aborting so "
             "you can apply Edit 3 by hand.")
replacement3 = (
    "            if self._board_cleared or emerald_gained:\n"
    "                time_bonus = max(0.0, (SS_MAX_STEPS - self.steps)\n"
    "                                 / SS_MAX_STEPS) * SS_TIME_BONUS_MAX\n"
    "                reward += SS_COMPLETION_BASE + time_bonus\n"
    "                self._log_episode(success=True,\n"
    "                                  emerald_type=emerald_type or \"cleared\")\n"
)
src = src.replace(anchor3, replacement3, 1)

# ---- backup, write, show diff ----
backup = f"{PATH}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
with open(backup, "w") as f:
    f.write(original)

with open(PATH, "w") as f:
    f.write(src)

diff = difflib.unified_diff(
    original.splitlines(keepends=True),
    src.splitlines(keepends=True),
    fromfile=f"{PATH} (before)", tofile=f"{PATH} (after)",
)
print("".join(diff))
print(f"\nBackup written: {backup}")
print("Patched: special_stage_env.py")
print("Next: python3 verify_win_fix.py   (expect >>> PASS)")
