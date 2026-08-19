#!/usr/bin/env python3
"""
ss_brief.py -- run an analyzer and print a terminal-friendly condensed view.

Keeps every section header and summary line; collapses the giant repeating
blocks (the ~400-row learning curve and the per-500 bar charts) down to just
the most recent rows. Does not modify the analyzer.

Usage:
  python3 ss_brief.py                          # runs analyze_ss.py
  python3 ss_brief.py analyze_curriculum.py    # runs the curriculum analyzer
  RECENT=20 python3 ss_brief.py                # show last 20 curve rows

Tip: alias it.   alias ssb='python3 ~/sonic-rl/ss_brief.py'
"""
import os, re, sys, subprocess

target = sys.argv[1] if len(sys.argv) > 1 else "analyze_ss.py"
passthru = sys.argv[2:]
RECENT = int(os.environ.get("RECENT", "10"))

proc = subprocess.run([sys.executable, target, *passthru],
                      capture_output=True, text=True)
if proc.returncode != 0:
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)

# rows we collapse to a recent tail (learning curve, curriculum gained rows)
keep_tail = re.compile(r"avg=")
# any enumerated per-bucket row: starts with "ep N-M" or "death N-M"
enum_row = re.compile(r"^\s*(ep|death)\s+\d+\s*-\s*\d+\b")
# pure bar/percent rows we drop entirely (bumper / trail / loop per-500)
bar_row = re.compile(r"[█░]|%\s*$")

out, buf = [], []

def flush():
    if not buf:
        return
    if len(buf) > RECENT:
        out.append(f"  ... ({len(buf) - RECENT} earlier rows hidden, RECENT={RECENT}) ...")
    out.extend(buf[-RECENT:])
    buf.clear()

for ln in proc.stdout.splitlines():
    if enum_row.match(ln):
        if keep_tail.search(ln):      # curve / gained row -> buffer for tail
            buf.append(ln)
        # else: per-500 bar row -> drop (handled by falling through)
        elif bar_row.search(ln):
            continue
        else:
            flush()
            out.append(ln)
        continue
    flush()
    out.append(ln)
flush()

print("\n".join(out))
