#!/usr/bin/env python3
"""Kill test renders of the given app that make no CPU progress for LIMIT seconds; log what was killed."""
import os, re, signal, subprocess, sys, time
app, limit, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seen = {}
while not os.path.exists(log + ".stop"):
    out = subprocess.run(["ps", "-axo", "pid=,time=,command="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(.*)", line)
        if not m or (app + "/Contents/MacOS/Blender --background") not in m.group(3):
            continue
        pid, cpu, cmd = int(m.group(1)), m.group(2), m.group(3)
        now = time.time()
        # CPU time in seconds; a stalled render still polls, so allow a little progress.
        parts = [float(x) for x in cpu.replace("-", ":").split(":")]
        seconds = sum(v * m for v, m in zip(reversed(parts), (1, 60, 3600, 86400)))
        if pid not in seen or seconds - seen[pid][0] > 2.0:
            seen[pid] = (seconds, now)
        elif now - seen[pid][1] >= limit:
            blend = re.findall(r"[^ /]*\.blend", cmd)
            with open(log, "a") as f:
                f.write(f"{time.strftime('%T')} killed stalled pid {pid}: {blend[:1]}\n")
            os.kill(pid, signal.SIGKILL)
            seen.pop(pid, None)
    time.sleep(10)
