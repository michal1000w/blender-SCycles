# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged coherence worker sequentially; preserve per-case failures."""
import contextlib
import datetime
import json
from pathlib import Path
import runpy
import sys
import traceback

jobs_path = Path(sys.argv[sys.argv.index("--") + 1])
request = json.loads(jobs_path.read_text())
state_path = Path(request["state"])
state = {"jobs": {}}

def save():
    state_path.write_text(json.dumps(state, indent=2) + "\n")

for job in request["jobs"]:
    entry = {"status": "running", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    state["jobs"][job["case"]] = entry
    save()
    with Path(job["stdout"]).open("w") as stdout, Path(job["stderr"]).open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                sys.argv = [request["worker"], "--", *job["arguments"]]
                runpy.run_path(request["worker"], run_name="__main__")
                entry.update(status="completed", exit_code=0)
            except BaseException as error:
                traceback.print_exc()
                entry.update(status="error", exit_code=73, error=f"{type(error).__name__}: {error}")
    entry["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(f"batch case {job['case']}: {entry['status']}", flush=True)
if any(job["exit_code"] != 0 for job in state["jobs"].values()):
    raise RuntimeError("One or more shared-process workers failed; see individual reports")
