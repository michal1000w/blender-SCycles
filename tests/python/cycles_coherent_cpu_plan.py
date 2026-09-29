#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive a CPU (or output-redirected Metal) plan from an existing coherent acceptance plan.

Scenes, references, hashes and prospective gates are copied unchanged. Only
path-tracing and path-guiding jobs are kept, because BDPT is a Metal feature.
Every output path is redirected into a new directory so no earlier evidence
is overwritten. The resulting plan runs with cycles_coherent_mirror_batch_worker.py
and is checked by cycles_coherent_polarizer_check.py like the Metal plans.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--metal", action="store_true",
                        help="Keep every job on Metal (PT, BDPT, guiding); only redirect outputs")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def redirect(path):
        path = Path(path)
        # Keep the last two components (transport folder and file name) unique.
        target = out_dir / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)

    jobs = []
    for job in plan["jobs"]:
        arguments = list(job["arguments"])
        if not args.metal and "--pt" not in arguments and "--pt-guiding" not in arguments:
            continue
        for flag in ("--render", "--report", "--phase0-data"):
            if flag in arguments:
                index = arguments.index(flag) + 1
                arguments[index] = redirect(arguments[index])
        if not args.metal:
            arguments.append("--cpu")
        new_job = dict(job)
        new_job["case"] = ("metal_" if args.metal else "cpu_") + job["case"]
        new_job["arguments"] = arguments
        new_job["stdout"] = str(out_dir / (new_job["case"] + "_stdout.log"))
        new_job["stderr"] = str(out_dir / (new_job["case"] + "_stderr.log"))
        jobs.append(new_job)
    derived = dict(plan)
    derived["scope"] = ("Metal output redirection of " if args.metal else "CPU derivation of ") + str(args.plan.resolve()) + ": " + plan.get("scope", "")
    derived["source_plan"] = str(args.plan.resolve())
    derived["state"] = str(out_dir / "batch_state.json")
    derived["jobs"] = jobs
    derived.pop("pending_transport_api", None)
    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    args.out_plan.write_text(json.dumps(derived, indent=1) + "\n")
    print(f"{len(jobs)} {'Metal' if args.metal else 'CPU'} jobs -> {args.out_plan}")


if __name__ == "__main__":
    main()
