#!/usr/bin/env python3
"""
Batch runner for APEX-Agents Pass@1 evaluation (AA-strict exclusion).

Iterates every task in mercor/apex-agents EXCEPT those belonging to the two
Investment Banking worlds that AA excluded due to external runtime dependencies
(Edgar SEC / FMP): "Investment Banking World 244" and "Investment Banking World 246".

For each remaining task, runs NUM_TRIALS independent trials by invoking main.py
with TRIAL_ID=trial_{N} set in the environment. Output lives at
examples/hugging_face_task/output/{task_id}/trial_{N}/.

Resume-safe: a (task, trial) combo is skipped if its grades.json already exists.

Usage:
    uv run python batch_run.py [--trials 3] [--start-index 0] [--limit N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

EXAMPLE_DIR = Path(__file__).parent
HF_DATASET = "mercor/apex-agents"

# AA-strict exclusion: world names that have external runtime deps (Edgar SEC / FMP).
EXCLUDED_WORLD_NAMES = {
    "Investment Banking World 244",
    "Investment Banking World 246",
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [batch] {msg}", flush=True)


def load_tasks_and_worlds():
    log("Downloading HF dataset metadata (tasks_and_rubrics.json, world_descriptions.json)...")
    tasks_path = hf_hub_download(HF_DATASET, "tasks_and_rubrics.json", repo_type="dataset")
    worlds_path = hf_hub_download(HF_DATASET, "world_descriptions.json", repo_type="dataset")
    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(worlds_path) as f:
        worlds_list = json.load(f)
    worlds = {w["world_id"]: w for w in worlds_list}
    return tasks, worlds


def filter_tasks(tasks, worlds, world_name_prefix: str = ""):
    """Drop tasks belonging to excluded worlds; optionally restrict to a
    world-name prefix (e.g. "Management Consulting"). Return kept + dropped."""
    excluded_world_ids = {
        wid for wid, w in worlds.items() if w["world_name"] in EXCLUDED_WORLD_NAMES
    }
    log(f"Excluded world_ids ({len(excluded_world_ids)}): {sorted(excluded_world_ids)}")
    missing = EXCLUDED_WORLD_NAMES - {worlds[wid]["world_name"] for wid in excluded_world_ids}
    if missing:
        log(f"WARNING: could not locate these excluded world_names: {missing}")

    if world_name_prefix:
        prefix_world_ids = {
            wid for wid, w in worlds.items()
            if w["world_name"].startswith(world_name_prefix)
        }
        log(f"world_name_prefix={world_name_prefix!r} -> {len(prefix_world_ids)} worlds match")
        kept = [t for t in tasks if t["world_id"] in prefix_world_ids and t["world_id"] not in excluded_world_ids]
        dropped = [t for t in tasks if t["world_id"] not in prefix_world_ids or t["world_id"] in excluded_world_ids]
    else:
        kept = [t for t in tasks if t["world_id"] not in excluded_world_ids]
        dropped = [t for t in tasks if t["world_id"] in excluded_world_ids]
    return kept, dropped


def grades_exist(task_id: str, trial_id: str) -> bool:
    grades = EXAMPLE_DIR / "output" / task_id / trial_id / "grades.json"
    return grades.exists() and grades.stat().st_size > 0


def trajectory_exists(task_id: str, trial_id: str) -> bool:
    traj = EXAMPLE_DIR / "output" / task_id / trial_id / "trajectory.json"
    return traj.exists() and traj.stat().st_size > 0


def run_single(task_id: str, trial_id: str, task_timeout: int) -> tuple[int, float]:
    env = os.environ.copy()
    env["TRIAL_ID"] = trial_id
    cmd = [sys.executable, str(EXAMPLE_DIR / "main.py"), task_id]
    start = time.time()
    try:
        result = subprocess.run(cmd, env=env, cwd=EXAMPLE_DIR, timeout=task_timeout)
        rc = result.returncode
    except subprocess.TimeoutExpired:
        log(f"  [TIMEOUT] {task_id} {trial_id} exceeded {task_timeout}s")
        rc = -1
    elapsed = time.time() - start
    return rc, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1, help="Number of trials per task")
    ap.add_argument("--start-index", type=int, default=0, help="Skip the first N kept tasks (for resume)")
    ap.add_argument("--limit", type=int, default=0, help="Max kept tasks to process (0=all)")
    ap.add_argument("--task-timeout", type=int, default=7200, help="Seconds per (task, trial) attempt")
    ap.add_argument("--only-task-ids", type=str, default="", help="Comma-separated task_ids (debug)")
    ap.add_argument("--world-name-prefix", type=str, default="", help='Restrict to worlds whose name starts with this prefix (e.g. "Management Consulting")')
    ap.add_argument("--summary-only", action="store_true", help="Just print stats about what's been graded so far")
    args = ap.parse_args()

    tasks, worlds = load_tasks_and_worlds()
    kept, dropped = filter_tasks(tasks, worlds, world_name_prefix=args.world_name_prefix)
    log(f"Total tasks in dataset: {len(tasks)}")
    log(f"Dropped (AA-excluded worlds): {len(dropped)}")
    log(f"Kept for evaluation: {len(kept)}")

    if args.only_task_ids:
        wanted = set(t.strip() for t in args.only_task_ids.split(",") if t.strip())
        kept = [t for t in kept if t["task_id"] in wanted]
        log(f"Filtered via --only-task-ids to {len(kept)} tasks")

    if args.limit:
        kept = kept[args.start_index : args.start_index + args.limit]
    else:
        kept = kept[args.start_index :]

    log(f"Will process {len(kept)} tasks x {args.trials} trials = {len(kept) * args.trials} runs")

    if args.summary_only:
        already_graded = 0
        for t in kept:
            for trial in range(args.trials):
                if grades_exist(t["task_id"], f"trial_{trial}"):
                    already_graded += 1
        log(f"SUMMARY: {already_graded}/{len(kept) * args.trials} (task,trial) pairs already have grades.json")
        return

    done = 0
    skipped = 0
    failed = 0
    total = len(kept) * args.trials
    start_all = time.time()

    for ti, task in enumerate(kept):
        for trial in range(args.trials):
            trial_id = f"trial_{trial}"
            idx = ti * args.trials + trial + 1

            if grades_exist(task["task_id"], trial_id):
                log(f"({idx}/{total}) SKIP (already graded): {task['task_id']} {trial_id}")
                skipped += 1
                continue

            log(f"({idx}/{total}) RUN: {task['task_id']} {trial_id} — {task['task_name'][:80]}")
            rc, elapsed = run_single(task["task_id"], trial_id, args.task_timeout)
            log(f"   → rc={rc} elapsed={elapsed:.0f}s")

            if rc == 0 and grades_exist(task["task_id"], trial_id):
                done += 1
            else:
                failed += 1

            elapsed_all = time.time() - start_all
            attempted = done + failed
            if attempted > 0:
                avg = elapsed_all / attempted
                remaining = (total - idx) * avg
                log(f"   progress: done={done} failed={failed} skipped={skipped} | avg={avg:.0f}s/run | ETA={remaining / 3600:.1f}h")

    log(f"FINAL: done={done} failed={failed} skipped={skipped} total={total}")


if __name__ == "__main__":
    main()
