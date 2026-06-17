import argparse
import json
from pathlib import Path

from text_only_tasks import (
    AA_CONSERVATIVE_TEXT_ONLY_TASK_COUNT,
    AA_EXCLUDED_TASK_IDS,
    AA_STRICT_TASK_COUNT,
    AA_TEXT_ONLY_TASK_COUNT,
    VISION_LIKELY_OR_RISKY_TASK_IDS,
    VISION_REQUIRED_TASK_IDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pass metrics for HuggingFace task outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="output_qwen397b", help="Output directory to score")
    parser.add_argument(
        "--include-vision-tasks",
        action="store_true",
        help="Include tasks classified as vision-required. Defaults to AA text-only evaluation.",
    )
    parser.add_argument(
        "--include-aa-excluded-worlds",
        action="store_true",
        help="Include the two AA-excluded worlds. Defaults to AA-strict evaluation.",
    )
    parser.add_argument(
        "--semantic-text-only",
        action="store_true",
        help="Use the broader 445-task semantic text-only subset instead of the default 420-task conservative subset.",
    )
    parser.add_argument(
        "--conservative-text-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def include_trial(task_id: str, args: argparse.Namespace) -> bool:
    if not args.include_aa_excluded_worlds and task_id in AA_EXCLUDED_TASK_IDS:
        return False
    if args.include_vision_tasks:
        return True
    if task_id in VISION_REQUIRED_TASK_IDS:
        return False
    if not args.semantic_text_only and task_id in VISION_LIKELY_OR_RISKY_TASK_IDS:
        return False
    return True


args = parse_args()
root = Path(args.output_dir)
all_trials = sorted(root.glob("task_*/trial_*"))
trials = [trial for trial in all_trials if include_trial(trial.parent.name, args)]

graded = 0
passed = 0

for t in trials:
    g = t / "grades.json"
    if not g.exists() or g.stat().st_size == 0:
        continue

    with g.open() as f:
        data = json.load(f)
    if data.get("grading_run_status") != "completed":
        continue

    graded += 1
    if data.get("scoring_results", {}).get("final_score") == 1.0:
        passed += 1

print(f"output_dir: {root}")
if args.include_aa_excluded_worlds:
    aa_scope = "all dataset worlds"
else:
    aa_scope = f"AA-strict ({AA_STRICT_TASK_COUNT} expected tasks)"

if args.include_vision_tasks:
    print(f"task_filter: {aa_scope}, vision tasks included")
elif args.semantic_text_only:
    print(f"task_filter: AA semantic text-only ({AA_TEXT_ONLY_TASK_COUNT} expected tasks)")
else:
    print(f"task_filter: AA conservative text-only ({AA_CONSERVATIVE_TEXT_ONLY_TASK_COUNT} expected tasks)")

print(f"trials_excluded_by_filter: {len(all_trials) - len(trials)}")
print(f"passed: {passed}")
print(f"graded: {graded}")
print(f"total_attempted: {len(trials)}")
print(f"pass@1 graded-only: {passed / graded:.4%}" if graded else "pass@1 graded-only: n/a")
print(f"pass@1 all-attempted: {passed / len(trials):.4%}" if trials else "pass@1 all-attempted: n/a")
