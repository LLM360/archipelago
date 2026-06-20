#!/usr/bin/env python3
"""
Parallel batch runner for mercor/apex-agents HuggingFace tasks.

This runs independent (task, trial) pairs across multiple Docker Compose
projects. Each worker gets its own COMPOSE_PROJECT_NAME, ENV_PORT, and ENV_URL,
so environment resets do not collide.

The script is resume-safe: already graded runs are skipped when
outputs/<model_name>/<task_id>/<trial_id>/grades.json exists and is non-empty.

Examples from the repository root:
    agents/.venv/bin/python examples/hugging_face_task/parallel_batch_run.py --workers 2 --trials 1
    agents/.venv/bin/python examples/hugging_face_task/parallel_batch_run.py --workers 4 --base-port 8080 --trials 3
    agents/.venv/bin/python examples/hugging_face_task/parallel_batch_run.py --workers 3 --exclude-task-ids task_a,task_b
    agents/.venv/bin/python examples/hugging_face_task/parallel_batch_run.py --workers 4 --no-preserve-thinking
    agents/.venv/bin/python examples/hugging_face_task/parallel_batch_run.py --summary-only --trials 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from text_only_tasks import (
    AA_CONSERVATIVE_TEXT_ONLY_TASK_COUNT,
    AA_TEXT_ONLY_TASK_COUNT,
    filter_text_only_tasks,
)


EXAMPLE_DIR = Path(__file__).parent.resolve()
ARCHIPELAGO_DIR = Path(os.environ.get("ARCHIPELAGO_DIR", EXAMPLE_DIR.parent.parent)).resolve()
ENVIRONMENT_DIR = Path(os.environ.get("ENVIRONMENT_DIR", ARCHIPELAGO_DIR / "environment")).resolve()
HF_DATASET = "mercor/apex-agents"

DEFAULT_EXCLUDED_WORLD_NAMES = {
    "Investment Banking World 244",
    "Investment Banking World 246",
}

ACTIVE_PROCS_LOCK = threading.Lock()
ACTIVE_PROCS: set[subprocess.Popen] = set()


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    project_name: str
    port: int


@dataclass(frozen=True)
class RunItem:
    index: int
    task_id: str
    task_name: str
    trial_id: str


@dataclass
class Stats:
    total: int
    done: int = 0
    failed: int = 0
    skipped: int = 0
    started: int = 0
    elapsed_sum: float = 0.0
    started_at: float = field(default_factory=time.time)
    completion_times: list[float] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.done + self.failed

    @property
    def finished(self) -> int:
        return self.attempted + self.skipped

    @property
    def running(self) -> int:
        return max(self.started - self.attempted, 0)

    @property
    def queued(self) -> int:
        return max(self.total - self.skipped - self.started, 0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [parallel] {msg}", flush=True)


def parse_csv(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def load_id_file(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value).expanduser()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                ids.add(stripped)
    return ids


def require_modules(modules: list[tuple[str, str]]) -> None:
    missing = [package for module, package in modules if importlib.util.find_spec(module) is None]
    if missing:
        packages = " ".join(f"--with {package}" for package in missing)
        raise SystemExit(
            f"Missing Python dependencies: {', '.join(missing)}. "
            f"Run with: uv run {packages} python examples/hugging_face_task/parallel_batch_run.py ..."
        )


def load_tasks_and_worlds() -> tuple[list[dict], dict[str, dict]]:
    require_modules([("huggingface_hub", "huggingface-hub")])
    from huggingface_hub import hf_hub_download

    log("Downloading HF dataset metadata...")
    tasks_path = hf_hub_download(HF_DATASET, "tasks_and_rubrics.json", repo_type="dataset")
    worlds_path = hf_hub_download(HF_DATASET, "world_descriptions.json", repo_type="dataset")
    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(worlds_path) as f:
        worlds_list = json.load(f)
    return tasks, {w["world_id"]: w for w in worlds_list}


def load_orchestrator_model() -> str:
    with open(EXAMPLE_DIR / "orchestrator_config.json") as f:
        config = json.load(f)
    model = config.get("model")
    if not model:
        raise SystemExit("orchestrator_config.json must define a non-empty model")
    return str(model)


def model_output_name(model: str) -> str:
    name = model.rsplit("/", 1)[-1].strip()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return safe_name or "unknown-model"


def resolve_output_root(output_root: str, model: str) -> Path:
    if output_root:
        root = Path(output_root).expanduser()
    else:
        root = EXAMPLE_DIR / "outputs" / model_output_name(model)

    if not root.is_absolute():
        root = EXAMPLE_DIR / root
    return root.resolve()


def output_dir_for(task_id: str, trial_id: str, output_root: Path) -> Path:
    return output_root / task_id / trial_id


def grades_exist(task_id: str, trial_id: str, output_root: Path) -> bool:
    grades = output_dir_for(task_id, trial_id, output_root) / "grades.json"
    return grades.exists() and grades.stat().st_size > 0


def select_tasks(args: argparse.Namespace) -> list[dict]:
    tasks, worlds = load_tasks_and_worlds()

    excluded_world_names = set()
    if not args.include_aa_excluded_worlds:
        excluded_world_names.update(DEFAULT_EXCLUDED_WORLD_NAMES)
    excluded_world_names.update(parse_csv(args.exclude_world_names))

    excluded_world_ids = {
        world_id
        for world_id, world in worlds.items()
        if world.get("world_name") in excluded_world_names
    }
    if excluded_world_ids:
        log(f"Excluded world_ids ({len(excluded_world_ids)}): {sorted(excluded_world_ids)}")

    selected = list(tasks)
    if args.world_name_prefix:
        prefix_world_ids = {
            world_id
            for world_id, world in worlds.items()
            if world.get("world_name", "").startswith(args.world_name_prefix)
        }
        selected = [task for task in selected if task["world_id"] in prefix_world_ids]
        log(f"world_name_prefix={args.world_name_prefix!r} -> {len(prefix_world_ids)} worlds match")

    if excluded_world_ids:
        selected = [task for task in selected if task["world_id"] not in excluded_world_ids]

    if not args.include_vision_tasks:
        before = len(selected)
        selected = filter_text_only_tasks(
            selected,
            include_aa_excluded=args.include_aa_excluded_worlds,
            conservative=not args.semantic_text_only,
        )
        expected = (
            AA_TEXT_ONLY_TASK_COUNT
            if args.semantic_text_only
            else AA_CONSERVATIVE_TEXT_ONLY_TASK_COUNT
        )
        label = "semantic" if args.semantic_text_only else "conservative"
        log(
            f"AA {label} text-only filter excluded {before - len(selected)} task(s), "
            f"{len(selected)} remain (expected {expected})"
        )

    only_task_ids = parse_csv(args.only_task_ids)
    if only_task_ids:
        selected = [task for task in selected if task["task_id"] in only_task_ids]
        log(f"Filtered via --only-task-ids to {len(selected)} tasks")

    excluded_task_ids = parse_csv(args.exclude_task_ids) | load_id_file(args.exclude_task_id_file)
    if excluded_task_ids:
        before = len(selected)
        selected = [task for task in selected if task["task_id"] not in excluded_task_ids]
        log(f"Excluded task_ids: {before - len(selected)} matched, {len(selected)} remain")

    if args.limit:
        selected = selected[args.start_index : args.start_index + args.limit]
    else:
        selected = selected[args.start_index :]

    log(f"Total tasks in dataset: {len(tasks)}")
    log(f"Tasks selected for evaluation: {len(selected)}")
    return selected


def build_run_plan(tasks: list[dict], trials: int) -> list[RunItem]:
    runs: list[RunItem] = []
    for task in tasks:
        for trial in range(trials):
            runs.append(
                RunItem(
                    index=len(runs) + 1,
                    task_id=task["task_id"],
                    task_name=task.get("task_name", ""),
                    trial_id=f"trial_{trial}",
                )
            )
    return runs


def ensure_environment_env_file() -> None:
    env_file = ENVIRONMENT_DIR / ".env"
    env_example = ENVIRONMENT_DIR / ".env.example"
    if env_file.exists():
        return
    if env_example.exists():
        log("Creating environment/.env from environment/.env.example...")
        shutil.copy(env_example, env_file)
    else:
        log("Creating empty environment/.env...")
        env_file.touch()


def compose_env(spec: WorkerSpec) -> dict[str, str]:
    env = os.environ.copy()
    env["COMPOSE_PROJECT_NAME"] = spec.project_name
    env["ENV_PORT"] = str(spec.port)
    return env


def cleanup_worker_projects(worker_specs: list[WorkerSpec], label: str) -> None:
    log(label)
    for spec in worker_specs:
        result = subprocess.run(
            ["docker", "compose", "down", "-v"],
            cwd=ENVIRONMENT_DIR,
            env=compose_env(spec),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            log(f"  w{spec.worker_id}: docker compose down exited {result.returncode}")
            if result.stdout.strip():
                log(f"  w{spec.worker_id}: {result.stdout.strip()[-500:]}")


def prebuild_environment() -> None:
    log("Building environment image once...")
    result = subprocess.run(["docker", "compose", "build"], cwd=ENVIRONMENT_DIR)
    if result.returncode != 0:
        raise SystemExit(f"docker compose build failed with exit code {result.returncode}")


def wait_for_worker_health(spec: WorkerSpec, timeout: int = 480) -> bool:
    import httpx

    url = f"http://localhost:{spec.port}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    return False


def start_worker_environment(spec: WorkerSpec, build: bool) -> None:
    compose_up_cmd = ["docker", "compose", "up", "-d"]
    compose_up_cmd.append("--build" if build else "--no-build")
    result = subprocess.run(
        compose_up_cmd,
        cwd=ENVIRONMENT_DIR,
        env=compose_env(spec),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        tail = result.stdout.strip()[-1200:]
        raise RuntimeError(f"docker compose up failed with exit code {result.returncode}: {tail}")

    if wait_for_worker_health(spec):
        return

    logs = subprocess.run(
        ["docker", "compose", "logs"],
        cwd=ENVIRONMENT_DIR,
        env=compose_env(spec),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail = logs.stdout.strip()[-2000:]
    raise RuntimeError(f"environment failed to become healthy: {tail}")


def configure_worker_mcp(spec: WorkerSpec, mcp_config: dict) -> float:
    import httpx

    url = f"http://localhost:{spec.port}"
    start = time.time()
    resp = httpx.post(f"{url}/apps", json=mcp_config, timeout=1200.0)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"MCP configuration failed: {e.response.text[:1200]}") from e
    return time.time() - start


def prepare_reusable_worker(spec: WorkerSpec, build: bool, mcp_config: dict) -> float:
    start_worker_environment(spec, build=build)
    return configure_worker_mcp(spec, mcp_config)


def prepare_reusable_worker_environments(worker_specs: list[WorkerSpec], args: argparse.Namespace) -> None:
    with open(EXAMPLE_DIR / "mcp_config_all_oss_servers.json") as f:
        mcp_config = json.load(f)

    server_names = list(mcp_config["mcpServers"].keys())
    setup_workers = min(args.setup_concurrency, len(worker_specs))
    log(
        f"Starting and configuring {len(worker_specs)} reusable worker environment(s) "
        f"with setup_concurrency={setup_workers}..."
    )
    log(f"  MCP servers: {server_names}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=setup_workers) as executor:
        future_to_spec = {
            executor.submit(prepare_reusable_worker, spec, args.build_per_run, mcp_config): spec
            for spec in worker_specs
        }
        for future in concurrent.futures.as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                mcp_elapsed = future.result()
            except Exception as e:
                raise SystemExit(f"w{spec.worker_id} failed reusable worker setup: {e}") from e
            log(f"  w{spec.worker_id}: ready on http://localhost:{spec.port} (MCP {mcp_elapsed:.1f}s)")


def port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def check_ports(worker_specs: list[WorkerSpec]) -> None:
    busy = [spec.port for spec in worker_specs if not port_is_available(spec.port)]
    if busy:
        ports = ", ".join(str(port) for port in busy)
        raise SystemExit(
            f"Ports are already in use: {ports}. Use --base-port, stop the service, "
            "or pass --skip-port-check if you know the stale workers will be cleaned up."
        )


def terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait()


def stop_active_processes() -> None:
    with ACTIVE_PROCS_LOCK:
        procs = list(ACTIVE_PROCS)
    for proc in procs:
        if proc.poll() is None:
            terminate_process_group(proc)


def run_process(
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    timeout: int,
    stdout_target,
) -> tuple[int, float, bool]:
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout_target,
        stderr=subprocess.STDOUT if stdout_target is not None else None,
        start_new_session=True,
    )
    with ACTIVE_PROCS_LOCK:
        ACTIVE_PROCS.add(proc)
    try:
        try:
            rc = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            terminate_process_group(proc)
            rc = -1
            timed_out = True
        return rc, time.time() - start, timed_out
    finally:
        with ACTIVE_PROCS_LOCK:
            ACTIVE_PROCS.discard(proc)


def run_single(item: RunItem, spec: WorkerSpec, args: argparse.Namespace) -> tuple[int, float, bool, Path | None]:
    output_dir = output_dir_for(item.task_id, item.trial_id, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = None if args.stream_logs else output_dir / "run.log"

    env = os.environ.copy()
    env["TRIAL_ID"] = item.trial_id
    env["COMPOSE_PROJECT_NAME"] = spec.project_name
    env["ENV_PORT"] = str(spec.port)
    env["ENV_URL"] = f"http://localhost:{spec.port}"
    if args.restart_env_per_task:
        env["ENVIRONMENT_BUILD"] = "1" if args.build_per_run else "0"
    else:
        env["ENVIRONMENT_BUILD"] = "0"
        env["REUSE_ENVIRONMENT"] = "1"
        env["SKIP_MCP_CONFIG"] = "1"
    env["OUTPUT_DIR"] = str(args.output_root)
    env["PRESERVE_THINKING"] = "1" if args.preserve_thinking else "0"

    cmd = [sys.executable, str(EXAMPLE_DIR / "main.py"), item.task_id]

    if args.stream_logs:
        rc, elapsed, timed_out = run_process(cmd, env, EXAMPLE_DIR, args.task_timeout, None)
    else:
        assert log_path is not None
        log_mode = "a" if log_path.exists() and log_path.stat().st_size > 0 else "w"
        with log_path.open(log_mode) as f:
            if log_mode == "a":
                f.write(
                    "\n\n"
                    + "=" * 80
                    + f"\nresume_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
            f.write(
                f"task_id={item.task_id}\n"
                f"trial_id={item.trial_id}\n"
                f"worker={spec.worker_id}\n"
                f"project={spec.project_name}\n"
                f"port={spec.port}\n"
                f"preserve_thinking={args.preserve_thinking}\n"
                f"cmd={' '.join(cmd)}\n\n"
            )
            f.flush()
            rc, elapsed, timed_out = run_process(cmd, env, EXAMPLE_DIR, args.task_timeout, f)
            if timed_out:
                f.write(f"\n[TIMEOUT] exceeded {args.task_timeout}s\n")
    return rc, elapsed, timed_out, log_path


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "pending"
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def completion_rate(stats: Stats, workers: int) -> tuple[float, str]:
    attempted = stats.attempted
    elapsed_wall = max(time.time() - stats.started_at, 1.0)
    overall_rate = attempted / elapsed_wall if attempted else 0.0

    window_size = min(max(workers * 2, 8), 40)
    window = stats.completion_times[-window_size:]
    min_recent = min(max(workers, 4), 10)
    if len(window) >= min_recent:
        span = max(window[-1] - window[0], 1.0)
        recent_rate = (len(window) - 1) / span
        if recent_rate > 0:
            return recent_rate, "recent"

    return overall_rate, "overall"


def progress_line(stats: Stats, workers: int) -> str:
    attempted = stats.attempted
    avg = stats.elapsed_sum / attempted if attempted else 0.0
    remaining = max(stats.total - stats.finished, 0)
    rate, rate_source = completion_rate(stats, workers)
    eta_seconds = remaining / rate if rate > 0 else None
    elapsed_wall = time.time() - stats.started_at
    throughput = rate * 3600
    return (
        f"progress: done={stats.done} failed={stats.failed} skipped={stats.skipped} "
        f"running={stats.running} queued={stats.queued} ({stats.finished}/{stats.total}) | "
        f"avg_run={avg:.0f}s | throughput={throughput:.1f}/h {rate_source} | "
        f"elapsed={format_duration(elapsed_wall)} | ETA={format_duration(eta_seconds)}"
    )


def worker_loop(
    spec: WorkerSpec,
    work_queue: queue.Queue[RunItem],
    stats: Stats,
    stats_lock: threading.Lock,
    args: argparse.Namespace,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            item = work_queue.get_nowait()
        except queue.Empty:
            return

        try:
            if grades_exist(item.task_id, item.trial_id, args.output_root):
                with stats_lock:
                    stats.skipped += 1
                    log(f"[w{spec.worker_id}] SKIP already graded: {item.task_id} {item.trial_id}")
                    log(progress_line(stats, args.active_workers))
                continue

            with stats_lock:
                stats.started += 1
                log(
                    f"[w{spec.worker_id} port={spec.port}] "
                    f"({item.index}/{stats.total}) RUN: {item.task_id} {item.trial_id} - "
                    f"{item.task_name[:80]}"
                )

            rc, elapsed, timed_out, log_path = run_single(item, spec, args)
            graded = grades_exist(item.task_id, item.trial_id, args.output_root)
            succeeded = rc == 0 and graded

            with stats_lock:
                stats.elapsed_sum += elapsed
                stats.completion_times.append(time.time())
                if len(stats.completion_times) > 200:
                    del stats.completion_times[:-200]
                if succeeded:
                    stats.done += 1
                    status = "DONE"
                else:
                    stats.failed += 1
                    status = "TIMEOUT" if timed_out else "FAILED"
                suffix = f" log={log_path}" if log_path else ""
                log(f"[w{spec.worker_id}] {status}: {item.task_id} {item.trial_id} rc={rc} elapsed={elapsed:.0f}s{suffix}")
                log(progress_line(stats, args.active_workers))
        finally:
            work_queue.task_done()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HuggingFace benchmark tasks in parallel with isolated Docker Compose workers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workers", type=positive_int, default=32, help="Number of parallel environment workers")
    parser.add_argument("--base-port", type=int, default=9090, help="First host port; workers use base-port + worker_id")
    parser.add_argument("--project-prefix", default="archipelago_bench", help="Compose project prefix for worker containers")
    parser.add_argument("--trials", type=positive_int, default=1, help="Number of trials per task")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N selected tasks")
    parser.add_argument("--limit", type=int, default=0, help="Max selected tasks to process; 0 means all")
    parser.add_argument("--task-timeout", type=int, default=7200, help="Seconds per (task, trial) attempt")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("OUTPUT_DIR", ""),
        help="Directory for run outputs; defaults to outputs/<model_name> under this example",
    )
    parser.add_argument("--only-task-ids", default="", help="Comma-separated task IDs to run")
    parser.add_argument("--exclude-task-ids", default="", help="Comma-separated task IDs to skip")
    parser.add_argument("--exclude-task-id-file", default="", help="File containing task IDs to skip, one per line")
    parser.add_argument("--exclude-world-names", default="", help="Additional comma-separated world names to skip")
    parser.add_argument("--include-aa-excluded-worlds", action="store_true", help="Do not apply the default AA-strict excluded worlds")
    parser.add_argument("--include-vision-tasks", action="store_true", help="Run tasks known to produce image inputs; defaults to text-only tasks")
    parser.add_argument("--semantic-text-only", action="store_true", help="Use the broader 445-task AA semantic text-only subset; default is the 420-task conservative subset")
    parser.add_argument("--conservative-text-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--world-name-prefix", default="", help="Restrict to worlds whose names start with this prefix")
    parser.add_argument("--summary-only", action="store_true", help="Only print resume stats; do not run Docker or tasks")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected work without running Docker or tasks")
    parser.add_argument("--no-prebuild", action="store_true", help="Skip the initial docker compose build")
    parser.add_argument("--build-per-run", action="store_true", help="Build during task startup in restart mode, or during worker startup in reuse mode")
    parser.add_argument("--restart-env-per-task", action="store_true", help="Restart Docker and reconfigure MCP for every task instead of reusing one environment per worker")
    parser.add_argument("--setup-concurrency", type=positive_int, default=8, help="Max reusable worker environments to start/configure at the same time")
    parser.add_argument("--skip-start-cleanup", action="store_true", help="Do not run docker compose down for worker projects before starting")
    parser.add_argument("--keep-env-running", action="store_true", help="Leave worker environments running at the end")
    parser.add_argument("--skip-port-check", action="store_true", help="Do not check whether worker ports are free before starting")
    parser.add_argument("--stream-logs", action="store_true", help="Stream child task logs instead of writing per-run run.log files")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--preserve-thinking",
        "--preserved-thinking",
        dest="preserve_thinking",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Preserve/fill assistant reasoning_content when replaying message history (default)",
    )
    thinking_group.add_argument(
        "--no-preserve-thinking",
        dest="preserve_thinking",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable assistant reasoning_content preservation",
    )
    args = parser.parse_args()
    if not hasattr(args, "preserve_thinking"):
        args.preserve_thinking = True

    if args.base_port < 1 or args.base_port + args.workers - 1 > 65535:
        raise SystemExit("Invalid --base-port/--workers combination; ports must be in 1..65535")
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.task_timeout < 1:
        raise SystemExit("--task-timeout must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    orchestrator_model = load_orchestrator_model()
    args.output_root = resolve_output_root(args.output_root, orchestrator_model)

    log(f"Orchestrator model: {orchestrator_model}")
    log(f"Output root: {args.output_root}")

    tasks = select_tasks(args)
    runs = build_run_plan(tasks, args.trials)
    already_graded = sum(1 for item in runs if grades_exist(item.task_id, item.trial_id, args.output_root))
    pending_runs = [item for item in runs if not grades_exist(item.task_id, item.trial_id, args.output_root)]

    log(f"Planned runs: {len(runs)} ({len(tasks)} tasks x {args.trials} trials)")
    log(f"Resume state: {already_graded} already graded, {len(pending_runs)} pending")

    if args.summary_only:
        return

    if args.dry_run:
        for item in pending_runs[:20]:
            log(f"DRY RUN pending: {item.task_id} {item.trial_id} - {item.task_name[:80]}")
        if len(pending_runs) > 20:
            log(f"DRY RUN omitted {len(pending_runs) - 20} additional pending runs")
        return

    if not pending_runs:
        log("Nothing to run.")
        return

    require_modules([("httpx", "httpx")])

    all_worker_specs = [
        WorkerSpec(
            worker_id=i,
            project_name=f"{args.project_prefix}_w{i}",
            port=args.base_port + i,
        )
        for i in range(args.workers)
    ]

    args.active_workers = min(args.workers, len(pending_runs))
    if args.active_workers < args.workers:
        log(f"Using {args.active_workers} active worker(s) for {len(pending_runs)} pending run(s)")

    worker_specs = all_worker_specs[: args.active_workers]

    log("Workers:")
    for spec in worker_specs:
        log(f"  w{spec.worker_id}: project={spec.project_name} ENV_URL=http://localhost:{spec.port}")

    ensure_environment_env_file()

    if not args.skip_start_cleanup:
        cleanup_worker_projects(all_worker_specs, "Cleaning worker Compose projects before start...")

    if not args.skip_port_check:
        check_ports(worker_specs)

    if not args.no_prebuild:
        prebuild_environment()

    stats = Stats(total=len(runs), skipped=already_graded)
    stats_lock = threading.Lock()
    stop_event = threading.Event()
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    futures: list[concurrent.futures.Future] = []

    try:
        if args.restart_env_per_task:
            log("Worker environment reuse disabled; each task will restart Docker and configure MCP.")
        else:
            prepare_reusable_worker_environments(worker_specs, args)

        stats.started_at = time.time()
        work_queue: queue.Queue[RunItem] = queue.Queue()
        for item in pending_runs:
            work_queue.put(item)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.active_workers)
        futures = [
            executor.submit(worker_loop, spec, work_queue, stats, stats_lock, args, stop_event)
            for spec in worker_specs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        stop_event.set()
        log("Interrupted. Terminating active child process groups...")
        stop_active_processes()
        for future in futures:
            future.cancel()
        raise
    except BaseException:
        stop_event.set()
        stop_active_processes()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if not args.keep_env_running:
            cleanup_worker_projects(worker_specs, "Cleaning worker Compose projects after run...")

    log(f"FINAL: done={stats.done} failed={stats.failed} skipped={stats.skipped} total={stats.total}")


if __name__ == "__main__":
    main()
