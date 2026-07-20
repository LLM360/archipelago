#!/usr/bin/env python3
"""
Run a task from the mercor/apex-agents HuggingFace dataset.

Usage:
    ./run.sh              # Run task index 0
    ./run.sh 42           # Run task index 42
    ./run.sh task_abc123  # Run task by ID
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

EXAMPLE_DIR = Path(os.environ.get("EXAMPLE_DIR", Path(__file__).parent))
ARCHIPELAGO_DIR = Path(os.environ.get("ARCHIPELAGO_DIR", EXAMPLE_DIR.parent.parent))
ENVIRONMENT_DIR = Path(
    os.environ.get("ENVIRONMENT_DIR", ARCHIPELAGO_DIR / "environment")
)
AGENTS_DIR = Path(os.environ.get("AGENTS_DIR", ARCHIPELAGO_DIR / "agents"))
GRADING_DIR = Path(os.environ.get("GRADING_DIR", ARCHIPELAGO_DIR / "grading"))

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8080")
HF_DATASET = "mercor/apex-agents"
SUBSYSTEMS = ["filesystem", ".apps_data"]

# Default task: Investment Banking World 221 - BBDC/TVPG accretion/dilution sensitivity analysis
DEFAULT_TASK = "task_9ba58a6197114140877a1df1754d2993"

AGENT_CONFIG_PROFILES = {
    "loop_agent": {
        "agent_name": "Loop Agent",
        "agent_config_values": {
            "timeout": 3600,
            "max_steps": 100,
            "tool_call_timeout": 60,
            "llm_response_timeout": 600,
        },
    },
    "react_toolbelt_agent": {
        "agent_name": "React Toolbelt Agent",
        "agent_config_values": {
            "timeout": 3600,
            "max_steps": 250,
            "preserve_thinking": True,
        },
    },
}

LOOP_AGENT_SYSTEM_PROMPT = """You are an AI assistant that completes tasks by reasoning and using tools.

## Think Before Acting

Before making tool calls, briefly explain your reasoning in 1-3 sentences:
- What you learned from the previous step
- What you're doing next and why

Don't over-explain. Be concise but show your thinking.

## Tools

All available domain tools are provided directly. Use them as needed to complete the task.

## Workflow

1. Understand the requested outcome
2. Use the available tools to gather information and make the required changes
3. Verify the result when possible
4. When the task is complete, respond with the final answer without calling another tool

## Rules

- Continue using tools while work remains
- Show your work for calculations
- A response without tool calls ends the run, so only provide it when you are finished
"""

REACT_TOOLBELT_SYSTEM_PROMPT = """You are an AI assistant that completes tasks by reasoning and using tools.

## Think Before Acting

Before making tool calls, briefly explain your reasoning in 1-3 sentences:
- What you learned from the previous step
- What you're doing next and why

Don't over-explain. Be concise but show your thinking.

## Tools

**Always Available (Meta-Tools):**
- `todo_write` - Task planning: create/update todos. Takes `todos` array [{id, content, status}] and `merge` boolean.
- `toolbelt_list_tools` / `toolbelt_inspect_tool` / `toolbelt_add_tool` / `toolbelt_remove_tool` - Tool management
- `final_answer` - Submit your answer (status: completed/blocked/failed)

**Domain Tools:** Use `toolbelt_list_tools` to discover, then `toolbelt_add_tool` to add them.

## Workflow

1. Plan: Use `todo_write` to create todos for complex tasks
2. Discover: Use `toolbelt_list_tools` to find relevant tools
3. Execute: Work through todos, use `todo_write` with `merge=true` to update status
4. Complete: Call `final_answer` (all todos must be completed/cancelled first)

## Rules

- Update todo status with `todo_write`: set `in_progress` when starting, `completed` when done
- Show your work for calculations
- `final_answer` is rejected if todos are incomplete
"""

AGENT_SYSTEM_PROMPTS = {
    "loop_agent": LOOP_AGENT_SYSTEM_PROMPT,
    "react_toolbelt_agent": REACT_TOOLBELT_SYSTEM_PROMPT,
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_agent_config() -> dict:
    """Load the base agent config and apply the optional environment selection."""
    with open(EXAMPLE_DIR / "agent_config.json") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise SystemExit("agent_config.json must contain a JSON object")

    configured_id = str(config.get("agent_config_id", "loop_agent")).strip()
    requested_id = os.environ.get("AGENT_CONFIG_ID", configured_id).strip()
    if requested_id not in AGENT_CONFIG_PROFILES:
        supported = ", ".join(sorted(AGENT_CONFIG_PROFILES))
        raise SystemExit(
            f"Unsupported AGENT_CONFIG_ID={requested_id!r}; choose one of: {supported}"
        )

    profile = AGENT_CONFIG_PROFILES[requested_id]
    configured_values = config.get("agent_config_values", {})
    if not isinstance(configured_values, dict):
        raise SystemExit("agent_config.json agent_config_values must be a JSON object")

    # Preserve checked-in/custom values when the file already targets this agent.
    # When the environment switches agent types, start from that agent's defaults
    # so ReAct-only settings do not leak into Loop (and vice versa).
    values = dict(profile["agent_config_values"])
    if requested_id == configured_id:
        values.update(configured_values)

    config["agent_config_id"] = requested_id
    config["agent_name"] = profile["agent_name"]
    config["agent_config_values"] = values
    return config


def project_python_cmd(project_dir: Path, override_env: str) -> list[str]:
    override = os.environ.get(override_env)
    if override:
        return [override]

    venv_python = project_dir / ".venv" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return [str(venv_python)]

    if shutil.which("uv"):
        return ["uv", "run", "python"]

    raise FileNotFoundError(
        f"Could not find {venv_python} or uv. Set {override_env} to the "
        f"Python executable for {project_dir}."
    )


def resolve_output_root() -> Path:
    output_dir = os.environ.get("OUTPUT_DIR")
    if not output_dir:
        return EXAMPLE_DIR / "output"

    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        root = EXAMPLE_DIR / root
    return root.resolve()


def archive_cache_root() -> Path:
    cache_dir = os.environ.get("ARCHIVE_CACHE_DIR") or os.environ.get(
        "PREPARED_ARCHIVE_CACHE_DIR"
    )
    if cache_dir:
        root = Path(cache_dir).expanduser()
        if not root.is_absolute():
            root = EXAMPLE_DIR / root
    else:
        root = EXAMPLE_DIR / ".cache" / "prepared_archives"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def safe_cache_component(value: str) -> str:
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in str(value)
    ).strip("._-")
    return safe or "unknown"


def cache_dir_for(kind: str, key: str) -> Path:
    return archive_cache_root() / kind / safe_cache_component(key)


@contextmanager
def cache_lock(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / "manifest.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_archive_manifest(cache_dir: Path) -> list[dict] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open() as f:
            manifest = json.load(f)
        archives = []
        for item in manifest.get("archives", []):
            subsystem = item.get("subsystem")
            archive_name = item.get("archive")
            if not isinstance(subsystem, str) or not isinstance(archive_name, str):
                return None
            archive_path = cache_dir / archive_name
            if not archive_path.exists():
                return None
            archives.append(
                {
                    "subsystem": subsystem,
                    "path": archive_path,
                    "objects": int(item.get("objects", 0)),
                    "files": int(item.get("files", 0)),
                    "bytes": int(item.get("bytes", archive_path.stat().st_size)),
                }
            )
        return archives
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_archive_manifest(cache_dir: Path, archives: list[dict]) -> None:
    manifest_path = cache_dir / "manifest.json"
    manifest = {
        "archives": [
            {
                "subsystem": archive["subsystem"],
                "archive": Path(archive["path"]).name,
                "objects": archive["objects"],
                "files": archive["files"],
                "bytes": archive["bytes"],
            }
            for archive in archives
        ]
    }
    tmp_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)


def create_subsystem_archive(
    subsystem_dir: Path,
    archive_path: Path,
    label: str,
    subsystem: str,
) -> dict | None:
    if not subsystem_dir.exists():
        return None

    entries = list(subsystem_dir.rglob("*"))
    if not entries:
        return None

    file_count = sum(1 for p in entries if p.is_file())
    log(
        f"  Preparing cached {label} {subsystem} "
        f"({file_count} files, {len(entries) - file_count} directories)..."
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_name(f".{archive_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tarfile.open(tmp_path, "w:gz", compresslevel=1) as tar:
            tar.dereference = True
            for entry in entries:
                tar.add(
                    entry,
                    arcname=str(entry.relative_to(subsystem_dir)),
                    recursive=False,
                )
        os.replace(tmp_path, archive_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "subsystem": subsystem,
        "path": archive_path,
        "objects": len(entries),
        "files": file_count,
        "bytes": archive_path.stat().st_size,
    }


def prepare_archives_from_root(root: Path, cache_dir: Path, label: str) -> list[dict]:
    archives = []
    for subsystem in SUBSYSTEMS:
        archive = create_subsystem_archive(
            root / subsystem,
            cache_dir / f"{subsystem}.tar.gz",
            label,
            subsystem,
        )
        if archive is not None:
            archives.append(archive)
    write_archive_manifest(cache_dir, archives)
    return archives


def cached_world_archives(world_id: str, zip_path: Path) -> list[dict]:
    cache_dir = cache_dir_for("worlds", world_id)
    with cache_lock(cache_dir):
        cached = read_archive_manifest(cache_dir)
        if cached is not None:
            log(f"Using cached world archives for {world_id}: {cache_dir}")
            return cached

        log(f"Creating world archive cache for {world_id}: {cache_dir}")
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
            return prepare_archives_from_root(Path(tmp), cache_dir, "world")


def cached_task_archives(task_id: str) -> list[dict]:
    cache_dir = cache_dir_for("tasks", task_id)
    with cache_lock(cache_dir):
        cached = read_archive_manifest(cache_dir)
        if cached is not None:
            log(f"Using cached task archives for {task_id}: {cache_dir}")
            return cached

        task_prefix = f"task_files/{task_id}"
        log(f"Creating task archive cache for {task_id}: {cache_dir}")
        snapshot_dir = snapshot_download(
            HF_DATASET, repo_type="dataset", allow_patterns=[f"{task_prefix}/**"]
        )
        task_dir = Path(snapshot_dir) / task_prefix
        if not task_dir.exists():
            write_archive_manifest(cache_dir, [])
            return []
        return prepare_archives_from_root(task_dir, cache_dir, "task")


def populate_archive(archive: dict, label: str) -> None:
    subsystem = archive["subsystem"]
    tar_path = Path(archive["path"])
    file_count = archive["files"]
    byte_count = archive["bytes"]
    log(
        f"  Populating {label} {subsystem} from cache "
        f"({file_count} files, {byte_count} bytes)..."
    )
    with tar_path.open("rb") as f:
        resp = httpx.post(
            f"{ENV_URL}/data/populate",
            files={"archive": (tar_path.name, f, "application/gzip")},
            params={"subsystem": subsystem},
            timeout=600.0,
        )
    if resp.status_code != 200:
        log(f"ERROR: Failed to populate {label} {subsystem}: {resp.text}")
        sys.exit(1)
    log(f"  {subsystem}: {resp.json()}")


def populate_archives(archives: list[dict], label: str) -> None:
    if not archives:
        log(f"  No {label} archives to populate")
        return
    for archive in archives:
        populate_archive(archive, label)


def copy_or_link(src: Path, dst: Path) -> None:
    """Materialize src at dst without preserving HF cache symlinks."""
    src = src.resolve(strict=True)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not dst.is_symlink():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return
        except OSError:
            pass

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    tmp_path = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(src, tmp_path)
        except OSError:
            shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)

    if dst.is_symlink() or not dst.exists():
        log(f"ERROR: Failed to materialize snapshot zip: {dst}")
        sys.exit(1)


def wait_for_health(url: str, timeout: int = 480) -> bool:
    """Wait for environment to be healthy."""
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


def start_environment():
    """Start a fresh environment container (always restarts)."""
    env_file = ENVIRONMENT_DIR / ".env"
    env_example = ENVIRONMENT_DIR / ".env.example"
    if not env_file.exists() and env_example.exists():
        log("Creating .env from .env.example...")
        shutil.copy(env_example, env_file)
    elif not env_file.exists():
        log("Creating empty .env file...")
        env_file.touch()

    log("Stopping any existing environment containers...")
    subprocess.run(
        ["docker", "compose", "down", "-v"], cwd=ENVIRONMENT_DIR, capture_output=True
    )

    build_setting = os.environ.get("ENVIRONMENT_BUILD", "1").strip().lower()
    build_each_run = build_setting not in {"0", "false", "no", "off"}
    compose_up_cmd = ["docker", "compose", "up", "-d"]
    compose_up_cmd.append("--build" if build_each_run else "--no-build")

    if build_each_run:
        log("Building and starting environment container...")
    else:
        log("Starting environment container without rebuilding...")
    result = subprocess.run(compose_up_cmd, cwd=ENVIRONMENT_DIR)
    if result.returncode != 0:
        log("ERROR: Failed to start environment")
        sys.exit(1)

    log("Waiting for environment to be healthy...")
    if not wait_for_health(ENV_URL):
        subprocess.run(["docker", "compose", "logs"], cwd=ENVIRONMENT_DIR)
        log("ERROR: Environment failed to start")
        sys.exit(1)

    log("Environment started")


def reset_environment_state():
    """Clear mutable state in an already-running environment."""
    log("Resetting environment state...")
    try:
        resp = httpx.post(f"{ENV_URL}/data/reset", timeout=600.0)
    except httpx.RequestError as e:
        log(f"ERROR: Failed to reset environment state: {e}")
        sys.exit(1)

    if resp.status_code != 200:
        log(f"ERROR: Failed to reset environment state: {resp.text}")
        sys.exit(1)
    log(f"Environment state reset: {resp.json()}")


def prepare_environment():
    """Start a fresh environment or reset an existing reusable worker."""
    if not env_flag("REUSE_ENVIRONMENT"):
        start_environment()
        return

    log("Reusing existing environment container")
    log("Waiting for environment to be healthy...")
    if not wait_for_health(ENV_URL):
        subprocess.run(["docker", "compose", "logs"], cwd=ENVIRONMENT_DIR)
        log("ERROR: Reusable environment is not healthy")
        sys.exit(1)
    reset_environment_state()


def configure_mcp_servers():
    """Configure MCP servers using the all-servers config."""
    log("Configuring MCP servers...")
    with open(EXAMPLE_DIR / "mcp_config_all_oss_servers.json") as f:
        mcp_config = json.load(f)
    server_names = list(mcp_config["mcpServers"].keys())
    log(f"  Servers: {server_names}")

    resp = httpx.post(f"{ENV_URL}/apps", json=mcp_config, timeout=1200.0)
    resp.raise_for_status()
    log("MCP servers configured")


def tar_gz_to_zip(tar_gz_path: Path) -> Path:
    """Convert tar.gz to zip for grading."""
    stem = tar_gz_path.stem
    if stem.endswith(".tar"):
        stem = stem[:-4]
    zip_path = tar_gz_path.parent / f"{stem}.zip"
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        zf.writestr(member.name, f.read())
    return zip_path


def save_final_snapshot(output_dir: Path) -> Path:
    """Save final snapshot as a grading-compatible zip."""
    if env_flag("DIRECT_ZIP_SNAPSHOT", True):
        compression = os.environ.get("SNAPSHOT_ZIP_COMPRESSION", "stored")
        final_zip = output_dir / "final_snapshot.zip"
        log(f"Saving final snapshot as zip (compression={compression})...")
        with httpx.stream(
            "POST",
            f"{ENV_URL}/data/snapshot/zip",
            params={"compression": compression},
            timeout=600.0,
        ) as resp:
            if resp.status_code != 404:
                resp.raise_for_status()
                with final_zip.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                return final_zip
            log("Zip snapshot endpoint unavailable; falling back to tar.gz snapshot")

    log("Saving final snapshot as tar.gz, then converting to zip...")
    with httpx.stream("POST", f"{ENV_URL}/data/snapshot", timeout=600.0) as resp:
        resp.raise_for_status()
        final_tar_gz = output_dir / "final_snapshot.tar.gz"
        with final_tar_gz.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)

    return tar_gz_to_zip(final_tar_gz)


def download_world_snapshot(world_id: str, output_dir: Path) -> tuple[Path, Path]:
    log(f"Downloading world snapshot: {world_id}")
    zip_path = Path(
        hf_hub_download(
            HF_DATASET, f"world_files_zipped/{world_id}.zip", repo_type="dataset"
        )
    )
    world_zip = output_dir / f"{world_id}.zip"
    copy_or_link(zip_path, world_zip)
    return zip_path, world_zip


def read_agent_status(trajectory_file: Path) -> str | None:
    if not trajectory_file.exists():
        return None
    try:
        with trajectory_file.open() as f:
            trajectory = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(f"WARNING: Could not read existing trajectory: {e}")
        return None

    agent_status = trajectory.get("status")
    log(f"Agent status: {agent_status}")
    return agent_status


def write_verifiers(task: dict, world_id: str, output_dir: Path) -> Path:
    verifiers = [
        {
            "verifier_id": c["verifier_id"],
            "verifier_version": 1,
            "world_id": world_id,
            "task_id": task["task_id"],
            "eval_config_id": "ec_output_llm",
            "verifier_values": {
                "criteria": c["criteria"],
                "is_primary_objective": i == 0,
            },
            "verifier_index": i,
            "verifier_dependencies": None,
        }
        for i, c in enumerate(task.get("rubric", []))
    ]
    verifiers_file = output_dir / "verifiers.json"
    with verifiers_file.open("w") as f:
        json.dump(verifiers, f, indent=2)
    return verifiers_file


def run_grading(
    task: dict,
    world_id: str,
    output_dir: Path,
    world_zip: Path,
    final_zip: Path,
    trajectory_file: Path,
    trajectory_id: str,
    grading_run_id: str,
) -> None:
    log("Running grading...")
    verifiers_file = write_verifiers(task, world_id, output_dir)
    grades_file = output_dir / "grades.json"

    grading_cmd = [
        *project_python_cmd(GRADING_DIR, "GRADING_PYTHON"),
        "-m",
        "runner.main",
        "--grading-run-id",
        grading_run_id,
        "--trajectory-id",
        trajectory_id,
        "--initial-snapshot",
        str(world_zip),
        "--final-snapshot",
        str(final_zip),
        "--trajectory",
        str(trajectory_file),
        "--grading-settings",
        str(EXAMPLE_DIR / "grading_settings.json"),
        "--verifiers",
        str(verifiers_file),
        "--eval-configs",
        str(EXAMPLE_DIR / "eval_configs.json"),
        "--scoring-config",
        str(EXAMPLE_DIR / "scoring_config.json"),
        "--output",
        str(grades_file),
    ]

    result = subprocess.run(grading_cmd, cwd=GRADING_DIR)
    if result.returncode != 0:
        log(f"WARNING: Grading exited with code {result.returncode}")

    if grades_file.exists():
        with grades_file.open() as f:
            grades = json.load(f)
        log("=" * 60)
        log("GRADING RESULTS")
        log("=" * 60)
        log(f"Status: {grades.get('grading_run_status')}")
        log(f"Final Score: {grades.get('scoring_results', {}).get('final_score')}")
        for vr in grades.get("verifier_results", []):
            log(f"  - {vr.get('verifier_id')}: {vr.get('score')}")


def log_done(output_dir: Path) -> None:
    log("=" * 60)
    log("DONE")
    log(f"Output: {output_dir}")
    log("=" * 60)


def main():
    # Parse task selector from command line (index, task ID, or use default)
    task_selector = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    # Load task and world data from HuggingFace
    log("Downloading task data from HuggingFace...")
    tasks_path = hf_hub_download(
        HF_DATASET, "tasks_and_rubrics.json", repo_type="dataset"
    )
    worlds_path = hf_hub_download(
        HF_DATASET, "world_descriptions.json", repo_type="dataset"
    )

    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(worlds_path) as f:
        worlds = {w["world_id"]: w for w in json.load(f)}

    # Find the task
    if task_selector.isdigit():
        task_index = int(task_selector)
        if task_index < 0 or task_index >= len(tasks):
            log(f"ERROR: Task index out of range (0-{len(tasks) - 1})")
            sys.exit(1)
        task = tasks[task_index]
    else:
        task = next((t for t in tasks if t["task_id"] == task_selector), None)
        if not task:
            log(f"ERROR: Task not found: {task_selector}")
            sys.exit(1)

    world_id = task["world_id"]
    world = worlds.get(world_id)
    if not world:
        log(f"ERROR: World not found: {world_id}")
        sys.exit(1)

    trajectory_id = f"hf_{task['task_id']}_{uuid.uuid4().hex[:8]}"
    grading_run_id = f"gr_{uuid.uuid4().hex[:8]}"
    trial_id = os.environ.get("TRIAL_ID", "")
    output_dir = resolve_output_root() / task["task_id"]
    if trial_id:
        output_dir = output_dir / trial_id
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_file = output_dir / "trajectory.json"
    grades_file = output_dir / "grades.json"
    existing_final_zip = output_dir / "final_snapshot.zip"

    log("=" * 60)
    log(f"Task: {task['task_name']}")
    log(f"Domain: {task['domain']}")
    log(f"World: {world['world_name']}")
    log(f"Prompt: {task['prompt'][:100]}...")
    log("=" * 60)

    if grades_file.exists() and grades_file.stat().st_size > 0:
        log(f"Grades already exist: {grades_file}")
        log_done(output_dir)
        return

    agent_status = read_agent_status(trajectory_file)
    if (
        agent_status == "completed"
        and existing_final_zip.exists()
        and existing_final_zip.stat().st_size > 0
    ):
        log("Existing completed trajectory and final snapshot found; running grading only")
        _, world_zip = download_world_snapshot(world_id, output_dir)
        run_grading(
            task,
            world_id,
            output_dir,
            world_zip,
            existing_final_zip,
            trajectory_file,
            trajectory_id,
            grading_run_id,
        )
        log_done(output_dir)
        return

    prepare_environment()

    # Download world snapshot and populate cached subsystem archives.
    zip_path, world_zip = download_world_snapshot(world_id, output_dir)

    # Populate world data, then overlay per-task files (order matters).
    log("Populating environment with world snapshot...")
    populate_archives(cached_world_archives(world_id, zip_path), "world")

    if task.get("task_input_files"):
        task_id = task["task_id"]
        log(f"Populating task input files: {task_id}")
        populate_archives(cached_task_archives(task_id), "task")

    if env_flag("SKIP_MCP_CONFIG"):
        log("Skipping MCP server configuration (already configured for this worker)")
    else:
        configure_mcp_servers()

    # Generate initial messages using instructions that match the selected harness.
    agent_config = load_agent_config()
    agent_config_id = agent_config["agent_config_id"]
    system_prompt = AGENT_SYSTEM_PROMPTS[agent_config_id]
    log(f"Agent harness: {agent_config_id}")
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["prompt"]},
    ]
    with open(output_dir / "initial_messages.json", "w") as f:
        json.dump(initial_messages, f, indent=2)

    # Load orchestrator config
    with open(EXAMPLE_DIR / "orchestrator_config.json") as f:
        orchestrator_config = json.load(f)

    if agent_config_id == "react_toolbelt_agent":
        preserve_thinking_env = os.environ.get("PRESERVE_THINKING")
        preserved_thinking_env = os.environ.get("PRESERVED_THINKING")
        if preserve_thinking_env is not None or preserved_thinking_env is not None:
            preserve_thinking = (
                env_flag("PRESERVE_THINKING")
                if preserve_thinking_env is not None
                else env_flag("PRESERVED_THINKING")
            )
            agent_config.setdefault("agent_config_values", {})[
                "preserve_thinking"
            ] = preserve_thinking

        preserve_thinking = bool(
            agent_config.get("agent_config_values", {}).get("preserve_thinking", True)
        )
        log(
            "Preserve thinking "
            + ("enabled" if preserve_thinking else "disabled")
            + " for agent message history"
        )

    agent_config_file = output_dir / "agent_config.json"
    with open(agent_config_file, "w") as f:
        json.dump(agent_config, f, indent=2)

    # Run agent
    log("Running agent...")
    agent_cmd = [
        *project_python_cmd(AGENTS_DIR, "AGENTS_PYTHON"),
        "-m",
        "runner.main",
        "--trajectory-id",
        trajectory_id,
        "--initial-messages",
        str(output_dir / "initial_messages.json"),
        "--mcp-gateway-url",
        f"{ENV_URL}/mcp/",
        "--agent-config",
        str(agent_config_file),
        "--orchestrator-model",
        orchestrator_config["model"],
        "--output",
        str(trajectory_file),
    ]

    # Add extra args if present
    if orchestrator_config.get("extra_args"):
        extra_args_file = output_dir / "orchestrator_extra_args.json"
        with open(extra_args_file, "w") as f:
            json.dump(orchestrator_config["extra_args"], f)
        agent_cmd.extend(["--orchestrator-extra-args", str(extra_args_file)])

    result = subprocess.run(agent_cmd, cwd=AGENTS_DIR)
    if result.returncode != 0:
        log(f"WARNING: Agent exited with code {result.returncode}")

    agent_status = read_agent_status(trajectory_file)

    # Save final snapshot
    final_zip = save_final_snapshot(output_dir)
    log(f"Saved: {final_zip}")

    # Run grading if agent completed
    if agent_status != "completed":
        log(f"Skipping grading (agent status: {agent_status})")
    else:
        run_grading(
            task,
            world_id,
            output_dir,
            world_zip,
            final_zip,
            trajectory_file,
            trajectory_id,
            grading_run_id,
        )

    log_done(output_dir)


if __name__ == "__main__":
    main()
