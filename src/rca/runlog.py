"""Run metadata and JSONL output.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

Every run gets its own folder under $FYP/runs/<experiment>/ holding meta.json
(config, model, revision, git commit, host, library versions, seed) and JSONL
results, so any number can be traced to the code and settings that made it.
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def run_dir(experiment: str, label: str) -> Path:
    root = Path(os.environ.get("FYP", ".")) / "runs" / experiment
    host = socket.gethostname().split(".")[0]
    path = root / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{host}_{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_metadata(config: dict, model_spec=None, seed: int | None = None) -> dict:
    import torch
    import transformers

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),
        "git": git_state(),
        "seed": seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": config,
    }
    if model_spec is not None:
        meta["model"] = {"name": model_spec.name, "revision": model_spec.revision, "dtype": str(model_spec.dtype)}
    return meta


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


class JsonlWriter:
    def __init__(self, path: Path):
        self._fh = open(path, "a")

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
