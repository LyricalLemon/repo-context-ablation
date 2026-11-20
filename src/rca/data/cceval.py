"""Fetch CrossCodeEval Python repositories at their pinned commits.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

CrossCodeEval ships prompts but not repositories. Each task names its
repository as "<owner>-<name>-<7-character commit>", and the licence map names
it as "<owner>/<name>". For each repository this module:

1. makes a blobless clone (commits and trees, file contents on demand) in a
   temporary folder;
2. resolves the short hash to a full commit, and records its commit date;
3. checks out only the .py files at that commit (sparse checkout, so only
   their contents are downloaded) and copies the regular files out;
4. deletes the clone, so no git history (which holds author names and emails)
   is kept.

Ethics rules from the build brief: .py files only, at a fixed commit, plus
commit dates and counts; no author names or emails; permissive licences only;
repositories with AI-assistant instruction files are flagged. The repository
owner appears in the benchmark's own identifiers and is kept as given.
"""

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

# OSI-approved permissive licences and public-domain dedications. Repositories
# with any licence outside this set are excluded and never fetched.
PERMISSIVE = frozenset({
    "MIT", "MIT-0", "X11", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "Zlib", "PSF-2.0", "Unlicense", "CC0-1.0",
})

# Same list as scripts/probe.py, checked at the repository root.
AI_FILES = ("CLAUDE.md", "AGENTS.md", "GEMINI.md", ".cursorrules", ".cursor",
            ".windsurfrules", ".github/copilot-instructions.md")

GITHUB_URL = "https://github.com/{slug}"


@dataclass(frozen=True)
class Task:
    task_id: str
    repo_id: str
    file: str
    context_start_lineno: int
    prompt: str
    groundtruth: str


def load_tasks(path: str | Path) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            x = json.loads(line)
            m = x["metadata"]
            tasks.append(Task(
                task_id=m["task_id"],
                repo_id=m["repository"],
                file=m["file"],
                context_start_lineno=m["context_start_lineno"],
                prompt=x["prompt"],
                groundtruth=x["groundtruth"],
            ))
    return tasks


def load_licence_map(path: str | Path) -> dict[str, frozenset[str]]:
    """Parse LICENSES/project_license_map.txt: "<owner>/<name> {'MIT', ...}" per line."""
    import ast

    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            slug, _, rest = line.partition(" ")
            out[slug] = frozenset(ast.literal_eval(rest)) if rest else frozenset()
    return out


def split_repo_id(repo_id: str, licence_map: dict) -> tuple[str, str]:
    """Return ("<owner>/<name>", short_hash) for a benchmark repository id.

    Owners and names can both contain hyphens, so the split is resolved by
    looking the id up in the licence map, and must be unique.
    """
    base, short_hash = repo_id.rsplit("-", 1)
    hits = [slug for slug in licence_map if slug.replace("/", "-") == base]
    if len(hits) != 1:
        raise ValueError(f"{repo_id!r} matches {len(hits)} licence-map entries: {hits}")
    return hits[0], short_hash


def is_permissive(licences: frozenset[str]) -> bool:
    return bool(licences) and licences <= PERMISSIVE


def ai_instruction_files(paths: list[str]) -> list[str]:
    found = set()
    for p in paths:
        for name in AI_FILES:
            if p == name or p.startswith(name + "/"):
                found.add(name)
    return sorted(found)


def nth_line_offset(text: str, n: int) -> int:
    """Character offset of the start of line n (0-based), counting "\\n" only."""
    pos = 0
    for _ in range(n):
        pos = text.index("\n", pos) + 1
    return pos


def verify_task(task: Task, repo_dir: Path) -> str:
    """Check that the exported file reproduces the task.

    The file, read from line context_start_lineno, must start with the prompt
    followed by the ground truth. Files are read as UTF-8 text with universal
    newlines, so CRLF files compare equal to the benchmark's LF text.
    Returns "ok", "file_missing", "decode_error" or "mismatch".
    """
    path = repo_dir / task.file
    if not path.is_file():
        return "file_missing"
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "decode_error"
    try:
        start = nth_line_offset(text, task.context_start_lineno)
    except ValueError:
        return "mismatch"
    return "ok" if text.startswith(task.prompt + task.groundtruth, start) else "mismatch"


def _git(*args: str, cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "advice.detachedHead=false", *args],
        cwd=cwd, capture_output=True, text=True, timeout=timeout,
        env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin"},
    )


def _last_line(s: str) -> str:
    lines = [line for line in s.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


@dataclass
class FetchResult:
    status: str                     # ok | clone_failed | commit_missing | checkout_failed
    commit: str | None = None       # full SHA
    commit_date: str | None = None  # committer date, ISO 8601
    n_py_files: int = 0
    py_bytes: int = 0
    py_symlinks_skipped: int = 0
    licence_files: tuple[str, ...] = ()
    ai_instruction_files: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_repo(url: str, short_hash: str, out_dir: Path, work_dir: str | None = None) -> FetchResult:
    """Export the regular .py files of `url` at `short_hash` into `out_dir`.

    out_dir is replaced only when the export succeeds: files go to a
    ".partial" folder first and are renamed into place at the end.
    """
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory(dir=work_dir, prefix="cceval_") as tmp:
        clone = Path(tmp) / "clone"
        r = _git("clone", "--filter=blob:none", "--no-checkout", "--quiet", url, str(clone))
        if r.returncode:
            return FetchResult("clone_failed", error=_last_line(r.stderr))

        r = _git("rev-parse", "--verify", "--quiet", f"{short_hash}^{{commit}}", cwd=clone)
        if r.returncode:
            return FetchResult("commit_missing", error=f"{short_hash} not found")
        sha = r.stdout.strip()
        date = _git("show", "-s", "--format=%cI", sha, cwd=clone).stdout.strip()

        r = _git("ls-tree", "-r", "-z", "--full-tree", sha, cwd=clone)
        entries = []
        for item in r.stdout.split("\0"):
            if item:
                meta, path = item.split("\t", 1)
                mode = meta.split()[0]
                entries.append((mode, path))
        paths = [p for _, p in entries]
        py_regular = [p for m, p in entries if p.endswith(".py") and m in ("100644", "100755")]
        py_links = sum(1 for m, p in entries if p.endswith(".py") and m == "120000")
        licence_files = tuple(sorted(
            p for p in paths if "/" not in p and p.lower().startswith(("license", "licence", "copying"))
        ))

        # Sparse checkout of *.py only, so only those blobs are downloaded.
        _git("config", "core.sparseCheckout", "true", cwd=clone)
        (clone / ".git" / "info").mkdir(exist_ok=True)
        (clone / ".git" / "info" / "sparse-checkout").write_text("*.py\n")
        r = _git("checkout", "--quiet", sha, cwd=clone)
        if r.returncode:
            return FetchResult("checkout_failed", commit=sha, commit_date=date, error=_last_line(r.stderr))

        partial = out_dir.with_name(out_dir.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        total = 0
        for p in py_regular:
            src = clone / p
            if src.is_symlink() or not src.is_file():
                continue
            dst = partial / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            total += dst.stat().st_size
        partial.mkdir(parents=True, exist_ok=True)  # a repository with no .py files still gets a folder
        if out_dir.exists():
            shutil.rmtree(out_dir)
        partial.rename(out_dir)

    return FetchResult(
        status="ok",
        commit=sha,
        commit_date=date,
        n_py_files=len(py_regular),
        py_bytes=total,
        py_symlinks_skipped=py_links,
        licence_files=licence_files,
        ai_instruction_files=tuple(ai_instruction_files(paths)),
    )


def latest_rows(path: Path, key: str) -> dict[str, dict]:
    """Read an append-only JSONL file; the last row for each key wins."""
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out[row[key]] = row
    return out
