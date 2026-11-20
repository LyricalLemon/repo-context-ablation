"""Fetch the CrossCodeEval Python repositories and verify every task against them.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.
See rca.data.cceval for what is fetched and what is never kept.

Usage (lab machine, inside the rca environment; network only, no GPU):
    python scripts/fetch_cceval.py --limit 5     # try a few repositories first
    python scripts/fetch_cceval.py               # all 471; resumable

Output in $FYP/data/cceval/:
    repos/<repo_id>/...   the .py files at the pinned commit
    manifest.jsonl        one row per repository (append-only; last row wins)
    tasks.jsonl           one row per task with its verification status
    summary.json          counts, written at the end of each run
    meta.json             run metadata (git commit, host, settings)

Re-running skips repositories already fetched with status "ok" and retries
the rest. Licence-excluded repositories are recorded but never cloned.
"""

import argparse
import collections
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rca.data.cceval import (
    GITHUB_URL, PERMISSIVE, fetch_repo, is_permissive, latest_rows, load_licence_map,
    load_tasks, split_repo_id, verify_task,
)
from rca.runlog import JsonlWriter, git_state, write_json

FYP = Path(os.environ.get("FYP", "/vol/research/Noah_Dias_FYP"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=str(FYP / "benchmarks" / "cceval"))
    ap.add_argument("--out", default=str(FYP / "data" / "cceval"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="fetch at most this many repositories this run")
    ap.add_argument("--work-dir", default=None, help="where temporary clones go (default: system temp, local disk)")
    args = ap.parse_args()

    bench, out = Path(args.benchmark), Path(args.out)
    (out / "repos").mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(bench / "python" / "line_completion.jsonl")
    licences = load_licence_map(bench / "LICENSES" / "project_license_map.txt")
    by_repo = collections.defaultdict(list)
    for t in tasks:
        by_repo[t.repo_id].append(t)

    write_json(out / "meta.json", {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename, "git": git_state(),
        "benchmark": str(bench), "permissive_licences": sorted(PERMISSIVE), "args": vars(args),
    })

    manifest_path, tasks_path = out / "manifest.jsonl", out / "tasks.jsonl"
    done = {k for k, v in latest_rows(manifest_path, "repo_id").items()
            if v["status"] in ("ok", "licence_excluded")}
    todo = [r for r in sorted(by_repo) if r not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(by_repo)} repositories, {len(done)} already done, {len(todo)} to fetch now")

    lock = threading.Lock()
    with JsonlWriter(manifest_path) as manifest, JsonlWriter(tasks_path) as task_log:

        def record(repo_id: str, row: dict, task_status: dict[str, str]) -> None:
            with lock:
                manifest.write(row)
                for t in by_repo[repo_id]:
                    task_log.write({"task_id": t.task_id, "repo_id": repo_id,
                                    "status": task_status.get(t.task_id, row["status"])})

        def work(repo_id: str) -> str:
            slug, short = split_repo_id(repo_id, licences)
            base = {"repo_id": repo_id, "slug": slug, "short_hash": short,
                    "licences": sorted(licences[slug]), "n_tasks": len(by_repo[repo_id]),
                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            if not is_permissive(licences[slug]):
                record(repo_id, {**base, "status": "licence_excluded"}, {})
                return "licence_excluded"
            t0 = time.time()
            res = fetch_repo(GITHUB_URL.format(slug=slug), short, out / "repos" / repo_id, args.work_dir)
            status = {}
            if res.status == "ok":
                status = {t.task_id: verify_task(t, out / "repos" / repo_id) for t in by_repo[repo_id]}
            row = {**base, **res.to_dict(), "seconds": round(time.time() - t0, 1),
                   "n_tasks_ok": sum(s == "ok" for s in status.values())}
            record(repo_id, row, status)
            return res.status

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(work, r): r for r in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                repo_id = futures[fut]
                try:
                    st = fut.result()
                except Exception as exc:  # keep going; the repository is retried next run
                    st = f"error: {exc}"
                print(f"[{i}/{len(todo)}] {repo_id}: {st}", flush=True)

    summarise(out, by_repo)


def summarise(out: Path, by_repo: dict) -> None:
    repos = latest_rows(out / "manifest.jsonl", "repo_id")
    task_rows = latest_rows(out / "tasks.jsonl", "task_id")
    summary = {
        "repositories_total": len(by_repo),
        "repositories_by_status": dict(collections.Counter(r["status"] for r in repos.values())),
        "repositories_not_yet_attempted": len(set(by_repo) - set(repos)),
        "tasks_total": sum(len(v) for v in by_repo.values()),
        "tasks_by_status": dict(collections.Counter(t["status"] for t in task_rows.values())),
        "repositories_with_ai_instruction_files": sorted(
            r["repo_id"] for r in repos.values() if r.get("ai_instruction_files")),
        "py_megabytes": round(sum(r.get("py_bytes", 0) for r in repos.values()) / 1e6, 1),
    }
    write_json(out / "summary.json", summary)
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
