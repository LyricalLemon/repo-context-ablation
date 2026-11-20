"""Tests for fetching and verifying CrossCodeEval repositories.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

No network: the end-to-end tests clone a small git repository built in a
temporary folder.
"""

import subprocess
from pathlib import Path

import pytest

from rca.data.cceval import (
    Task, ai_instruction_files, fetch_repo, is_permissive, latest_rows, nth_line_offset,
    split_repo_id, verify_task,
)

LICENCES = {
    "a-b/c-d": frozenset({"MIT"}),
    "a/b-c-d": frozenset({"MIT"}),
    "owner/tool": frozenset({"Apache-2.0"}),
}


def test_split_repo_id_resolves_hyphens_through_the_licence_map():
    assert split_repo_id("owner-tool-abc1234", LICENCES) == ("owner/tool", "abc1234")


def test_split_repo_id_rejects_ambiguous_ids():
    # "a-b-c-d" could be a-b/c-d or a/b-c-d
    with pytest.raises(ValueError, match="2 licence-map entries"):
        split_repo_id("a-b-c-d-abc1234", LICENCES)


def test_is_permissive():
    assert is_permissive(frozenset({"MIT", "Apache-2.0"}))
    assert not is_permissive(frozenset({"MIT", "CC-BY-4.0"}))
    assert not is_permissive(frozenset())


def test_ai_instruction_files():
    paths = ["src/a.py", "CLAUDE.md", ".cursor/rules/x.mdc", "docs/AGENTS.md", ".github/copilot-instructions.md"]
    assert ai_instruction_files(paths) == [".cursor", ".github/copilot-instructions.md", "CLAUDE.md"]


def test_nth_line_offset_counts_newlines_only():
    text = "a\x0cb\nc\nd"          # a form feed is not a line break here
    assert nth_line_offset(text, 0) == 0
    assert nth_line_offset(text, 1) == 4
    assert nth_line_offset(text, 2) == 6


def _task(file="pkg/m.py", start=0, prompt="import x\n\nx.", gt="run()"):
    return Task("t/1", "r", file, start, prompt, gt)


def test_verify_task(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("# header\nimport x\n\nx.run()\nprint(1)\n")
    assert verify_task(_task(start=1), tmp_path) == "ok"
    assert verify_task(_task(start=0), tmp_path) == "mismatch"
    assert verify_task(_task(start=1, gt="stop()"), tmp_path) == "mismatch"
    assert verify_task(_task(start=99), tmp_path) == "mismatch"
    assert verify_task(_task(file="pkg/none.py"), tmp_path) == "file_missing"


def test_verify_task_reads_crlf_files_as_lf(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_bytes(b"import x\r\n\r\nx.run()\r\n")
    assert verify_task(_task(), tmp_path) == "ok"


def test_verify_task_decode_error(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_bytes(b"\xff\xfe\x00bad")
    assert verify_task(_task(), tmp_path) == "decode_error"


def test_latest_rows_last_wins(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"repo_id": "a", "status": "clone_failed"}\n{"repo_id": "a", "status": "ok"}\n')
    assert latest_rows(p, "repo_id")["a"]["status"] == "ok"
    assert latest_rows(tmp_path / "none.jsonl", "repo_id") == {}


# --- end to end on a local repository ----------------------------------------

def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def source_repo(tmp_path):
    src = tmp_path / "src_repo"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "a.py").write_text("def f():\n    return 1\n")
    (src / "pkg" / "b.py").write_text("from pkg import a\n\nx = a.f()\n")
    (src / "README.md").write_text("readme\n")
    (src / "LICENSE").write_text("MIT\n")
    (src / "CLAUDE.md").write_text("instructions\n")
    (src / "link.py").symlink_to("pkg/a.py")
    _git(src, "init", "-q", "-b", "main")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "first")
    first = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    (src / "pkg" / "a.py").write_text("def f():\n    return 2\n")
    _git(src, "commit", "-q", "-am", "second")
    return src, first


def test_fetch_repo_exports_py_at_the_pinned_commit(source_repo, tmp_path):
    src, first = source_repo
    out = tmp_path / "out" / "repo"
    res = fetch_repo(f"file://{src}", first[:7], out, work_dir=str(tmp_path))
    assert res.status == "ok" and res.commit == first
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["pkg/a.py", "pkg/b.py"]                    # no README, no symlink, no .git
    assert (out / "pkg" / "a.py").read_text() == "def f():\n    return 1\n"   # first commit, not HEAD
    assert res.n_py_files == 2 and res.py_symlinks_skipped == 1
    assert res.licence_files == ("LICENSE",)
    assert res.ai_instruction_files == ("CLAUDE.md",)
    assert res.commit_date
    assert not out.with_name("repo.partial").exists()
    assert not list(tmp_path.glob("cceval_*"))                  # clone deleted


def test_fetch_repo_missing_commit(source_repo, tmp_path):
    src, _ = source_repo
    res = fetch_repo(f"file://{src}", "0000000", tmp_path / "out", work_dir=str(tmp_path))
    assert res.status == "commit_missing"
    assert not (tmp_path / "out").exists()


def test_fetch_repo_clone_failed(tmp_path):
    res = fetch_repo(f"file://{tmp_path}/does_not_exist", "abc1234", tmp_path / "out", work_dir=str(tmp_path))
    assert res.status == "clone_failed" and res.error
