#!/usr/bin/env python3
"""Feasibility probe (throwaway, AI-generated; disclose in the AI-use log).

Question: in each candidate domain, how much cross-file structure does the Python
code contain, i.e. how many call sites reach a function or class defined in a
DIFFERENT file of the same repository? Those are the sites a repository-context
experiment could mask, so they are the raw material for the task set.

Per repository it records (no author data is read or stored):
  - licence label guessed from the LICENSE text (check by eye before relying on it)
  - non-test .py files and lines of code
  - call sites by origin: repo-internal cross-file / third-party / stdlib / other
  - number of commits since two dates (counts only)
  - whether AI-assistant instruction files exist at HEAD

Usage: python3 -I probe.py <clone_dir> <output_json>
"""
from __future__ import annotations

import ast
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("Usage: python3 -I probe.py <clone_dir> <output_json>")
CLONES = Path(sys.argv[1])
OUT = Path(sys.argv[2])

# Hand-picked from memory, so biased towards well-known, mature repositories.
REPOS = {
    "finance": [
        "PyPortfolio/PyPortfolioOpt", "ranaroussi/yfinance", "microsoft/qlib",
        "pmorissette/bt", "bukosabino/ta", "ranaroussi/quantstats",
        "AI4Finance-Foundation/FinRL",
    ],
    "biology": [
        "biopython/biopython", "scverse/scanpy", "scverse/anndata",
        "scikit-bio/scikit-bio", "biotite-dev/biotite",
        "pysam-developers/pysam", "MouseLand/cellpose",
    ],
    "sustainability": [
        "PyPSA/PyPSA", "pvlib/pvlib-python", "e2nIEE/pandapower",
        "oemof/oemof-solph", "calliope-project/calliope",
        "mlco2/codecarbon", "IAMconsortium/pyam",
    ],
}
SINCE = ["2025-02-01", "2025-06-01", "2026-01-01"]
STDLIB = set(sys.stdlib_module_names)
AI_FILES = ["CLAUDE.md", "AGENTS.md", "GEMINI.md", ".cursorrules", ".cursor",
            ".windsurfrules", ".github/copilot-instructions.md"]
SKIP_DIRS = {"build", "dist", "docs", "doc", "examples", "benchmarks", "node_modules"}


def sh(args, cwd=None, timeout=600):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def clone(slug):
    dest = CLONES / slug.replace("/", "__")
    if not dest.exists():
        # blob:none keeps full commit history (for counts) but only fetches files for HEAD.
        r = sh(["git", "clone", "--filter=blob:none", "--no-tags", "--quiet",
                f"https://github.com/{slug}.git", str(dest)], timeout=900)
        if r.returncode:
            return slug, None, r.stderr[-300:]
    return slug, dest, ""


def licence(root):
    cands = sorted(root.glob("LICEN[CS]E*")) + sorted(root.glob("COPYING*"))
    files = [p for p in cands if p.is_file()]
    for d in (p for p in cands if p.is_dir()):  # REUSE-style LICENSES/ folder
        files += sorted(q for q in d.iterdir() if q.is_file())
    for p in files:
        t = p.read_text(errors="ignore")[:8000].lower()
        if "gnu affero" in t: return "AGPL"
        if "gnu lesser" in t: return "LGPL"
        if "gnu general public" in t: return "GPL"
        if "apache license" in t: return "Apache-2.0"
        if "permission is hereby granted, free of charge" in t: return "MIT"
        if "redistribution and use in source and binary forms" in t: return "BSD-style"
        return f"other ({p.name})"
    return "none found"


def is_test(rel):
    return (any(p in {"test", "tests", "testing"} or p.startswith("test_") for p in rel.parts)
            or rel.name.endswith("_test.py") or rel.name == "conftest.py")


def dotted(root, rel):
    """Dotted module name, using __init__.py to find the package root."""
    d, pkg = root / rel.parent, []
    while d != root and (d / "__init__.py").exists():
        pkg.insert(0, d.name)
        d = d.parent
    if rel.stem == "__init__":
        return ".".join(pkg) or None
    return ".".join(pkg + [rel.stem])


def load(root):
    files, fails = {}, 0
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if p.stat().st_size > 1_000_000:
            continue
        try:
            src = p.read_text(errors="ignore")
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            fails += 1
            continue
        files[rel] = {"tree": tree, "loc": len(src.splitlines()), "test": is_test(rel),
                      "dotted": dotted(root, rel), "init": rel.name == "__init__.py"}
    return files, fails


def resolve_target(f, node):
    """Absolute dotted module that an ImportFrom node points at."""
    if not node.level:
        return node.module
    d = f["dotted"]
    if d is None:
        return None
    base = d if f["init"] else d.rpartition(".")[0]
    parts = base.split(".") if base else []
    up = node.level - 1
    if up > len(parts):
        return None
    parts = parts[:len(parts) - up]
    return ".".join(parts + ([node.module] if node.module else []))


def analyse(root):
    files, fails = load(root)
    # dotted name -> file; drop ambiguous names (two files with the same one)
    dmap, seen = {}, set()
    for rel, f in files.items():
        d = f["dotted"]
        if d is None:
            continue
        if d in seen:
            dmap.pop(d, None)
        else:
            dmap[d] = rel
            seen.add(d)
    tops = {d.split(".")[0] for d in dmap}

    # what each file defines at top level, then what its __init__ re-exports
    exports = {rel: {n.name: (rel, n.name) for n in f["tree"].body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
               for rel, f in files.items()}
    for _ in range(3):
        for rel, f in files.items():
            if not f["init"]:
                continue
            for n in ast.walk(f["tree"]):
                if isinstance(n, ast.ImportFrom):
                    t = resolve_target(f, n)
                    if t in dmap:
                        src_exports = exports[dmap[t]]
                        for a in n.names:
                            if a.name in src_exports:
                                exports[rel].setdefault(a.asname or a.name, src_exports[a.name])

    stats = {"internal": 0, "external": 0, "stdlib": 0, "internal_unresolved": 0, "other": 0}
    per_file_internal, targets, files_n, loc = [], set(), 0, 0
    for rel, f in files.items():
        if f["test"]:
            continue
        files_n += 1
        loc += f["loc"]
        sym, mod, ext = {}, {}, {}
        for n in ast.walk(f["tree"]):
            if isinstance(n, ast.ImportFrom):
                t = resolve_target(f, n)
                if t is None:
                    continue
                top = t.split(".")[0]
                for a in n.names:
                    local = a.asname or a.name
                    if t in dmap and a.name in exports[dmap[t]] and exports[dmap[t]][a.name][0] != rel:
                        sym[local] = exports[dmap[t]][a.name]
                    elif f"{t}.{a.name}" in dmap:
                        mod[local] = dmap[f"{t}.{a.name}"]
                    elif top in tops:
                        ext[local] = "internal_unresolved"
                    else:
                        ext[local] = "stdlib" if top in STDLIB else "external"
            elif isinstance(n, ast.Import):
                for a in n.names:
                    top = a.name.split(".")[0]
                    local = a.asname or top
                    if a.asname and a.name in dmap:
                        mod[local] = dmap[a.name]
                    elif top in tops:
                        ext[local] = "internal_unresolved"
                    else:
                        ext[local] = "stdlib" if top in STDLIB else "external"
        n_int = 0
        for n in ast.walk(f["tree"]):
            if not isinstance(n, ast.Call):
                continue
            fn, cat = n.func, "other"
            if isinstance(fn, ast.Name):
                if fn.id in sym:
                    cat = "internal"; targets.add(sym[fn.id])
                elif fn.id in ext:
                    cat = ext[fn.id]
            elif isinstance(fn, ast.Attribute):
                root_n = fn
                while isinstance(root_n, ast.Attribute):
                    root_n = root_n.value
                if isinstance(root_n, ast.Name):
                    if (fn.value is root_n and root_n.id in mod
                            and fn.attr in exports[mod[root_n.id]]
                            and mod[root_n.id] != rel):
                        cat = "internal"; targets.add(exports[mod[root_n.id]][fn.attr])
                    elif root_n.id in ext:
                        cat = ext[root_n.id]
                    elif root_n.id in mod:
                        cat = "internal_unresolved"
            stats[cat] += 1
            n_int += cat == "internal"
        per_file_internal.append(n_int)

    total = sum(stats.values()) or 1
    return {
        "py_files_nontest": files_n, "loc_nontest": loc, "parse_fails": fails,
        "calls_total": total if sum(stats.values()) else 0, **{f"calls_{k}": v for k, v in stats.items()},
        "pct_internal": round(100 * stats["internal"] / total, 1),
        "pct_external": round(100 * stats["external"] / total, 1),
        "files_with_internal": sum(1 for x in per_file_internal if x),
        "median_internal_per_file": statistics.median(per_file_internal) if per_file_internal else 0,
        "distinct_internal_targets": len(targets),
    }


def git_facts(root):
    out = {"head_date": sh(["git", "log", "-1", "--format=%cs"], cwd=root).stdout.strip()}
    for d in SINCE:
        out[f"commits_since_{d}"] = int(sh(["git", "rev-list", "--count", f"--since={d}", "HEAD"],
                                           cwd=root).stdout.strip() or 0)
    out["ai_files"] = [a for a in AI_FILES if (root / a).exists()]
    return out


def main():
    jobs = [(dom, slug) for dom, slugs in REPOS.items() for slug in slugs]
    with ThreadPoolExecutor(max_workers=6) as ex:
        cloned = list(ex.map(lambda j: (j, clone(j[1])), jobs))
    results = []
    for (dom, slug), (_, dest, err) in cloned:
        if dest is None:
            results.append({"domain": dom, "repo": slug, "error": err})
            print("FAILED", slug, err, flush=True)
            continue
        row = {"domain": dom, "repo": slug, "licence": licence(dest),
               **git_facts(dest), **analyse(dest)}
        results.append(row)
        print(f"{dom:15s} {slug:36s} {row['licence']:12s} files={row['py_files_nontest']:4d} "
              f"loc={row['loc_nontest']:6d} internal={row['pct_internal']:5.1f}% "
              f"external={row['pct_external']:5.1f}%", flush=True)
    OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
