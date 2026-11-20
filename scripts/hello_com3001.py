"""Hello COM3001: surprisal of a hidden cross-file line with and without its definition.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

For each made-up task in configs/hello.toml and each condition (no context,
random unrelated file, oracle definition file), it records the surprisal of
every token of the true line, the summed bits of each cross-file name, and
whether greedy generation reproduces the line.

Usage (lab machine with a 20 GB GPU, inside the rca environment):
    python scripts/hello_com3001.py --model gemma
    python scripts/hello_com3001.py --model qwen
    python scripts/hello_com3001.py --model gemma --check-logits-to-keep

Output: $FYP/runs/hello/<time>_<host>_<model>/ with meta.json, tokens.jsonl
and summary.json.
"""

import argparse
import time
from pathlib import Path

import torch

from rca.generation import exact_match, greedy_line
from rca.hello import CONDITIONS, build_context, load_hello, target_span
from rca.models import get_spec, load_model, load_tokenizer
from rca.runlog import REPO_ROOT, JsonlWriter, run_dir, run_metadata, write_json
from rca.surprisal import score_completion, span_bits

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["gemma", "qwen"])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "hello.toml"))
    ap.add_argument("--check-logits-to-keep", action="store_true",
                    help="also score with full logits and report the largest difference in bits")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    spec = get_spec(args.model)
    tasks, random = load_hello(args.config)
    out = run_dir("hello", args.model)
    config = {"config_file": args.config, "conditions": list(CONDITIONS),
              "check_logits_to_keep": args.check_logits_to_keep, "max_new_tokens": 48,
              "prompt_format": "plain"}
    meta = run_metadata(config, spec, seed=SEED)
    write_json(out / "meta.json", meta)
    if meta["git"]["dirty"]:
        print("warning: uncommitted changes in the repository; meta.json records dirty=true")

    tok = load_tokenizer(args.model)
    t0 = time.time()
    model = load_model(args.model)
    print(f"{spec.name} loaded in {time.time() - t0:.0f} s -> {out}")

    summary = []
    max_diff = 0.0
    with JsonlWriter(out / "tokens.jsonl") as tokens_out:
        for task in tasks:
            for cond in CONDITIONS:
                context = build_context(task, cond, random)
                scores = score_completion(model, tok, context, task.line)
                if args.check_logits_to_keep:
                    full = score_completion(model, tok, context, task.line, keep_logits_only=False)
                    diff = max(abs(a.bits - b.bits) for a, b in zip(scores, full))
                    max_diff = max(max_diff, diff)
                generated = greedy_line(model, tok, context)
                prompt_tokens = len(tok(context)["input_ids"])

                for t in scores:
                    tokens_out.write({"task": task.id, "condition": cond, **t.to_dict()})
                spans = {name: span_bits(scores, *target_span(task, name), text=name).to_dict()
                         for name in task.targets}
                summary.append({
                    "task": task.id,
                    "condition": cond,
                    "prompt_tokens": prompt_tokens,
                    "line_tokens": len(scores),
                    "line_bits": sum(t.bits for t in scores),
                    "straddling_tokens": sum(t.straddles for t in scores),
                    "targets": spans,
                    "generated": generated,
                    "exact_match": exact_match(generated, task.line),
                })

    result = {"summary": summary}
    if args.check_logits_to_keep:
        result["logits_to_keep_max_abs_diff_bits"] = max_diff
    result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    write_json(out / "summary.json", result)
    report(tasks, summary, out, result)


def report(tasks, summary, out: Path, result: dict) -> None:
    by_key = {(r["task"], r["condition"]): r for r in summary}
    print("\nBits per cross-file name (summed over overlapping tokens; * = tokens spill outside the name)")
    print(f"{'task':<10} {'name':<26}" + "".join(f"{c:>10}" for c in CONDITIONS))
    for task in tasks:
        for name in task.targets:
            cells = []
            for c in CONDITIONS:
                s = by_key[(task.id, c)]["targets"][name]
                cells.append(f"{s['bits']:>9.2f}{'' if s['exact'] else '*'}")
            print(f"{task.id:<10} {name:<26}" + "".join(cells))

    print("\nWhole line: bits / exact match / prompt tokens")
    for task in tasks:
        row = "  ".join(
            f"{c}: {by_key[(task.id, c)]['line_bits']:6.2f} "
            f"{'EM' if by_key[(task.id, c)]['exact_match'] else '--'} "
            f"{by_key[(task.id, c)]['prompt_tokens']:>4}t"
            for c in CONDITIONS
        )
        print(f"{task.id:<10} {row}")

    print("\nGreedy lines")
    for r in summary:
        print(f"  {r['task']:<10} {r['condition']:<7} {r['generated']!r}")

    straddles = sum(r["straddling_tokens"] for r in summary)
    print(f"\nstraddling tokens: {straddles} | peak GPU memory: {result['peak_gpu_gib']:.2f} GiB")
    if "logits_to_keep_max_abs_diff_bits" in result:
        print(f"logits_to_keep vs full logits, max difference: {result['logits_to_keep_max_abs_diff_bits']:.4f} bits")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
