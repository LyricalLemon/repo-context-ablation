"""Environment and model load checks for the lab machines.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.
This is a setup check, not part of the measurement code.

Usage (lab machine, inside the rca environment):
    python scripts/check_models.py smoke   # GPU and library check
    python scripts/check_models.py gemma   # load test for Gemma 4 E4B
    python scripts/check_models.py qwen    # load test for Qwen2.5-Coder-7B

Each model check loads the pinned revision from the local cache only
(no network), in bfloat16, and records GPU memory, whether a text-only
prompt returns logits, the top next-token guesses and a short greedy
continuation. Results are printed and saved as JSON under $FYP/logs.
"""

import json
import os
import socket
import sys
import time

# Use only the cached, pinned snapshot. Must be set before importing transformers.
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODELS = {
    "gemma": ("google/gemma-4-E4B", "11b8250ef59eb3ebc5dae7b21144d7b84f2c06dd"),
    "qwen": ("Qwen/Qwen2.5-Coder-7B", "0396a76181e127dfc13e5c5ec48a8cee09938b02"),
}

PROMPT = (
    "def celsius_to_kelvin(celsius: float) -> float:\n"
    '    """Convert a temperature from Celsius to Kelvin."""\n'
    "    return "
)

GIB = 1024**3


def smoke() -> dict:
    print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU ok:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0), "| transformers", transformers.__version__)
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    ok = (x @ x).float().abs().sum().item() > 0
    print("matmul ok:", ok)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free / GIB:.2f} GiB free of {total / GIB:.2f} GiB")
    return {"matmul_ok": ok, "gpu_free_gib": free / GIB, "gpu_total_gib": total / GIB}


def load_test(key: str) -> dict:
    name, revision = MODELS[key]
    print(f"\n== {name} @ {revision[:8]}")

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    load_s = time.time() - t0
    mem_loaded = torch.cuda.memory_allocated() / GIB
    print(f"class: {type(model).__name__} | dtype: {model.dtype} | load time {load_s:.0f} s")
    print(f"GPU memory after load: {mem_loaded:.2f} GiB allocated")

    enc = tok(PROMPT, return_tensors="pt").to("cuda")
    ids = enc["input_ids"][0].tolist()
    first = tok.convert_ids_to_tokens(ids[:3])
    print(f"prompt tokens: {len(ids)} | first three: {first} | BOS id: {tok.bos_token_id}")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        logits = model(**enc).logits
    print(f"logits shape: {tuple(logits.shape)} | dtype: {logits.dtype} | finite: {torch.isfinite(logits).all().item()}")

    logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(logp, 5)
    top5 = [(tok.decode([i]), round(-lp / torch.log(torch.tensor(2.0)).item(), 2))
            for i, lp in zip(top.indices.tolist(), top.values.tolist())]
    print("top-5 next tokens (token, surprisal in bits):", top5)

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=20, do_sample=False)
    continuation = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print("greedy continuation:", repr(continuation))

    peak = torch.cuda.max_memory_allocated() / GIB
    free, total = torch.cuda.mem_get_info()
    print(f"peak GPU memory during forward and generate: {peak:.2f} GiB | free now: {free / GIB:.2f} of {total / GIB:.2f} GiB")

    return {
        "model": name,
        "revision": revision,
        "class": type(model).__name__,
        "dtype": str(model.dtype),
        "load_seconds": round(load_s, 1),
        "gpu_gib_after_load": round(mem_loaded, 2),
        "gpu_gib_peak": round(peak, 2),
        "gpu_gib_total": round(total / GIB, 2),
        "prompt_tokens": len(ids),
        "first_tokens": first,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "top5_next": top5,
        "greedy_continuation": continuation,
    }


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("smoke", *MODELS):
        sys.exit(__doc__)
    what = sys.argv[1]
    result = smoke() if what == "smoke" else load_test(what)
    result.update({
        "check": what,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    log_dir = os.path.join(os.environ.get("FYP", "."), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"check_{what}_{socket.gethostname().split('.')[0]}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
