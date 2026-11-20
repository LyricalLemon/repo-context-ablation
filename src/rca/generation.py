"""Greedy generation of the next line, and exact match.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.
"""

import torch


def greedy_line(model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    """Greedily continue `prompt` and return the text up to the first newline.

    Generation always runs for max_new_tokens and is then cut, which is
    simpler than stopping criteria and cheap at this length.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
        )
    text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("\n", 1)[0]


def exact_match(generated: str, reference: str) -> bool:
    """Lines match if they are equal after stripping surrounding whitespace."""
    return generated.strip() == reference.strip()
