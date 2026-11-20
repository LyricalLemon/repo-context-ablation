"""Pinned model registry and loaders.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

Every model is loaded from the local Hugging Face cache only, at a pinned
revision, so a run can never silently pick up a newer checkpoint or tokeniser.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    revision: str
    dtype: torch.dtype = torch.bfloat16


MODELS: dict[str, ModelSpec] = {
    "gemma": ModelSpec("gemma", "google/gemma-4-E4B", "11b8250ef59eb3ebc5dae7b21144d7b84f2c06dd"),
    "qwen": ModelSpec("qwen", "Qwen/Qwen2.5-Coder-7B", "0396a76181e127dfc13e5c5ec48a8cee09938b02"),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"unknown model {key!r}; choose from {sorted(MODELS)}") from None


def load_tokenizer(key: str):
    from transformers import AutoTokenizer

    spec = get_spec(key)
    return AutoTokenizer.from_pretrained(spec.name, revision=spec.revision, local_files_only=True)


def load_model(key: str, device: str = "cuda"):
    """Load the full model in its pinned precision, in eval mode.

    Gemma 4 resolves to Gemma4ForConditionalGeneration through
    AutoModelForCausalLM (build brief, section 4, step 2).
    """
    from transformers import AutoModelForCausalLM

    spec = get_spec(key)
    model = AutoModelForCausalLM.from_pretrained(
        spec.name,
        revision=spec.revision,
        dtype=spec.dtype,
        device_map=device,
        local_files_only=True,
    )
    return model.eval()
