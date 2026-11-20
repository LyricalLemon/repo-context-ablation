"""Tests for surprisal scoring, span sums, prompts and the Hello COM3001 tasks.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

CPU only. A fake model with known logits checks the indexing; the real pinned
tokenisers (loaded offline from the cache) check the boundary rule.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from rca.generation import exact_match
from rca.hello import CONDITIONS, build_context, load_hello, target_span
from rca.prompt import plain_prompt
from rca.runlog import REPO_ROOT
from rca.surprisal import TokenScore, completion_positions, score_completion, span_bits

HELLO = REPO_ROOT / "configs" / "hello.toml"


class OneBitModel:
    """Puts exactly half the probability on the true next token, so every
    correctly indexed token scores 1 bit. An off-by-one shift gives a
    different value (the target then gets 1 / (2 (V - 1)))."""

    device = torch.device("cpu")

    def __init__(self, vocab_size: int):
        self.v = vocab_size
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, logits_to_keep=0, **_):
        n = input_ids.shape[1]
        logits = torch.zeros(1, n, self.v)
        a = math.log(self.v - 1)  # e^a / (e^a + V - 1) = 1/2
        for i in range(n - 1):
            logits[0, i, input_ids[0, i + 1]] = a
        self.calls.append(logits_to_keep)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


# --- pure functions, no tokeniser -------------------------------------------

def test_completion_positions_skips_special_and_context_tokens():
    # <bos> (0,0), "ab" (0,2), "c" (2,3) | boundary 3 | "de" (3,5)
    assert completion_positions([(0, 0), (0, 2), (2, 3), (3, 5)], 3) == [3]


def test_completion_positions_keeps_straddling_token():
    assert completion_positions([(0, 2), (2, 5), (5, 6)], 3) == [1, 2]


def test_span_bits_overlap_and_exact():
    toks = [
        TokenScore(0, 1, " x", 0, 2, 1.0, False),
        TokenScore(1, 2, ".foo", 2, 6, 2.0, False),
        TokenScore(2, 3, "_bar", 6, 10, 3.0, False),
        TokenScore(3, 4, "(", 10, 11, 0.5, False),
    ]
    s = span_bits(toks, 3, 10, "foo_bar")    # ".foo" spills over the dot
    assert s.bits == 5.0 and s.n_tokens == 2 and not s.exact
    s = span_bits(toks, 6, 10, "_bar")
    assert s.bits == 3.0 and s.exact


def test_plain_prompt_format():
    p = plain_prompt([("a.py", "x = 1")], "b.py", "import a\n")
    assert p == "# file: a.py\nx = 1\n\n# file: b.py\nimport a\n"
    assert plain_prompt([], "b.py", "y\n") == "# file: b.py\ny\n"


def test_exact_match_ignores_surrounding_whitespace():
    assert exact_match("    x = f(1)", "    x = f(1)\n")
    assert not exact_match("x = f(2)", "x = f(1)")


def test_hello_tasks_are_well_formed():
    tasks, random = load_hello(HELLO)
    assert len(tasks) == 3
    for task in tasks:
        assert task.prefix.endswith("\n") and task.line.endswith("\n")
        assert task.line.count("\n") == 1
        for name in task.targets:
            s, e = target_span(task, name)
            assert task.line[s:e] == name
            assert name in task.definition          # the oracle really defines it
            assert name not in task.prefix          # and the current file does not show it
            assert name not in random[1]
        for cond in CONDITIONS:
            ctx = build_context(task, cond, random)
            assert ctx.endswith(task.prefix)


# --- real tokenisers ---------------------------------------------------------

@pytest.fixture(scope="module", params=["gemma", "qwen"])
def tok(request):
    from rca.models import load_tokenizer

    try:
        t = load_tokenizer(request.param)
    except Exception as exc:  # not on a lab machine, or cache missing
        pytest.skip(f"tokeniser {request.param} unavailable: {exc}")
    t.key = request.param
    return t


@pytest.mark.tokenizers
@pytest.mark.parametrize("keep_logits_only", [True, False])
def test_every_hello_token_scores_one_bit(tok, keep_logits_only):
    """Checks the shift between logits and targets, and the logits_to_keep arithmetic."""
    tasks, random = load_hello(HELLO)
    model = OneBitModel(len(tok))
    for task in tasks:
        for cond in CONDITIONS:
            scores = score_completion(model, tok, build_context(task, cond, random), task.line,
                                      keep_logits_only=keep_logits_only)
            assert all(abs(t.bits - 1.0) < 1e-6 for t in scores)
            assert "".join(t.text for t in scores) == task.line
            assert not any(t.straddles for t in scores)
    if keep_logits_only:
        assert all(k > 0 for k in model.calls)


@pytest.mark.tokenizers
def test_straddling_token_is_kept_and_flagged(tok):
    scores = score_completion(OneBitModel(len(tok)), tok, "x = calib", "rate\n")
    assert scores[0].straddles and scores[0].start < 0
    assert "".join(t.text for t in scores).endswith("rate\n")
    assert all(abs(t.bits - 1.0) < 1e-6 for t in scores)


@pytest.mark.tokenizers
def test_empty_context_without_start_token_raises(tok):
    model = OneBitModel(len(tok))
    if tok.key == "qwen":
        with pytest.raises(ValueError, match="no prediction"):
            score_completion(model, tok, "", "x = 1\n")
    else:  # Gemma adds <bos>, so the first token does get a prediction
        assert len(score_completion(model, tok, "", "x = 1\n")) > 0


@pytest.mark.tokenizers
def test_target_spans_map_to_tokens(tok):
    tasks, random = load_hello(HELLO)
    model = OneBitModel(len(tok))
    for task in tasks:
        scores = score_completion(model, tok, build_context(task, "oracle", random), task.line)
        for name in task.targets:
            s = span_bits(scores, *target_span(task, name), text=name)
            assert s.n_tokens >= 1
            assert name in "".join(s.tokens)
