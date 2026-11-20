"""Teacher-forced surprisal of a completion given its context.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

The context and completion are tokenised together, as the model would see them
in use, and character offsets decide which tokens belong to the completion.

Boundary rule: a token belongs to the completion if it ends after the boundary
(the first character of the completion). A token that starts before the
boundary and ends after it "straddles": it is kept, because the model has to
predict it to produce the completion, and it is flagged so straddles can be
counted. Zero-width special tokens such as <bos> never belong to the completion.

Surprisal of a token is -log2 p(token | everything before it). The first token
of the input has no prediction, so a completion that starts there raises an
error rather than returning a wrong number. This happens with Qwen, which adds
no start token, when the context is empty.
"""

import math
from dataclasses import asdict, dataclass

import torch

LN2 = math.log(2)


@dataclass(frozen=True)
class TokenScore:
    position: int    # index in the full input
    token_id: int
    text: str        # source characters the token covers
    start: int       # character offsets relative to the start of the completion;
    end: int         # start is negative for a straddling token
    bits: float
    straddles: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SpanScore:
    text: str
    bits: float          # summed over every token that overlaps the span
    n_tokens: int
    exact: bool          # True if those tokens cover the span and nothing else
    tokens: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def completion_positions(offsets: list[tuple[int, int]], boundary: int) -> list[int]:
    """Indices of the tokens that belong to the completion (see module docstring)."""
    positions = [i for i, (s, e) in enumerate(offsets) if e > s and e > boundary]
    if not positions:
        raise ValueError("no tokens fall after the boundary; is the completion empty?")
    expected = list(range(positions[0], positions[-1] + 1))
    if positions != expected:
        raise ValueError(f"completion tokens are not contiguous: {positions}")
    return positions


def score_completion(
    model,
    tokenizer,
    context: str,
    completion: str,
    *,
    keep_logits_only: bool = True,
) -> list[TokenScore]:
    """Per-token surprisal, in bits, of `completion` given `context`.

    With keep_logits_only, the model returns logits only for the positions
    needed (the one before the first completion token, onwards), using
    `logits_to_keep`. This keeps long prompts within GPU memory. Set it to
    False to compute full logits, for checking that both give the same answer.
    """
    text = context + completion
    boundary = len(context)
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    input_ids = enc["input_ids"]
    n = input_ids.shape[1]

    positions = completion_positions(offsets, boundary)
    if positions[0] == 0:
        raise ValueError(
            "the first completion token is the first input token, so it has no prediction "
            "(the tokeniser adds no start token and the context is empty)"
        )
    if positions[-1] != n - 1:
        raise ValueError("tokens follow the completion; the completion must end the input")

    covered = "".join(text[max(offsets[p][0], boundary):offsets[p][1]] for p in positions)
    if covered != completion:
        raise ValueError("completion tokens do not cover the completion text exactly")

    # Logits at position i predict token i + 1. We need rows positions[0] - 1 .. n - 2.
    first_row = positions[0] - 1
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        if keep_logits_only:
            keep = n - first_row
            logits = model(**enc, logits_to_keep=keep).logits[0]
            row_offset = n - logits.shape[0]
        else:
            logits = model(**enc).logits[0]
            row_offset = 0
    rows = torch.tensor([p - 1 - row_offset for p in positions], device=logits.device)
    # float64 for the softmax: float32 summation over a 150k-262k vocabulary was
    # measured off by about 3e-4 bits. Only the kept rows are converted, so it is cheap.
    logp = torch.log_softmax(logits[rows].double(), dim=-1)
    targets = input_ids[0, positions].to(logits.device)
    bits = (-logp.gather(1, targets[:, None]).squeeze(1) / LN2).cpu().tolist()

    scores = []
    for p, b in zip(positions, bits):
        s, e = offsets[p]
        scores.append(TokenScore(
            position=p,
            token_id=int(input_ids[0, p]),
            text=text[s:e],
            start=s - boundary,
            end=e - boundary,
            bits=b,
            straddles=s < boundary,
        ))
    return scores


def span_bits(scores: list[TokenScore], start: int, end: int, text: str = "") -> SpanScore:
    """Sum the bits of every token overlapping completion characters [start, end).

    `exact` is False when a token spills outside the span, for example Qwen's
    ".normal", which merges the dot into the identifier. Bits are summed over
    the whole token in that case, so the two tokenisers are not split the same
    way. The full span rule belongs to the token-labelling module.

    The sum is p(these tokens), not p(this word): it leaves out the chance
    that the word ends here (Pimentel and Meister, 2024).
    """
    hit = [t for t in scores if t.start < end and t.end > start]
    if not hit:
        raise ValueError(f"no tokens overlap span [{start}, {end})")
    exact = hit[0].start == start and hit[-1].end == end
    return SpanScore(
        text=text,
        bits=sum(t.bits for t in hit),
        n_tokens=len(hit),
        exact=exact,
        tokens=tuple(t.text for t in hit),
    )
