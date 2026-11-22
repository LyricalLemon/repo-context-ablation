"""Tests for the code tokeniser and BM25. AI-written (Claude Code, 22 Nov 2026), disclosed.

The scores in test_scores_match_hand_computation were worked out by hand from
the formula in rca.bm25 (k1 = 1.2, b = 0.75), not by running the code.
The reference test compares with the independent bm25s library (dev only).
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import pytest

from rca.bm25 import BM25Index, code_tokens, identifier_parts

# --- tokeniser -------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("FluxBinner", ["fluxbinner", "flux", "binner"]),
    ("normalise_reading_v2", ["normalise_reading_v2", "normalise", "reading", "v2"]),
    ("HTTPServer2", ["httpserver2", "http", "server2"]),
    ("getHTTP", ["gethttp", "get", "http"]),
    ("MAX_PROBE_TEMPERATURE_K", ["max_probe_temperature_k", "max", "probe", "temperature", "k"]),
    ("_private", ["_private", "private"]),
    ("__init__", ["__init__", "init"]),
    ("x", ["x"]),
    ("self", ["self"]),
    ("273.15", ["273", "15"]),
])
def test_code_tokens_single_words(text, expected):
    assert code_tokens(text) == expected


def test_code_tokens_splits_on_punctuation_and_keeps_order():
    assert code_tokens("value = calibrate.normalise(raw, gain=2)") == [
        "value", "calibrate", "normalise", "raw", "gain", "2",
    ]


def test_code_tokens_comments_and_strings_count_as_words():
    assert code_tokens('# Scale the "raw" reading') == ["scale", "the", "raw", "reading"]


def test_identifier_parts_non_ascii_piece_is_kept_whole():
    assert identifier_parts("café_total") == ["café", "total"]


# --- BM25 scoring ----------------------------------------------------------

DOCS = [["a", "b", "a"], ["b", "c"], ["c", "c", "c", "d"]]  # lengths 3, 2, 4; avgdl 3


def test_idf_is_lucene_form():
    idx = BM25Index(DOCS)
    assert idx.idf("a") == pytest.approx(math.log(1 + 2.5 / 1.5))  # df 1
    assert idx.idf("c") == pytest.approx(math.log(1.6))           # df 2
    assert idx.idf("zzz") == pytest.approx(math.log(1 + 3.5 / 0.5))  # unseen
    # Positive even for a term in every document, unlike Robertson-Sparck Jones IDF.
    assert BM25Index([["t"], ["t"]]).idf("t") > 0


def test_scores_match_hand_computation():
    # Query "a c". idf(a) = ln(8/3) = 0.980829, idf(c) = ln(1.6) = 0.470004.
    # d0: a, tf 2, |d| 3: K = 1.2*(0.25 + 0.75*3/3) = 1.2;  2*2.2/(2+1.2) = 1.375     -> 1.348640
    # d1: c, tf 1, |d| 2: K = 1.2*(0.25 + 0.75*2/3) = 0.9;  1*2.2/(1+0.9) = 1.157895  -> 0.544215
    # d2: c, tf 3, |d| 4: K = 1.2*(0.25 + 0.75*4/3) = 1.5;  3*2.2/(3+1.5) = 1.466667  -> 0.689339
    idx = BM25Index(DOCS, k1=1.2, b=0.75)
    s = idx.scores(["a", "c"])
    assert s == pytest.approx([1.348640, 0.544215, 0.689339], abs=1e-6)
    assert idx.top_k(["a", "c"]) == [(0, pytest.approx(1.348640, abs=1e-6)),
                                      (2, pytest.approx(0.689339, abs=1e-6)),
                                      (1, pytest.approx(0.544215, abs=1e-6))]


def test_query_repeats_add_unless_turned_off():
    with_repeats = BM25Index(DOCS).scores(["a", "a"])
    once = BM25Index(DOCS, count_query_repeats=False).scores(["a", "a"])
    assert with_repeats[0] == pytest.approx(2 * 1.348640, abs=1e-6)
    assert once[0] == pytest.approx(1.348640, abs=1e-6)


def test_b_zero_turns_off_length_normalisation():
    idx = BM25Index([["a"], ["a", "x", "x", "x"]], b=0.0)
    s = idx.scores(["a"])
    assert s[0] == pytest.approx(s[1])


def test_ties_keep_document_order_and_non_matches_are_left_out():
    idx = BM25Index([["q"], ["z"], ["q"], ["q"]])
    assert [d for d, _ in idx.top_k(["q"])] == [0, 2, 3]
    assert [d for d, _ in idx.top_k(["q"], k=2)] == [0, 2]
    assert idx.top_k(["nothing"]) == []


def test_empty_index_and_empty_documents():
    assert BM25Index([]).top_k(["a"]) == []
    idx = BM25Index([[], ["a"]])
    assert [d for d, _ in idx.top_k(["a"])] == [1]
    assert BM25Index([[], []]).top_k(["a"]) == []


# --- reference check against bm25s ---------------------------------------
# bm25s's "lucene" method uses the same IDF and adds repeated query terms, but
# its term weight is tf / (tf + K), without the (k1 + 1) factor kept here.
# So our scores should equal (k1 + 1) times its scores, document by document.

bm25s = pytest.importorskip("bm25s", reason="bm25s is a dev-only reference (requirements-dev.txt)")


def _bm25s_scores(docs, query, k1, b):
    ref = bm25s.BM25(k1=k1, b=b, method="lucene", dtype="float64")
    ref.index(docs, show_progress=False)
    vocab = ref.vocab_dict
    q = [t for t in query if t in vocab]  # bm25s cannot score unseen terms; they add nothing anyway
    if not q:
        return np.zeros(len(docs))
    return np.asarray(ref.get_scores(q), dtype=np.float64)


def _compare(docs, queries, k1=1.2, b=0.75):
    ours = BM25Index(docs, k1=k1, b=b)
    for q in queries:
        expected = (k1 + 1) * _bm25s_scores(docs, q, k1, b)
        np.testing.assert_allclose(ours.scores(q), expected, rtol=1e-9, atol=1e-9)


def test_matches_bm25s_on_small_corpus():
    _compare(DOCS, [["a", "c"], ["a", "a", "b"], ["d"], ["c", "zzz"]])
    _compare(DOCS, [["a", "c"]], k1=2.0, b=0.3)


CCEVAL_REPOS = Path(os.environ.get("FYP", "/vol/research/Noah_Dias_FYP")) / "data" / "cceval" / "repos"


@pytest.mark.skipif(not CCEVAL_REPOS.is_dir(), reason="fetched CrossCodeEval repositories not available")
def test_matches_bm25s_on_real_repository_chunks():
    from rca.chunking import fixed_windows

    repos = sorted(p for p in CCEVAL_REPOS.iterdir() if p.is_dir())
    rng = random.Random(0)
    for repo in rng.sample(repos, 3):
        files = {}
        for f in sorted(repo.rglob("*.py")):
            try:
                files[str(f.relative_to(repo))] = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
        chunks = fixed_windows(files)
        if len(chunks) < 5:
            continue
        docs = [code_tokens(c.text) for c in chunks]
        queries = [code_tokens(chunks[i].text) for i in rng.sample(range(len(chunks)), 5)]
        _compare(docs, queries)
