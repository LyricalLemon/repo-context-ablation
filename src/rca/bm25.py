"""Code tokeniser and Okapi BM25, written from scratch (no retrieval library).

AI-written (Claude Code, 22 Nov 2026), disclosed in the AI-use log.

Tokeniser (decision 3, 22 Nov 2026)
-----------------------------------
Text is split into words: identifiers (a letter or underscore followed by
letters, digits or underscores) and runs of digits. Everything else is a
separator. Each word is lowercased. An identifier made of more than one part
also adds its parts, split at underscores and at camelCase boundaries, so a
query that mentions `FluxBinner` also matches code that says `flux_binner`
or "binner". Digits stay attached to the letters before them:

    FluxBinner            -> fluxbinner, flux, binner
    normalise_reading_v2  -> normalise_reading_v2, normalise, reading, v2
    HTTPServer2           -> httpserver2, http, server2
    _private              -> _private, private
    273.15                -> 273, 15

There is no stopword list: BM25's IDF already gives very common words (self,
def, return) little weight.

Scoring (decision 4, 22 Nov 2026)
---------------------------------
Okapi BM25 with Lucene's IDF (Robertson and Zaragoza, 2009):

    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = sum over query terms t of
                  qtf(t) * idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

N is the number of documents, df(t) the number holding t, tf(t,d) how often t
occurs in d, |d| the length of d in tokens and avgdl the mean length. Lucene's
IDF stays positive even for a term found in every document; the classic
Robertson-Sparck Jones IDF goes negative there. The (k1 + 1) factor is kept as
in Robertson and Zaragoza; Lucene 8+ drops it, which rescales every score by
the same constant and leaves rankings unchanged.

qtf(t) is how often t occurs in the query when `count_query_repeats` is True
(the default), which is what Lucene does when the same term clause is added
twice. Set it to False to count each distinct query term once.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

_WORD = re.compile(r"[^\W\d]\w*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+\d*|[A-Z]+\d*|\d+")


def identifier_parts(word: str) -> list[str]:
    """Split an identifier at underscores and camelCase boundaries (case kept).

    The camelCase pattern is ASCII-only, so a piece with non-ASCII letters
    (Python allows them in identifiers) is kept whole rather than cut short.
    """
    parts = []
    for piece in word.split("_"):
        if not piece:
            continue
        if piece.isascii():
            parts.extend(_CAMEL.findall(piece) or [piece])
        else:
            parts.append(piece)
    return parts


def code_tokens(text: str) -> list[str]:
    """Tokenise code for BM25: each word lowercased, plus its parts (see module docstring)."""
    out = []
    for word in _WORD.findall(text):
        whole = word.lower()
        out.append(whole)
        if word[0].isdigit():
            continue
        parts = [p.lower() for p in identifier_parts(word)]
        if parts != [whole]:
            out.extend(parts)
    return out


class BM25Index:
    """An inverted index over pre-tokenised documents, scored with Okapi BM25."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75,
                 count_query_repeats: bool = True):
        self.k1 = k1
        self.b = b
        self.count_query_repeats = count_query_repeats
        self.n_docs = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs_tokens):
            for term, tf in Counter(doc).items():
                postings[term].append((i, tf))
        self._postings = {
            t: (np.array([d for d, _ in p], dtype=np.int64), np.array([f for _, f in p], dtype=np.float64))
            for t, p in postings.items()
        }

    def df(self, term: str) -> int:
        p = self._postings.get(term)
        return 0 if p is None else len(p[0])

    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every document for the query, as an array of length n_docs."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0:
            return out
        q = Counter(query_tokens)
        # Length normalisation; a corpus whose documents are all empty has avgdl 0,
        # and then no document can match, so the value is never used.
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term, qtf in q.items():
            p = self._postings.get(term)
            if p is None:
                continue
            docs, tf = p
            weight = (qtf if self.count_query_repeats else 1) * self.idf(term)
            out[docs] += weight * tf * (self.k1 + 1.0) / (tf + norm[docs])
        return out

    def top_k(self, query_tokens: list[str], k: int | None = None) -> list[tuple[int, float]]:
        """Documents sharing at least one term with the query, best first.

        Ties are broken by document index (a stable sort), so the ranking is
        deterministic. k=None returns every matching document.
        """
        s = self.scores(query_tokens)
        order = np.argsort(-s, kind="stable")
        ranked = [(int(i), float(s[i])) for i in order if s[i] > 0]
        return ranked if k is None else ranked[:k]
