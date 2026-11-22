"""Fixed-window chunking of repository files.

AI-written (Claude Code, 22 Nov 2026), disclosed in the AI-use log.

A file is cut into windows of `window` lines that start every `stride` lines
(0, stride, 2*stride, ...). With window 20 and stride 10, each line appears in
two windows, so a definition near a window edge is still seen whole in one of
them. The last window of a file may be shorter, and a file shorter than one
window gives a single chunk. Windows holding only whitespace are dropped.

Lines are counted on "\\n" only, and each chunk's text keeps its newlines, so
joining a file's non-overlapping windows gives the file back.

Which files go in is the caller's decision: the current file is removed
before chunking, because its text after the cursor holds the answer
(decision of 22 Nov 2026).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    file: str        # path relative to the repository root
    start_line: int  # 0-based, inclusive
    end_line: int    # 0-based, exclusive
    text: str

    @property
    def id(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line}"

    def contains_line(self, line: int) -> bool:
        return self.start_line <= line < self.end_line


def split_lines(text: str) -> list[str]:
    """Split on "\\n", keeping each line's newline (unlike str.splitlines, which
    also splits on form feeds and other separators)."""
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def fixed_windows(files: dict[str, str], window: int = 20, stride: int = 10) -> list[Chunk]:
    """Cut every file into windows of `window` lines, starting every `stride` lines.

    Files are visited in sorted path order, so chunk order (and BM25 tie
    order) does not depend on how the dictionary was built.
    """
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be at least 1")
    if stride > window:
        raise ValueError("stride larger than window would skip lines")
    chunks = []
    for path in sorted(files):
        lines = split_lines(files[path])
        n = len(lines)
        start = 0
        while True:
            end = min(start + window, n)
            text = "".join(lines[start:end])
            if text.strip():
                chunks.append(Chunk(path, start, end, text))
            if end >= n:
                break
            start += stride
    return chunks
