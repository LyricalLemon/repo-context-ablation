"""Tests for fixed-window chunking. AI-written (Claude Code, 22 Nov 2026), disclosed."""

import pytest

from rca.chunking import fixed_windows, split_lines


def numbered(n: int) -> str:
    return "".join(f"line{i}\n" for i in range(n))


def spans(chunks):
    return [(c.start_line, c.end_line) for c in chunks]


def test_split_lines_keeps_newlines_and_unterminated_last_line():
    assert split_lines("a\nb\n") == ["a\n", "b\n"]
    assert split_lines("a\nb") == ["a\n", "b"]
    assert split_lines("") == []
    # A form feed is not a line break here (str.splitlines would split on it).
    assert split_lines("a\x0cb\nc\n") == ["a\x0cb\n", "c\n"]


def test_windows_start_every_stride_and_last_one_is_shorter():
    chunks = fixed_windows({"m.py": numbered(45)}, window=20, stride=10)
    assert spans(chunks) == [(0, 20), (10, 30), (20, 40), (30, 45)]
    assert chunks[0].text == numbered(20)
    assert chunks[-1].text.startswith("line30\n") and chunks[-1].text.endswith("line44\n")


def test_window_ending_exactly_at_the_end_of_the_file_is_the_last():
    assert spans(fixed_windows({"m.py": numbered(30)}, 20, 10)) == [(0, 20), (10, 30)]
    assert spans(fixed_windows({"m.py": numbered(20)}, 20, 10)) == [(0, 20)]


def test_short_file_gives_one_chunk():
    chunks = fixed_windows({"m.py": numbered(3)}, 20, 10)
    assert spans(chunks) == [(0, 3)]
    assert chunks[0].text == numbered(3)


def test_whitespace_only_windows_and_empty_files_are_dropped():
    text = numbered(10) + "\n" * 25 + numbered(5)
    chunks = fixed_windows({"m.py": text, "empty.py": ""}, 10, 10)
    assert all(c.text.strip() for c in chunks)
    assert {c.file for c in chunks} == {"m.py"}
    assert spans(chunks) == [(0, 10), (30, 40)]


def test_files_in_sorted_order_and_ids():
    chunks = fixed_windows({"b.py": "x\n", "a.py": "y\n"})
    assert [c.file for c in chunks] == ["a.py", "b.py"]
    assert chunks[0].id == "a.py:0-1"


def test_every_line_is_covered():
    text = numbered(57)
    chunks = fixed_windows({"m.py": text}, 20, 10)
    covered = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line))
    assert covered == set(range(57))
    assert chunks[0].contains_line(0) and not chunks[0].contains_line(20)


def test_bad_settings_raise():
    with pytest.raises(ValueError):
        fixed_windows({"m.py": "x\n"}, window=10, stride=20)
    with pytest.raises(ValueError):
        fixed_windows({"m.py": "x\n"}, window=0, stride=1)
