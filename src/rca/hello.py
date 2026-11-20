"""Hello COM3001 task loading and prompt building (no model needed).

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.
Kept separate from the script so the tests can check the tasks on the CPU.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from rca.prompt import plain_prompt

CONDITIONS = ("none", "random", "oracle")


@dataclass(frozen=True)
class HelloTask:
    id: str
    targets: tuple[str, ...]
    definition_path: str
    definition: str
    current_path: str
    prefix: str
    line: str


def load_hello(path: str | Path) -> tuple[list[HelloTask], tuple[str, str]]:
    data = tomllib.loads(Path(path).read_text())
    tasks = [
        HelloTask(
            id=t["id"],
            targets=tuple(t["targets"]),
            definition_path=t["definition_path"],
            definition=t["definition"],
            current_path=t["current_path"],
            prefix=t["prefix"],
            line=t["line"],
        )
        for t in data["tasks"]
    ]
    random = (data["random"]["path"], data["random"]["text"])
    return tasks, random


def build_context(task: HelloTask, condition: str, random: tuple[str, str]) -> str:
    if condition == "none":
        snippets = []
    elif condition == "random":
        snippets = [random]
    elif condition == "oracle":
        snippets = [(task.definition_path, task.definition)]
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return plain_prompt(snippets, task.current_path, task.prefix)


def target_span(task: HelloTask, name: str) -> tuple[int, int]:
    """Character span of the first occurrence of `name` as a whole word in the hidden line."""
    import re

    m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", task.line)
    if m is None:
        raise ValueError(f"target {name!r} not found in task {task.id!r}")
    return m.span()
