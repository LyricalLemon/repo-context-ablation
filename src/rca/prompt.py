"""Plain prompt format shared by both models.

AI-written (Claude Code, 20 Nov 2026), disclosed in the AI-use log.

Each file is introduced by a "# file: <path>" comment line. Retrieved files
come first, then the current file up to the cursor. Qwen's native
<|repo_name|> / <|file_sep|> format is a pilot comparison, not used here.
"""


def plain_prompt(snippets: list[tuple[str, str]], current_path: str, current_prefix: str) -> str:
    parts = [f"# file: {path}\n{_with_newline(text)}" for path, text in snippets]
    parts.append(f"# file: {current_path}\n{current_prefix}")
    return "\n".join(parts)


def _with_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
