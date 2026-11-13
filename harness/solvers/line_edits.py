import re

MIN_SOURCE_BYTES = 600
_HEADER = re.compile(r"@@ ([^\s]+) ([1-9][0-9]{0,6})(?:-([1-9][0-9]{0,6}))?")
FORMAT = """Return only the smallest line replacements that repair every independent bug.
Start each block with @@ filename.py START-END (inclusive ORIGINAL line numbers, or just START for one line).
Then write the new Python lines with their actual indentation, without line numbers. To delete, leave the block
empty. To insert, replace a nearby line and include it. Use only the listed implementation files. No code fences
or explanation. Preserve the surrounding definitions. Example:
@@ prices.py 7
        total += price * quantity"""


def numbered(source: str) -> str:
    return "\n".join(f"{n}|{line}" for n, line in enumerate(source.splitlines(), 1))


def apply(reply: str, sources: dict[str, str]) -> dict[str, str] | None:
    text = reply.strip("\r\n")
    fence = re.fullmatch(r"```(?:python|diff|text)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*", text, re.S)
    if fence:
        text = fence[1]
    edits = {}
    replacement = None
    for line in text.splitlines():
        if line.startswith("@@"):
            match = _HEADER.fullmatch(line.strip())
            if not match or match[1] not in sources:
                return None
            start = int(match[2]) - 1
            end = int(match[3] or match[2])
            if not start < end <= len(sources[match[1]].splitlines()):
                return None
            replacement = []
            edits.setdefault(match[1], []).append((start, end, replacement))
        elif replacement is None:
            return None
        else:
            replacement.append(line)
    if not edits:
        return None
    patched = {}
    for name, blocks in edits.items():
        lines = sources[name].splitlines(keepends=True)
        boundary = len(lines)
        for start, end, new_lines in sorted(blocks, key=lambda b: b[0], reverse=True):
            if end > boundary:
                return None
            newline = "\r\n" if lines[start].endswith("\r\n") else "\n"
            new = newline.join(new_lines)
            if new_lines and lines[end - 1].endswith(("\r", "\n")):
                new += newline
            lines[start:end] = [new]
            boundary = start
        source = "".join(lines)
        try:
            compile(source, name, "exec")
        except (ValueError, SyntaxError):
            return None
        if source != sources[name]:
            patched[name] = source
    return patched or None
