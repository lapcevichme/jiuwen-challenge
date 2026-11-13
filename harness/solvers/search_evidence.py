import json
import re
from dataclasses import dataclass
from pathlib import Path

from harness.solvers.linked_docs import connected_indices


SYSTEM = """Answer the question using only the provided documents. Each content line has an ID in brackets.
Read across all documents, including tables and prose. Resolve nicknames and joins first. Return one JSON object.
- To look up a number or name, return {"op":"lookup","line":12,"value":"literal answer from line 12"}.
- To count, return {"op":"count","lines":[4,17,33]}: one distinct supporting line per matching record.
- To sum, return {"op":"sum","terms":[[4,"3000"],[17,"4500"]],"divisor":1}. Copy each number from its line.
  For a total requested in thousands rounded down use divisor 1000. The harness performs the arithmetic.
Use only IDs from the documents. If a necessary fact is missing, the result needs a different calculation,
or a count has no matches, return UNSOLVABLE to request a separate program-based check."""


LOOKUP_SYSTEM = """Follow the named entities through the provided documents to find the requested fact.
Reply with just {"op":"lookup","line":12,"value":"exact value from line 12"} using the final supporting line.
Check the subject and field carefully. If these documents do not establish the answer, return UNSOLVABLE."""


@dataclass
class Evidence:
    prompt: str
    lines: list[str]
    has_tables: bool = False
    focused: bool = False

    @classmethod
    def read(cls, root: Path | None, limit: int, question: str = "") -> "Evidence | None":
        if root is None or limit <= 0:
            return None
        documents = []
        size = 0
        try:
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    return None
                if path.is_dir():
                    continue
                if (not path.is_file() or not path.resolve().is_relative_to(root.resolve())
                        or path.suffix.lower() not in {".md", ".markdown", ".txt"}):
                    return None
                size += path.stat().st_size
                if size > min(limit * (4 if question else 1), 400_000) or len(documents) >= 60:
                    return None
                content = path.read_text(encoding="utf-8-sig").splitlines()
                documents.append((str(path.relative_to(root)), [line.strip() for line in content if line.strip()]))
        except (OSError, UnicodeError):
            return None
        indices = connected_indices(question, documents) if question else None
        focused = indices is not None
        if focused:
            documents = [documents[i] for i in indices]
        elif size > limit:
            return None
        sections, lines = [], []
        for name, content in documents:
            section = [f"Document {name}"]
            for line in content:
                lines.append(line)
                section.append(f"[{len(lines)}] {line}")
            sections.append("\n".join(section))
        prompt = "\n\n".join(sections)
        if not lines or len(prompt) > 2 * limit:
            return None
        has_tables = any(re.fullmatch(r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?", line) for line in lines)
        return cls(prompt, lines, has_tables, focused)

    def line(self, index) -> str:
        if type(index) is not int or not 1 <= index <= len(self.lines):
            raise ValueError("unknown evidence line")
        return self.lines[index - 1]

    def answer(self, reply: str, check: str, question: str) -> str | None:
        blocks = re.findall(r"^```(?:json)?[ \t]*\r?\n(.*?)^```[ \t]*$", reply.strip(), re.MULTILINE | re.DOTALL)
        try:
            spec = json.loads(blocks[0] if len(blocks) == 1 else reply)
            if not isinstance(spec, dict):
                return None
            if spec.get("op") == "lookup":
                value = spec["value"]
                if not isinstance(value, (str, int)) or isinstance(value, bool):
                    return None
                value = str(value).strip()
                if not value or len(value) > 200 or "\n" in value:
                    return None
                line = self.line(spec["line"])
                if check == "exact_int":
                    return str(_integer(line, value))
                match = re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", line, re.IGNORECASE)
                return match[0] if match else None
            if self.focused or check != "exact_int":
                return None
            if spec.get("op") == "count":
                ids = spec["lines"]
                if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
                    return None
                for index in ids:
                    self.line(index)
                return str(len(ids))
            if spec.get("op") == "sum":
                terms = spec["terms"]
                divisor = spec.get("divisor", 1)
                if type(divisor) is not int or divisor not in (1, 1000):
                    return None
                if divisor == 1000 and not re.search(r"\bthousands?\b", question, re.I):
                    return None
                if not isinstance(terms, list) or not terms:
                    return None
                seen = set()
                total = 0
                for index, value in terms:
                    if index in seen:
                        return None
                    total += _integer(self.line(index), str(value))
                    seen.add(index)
                if total % divisor and not re.search(r"\b(?:round(?:ed)? down|floor)\b", question, re.I):
                    return None
                return str(total // divisor)
        except (KeyError, TypeError, ValueError):
            return None
        return None


def _integer(line: str, value: str) -> int:
    if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)", value):
        raise ValueError("expected integer evidence")
    if not re.search(r"(?<![\w+\-])(?<!\d[.,])" + re.escape(value) + r"(?!\w|[.,]\d)", line):
        raise ValueError("number absent from cited line")
    return int(value.replace(",", ""))
