import re
from pathlib import Path

from harness.contracts import SolveResult, TaskSpec
from harness.solvers._instruction import question_text
from harness.task_io import resolve_inside

_INTRO = re.compile(
    r"The directory `?(?P<kb>[\w./-]+)`? contains a (?:small )?knowledge base of "
    r"markdown files\.\s+(?P<question>.+)", re.IGNORECASE | re.DOTALL,
)
_QUERIES = [
    (r"In which year was (.+) born", ("born",)),
    (r"Who is (.+)['’]s mentor", ("mentor",)),
    (r"(?:Who|What) is (.+)['’]s employer", ("employer",)),
    (r"In which city is the headquarters of (.+)['’]s employer", ("employer", "headquarters")),
    (r"In which city is the headquarters of (.+)", ("headquarters",)),
    (r"In which year was (.+) founded", ("founded",)),
]
_HEADING = re.compile(r"#\s+(.+)")
_FIELD = re.compile(r"([\w][\w -]*):\s*(\S.*)")
_ATTRIBUTES = {"born", "mentor", "employer", "headquarters", "founded"}


def _plain_name(text: str) -> bool:
    words = text.split()
    return bool(words) and all(word[0].isupper() for word in words) and not re.search(r"['’]s\b", text)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _read_index(kb: Path) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for entry in sorted(kb.iterdir()):
        path = resolve_inside(kb, entry.name)
        if path.is_dir():
            raise ValueError(f"nested knowledge base directory {entry.name!r}")
        if entry.suffix.lower() not in (".md", ".markdown"):
            continue
        if not path.is_file():
            raise OSError(f"unreadable Markdown entry {entry.name!r}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        heading = _HEADING.fullmatch(lines[0]) if lines else None
        if heading is None:
            raise ValueError(f"unknown Markdown heading format in {entry.name!r}")
        entity = _normalize(heading[1])
        for line in lines[1:]:
            field = _FIELD.fullmatch(line)
            if field is None:
                raise ValueError(f"unknown Markdown field format in {entry.name!r}")
            attribute = _normalize(field[1])
            if attribute not in _ATTRIBUTES:
                raise ValueError(f"unknown Markdown attribute in {entry.name!r}")
            key = (entity, attribute)
            index.setdefault(key, {}).setdefault(field[2], entry.name)
    return index


def solve(task: TaskSpec) -> SolveResult:
    question = question_text(task.instruction)
    intro = _INTRO.fullmatch(question)
    directory = intro["kb"] if intro else "kb"
    if intro:
        question = intro["question"]
    for pattern, attributes in _QUERIES:
        match = re.fullmatch(pattern, question, re.IGNORECASE)
        if match:
            entity = match[1].strip()
            break
    else:
        return SolveResult.unsupported("unrecognized knowledge base question")

    if task.workspace is None:
        if intro:
            return SolveResult.unsolvable(f"no workspace containing knowledge base {directory!r}")
        return SolveResult.unsupported("no workspace to locate the knowledge base")
    try:
        kb = resolve_inside(task.workspace, directory)
        if not kb.exists():
            if intro:
                return SolveResult.unsolvable(f"knowledge base {directory!r} is missing")
            return SolveResult.unsupported("no knowledge base at the default path")
        index = _read_index(kb)
    except (OSError, UnicodeError) as error:
        return SolveResult.error(f"cannot read knowledge base: {error}")
    except ValueError as error:
        return SolveResult.unsupported(str(error))

    known = {name for name, _ in index}
    evidence = []
    for hop, attribute in enumerate(attributes):
        values = index.get((_normalize(entity), attribute), {})
        if not values:
            if hop == 0 and _normalize(entity) not in known and not _plain_name(entity):
                return SolveResult.unsupported(f"question subject {entity!r} is a description, not a name")
            return SolveResult.unsolvable(
                f"fully read {directory!r}; no {attribute} fact for {entity!r}",
            )
        if len(values) > 1:
            return SolveResult.unsupported(f"conflicting {attribute} facts for {entity!r}")
        value, source = next(iter(values.items()))
        if attribute in ("born", "founded") and not re.fullmatch(r"[0-9]{4}", value):
            return SolveResult.unsupported(f"unrecognized year format in {source!r}")
        evidence.append(f"{directory.rstrip('/')}/{source}: {entity} {attribute} = {value}")
        entity = value

    return SolveResult.solved(entity, " → ".join(evidence))
