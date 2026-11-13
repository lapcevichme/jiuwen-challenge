import re
from pathlib import PurePosixPath

_COLLECTION = re.compile(
    r"\b(?:all|each|every|across|compare|comparison|between|both|sum|total|count|difference|"
    r"how many|how much|number of|average|median|earli(?:er|est)|lat(?:er|est)|"
    r"high(?:er|est)|low(?:er|est)|larg(?:er|est)|small(?:er|est)|long(?:er|est)|"
    r"short(?:er|est)|most|least|more|fewer|older|oldest|younger|youngest|maximum|minimum)\b", re.I,
)
_LOOKUP = re.compile(r"\b(?:who|which|where|what|find|identify|look up|determine|report|give)\b", re.I)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_MENTION = re.compile(r"""['"“‘]([^'"“”‘’\n]{3,70})['"”’]|\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b|\b([a-zA-Z]+-?\d+)\b""")
_TYPES = {"captain", "professor", "doctor", "dr", "player", "ship", "station", "company", "person",
          "department", "building", "tournament", "club", "team", "city", "band", "museum", "curator",
          "exhibition", "project", "paper", "author", "publisher", "novel", "album", "registry", "report"}
_GENERIC = _TYPES | {"the", "of", "at", "in", "knowledge", "base", "profile", "summary", "overview",
                     "directory", "records", "assignments", "index", "list", "results", "readme", "current"}


def _words(text):
    return " " + " ".join(_WORD.findall(text.casefold())) + " "


def _names(path, lines):
    title = next((line[2:] for line in lines if line.startswith("# ")), "")
    phrases = set()
    for value in (title, PurePosixPath(path).stem):
        words = _WORD.findall(value.casefold())
        while len(words) > 1 and words[0] in _TYPES | {"the", "of"}:
            words.pop(0)
        if len(words) >= 2 and any(w not in _GENERIC and not w.isdigit() for w in words):
            phrases.add(" " + " ".join(words) + " ")
        elif len(words) == 1 and len(words[0]) >= 4 and words[0] not in _GENERIC and not words[0].isdigit():
            phrases.add(" " + words[0] + " ")
    return phrases


def _referenced_ids(lines, mention):
    id_columns = []
    previous = []
    for line in lines:
        if not line.startswith("|"):
            id_columns, previous = [], []
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if previous and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            id_columns = [i for i, col in enumerate(previous) if _words(col).strip().split()[-1:] == ["id"]]
        elif mention in (_words(cell) for cell in cells):
            for i in id_columns:
                if i < len(cells) and re.fullmatch(r"[a-zA-Z]*[0-9]+", cells[i]):
                    yield cells[i].casefold()
        previous = cells


def connected_indices(question: str, documents: list[tuple[str, list[str]]]) -> list[int] | None:
    scope = re.sub(r"\b(?:(?:how many|number of) floors|floor count)\b", "floors", question, flags=re.I)
    if _COLLECTION.search(scope) or not _LOOKUP.search(question) or len(documents) < 3:
        return None
    query = _words(question)
    names = [_names(path, lines) for path, lines in documents]
    content = [_words(" ".join(lines)) for _, lines in documents]
    selected = {i for i, variants in enumerate(names) if any(v in query for v in variants)}
    anchored = set()
    record_keys = [re.split(r"[-_]", PurePosixPath(path).stem.casefold())[-1] for path, _ in documents]
    for match in _MENTION.finditer(question):
        mention = _words(next(v for v in match.groups() if v))
        hits = {i for i, body in enumerate(content) if mention in body}
        if len(hits) <= 3:
            selected.update(hits)
            anchored.update(hits)
            for i in hits:
                for key in _referenced_ids(documents[i][1], mention):
                    selected.update(j for j, record_key in enumerate(record_keys) if record_key == key)
    if not selected or len(selected) > 4:
        return None
    frontier = selected.copy()
    for _ in range(3):
        related = {j for j in range(len(names)) if j not in selected and any(
            any(v in content[j] for v in names[i]) or any(v in content[i] for v in names[j])
            for i in frontier
        )}
        if not related:
            break
        selected.update(related)
        if len(selected) > 8 or len(selected) * 2 > len(documents):
            return None
        frontier = related
    if len(selected) == 1:
        lines = documents[next(iter(selected))][1]
        if not anchored or any(line.startswith("|") for line in lines):
            return None
    if len(selected) * 2 > len(documents):
        return None
    return sorted(selected)
