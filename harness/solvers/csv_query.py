import csv
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

SYSTEM = """Solve the task with one short SQLite SELECT. Return only SQL. Tables use each filename's stem.
Empty cells are NULL; integer and decimal columns are numeric; codes with leading zeros remain text.
Use single quotes for strings, bare column names, 1.0 for fractional division, and printf when trailing zeros
are required. Read all relevant rows. If a required fact or column is absent return UNSOLVABLE.
If SQLite cannot express the exact calculation simply, return a minimal standard-library Python script
in a fenced python block instead. It can open the listed files and must print only the final answer."""

_NAME = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]+")
MAX_BYTES = 1024 * 1024


def files(root: Path | None) -> tuple[str, ...]:
    if root is None:
        return ()
    entries = sorted(root.iterdir())
    if not 1 <= len(entries) <= 4:
        return ()
    names = set()
    size = 0
    for path in entries:
        name = path.stem.casefold()
        if (path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}
                or not _NAME.fullmatch(path.stem) or name in names or name.startswith("sqlite_")):
            return ()
        names.add(name)
        size += path.stat().st_size
    return tuple(path.name for path in entries) if size <= MAX_BYTES else ()


def eligible(root: Path | None, question: str) -> bool:
    if re.search(r"\b(?:median|nearest|half[- ]even|banker'?s? rounding)\b", question, re.I):
        return False
    return bool(files(root))


def extract(reply: str) -> str | None:
    text = reply.strip()
    fenced = re.fullmatch(r"```(?:sql|sqlite)?[ \t]*\r?\n(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced[1].strip()
    if re.match(r"(?:SELECT\b|WITH\s+(?:RECURSIVE\s+)?\w+\s*(?:\([^()]*\)\s*)?AS\s*\()", text, re.I):
        return text
    return None


def _type(values) -> str:
    values = {v for v in values if v}
    if values and all(_INTEGER.fullmatch(v) and len(v) <= 18 for v in values):
        return "INTEGER"
    if values and all(len(v) <= 15 and (_DECIMAL.fullmatch(v) or _INTEGER.fullmatch(v)) for v in values):
        return "REAL"
    return "TEXT"


def _load(db, path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ",", strict=True)
        header = next(reader, [])
        if (not 1 <= len(header) <= 80 or any(not _NAME.fullmatch(col) for col in header)
                or len({col.casefold() for col in header}) != len(header)):
            raise ValueError("unsupported column names")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header) or len(rows) >= 20000:
                raise ValueError("unsupported table shape")
            rows.append(row)
    columns = [f'"{col}" {_type(row[i] for row in rows)}' for i, col in enumerate(header)]
    db.execute(f'CREATE TABLE "{path.stem}" ({", ".join(columns)})')
    slots = ",".join("?" for _ in header)
    db.executemany(f'INSERT INTO "{path.stem}" VALUES ({slots})', ([v or None for v in row] for row in rows))


_SQL_TOKEN = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*"|`[^`]*`|\[[^\]]*\]|--[^\n]*|/\*.*?\*/|[\w$]+|\s+|.""", re.S)


def _file_references(query: str, names: tuple[str, ...]) -> str:
    by_name = {name.casefold(): Path(name).stem for name in names}
    tokens = _SQL_TOKEN.findall(query)
    for i, token in enumerate(tokens):
        if token.upper() not in {"FROM", "JOIN"}:
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].isspace():
            j += 1
        if j == len(tokens):
            continue
        end = j + 1
        if end + 1 < len(tokens) and tokens[end] == ".":
            end += 2
        name = "".join(tokens[j:end])
        if name[:1] in {'"', "`", "["}:
            name = name[1:-1]
        if stem := by_name.get(name.casefold()):
            tokens[j:end] = [f'"{stem}"'] + [""] * (end - j - 1)
    return "".join(tokens)


def scalar(root: Path, query: str):
    names = files(root)
    if not names or not re.match(r"(?:SELECT|WITH)\b", query.strip(), re.I):
        raise ValueError("expected a query on the provided tables")
    tables = {Path(name).stem.casefold() for name in names}
    read_table = False

    def authorize(action, arg, function, database, trigger):
        nonlocal read_table
        if action == sqlite3.SQLITE_READ and arg.casefold() in tables:
            read_table = True
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_FUNCTION and function in {
            "abs", "avg", "cast", "ceil", "coalesce", "count", "date", "dense_rank", "floor", "ifnull", "instr",
            "julianday", "length", "like", "lower", "max", "min", "nullif", "printf", "rank", "replace",
            "round", "row_number", "strftime", "substr", "sum", "total", "trim", "upper",
        }):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    db = sqlite3.connect(":memory:")
    try:
        for name in names:
            _load(db, root / name)
        db.execute("PRAGMA query_only=ON")
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_BYTES)
        db.set_authorizer(authorize)
        db.set_progress_handler(lambda: 1, 2_000_000)
        try:
            cursor = db.execute(query)
        except sqlite3.OperationalError as error:
            if "no such table:" not in str(error):
                raise
            corrected = _file_references(query, names)
            if corrected == query:
                raise
            cursor = db.execute(corrected)
        rows = cursor.fetchmany(2)
        if not read_table or len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] is None:
            raise ValueError("query must return one non-null cell from the data")
        value = rows[0][0]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite result")
        if not isinstance(value, (int, float, str)) or len(str(value).splitlines()) != 1:
            raise ValueError("invalid result type")
        return value
    finally:
        db.close()


if __name__ == "__main__":
    print(scalar(Path.cwd(), json.load(sys.stdin)))
