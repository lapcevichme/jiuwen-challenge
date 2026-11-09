import csv
import re

from harness.contracts import SolveResult, TaskSpec
from harness.solvers._instruction import question_text
from harness.task_io import resolve_inside

_QUESTION = re.compile(
    r"The file `?(?P<file>[\w./-]+\.csv)`?"
    r"(?: has columns [\w, ]+(?: \([0-9]+ rows?\))?)?\.\s+"
    r"How many units of product (?P<quote>['\"])(?P<product>.+?)(?P=quote) were sold "
    r"in months (?P<first>[0-9]+) through (?P<last>[0-9]+) inclusive",
    re.IGNORECASE,
)


def solve(task: TaskSpec) -> SolveResult:
    question = _QUESTION.fullmatch(question_text(task.instruction))
    if not question:
        return SolveResult.unsupported("unrecognized CSV aggregation")
    filename, product = question["file"], question["product"]
    first, last = int(question["first"]), int(question["last"])
    if first > last:
        return SolveResult.unsupported("month range is reversed")
    if task.workspace is None:
        return SolveResult.unsolvable(f"no workspace containing the required {filename!r}")

    total = count = 0
    try:
        path = resolve_inside(task.workspace, filename)
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = csv.DictReader(file, strict=True)
            headers = rows.fieldnames or []
            if not {"product", "month", "units"}.issubset(headers) or len(headers) != len(set(headers)):
                return SolveResult.error("CSV needs unique product, month and units columns")
            for row in rows:
                if None in row or None in row.values():
                    return SolveResult.error(f"malformed CSV row {rows.line_num}")
                if row["product"] == product and first <= int(row["month"]) <= last:
                    total += int(row["units"])
                    count += 1
    except FileNotFoundError:
        return SolveResult.unsolvable(f"required CSV {filename!r} is missing")
    except (OSError, csv.Error, ValueError) as error:
        return SolveResult.error(f"cannot aggregate {filename!r}: {error}")

    return SolveResult.solved(
        str(total), f"{filename!r}, product={product!r}, months {first}–{last}: {count} rows, {total} units",
    )
