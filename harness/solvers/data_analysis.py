import csv
import re
from collections import Counter

from harness.contracts import SolveResult, TaskSpec
from harness.solvers._instruction import question_text
from harness.task_io import resolve_inside


_FILE = re.compile(
    r"The file `?(?P<file>[\w./-]+\.csv)`?"
    r"(?: has columns [\w, ]+(?: \([0-9]+ rows?\))?)?\.\s+(?P<query>.+)", re.I,
)
_PRODUCT = r"product (?P<quote>['\"])(?P<product>.+?)(?P=quote)"
_MONTHS = r"in months (?P<first>[0-9]+) through (?P<last>[0-9]+) inclusive"
_REGION = r"in the (?P<region>[\w -]+) region"
_QUERIES = (
    (rf"How many units of {_PRODUCT} were sold {_MONTHS}", "units"),
    (rf"How many units of {_PRODUCT} were sold {_REGION}", "units"),
    (rf"What is the total revenue \(units times price\) of {_PRODUCT} {_MONTHS}", "revenue"),
    (rf"How many orders of {_PRODUCT} were placed {_REGION}", "orders"),
    (rf"What is the largest number of units in a single order of {_PRODUCT}", "maximum"),
    (rf"In which month were the most units of {_PRODUCT} sold", "month"),
    (rf"Which region sold the most units of {_PRODUCT}", "region"),
)


def solve(task: TaskSpec) -> SolveResult:
    document = _FILE.fullmatch(question_text(task.instruction))
    if document is None:
        return SolveResult.unsupported("unrecognized CSV aggregation")
    for pattern, operation in _QUERIES:
        query = re.fullmatch(pattern, document["query"], re.I)
        if query:
            break
    else:
        return SolveResult.unsupported("unrecognized CSV question")
    params = query.groupdict()
    period = (int(params["first"]), int(params["last"])) if "first" in params else None
    if period and period[0] > period[1]:
        return SolveResult.unsupported("month range is reversed")
    filename = document["file"]
    if task.workspace is None:
        return SolveResult.unsolvable(f"no workspace containing the required {filename!r}")

    required = {"product", "order_id" if operation == "orders" else "units"}
    if "region" in params or operation == "region":
        required.add("region")
    if period or operation == "month":
        required.add("month")
    if operation == "revenue":
        required.add("price")

    total = 0
    maximum = None
    groups = Counter()
    orders = set()
    try:
        path = resolve_inside(task.workspace, filename)
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = csv.DictReader(file, strict=True)
            headers = rows.fieldnames or []
            if not required.issubset(headers) or len(headers) != len(set(headers)):
                return SolveResult.error(f"CSV needs unique columns including {sorted(required)}")
            for row in rows:
                if None in row or None in row.values():
                    return SolveResult.error(f"malformed CSV row {rows.line_num}")
                if row["product"] != params["product"]:
                    continue
                if "region" in params and row["region"].casefold() != params["region"].casefold():
                    continue
                if period and not period[0] <= int(row["month"]) <= period[1]:
                    continue
                if operation == "orders":
                    if not row["order_id"].strip():
                        return SolveResult.error("empty order identifier")
                    orders.add(row["order_id"])
                    continue
                units = int(row["units"])
                if operation == "maximum":
                    if "order_id" in row and (not row["order_id"].strip() or row["order_id"] in orders):
                        return SolveResult.unsupported("ambiguous units per order")
                    orders.add(row.get("order_id"))
                    maximum = units if maximum is None else max(maximum, units)
                elif operation in {"month", "region"}:
                    key = str(int(row["month"])) if operation == "month" else row["region"].strip()
                    if not key:
                        return SolveResult.error("empty grouping key")
                    groups[key] += units
                else:
                    total += units * int(row["price"]) if operation == "revenue" else units
    except FileNotFoundError:
        return SolveResult.unsolvable(f"required CSV {filename!r} is missing")
    except (OSError, csv.Error, ValueError) as error:
        return SolveResult.error(f"cannot aggregate {filename!r}: {error}")

    if operation in {"month", "region"}:
        leaders = groups.most_common(2)
        if not leaders or len(leaders) == 2 and leaders[0][1] == leaders[1][1]:
            return SolveResult.unsupported("no unique leading group")
        answer = leaders[0][0]
    elif operation == "maximum":
        if maximum is None:
            return SolveResult.unsupported("no qualifying order for maximum")
        answer = str(maximum)
    else:
        answer = str(len(orders) if operation == "orders" else total)
    return SolveResult.solved(answer, f"{filename!r}, {operation} for {params['product']!r}: {answer}")
