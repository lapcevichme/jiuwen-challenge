import re

from harness.contracts import SolveResult, TaskSpec
from harness.solvers._instruction import question_text


_INITIAL = re.compile(
    r"(?:A|An) (?P<site>[a-z]+) (?:starts with|begins with|initially has) "
    r"(?P<count>[0-9]+) (?P<items>[a-z]+)", re.IGNORECASE,
)


def _units(noun: str) -> str:
    if noun.endswith(("sses", "xes", "ches", "shes")):
        singular = noun[:-2]
    elif noun.endswith("ies"):
        singular = noun[:-3] + "y"
    elif noun.endswith("s") and not noun.endswith("ss"):
        singular = noun[:-1]
    else:
        singular = noun
    if singular.endswith(("s", "x", "ch", "sh")):
        plural = singular + "es"
    elif singular.endswith("y") and len(singular) > 1 and singular[-2] not in "aeiou":
        plural = singular[:-1] + "ies"
    else:
        plural = singular + "s"
    return "(?:" + "|".join(sorted({noun, singular, plural})) + ")"


def solve(task: TaskSpec) -> SolveResult:
    sentences = re.split(r"\.\s+", question_text(task.instruction).lower())
    initial = _INITIAL.fullmatch(sentences[0])
    if len(sentences) < 2 or not initial:
        return SolveResult.unsupported("unrecognized inventory problem")
    site = initial["site"]
    sites, items = _units(site), _units(initial["items"])
    if not re.fullmatch(rf"how many {items} does (?:it|the {site}) have at the end", sentences[-1]):
        return SolveResult.unsupported("unrecognized inventory question")

    operations = (
        (rf"the stock is doubled(?: by merging with a sister {site})?", "double"),
        (r"the stock is tripled", "triple"),
        (rf"the stock is split equally into ([0-9]+) {sites} and it keeps one share \(rounded down\)", "split"),
        (rf"it (?:ships|dispatches) ([0-9]+) {items}(?: per day for ([0-9]+) days?)?", "subtract"),
        (rf"it (?:receives|acquires) ([0-9]+) {items}(?: per day for ([0-9]+) days?)?", "add"),
        (rf"it receives ([0-9]+) deliver(?:y|ies) of ([0-9]+) {items} each", "add"),
        (rf"it loses ([0-9]+)% of its stock to damage \(the number of lost {items} is rounded down\)", "loss"),
        (r"it gives away half of its stock \(the number given away is rounded down\)", "half"),
    )
    stock = int(initial["count"])
    steps = []
    for sentence in sentences[1:-1]:
        for pattern, operation in operations:
            match = re.fullmatch(pattern, sentence.removeprefix("then "))
            if match:
                steps.append((operation, [int(v) for v in match.groups() if v is not None]))
                break
        else:
            return SolveResult.unsupported("unrecognized inventory operation")

    evidence = [str(stock)]
    for operation, values in steps:
        if operation == "split":
            if values[0] == 0:
                return SolveResult.unsolvable("division into zero sites is undefined")
            stock //= values[0]
        elif operation in {"double", "triple"}:
            stock *= 2 if operation == "double" else 3
        elif operation == "loss":
            if values[0] > 100:
                return SolveResult.unsupported("inventory loss exceeds stock")
            stock -= stock * values[0] // 100
        elif operation == "half":
            stock -= stock // 2
        else:
            amount = values[0] * (values[1] if len(values) == 2 else 1)
            stock += amount if operation == "add" else -amount
        evidence.append(f"{operation} {values} -> {stock}")
    return SolveResult.solved(str(stock), ", ".join(evidence))
