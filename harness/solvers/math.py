import re

from harness.contracts import SolveResult, TaskSpec
from harness.solvers._instruction import question_text


_INITIAL = (
    r"(?:a|an) (?P<site>[a-z]+) (?:starts with|begins with|initially has|is initially stocked with) "
    r"(?P<count>[0-9]+) (?P<items>[a-z]+)",
    r"the starting quantity of (?P<items>[a-z]+) in (?:a|an) (?P<site>[a-z]+) is (?P<count>[0-9]+)",
    r"at the (?:beginning|start), there are (?P<count>[0-9]+) (?P<items>[a-z]+) at (?:a|an) (?P<site>[a-z]+)",
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
    initial = next((m for pattern in _INITIAL if (m := re.fullmatch(pattern, sentences[0]))), None)
    if len(sentences) < 2 or not initial:
        return SolveResult.unsupported("unrecognized inventory problem")
    site = initial["site"]
    sites, items = _units(site), _units(initial["items"])
    questions = (
        rf"how many {items} does (?:it|the {site}) have at the end",
        rf"at the end, how many {items} does it have remaining",
        rf"how many {items} does it end up with",
        rf"what is the number of {items} it has at the end",
        rf"how many {items} (?:are left|remain with it) at the end",
    )
    if not any(re.fullmatch(pattern, sentences[-1]) for pattern in questions):
        return SolveResult.unsupported("unrecognized inventory question")

    operations = (
        (rf"the stock is doubled(?: by merging with a sister {site})?", "double"),
        (rf"the stock is (?:then )?doubled (?:via|through) a merger with a sister {site}", "double"),
        (r"the stock is tripled", "triple"),
        (rf"the stock is split equally into ([0-9]+) {sites} and it keeps one share \(rounded down\)", "split"),
        (rf"the stock is (?:subsequently |then )?divided evenly (?:across|among) ([0-9]+) {sites}, "
         r"(?:with one share (?:held back|retained)|and one share is kept) \(rounded down\)", "split"),
        (rf"(?:after that, )?the shares are divided evenly into ([0-9]+) {sites}, "
         r"and a single share is kept \(rounded down\)", "split"),
        (rf"([0-9]+) {items} are shipped out", "subtract"),
        (rf"([0-9]+) {items} are received by it", "add"),
        (rf"it receives ([0-9]+) separate deliveries, each consisting of ([0-9]+) {items}", "add"),
        (r"the number of items given away is half of its stock, rounded down", "half"),
        (rf"of its stock, ([0-9]+)% is lost due to damage, with the number of damaged {items} rounded down", "loss"),
        (r"half of its stock is handed out, with the result rounded down", "half"),
        (rf"it (?:ships|dispatches) ([0-9]+) {items} each day (?:over a span of|for) ([0-9]+) days?", "subtract"),
        (rf"it (?:ships|dispatches) ([0-9]+) {items}(?: per day for ([0-9]+) days?)?", "subtract"),
        (rf"it (?:receives|acquires|obtains) ([0-9]+) {items}(?: per day for ([0-9]+) days?)?", "add"),
        (rf"it receives ([0-9]+) deliver(?:y|ies) of ([0-9]+) {items} each", "add"),
        (rf"it is sent ([0-9]+) deliveries, each of which contains ([0-9]+) {items}", "add"),
        (r"half of its stock is given away, with the number given away rounded down", "half"),
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
