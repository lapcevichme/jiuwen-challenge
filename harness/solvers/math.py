import re

from harness.contracts import SolveResult, TaskSpec
from harness.solvers._instruction import question_text

_INITIAL = re.compile(r"A warehouse starts with ([0-9]+) box(?:es)?", re.IGNORECASE)
_QUESTION = re.compile(r"How many boxes does (?:it|the warehouse) have at the end", re.IGNORECASE)
_STEPS = [
    (re.compile(r"(?:Then )?the stock is doubled(?: by merging with a sister warehouse)?", re.IGNORECASE), "*"),
    (re.compile(
        r"(?:Then )?the stock is split equally into ([0-9]+) warehouses? "
        r"and it keeps one share \(rounded down\)", re.IGNORECASE,
    ), "//"),
    (re.compile(r"(?:Then )?it ships ([0-9]+) box(?:es)?(?: per day for ([0-9]+) days?)?", re.IGNORECASE), "-"),
    (re.compile(r"(?:Then )?it receives ([0-9]+) deliver(?:y|ies) of ([0-9]+) box(?:es)? each", re.IGNORECASE), "+"),
    (re.compile(r"(?:Then )?it receives ([0-9]+) box(?:es)?", re.IGNORECASE), "+"),
]


def solve(task: TaskSpec) -> SolveResult:
    sentences = re.split(r"\.\s+", question_text(task.instruction))
    initial = _INITIAL.fullmatch(sentences[0])
    if len(sentences) < 2 or not initial or not _QUESTION.fullmatch(sentences[-1]):
        return SolveResult.unsupported("unrecognized warehouse problem")

    steps = []
    for sentence in sentences[1:-1]:
        for pattern, operation in _STEPS:
            match = pattern.fullmatch(sentence)
            if match:
                operands = [int(value) for value in match.groups() if value is not None]
                operand = operands[0] if operands else 2
                if len(operands) == 2:
                    operand *= operands[1]
                steps.append((operation, operand))
                break
        else:
            return SolveResult.unsupported("unrecognized warehouse operation")

    stock = int(initial[1])
    evidence = [str(stock)]
    for operation, operand in steps:
        if operation == "//":
            if operand == 0:
                return SolveResult.unsolvable("an equal split into zero warehouses is undefined")
            stock //= operand
        elif operation == "*":
            stock *= operand
        elif operation == "-":
            stock -= operand
        else:
            stock += operand
        evidence.append(f"{operation} {operand} -> {stock}")
    return SolveResult.solved(str(stock), ", ".join(evidence))
