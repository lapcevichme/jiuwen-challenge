import ast
import re

_VAGUE = re.compile(r"\b(?:some|several|an unspecified number of|an unknown number of)\s+\w+", re.I)


def vague_quantity(question: str) -> bool:
    return bool(_VAGUE.search(question))


def stock_calculation(question: str) -> bool:
    return (bool(re.search(r"\b(?:stock|inventory)\b", question, re.I))
            and bool(re.search(r"\b(?:ships?|shipped|receives?|deliveries|dispatches|share|given away)\b",
                               question, re.I)))


def unused_quantity(code: str, question: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in used or name.startswith("_") or not isinstance(node.value, ast.Constant):
            continue
        value = node.value.value
        if type(value) is int and value > 1 and re.search(rf"(?<![\w.,]){value}(?!\w|[.,]\d)", question):
            return f"The script assigns {name} = {value} from the question but never reads it. Check all constraints."
    return None
