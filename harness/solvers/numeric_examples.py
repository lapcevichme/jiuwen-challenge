import ast
import math
import statistics


def _numeric_list(value):
    return type(value) is list and 0 < len(value) <= 32 and all(type(n) is int and abs(n) <= 100 for n in value)


_LISTS = {
    "median": statistics.median, "mean": statistics.mean, "average": statistics.mean,
    "minimum": min, "maximum": max, "product": math.prod, "total": sum,
    "is_sorted": lambda xs: xs == sorted(xs),
    "count_even": lambda xs: sum(n % 2 == 0 for n in xs),
    "count_odd": lambda xs: sum(n % 2 != 0 for n in xs),
    "sum_of_squares": lambda xs: sum(n * n for n in xs),
}
_SCALARS = {"factorial": math.factorial, "gcd": math.gcd}


def examples(tree, modules):
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update((a.asname or a.name, a.name) for a in node.names if a.name in modules)
        elif isinstance(node, ast.ImportFrom) and node.module in modules:
            imports.update((a.asname or a.name, f"{node.module}.{a.name}") for a in node.names)
    samples = {}
    conflicts = set()
    for assertion in ast.walk(tree):
        if not (isinstance(assertion, ast.Call) and isinstance(assertion.func, ast.Attribute)
                and assertion.func.attr == "assertEqual" and len(assertion.args) == 2):
            continue
        call, result = assertion.args
        if not isinstance(call, ast.Call) or call.keywords:
            continue
        if isinstance(call.func, ast.Name):
            qualified = imports.get(call.func.id, "")
        elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            qualified = f"{imports.get(call.func.value.id, '')}.{call.func.attr}"
        else:
            continue
        module, _, name = qualified.rpartition(".")
        if module not in modules or name not in _LISTS and name not in _SCALARS:
            continue
        try:
            args = tuple(ast.literal_eval(arg) for arg in call.args)
            expected = ast.literal_eval(result)
        except (ValueError, SyntaxError):
            continue
        if name in _LISTS and len(args) == 1 and _numeric_list(args[0]):
            op = _LISTS[name]
            probes = [([2, 5],), ([3, 1],), ([4, 4],), ([3, 1, 4, 2],)]
        elif name in _SCALARS and all(type(n) is int and 0 <= n <= 32 for n in args):
            op = _SCALARS[name]
            probes = [(0,), (1,), (4,)] if name == "factorial" else [(15, 6), (0, 5), (18, 24)]
        else:
            continue
        try:
            if type(expected) not in (bool, int, float) or op(*args) != expected:
                conflicts.add(qualified)
                continue
        except (TypeError, ValueError):
            conflicts.add(qualified)
            continue
        samples[qualified] = (module, name, [(repr(a), op(*a)) for a in probes])
    return [case for key, case in samples.items() if key not in conflicts][:10]
