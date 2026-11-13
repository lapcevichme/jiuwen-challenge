import ast
import re
from pathlib import Path

from harness.task_io import resolve_inside


def data_examples(root: Path, sources: dict[str, str], tests: str) -> str:
    requested = set()
    for text in [*sources.values(), tests]:
        try:
            tree = ast.parse(text)
        except (ValueError, SyntaxError):
            continue
        requested.update(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                         and isinstance(n.value, str) and len(n.value) <= 150
                         and re.fullmatch(r"[\w./ -]+\.(?:csv|tsv|txt|json)", n.value, re.I))
    parts = []
    for name in sorted(requested):
        try:
            path = resolve_inside(root, name)
            if (root / name).is_symlink() or not path.is_file():
                continue
            with path.open("rb") as f:
                data = f.read(1025)
            value = data[:1024].decode("utf-8-sig")
        except (OSError, ValueError):
            continue
        parts.append(f"Data file {name} ({'sample' if len(data) > 1024 else 'complete'}):\n{value}")
        if len(parts) == 3:
            break
    return "\n\n".join(parts)


def empty_averages(tree: ast.Module, modules: dict[str, str]):
    def literal(node):
        try:
            ast.literal_eval(node)
        except (ValueError, SyntaxError):
            return False
        return True

    cases = []
    for filename, src in modules.items():
        module = ".".join(Path(filename).with_suffix("").parts)
        try:
            classes = [c for c in ast.parse(src).body if isinstance(c, ast.ClassDef)]
        except (ValueError, SyntaxError):
            continue
        for cls in classes:
            if not any(isinstance(n, ast.ImportFrom) and n.module == module and
                       any(a.name == cls.name and not a.asname for a in n.names) for n in ast.walk(tree)):
                continue
            methods = [m for m in cls.body if isinstance(m, ast.FunctionDef)
                       and re.search(r"(?:^|_)(?:average|mean)(?:_|$)", m.name) and len(m.args.args) == 1
                       and not m.decorator_list and any(isinstance(n, ast.Div) for n in ast.walk(m))]
            for method in methods:
                if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                           and n.func.attr == method.name for n in ast.walk(tree)):
                    continue
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == cls.name
                            and not node.keywords and len(node.args) <= 3 and all(literal(x) for x in node.args)):
                        args = repr(tuple(ast.literal_eval(x) for x in node.args))
                        if len(args) <= 128:
                            cases.append((module, cls.name, args, method.name))
                            break
    return cases[:4]
