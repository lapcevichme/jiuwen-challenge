import ast
import textwrap

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)


def _start(node):
    return min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1


def merge(module: str, patch: str, original: ast.Module, proposed: ast.Module) -> tuple[str, str] | None:
    classes = {node.name: node for node in original.body if isinstance(node, ast.ClassDef)}
    if not classes:
        return module, patch
    functions = {node.name for node in original.body if isinstance(node, _FUNCTION)}
    methods = {name: {} for name in classes}
    for name, cls in classes.items():
        for method in cls.body:
            if isinstance(method, _FUNCTION):
                methods[name].setdefault(method.name, []).append(method)
    lines = module.splitlines(keepends=True)
    fragments = patch.splitlines(keepends=True)
    edits = []
    consumed = []
    touched = set()
    whole = set()
    for node in proposed.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            owner = node.name
            if any(not isinstance(m, _FUNCTION) for m in node.body):
                whole.add(owner)
                continue
            if node.bases or node.keywords or node.decorator_list:
                whole.add(owner)
                continue
            updates = node.body
        elif isinstance(node, _FUNCTION) and node.name not in functions:
            args = node.args.posonlyargs + node.args.args
            owners = [name for name in classes if node.name in methods[name]]
            if not owners or not args or args[0].arg not in {"self", "cls"}:
                continue
            if len(owners) != 1:
                return None
            owner, = owners
            updates = [node]
        else:
            continue
        consumed.append((_start(node), node.end_lineno))
        for update in updates:
            candidates = methods[owner].get(update.name, [])
            if len(candidates) > 1 or (owner, update.name) in touched:
                return None
            touched.add((owner, update.name))
            target = candidates[0] if candidates else classes[owner].body[-1]
            original_line = lines[target.lineno - 1]
            indent = original_line[:len(original_line) - len(original_line.lstrip())]
            start = target.end_lineno
            if candidates:
                start = _start(target) if update.decorator_list else target.lineno - 1
            if original_line.lstrip().startswith("class "):
                return None
            replacement = textwrap.indent(ast.unparse(update), indent) + "\n"
            if not candidates and not original_line.endswith(("\n", "\r")):
                replacement = "\n" + replacement
            edits.append((start, target.end_lineno, replacement))
    if whole.intersection(owner for owner, _ in touched):
        return None
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    for start, end in reversed(consumed):
        del fragments[start:end]
    return "".join(lines), "".join(fragments)
