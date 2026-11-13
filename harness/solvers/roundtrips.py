import ast


def _qualified(name: ast.expr, imported: dict[str, str]) -> str | None:
    if isinstance(name, ast.Name):
        return imported.get(name.id)
    if isinstance(name, ast.Attribute) and isinstance(name.value, ast.Name):
        module = imported.get(name.value.id)
        if module:
            return f"{module}.{name.attr}"
    return None


def examples(tree: ast.Module, modules: set[str]) -> list[tuple[str, str, str, list[str]]]:
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in modules:
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    imports[alias.asname or alias.name] = alias.name

    found = []
    for test in ast.walk(tree):
        if not isinstance(test, ast.FunctionDef) or not test.name.startswith("test"):
            continue
        strings = {}
        for statement in test.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        value = statement.value
                        strings[target.id] = value.value if isinstance(value, ast.Constant) else None
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            assertion = statement.value
            if not (isinstance(assertion.func, ast.Attribute) and assertion.func.attr == "assertEqual"
                    and len(assertion.args) == 2):
                continue
            outer, original = assertion.args
            if not isinstance(outer, ast.Call) or len(outer.args) != 1 or outer.keywords:
                continue
            inner = outer.args[0]
            if not isinstance(inner, ast.Call) or len(inner.args) != 1 or inner.keywords:
                continue
            if ast.dump(original) != ast.dump(inner.args[0]):
                continue
            seed = strings.get(original.id) if isinstance(original, ast.Name) else (
                original.value if isinstance(original, ast.Constant) else None)
            inverse = _qualified(outer.func, imports)
            forward = _qualified(inner.func, imports)
            if not (isinstance(seed, str) and 0 < len(seed) <= 32 and forward and inverse):
                continue
            module, _, encode = forward.rpartition(".")
            other, _, decode = inverse.rpartition(".")
            if module == other and module in modules and encode != decode:
                probes = [seed, seed[:1] * 12, (seed * 2)[:48]]
                found.append((module, encode, decode, list(dict.fromkeys(probes))))
                if len(found) == 4:
                    return found
    return found
