import ast
import codecs


def isolated_defaults(source: bytes) -> bytes | None:
    bom = codecs.BOM_UTF8 if source.startswith(codecs.BOM_UTF8) else b""
    source = source[len(bom):]
    if b"[" not in source and b"{" not in source:
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    if not tree.body:
        return None
    sentinel = b"_missing_default"
    if sentinel in source:
        return None
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = function.body
        has_doc = ast.get_docstring(function, clean=False) is not None
        if len(body) <= has_doc:
            continue
        first = body[int(has_doc)]
        line_index = min([first.lineno] + [d.lineno for d in getattr(first, "decorator_list", [])]) - 1
        indent = lines[line_index][:first.col_offset]
        if not indent.isspace() or line_index < function.lineno:
            continue
        args = function.args
        positional = args.posonlyargs + args.args
        bindings = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
        bindings.extend(zip(args.kwonlyargs, args.kw_defaults))
        guards = []
        for arg, default in bindings:
            if isinstance(default, ast.List) and not default.elts:
                literal = b"[]"
            elif isinstance(default, ast.Dict) and not default.keys:
                literal = b"{}"
            else:
                continue
            if not _can_share_state(function, arg.arg):
                continue
            name = arg.arg.encode()
            start = offsets[default.lineno - 1] + default.col_offset
            end = offsets[default.end_lineno - 1] + default.end_col_offset
            if source[start:end].strip() != literal:
                continue
            changes.append((start, end, sentinel))
            guards.append(indent + b"if " + name + b" is " + sentinel + b":" + newline
                          + indent + (b"\t" if indent.endswith(b"\t") else b"    ") + name + b" = " + literal + newline)
        if guards:
            pos = offsets[line_index]
            changes.append((pos, pos, b"".join(guards)))
    if not changes:
        return None
    first = next((n for i, n in enumerate(tree.body)
                  if not (i == 0 and ast.get_docstring(tree, clean=False) is not None)
                  and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")), None)
    if first is None:
        return None
    start_line = min([first.lineno] + [d.lineno for d in getattr(first, "decorator_list", [])])
    pos = offsets[start_line - 1]
    changes.append((pos, pos, sentinel + b" = object()" + newline))
    for start, end, replacement in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[end:]
    try:
        compile(source, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return None
    return bom + source


def _can_share_state(function, name: str) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == name:
            if node.attr in {"append", "extend", "insert", "pop", "remove", "clear", "update", "setdefault", "sort", "reverse"}:
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if isinstance(node.value, ast.Name) and node.value.id == name:
                return True
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == name:
            if any(isinstance(t, ast.Attribute) for t in node.targets):
                return True
    return False
