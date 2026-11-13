import ast


def _tree(expr):
    return ast.dump(ast.parse(expr, mode="eval").body)


def edits(tree, tests):
    functions = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    forward, inverse = functions.get("celsius_to_fahrenheit"), functions.get("fahrenheit_to_celsius")
    if forward is None or inverse is None or any(len(f.args.args) != 1 for f in (forward, inverse)):
        return
    if not all(f.name.encode() in tests for f in (forward, inverse)):
        return
    x, y = forward.args.args[0].arg, inverse.args.args[0].arg
    def first_return(f):
        return next((s.value for s in f.body if isinstance(s, ast.Return)), None)
    fwd, inv = first_return(forward), first_return(inverse)
    if fwd is None or inv is None:
        return
    def formula(node):
        values = ast.parse(ast.unparse(node), mode="eval").body
        for constant in ast.walk(values):
            if isinstance(constant, ast.Constant) and type(constant.value) is int:
                constant.value = float(constant.value)
        return ast.dump(values)
    if (formula(fwd) == _tree(f"{x} * 9.0 / 5.0 + 32.0") and
            formula(inv) == _tree(f"({y} + 32.0) * 5.0 / 9.0")):
        yield inv.left.left, inv.left.left, f"({y} - 32)"
    validate = functions.get("validate_temp")
    kelvin = functions.get("celsius_to_kelvin")
    if validate is None or kelvin is None or len(kelvin.args.args) != 1 or len(validate.args.args) != 2:
        return
    scale = first_return(kelvin)
    if scale is None or formula(scale) != _tree(f"{kelvin.args.args[0].arg} + 273.15"):
        return
    value, unit = (arg.arg for arg in validate.args.args)
    for gate in ast.walk(validate):
        if not isinstance(gate, ast.If) or ast.dump(gate.test) != _tree(f"{unit} == 'C'"):
            continue
        for stmt in gate.body:
            if isinstance(stmt, ast.If) and formula(stmt.test) == _tree(f"{value} < -273.0"):
                bound = stmt.test.comparators[0]
                yield bound, bound, "-273.15"
