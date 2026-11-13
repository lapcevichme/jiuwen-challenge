import ast
import re


def _expr(text):
    return ast.dump(ast.parse(text, mode="eval").body)


def _arguments(fn):
    return [a.arg for a in fn.args.args]


def _empty_clock(tests, name):
    for node in ast.walk(tests):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "assertEqual" and len(node.args) == 2):
            continue
        call, expected = node.args
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
                and len(call.args) == 1 and isinstance(call.args[0], ast.Constant) and call.args[0].value == ""
                and ast.dump(expected) == _expr("(0, 0, 0)")):
            return True
    return False


def _duration_examples(fn, tests_tree):
    examples = []
    for assertion in ast.walk(tests_tree):
        if (isinstance(assertion, ast.Call) and isinstance(assertion.func, ast.Attribute)
                and assertion.func.attr == "assertEqual" and len(assertion.args) == 2):
            call, expected = assertion.args
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == fn.name
                    and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                    and type(call.args[0].value) in (int, float) and isinstance(expected, ast.Constant)
                    and isinstance(expected.value, str)):
                examples.append((call.args[0].value, expected.value))
    return examples


def edits(tree, tests, source):
    try:
        tests_tree = ast.parse(tests)
    except (ValueError, SyntaxError):
        return
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        args = _arguments(fn)
        if fn.name == "is_leap_year" and len(args) == 1 and b"days_in_month" in tests:
            year = args[0]
            simple = ast.parse(f"if {year} % 4 == 0:\n    return True\nreturn False").body
            century = ast.parse(f"if {year} % 4 == 0:\n    if {year} % 100 == 0:\n        return False\n"
                                "    return True\nreturn False").body
            if ([ast.dump(n) for n in fn.body] in ([ast.dump(n) for n in simple], [ast.dump(n) for n in century])
                    and re.search(rb"\b(?:1[6-9]|2[0-4])00\b", tests)):
                first, last = fn.body[0], fn.body[-1]
                yield first, last, f"return {year} % 4 == 0 and ({year} % 100 != 0 or {year} % 400 == 0)"
        elif fn.name == "time_to_seconds" and args == ["hours", "minutes", "seconds"]:
            for stmt in fn.body:
                if isinstance(stmt, ast.Return) and ast.dump(stmt.value) == _expr("hours * 60 + minutes + seconds"):
                    yield stmt, stmt, "return hours * 3600 + minutes * 60 + seconds"
        elif fn.name == "parse_time" and len(args) == 1 and _empty_clock(tests_tree, fn.name):
            stmt = fn.body[0]
            if isinstance(stmt, ast.Assign) and ast.dump(stmt.value) == _expr(f"{args[0]}.split(':')"):
                indent = " " * stmt.col_offset
                yield stmt, stmt, f"if {args[0]} == '':\n{indent}    return (0, 0, 0)\n{indent}{ast.unparse(stmt)}"
        elif fn.name == "add_days" and len(args) == 4 and b"negative" in tests.lower():
            year, month, day, delta = args
            for loop in fn.body:
                if (isinstance(loop, ast.For) and ast.dump(loop.iter) == _expr(f"range({delta})")
                        and len(loop.body) == 2
                        and ast.dump(loop.body[0]) == ast.dump(ast.parse(f"{day} += 1").body[0])
                        and isinstance(loop.body[1], ast.If)
                        and ast.dump(loop.body[1].test) == _expr(f"{day} > days_in_month({year}, {month})")):
                    indent = " " * loop.col_offset
                    reverse = (f"for _ in range(-{delta}):\n    {day} -= 1\n    if {day} == 0:\n"
                               f"        {month} -= 1\n        if {month} == 0:\n            {year} -= 1\n"
                               f"            {month} = 12\n        {day} = days_in_month({year}, {month})")
                    original = ast.get_source_segment(source, loop)
                    yield loop, loop, reverse.replace("\n", "\n" + indent) + "\n" + indent + original
        elif fn.name == "format_duration" and len(args) == 1:
            for value, expected in _duration_examples(fn, tests_tree):
                if value < 0 or not re.fullmatch(r"[0-9]+:[0-5][0-9]:[0-5][0-9]", expected):
                    continue
                h, m, sec = map(int, expected.split(":"))
                if int(value) != 3600 * h + 60 * m + sec or round(value) == int(value):
                    continue
                for call in ast.walk(fn):
                    if isinstance(call, ast.Call) and ast.dump(call) == _expr(f"round({args[0]})"):
                        yield call, call, f"int({args[0]})"
                break
        elif fn.name == "humanize_duration" and len(args) == 1:
            examples = _duration_examples(fn, tests_tree)
            for guard in ast.walk(fn):
                if not isinstance(guard, ast.If) or not isinstance(guard.test, ast.Compare):
                    continue
                var = guard.test.left
                if not isinstance(var, ast.Name) or ast.dump(guard.test) != _expr(f"{var.id} > 0"):
                    continue
                for fstring in (n for n in ast.walk(guard) if isinstance(n, ast.JoinedStr)):
                    for prev, field in zip(fstring.values, fstring.values[1:]):
                        if not (isinstance(prev, ast.Constant) and str(prev.value).strip() in {"hour", "minute", "second"}
                                and isinstance(field, ast.FormattedValue) and isinstance(field.value, ast.IfExp)):
                            continue
                        unit = prev.value.strip()
                        expression = f"'s' if {var.id} != 0 else ''"
                        if (ast.dump(field.value) == _expr(expression)
                                and any(re.search(rf"\b1 {unit}\b", text) for _, text in examples)):
                            literal = field.value.test.comparators[0]
                            yield literal, literal, "1"
        elif fn.name == "format_utc_offset" and len(args) == 1 and b"negative" in tests.lower():
            for call in ast.walk(fn):
                if isinstance(call, ast.Call) and ast.dump(call) == _expr(f"divmod({args[0]}, 60)"):
                    if any(isinstance(n, ast.IfExp) and ast.dump(n.test) == _expr(f"{args[0]} < 0")
                           for n in ast.walk(fn)):
                        yield call, call, f"divmod(abs({args[0]}), 60)"
