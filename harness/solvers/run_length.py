import ast

from harness.solvers import roundtrips


def _expr(source):
    return ast.dump(ast.parse(source, mode="eval").body)


def edits(tree, test_tree, module):
    pairs = roundtrips.examples(test_tree, {module})
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and len(n.args.args) == 1}
    for _, encode, decode, _ in pairs[:1]:
        writer, reader = functions.get(encode), functions.get(decode)
        if writer is None or reader is None:
            return
        text, encoded = writer.args.args[0].arg, reader.args.args[0].arg
        for field in ast.walk(writer):
            if not isinstance(field, ast.JoinedStr) or len(field.values) != 2:
                continue
            number, char = field.values
            if not all(isinstance(n, ast.FormattedValue) and n.conversion == -1 and n.format_spec is None
                       for n in (number, char)):
                continue
            if not (isinstance(number.value, ast.Name) and isinstance(char.value, ast.Subscript)
                    and isinstance(char.value.value, ast.Name) and char.value.value.id == text
                    and isinstance(char.value.slice, ast.Name)):
                continue
            count, index = number.value.id, char.value.slice.id
            break
        else:
            return
        for node in ast.walk(writer):
            if isinstance(node, ast.While) and ast.dump(node.test) == _expr(f"{index} < len({text}) - 1"):
                if any(isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add)
                       and ast.dump(n.value) == _expr(count) and isinstance(n.target, ast.Name)
                       and n.target.id == index for n in node.body):
                    yield node.test, node.test, f"{index} < len({text})"
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                comp = node.test
                if (len(comp.ops) == 1 and isinstance(comp.ops[0], ast.Gt)
                        and ast.dump(comp.left) == _expr(count) and isinstance(comp.comparators[0], ast.Name)
                        and comp.comparators[0].id.startswith(("min_", "minimum_"))):
                    yield comp, comp, f"{count} >= {comp.comparators[0].id}"
        for branch in ast.walk(reader):
            if not isinstance(branch, ast.If) or len(branch.body) != 2:
                continue
            assign, advance = branch.body
            if not (isinstance(advance, ast.AugAssign) and isinstance(advance.target, ast.Name)
                    and isinstance(advance.op, ast.Add) and ast.dump(advance.value) == _expr("1")
                    and isinstance(assign, ast.Assign) and len(assign.targets) == 1
                    and isinstance(assign.targets[0], ast.Name)):
                continue
            i, n = advance.target.id, assign.targets[0].id
            digit = f"{encoded}[{i}]"
            if ast.dump(branch.test) == _expr(digit + ".isdigit()") and ast.dump(assign.value) == _expr(f"int({digit})"):
                indent = " " * advance.col_offset
                extra = (f"{i} += 1\n{indent}while {i} < len({encoded}) and {digit}.isdigit():\n"
                         f"{indent}    {n} = 10 * {n} + int({digit})\n{indent}    {i} += 1")
                yield advance, advance, extra
