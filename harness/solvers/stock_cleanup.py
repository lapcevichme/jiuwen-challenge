import ast
from collections import defaultdict


def edits(cls):
    by_slot = defaultdict(list)
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef) or not method.args.args:
            continue
        guards = set()
        receiver = method.args.args[0].arg
        decremented = None
        for stmt in method.body:
            if isinstance(stmt, ast.If) and not stmt.orelse and isinstance(stmt.test, ast.Compare):
                cmp = stmt.test
                if len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Gt) and len(stmt.body) == 1:
                    if isinstance(cmp.left, ast.Name) and isinstance(stmt.body[0], ast.Raise):
                        guards.add((cmp.left.id, ast.unparse(cmp.comparators[0])))
                if (decremented and len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Lt)
                        and isinstance(cmp.comparators[0], ast.Constant) and cmp.comparators[0].value == 0
                        and ast.unparse(cmp.left) == decremented and isinstance(cmp.left, ast.Subscript)
                        and isinstance(cmp.left.value, ast.Subscript)
                        and isinstance(cmp.left.value.value, ast.Attribute)
                        and isinstance(cmp.left.value.value.value, ast.Name)
                        and cmp.left.value.value.value.id == receiver
                        and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Delete)
                        and len(stmt.body[0].targets) == 1
                        and ast.unparse(stmt.body[0].targets[0]) == ast.unparse(cmp.left.value)):
                    by_slot[decremented].append((method.name, cmp))
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Subscript):
                target = ast.unparse(stmt.target)
                if (isinstance(stmt.op, ast.Sub) and isinstance(stmt.value, ast.Name)
                        and (stmt.value.id, target) in guards):
                    decremented = target
                else:
                    decremented = None
            elif isinstance(stmt, (ast.Assign, ast.Delete, ast.For, ast.While, ast.Try, ast.With)):
                decremented = None
                guards.clear()
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                target = stmt.value.func
                if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id in {arg.arg for arg in method.args.args[1:]}):
                    decremented = None
                    guards.clear()
    for matches in by_slot.values():
        if len({name for name, _ in matches}) >= 2:
            for _, cmp in matches:
                yield cmp, cmp, f"{ast.unparse(cmp.left)} <= 0"
