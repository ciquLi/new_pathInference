"""Python-version-independent identities for migrated numerical definitions."""
import ast
import hashlib
import json


def canonical_ast(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, ast.AST):
        if type(value).__name__ == "Index":
            return canonical_ast(value.value)
        fields = dict(ast.iter_fields(value))
        if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = fields.get("body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                fields["body"] = body[1:]
        return {
            "node": type(value).__name__,
            **{
                name: canonical_ast(item)
                for name, item in fields.items()
                if item is not None and item != []
            },
        }
    if isinstance(value, list):
        return [canonical_ast(item) for item in value]
    return value


def symbol_sha256(node):
    payload = json.dumps(canonical_ast(node), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
