"""No Cypher outside graph_queries.py, and no interpolated Cypher anywhere (R-01, R-03,
06 section 2, Definition of done for v1)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
GRAPH_QUERIES = APP / "services" / "retrieval" / "graph_queries.py"

# Methods that send query text to Neo4j: the driver's own and Neo4jService/ScopedGraph's.
QUERY_METHODS = {
    "run", "execute_query", "read", "write", "write_batches", "read_sync", "write_sync",
    "write_batches_sync", "run_autocommit_sync", "queryNodes",
}
CYPHER = re.compile(
    r"\b(MATCH|MERGE|CREATE|OPTIONAL MATCH|DETACH DELETE)\s*\(|\bCALL\s+db\.|\bUNWIND\s+\$"
    r"|\bSHOW\s+(INDEXES|CONSTRAINTS)\b|\bCREATE\s+(CONSTRAINT|VECTOR INDEX|FULLTEXT INDEX|INDEX)\b"
)


def _py_files() -> list[Path]:
    return sorted(p for p in APP.rglob("*.py") if "__pycache__" not in p.parts)


def _is_stringy(node: ast.AST) -> bool:
    if isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return True
    return isinstance(node, ast.BinOp) and (_is_stringy(node.left) or _is_stringy(node.right))


def _is_interpolated(node: ast.AST) -> bool:
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return True
    # '+' builds query text when a string is involved (not 'MAX_BYTES + 1').
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and _is_stringy(node):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and isinstance(node.right, ast.Name) \
            and isinstance(node.left, (ast.Name, ast.Attribute)):
        # 'Q + user_value' with two names: cannot tell, so treat it as interpolation.
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"format", "format_map", "join", "replace"}
    ):
        return True
    return False


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def test_no_interpolated_query_passed_to_neo4j() -> None:
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in QUERY_METHODS
                and node.args
                and _is_interpolated(node.args[0])
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == [], f"interpolated query text passed to Neo4j: {offenders}"


def test_cypher_text_only_in_graph_queries() -> None:
    offenders = []
    for path in _py_files():
        if path == GRAPH_QUERIES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and CYPHER.search(node.value)
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == [], f"Cypher text outside graph_queries.py: {offenders}"


def test_graph_queries_has_no_interpolation() -> None:
    tree = ast.parse(GRAPH_QUERIES.read_text(encoding="utf-8"))
    bad = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        or (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)))
        or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"format", "format_map"})
    ]
    assert bad == [], f"graph_queries.py builds query text dynamically at lines {bad}"


def test_scanner_detects_violations(tmp_path) -> None:
    # Guard against the scanner silently matching nothing.
    sample = (
        "def f(session, x):\n"
        "    session.run(f'MATCH (n {{id: \"{x}\"}}) RETURN n')\n"
        "    session.run('MATCH (n) WHERE n.id = ' + x)\n"
        "Q = 'MATCH (n) RETURN n'\n"
    )
    tree = ast.parse(sample)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args and _is_interpolated(n.args[0])]
    consts = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and CYPHER.search(n.value)]
    assert len(calls) == 2
    assert len(consts) >= 1


def test_no_auth_code_in_app() -> None:
    # 3.2: no authentication of any kind in v1.
    pattern = re.compile(r"\b(import jwt|OAuth2PasswordBearer|HTTPBearer|APIKeyHeader)\b")
    offenders = [
        str(p.relative_to(ROOT)) for p in _py_files() if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
