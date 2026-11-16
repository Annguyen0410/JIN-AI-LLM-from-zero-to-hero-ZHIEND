"""
tools.py — Phase 6 tool registry with JSON schemas.

Tools are stdlib-only. Python execution is restricted defense-in-depth,
NOT an OS security sandbox, and is disabled in agent dispatch by default:

- ``calculator``  — recursive-descent expression parser. **No eval**,
  no Python builtins reachable, bounded recursion depth.
- ``python_exec`` — runs user code in a *subprocess* sandbox:
  isolated interpreter (``-I``), import whitelist, network disabled,
  file writes blocked, hard wall-clock timeout, output byte cap.
- ``web_search`` / ``fetch_page`` — wrap ``web_search.py`` (SSRF
  guard + robots.txt + per-domain rate limit already enforced there).
- ``memory_search`` / ``memory_write`` — wrap ``MemoryStore``.
- ``now`` — current UTC/local time.

``execute_tool(name, args, ctx)`` is the single dispatch entry point;
``TOOL_SCHEMAS`` declares each tool's JSON schema (name + properties)
for the model to follow when emitting ``<|tool|>{...}<|end|>``.
"""

from __future__ import annotations

import ast
import base64
import datetime
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile

# ── calculator: recursive-descent parser (no eval) ────────────────────

_MAX_CALC_DEPTH = 60


class CalculatorError(ValueError):
    """Raised on any calculator failure (bad syntax, overflow, ...)."""


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            j = i
            while j < n and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        if expr[i : i + 2] in ("**", "//"):
            tokens.append(expr[i : i + 2])
            i += 2
            continue
        tokens.append(ch)
        i += 1
    return tokens


class _Parser:
    """Recursive-descent parser. Grammar (precedence low→high):

        expr    := term (('+'|'-') term)*
        term    := factor (('*'|'/'|'//'|'%') factor)*
        factor  := unary ('**' factor)?
        unary   := ('+'|'-') unary | atom
        atom    := NUMBER | NAME '(' expr ')' | '(' expr ')' | NAME
    """

    _FUNCS = {
        "abs": abs, "sqrt": math.sqrt, "cbrt": lambda x: x ** (1 / 3),
        "exp": math.exp, "log": math.log, "log10": math.log10, "log2": math.log2,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "floor": math.floor, "ceil": math.ceil, "round": round,
        "min": min, "max": max, "pow": lambda a, b: a ** b if abs(b) <= 10000 else (_ for _ in ()).throw(ValueError("exponent too large")), "hypot": math.hypot,
        "degrees": math.degrees, "radians": math.radians,
        "factorial": lambda x: math.factorial(int(x)) if x == int(x) and 0 <= x <= 170 else (_ for _ in ()).throw(ValueError("factorial range is 0..170")), "gcd": lambda a, b: math.gcd(int(a), int(b)),
    }
    _CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

    def __init__(self, tokens: list[str]):
        self.toks = tokens
        self.pos = 0
        self.depth = 0

    def _peek(self) -> str:
        return self.toks[self.pos] if self.pos < len(self.toks) else ""

    def _next(self) -> str:
        tok = self._peek()
        self.pos += 1
        return tok

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > _MAX_CALC_DEPTH:
            raise CalculatorError("expression too deeply nested")

    def parse(self) -> float:
        val = self._expr()
        if self.pos != len(self.toks):
            raise CalculatorError(f"unexpected token: {self._peek()!r}")
        return val

    def _expr(self) -> float:
        val = self._term()
        while self._peek() in ("+", "-"):
            op = self._next()
            rhs = self._term()
            val = val + rhs if op == "+" else val - rhs
        return val

    def _term(self) -> float:
        val = self._factor()
        while self._peek() in ("*", "/", "//", "%"):
            op = self._next()
            rhs = self._factor()
            if op == "*":
                val = val * rhs
            elif op == "/":
                if rhs == 0:
                    raise CalculatorError("division by zero")
                val = val / rhs
            elif op == "//":
                if rhs == 0:
                    raise CalculatorError("division by zero")
                val = math.floor(val / rhs)
            else:
                if rhs == 0:
                    raise CalculatorError("modulo by zero")
                val = val % rhs
        return val

    def _factor(self) -> float:
        """Unary level: ``-2**2 == -(2**2) == -4`` (power binds tighter)."""
        self._enter()
        tok = self._peek()
        if tok in ("+", "-"):
            self._next()
            val = self._factor()
            return val if tok == "+" else -val
        return self._power()

    def _power(self) -> float:
        self._enter()
        base = self._atom()
        if self._peek() == "**":
            self._next()
            exp = self._factor()  # right-assoc; exponent may itself be unary
            if abs(exp) > 10000:
                raise CalculatorError("exponent too large")
            if base == 0 and exp < 0:
                raise CalculatorError("0 cannot be raised to a negative power")
            try:
                return base ** exp
            except (OverflowError, ValueError):
                raise CalculatorError("power result out of range")
        return base

    def _atom(self) -> float:
        tok = self._next()
        if not tok:
            raise CalculatorError("unexpected end of expression")
        if tok[0].isdigit() or tok[0] == ".":
            try:
                return float(tok)
            except ValueError:
                raise CalculatorError(f"invalid number: {tok!r}")
        if tok in self._CONSTS:
            return self._CONSTS[tok]
        if tok in self._FUNCS:
            if self._peek() != "(":
                raise CalculatorError(f"{tok} needs parentheses, e.g. {tok}(...)")
            self._next()
            args = []
            if self._peek() != ")":
                args.append(self._expr())
                while self._peek() == ",":
                    self._next()
                    args.append(self._expr())
            if self._peek() != ")":
                raise CalculatorError(f"missing ')' after {tok}(")
            self._next()
            try:
                return float(self._FUNCS[tok](*args))
            except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
                raise CalculatorError(f"{tok}(...) failed: {exc}")
        if tok == "(":
            val = self._expr()
            if self._peek() != ")":
                raise CalculatorError("missing ')'")
            self._next()
            return val
        raise CalculatorError(f"unknown symbol: {tok!r}")


def _fmt_number(x: float) -> str:
    """Format a float readably: integers without decimals, big numbers
    with thousands separators, small magnitudes as-is."""
    if math.isnan(x) or math.isinf(x):
        raise CalculatorError("result is not a finite number")
    if abs(x) >= 1e15 or (abs(x) < 1e-9 and x != 0):
        return f"{x:.6e}"
    if float(x).is_integer():
        return f"{int(x):,}"
    return f"{x:.10f}".rstrip("0").rstrip(".")


def calculate(expression: str) -> str:
    """Evaluate ``expression``; returns a *string* result (safe to log)."""
    expr = (expression or "").strip()
    if not expr:
        raise CalculatorError("empty expression")
    if len(expr) > 500:
        raise CalculatorError("expression too long")
    parser = _Parser(_tokenize(expr))
    try:
        return _fmt_number(parser.parse())
    except RecursionError:
        raise CalculatorError("expression too deeply nested")
    except (OverflowError, TypeError, ValueError) as exc:
        raise CalculatorError(str(exc)) from exc


# ── python_exec: subprocess sandbox ───────────────────────────────────

_SANDBOX_IMPORT_WHITELIST = frozenset({
    "math", "json", "random", "re", "string", "itertools", "collections",
    "datetime", "fractions", "statistics", "decimal", "textwrap",
    "unicodedata", "bisect", "heapq", "operator", "functools", "typing",
    "numbers", "dataclasses", "enum", "array", "cmath", "time", "hashlib",
})

# Runs INSIDE the sandboxed subprocess. Installs an import hook that only
# allows the whitelist, blocks all network socket calls, and forbids file
# writes. Read-only file access stays allowed (harmless).
_SANDBOX_WRAPPER = r"""
import base64, sys, json, builtins, io, importlib
_code = base64.b64decode(sys.argv[1]).decode("utf-8")
_ALLOWED = frozenset(__ALLOWED__)
# Preload approved stdlib modules before blocking their internal imports.
for _name in _ALLOWED:
    importlib.import_module(_name)
def _audit(event, args):
    if event == "open" or event.startswith(("socket.", "subprocess.", "ctypes.", "os.")):
        raise PermissionError("host I/O disabled by restricted executor")
sys.addaudithook(_audit)

# 1. Import whitelist: reject any top-level module not in the list.
_real_import = builtins.__import__
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if level == 0 and top not in _ALLOWED:
        raise ImportError("import of %r blocked by sandbox whitelist" % top)
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _safe_import

# 2. Network disabled: socket connect/create_connection always raise.
try:
    import socket
    def _no_net(*a, **k):
        raise RuntimeError("network access disabled by sandbox")
    socket.socket.connect = _no_net
    socket.socket.connect_ex = lambda *a, **k: -1
    socket.create_connection = _no_net
except Exception:
    pass

# 3. File writes forbidden; reads allowed.
_real_open = builtins.open
def _safe_open(file, mode="r", *a, **k):
    raise PermissionError("file access disabled by restricted executor")
builtins.open = _safe_open

# 4. subprocess / os.system are unreachable anyway (import blocked), but
#    belt-and-braces for anything that already holds references.
try:
    import os
    os.system = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("os.system disabled by sandbox"))
    os.popen = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("os.popen disabled by sandbox"))
except Exception:
    pass

try:
    exec(compile(_code, "<sandbox>", "exec"), {"__name__": "__sandbox__"})
except SystemExit:
    pass
except BaseException as exc:
    print(type(exc).__name__ + ": " + str(exc))
"""


def python_exec(code: str, *, timeout: float = 8.0, max_output: int = 4000) -> str:
    """Run ``code`` in an isolated subprocess sandbox; returns stdout.

    Raises ``RuntimeError`` for non-zero exit / timeout / overlong output.
    """
    code = (code or "").strip()
    if not code:
        raise RuntimeError("empty code")
    if len(code) > 8_000:
        raise RuntimeError("code too long (max 8000 chars)")
    # Reject introspection and dynamic execution before spawning. This is a
    # restricted language subset; hostile arbitrary Python needs OS isolation.
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    forbidden = {"eval", "exec", "compile", "globals", "locals", "vars", "getattr",
                 "setattr", "delattr", "type", "object", "super", "breakpoint",
                 "help", "input", "memoryview", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return "PermissionError: introspection blocked by restricted executor"
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in forbidden):
            return "PermissionError: dynamic execution blocked by restricted executor"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if getattr(node, "level", 0) or any(n.split(".")[0] not in _SANDBOX_IMPORT_WHITELIST for n in names):
                return "ImportError: import blocked by restricted executor"
    timeout = max(0.1, min(float(timeout), 8.0))
    max_output = max(128, min(int(max_output), 16000))
    # Inject the whitelist with plain replace() — ``%``-formatting would
    # also try to consume the wrapper's OWN nested format placeholders.
    wrapper = _SANDBOX_WRAPPER.replace(
        "__ALLOWED__", repr(sorted(_SANDBOX_IMPORT_WHITELIST)))
    b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    cmd = [sys.executable, "-I", "-c", wrapper, b64]
    from storage import configure_runtime
    configure_runtime()
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "TMPDIR") if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    import time
    with tempfile.TemporaryDirectory(prefix="jin-exec-") as td, tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.Popen(cmd, stdout=output, stderr=output, env=env, cwd=td)
        except OSError as exc:
            raise RuntimeError(f"executor failed to start: {exc}") from exc
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"code timed out after {timeout}s")
                if os.fstat(output.fileno()).st_size > max_output * 4:
                    raise RuntimeError("output budget exceeded")
                time.sleep(0.01)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        output.seek(0)
        out = output.read(max_output * 4).decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RuntimeError(f"code failed (exit {proc.returncode}): {out[:500]}")
    return out[:max_output]


# ── memory tools ──────────────────────────────────────────────────────


def memory_search(store, query: str, k: int = 3) -> str:
    """Keyword-overlap retrieval over ``MemoryStore.all()`` entries.

    Deliberately model-free (cheap, always available). Returns ranked
    entries with tier / confidence / source so the model can cite them.
    """
    if store is None:
        return "memory store is not available"
    query = (query or "").strip()
    if not query:
        return "empty query"
    q_terms = set(re.findall(r"\w+", query.lower()))
    scored: list[tuple[float, dict]] = []
    for rec in store.all():
        if rec.get("needs_verification") or rec.get("contradicts"):
            continue
        text = (rec.get("text") or "") + " " + (rec.get("claim") or "")
        t_terms = set(re.findall(r"\w+", text.lower()))
        if not t_terms:
            continue
        overlap = len(q_terms & t_terms)
        if overlap == 0:
            continue
        score = overlap / max(len(q_terms), 1) * float(rec.get("confidence", 0.5))
        scored.append((score, rec))
    scored.sort(key=lambda x: -x[0])
    lines = []
    for score, rec in scored[:k]:
        lines.append(
            f"- [{rec.get('tier')}] (conf {rec.get('confidence', 0.5):.2f}) "
            f"{rec.get('text', '')[:400]}"
            + (f"  (source: {rec.get('source_url')})" if rec.get("source_url") else "")
        )
    return "\n".join(lines) if lines else "no memory entries matched"


def memory_write(store, text: str, *, claim: str | None = None,
                 source_url: str | None = None, confidence: float = 0.6,
                 tier: str = "semantic") -> str:
    """Store one verified knowledge item in long-term memory."""
    if store is None:
        return "memory store is not available"
    if not (text or "").strip():
        return "empty text — nothing stored"
    try:
        mid = store.add(
            tier=tier, text=text, claim=claim or text,
            source_url=source_url, confidence=float(confidence),
        )
        return f"stored as {mid} (tier={tier}, confidence={confidence})"
    except ValueError as exc:
        return f"store failed: {exc}"


# ── tool registry ─────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression. Supports + - * / // % **, "
                       "parentheses, functions sqrt/abs/log/log10/exp/sin/cos/"
                       "tan/floor/ceil/round/min/max/pow/gcd, constants pi/e.",
        "properties": {"expression": {"type": "string", "required": True}},
    },
    {
        "name": "python_exec",
        "description": "Run a short Python program in a safe sandbox "
                       "(no network, no file writes, no external packages). "
                       "Prints its stdout. Whitelisted imports: math json "
                       "random re itertools collections datetime fractions "
                       "statistics decimal heapq bisect etc.",
        "properties": {"code": {"type": "string", "required": True}},
    },
    {
        "name": "web_search",
        "description": "Search the live web. Use for recent events, prices, "
                       "facts outside training, or anything uncertain.",
        "properties": {"query": {"type": "string", "required": True}},
    },
    {
        "name": "fetch_page",
        "description": "Fetch a URL and return cleaned readable text "
                       "(max ~4000 chars).",
        "properties": {"url": {"type": "string", "required": True}},
    },
    {
        "name": "memory_search",
        "description": "Search Jin's long-term memory for prior verified "
                       "knowledge on a topic.",
        "properties": {"query": {"type": "string", "required": True},
                       "k": {"type": "integer"}},
    },
    {
        "name": "memory_write",
        "description": "Save a verified fact to long-term memory so future "
                       "answers can cite it. Only write claims you are "
                       "confident about (cross-checked with sources).",
        "properties": {"text": {"type": "string", "required": True},
                       "source_url": {"type": "string"},
                       "confidence": {"type": "number"}},
    },
    {
        "name": "now",
        "description": "Get the current date and time.",
        "properties": {},
    },
]

TOOL_NAMES: set[str] = {s["name"] for s in TOOL_SCHEMAS}


def tool_schema_json() -> str:
    """Render the registry as one JSON string for the system prompt."""
    return json.dumps(TOOL_SCHEMAS, ensure_ascii=False)


def execute_tool(name: str, args: dict, ctx: dict | None = None) -> str:
    """Dispatch one tool call; returns a plain-text result string.

    ``ctx`` may carry ``store`` (MemoryStore), ``memory`` (same), and
    ``search_enabled`` (bool) so web tools can be disabled by policy.
    """
    ctx = ctx or {}
    args = args or {}
    try:
        if not isinstance(args, dict):
            raise ValueError("tool args must be an object")
        schema = next((s for s in TOOL_SCHEMAS if s["name"] == name), None)
        if schema is None:
            raise ValueError("unknown tool")
        props = schema["properties"]
        if set(args) - set(props):
            raise ValueError("unknown arguments")
        types = {"string": str, "integer": int, "number": (int, float)}
        for key, prop in props.items():
            if prop.get("required") and key not in args:
                raise ValueError(f"missing argument: {key}")
            if key in args and (not isinstance(args[key], types[prop["type"]]) or isinstance(args[key], bool)):
                raise ValueError(f"invalid argument type: {key}")
        if name == "calculator":
            return calculate(str(args.get("expression", "")))
        if name == "python_exec":
            if not ctx.get("allow_python_exec", False):
                return "[tool python_exec error: disabled; OS isolation required for untrusted code]"
            return python_exec(str(args.get("code", "")), timeout=ctx.get("timeout", 8.0))
        if name == "web_search":
            if not ctx.get("search_enabled", True):
                return "web search is disabled"
            import web_search
            block, sources = web_search.web_context(
                str(args.get("query", "")), max_results=3, fetch_top=1,
                page_chars=1_200,
            )
            if not sources:
                return block
            lines = [f"Results for {args.get('query')}:"]
            for i, src in enumerate(sources, 1):
                lines.append(f"[{i}] {src.get('title')} — {src.get('url')}")
                if src.get("snippet"):
                    lines.append(f"    {src.get('snippet')}")
            return "\n".join(lines)
        if name == "fetch_page":
            if not ctx.get("search_enabled", True):
                return "page fetching is disabled"
            import web_search
            return web_search.fetch_text(str(args.get("url", "")), max_chars=4_000)
        if name == "memory_search":
            store = ctx.get("store") or ctx.get("memory")
            return memory_search(store, str(args.get("query", "")),
                                 k=max(1, min(int(args.get("k", 3)), 20)))
        if name == "memory_write":
            if not ctx.get("allow_memory_write", False):
                return "[tool memory_write error: explicit write permission required]"
            store = ctx.get("store") or ctx.get("memory")
            conf = args.get("confidence", 0.6)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.6
            return memory_write(
                store, str(args.get("text", "")),
                claim=args.get("claim"),
                source_url=args.get("source_url"),
                confidence=conf,
            )
        if name == "now":
            now = datetime.datetime.now().astimezone()
            return (f"{now.isoformat(timespec='seconds')} "
                    f"({now.strftime('%A')})")
        raise RuntimeError(f"unknown tool: {name!r}")
    except (CalculatorError, RuntimeError, ValueError, TypeError, OverflowError, OSError) as exc:
        return f"[tool {name} error: {exc}]"