"""Phase 17 §8.1 — deterministic intent routing for the JIN assistant.

The Phase 16 benchmark called `tools.calculate()` directly and reported 14/14.
That measured arithmetic, not routing: nothing proved the assistant could tell a
user's arithmetic request from a date, a phone number or an ordinary question.

This module is that router. It is deliberately deterministic and auditable:

* no `eval`, no `exec` — arithmetic is parsed to an AST and only whitelisted
  node types are evaluated;
* dates, phone numbers, identifiers, versions and percentages-as-numbers are
  rejected rather than computed;
* every result carries a trace naming which component produced the answer;
* a router/calculator answer is *never* attributed to the base model.

Usage
-----
    python scripts/phase17_intent_router.py "what is (12 + 8) * 3?"
    python scripts/phase17_intent_router.py --benchmark
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import sys
import time
from dataclasses import dataclass, field, asdict

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

INTENT_CALCULATOR = "calculator"
INTENT_PERCENT = "percentage"
INTENT_CONVERT = "unit_conversion"
INTENT_RETRIEVAL = "retrieval"
INTENT_NONE = "none"

STATUS_OK = "ok"
STATUS_UNSUPPORTED = "unsupported"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_ERROR = "error"
STATUS_NO_TOOL = "no_tool"

MAX_EXPONENT = 64
MAX_ABS_RESULT = 1e18


@dataclass
class Route:
    request: str
    intent: str
    tool: str | None
    args: dict = field(default_factory=dict)
    status: str = STATUS_NO_TOOL
    value: float | int | None = None
    result_text: str | None = None
    explanation: str = ""
    executor: str = "router"
    evidence: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ── normalisation ───────────────────────────────────────────────────────────
_CHAR_MAP = {"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-", "^": "**"}
_WORD_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?(?:calculate|compute|evaluate|what\s+is|"
    r"what's|how much is|how many is|tính|tinh|hãy tính|tính giúp|bằng mấy|"
    r"ket qua|kết quả)\b[:\s]*", re.I)
_TRAILING = re.compile(r"[\s?=.!]+$")


def normalise(request: str) -> str:
    text = request.strip()
    for bad, good in _CHAR_MAP.items():
        text = text.replace(bad, good)
    # thousands separators only between digit groups
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def strip_framing(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _WORD_PREFIX.sub("", text)
        text = _TRAILING.sub("", text)
        text = text.strip()
    return text


# ── arithmetic ──────────────────────────────────────────────────────────────
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_EXPR_CHARS = re.compile(r"^[0-9\s+\-*/%.()]+$")
EXPR_CANDIDATE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[-+*/%]\s*\(?\s*-?\d+(?:\.\d+)?\)?)+")
DATE_LIKE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")
PHONE_LIKE = re.compile(r"(?:\+?\d[\s-]?){7,}\d")
VERSION_LIKE = re.compile(r"\b[a-z]*\d+(?:\.\d+){2,}\b", re.I)


class UnsupportedExpression(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsupportedExpression("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise UnsupportedExpression(
                    f"exponent above {MAX_EXPONENT} is not supported")
            if abs(left) > 1e6:
                raise UnsupportedExpression("base too large for a safe power")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise UnsupportedExpression(
        f"{type(node).__name__} is not allowed in an arithmetic expression")


def evaluate_expression(expression: str) -> float:
    """Evaluate a numeric expression without `eval`."""
    text = expression.strip()
    if not text or not _EXPR_CHARS.match(text):
        raise UnsupportedExpression("expression contains unsupported characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise UnsupportedExpression(f"invalid syntax: {exc.msg}") from exc
    value = _eval_node(tree)
    if not math.isfinite(value):
        raise UnsupportedExpression("result is not finite")
    if abs(value) > MAX_ABS_RESULT:
        raise UnsupportedExpression("result is out of the supported range")
    return value


def format_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def _numbers_in(text: str) -> int:
    return len(re.findall(r"\d+(?:\.\d+)?", text))


def _looks_non_arithmetic(text: str) -> str | None:
    """Return a reason when the text merely *contains* digits."""
    if DATE_LIKE.search(text):
        return "looks like a date"
    if PHONE_LIKE.search(text):
        return "looks like a phone number or identifier"
    if VERSION_LIKE.search(text):
        return "looks like a version string"
    return None


def _percentage(text: str) -> Route | None:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%\s*(of|off|của|giảm)\s*"
                  r"(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        pct, mode, base = float(m.group(1)), m.group(2).lower(), float(m.group(3))
        if mode in ("off", "giảm"):
            value = base * (1 - pct / 100.0)
            return Route(text, INTENT_PERCENT, "calculator",
                         {"percent": pct, "op": "off", "base": base},
                         STATUS_OK, value, format_number(value),
                         f"{format_number(pct)}% off {format_number(base)} = "
                         f"{format_number(value)}")
        value = base * pct / 100.0
        return Route(text, INTENT_PERCENT, "calculator",
                     {"percent": pct, "op": "of", "base": base},
                     STATUS_OK, value, format_number(value),
                     f"{format_number(pct)}% of {format_number(base)} = "
                     f"{format_number(value)}")
    # "17 % 5" is modulo, not a percentage: a percent sign followed by a number
    # is an operator here, only a trailing percent sign is ambiguous.
    if re.search(r"\d\s*%(?!\s*\d)", text):
        # "50 + 10%" has no defined base: refuse instead of inventing one.
        return Route(text, INTENT_PERCENT, "calculator", {"raw": text},
                     STATUS_AMBIGUOUS, None, None,
                     "a bare percentage needs a base (use '15% of 240')")
    return None


# ── unit conversion ─────────────────────────────────────────────────────────
# factor = value in the base unit of its dimension
UNITS = {
    "km": ("length", 1000.0), "kilometre": ("length", 1000.0),
    "kilometer": ("length", 1000.0), "m": ("length", 1.0), "meter": ("length", 1.0),
    "metre": ("length", 1.0), "cm": ("length", 0.01), "mm": ("length", 0.001),
    "mile": ("length", 1609.344), "miles": ("length", 1609.344),
    "mi": ("length", 1609.344), "ft": ("length", 0.3048), "feet": ("length", 0.3048),
    "foot": ("length", 0.3048), "inch": ("length", 0.0254),
    "inches": ("length", 0.0254), "yd": ("length", 0.9144),
    "kg": ("mass", 1.0), "kilogram": ("mass", 1.0), "g": ("mass", 0.001),
    "gram": ("mass", 0.001), "lb": ("mass", 0.45359237),
    "lbs": ("mass", 0.45359237), "pound": ("mass", 0.45359237),
    "pounds": ("mass", 0.45359237), "oz": ("mass", 0.028349523125),
    "l": ("volume", 1.0), "litre": ("volume", 1.0), "liter": ("volume", 1.0),
    "ml": ("volume", 0.001), "gal": ("volume", 3.785411784),
    "gallon": ("volume", 3.785411784),
    "hour": ("time", 3600.0), "hours": ("time", 3600.0), "h": ("time", 3600.0),
    "minute": ("time", 60.0), "minutes": ("time", 60.0), "min": ("time", 60.0),
    "second": ("time", 1.0), "seconds": ("time", 1.0), "s": ("time", 1.0),
    "day": ("time", 86400.0), "days": ("time", 86400.0),
}
TEMPERATURE = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
_CONVERT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([a-z°]+)\s*(?:->|to|in|sang|bằng)\s*([a-z°]+)", re.I)


def _convert_temperature(value: float, src: str, dst: str) -> float:
    src = "c" if src.startswith("c") else "f" if src.startswith("f") else "k"
    dst = "c" if dst.startswith("c") else "f" if dst.startswith("f") else "k"
    celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[src]
    return {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[dst]


def _conversion(text: str) -> Route | None:
    m = _CONVERT_RE.search(text)
    if not m:
        return None
    value, src, dst = float(m.group(1)), m.group(2).lower().lstrip("°"), m.group(3).lower().lstrip("°")
    if src in TEMPERATURE and dst in TEMPERATURE:
        result = _convert_temperature(value, src, dst)
        return Route(text, INTENT_CONVERT, "unit_converter",
                     {"value": value, "from": src, "to": dst}, STATUS_OK, result,
                     format_number(round(result, 6)),
                     f"{format_number(value)} {src} = {format_number(round(result, 6))} {dst}")
    if src in UNITS and dst in UNITS:
        dim_s, factor_s = UNITS[src]
        dim_d, factor_d = UNITS[dst]
        if dim_s != dim_d:
            return Route(text, INTENT_CONVERT, "unit_converter",
                         {"value": value, "from": src, "to": dst},
                         STATUS_UNSUPPORTED, None, None,
                         f"{src} and {dst} measure different things")
        result = value * factor_s / factor_d
        return Route(text, INTENT_CONVERT, "unit_converter",
                     {"value": value, "from": src, "to": dst}, STATUS_OK, result,
                     format_number(round(result, 6)),
                     f"{format_number(value)} {src} = {format_number(round(result, 6))} {dst}")
    return Route(text, INTENT_CONVERT, "unit_converter",
                 {"value": value, "from": src, "to": dst}, STATUS_UNSUPPORTED, None, None,
                 f"unknown unit {src!r} or {dst!r}")


# ── routing ─────────────────────────────────────────────────────────────────
def route(request: str, *, sources: list[dict] | None = None, k: int = 3) -> Route:
    started = time.perf_counter()
    text = normalise(request)
    body = strip_framing(text)

    try:
        if body in ("", "?"):
            return Route(request, INTENT_NONE, None, status=STATUS_NO_TOOL,
                         explanation="empty request")

        non_arithmetic = _looks_non_arithmetic(body)
        percent = _percentage(body)
        conversion = _conversion(body) if re.search(
            r"\b(to|in|sang)\b", body, re.I) else None

        if conversion is not None and conversion.status == STATUS_OK:
            conversion.request = request
            conversion.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            return conversion

        if percent is not None:
            percent.request = request
            percent.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            return percent

        # Prose around a number is not an arithmetic request. "1 + 1 = 2 is a
        # true statement" is a statement about arithmetic, not a request for it.
        # Only relevant when no retrieval sources are supplied: with sources a
        # prose question is exactly what retrieval is for.
        if not sources and re.search(r"[^\W\d_]{2,}", body, re.UNICODE):
            return Route(request, INTENT_NONE, None, {"reason": "prose"},
                         STATUS_NO_TOOL, None, None,
                         "request contains prose, not a bare expression",
                         latency_ms=round((time.perf_counter() - started) * 1000, 3))

        if re.fullmatch(r"[\d\s+\-*/%().]+", body) and _numbers_in(body) >= 2 \
                and re.search(r"\*\*|[-+*/%]", body):
            expression, candidate_match = body, None
        else:
            candidate_match = EXPR_CANDIDATE.search(body)
            expression = candidate_match.group(0).strip() if candidate_match else ""
        has_operator = bool(expression)

        if non_arithmetic and not (candidate_match and has_operator
                                   and candidate_match.group(0) == body):
            return Route(request, INTENT_NONE, None, {"reason": non_arithmetic},
                         STATUS_NO_TOOL, None, None,
                         f"not arithmetic: {non_arithmetic}",
                         latency_ms=round((time.perf_counter() - started) * 1000, 3))

        if has_operator:
            if expression != body and _numbers_in(body) != _numbers_in(expression):
                return Route(request, INTENT_CALCULATOR, "calculator",
                             {"expression": expression}, STATUS_AMBIGUOUS, None, None,
                             "more than one arithmetic reading is possible",
                             latency_ms=round((time.perf_counter() - started) * 1000, 3))
            try:
                value = evaluate_expression(expression)
            except ZeroDivisionError:
                return Route(request, INTENT_CALCULATOR, "calculator",
                             {"expression": expression}, STATUS_ERROR, None, None,
                             "division by zero", executor="calculator",
                             latency_ms=round((time.perf_counter() - started) * 1000, 3))
            except UnsupportedExpression as exc:
                return Route(request, INTENT_CALCULATOR, "calculator",
                             {"expression": expression}, STATUS_UNSUPPORTED, None, None,
                             str(exc), executor="calculator",
                             latency_ms=round((time.perf_counter() - started) * 1000, 3))
            value = int(value) if float(value).is_integer() else value
            return Route(request, INTENT_CALCULATOR, "calculator",
                         {"expression": expression}, STATUS_OK, value,
                         format_number(value), f"{expression} = {format_number(value)}",
                         executor="calculator",
                         latency_ms=round((time.perf_counter() - started) * 1000, 3))

        if conversion is not None:
            conversion.request = request
            conversion.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            return conversion

        if sources:
            from grounding import select_evidence
            selected, issues = select_evidence(body, sources, k=k)
            if selected:
                return Route(request, INTENT_RETRIEVAL, "retrieval",
                             {"query": body, "k": k}, STATUS_OK, None, None,
                             f"{len(selected)} evidence passage(s) selected",
                             executor="retrieval",
                             evidence=[{"citation_id": s.get("citation_id"),
                                        "source": s.get("title") or s.get("id"),
                                        "text": s.get("text", "")[:400],
                                        "score": s.get("retrieval_score")}
                                       for s in selected],
                             latency_ms=round((time.perf_counter() - started) * 1000, 3))
            return Route(request, INTENT_RETRIEVAL, "retrieval", {"query": body, "k": k},
                         STATUS_UNSUPPORTED, None, None,
                         "no evidence above the coverage threshold; abstain",
                         executor="retrieval", latency_ms=round(
                             (time.perf_counter() - started) * 1000, 3))

        return Route(request, INTENT_NONE, None, status=STATUS_NO_TOOL, explanation=(
            "no supported tool: arithmetic needs an expression with two numbers"),
            latency_ms=round((time.perf_counter() - started) * 1000, 3))
    finally:
        pass


# ── benchmark cases ─────────────────────────────────────────────────────────
def supported_cases() -> list[tuple[str, float]]:
    """200+ expressions with exact expected values (generated, deterministic)."""
    cases: list[tuple[str, float]] = []
    pairs = [(2, 3), (7, 8), (12, 12), (45, 9), (100, 25), (9, 3), (14, 6),
             (33, 11), (8, 4), (60, 15), (17, 5), (81, 9), (24, 8), (55, 5),
             (120, 40), (6, 7), (90, 10), (13, 4), (48, 6), (200, 8)]
    for a, b in pairs:
        cases.append((f"{a} + {b}", a + b))
        cases.append((f"{a} - {b}", a - b))
        cases.append((f"{a} * {b}", a * b))
        cases.append((f"{a} / {b}", a / b))
        cases.append((f"({a} + {b}) * 2", (a + b) * 2))
        cases.append((f"{a} * {b} - {a}", a * b - a))
        cases.append((f"{a} + {b} * 2", a + b * 2))
        cases.append((f"({a} - {b}) / 2", (a - b) / 2))
        cases.append((f"what is {a} + {b}?", a + b))
        cases.append((f"calculate {a} * {b}", a * b))
        cases.append((f"tính {a} + {b}", a + b))
    for base, exp in ((2, 10), (3, 4), (5, 3), (2, 8), (10, 3), (4, 5)):
        cases.append((f"{base}^{exp}", base ** exp))
        cases.append((f"{base} ** {exp}", base ** exp))
    for a, b in ((17, 5), (29, 7), (100, 9), (45, 4)):
        cases.append((f"{a} % {b}", a % b))
        cases.append((f"{a} // {b}", a // b))
    for txt, val in (("15% of 240", 36.0), ("25% of 80", 20.0), ("10% of 55", 5.5),
                     ("50% of 130", 65.0), ("5% of 900", 45.0),
                     ("20% off 250", 200.0), ("10% off 90", 81.0),
                     ("30% của 400", 120.0), ("12.5% of 400", 50.0),
                     ("75% of 128", 96.0)):
        cases.append((txt, val))
    for txt, val in (("5 km to miles", 5 * 1000 / 1609.344),
                     ("2 kg to lb", 2 / 0.45359237),
                     ("100 c to f", 212.0),
                     ("32 f to c", 0.0),
                     ("1 hour to minutes", 60.0),
                     ("3 ft to inches", 36.0),
                     ("1500 m to km", 1.5),
                     ("2 l to ml", 2000.0)):
        cases.append((txt, val))
    return cases


NEGATIVE_CASES = [
    "12/03/2024", "on 2024-03-12 what happened", "090-123-4567",
    "call +84 90 123 4567", "GPT-2 is a model", "v1.2.3", "ISO-8601 dates",
    "what is the capital of France?", "explain transformers", "who wrote Hamlet",
    "1 + 1 = 2 is a true statement", "open the file named report-2024",
    "what is JIN AI", "tell me a story",
]
AMBIGUOUS_CASES = ["50 + 10%", "20% more than 30", "what is 5"]
INVALID_CASES = ["1/0", "5 +", "((3+4)", "sqrt(9)", "2 ^ 99999", "3 + * 4"]


def run_router_benchmark() -> dict:
    cases = supported_cases()
    router_ok = 0
    arithmetic_ok = 0
    failures: list[dict] = []
    for text, expected in cases:
        r = route(text)
        if r.intent in (INTENT_CALCULATOR, INTENT_PERCENT, INTENT_CONVERT) and \
                r.status == STATUS_OK:
            router_ok += 1
            if abs(float(r.value) - float(expected)) <= 1e-6:
                arithmetic_ok += 1
            else:
                failures.append({"request": text, "expected": expected,
                                 "got": r.value})
        else:
            failures.append({"request": text, "expected": expected,
                             "got": f"{r.intent}/{r.status}"})

    negatives_ok = sum(1 for t in NEGATIVE_CASES if route(t).intent in (INTENT_NONE,))
    ambiguous_ok = sum(1 for t in AMBIGUOUS_CASES
                       if route(t).status in (STATUS_AMBIGUOUS, STATUS_NO_TOOL))
    invalid_handled = 0
    for text in INVALID_CASES:
        r = route(text)
        if r.status in (STATUS_ERROR, STATUS_UNSUPPORTED, STATUS_NO_TOOL):
            invalid_handled += 1

    return {
        "supported_cases": len(cases),
        "router_accuracy": round(router_ok / len(cases), 4),
        "arithmetic_accuracy": round(arithmetic_ok / len(cases), 4),
        "negative_cases": len(NEGATIVE_CASES),
        "negative_rejection_rate": round(negatives_ok / len(NEGATIVE_CASES), 4),
        "ambiguous_cases": len(AMBIGUOUS_CASES),
        "ambiguous_handled_rate": round(ambiguous_ok / len(AMBIGUOUS_CASES), 4),
        "invalid_cases": len(INVALID_CASES),
        "invalid_handled_rate": round(invalid_handled / len(INVALID_CASES), 4),
        "failures": failures[:20],
        "credited_to_model": 0,
        "attribution": "router + calculator; the base model generated none of these",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 17 deterministic intent router")
    ap.add_argument("request", nargs="?", help="a user request to route")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.request:
        print(json.dumps(route(args.request).to_dict(), ensure_ascii=False, indent=2))
        return
    report = run_router_benchmark()
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
