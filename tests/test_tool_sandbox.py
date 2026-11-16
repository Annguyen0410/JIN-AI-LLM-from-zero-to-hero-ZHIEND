"""
Phase 6 tests — tool safety, sandbox escapes, agent ReAct loop.

Covers:
- calculator: correctness (precedence, functions, right-assoc power),
  safety (no eval, unknown symbols, division by zero, depth cap).
- python_exec sandbox: import whitelist, network blocked, file writes
  blocked, normal execution, timeout.
- agent loop: parse_tool_call (incl. nested-dict args), tool-call
  execution, max_tool_calls budget, honest stop, injected generator.
- self-check verifier: math re-check, citation requirement (high),
  self-consistency, fast = no checks.
- memory tools: search + write against a real MemoryStore.

Run:  python -m unittest tests.test_tool_sandbox -v   (stdlib, CPU)
"""

from __future__ import annotations

import unittest

from agent import Agent, _trim_at_stop, parse_tool_call
from memory import MemoryStore
from tools import (
    CalculatorError,
    calculate,
    execute_tool,
    memory_search,
    memory_write,
    python_exec,
)


class TestCalculator(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(calculate("2+3*4"), "14")
        self.assertEqual(calculate("(2+3)*4"), "20")
        self.assertEqual(calculate("10/4"), "2.5")
        self.assertEqual(calculate("2**10"), "1,024")
        self.assertEqual(calculate("17 % 5"), "2")

    def test_right_associative_power(self):
        # 2**3**2 = 2**(3**2) = 512, NOT (2**3)**2 = 64.
        self.assertEqual(calculate("2**3**2"), "512")

    def test_unary_minus_and_constants(self):
        self.assertEqual(calculate("-3 + 5"), "2")
        self.assertEqual(calculate("2 * pi"), "6.2831853072")
        self.assertEqual(calculate("-2**2"), "-4")

    def test_functions(self):
        self.assertEqual(calculate("sqrt(16)"), "4")
        self.assertEqual(calculate("max(3, 7, 2)"), "7")
        self.assertEqual(calculate("floor(3.7)"), "3")
        self.assertEqual(calculate("round(2.5)"), "2")  # banker's rounding

    def test_errors_never_eval(self):
        with self.assertRaises(CalculatorError):
            calculate("__import__('os').system('echo hi')")
        with self.assertRaises(CalculatorError):
            calculate("1/0")
        with self.assertRaises(CalculatorError):
            calculate("foo(3)")
        with self.assertRaises(CalculatorError):
            calculate("2 +")  # dangling operator
        with self.assertRaises(CalculatorError):
            calculate("2 ** (3")  # unbalanced parens
        with self.assertRaises(CalculatorError):
            calculate("x" * 600)  # overlong

    def test_no_builtins_leak(self):
        with self.assertRaises(CalculatorError):
            calculate("open('/tmp/x')")
        with self.assertRaises(CalculatorError):
            calculate("len([1,2])")


class TestPythonExecSandbox(unittest.TestCase):
    def test_normal_execution(self):
        out = python_exec("print(1+1)\nprint('hello')")
        self.assertIn("2", out)
        self.assertIn("hello", out)

    def test_import_whitelist_blocks_os_subprocess(self):
        for code in ("import os", "import subprocess", "import sys",
                     "import socket", "import requests"):
            out = python_exec(code + "\nprint('leaked')")
            self.assertNotIn("leaked", out, f"{code} must not run")
            self.assertIn("blocked", out, f"{code} must be rejected")

    def test_allowed_imports_still_work(self):
        out = python_exec(
            "import math, json\nprint(math.sqrt(9))\nprint(json.dumps({'a': 1}))")
        self.assertIn("3.0", out)
        self.assertIn('{"a": 1}', out)

    def test_network_blocked(self):
        out = python_exec(
            "import socket\ns = socket.socket()\n"
            "s.connect(('example.com', 80))\nprint('leaked')")
        self.assertNotIn("leaked", out)
        self.assertIn("blocked", out)

    def test_file_writes_blocked_reads_allowed(self):
        out = python_exec(
            "open('C:/temp/jin_sandbox_probe.txt', 'w').write('x')\nprint('wrote')")
        self.assertNotIn("wrote", out)
        self.assertIn("disabled", out)

    def test_timeout_kills_hang(self):
        with self.assertRaises(RuntimeError):
            python_exec("while True: pass", timeout=1)

    def test_exception_reported_not_crash(self):
        out = python_exec("1/0")
        self.assertIn("ZeroDivisionError", out)

    def test_os_system_blocked_via_whitelist(self):
        out = python_exec("import os\nos.system('echo pwned')")
        self.assertIn("blocked", out)


class TestParseToolCall(unittest.TestCase):
    def test_plain_call(self):
        text = '<|tool|>{"name": "calculator", "args": {"expression": "2+2"}}<|end|>'
        name, args, _end = parse_tool_call(text)
        self.assertEqual(name, "calculator")
        self.assertEqual(args, {"expression": "2+2"})

    def test_nested_dict_args_not_truncated(self):
        # The inner {} of ``args`` must not truncate the JSON.
        text = ('prefix <|tool|>{"name": "memory_search", '
                '"args": {"query": "capital of France", "k": 3}}<|end|> tail')
        name, args, _end = parse_tool_call(text)
        self.assertEqual(name, "memory_search")
        self.assertEqual(args, {"query": "capital of France", "k": 3})

    def test_unknown_tool_rejected(self):
        text = '<|tool|>{"name": "rm_rf", "args": {}}<|end|>'
        self.assertIsNone(parse_tool_call(text))

    def test_malformed_json_rejected(self):
        self.assertIsNone(parse_tool_call('<|tool|>{not json}<|end|>'))
        self.assertIsNone(parse_tool_call("no tool call here"))

    def test_braces_inside_strings_ignored(self):
        text = '<|tool|>{"name": "web_search", "args": {"query": "a {b} c"}}<|end|>'
        name, args, _end = parse_tool_call(text)
        self.assertEqual(args, {"query": "a {b} c"})


class TestToolRegistry(unittest.TestCase):
    def test_now(self):
        out = execute_tool("now", {}, {})
        self.assertIn("20", out[:4])  # year prefix

    def test_unknown_tool_returns_error_string(self):
        out = execute_tool("nope", {}, {})
        self.assertIn("error", out.lower())

    def test_calculator_via_registry(self):
        out = execute_tool("calculator", {"expression": "6*7"}, {})
        self.assertEqual(out, "42")

    def test_search_disabled_by_policy(self):
        out = execute_tool("web_search", {"query": "x"}, {"search_enabled": False})
        self.assertIn("disabled", out)

    def test_memory_write_and_search(self):
        store = MemoryStore(path=None)
        memory_write(store, "The capital of France is Paris.",
                     source_url="https://example.com/france", confidence=0.9)
        res = memory_search(store, "capital of France")
        self.assertIn("Paris", res)
        self.assertIn("https://example.com/france", res)
        # No match -> honest empty.
        self.assertIn("no memory", memory_search(store, "quantum computing"))


class TestAgentLoop(unittest.TestCase):
    def _agent(self, script: list[str], max_tool_calls=3, rigor="fast"):
        """script[i] is the text returned on the i-th generation call."""
        calls = {"n": 0}

        def gen(prompt_ids, max_new_tokens=None):
            text = script[min(calls["n"], len(script) - 1)]
            calls["n"] += 1
            return text

        return Agent(generate_fn=gen, max_tool_calls=max_tool_calls,
                     rigor=rigor, tokenizer=None)

    def test_tool_then_final_answer(self):
        agent = self._agent([
            '<|tool|>{"name": "calculator", "args": {"expression": "6*7"}}<|end|>',
            "The answer is 42.",
        ])
        out = agent.run("What is 6 times 7?")
        self.assertEqual(out["answer"], "The answer is 42.")
        types = [e["type"] for e in out["trace"]]
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        self.assertIn("final", types)
        # Calculator actually ran.
        result = next(e["text"] for e in out["trace"]
                      if e["type"] == "tool_result")
        self.assertEqual(result, "42")

    def test_max_tool_calls_enforced(self):
        script = ['<|tool|>{"name": "now", "args": {}}<|end|>'] * 10
        agent = self._agent(script, max_tool_calls=2, rigor="fast")
        out = agent.run("keep calling tools")
        n_calls = sum(1 for e in out["trace"] if e["type"] == "tool_call")
        self.assertEqual(n_calls, 2)
        self.assertIn("ran out of tool calls", out["answer"])

    def test_no_tool_call_returns_directly(self):
        agent = self._agent(["Just a plain answer."])
        out = agent.run("hi")
        self.assertEqual(out["answer"], "Just a plain answer.")
        self.assertEqual([e["type"] for e in out["trace"]], ["final"])

    def test_stream_cb_events(self):
        agent = self._agent([
            '<|tool|>{"name": "calculator", "args": {"expression": "1+1"}}<|end|>',
            "It is 2.",
        ])
        events: list[str] = []
        out = agent.run("1+1?", stream_cb=lambda ev: events.append(ev["type"]))
        self.assertEqual(events, ["tool_call", "tool_result", "final"])


class TestSelfCheck(unittest.TestCase):
    def test_fast_skips_checks(self):
        agent = Agent(rigor="fast", tokenizer=None)
        self.assertEqual(agent.self_check("What is 2+2?", "Four."), [])

    def test_math_mismatch_detected(self):
        agent = Agent(rigor="normal", tokenizer=None)
        issues = agent.self_check("What is 2+2?", "The answer is 5.")
        self.assertTrue(any("calculator" in i for i in issues))

    def test_math_correct_passes(self):
        agent = Agent(rigor="normal", tokenizer=None)
        self.assertEqual(agent.self_check("What is 2+2?", "It is 4."), [])

    def test_high_requires_citation_for_factual(self):
        agent = Agent(rigor="high", tokenizer=None)
        issues = agent.self_check(
            "Who is the current president of France?", "Emmanuel Macron.")
        self.assertTrue(any("citation" in i for i in issues))
        # With a citation -> clean.
        issues = agent.self_check(
            "Who is the current president of France?",
            "Emmanuel Macron [1]. https://example.com")
        self.assertFalse(any("citation" in i for i in issues))

    def test_low_self_consistency_flagged(self):
        # The resample (first _generate call inside self_check) returns a
        # completely different answer -> low jaccard -> flagged.
        agent = self._consistency_agent("completely different words here")
        issues = agent.self_check("some query", "first answer one two three four")
        self.assertTrue(any("self-consistency" in i for i in issues))

    def test_high_consistency_agrees_passes(self):
        agent = self._consistency_agent("first answer one two three four")
        issues = agent.self_check("some query", "first answer one two three four")
        self.assertFalse(any("self-consistency" in i for i in issues))

    def _consistency_agent(self, resample: str):
        def gen(prompt_ids, max_new_tokens=None):
            return resample

        return Agent(generate_fn=gen, rigor="high", tokenizer=None)


class TestTrimStop(unittest.TestCase):
    def test_trims_end_tags(self):
        self.assertEqual(_trim_at_stop("hello <|end|> world"), "hello")
        self.assertEqual(_trim_at_stop("a <|endoftext|> b"), "a")
        self.assertEqual(_trim_at_stop("no tags"), "no tags")


if __name__ == "__main__":
    unittest.main()