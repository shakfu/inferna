"""Tests for the safe ``calculator`` tool in inferna.agents.tools.

The calculator is implemented as an AST allowlist (not ``eval``), so the
tests focus on three things:

1. Correctness on legal arithmetic.
2. Rejection of every disallowed Python construct (names, calls,
   attribute access, comparisons, comprehensions, etc.).
3. Resource caps: expression length and exponent magnitude.
"""

from __future__ import annotations

import pytest

from inferna.agents.tools import calculator


def _call(expr: str) -> str:
    """Invoke the underlying callable; ``calculator`` is a Tool wrapper."""
    return calculator(expression=expr)


class TestCalculatorCorrectness:
    def test_integer_addition(self):
        assert _call("2 + 3") == "5"

    def test_integer_subtraction(self):
        assert _call("10 - 4") == "6"

    def test_integer_multiplication(self):
        assert _call("6 * 7") == "42"

    def test_true_division_returns_float(self):
        # 10 / 4 -> 2.5 (Python's ``/`` is true division)
        assert _call("10 / 4") == "2.5"

    def test_floor_division(self):
        assert _call("10 // 4") == "2"

    def test_modulo(self):
        assert _call("10 % 3") == "1"

    def test_power(self):
        assert _call("2 ** 10") == "1024"

    def test_unary_plus(self):
        assert _call("+5") == "5"

    def test_unary_minus(self):
        assert _call("-5") == "-5"

    def test_double_unary_minus(self):
        assert _call("--5") == "5"

    def test_parentheses_and_precedence(self):
        assert _call("(2 + 3) * 4") == "20"
        assert _call("2 + 3 * 4") == "14"

    def test_float_literals(self):
        assert _call("1.5 + 2.5") == "4.0"

    def test_mixed_int_and_float(self):
        assert _call("2 * 1.5") == "3.0"


class TestCalculatorRejectsUnsafeConstructs:
    """Every rejected construct should raise ValueError -- not eval, not crash."""

    def test_rejects_name_lookup(self):
        with pytest.raises(ValueError):
            _call("x + 1")

    def test_rejects_function_call(self):
        with pytest.raises(ValueError):
            _call("abs(-5)")

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError):
            _call("(1).bit_length")

    def test_rejects_subscript(self):
        with pytest.raises(ValueError):
            _call("[1, 2, 3][0]")

    def test_rejects_comparison(self):
        with pytest.raises(ValueError):
            _call("1 < 2")

    def test_rejects_boolean_op(self):
        with pytest.raises(ValueError):
            _call("1 and 2")

    def test_rejects_lambda(self):
        with pytest.raises(ValueError):
            _call("(lambda: 1)()")

    def test_rejects_conditional_expression(self):
        with pytest.raises(ValueError):
            _call("1 if True else 2")

    def test_rejects_list_comprehension(self):
        with pytest.raises(ValueError):
            _call("[i for i in range(3)]")

    def test_rejects_string_literal(self):
        with pytest.raises(ValueError, match="disallowed type"):
            _call("'hello'")

    def test_rejects_bytes_literal(self):
        with pytest.raises(ValueError, match="disallowed type"):
            _call("b'x'")

    def test_rejects_bitwise_or(self):
        # Bitwise ops are deliberately not in _CALC_BINOPS.
        with pytest.raises(ValueError, match="disallowed binary operator"):
            _call("1 | 2")

    def test_rejects_bitshift(self):
        with pytest.raises(ValueError, match="disallowed binary operator"):
            _call("1 << 2")

    def test_rejects_bitwise_not(self):
        with pytest.raises(ValueError, match="disallowed unary operator"):
            _call("~1")


class TestCalculatorErrorReporting:
    def test_syntax_error_becomes_value_error(self):
        with pytest.raises(ValueError, match="invalid arithmetic expression"):
            _call("2 +")

    def test_empty_expression(self):
        with pytest.raises(ValueError):
            _call("")


class TestCalculatorResourceCaps:
    def test_rejects_overlong_expression(self):
        # 200 is the documented limit.
        expr = "1+" * 150 + "1"  # 451 chars, well over the cap
        with pytest.raises(ValueError, match="expression too long"):
            _call(expr)

    def test_at_length_limit_is_accepted(self):
        # Build an arithmetic expression of exactly 200 chars.
        body = "1+" * 99 + "1"  # 99*2 + 1 = 199 chars
        assert len(body) == 199
        # Sum of 100 ones.
        assert _call(body) == "100"

    def test_rejects_huge_exponent(self):
        with pytest.raises(ValueError, match="exponent"):
            _call("2 ** 100000")

    def test_negative_huge_exponent_also_rejected(self):
        with pytest.raises(ValueError, match="exponent"):
            _call("2 ** -100000")

    def test_exponent_at_cap_is_accepted(self):
        # 2 ** 1000 is large but instant. Just verify it doesn't raise.
        result = _call("2 ** 1000")
        assert result.startswith("1")  # 2**1000 starts with "10715..."
        assert int(result) > 0
