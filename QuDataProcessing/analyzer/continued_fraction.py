import numpy as np


def _continued_fraction(x, acc=1e-6):
    """Find the continued fraction expansion of x with accuracy acc"""
    a = np.floor(x)
    b = x - a
    cf = [a]
    x0 = x
    while abs(x0 - _evaluate_cf(cf)) > acc:
        x = 1 / b
        a = np.floor(x)
        b = x - a
        cf.append(a)
    return cf


def _cf_to_fraction(cf):
    """Convert a continued fraction to a fraction"""
    if len(cf) == 1:
        return cf[0], 1
    else:
        a, b = _cf_to_fraction(cf[1:])
        return cf[0] * a + b, a


def _evaluate_cf(cf):
    """Evaluate the continued fraction."""
    x, y = _cf_to_fraction(cf)
    return x / y


def float_to_fraction(x, acc=1e-6):
    """Converts an input float to a fraction with accuracy acc, using the continued fraction method"""
    cf = _continued_fraction(x, acc)
    a, b = _cf_to_fraction(cf)
    return a, b


def integer_multiplier(x, max_m=100):
    """finds the integer in range(1, max_m) that multiplies x to closest to an integer"""
    m = np.arange(1, max_m + 1, dtype=int)
    xm = x * m
    xm_residue = np.abs(xm - np.round(xm))
    return m[np.argmin(xm_residue)]
