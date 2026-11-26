"""Exact Choquet upper expectation on a finite space (for testing and teaching)."""
from fractions import Fraction


def choquet_finite(w, f):
    """Upper expectation of f under the possibility distribution w (sequences)."""
    levels = sorted({Fraction(0), *map(Fraction, w)})
    total = Fraction(0)
    for x, y in zip(levels, levels[1:]):
        total += (y - x) * max(v for p, v in zip(w, f) if Fraction(p) >= y)
    return total
