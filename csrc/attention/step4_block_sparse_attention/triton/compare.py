"""Comparison reporting for the parity harness.

Relative error is only meaningful where the expected value is above a floor.
Post-ReLU indexer logits are exactly zero for a large fraction of entries, so a
naive ``diff / expected`` reports 1e8 on cases that pass cleanly -- a metric that
trains the reader to ignore it. Report it over the above-floor subset only, and
report the exact-zero fraction separately, since that fraction is what determines
how much the top-k tie-break rule matters.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

REL_FLOOR = 1e-3


class Comparison(NamedTuple):
    passed: bool
    max_abs: float
    max_rel: float
    rel_count: int
    zero_fraction: float
    nonfinite: int

    def format(self) -> str:
        rel = f"{self.max_rel:.3e}" if self.rel_count else "n/a"
        text = (
            f"max_abs={self.max_abs:.3e}  "
            f"max_rel={rel} (n={self.rel_count})  "
            f"zeros={self.zero_fraction:.1%}"
        )
        if self.nonfinite:
            text += f"  nonfinite={self.nonfinite}(matched)"
        return text


def compare(
    actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float
) -> Comparison:
    actual_f = actual.float()
    expected_f = expected.float()

    # ``lse`` is legitimately -inf wherever a query selected nothing, and -inf minus -inf
    # is NaN. Reporting that NaN as the error statistic would blank out the metric on
    # exactly the rows the empty-selection cases exist to exercise, so the difference is
    # measured over the finite subset and the non-finite entries are counted instead.
    # ``allclose`` still decides pass/fail, and it treats matching infinities as equal
    # while rejecting an infinity paired with a finite value or a NaN.
    finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    diff = (actual_f - expected_f).abs()[finite]

    above_floor = expected_f.abs() >= REL_FLOOR
    rel_count = int((above_floor & finite).sum())
    if rel_count:
        subset = above_floor & finite
        max_rel = float(((actual_f - expected_f).abs()[subset] / expected_f[subset].abs()).max())
    else:
        max_rel = float("nan")

    return Comparison(
        passed=torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol),
        max_abs=float(diff.max()) if diff.numel() else 0.0,
        max_rel=max_rel,
        rel_count=rel_count,
        zero_fraction=float((expected_f == 0).float().mean()),
        nonfinite=int((~finite).sum()),
    )


class UlpComparison(NamedTuple):
    passed: bool
    max_abs: float
    bound: float
    peak: float
    exceeding: int
    total: int

    def format(self) -> str:
        ratio = self.max_abs / self.bound if self.bound else float("inf")
        return (
            f"max_abs={self.max_abs:.3e} bound={self.bound:.3e} ({ratio:.2f}x)  "
            f"peak={self.peak:.3g}  over_tenth_bound={self.exceeding}/{self.total}"
        )


BF16_RELATIVE_ULP = 2.0**-7


def compare_scaled(
    actual: torch.Tensor, expected: torch.Tensor, *, ulps: float = 1.0
) -> UlpComparison:
    """Compare two bfloat16 results against one ulp of the tensor's *peak* magnitude.

    Neither a relative tolerance nor an ulp distance works for this stage, and both
    failure modes are instructive:

    * **Relative error** blows up under cancellation. RoPE computes
      ``real * cos - imag * sin``; where those products nearly cancel, an operand
      difference of one ulp becomes a relative error of several hundred percent on a
      result that is itself near zero.
    * **Ulp distance** blows up for the same elements, because bfloat16 has an enormous
      number of representable values near zero -- two tiny results of opposite sign are
      tens of thousands of ulps apart while differing by a millionth in absolute terms.

    What *is* bounded is the absolute error relative to the magnitude of the operands
    that went in. Using the tensor's peak magnitude as a stand-in for that scale gives a
    single honest statement: "the two agree to within N ulps of the largest value
    present". ``over_tenth_bound`` counts elements past a tenth of that band, so a
    systematic bias cannot hide behind a bound sized for the worst case.
    """
    actual_f = actual.float()
    expected_f = expected.float()
    error = (actual_f - expected_f).abs()
    peak = float(expected_f.abs().max()) if expected_f.numel() else 0.0
    bound = ulps * BF16_RELATIVE_ULP * peak
    max_abs = float(error.max()) if error.numel() else 0.0
    return UlpComparison(
        passed=max_abs <= bound,
        max_abs=max_abs,
        bound=bound,
        peak=peak,
        exceeding=int((error > bound / 10.0).sum()),
        total=int(error.numel()),
    )
