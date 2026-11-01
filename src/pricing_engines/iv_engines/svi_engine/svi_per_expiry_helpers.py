"""
Per-Slice SVI Helpers
=====================
Stateless helpers for raw SVI: the formula, parameter encoding, and
no-arbitrage penalties. Mirrors the SSVIHelpers pattern.
"""

import math
import numpy as np


class SSVIPEHelpers:

    # Raw-space clip: a, b, sigma ∈ [e^-30, e^10], m ∈ [-1, 1].
    # Far wider than any realistic value; prevents math.exp overflow.
    RAW_CLIP: float = 10.0
    M_CLIP:   float = 1.0

    # ── SVI formula ───────────────────────────────────────────────────────

    @staticmethod
    def svi_w(k: np.ndarray, a: float, b: float,
              rho: float, m: float, sigma: float) -> np.ndarray:
        """
        Raw SVI total variance  w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).

        a: level · b: wing slope · rho: skew · m: horizontal shift · sigma: ATM curvature
        """
        k    = np.asarray(k, dtype=float)
        diff = k - m
        return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))

    # ── parameter encoding ────────────────────────────────────────────────

    @staticmethod
    def pack(a: float, b: float, rho: float,
             m: float, sigma: float) -> np.ndarray:
        """(a, b, rho, m, sigma) → unconstrained x: log for a, b, sigma; atanh for rho."""
        return np.array([
            math.log(max(a, 1e-12)),
            math.log(max(b, 1e-12)),
            math.atanh(max(-0.999, min(0.999, rho))),
            m,
            math.log(max(sigma, 1e-12)),
        ])

    @staticmethod
    def unpack(x: np.ndarray):
        """Unconstrained x → (a, b, rho, m, sigma), clipping before exp()."""
        a     = math.exp(float(np.clip(x[0], -30, SSVIPEHelpers.RAW_CLIP)))
        b     = math.exp(float(np.clip(x[1], -30, SSVIPEHelpers.RAW_CLIP)))
        rho   = math.tanh(float(x[2]))
        m     = float(np.clip(x[3], -SSVIPEHelpers.M_CLIP, SSVIPEHelpers.M_CLIP))
        sigma = math.exp(float(np.clip(x[4], -30, SSVIPEHelpers.RAW_CLIP)))
        return a, b, rho, m, sigma

    # ── no-arbitrage penalties ────────────────────────────────────────────

    @staticmethod
    def butterfly_penalty(b: float, rho: float) -> float:
        """
        Soft penalty on the SVI wing slope b·(1+|rho|) above 1.

        Note: this is a wing (Lee moment) bound, not a butterfly condition, and
        Lee's bound on total-variance wing slope is 2, not 1. TODO: revisit.
        """
        val = b * (1.0 + abs(rho))
        return 1e4 * max(0.0, val - 1.0) ** 2

    @staticmethod
    def calendar_penalty(k_cal: np.ndarray, a: float, b: float, rho: float,
                         m: float, sigma: float, prev_fits: list) -> float:
        """
        Soft calendar penalty: 50 · mean(max(0, w_prev − w_curr)²) over k_cal,
        summed over earlier slices (prev_fits).

        Note: too weak to bind — realistic violations cost ~1e-5 against a
        vol-pt MSE of O(1) — and it can only push later slices up. TODO: replace
        with a hard constraint or a joint fit.
        """
        if not prev_fits:
            return 0.0
        w_curr = SSVIPEHelpers.svi_w(k_cal, a, b, rho, m, sigma)
        pen = 0.0
        for pf in prev_fits:
            w_prev     = pf.w(k_cal)
            violations = np.maximum(0.0, w_prev - w_curr)
            pen += 50.0 * float(np.mean(violations ** 2))
        return pen
