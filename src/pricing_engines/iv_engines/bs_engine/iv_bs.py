import numpy as np
from src.pricing_engines.bs_pricer import BlackScholes

class BSIV:

    @staticmethod
    def _initial_guess(price, S, K, r, q, t, flag):
        """
        Brenner-Subrahmanyam (1988) ATM approximation as seed.
        sigma ≈ price / (S * sqrt(T / 2π))  for ATM options.
        Works well as a universal seed; keeps Newton converging fast.
        """
        F = S * np.exp((r - q) * t)
        atm_vega_approx = F * np.exp(-r * t) * np.sqrt(t / (2 * np.pi))
        if atm_vega_approx < 1e-12:
            return 0.3
        sigma = price / atm_vega_approx
        return float(np.clip(sigma, 1e-4, 4.0))

    @staticmethod
    def compute(price, S, K, r, q, t, flag, tol=1e-6, max_iter=500):
        """
        Newton-Raphson with bisection fallback.
        Returns NaN if IV cannot be determined (not 0 — callers must filter NaN).
        """
        if t < 1e-10:
            return np.nan

        # -----------------------------
        # No-arbitrage bounds check
        # -----------------------------
        disc_S = S * np.exp(-q * t)
        disc_K = K * np.exp(-r * t)

        if flag.lower() == "c":
            intrinsic = max(0.0, disc_S - disc_K)
            upper = disc_S
        else:
            intrinsic = max(0.0, disc_K - disc_S)
            upper = disc_K

        # Price must have positive time value; below intrinsic or above theoretical max → no IV
        if price <= intrinsic or price >= upper:
            return np.nan

        # -----------------------------
        # Volatility bracket
        # -----------------------------
        vol_low = 1e-6
        vol_high = 5.0

        f_low  = BlackScholes.bs_price(S, K, r, q, t, vol_low,  flag) - price
        f_high = BlackScholes.bs_price(S, K, r, q, t, vol_high, flag) - price

        if f_low * f_high > 0:
            return np.nan

        # Better initial guess → fewer Newton iterations
        sigma = BSIV._initial_guess(price, S, K, r, q, t, flag)

        # -----------------------------
        # Newton-Raphson + bisection fallback
        # -----------------------------
        for _ in range(max_iter):
            model_price = BlackScholes.bs_price(S, K, r, q, t, sigma, flag)
            vega        = BlackScholes.bs_vega(S, K, r, q, t, sigma)
            diff        = model_price - price

            if abs(diff) < tol:
                return sigma

            # Newton step when vega is reliable
            if vega > 1e-8:
                sigma_new = sigma - diff / vega
                if vol_low < sigma_new < vol_high:
                    sigma = sigma_new
                    continue

            # Bisection fallback
            if diff * f_low < 0:
                vol_high = sigma
                f_high   = diff
            else:
                vol_low = sigma
                f_low   = diff

            sigma = 0.5 * (vol_low + vol_high)

        return np.nan  # Failed to converge → NaN, not 0