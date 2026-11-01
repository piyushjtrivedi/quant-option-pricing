import numpy as np
from scipy.stats import norm

class BlackScholes:
    @staticmethod
    def bs_price(S, K, r, q, T, sigma, flag):
        if T < 1e-10 or sigma < 1e-10:
            # At expiry: return intrinsic value
            if flag.lower() == 'c':
                return float(max(0.0, S - K))
            else:
                return float(max(0.0, K - S))
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if flag.lower() == 'c':
            return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

    @staticmethod
    def bs_vega(S, K, r, q, T, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q*T)* norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def bs_call_prices_from_iv_grid(S, K_grid, T, r, q, sigma_grid):
        sigma = np.maximum(sigma_grid, 1e-12)
        sqrtT = np.sqrt(max(T, 1e-12))

        d1 = (np.log(S/K_grid) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrtT)
        d2 = d1 - sigma*sqrtT

        return S*np.exp(-q*T)*norm.cdf(d1) - K_grid*np.exp(-r*T)*norm.cdf(d2)