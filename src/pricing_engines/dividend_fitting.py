import numpy as np

class DividendFitter:

    def __init__(self, tenors, dividends):

        # Sort by tenor
        sorted_pairs = sorted(zip(tenors, dividends))
        self.tenors = np.array([t for t, _ in sorted_pairs])
        self.dividends = np.array([d for _, d in sorted_pairs])

    def forecast(self, tenor):
        tenor = np.atleast_1d(np.asarray(tenor, dtype=float))
        result = np.empty_like(tenor, dtype=float)

        below = tenor <= self.tenors[0]
        above = tenor >= self.tenors[-1]
        inside = ~(below | above)

        result[below]  = self.dividends[0]
        result[above]  = self.dividends[-1]
        if inside.any():
            result[inside] = np.interp(tenor[inside], self.tenors, self.dividends)

        return float(result[0]) if result.size == 1 else result