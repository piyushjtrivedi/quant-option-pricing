import numpy as np
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from scipy.optimize import least_squares

class NSSCurveFitter:

    @staticmethod    
    def fit_curve(maturities, zero_rates):
        # Convert Pandas Series to NumPy arrays
        maturities = np.asarray(maturities, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)

        # Initial parameters
        beta0 = zero_rates[-1]
        beta1 = -0.02
        beta2 = 0.02
        tau1 = 0.2
        tau2 = 1.0
        
        initial_params = [beta0, beta1, beta2, 0.01, tau1, tau2]

        def nss_residuals(params):
            curve = NelsonSiegelSvenssonCurve(*params)
            fitted = curve(maturities)
            return fitted - zero_rates

        result = least_squares(nss_residuals, initial_params)
        return NelsonSiegelSvenssonCurve(*result.x)

    
