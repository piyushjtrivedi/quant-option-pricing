import pandas as pd
import numpy as np
from src.pricing_engines.bs_pricer import BlackScholes

class SSVIHelpers:

    @staticmethod
    def ssvi_w(k, theta, phi, rho):
        '''
        Vectorised SSVI Total variance for one T slice
        k : arrayed log-moneyness
        returns : w(k) as numpy array
        '''
        k = np.asarray(k, dtype=float)
        x = phi*k + rho
        inside = x*x + 1.0 - rho*rho
        return 0.5*theta*(1+rho*phi*k+np.sqrt(inside))
    
    @staticmethod
    def ssvi_iv_from_ssvi_params(k_grid,T, theta, phi, rho):
        w = np.maximum(SSVIHelpers.ssvi_w(k_grid,theta, phi, rho), 1e-12)
        return np.sqrt(w/T)
    
    @staticmethod
    def local_iv_from_ssvi_params(k_grid,T, theta, theta_dot, phi, rho):

        one_minus_rho2 = max(1e-16, 1.0 - rho*rho)
        a = phi*k_grid + rho
        u = a*a + one_minus_rho2
        u = np.maximum(u, 1e-20)
        s = np.sqrt(u)

        G = 0.5*(1.0 + rho*phi*k_grid + s)

        w_row = np.maximum(G*theta, 1e-12)
        iv_row = np.sqrt(w_row/T)

        # derivative wrt k:
        # w_k = theta/2 * phi* (rho + a/s)
        w_k_row = (theta * 0.5)*phi*(rho + (a /s))

        #w_kk_row = theta/2 * phi^2 * (1 - rho^2) / u^(3/2)
        w_kk_row = (theta*0.5)*(phi**2)*(one_minus_rho2)/(u**1.5)

        # time derivative: w_t = theta_dot*G
        w_t_row = theta_dot*G

        return w_row, iv_row, w_k_row, w_kk_row, w_t_row

    
    @staticmethod
    def positive_density_on_grid(k_grid, w_grid, T, F, S, r, q, tol=-1e-12):
        '''
        Check discrete second derivative of call price => tolerance on k grid
        Returns True if pass
        '''
        K_grid = F*np.exp(k_grid)
        iv_grid = np.sqrt(np.maximum(w_grid/max(T,1e-12),0.0))
        C = BlackScholes.bs_call_prices_from_iv_grid(S, K_grid, T, r, q, iv_grid)
        # Central finite-difference second derivative in k
        K = K_grid; C = C
        for i in range(1, len(K)-1):
            k1, k0, k_1 = K[i+1], K[i], K[i-1]
            if (k1 - k0) <= 0 or (k0 - k_1) <= 0:
                return False
            s1 = (C[i+1] - C[i]) / (k1 - k0)
            s0 = (C[i] - C[i-1]) / (k0 - k_1)
            d2 = 2.0 * (s1 - s0) / (k1 - k_1)
            if d2 < tol:
                return False
        return True

    @staticmethod
    def softplus_np(x):
        return np.log1p(np.exp(x))
    
    @staticmethod
    def pack_to_theta(deltas):
        inc = np.log1p(np.exp(deltas))
        return np.cumsum(inc)
    
 





    