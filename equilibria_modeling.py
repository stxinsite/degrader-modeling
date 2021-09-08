import numpy as np
import pandas as pd
import scipy
from math import sqrt
from lmfit import minimize, Parameters

"""
ABBREVIATIONS:
- E: E3 ligase
- D: degrader
- T: target protein

INDEPENDENT VARIABLES:
- D

DEPENDENT VARIABLES:
- EDT

PARAMETERS:
- E_t
- T_t
- KAPPA
- alpha
- BETA

CONSTANTS:
- K_ED
- K_DT
"""

nanobret_df = pd.read_csv("data/2021-08-26_nano_bret_test.reformed.csv")
nanobret_df['mBU'] = nanobret_df['Lum_610'] / nanobret_df['Lum_450'] * 1000
nanobret_df.head()

EDT = .00000000005
alpha = 500
E_t = 0.00000000005
D_t = 0.00000796
T_t = 0.00001
K_ED = 0.0000018
K_DT = 0.00000025

def ternary_quintic( D_in, EDT, E_t, T_t, K_ED, K_DT, alpha, beta, kappa):
    D_t = kappa * D_in

    (
    EDT**5 * (alpha - 1) * alpha**2 - \
    EDT**4 * alpha * (alpha**2 * (2*E_t + D_t + 2*T_t) + \
                      2*alpha * (K_ED + K_DT - E_t - T_t) - \
                      2 * (K_ED + K_DT)) + \
    EDT**3 * (alpha**3 * (E_t**2 + 2*E_t * (D_t + 2*T_t) + T_t * (2*D_t + T_t)) + \
              alpha**2 * (-E_t * (D_t + 3*T_t - 2*K_ED - 3*K_DT) + \
                          K_DT * (D_t + 2*T_t + K_ED) - \
                          E_t**2 + D_t**2 - D_t*T_t - T_t**2 + \
                          D_t*K_ED + 3*T_t*K_ED) + \
              alpha * (K_DT**2 - 2*K_DT*(E_t + T_t + K_ED) + \
                       K_ED*(K_ED - 2*(E_t + T_t))) - (K_ED - K_DT)**2) - \
    EDT**2 * alpha *(alpha**2 * (D_t*T_t**2 + 2*E_t*T_t*(2*D_t+T_t) + E_t**2 *(D_t + 2*T_t)) - \
                     (E_t*(D_t + T_t) + T_t*(T_t - D_t - K_ED))*K_ED - (E_t**2 + T_t*(D_t + K_ED) + E_t*(T_t - D_t + K_ED))*K_DT + \
                     alpha * (-E_t**2 * (D_t + T_t - K_DT) + \
                              E_t * (D_t**2 + D_t*(K_ED + K_DT - 2*T_t) - T_t**2 + 3*T_t*(K_ED + K_DT) + K_ED*K_DT) + \
                              T_t * (D_t**2 + D_t*(K_ED + K_DT - T_t) + K_ED*(T_t + K_DT))) + \
                     E_t * K_DT**2) + \
    EDT * alpha**2 * E_t * T_t * (D_t**2 + \
                                  D_t*(2*alpha*T_t + K_ED + K_DT - T_t) + \
                                  E_t * ((2*alpha - 1)*D_t + alpha*T_t + K_DT) + \
                                  K_ED*(T_t + K_DT)) - alpha**3 * E_t**2 * D_t * T_t**2
    ) / alpha

    return

def fit_ABC( data_obs, K_ED, K_DT, alpha = 1., beta = 0.5, kappa = 0.5):
    """
    Fit equilibrium binding parameters to NanoBRET data.

    Args:

    data_obs: 2-d array; data_obs[:,0] gives the degrader concentrations and
    data_obs[:,1] gives the corresponding observed response units (mBU).

    K_ED: float; constant of dissociation for binary complex of E3 ligase
    and degrader.

    K_DT: float; constant of dissociation for binary complex of target protein
    and degrader.

    alpha, beta, kappa: floats; initial guesses for the values of the fitting
    parameters.

    Returns:

    results: fitting results of the binding parameters.

    results.params contains the fitted parameters for alpha, beta, kappa.
    """

    def residual( params, data_obs, K_ED, K_DT):
        EDT = params['EDT']
        alpha = params['alpha']
        beta = params['beta']
        kappa = params['kappa']

        y_pred = ternary_quintic( data_obs[:,0], K_ED, K_DT, alpha, beta, kappa)

        return data_obs[:,1] - y_pred

    params = Parameters()

    params.add( 'EDT', value = EDT min = 0., max = min(E_t, D_t, T_t))
    params.add( 'alpha', value = alpha, min = 1.)
    params.add( 'beta', value = beta, min = 0., max = 1.)
    params.add( 'kappa', value = kappa, min = 0., max = 1.)

    result = minimize( residual, params, args = (y_obs, K_ED, K_DT),
                       method = 'least_squares', nan_policy = 'omit')

    return result

# COOPERATIVE EQUILIBRIA

# ## S18
# E = K_ED * (E_t - EDT) / (K_ED + D)
# ## S19
# T = K_DT * (T_t - EDT) / (K_DT + D)
# ## S21
# EDT**2 - EDT*((E_t + T_t + (K_ED + D)*(K_DT + D)) / (alpha * D)) + E_t * T_t = 0
#
# ## S38
# ED * DT = (E * D**2 * T) / (K_ED * K_DT)
#
# ## S164
# D_t = ((E_t + T_t - K_ED - K_DT +
#     (alpha*(EDT - E_t)*(EDT - T_t) / EDT) +
#     (E_t * (K_DT - K_ED) / (alpha*(EDT - E_t))) +
#     (T_t * (K_ED - K_DT) / (alpha*(EDT - T_t))) +- # subtracted form represents left side of curve, summed form right side
#     (alpha / EDT + 1 / (E_t - EDT) + 1 / (T_t - EDT)) *
#     sqrt((alpha*(EDT - E_t)*(EDT - T_t) - EDT * K_ED)**2 -
#          2*EDT*K_DT*(alpha*(EDT-E_t)*(EDT-T_t) + EDT*K_ED) +
#          EDT**2 * K_DT**2
#          )) / alpha
# ) / 2

# # NON-COOPERATIVE EQUILIBRIA
# PHI_ED = (E_t + B_t + K_ED - sqrt((E_t + D_t + K_ED)**2 - 4 * E_t * D_t)) / 2
# PHI_DT = (T_t + B_t + K_DT - sqrt((T_t + D_t + K_DT)**2 - 4 + T_t + D_t)) / 2
#
# EDT = PHI_ED * PHI_DT / D_t
