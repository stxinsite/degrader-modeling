import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from math import sqrt
from lmfit import minimize, Parameters

"""
ABBREVIATIONS:
- A: E3 ligase (target protein)
- B: degrader
- C: target protein (E3 ligase)
"""

nanobret_df = pd.read_csv("data/2021-08-26_nano_bret_test.reformed.csv")
nanobret_df['mBU'] = nanobret_df['Lum_610'] / nanobret_df['Lum_450'] * 1000
nanobret_df.head()

"""
GLOBAL CONSTANTS
"""
A_t = 5e-11
C_t = 1e-5
K_AB = 1.8e-6
K_BC = 2.5e-7
alpha = 10

B_tmax = sqrt(K_BC) / (sqrt(K_AB) + sqrt(K_BC)) * (A_t + K_AB) + sqrt(K_AB) / (sqrt(K_AB) + sqrt(K_BC)) * (C_t + K_BC)
TF_50 = A_t / 2 + K_AB / alpha
TI_50 = C_t * (1 + alpha)

def equilibrium_sys(params, *constants):
    A_t, C_t, K_AB, K_BC, alpha, B_t = constants
    A = params[0]
    C = params[1]
    ABC = params[2]

    F = np.empty((3))
    F[0] = A + K_BC * ABC / (alpha * C) + ABC - A_t
    F[1] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
    F[2] = C + K_AB * ABC / (alpha * A) + ABC - C_t

    return F

init_params = np.array([A_t, C_t, 0.5*min(A_t, C_t)])  # initial guesses for roots
sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_tmax)

roots = fsolve(equilibrium_sys, init_params, args=sys_args)  # solutions at B_tmax
print(roots)
np.isclose(equilibrium_sys(roots, *sys_args), [0.0, 0.0, 0.0])  # checks whether system is close to 0 at solutions

sys_args = (A_t, C_t, K_AB, K_BC, alpha, TF_50)
roots_1 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_1)

# [ABC] at TF_50 should be approximately 0.5 * [ABC]_max
roots_1[2] / roots[2]

## for B_t > B_tmax, fsolve() is sensitive to the inital parameters
sys_args = (A_t, C_t, K_AB, K_BC, alpha, TI_50)
roots_2 = fsolve(equilibrium_sys, np.array([A_t * 1e-2, C_t * 1e-2, 0.5*min(A_t, C_t)]), args=sys_args)
print(roots_2)
np.isclose(equilibrium_sys(roots_2, *sys_args), [0.0, 0.0, 0.0])  # checks whether system is close to 0 at solutions

# [ABC] at TI_50 should be approximately 0.5 * [ABC]_max
roots_2[2] / roots[2]

# [ABC] at TF_50 and TI_50 should be approximately equal
np.isclose(roots_1[2], roots_2[2], rtol=1e-6)

"""NON-COOPERATIVE EQUILIBRIA"""
def noncoop_equilibrium(B_t):
    A = A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

    phi_AB = A_t - A
    phi_BC = C_t - C
    ABC = phi_AB * phi_BC / B_t

    return np.array([A, C, ABC])

"""
Check that non-coop(B_t) = coop(B_t | alpha = 1)
"""
noncoop_equilibrium(B_tmax)
noncoop_equilibrium(B_t = 1.12e-6)
noncoop_equilibrium(B_t = 2.45e-05)

sys_args = (A_t, C_t, K_AB, K_BC, 1, B_tmax)
roots_3 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_3)

sys_args = (A_t, C_t, K_AB, K_BC, 1, 1.12e-6)
roots_4 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_4)

sys_args = (A_t, C_t, K_AB, K_BC, 1, 2.45e-05)
roots_5 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_5)


def fit_equilibrium_sys(data_obs, K_AB, K_BC, alpha = 1., beta = 0.5, kappa = 0.5):
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

    def residual(params, data_obs, K_AB, K_BC):
        A_t = params['A_t']
        C_t = params['C_t']
        alpha = params['alpha']
        beta = params['beta']
        kappa = params['kappa']

        B_t = kappa * data_obs[:,0]

        init_params = np.array([A_t, C_t, 0.5*min(A_t, C_t)])  # initial guesses for roots
        sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_t)
        roots = fsolve(equilibrium_sys, init_params, args=sys_args)

        ABC_pred = roots[2]
        y_pred = ABC_pred * beta

        return data_obs[:,1] - y_pred

    params = Parameters()

    params.add( 'A_t', value = A_t, min = 0.)
    params.add( 'C_t', value = C_t, min = 0.)
    params.add( 'alpha', value = alpha, min = 1.)
    params.add( 'beta', value = beta, min = 0.)
    params.add( 'kappa', value = kappa, min = 0., max = 1.)

    result = minimize( residual, params, args = (data_obs, K_AB, K_BC),
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
