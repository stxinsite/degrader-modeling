import numpy as np
import pandas as pd
from scipy.optimize import fsolve, root
from math import sqrt
from lmfit import Minimizer, minimize, fit_report, Parameters
import matplotlib.pyplot as plt

"""
ABBREVIATIONS:
- A: E3 ligase (target protein)
- B: degrader
- C: target protein (E3 ligase)
"""

nanobret_df = pd.read_csv("data/2021-08-26_nano_bret_test.reformed.csv")
nanobret_df['Time'] = pd.to_datetime(nanobret_df['Time'])
nanobret_df['mBU'] = nanobret_df['Lum_610'] / nanobret_df['Lum_450'] * 1000
nanobret_df['B_x'] = nanobret_df['uM'] * 1e-6  # convert uM (micromolar) to molar units
nanobret_df.head()

nanobret_df['Construct'].unique().tolist()

nanobret_df['Time'].unique()

"""GENERAL SOLUTION (COOPERATIVE EQUILIBRIA)"""
def equilibrium_sys(params, *constants):
    """"System of equations describes concentrations at equilibrium"""
    A_t, C_t, K_AB, K_BC, alpha, B_t = constants
    A = params[0]
    C = params[1]
    ABC = params[2]

    F = np.empty((3))
    F[0] = A + K_BC * ABC / (alpha * C) + ABC - A_t
    F[1] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
    F[2] = C + K_AB * ABC / (alpha * A) + ABC - C_t

    return F

def f(params, B_t, *constants):
    """"System of equations describes concentrations at equilibrium"""
    A_t, C_t, K_AB, K_BC, alpha = constants
    A = params[0]
    C = params[1]
    ABC = params[2]

    F = np.empty((3))
    F[0] = A + K_BC * ABC / (alpha * C) + ABC - A_t
    F[1] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
    F[2] = C + K_AB * ABC / (alpha * A) + ABC - C_t
    return F

def jac_equilibrium(params, *constants):
    A_t, C_t, K_AB, K_BC, alpha, B_t = constants
    A = params[0]
    C = params[1]
    ABC = params[2]

    return np.array([[1, -K_BC*ABC/(alpha*C**2), K_BC/(alpha*C) + 1],
                     [-K_AB*K_BC*ABC/(alpha*A**2*C)-K_AB*ABC/(alpha*A**2),
                      -K_AB*K_BC*ABC/(alpha*A*C**2)-K_BC*ABC/(alpha*C**2),
                      K_AB*K_BC/(alpha*A*C)+K_BC/(alpha*C)+K_AB/(alpha*A) + 1],
                     [-K_AB*ABC/(alpha*A**2), 1, K_AB/(alpha*A) + 1]])

def noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_t):
    """NON-COOPERATIVE EQUILIBRIUM"""
    A = A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

    phi_AB = A_t - A
    phi_BC = C_t - C
    ABC = phi_AB * phi_BC / B_t

    return np.array([A, C, ABC])

def residual(params, data_obs, K_AB, K_BC):
    A_t = params['A_t']
    C_t = params['C_t']
    alpha = params['alpha']
    beta = params['beta']
    kappa = params['kappa']

    B_t = np.multiply(data_obs[:,0], kappa)  # B_t = kappa * B_x

    ABC_pred = []  # predicted [ABC] that satisfies equilibrium system
    init_params = np.array([A_t*0.3, C_t*0.3, 1])  # initial guesses for [A], [C], [ABC]
    for B_i in B_t:
        sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_i)
        roots = root(equilibrium_sys, init_params, jac=jac_equilibrium, args=sys_args)
        # print(roots)
        # print(np.isclose(equilibrium_sys(roots, *sys_args), [0.0, 0.0, 0.0]).all())
        ABC_pred.append(roots.x[2])

    y_pred = np.multiply(np.array(ABC_pred), beta)  # ^mBU = ^[ABC] * beta
    resid = data_obs[:,1] - y_pred
    squared_resid = np.square(resid)
    # print(squared_resid.mean())
    return resid

def df_dx(params, data_obs, K_AB, K_BC, y):
    A_t = params['A_t']
    C_t = params['C_t']
    alpha = params['alpha']
    beta = params['beta']
    kappa = params['kappa']

    return np.array([[1, 0, K_BC*ABC/(alpha**2*C), 0],
                     [0, 0, K_AB*K_BC*ABC/(alpha**2*A*C)+K_AB*ABC/(alpha**2*A)+K_BC*ABC/(alpha**2*C), B_x],
                     -, 1, K_AB*ABC/(alpha**2*A), 0])

solve_ABC()

# def residual(params, data, K_AB, K_BC):
#         A_t = params['A_t']
#         C_t = params['C_t']
#         alpha = params['alpha']
#         beta = params['beta']
#         kappa = params['kappa']
#         B_t = kappa * data[:,0]
#
#         solve_ABC(A_t, C_t, K_AB, K_BC, alpha, B_t)
#         return ABC_pred - data[:,1]

def fit_equilibrium_sys(data_obs, K_AB, K_BC, A_t=1., C_t=1., alpha=1., beta=1., kappa=0.5):
    """
    Fit equilibrium binding parameters to NanoBRET data.

    Args:

    data_obs: 2-d array; data_obs[:,0] gives the concentrations of degrader (B_x) and
    data_obs[:,1] gives the corresponding observed response units (mBU).

    K_AB, K_BC: floats; constants of dissociation for binary complex of E3 ligase (target protein)
    and degrader.

    alpha, beta, kappa: floats; initial guesses for the values of the fitting
    parameters.

    Returns:

    results: fitting results of the binding parameters.

    results.params contains the fitted parameters for A_t, C_t, alpha, beta, kappa.
    """
    params = Parameters()

    params.add( 'A_t', value = A_t, min = 0.)
    params.add( 'C_t', value = C_t, min = 0.)
    params.add( 'alpha', value = alpha, min = 0.)
    params.add( 'beta', value = beta, min = 0.)
    params.add( 'kappa', value = kappa, min = 0., max = 1.)

    # def mse(params, iter, resid, *fcn_args, **fcn_kws):
    #     squared_resid = np.square(resid)
    #     return squared_resid.mean()

    result = minimize( residual, params, args = (data_obs, K_AB, K_BC),
                       method = 'leastsq', nan_policy = 'omit')
    return result

"""
UNIT TESTING
units in micromolar
"""
A_t = 1
B_x = np.array([0.0, 0.05, 0.1, 0.5, 1, 1.5, 3, 5])
C_t = 2
alpha = 1.5
kappa = 0.75
K_AB = 250e-3
K_BC = 1800e-3
beta = 5
B_t = np.multiply(B_x, kappa)
init_params = np.array([0.5*A_t, 0.5*C_t, 1])  # initial guesses for [A], [C], [ABC]
ABC = []  # predicted [ABC] that satisfies equilibrium system
for B_i in B_t:
    sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_i)
    roots = fsolve(equilibrium_sys, init_params, args=sys_args)
    print(roots)
    print(np.isclose(equilibrium_sys(roots, *sys_args), [0.0, 0.0, 0.0]).all())
    ABC.append(roots[2])

mBU = np.multiply(np.array(ABC), beta)  # mBU = [ABC] * beta
plt.plot(B_x, mBU, label = 'simulated', color = 'cyan')
plt.xscale('log')
plt.legend()
plt.title('Simulated mBU data')
plt.xlabel('B_t')
plt.show()

true_data = np.stack((B_x, mBU), axis=1)
# fit with init vals = true vals
fit_equilibrium_sys(data_obs = true_data, K_AB=K_AB, K_BC=K_BC, A_t=A_t, C_t=C_t, alpha=alpha, beta=beta, kappa=kappa)


# good fit
fit_equilibrium_sys(data_obs = true_data, K_AB=K_AB, K_BC=K_BC, A_t=5, C_t=5, alpha=3, beta=5, kappa=0.5)

fit_equilibrium_sys(data_obs=true_data, K_AB=K_AB, K_BC=K_BC, A_t=7, C_t=10, alpha=2, beta=5, kappa=0.5)

fit_equilibrium_sys(data_obs=true_data, K_AB=K_AB, K_BC=K_BC, A_t=10, C_t=20, alpha=2, beta=10, kappa=0.5)

# init vals for A_t, C_t greater than true vals by factor of 5 is unstable
res_4 = fit_equilibrium_sys(data_obs=true_data, K_AB=K_AB, K_BC=K_BC, A_t=10, C_t=10, alpha=2, beta=5, kappa=0.5)
v = res_4.params
res_5 = fit_equilibrium_sys(data_obs=true_data, K_AB=K_AB, K_BC=K_BC,
                            A_t = v['A_t'].value, C_t = v['C_t'].value, alpha=v['alpha'].value,
                            beta=v['beta'].value, kappa=v['kappa'].value)
res_5
# very poor fit
res_2 = fit_equilibrium_sys(data_obs = true_data, K_AB=K_AB, K_BC=K_BC, A_t=A_t*10, C_t=C_t*10, alpha=1, beta=1, kappa=0.5)
res_2

# ## S18
# E = K_ED * (E_t - EDT) / (K_ED + D)
# ## S19
# T = K_DT * (T_t - EDT) / (K_DT + D)
#
# ## S38
# ED * DT = (E * D**2 * T) / (K_ED * K_DT)
