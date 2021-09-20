import numpy as np
import pandas as pd
import random as rand
from scipy.optimize import fsolve, root, check_grad
from scipy.linalg import solve
from math import sqrt, exp, log
from lmfit import Minimizer, minimize, fit_report, Parameters
from collections import OrderedDict
import matplotlib.pyplot as plt

"""
ABBREVIATIONS:
- A: E3 ligase (target protein)
- B: degrader
- C: target protein (E3 ligase)
"""

"""GENERAL SOLUTION (COOPERATIVE EQUILIBRIA)"""
def equilibrium_sys(variables, *constants):
    """"System of equations describes concentrations at equilibrium"""
    A_t, C_t, K_AB, K_BC, alpha, B_t = constants
    A = variables[0]
    C = variables[1]
    ABC = variables[2]

    F = np.empty((3))
    F[0] = A + K_BC * ABC / (alpha * C) + ABC - A_t
    F[1] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
    F[2] = C + K_AB * ABC / (alpha * A) + ABC - C_t

    return F

def jac_equilibrium(variables, *constants):
    A_t, C_t, K_AB, K_BC, alpha, B_t = constants
    A = variables[0]
    C = variables[1]
    ABC = variables[2]
    df_dy = solve_dfdy(A, C, ABC, K_AB, K_BC, alpha)
    return df_dy

def solve_dfdy(A, C, ABC, K_AB, K_BC, alpha):
    return np.array([[ 1, -K_BC * ABC / ( alpha * C**2 ), K_BC / ( alpha * C ) + 1 ],
                     [ -K_AB * K_BC * ABC / ( alpha * A**2 * C ) -K_AB * ABC / ( alpha * A**2 ),
                       -K_AB * K_BC * ABC / ( alpha * A * C**2 ) -K_BC * ABC / ( alpha * C**2 ),
                       K_AB * K_BC / ( alpha * A * C ) +K_BC / ( alpha * C ) +K_AB / ( alpha * A ) + 1 ],
                     [ -K_AB * ABC / ( alpha * A**2 ), 1, K_AB / ( alpha * A ) + 1 ]])

def solve_dABCdx(x, y, B_x):
    """solves for dy/dx"""
    A_t, C_t, alpha, kappa = x
    A, C, ABC = y

    df_dy = solve_dfdy(A, C, ABC, K_AB, K_BC, alpha)
    neg_df_dx = np.array([[ 1, 0, K_BC * ABC / ( alpha**2 * C ), 0 ],
                          [ 0, 0, K_AB * K_BC *ABC / ( alpha**2 * A * C ) +K_AB * ABC / ( alpha**2 * A ) +K_BC * ABC / ( alpha**2 * C ), B_x ],
                          [ 0, 1, K_AB * ABC / ( alpha**2 * A ), 0 ]])

    dy_dx = solve(df_dy, neg_df_dx)  # [ d[ABC]/d[A]_t d[ABC]/d[C]_t d[ABC]/dalpha d[ABC]/dkappa ] is the third row
    return dy_dx[2,:]

def noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_t):
    """NON-COOPERATIVE EQUILIBRIUM"""
    A = A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

    phi_AB = A_t - A
    phi_BC = C_t - C
    if B_t == 0:
        ABC = 0
    else:
        ABC = phi_AB * phi_BC / B_t
    return np.array([A, C, ABC])

def residual(params, *constants, data, log_conc=False):
    if log_conc:
        A_t = np.exp(params['log_A_t'])
        C_t = np.exp(params['log_C_t'])
    else:
        A_t = params['A_t']
        C_t = params['C_t']
    alpha = params['alpha']
    kappa = params['kappa']
    beta = params['beta']

    B_x = data[:,0]
    B_t = kappa * B_x

    K_AB, K_BC = constants

    equilibrium_solutions = np.zeros((len(B_x), 3))  # predicted roots ([A],[C],[ABC]) that satisfy equilibrium system
    for i, B_i in np.ndenumerate(B_t):
        init_guess = noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_i)  # initial guesses for [A], [C], [ABC]
        sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_i)
        roots = root(equilibrium_sys, init_guess, jac=jac_equilibrium, args=sys_args)
        equilibrium_solutions[i] = roots.x

    pred = beta * equilibrium_solutions[:,2]  # predicted response_i = beta * [ABC]_i
    resid = pred - data[:,1]  # residuals = predicted - observed
    return resid

def residual_jacobian(params, *constants, data, log_conc=False):
    if log_conc:
        A_t = exp(params['log_A_t'])
        C_t = exp(params['log_C_t'])
    else:
        A_t = params['A_t']
        C_t = params['C_t']
    alpha = params['alpha']
    kappa = params['kappa']
    beta = params['beta']

    B_x = data[:,0]
    B_t = kappa * B_x

    K_AB, K_BC = constants

    dL_dx = np.empty((len(B_x), len(params)))  # derivatives of residual function w.r.t. params
    for i, B_i in np.ndenumerate(B_t):
        init_guess = noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_i)  # initial guesses for [A], [C], [ABC]
        sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_i)
        roots = root(equilibrium_sys, init_guess, jac=jac_equilibrium, args=sys_args)
        A, C, ABC = roots.x  # predicted roots that satisfy equilibrium system
        dABC_dx = beta * solve_dABCdx(x=(A_t, C_t, alpha, kappa), y=(A, C, ABC), B_x=B_x[i])
        dL_dx[i] = np.append(dABC_dx, ABC)
    return dL_dx

def fit_equilibrium_sys(data, K_AB, K_BC, A_t=1., C_t=1., alpha=1., kappa=0.5, beta=1., log_conc=False):
    """
    Fit equilibrium binding parameters to experimental data.

    Args:

    data_obs: 2-d array; data_obs[:,0] gives the extracellular concentrations of degrader (B_x)
    and data_obs[:,1] gives the corresponding observed response units (mBU).

    K_AB, K_BC: floats; constants of dissociation for binary complex of E3 ligase (A)
    or target protein (C) and degrader (B).

    A_t, C_t: floats: initial values for concentrations of E3 ligase and target protein.

    alpha, beta, kappa: floats; initial values for model parameters.

    log_conc: boolean; if true, A_t and C_t will be log-transformed parameters.

    Returns:

    results: MinimizerResult object with fitting results of the binding parameters.
    results.params contains the fitted parameters for A_t, C_t, alpha, beta, kappa.

    sse: sum of squared residuals.
    """
    params = Parameters()
    if log_conc:
        params.add( 'log_A_t', value = log(A_t))
        params.add( 'log_C_t', value = log(C_t))
    else:
        params.add( 'A_t', value = A_t, min = 0., max= 500)
        params.add( 'C_t', value = C_t, min = 0., max= 500)
    params.add( 'alpha', value = alpha, min = 0.)
    params.add( 'kappa', value = kappa, min = 0., max = 1.)
    params.add( 'beta', value = beta, min = 0.)

    # result = minimize( residual, params, args = (, data_obs),
    #                    method = 'leastsq', nan_policy = 'omit')
    min = Minimizer(residual, params, fcn_args=(K_AB, K_BC), fcn_kws={'data': data, 'log_conc': log_conc})
    if log_conc:
        out = min.leastsq()
    else:
        out = min.leastsq(Dfun=residual_jacobian)

    sse = np.sum(np.square(out.residual))
    pred = data[:,1] + out.residual

    # plt.plot(data[:,0], data[:,1], label = 'simulated', color = 'cyan')
    # plt.plot(data[:,0], pred, label = 'predicted', color = 'coral')
    # plt.xscale('log')
    # plt.legend()
    # plt.show()

    return out, sse

def create_params_dict(arr):
    params = OrderedDict()
    params['A_t'] = arr[0]
    params['C_t'] = arr[1]
    params['alpha'] = arr[2]
    params['kappa'] = arr[3]
    params['beta'] = arr[4]
    return params

def residual_wrapper(arr, K_AB, K_BC, data, data_idx):
    params = create_params_dict(arr)
    res = residual(params, K_AB, K_BC, data=data)
    if data_idx >= len(data):
        return None
    else:
        return res[data_idx]

def residual_jacobian_wrapper(arr, K_AB, K_BC, data, data_idx):
    params = create_params_dict(arr)
    res = residual_jacobian(params, K_AB, K_BC, data=data)
    if data_idx >= len(data):
        return None
    else:
        return res[data_idx,:]

"""
UNIT TESTING
units in micromolar
"""
A_t = 1
B_x = np.linspace(0, 5, 16)
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
    # print(roots)
    # print(np.isclose(equilibrium_sys(roots, *sys_args), [0.0, 0.0, 0.0]).all())
    ABC.append(roots[2])

mBU = np.multiply(np.array(ABC), beta)  # mBU = [ABC] * beta
plt.plot(B_x, mBU, label = 'simulated', color = 'cyan')
plt.xscale('log')
plt.legend()
plt.title('Simulated mBU data')
plt.xlabel('B_x (extracellular)')
plt.show()

true_data = np.stack((B_x, mBU), axis=1)

# check gradients
arr = [7, 10, 2, 0.5, 5]
for i in range(len(true_data)):
    grad_diff = check_grad(residual_wrapper, residual_jacobian_wrapper, arr, K_AB, K_BC, true_data, i)
    if grad_diff > 1e-6:
        print(f"Index: {i}, Difference: {grad_diff}")

# fit with init vals = true vals
unit_res, unit_mse = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=A_t, C_t=C_t, alpha=alpha, kappa=kappa, beta=beta)
unit_res.params
unit_mse

"""MONTE CARLO"""
N = 500
sse_tol = 1e-15
A_t_0 = rand.uniform(0, 20)
C_t_0 = rand.uniform(0, 20)
alpha_0 = rand.uniform(0, 10)
kappa_0 = rand.random()
beta_0 = rand.uniform(0, 10)
res_0, sse_0 = fit_equilibrium_sys(true_data, K_AB, K_BC, A_t=A_t_0, C_t=C_t_0, alpha=alpha_0, kappa=kappa_0, beta=beta_0)

for i in range(N):
    if sse_0 <= sse_tol:
        print("convergence")
        break

    res_dict = res_0.params.valuesdict()
    A_t_i = res_dict['A_t'] * rand.lognormvariate(0, .3)
    C_t_i = res_dict['C_t'] * rand.lognormvariate(0, .3)
    alpha_i = res_dict['alpha'] * rand.lognormvariate(0, .3)
    kappa_i = res_dict['kappa'] * rand.lognormvariate(0, .3)
    beta_i = res_dict['beta'] * rand.lognormvariate(0, .3)

    res_i, sse_i = fit_equilibrium_sys(true_data, K_AB, K_BC, A_t=A_t_i, C_t=C_t_i, alpha=alpha_i, kappa=kappa_i, beta=beta_i)

    if sse_i <= sse_0:
        A_t_0, C_t_0, alpha_0, kappa_0, beta_0 = A_t_i, C_t_i, alpha_i, kappa_i, beta_i
        res_0, sse_0 = res_i, sse_i
        pass
    else:
        a = rand.random()
        if np.exp(-(sse_i - sse_0)) > a:
            A_t_0, C_t_0, alpha_0, kappa_0, beta_0 = A_t_i, C_t_i, alpha_i, kappa_i, beta_i
            res_0, sse_0 = res_i, sse_i
    if i % 10 == 0:
        print(f"{i / N * 100}%")

res_0
sse_0


res_12, mse_12 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=1, C_t=2, alpha=5, kappa=0.75, beta=5)
res_12.params
mse_12


res_1, mse_1 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=1, C_t=2, alpha=3, kappa=0.5, beta=5)
res_1.params
mse_1

res_2, mse_2 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=2, C_t=4, alpha=1.5, kappa=0.75, beta=5)
res_2.params
mse_2

res_3, mse_3 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=5, C_t=10, alpha=1.5, kappa=0.75, beta=5)
res_3.params
mse_3

res_4, mse_4 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=7, C_t=10, alpha=2, kappa=0.5, beta=5)
res_4.params
mse_4


res_5, mse_5 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=7, C_t=14, alpha=2, beta=10, kappa=0.5)
res_5.params
mse_5

res_6, mse_6 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=0.5, C_t=1, alpha=1, kappa=0.5, beta=1)
res_6.params
mse_6

res_11, mse_11 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=0.5, C_t=1, alpha=1, kappa=0.5, beta=1, log_conc=False)

res_11.params
mse_11

# init vals for A_t, C_t greater than true vals by factor of 5 is unstable
res_7, mse_7 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC, A_t=10, C_t=10, alpha=2, beta=5, kappa=0.5)
res_7.params
mse_7
v = res_7.params
res_8, mse_8 = fit_equilibrium_sys(data=true_data, K_AB=K_AB, K_BC=K_BC,
                            A_t = v['A_t'].value, C_t = v['C_t'].value, alpha=v['alpha'].value,
                            beta=v['beta'].value, kappa=v['kappa'].value)
res_8.params
mse_8

res_9, mse_9 = fit_equilibrium_sys(data = true_data, K_AB=K_AB, K_BC=K_BC, A_t=A_t*5, C_t=C_t*10, alpha=1, beta=1, kappa=0.5)
res_9.params
mse_9

res_10, mse_10 = fit_equilibrium_sys(data = true_data, K_AB=K_AB, K_BC=K_BC, A_t=A_t*10, C_t=C_t*10, alpha=1, beta=1, kappa=0.5)
res_10.params
mse_10
