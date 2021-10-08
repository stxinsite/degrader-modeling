import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import root, check_grad
from scipy.linalg import solve, LinAlgWarning
from math import sqrt, exp, log
from lmfit import Minimizer, minimize, fit_report, Parameters

"""
ABBREVIATIONS:
- A: target protein
- B: degrader
- C: E3 ligase
"""

"""GENERAL SOLUTION (COOPERATIVE EQUILIBRIA)"""
def equilibrium_sys(variables, *constants):
    """"System of equations describes concentrations at equilibrium"""
    A_t, B_t, C_t, alpha, K_AB, K_BC = constants
    A = np.exp(variables[0])
    C = np.exp(variables[1])
    ABC = np.exp(variables[2])

    F = np.empty(3)
    F[0] = A + K_BC * ABC / (alpha * C) + ABC - A_t
    F[1] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
    F[2] = C + K_AB * ABC / (alpha * A) + ABC - C_t

    return F

def jac_equilibrium(variables, *constants):
    A_t, B_t, C_t, alpha, K_AB, K_BC = constants
    A = np.exp(variables[0])
    C = np.exp(variables[1])
    ABC = np.exp(variables[2])

    jac = [
        [ A, -K_BC * ABC / (alpha * C), K_BC * ABC / (alpha * C) + ABC ],
        [ -K_AB * K_BC * ABC / (alpha * A * C) - K_AB * ABC / (alpha * A),
          -K_AB * K_BC * ABC / (alpha * A * C) - K_BC * ABC / (alpha * C),
          K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC ],
        [ -K_AB * ABC / (alpha * A), C, K_AB * ABC / (alpha * A) + ABC ]
    ]

    return jac

def noncoop_equilibrium(A_t, B_t, C_t, K_AB, K_BC):
    """NON-COOPERATIVE EQUILIBRIUM"""
    A = A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

    phi_AB = A_t - A
    phi_BC = C_t - C
    if B_t == 0:
        ABC = 1e-19
    else:
        ABC = phi_AB * phi_BC / B_t
    return np.array([A, C, ABC])

def residual(params, *constants, data):
    paramvals = params.valuesdict()

    C_t = paramvals['C_t']
    A_t = paramvals['A_t']
    alpha_0 = paramvals['alpha_0']
    alpha_1 = paramvals['alpha_1']
    kappa = 1
    beta = 1

    K_AB, K_BC = constants

    B_x = data[:,0]
    B_t = kappa * B_x

    pred_ABC = np.empty(len(B_t))  # predicted root [ABC] that satisfies equilibrium system
    for i, B_i in np.ndenumerate(B_t):
        noncoop_solutions = noncoop_equilibrium(A_t, B_i, C_t, K_AB, K_BC)
        init_guess = np.log(noncoop_solutions)

        alpha_idx = data[i,1]
        if alpha_idx == 0:
            sys_args = (A_t, B_i, C_t, K_AB, K_BC, alpha_0)
        elif alpha_idx == 1:
            sys_args = (A_t, B_i, C_t, K_AB, K_BC, alpha_1)

        roots = root(equilibrium_sys, init_guess, jac=jac_equilibrium, args=sys_args)
        pred_ABC[i] = np.exp(roots.x[2])

    pred_mBU = beta * pred_ABC
    resid = pred_mBU - data[:,2]  # residuals = predicted - observed
    return resid

def fit_equilibrium(data, K_AB, K_BC, C_t=1., alpha=(1., 1.), kappa=0.5, beta=1.):
    params = Parameters()

    params.add( 'C_t', value = C_t, min = 0., max = 10.)
    params.add( 'delta', value=0.35, min = 0.2, max=0.5 )
    params.add( 'A_t', expr='delta * C_t' )
    params.add( 'alpha_0', value = alpha[0], min = 1., max = 50.)
    params.add( 'alpha_1', value = alpha[1], min = 25., max = 35.)
    # params.add( 'kappa', value = kappa, min = 0., max = 1.)
    # params.add( 'beta', value = beta, min = 0.)

    result = minimize( residual, params, method = 'leastsq',
                       args = (K_AB, K_BC), kws = {'data': data}, nan_policy = 'omit')

    # sse = np.sum(np.square(out.residual))
    return result

def plot_fit(x, y, pred):

    plt.scatter(x, y, label = "observed", color = "cyan")
    plt.plot(x, pred, label = "predicted", color = "coral")
    plt.xscale('log')
    plt.legend()
    plt.show()
    return

"""
NANOBRET DATA
"""
K_AB = 1.8
K_BC = 0.25

corrected_nanobret_df = pd.read_csv("data/corrected_nanobret_df.csv")
sorted_min = sorted(corrected_nanobret_df['Minutes'].unique().tolist())
sorted_min[5]
sorted_min[6]

nanobret_subset = corrected_nanobret_df[corrected_nanobret_df['Minutes'].isin([sorted_min[5], sorted_min[6]])]
# nanobret_subset.loc[:,'mBU_corrected'] = nanobret_subset.mBU_corrected - min(nanobret_subset.mBU_corrected)
nanobret_subset.loc[:,'mBU_corrected'] = nanobret_subset.mBU_corrected / 20

construct_dict = {
    'VHL_WT SMARCA2_L1415S': 0,
    'VHL_WT SMARCA2_E1420S': 1,
    'VHL_R69Q SMARCA2_WT': 0,
    'VHL_Y112F SMARCA2_WT': 0,
    'VHL_WT SMARCA2_WT': 1
}
nanobret_subset.loc[:,'construct_idx'] = [construct_dict[item] for item in nanobret_subset.Construct]

data = nanobret_subset.loc[:,['uM', 'construct_idx', 'mBU_corrected']]

g = sns.relplot(data=data, x='uM', y='mBU_corrected', hue='construct_idx')
g.set(xscale = 'log')

data_arr = data.to_numpy()
fit = fit_equilibrium(data_arr, K_AB, K_BC, C_t = 1, alpha = (25, 30))

fit.params

fit.residual
