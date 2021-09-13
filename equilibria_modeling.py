import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from math import sqrt
from lmfit import minimize, fit_report, Parameters
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

"""GENERAL SOLUTION (COOPERATIVE EQUILIBRIA)"""
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
def noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_t):
    A = A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

    phi_AB = A_t - A
    phi_BC = C_t - C
    ABC = phi_AB * phi_BC / B_t

    return np.array([A, C, ABC])

"""
Check that non-coop(B_t) = coop(B_t | alpha = 1)
"""
noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_tmax)
noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_t = 1.12e-6)
noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_t = 2.45e-05)

sys_args = (A_t, C_t, K_AB, K_BC, 1, B_tmax)
roots_3 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_3)

sys_args = (A_t, C_t, K_AB, K_BC, 1, 1.12e-6)
roots_4 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_4)

sys_args = (A_t, C_t, K_AB, K_BC, 1, 2.45e-05)
roots_5 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_5)

"""FITTING PARAMETERS OF EQUILIBRIUM SYSTEM TO EMPIRICAL DATA"""
"""DATA PROCESSING"""
construct = 'VHL_WT SMARCA2_WT'
construct_data = nanobret_df[nanobret_df['Construct'] == construct]
min_time = construct_data['Time'].min()
construct_data.loc[:, 'Minutes'] = (construct_data.loc[:,'Time'] - min_time) / pd.Timedelta(minutes=1)
timepoints = []
mean_corrected_mbus = []
concentrations = []
sds = []
for timepoint in construct_data['Minutes'].unique().tolist():
    construct_data_timepoint = construct_data[construct_data['Minutes'] == timepoint]
    control_mean_mBU = construct_data_timepoint[~construct_data_timepoint['With_618']]['mBU'].mean()
    control_sd_mBU = construct_data_timepoint[~construct_data_timepoint['With_618']]['mBU'].std()
    construct_data_timepoint = construct_data_timepoint[construct_data_timepoint['With_618']]
    for concentration in construct_data_timepoint['uM'].unique().tolist():
        sample_mean_mBU = construct_data_timepoint[construct_data_timepoint['uM'] == concentration]['mBU'].mean()
        sample_sd_mBU = construct_data_timepoint[construct_data_timepoint['uM'] == concentration]['mBU'].std()
        sample_mean_corrected_mBU = sample_mean_mBU - control_mean_mBU
        timepoints.append(timepoint)
        mean_corrected_mbus.append(sample_mean_corrected_mBU)
        concentrations.append(concentration)
        sds.append(sample_sd_mBU)

construct_df = pd.DataFrame({'Minutes': timepoints,
                             'uM': concentrations,
                             'M': conc_molar,
                             'mBU_corrected': mean_corrected_mbus,
                             'stdev': sds,
                             'Construct': construct})
construct_df = construct_df.sort_values(by=['Minutes','uM'])
construct_df.head()
construct_df.shape

construct_df['Minutes'].unique().tolist()

data_obs = construct_df[construct_df['Minutes'] == 96.05][['uM', 'mBU_corrected']].to_numpy()
data_obs

# SPR measurements of ACBI1 PROTAC K_d's from http://cadd.zju.edu.cn/protacdb/compound/dataset=protac&id=798
# convert to micromolar
# original units in nanomolar
K_AB = 250e-3
K_BC = 1800e-3

def residual(params, data_obs, K_AB, K_BC):
    A_t = params['A_t']
    C_t = params['C_t']
    alpha = params['alpha']
    beta = params['beta']
    kappa = params['kappa']

    B_t = np.multiply(data_obs[:,0], kappa)  # B_t = kappa * B_x

    init_params = np.array([A_t, C_t, 0.5*min(A_t, C_t)])  # initial guesses for [A], [C], [ABC]
    ABC_pred = []  # predicted [ABC] that satisfies equilibrium system
    for B_i in B_t:
        sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_i)
        roots = fsolve(equilibrium_sys, init_params, args=sys_args)
        # print(roots)
        # print(np.isclose(equilibrium_sys(roots, *sys_args), [0.0, 0.0, 0.0]).all())
        ABC_pred.append(roots[2])

    y_pred = np.multiply(np.array(ABC_pred), beta)  # ^mBU = ^[ABC] * beta
    return data_obs[:,1] - y_pred

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
    params.add( 'alpha', value = alpha, min = 1.)
    params.add( 'beta', value = beta, min = 0.)
    params.add( 'kappa', value = kappa, min = 0., max = 1.)

    result = minimize( residual, params, args = (data_obs, K_AB, K_BC),
                       method = 'least_squares', nan_policy = 'omit')
    return result

res_all = []
for minute in construct_df['Minutes'].unique().tolist():
    data_obs = construct_df[construct_df['Minutes'] == minute][['uM', 'mBU_corrected']].to_numpy()

    res = fit_equilibrium_sys(data_obs, K_AB, K_BC, A_t = 10, C_t = 10, beta = 1)
    res_all.append(res)

len(res_all)
minutes_list = construct_df['Minutes'].unique().tolist()
for i in range(len(res_all)):
    print(f"Minute: {minutes_list[i]}")
    res = res_all[i]
    res.params.pretty_print()

    data_obs = construct_df[construct_df['Minutes'] == minutes_list[i]][['uM', 'mBU_corrected']].to_numpy()
    resid_k = residual(params=res.params, data_obs=data_obs, K_AB=K_AB, K_BC=K_BC)
    mBU_pred = data_obs[:,1] - resid_k

    plt.plot(data_obs[:,0], data_obs[:,1], label = 'observed', color = 'cyan')
    plt.plot(data_obs[:,0], mBU_pred, label = 'predicted', color = 'coral')
    plt.xscale('log')
    plt.legend()
    plt.show()

    # print(fit_report(res))


# ## S18
# E = K_ED * (E_t - EDT) / (K_ED + D)
# ## S19
# T = K_DT * (T_t - EDT) / (K_DT + D)
#
# ## S38
# ED * DT = (E * D**2 * T) / (K_ED * K_DT)
