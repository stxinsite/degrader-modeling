import numpy as np
import pandas as pd
from scipy.optimize import fsolve, root
from math import sqrt
from lmfit import Minimizer, minimize, fit_report, Parameters
import matplotlib.pyplot as plt
from importlib import reload
from equilibria_modeling import fit_equilibrium_sys

"""FITTING PARAMETERS OF EQUILIBRIUM SYSTEM TO EMPIRICAL DATA"""
"""DATA PROCESSING"""
nanobret_df = pd.read_csv("data/2021-08-26_nano_bret_test.reformed.csv")
nanobret_df['Time'] = pd.to_datetime(nanobret_df['Time'])
nanobret_df['mBU'] = nanobret_df['Lum_610'] / nanobret_df['Lum_450'] * 1000
nanobret_df['B_x'] = nanobret_df['uM'] * 1e-6  # convert uM (micromolar) to molar units
nanobret_df.head()

corrected_nanobret_df = pd.DataFrame(columns=['Minutes', 'uM', 'mBU_corrected', 'stdev', 'Construct'])

construct_list = nanobret_df['Construct'].unique().tolist()
construct_list

for construct in construct_list:
    construct_data = nanobret_df[nanobret_df['Construct'] == construct]
    min_time = construct_data['Time'].min()
    construct_data.loc[:, 'Minutes'] = (construct_data.loc[:,'Time'] - min_time) / pd.Timedelta(minutes=1)
    timepoints = []
    mean_corrected_mbus = []
    concentrations = []
    sds = []
    for timepoint in construct_data['Minutes'].unique().tolist():
        construct_data_timepoint = construct_data[construct_data['Minutes'] == timepoint]
        control_mean_mBU = construct_data_timepoint[~construct_data_timepoint['With_618']]['mBU'].mean()  # not With_618
        control_sd_mBU = construct_data_timepoint[~construct_data_timepoint['With_618']]['mBU'].std()  # not With_618
        construct_data_timepoint = construct_data_timepoint[construct_data_timepoint['With_618']]  # With_618
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
                                 'mBU_corrected': mean_corrected_mbus,
                                 'stdev': sds,
                                 'Construct': construct})
    construct_df = construct_df.sort_values(by=['Minutes','uM'])
    corrected_nanobret_df = corrected_nanobret_df.append(construct_df)
    print(construct)
    print(construct_df['Minutes'].unique().tolist())
    print(construct_df.shape)

corrected_nanobret_df.shape
corrected_nanobret_df.to_pickle("data/corrected_nanobret_df.pkl")


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
                             'mBU_corrected': mean_corrected_mbus,
                             'stdev': sds,
                             'Construct': construct})
construct_df = construct_df.sort_values(by=['Minutes','uM'])
construct_df.head()
construct_df.shape

"""
Save VHL_WT, SMARCA2_WT
"""
construct_df.to_pickle("data/VHL_WT_SMARCA2_WT.pkl")


construct_df['Minutes'].unique().tolist()

data_obs = construct_df[construct_df['Minutes'] == 96.05][['uM', 'mBU_corrected']].to_numpy()
data_obs

# SPR measurements of ACBI1 PROTAC K_d's from http://cadd.zju.edu.cn/protacdb/compound/dataset=protac&id=798
# convert to micromolar
# original units in nanomolar
K_AB = 250e-3
K_BC = 1800e-3

fit_equilibrium_sys(data_obs, K_AB, K_BC, A_t=3., C_t=5., alpha=1., beta=1., kappa=0.5)

res_all = []
for minute in construct_df['Minutes'].unique().tolist():
    data_obs = construct_df[construct_df['Minutes'] == minute][['uM', 'mBU_corrected']].to_numpy()

    res = fit_equilibrium_sys(data_obs, K_AB, K_BC, A_t = 5, C_t = 5, alpha = 1, beta = 1, kappa = 1)
    res_all.append(res)

len(res_all)
minutes_list = construct_df['Minutes'].unique().tolist()
for i in range(len(res_all)):
    print(f"Minute: {minutes_list[i]}")
    res = res_all[i]
    res.params.pretty_print()

    mBU_pred = data_obs[:,1] - res.residual

    plt.plot(data_obs[:,0], data_obs[:,1], label = 'observed', color = 'cyan')
    plt.plot(data_obs[:,0], mBU_pred, label = 'predicted', color = 'coral')
    plt.xscale('log')
    plt.legend()
    plt.show()

    # print(fit_report(res))


"""
TESTING: finding solutions to system of equations for equilibrium concentrations
"""
A_t = 5e-11
C_t = 1e-5
K_AB = 1.8e-6
K_BC = 2.5e-7
alpha = 10

B_tmax = sqrt(K_BC) / (sqrt(K_AB) + sqrt(K_BC)) * (A_t + K_AB) + sqrt(K_AB) / (sqrt(K_AB) + sqrt(K_BC)) * (C_t + K_BC)
TF_50 = A_t / 2 + K_AB / alpha
TI_50 = C_t * (1 + alpha)

init_params = np.array([A_t, C_t, 0.5*min(A_t, C_t)])  # initial guesses for roots
sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_tmax)

sol = root(equilibrium_sys, init_params, jac=jac_equilibrium, args=sys_args)
sol.x
sol.success
roots = fsolve(equilibrium_sys, init_params, args=sys_args)  # solutions at B_tmax
print(roots)
np.isclose(equilibrium_sys(roots, *sys_args), [0.0, 0.0, 0.0]).all()  # checks whether system is close to 0 at solutions

sys_args = (A_t, C_t, K_AB, K_BC, alpha, TF_50)
sol_1 = root(equilibrium_sys, init_params, args=sys_args)
sol_1.x
sol_1.success
roots_1 = fsolve(equilibrium_sys, init_params, args=sys_args)
print(roots_1)

# [ABC] at TF_50 should be approximately 0.5 * [ABC]_max
sol_1.x[2] / sol.x[2]
roots_1[2] / roots[2]

## for B_t > B_tmax, fsolve() is sensitive to the inital parameters
sys_args = (A_t, C_t, K_AB, K_BC, alpha, TI_50)
sol_2 = root(equilibrium_sys, x0=np.array([A_t*.5, C_t*.5, 0.5*TI_50]), jac=jac_equilibrium, args=sys_args)
sol_2.x
roots_2 = fsolve(equilibrium_sys, np.array([A_t * 1e-2, C_t * 1e-2, 0.5*min(A_t, C_t)]), args=sys_args)
print(roots_2)
np.allclose(equilibrium_sys(roots_2, *sys_args), 0)  # checks whether system is close to 0 at solutions

# [ABC] at TI_50 should be approximately 0.5 * [ABC]_max
roots_2[2] / roots[2]

# [ABC] at TF_50 and TI_50 should be approximately equal
np.isclose(roots_1[2], roots_2[2], rtol=1e-6)

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
