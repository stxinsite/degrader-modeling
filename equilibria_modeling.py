import numpy as np
import pandas as pd
import random as rand
import seaborn as sns
from scipy.optimize import fsolve, root, check_grad
from scipy.linalg import solve, LinAlgWarning
from math import sqrt, exp, log
from lmfit import Minimizer, minimize, fit_report, Parameters
import pymc3 as pm
import arviz as az
# import theano.tensor as tt
from theano import shared
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

    try:
        dy_dx = solve(df_dy, neg_df_dx)  # [ d[ABC]/dx ] is the third row
    except LinAlgWarning:
        dy_dx = np.empty(len(x))
        dy_dx[:] = np.NaN
        return dy_dx

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

    if not C_t.vary:
        return np.delete(dL_dx, 1, 1)  # delete column d[ABC]/d[C]_t

    return dL_dx

def fit_equilibrium_sys(data, K_AB, K_BC, A_t=1., A_t_max=np.inf, C_t=1., C_t_expr=None, C_t_max=np.inf, alpha=1., alpha_max=np.inf, kappa=0.5, beta=1., beta_max=np.inf, log_conc=False):
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
        params.add( 'A_t', value = A_t, min = 0., max = A_t_max)
        if C_t_expr:  # expr overrides provided C_t value
            params.add( 'C_t', expr = C_t_expr, min = 0., max = C_t_max)
        else:
            params.add( 'C_t', value = C_t, min = 0., max = C_t_max)
    params.add( 'alpha', value = alpha, min = 0., max = alpha_max)
    params.add( 'kappa', value = kappa, min = 0., max = 1.)
    params.add( 'beta', value = beta, min = 0., max = beta_max)

    # result = minimize( residual, params, args = (, data_obs),
    #                    method = 'leastsq', nan_policy = 'omit')
    min = Minimizer(residual, params, fcn_args=(K_AB, K_BC), fcn_kws={'data': data, 'log_conc': log_conc})
    if log_conc:
        out = min.leastsq()
    else:
        out = min.leastsq(Dfun=residual_jacobian)

    sse = np.sum(np.square(out.residual))
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

def plot_fit(x, y, pred):
    # A_t = params['A_t']
    # C_t = params['C_t']
    # alpha = params['alpha']
    # kappa = params['kappa']
    # beta = params['beta']
    # K_AB, K_BC = constants
    #
    # B_t = kappa * x
    #
    # equilibrium_solutions = np.zeros((len(B_t), 3))  # predicted roots ([A],[C],[ABC]) that satisfy equilibrium system
    # for i, B_i in np.ndenumerate(B_t):
    #     init_guess = noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_i)  # initial guesses for [A], [C], [ABC]
    #     # equilibrium_solutions[i] = init_guess
    #     sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_i)
    #     roots = root(equilibrium_sys, init_guess, jac=jac_equilibrium, args=sys_args)
    #     equilibrium_solutions[i] = roots.x
    #
    # pred = beta * equilibrium_solutions[:,2]  # predicted response_i = beta * [ABC]_i

    plt.scatter(x, y, label = "observed", color = "cyan")
    plt.plot(x, pred, label = "predicted", color = "coral")
    plt.xscale('log')
    plt.legend()
    plt.show()
    return

"""MONTE CARLO"""
def equilibrium_mc(fit_data, K_AB, K_BC, N=500, sse_tol=1e-15, seed=0, A_t_max=np.inf, C_t_max=np.inf, C_t_expr=None, alpha_max=np.inf, beta_max=np.inf):
    """
    N: int; number of Monte Carlo trials.
    sse_tol: float; tolerance for sum of squared residuals from fit.
    seed: int; seed for random library.
    fit_data: 2-d array; data arg for fit_equilibrium_sys().
    sigma: float; magnitude of average random displacement.
    dispinc: float; rate constant for 'sigma' adjustment.
    dispconf: int; number of configurations between adjusting 'sigma' value.
    """
    sigma = 1
    dispinc = log(2.0)
    dispconf = 100
    n_accept = 0
    n_reject = 0
    # initial guesses
    A_t_0 = rand.uniform(0, 20)
    C_t_0 = rand.uniform(0, 20)
    alpha_0 = rand.uniform(0, 30)
    kappa_0 = rand.random()
    beta_0 = rand.uniform(0, 20)
    # initial fit
    res_0, sse_0 = fit_equilibrium_sys(fit_data, K_AB, K_BC, A_t=A_t_0, A_t_max=A_t_max, C_t=C_t_0, C_t_max=C_t_max, C_t_expr=C_t_expr,
                                       alpha=alpha_0, alpha_max=alpha_max, kappa=kappa_0, beta=beta_0, beta_max=beta_max)
    for i in range(N):
        if sse_0 <= sse_tol:
            print(f"convergence at trial {i}")
            res_0.params.pretty_print()
            return res_0, sse_0

        res_dict = res_0.params.valuesdict()  # OrderedDict of parameters
        # random perturbation to parameters
        dA_t = rand.normalvariate(0, sigma)
        dC_t = rand.normalvariate(0, sigma)
        dalpha = rand.normalvariate(0, sigma)
        dbeta = rand.normalvariate(0, sigma)
        # proposed new initial values
        A_t_prop = res_dict['A_t'] + dA_t
        C_t_prop = res_dict['C_t'] + dC_t
        alpha_prop = res_dict['alpha'] + dalpha
        beta_prop = res_dict['beta'] + dbeta

        A_t_i = A_t_prop if A_t_prop > 0 else res_dict['A_t'] - dA_t
        C_t_i = C_t_prop if C_t_prop > 0 else res_dict['C_t'] - dC_t
        alpha_i = alpha_prop if alpha_prop > 0 else res_dict['alpha'] - dalpha
        kappa_i = rand.random()  # proposed kappa ~ Unif(0,1)
        beta_i = beta_prop if beta_prop > 0 else res_dict['beta'] - beta_prop

        res_i, sse_i = fit_equilibrium_sys(fit_data, K_AB, K_BC, A_t=A_t_i, A_t_max=A_t_max, C_t=C_t_i, C_t_max=C_t_max, C_t_expr=C_t_expr,
                                           alpha=alpha_i, alpha_max=alpha_max, kappa=kappa_i, beta=beta_i, beta_max=beta_max)
        if sse_i <= sse_0:
            # if SSE improves, accept proposal
            A_t_0, C_t_0, alpha_0, kappa_0, beta_0 = A_t_i, C_t_i, alpha_i, kappa_i, beta_i
            res_0, sse_0 = res_i, sse_i
            n_accept += 1
        else:
            n_reject += 1
        # else:
        #     # no SSE improvement, but random accept if difference is small
        #     if np.exp(-(sse_i - sse_0)) > rand.random():
        #         A_t_0, C_t_0, alpha_0, kappa_0, beta_0 = A_t_i, C_t_i, alpha_i, kappa_i, beta_i
        #         res_0, sse_0 = res_i, sse_i

        if (i+1) % dispconf == 0:
            print(f"{round((i+1) / N * 100)}%")
            print(f"SSE: {sse_0}")
            p_accept = float(n_accept / (n_reject + n_accept))
            n_accept, n_reject = 0, 0
            sigma *= np.exp(2.0 * dispinc * (p_accept - 0.5))

    print("convergence under tolerance not reached")
    return res_0, sse_0

"""
NANOBRET DATA
"""
K_AB = 250e-3
K_BC = 1800e-3

corrected_nanobret_df = pd.read_pickle("data/corrected_nanobret_df.pkl")
sorted_min = sorted(corrected_nanobret_df['Minutes'].unique().tolist())
sorted_min[5]
sorted_min[6]

nanobret_subset = corrected_nanobret_df[corrected_nanobret_df['Minutes'].isin([sorted_min[5], sorted_min[6]])]
nanobret_subset.loc[:,['mBU_corrected']] = nanobret_subset['mBU_corrected'] - min(nanobret_subset['mBU_corrected'])

g = sns.relplot(data=nanobret_subset, x='uM', y='mBU_corrected', hue='Construct')
g.set(xscale = 'log')

nanobret_subset.head()

construct_dict = {
    'VHL_WT SMARCA2_L1415S': 0,
    'VHL_WT SMARCA2_E1420S': 1,
    'VHL_R69Q SMARCA2_WT': 2,
    'VHL_Y112F SMARCA2_WT': 3,
    'VHL_WT SMARCA2_WT': 4
}
nanobret_subset.Construct = [construct_dict[item] for item in nanobret_subset.Construct]
nanobret_subset = nanobret_subset.loc[:,['uM', 'mBU_corrected', 'Construct']]
nanobret_subset.head()

"""PyMC3 Model"""
# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500, return_inferencedata=False)


with pm.Model() as model:
    A_t = pm.Gamma('A_t', alpha=0.001, beta=0.001)  # weak prior for [A]_t
    C_t = pm.Bound(pm.Gamma, lower=A_t)('C_t', alpha=0.001, beta=0.001)  # weak prior for [C]_t constrained to greater than [A]_t
    alpha = pm.Gamma('alpha', mu=30, sigma=3, shape=(5))  # vector of length 5: prior mean = 30
    kappa = pm.Uniform('kappa', lower=0, upper=1)
    beta = pm.Gamma('beta', alpha=0.001, beta=0.001)  # weak prior for beta

    # data[:,0] : uM values
    # data[:,1] : mBU values (observed)
    # data[:,2] : Construct integer values
    data = pm.Data('data', value=nanobret_subset)
    for i in range(len(data)):
        init_guess = noncoop_equilibrium(A_t, C_t, K_AB, K_BC, data[i,0])
        X = shared(init_guess)

    equilibrium_solutions = np.zeros((len(B_x), 3))  # predicted roots ([A],[C],[ABC]) that satisfy equilibrium system
    for i, B_i in np.ndenumerate(B_t):
        init_guess = noncoop_equilibrium(A_t, C_t, K_AB, K_BC, B_i)  # initial guesses for [A], [C], [ABC]
        sys_args = (A_t, C_t, K_AB, K_BC, alpha, B_i)
        roots = root(equilibrium_sys, init_guess, jac=jac_equilibrium, args=sys_args)
        equilibrium_solutions[i] = roots.x

    print(len(data))

    a = noncoop_equilibrium(5e-11, 1e-5, 1.8e-6, 2.5e-7, 1.12e-6)
    x = shared(a)
    pred = pm.Deterministic('ABC', x[2])

    # x = pm.Data('x', [1., 2., 3.])
    y = pm.Data('y', [1., 2., 3.])

    mu = pm.Normal('mu', mu=0, sigma=1)
    obs = pm.Normal('obs', mu=mu, sigma=1, observed=np.random.randn(100))

x.get_value()
pred
data[:,1]
model.basic_RVs
a
obs

nanobret_subset_arr = nanobret_subset[['uM', 'mBU_corrected', 'Construct']].to_numpy()

for construct in nanobret_subset['Construct'].unique():
    construct_subset = nanobret_subset_arr[nanobret_subset_arr[:,2] == construct, 0:2]
    # construct_subset = nanobret_subset[, [1,2]]
    print(construct_subset)

"""
Metropolis MC from scratch
"""
construct_df = pd.read_pickle("data/VHL_WT_SMARCA2_WT.pkl")
minutes_list = construct_df['Minutes'].unique().tolist()
minutes_list

construct_data = construct_df[construct_df['Minutes'] == 96.05][['uM', 'mBU_corrected']].to_numpy()
construct_data[:,1] -= np.min(construct_data[:,1])
construct_data[:,1] /= 10
construct_data
# construct_data = np.delete(construct_data, 0, 0)

res, sse = equilibrium_mc(construct_data, K_AB, K_BC, N = 5000, sse_tol=1e-2, A_t_max=1000, C_t_max=1000, alpha_max = 50)

res.params

plot_fit(construct_data[:,0], construct_data[:,1], res.params, K_AB, K_BC)

res1, sse1 = equilibrium_mc(construct_data, K_AB, K_BC, N = 5000, sse_tol=1e-2, A_t_max=500, C_t_max=500, alpha_max = 40)

res1.params

plot_fit(construct_data[:,0], construct_data[:,1], construct_data[:,1]+res1.residual)

res1_1, sse1_1 = fit_equilibrium_sys(construct_data, K_AB, K_BC, A_t=1.5e-4, A_t_max=500, C_t=2, C_t_max=500,
                                   alpha=30, alpha_max=35, kappa=0.5, beta=2000)
res1_1.params
plot_fit(construct_data[:,0], construct_data[:,1], construct_data[:,1]+res1_1.residual)


res2, sse2 = equilibrium_mc(construct_data, K_AB, K_BC, N = 5000, sse_tol=1e-3, A_t_max=500, C_t_max=500, alpha_max = 35)

res2.params

plot_fit(construct_data[:,0], construct_data[:,1], construct_data[:,1]+res2.residual)

res3, sse3 = equilibrium_mc(construct_data, K_AB, K_BC, N = 5000, sse_tol=1e-3, A_t_max=500, C_t_max=500, alpha_max = 30)

res3.params

plot_fit(construct_data[:,0], construct_data[:,1], construct_data[:,1]+res3.residual)


resid3 = residual({'A_t': .00013, 'C_t': 2.08, 'alpha': 40, 'kappa': 1, 'beta': 2364}, K_AB, K_BC, data=construct_data)
plot_fit(construct_data[:,0], construct_data[:,1], construct_data[:,1]+resid3)


res_all = []
for minute in minutes_list:
    print(f"Minute: {minute}")
    construct_data = construct_df[construct_df['Minutes'] == minute][['uM', 'mBU_corrected']].to_numpy()
    res, sse = equilibrium_mc(construct_data, K_AB, K_BC, N = 1000, C_t_expr = "0.5 * A_t",
                              A_t_max = 1000, alpha_max = 50)
    res_all.append((res, sse))

K_AB
r = res_all[4]
rres = r[0]
rres
r[1]


len(res_all)
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

true_data = np.stack((B_x, mBU), axis=1)  # 16 data points

sub_true_data = true_data[range(0, len(true_data), 2),:]  # subset of 8 data points

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

mc_res, mc_sse = equilibrium_mc(sub_true_data)
mc_res_1, mc_sse_1 = equilibrium_mc(sub_true_data)
mc_res_2, mc_sse_2 = equilibrium_mc(sub_true_data)
mc_res_3, mc_sse_3 = equilibrium_mc(sub_true_data)



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
