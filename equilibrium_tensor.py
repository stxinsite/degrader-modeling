import numpy as np
import pandas as pd
import theano
import theano.tensor as tt
import theano.tests.unittest_tools
import pymc3 as pm
import arviz as az
from scipy import special
from scipy.optimize import root, newton, check_grad
from scipy.linalg import solve, LinAlgWarning
import sympy
from math import sqrt, exp, log

"""
EQUILIBRIUM SYSTEM AND JACOBIAN OF F WRT TO [A], [C], [ABC]
"""
def equilibrium_f(variables, A_t, B_t, C_t, K_AB, K_BC, alpha):
    """"System of equations describes concentrations at equilibrium"""
    A = variables[0]
    C = variables[1]
    ABC = variables[2]

    F = np.empty((3))
    F[0] = A + K_BC * ABC / (alpha * C) + ABC - A_t
    F[1] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
    F[2] = C + K_AB * ABC / (alpha * A) + ABC - C_t
    return F

def equilibrium_jac(variables, A_t, B_t, C_t, K_AB, K_BC, alpha):
    A = variables[0]
    C = variables[1]
    ABC = variables[2]

    df_dy = solve_dfdy(A, C, ABC, K_AB, K_BC, alpha)
    return df_dy

def solve_dfdy(A, C, ABC, K_AB, K_BC, alpha):
    return np.array([[ 1, -K_BC * ABC / ( alpha * C**2 ), K_BC / ( alpha * C ) + 1 ],
                     [ -K_AB * K_BC * ABC / ( alpha * A**2 * C ) -K_AB * ABC / ( alpha * A**2 ),
                       -K_AB * K_BC * ABC / ( alpha * A * C**2 ) -K_BC * ABC / ( alpha * C**2 ),
                       K_AB * K_BC / ( alpha * A * C ) + K_BC / ( alpha * C ) + K_AB / ( alpha * A ) + 1 ],
                     [ -K_AB * ABC / ( alpha * A**2 ), 1, K_AB / ( alpha * A ) + 1 ]])

"""
ANALYTICAL SOLUTIONS FOR [A], [C], [ABC] IN NON-COOPERATIVE EQUILIBRIUM
"""
def noncooperative_f(A_t, B_t, C_t, K_AB, K_BC):
    """NON-COOPERATIVE EQUILIBRIUM"""
    A = A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

    phi_AB = A_t - A
    phi_BC = C_t - C
    ABC = 0 if B_t == 0 else phi_AB * phi_BC / B_t
    return np.array([A, C, ABC])

# noncooperative_f(5e-11, 7.5e-2, 1e-5, 5e-8, 1e-9)
# noncooperative_f(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9)
# noncooperative_f(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9)
# noncooperative_f(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9)

"""
SOLVE EQUILIBRIUM SYSTEM FOR [A], [C], [ABC] GIVEN THETA
"""
def abc_from_theta(A_t, B_t, C_t, K_AB, K_BC, alpha):
    init_guess = noncooperative_f(A_t, B_t, C_t, K_AB, K_BC)  # initial guesses for [A], [C], [ABC]
    root_args = (A_t, B_t, C_t, K_AB, K_BC, alpha)
    roots = root(equilibrium_f, init_guess, jac=equilibrium_jac, args=root_args)
    return roots.x[2]  # returns solution for [ABC]

abc_from_theta(5e-11, 7.5e-2, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 1)

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

"""
EQUILIBRIUM PARAMETER FITTING MCMC
"""
corrected_nanobret_df = pd.read_csv("data/corrected_nanobret_df.csv")
sorted_min = sorted(corrected_nanobret_df['Minutes'].unique().tolist())
nanobret_subset = corrected_nanobret_df[corrected_nanobret_df['Minutes'].isin([sorted_min[5], sorted_min[6]])]
nanobret_subset.mBU_corrected = nanobret_subset.mBU_corrected - min(nanobret_subset.mBU_corrected)
nanobret_subset.shape

construct_dict = {
    'VHL_WT SMARCA2_L1415S': 0,
    'VHL_WT SMARCA2_E1420S': 1,
    'VHL_R69Q SMARCA2_WT': 2,
    'VHL_Y112F SMARCA2_WT': 3,
    'VHL_WT SMARCA2_WT': 4
}

nanobret_subset.Construct = [construct_dict[item] for item in nanobret_subset.Construct]
nanobret_subset = nanobret_subset.loc[:,['uM', 'mBU_corrected', 'Construct']]
nanobret_subset.shape
nanobret_subset.head()

class ABCFromTheta(tt.Op):
    itypes = [tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar]
    otypes = [tt.dscalar]

    def perform(self, node, inputs, outputs):
        A_t = inputs[0]
        B_t = inputs[1]
        C_t = inputs[2]
        K_AB = inputs[3]
        K_BC = inputs[4]
        alpha = inputs[5]
        ABC = abc_from_theta(A_t, B_t, C_t, K_AB, K_BC, alpha)

        z = outputs[0]
        z[0] = ABC


A_t = tt.dscalar('A_t')
B_t = tt.dscalar('B_t')
C_t = tt.dscalar('C_t')
K_AB = tt.dscalar('K_AB')
K_BC = tt.dscalar('K_BC')
alpha = tt.dscalar('alpha')
z = ABCFromTheta(A_t, B_t, C_t, K_AB, K_BC, alpha)
f = theano.function([x, y], z)
    # def grad(self, inputs, g):
    #     pass

theano.config.compute_test_value = "ignore"

A_t = tt.scalar('A_t')
C_t = tt.scalar('C_t')
K_AB = tt.scalar('K_AB')
K_BC = tt.scalar('K_BC')
B_t = tt.vector('B_t')

A = A_t - (A_t + B_t + K_AB - tt.sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
C = C_t - (C_t + B_t + K_BC - tt.sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2
phi_AB = A_t - A
phi_BC = C_t - C
ABC = tt.switch(tt.eq(B_t, 0), 0, phi_AB * phi_BC / B_t)

noncoop_f = theano.function([A_t, C_t, K_AB, K_BC, B_t], [A, C, ABC])

noncoop_sols = noncoop_f(5e-11, 1e-5, 1.8e-6, 2.5e-7, [1.12e-6, 8e-6, 2.45e-05])
noncoop_sols
noncoop_sols


K_AB = 1.8e-6
K_BC = 2.5e-7
B_t = np.array([1.12e-6, 2.45e-05])
with pm.Model() as model:
    A_t = pm.Gamma('A_t', alpha=0.001, beta=0.001)  # weak prior for [A]_t
    C_t = pm.Bound(pm.Gamma, lower=A_t)('C_t', alpha=0.001, beta=0.001)  # weak prior for [C]_t constrained to greater than [A]_t
    alpha = pm.Gamma('alpha', mu=30, sigma=3, shape=5)  # vector of length 5: prior mean = 30
    kappa = pm.Uniform('kappa', lower=0, upper=1)
    beta = pm.Gamma('beta', alpha=0.001, beta=0.001)  # weak prior for beta

    A = A_t - (A_t + B_t + K_AB - tt.sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - tt.sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2
    phi_AB = A_t - A
    phi_BC = C_t - C
    ABC = tt.switch(tt.eq(B_t, 0), 0, phi_AB * phi_BC / B_t)



A_t = 5e-11
C_t = 1e-5
B_t = 1.12e-6
B_t = 2.45e-05

"""
EXAMPLE
"""
def func(mu, theta):
    thetamu = theta * mu
    value = np.log(mu) + np.logaddexp(0, thetamu)
    return value

def jac(mu, theta):
    thetamu = theta * mu
    jac = theta * special.expit(thetamu) + 1 / mu
    return jac

def mu_from_theta(theta):
    return newton(func, 1, fprime=jac, args=(theta,))

class MuFromTheta(theano.Op):
    itypes = [tt.dscalar]
    otypes = [tt.dscalar]

    def perform(self, node, inputs, outputs):
        theta, = inputs
        mu = mu_from_theta(theta)
        outputs[0][0] = np.array(mu)

    def grad(self, inputs, g):
        theta, = inputs
        mu = self(theta)
        thetamu = theta * mu
        return [- g[0] * mu ** 2 / (1 + thetamu + tt.exp(-thetamu))]

theano.tests.unittest_tools.verify_grad(MuFromTheta(), [np.array(0.2)])
theano.tests.unittest_tools.verify_grad(MuFromTheta(), [np.array(1e-5)])
theano.tests.unittest_tools.verify_grad(MuFromTheta(), [np.array(1e5)])

tt_mu_from_theta = MuFromTheta()

with pm.Model() as model:
    theta = pm.HalfNormal('theta', sigma=1)
    mu = pm.Deterministic('mu', tt_mu_from_theta(theta))
    pm.Normal('y', mu=mu, sigma=0.1, observed=[0.2, 0.21, 0.3])

    trace = pm.sample()

az.plot_trace(trace)
az.summary(trace)

az.plot_energy(trace)

az.plot_forest(trace)
