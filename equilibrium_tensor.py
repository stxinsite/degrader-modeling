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
    A = np.square(variables[0])
    C = np.square(variables[1])
    ABC = np.square(variables[2])

    F = np.empty((3))
    F[0] = A + K_BC * ABC / (alpha * C) + ABC - A_t
    F[1] = K_AB * K_BC * ABC / (alpha * A * C) + K_BC * ABC / (alpha * C) + K_AB * ABC / (alpha * A) + ABC - B_t
    F[2] = C + K_AB * ABC / (alpha * A) + ABC - C_t
    return F

def equilibrium_jac(variables, A_t, B_t, C_t, K_AB, K_BC, alpha):
    v1 = variables[0]
    v2 = variables[1]
    v3 = variables[2]
    A = np.square(variables[0])
    C = np.square(variables[1])
    ABC = np.square(variables[2])

    D = [ [ 2 * v1, -2 * K_BC * ABC / (alpha * v2**3), 2 * K_BC * v3 / (alpha * C) + 2 * v3 ],
          [ -2 * K_AB * K_BC * ABC / (alpha * v1**3 * C) - 2 * K_AB * ABC / (alpha * v1**3),
            -2 * K_AB * K_BC * ABC / (alpha * A * v2**3) - 2 * K_BC * ABC / (alpha * v2**3),
            2 * K_AB * K_BC * v3 / (alpha * A * C) + 2 * K_BC * v3 / (alpha * C) + 2 * K_AB * v3 / (alpha * A) + 2 * v3 ],
          [ -2 * K_AB * ABC / (alpha * v1**3), 2 * v2, 2 * K_AB * v3 / (alpha * A) + 2 * v3 ] ]

    return D

"""
ANALYTICAL SOLUTIONS FOR [A], [C], [ABC] IN NON-COOPERATIVE EQUILIBRIUM
"""
def noncooperative_f(A_t, B_t, C_t, K_AB, K_BC):
    """NON-COOPERATIVE EQUILIBRIUM"""
    A = A_t - (A_t + B_t + K_AB - sqrt((A_t + B_t + K_AB)**2 - 4*A_t*B_t)) / 2
    C = C_t - (C_t + B_t + K_BC - sqrt((C_t + B_t + K_BC)**2 - 4*C_t*B_t)) / 2

    A = A if A >= 0 else 0
    C = C if C >= 0 else 0

    phi_AB = A_t - A
    phi_BC = C_t - C
    ABC = 0 if B_t == 0 else phi_AB * phi_BC / B_t

    return np.array([A, C, ABC])

noncooperative_f(5e-11, 0, 1e-5, 5e-8, 1e-9)
noncooperative_f(5e-11, 7.5e-2, 1e-5, 5e-8, 1e-9)
noncooperative_f(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9)
noncooperative_f(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9)
noncooperative_f(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9)
noncooperative_f(5e-11, 2.81, 1e-5, 5e-8, 1e-9)


"""
SOLVE EQUILIBRIUM SYSTEM FOR [A], [C], [ABC] GIVEN THETA
"""
def abc_from_theta(A_t, B_t, C_t, K_AB, K_BC, alpha):
    noncoop_sols = noncooperative_f(A_t, B_t, C_t, K_AB, K_BC)
    init_guess = np.sqrt(noncoop_sols)  # initial guesses for [A], [C], [ABC]
    root_args = (A_t, B_t, C_t, K_AB, K_BC, alpha)
    roots = root(equilibrium_f, init_guess, jac=equilibrium_jac, args=root_args, options={"maxfev": 5000})
    assert(roots.success, "scipy.optimize.root() did not exit successfully")
    return np.square(roots.x[2])  # returns solution for [ABC]

abc_from_theta(5e-11, 0, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 7.5e-2, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 2.81, 1e-5, 5e-8, 1e-9, 1)

abc_from_theta(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 30)
abc_from_theta(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 50)
abc_from_theta(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 200)

abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 30)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 50)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 200)

abc_from_theta(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9, 30)
abc_from_theta(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9, 50)
abc_from_theta(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9, 200)

abc_from_theta(5e-11, 2.81, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 2.81, 1e-5, 5e-8, 1e-9, 30)
abc_from_theta(5e-11, 2.81, 1e-5, 5e-8, 1e-9, 200)

"""DERIVATIVES"""
def solve_dfdy(A, C, ABC, K_AB, K_BC, alpha):
    return np.array([[ 1, -K_BC * ABC / ( alpha * C**2 ), K_BC / ( alpha * C ) + 1 ],
                     [ -K_AB * K_BC * ABC / ( alpha * A**2 * C ) -K_AB * ABC / ( alpha * A**2 ),
                       -K_AB * K_BC * ABC / ( alpha * A * C**2 ) -K_BC * ABC / ( alpha * C**2 ),
                       K_AB * K_BC / ( alpha * A * C ) + K_BC / ( alpha * C ) + K_AB / ( alpha * A ) + 1 ],
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

"""NANOBRET DATA PROCESSING"""
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

"""CUSTOM THEANO OP"""

class ABCFromTheta(theano.Op):
    itypes = [tt.dscalar, tt.dscalar, tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, B_t, construct_idx, K_AB, K_BC):
        self.B_t = B_t
        self.construct_idx = construct_idx
        self.K_AB = K_AB
        self.K_BC = K_BC

    def perform(self, node, inputs, outputs):
        A_t = inputs[0]
        C_t = inputs[1]
        alpha = inputs[2]

        B_t = self.B_t.get_value()
        construct_idx = self.construct_idx.get_value()
        N = len(B_t)
        ABC = np.empty(N)
        for i in range(N):
            c = construct_idx[i]
            ABC[i] = abc_from_theta(A_t, B_t[i], C_t, self.K_AB.get_value(), self.K_BC.get_value(), alpha[c])

        z = outputs[0]
        z[0] = ABC

    def grad(self, inputs, g):
        pass

"""TESTING CUSTOM THEANO OP"""
theano.config.compute_test_value = "ignore"
A_t = tt.dscalar('A_t')
C_t = tt.dscalar('C_t')
alpha = tt.dvector('alpha')

B_t = theano.shared(np.array([0, 7.5e-2, 1.2e-7, 9.18e-9, 6.24e-15, 2.81, 9.18e-9, 9.18e-9, 9.18e-9, 9.18e-9]))
construct_idx = theano.shared(np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3], dtype=np.int64))
K_AB = theano.shared(5e-8)
K_BC = theano.shared(1e-9)

ABC_Op = ABCFromTheta(B_t, construct_idx, K_AB, K_BC)

f = theano.function([A_t, C_t, alpha], ABC_Op(A_t, C_t, alpha))

f(5e-11, 1e-5, np.array([1, 30, 50, 200]))

abc_from_theta(5e-11, 0, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 7.5e-2, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 1.2e-7, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 6.24e-15, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 2.81, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 1)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 30)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 50)
abc_from_theta(5e-11, 9.18e-9, 1e-5, 5e-8, 1e-9, 200)

"""EQUILIBRIUM PARAMETER FITTING MCMC"""
K_AB = 1.8e-6
K_BC = 2.5e-7
B_t = np.array([1.12e-6, 2.45e-05])
with pm.Model() as model:
    A_t = pm.Gamma('A_t', alpha=0.001, beta=0.001)  # weak prior for [A]_t
    C_t = pm.Bound(pm.Gamma, lower=A_t)('C_t', alpha=0.001, beta=0.001)  # weak prior for [C]_t constrained to greater than [A]_t
    alpha = pm.Gamma('alpha', mu=30, sigma=3, shape=5)  # vector of length 5: prior mean = 30
    kappa = pm.Uniform('kappa', lower=0, upper=1)
    beta = pm.Gamma('beta', alpha=0.001, beta=0.001)  # weak prior for beta

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
