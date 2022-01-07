"""This module contains functions that mathematically model
concentrations of species in ternary complex formation at equilibrium.
"""

import math

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import root
from xitorch.optimize import rootfinder

"""
GENERAL SOLUTION (COOPERATIVE EQUILIBRIA)
"""


def equilibrium_system(variables: NDArray[float],
                       total_target: float,
                       total_protac: float,
                       total_e3: float,
                       kd_target: float,
                       kd_e3: float,
                       alpha: float) -> NDArray[float]:
    """"System of equations describes concentrations at equilibrium."""
    target: float = variables[0]
    e3: float = variables[1]
    ternary: float = variables[2]

    F = np.empty(3)
    F[0] = target + kd_e3 * ternary / (alpha * e3) + ternary - total_target
    F[1] = kd_target * kd_e3 * ternary / (alpha * target * e3) + kd_e3 * ternary / (
            alpha * e3) + kd_target * ternary / (alpha * target) + ternary - total_protac
    F[2] = e3 + kd_target * ternary / (alpha * target) + ternary - total_e3
    return F


def jac_equilibrium(variables: NDArray[float],
                    total_target: float,
                    total_protac: float,
                    total_e3: float,
                    kd_target: float,
                    kd_e3: float,
                    alpha: float) -> NDArray[float]:
    """Jacobian of equilibrium system of equations."""
    target: float = variables[0]
    e3: float = variables[1]
    ternary: float = variables[2]

    jacobian = solve_jacobian(target, e3, ternary, kd_target, kd_e3, alpha)
    return jacobian


def solve_jacobian(target: float,
                   e3: float,
                   ternary: float,
                   kd_target: float,
                   kd_e3: float,
                   alpha: float) -> NDArray[float]:
    """Computes the Jacobian of equilibrium system of equations."""
    return np.array(
        [
            [1, -kd_e3 * ternary / (alpha * e3 ** 2), kd_e3 / (alpha * e3) + 1],
            [-kd_target * kd_e3 * ternary / (alpha * target ** 2 * e3) - kd_target * ternary / (alpha * target ** 2),
             -kd_target * kd_e3 * ternary / (alpha * target * e3 ** 2) - kd_e3 * ternary / (alpha * e3 ** 2),
             kd_target * kd_e3 / (alpha * target * e3) + kd_e3 / (alpha * e3) + kd_target / (alpha * target) + 1],
            [-kd_target * ternary / (alpha * target ** 2), 1, kd_target / (alpha * target) + 1]
        ]
    )


def noncooperative_equilibrium(total_target: float,
                               total_protac: float,
                               total_e3: float,
                               kd_target: float,
                               kd_e3: float) -> NDArray[float]:
    """Analytical solution for non-cooperative equilibrium."""
    target = total_target - (total_target + total_protac + kd_target - math.sqrt(
        (total_target + total_protac + kd_target) ** 2 - 4 * total_target * total_protac)) / 2
    e3 = total_e3 - (total_e3 + total_protac + kd_e3 - math.sqrt(
        (total_e3 + total_protac + kd_e3) ** 2 - 4 * total_e3 * total_protac)) / 2

    phi_ab = total_target - target
    phi_bc = total_e3 - e3
    ternary = phi_ab * phi_bc / total_protac if total_protac > 0 else 0

    return np.array([target, e3, ternary])


def solve_ternary(total_target: float,
                  total_protac: float,
                  total_e3: float,
                  kd_target: float,
                  kd_e3: float,
                  alpha: float) -> float:
    """Solves for ternary complex concentration at equilibrium."""
    initial_guess: NDArray[float] = noncooperative_equilibrium(total_target, total_protac, total_e3, kd_target, kd_e3)
    system_args = (total_target, total_protac, total_e3, kd_target, kd_e3, alpha)
    roots = root(fun=equilibrium_system, x0=initial_guess, args=system_args, jac=jac_equilibrium)
    solutions = roots.x
    ternary = solutions[2]
    return ternary


"""TENSOR FUNCTIONS"""


def noncooperative_solution(total_target: torch.Tensor,
                            total_protac: torch.Tensor,
                            total_e3: torch.Tensor,
                            kd_target: float,
                            kd_e3: float) -> torch.Tensor:
    """Computes the equilibrium solution for a non-cooperative system.

    Parameters
    ----------
    total_target : torch.Tensor
        Total target protein concentration.
    total_protac : torch.Tensor
        Total PROTAC concentrations.
    total_e3 : torch.Tensor
        Total E3 ligase concentration.
    kd_target : float
        Binary target protein Kd.
    kd_e3 : float
        Binary E3 ligase Kd.

    Returns
    -------
    torch.Tensor
        Solutions for unbound target, unbound E3 ligase, ternary complex concentrations.

    """
    phi_ab: torch.Tensor = (
                                   total_target
                                   + total_protac
                                   + kd_target
                                   - torch.sqrt(
                               torch.square(total_target + total_protac + kd_target)
                               - 4 * total_target * total_protac
                           )
                           ) / 2

    phi_bc: torch.Tensor = (
                                   total_e3
                                   + total_protac
                                   + kd_e3
                                   - torch.sqrt(
                               torch.square(total_e3 + total_protac + kd_e3)
                               - 4 * total_e3 * total_protac
                           )
                           ) / 2

    # set negative elements to 0
    torch.nn.functional.relu_(phi_ab)
    torch.nn.functional.relu_(phi_bc)

    target: torch.Tensor = total_target - phi_ab
    e3: torch.Tensor = total_e3 - phi_bc

    torch.nn.functional.relu_(target)
    torch.nn.functional.relu_(e3)

    ternary: torch.Tensor = phi_ab * phi_bc / total_protac
    ternary.nan_to_num_(posinf=0, neginf=0)  # handle division by 0

    result = torch.column_stack((target, e3, ternary))
    return result


def equilibrium(y: torch.Tensor,
                total_target: torch.Tensor,
                total_protac: torch.Tensor,
                total_e3: torch.Tensor,
                kd_target: float,
                kd_e3: float,
                alpha: torch.Tensor) -> torch.Tensor:
    """Computes cooperative ternary complex model system of equations.

    Parameters
    ----------
    y : torch.Tensor
        Values for target protein, E3 ligase, ternary complex concentrations.
    total_target : torch.Tensor
        Total target protein concentration.
    total_protac : torch.Tensor
        Total PROTAC concentrations.
    total_e3 : torch.Tensor
        Total E3 ligase concentration.
    kd_target : float
        Binary target protein Kd.
    kd_e3 : float
        Binary E3 ligase Kd.
    alpha : torch.Tensor
        Cooperativity values.

    Returns
    -------
    torch.Tensor
        Differences from equilibrium solution.

    """
    n = y.shape[0]

    target: torch.Tensor = y[:, 0]
    e3: torch.Tensor = y[:, 1]
    ternary: torch.Tensor = y[:, 2]

    f: torch.Tensor = torch.empty((n, 3), dtype=torch.float64)
    f[:, 0] = target + kd_e3 * ternary / (alpha * e3) + ternary - total_target
    f[:, 1] = (
            kd_target * kd_e3 * ternary / (alpha * target * e3)
            + kd_e3 * ternary / (alpha * e3)
            + kd_target * ternary / (alpha * target)
            + ternary
            - total_protac
    )
    f[:, 2] = e3 + kd_target * ternary / (alpha * target) + ternary - total_e3
    f.nan_to_num_(posinf=0, neginf=0)  # handle division by 0

    return f


def solve_equilibrium(total_target: torch.Tensor,
                      total_protac: torch.Tensor,
                      total_e3: torch.Tensor,
                      kd_target: float,
                      kd_e3: float,
                      alpha: torch.Tensor) -> torch.Tensor:
    """Solves equilibrium solution concentrations.

    Parameters
    ----------
    total_target : torch.Tensor
        Total target protein concentration.
    total_protac : torch.Tensor
        Total PROTAC concentrations.
    total_e3 : torch.Tensor
        Total E3 ligase concentration.
    kd_target : float
        Binary target protein Kd.
    kd_e3 : float
        Binary E3 ligase Kd.
    alpha : torch.Tensor
        Cooperativity values.

    Returns
    -------
    torch.Tensor
        Equilibrium concentrations of ternary complex.

    """
    initial_guess: torch.Tensor = noncooperative_solution(total_target, total_protac, total_e3, kd_target, kd_e3)
    params = [total_target, total_protac, total_e3, kd_target, kd_e3, alpha]
    solutions = rootfinder(fcn=equilibrium, y0=initial_guess, params=params, method='broyden2',
                           ftol=1e-15, maxiter=5000, verbose=False)
    ternary = solutions[:, 2]
    return ternary


"""
TRANSFORMED VARIABLES
"""


def equilibrium_f(variables: NDArray[float],
                  total_target: float,
                  total_protac: float,
                  total_e3: float,
                  kd_target: float,
                  kd_e3: float,
                  alpha: float) -> NDArray[float]:
    """System of equations describes concentrations at equilibrium"""
    target = np.square(variables[0])
    e3 = np.square(variables[1])
    ternary = np.square(variables[2])

    f = np.empty(3)
    f[0] = target + kd_e3 * ternary / (alpha * e3) + ternary - total_target
    f[1] = kd_target * kd_e3 * ternary / (alpha * target * e3) \
           + kd_e3 * ternary / (alpha * e3) \
           + kd_target * ternary / (alpha * target) \
           + ternary \
           - total_protac
    f[2] = e3 + kd_target * ternary / (alpha * target) + ternary - total_e3

    return f


def equilibrium_jac(variables: NDArray[float],
                    total_target: float,
                    total_protac: float,
                    total_e3: float,
                    kd_target: float,
                    kd_e3: float,
                    alpha: float) -> NDArray[float]:
    """Computes Jacobian of equilibrium_f()"""
    v1 = variables[0]
    v2 = variables[1]
    v3 = variables[2]
    target = np.square(v1)
    e3 = np.square(v2)
    ternary = np.square(v3)

    jac = [[2 * v1, -2 * kd_e3 * ternary / (alpha * v2 ** 3), 2 * kd_e3 * v3 / (alpha * e3) + 2 * v3],
           [-2 * kd_target * kd_e3 * ternary / (alpha * v1 ** 3 * e3) - 2 * kd_target * ternary / (alpha * v1 ** 3),
            -2 * kd_target * kd_e3 * ternary / (alpha * target * v2 ** 3) - 2 * kd_e3 * ternary / (alpha * v2 ** 3),
            2 * kd_target * kd_e3 * v3 / (alpha * target * e3)
            + 2 * kd_e3 * v3 / (alpha * e3)
            + 2 * kd_target * v3 / (alpha * target)
            + 2 * v3],
           [-2 * kd_target * ternary / (alpha * v1 ** 3), 2 * v2, 2 * kd_target * v3 / (alpha * target) + 2 * v3]]

    return jac


def noncooperative_f(total_target: float,
                     total_protac: float,
                     total_e3: float,
                     kd_target: float,
                     kd_e3: float) -> NDArray[float]:
    """Non-cooperative equilibrium system of equations"""
    target = total_target \
        - (total_target + total_protac + kd_target
           - np.sqrt((total_target + total_protac + kd_target) ** 2 - 4 * total_target * total_protac)) / 2
    e3 = total_e3 \
        - (total_e3 + total_protac + kd_e3
           - np.sqrt((total_e3 + total_protac + kd_e3) ** 2 - 4 * total_e3 * total_protac)) / 2

    target = target if target >= 0 else 0
    e3 = e3 if e3 >= 0 else 0

    phi_ab = total_target - target if total_target > target else 0
    phi_bc = total_e3 - e3 if total_e3 > e3 else 0
    ternary = phi_ab * phi_bc / total_protac if total_protac > 0 else 0

    return np.array([target, e3, ternary])


def predict_ternary(total_target, total_protac, total_e3, kd_target, kd_e3, alpha, return_all=False):
    noncoop_sols = noncooperative_f(total_target, total_protac, total_e3, kd_target, kd_e3)
    init_guess = np.sqrt(noncoop_sols)  # initial guesses for sqrt([target], [e3], [ternary])
    args = (total_target, total_protac, total_e3, kd_target, kd_e3, alpha)
    roots = root(equilibrium_f, init_guess, jac=equilibrium_jac, args=args, options={"maxfev": 5000})

    assert(roots.success, "scipy.optimize.root() did not exit successfully")

    if return_all:
        return np.square(roots.x)  # solutions for [target], [e3], [ternary]

    return np.square(roots.x[2])  # solution for [ternary]
