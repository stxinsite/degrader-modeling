"""This module contains functions that mathematically model
concentrations of species in ternary complex formation at equilibrium.
"""

import math
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

"""GENERAL SOLUTION (COOPERATIVE EQUILIBRIA)"""


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
    F[1] = kd_target * kd_e3 * ternary / (alpha * target * e3) + kd_e3 * ternary / (alpha * e3) + kd_target * ternary / (alpha * target) + ternary - total_protac
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
    target = total_target - (total_target + total_protac + kd_target - math.sqrt((total_target + total_protac + kd_target) ** 2 - 4 * total_target * total_protac)) / 2
    e3 = total_e3 - (total_e3 + total_protac + kd_e3 - math.sqrt((total_e3 + total_protac + kd_e3) ** 2 - 4 * total_e3 * total_protac)) / 2

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