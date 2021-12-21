import unittest

import numpy as np
import torch
from torch import tensor

import equilibrium_functions as eq

total_target = 5e-11
total_e3 = 1e-5
kd_target = 5e-8
kd_e3 = 1e-9

total_target_tensor = tensor(5e-11, requires_grad=True)
total_e3_tensor = tensor(1e-5, requires_grad=True)

torch.set_default_dtype(torch.float64)

class MyTestCase(unittest.TestCase):
    def test_noncooperative(self):
        """Tests that cooperative solution with alpha = 1 is equivalent to
        non-cooperative solution.
        """
        alpha = 1
        total_protac = 8.76808e-6
        ternary = eq.solve_ternary(total_target, total_protac, total_e3, kd_target, kd_e3, alpha)
        noncooperative_sol = eq.noncooperative_equilibrium(total_target, total_protac, total_e3, kd_target, kd_e3)
        noncooperative_ternary = noncooperative_sol[2]
        self.assertTrue(np.isclose(ternary, noncooperative_ternary, atol=1e-15))

    def test_cooperative(self):
        """Tests that cooperative solution is equivalent to Douglass et al.'s
        solution in supplementary excel file.
        """
        alpha = 100
        total_protac = 8.76808e-6
        ternary = eq.solve_ternary(total_target, total_protac, total_e3, kd_target, kd_e3, alpha)
        self.assertTrue(np.isclose(a=ternary, b=4.99967e-11, atol=1e-15))

    def test_noncooperative_tensor(self):
        total_protac = tensor(1.14e-10)
        noncooperative_sol = eq.noncooperative_solution(
            total_target=total_target_tensor,
            total_protac=total_protac,
            total_e3=total_e3_tensor,
            kd_target=kd_target,
            kd_e3=kd_e3
        )
        print(noncooperative_sol)

        success = torch.allclose(noncooperative_sol, tensor([4.98859e-11, 9.99989e-6, 1.14e-13]))
        self.assertTrue(success)

    def test_cooperative_tensor(self):
        alpha = tensor([100., 100, 100.], requires_grad=True)
        total_protac = tensor([1.01e-8, 8.76708e-6, 9.84e-1], requires_grad=True)

        ternary = eq.solve_equilibrium(total_target_tensor, total_protac, total_e3_tensor, kd_target, kd_e3, alpha)

        print(ternary)

        success = torch.isclose(ternary, tensor([8.39e-12, 4.99967e-11, 5.08e-16]), rtol=1e-10)
        self.assertTrue(torch.all(success))


if __name__ == '__main__':
    unittest.main()
