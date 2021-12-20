import unittest

import numpy as np
import equilibrium_functions as eq

total_target = 5e-11
total_e3 = 1e-5
kd_target = 5e-8
kd_e3 = 1e-9


class MyTestCase(unittest.TestCase):
    def test_noncooperative(self):
        """Tests that cooperative solution with alpha = 1 is equivalent to
        non-cooperative solution.
        """
        alpha = 1
        total_protac = 8.77e-6
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


if __name__ == '__main__':
    unittest.main()
