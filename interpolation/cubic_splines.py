"""
Задание 4: интерполяция кубическими сплайнами
"""

from linear_systems.gauss import GaussSolver

from .base_interpolation import BaseInterpolationSolver


class CubicSplinesSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)
        self.spline_coefficients = None
        self._build_splines()

    @property
    def method_name(self) -> str:
        return "Cubic Splines Interpolation"

    def _build_splines(self):
        # Проверяем равномерность сетки
        n = self.n
        h = self.nodes_x[1] - self.nodes_x[0]
        for i in range(1, n - 1):
            if abs((self.nodes_x[i + 1] - self.nodes_x[i]) - h) > 1e-10:
                raise ValueError("Cubic splines require uniform grid")

        # Формируем систему уравнений для нахождения коэффициентов
        # Используем граничные условия: S''(x0) = S''(xn) = 0
        matrix_size = 4 * (n - 1)
        A = [[0.0] * matrix_size for _ in range(matrix_size)]
        b = [0.0] * matrix_size

        # Уравнения для совпадения значений в узлах
        eq_index = 0
        for i in range(n - 1):
            # S_i(x_i) = f_i
            A[eq_index][4 * i] = 1.0  # a_i
            b[eq_index] = self.nodes_y[i]
            eq_index += 1

            # S_i(x_{i+1}) = f_{i+1}
            A[eq_index][4 * i] = 1.0  # a_i
            A[eq_index][4 * i + 1] = h  # b_i * h
            A[eq_index][4 * i + 2] = h**2  # c_i * h^2
            A[eq_index][4 * i + 3] = h**3  # d_i * h^3
            b[eq_index] = self.nodes_y[i + 1]
            eq_index += 1

        # Уравнения непрерывности первых производных
        for i in range(n - 2):
            # S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})
            A[eq_index][4 * i + 1] = 1.0  # b_i
            A[eq_index][4 * i + 2] = 2 * h  # 2c_i * h
            A[eq_index][4 * i + 3] = 3 * h**2  # 3d_i * h^2
            A[eq_index][4 * (i + 1) + 1] = -1.0  # -b_{i+1}
            b[eq_index] = 0.0
            eq_index += 1

        # Уравнения непрерывности вторых производных
        for i in range(n - 2):
            # S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})
            A[eq_index][4 * i + 2] = 2.0  # 2c_i
            A[eq_index][4 * i + 3] = 6 * h  # 6d_i * h
            A[eq_index][4 * (i + 1) + 2] = -2.0  # -2c_{i+1}
            b[eq_index] = 0.0
            eq_index += 1

        # S''_0(x_0) = 0
        A[eq_index][2] = 2.0  # 2c_0
        b[eq_index] = 0.0
        eq_index += 1

        # S''_{n-2}(x_{n-1}) = 0
        A[eq_index][4 * (n - 2) + 2] = 2.0  # 2c_{n-2}
        A[eq_index][4 * (n - 2) + 3] = 6 * h  # 6d_{n-2} * h
        b[eq_index] = 0.0

        # Решаем систему
        result = GaussSolver().solve(A, b)

        if not result.success:
            raise ValueError(f"Failed to solve spline system: {result.message}")

        self.spline_coefficients = []
        coeffs = result.result
        for i in range(n - 1):
            self.spline_coefficients.append(
                (
                    coeffs[4 * i],  # a_i
                    coeffs[4 * i + 1],  # b_i
                    coeffs[4 * i + 2],  # c_i
                    coeffs[4 * i + 3],  # d_i
                )
            )

    def _find_spline_index(self, x: float) -> int:
        if x < self.nodes_x[0] or x > self.nodes_x[-1]:
            raise ValueError(f"Point {x} is outside interpolation range")

        for i in range(self.n - 1):
            if self.nodes_x[i] <= x <= self.nodes_x[i + 1]:
                return i
        return self.n - 2

    def evaluate(self, x: float) -> float:
        if self.spline_coefficients is None:
            raise ValueError("Spline coefficients not built")

        i = self._find_spline_index(x)
        a, b, c, d = self.spline_coefficients[i]
        dx = x - self.nodes_x[i]

        return a + b * dx + c * dx**2 + d * dx**3
