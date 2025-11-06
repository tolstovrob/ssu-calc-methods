"""
Задание 1: построить интерполирующий многочлен в общем виде
"""

from core.linear_systems.gauss import GaussSolver

from .base_interpolation import BaseInterpolationSolver


class GeneralPolynomialSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)
        self.coefficients = None
        self._build_coefficients()

    @property
    def method_name(self) -> str:
        return "General Polynomial Interpolation"

    def _build_coefficients(self):
        n = self.n

        # Формируем матрицу Вандермонда и вектор правой части для полинома
        A = []
        b = []
        for i in range(n):
            row = [self.nodes_x[i] ** j for j in range(n)]
            A.append(row)
            b.append(self.nodes_y[i])

        # Решаем СЛАУ методом Гаусса
        result = GaussSolver().solve(A, b)

        if not result.success:
            raise ValueError(f"Failed to solve linear system: {result.message}")

        self.coefficients = result.result

    def evaluate(self, x: float) -> float:
        if self.coefficients is None:
            raise ValueError("Polynomial coefficients not built")

        result = 0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (x**i)
        return result
