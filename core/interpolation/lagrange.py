"""
Задание 2: построить интерполирующий многочлен в форме Лагранжа
"""

from .base_interpolation import BaseInterpolationSolver


class LagrangeSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)

    @property
    def method_name(self) -> str:
        return "Lagrange Interpolation"

    def evaluate(self, x: float) -> float:
        result = 0.0

        for i in range(self.n):
            # Вычисляем базисный полином l_i(x)
            basis_poly = 1.0
            for j in range(self.n):
                if i != j:
                    basis_poly *= (x - self.nodes_x[j]) / (
                        self.nodes_x[i] - self.nodes_x[j]
                    )

            # Добавляем слагаемое f(x_i) * l_i(x)
            result += self.nodes_y[i] * basis_poly

        return result
