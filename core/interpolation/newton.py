"""
Задание 3: построить интерполирующий многочлен в форме Ньютона
"""

from .base_interpolation import BaseInterpolationSolver


class NewtonSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)
        self.divided_diffs = self._build_divided_differences()

    @property
    def method_name(self) -> str:
        return "Newton Interpolation"

    def _build_divided_differences(self) -> list[float]:
        # Создаем копию значений узлов для работы
        n = self.n
        diff_table = [self.nodes_y.copy()]

        # Строим таблицу разделенных разностей
        for i in range(1, n):
            row = []
            for j in range(n - i):
                numerator = diff_table[i - 1][j + 1] - diff_table[i - 1][j]
                denominator = self.nodes_x[j + i] - self.nodes_x[j]
                row.append(numerator / denominator)
            diff_table.append(row)

        # Возвращаем первую строку (разности для полинома Ньютона)
        return [row[0] for row in diff_table]

    def evaluate(self, x: float) -> float:
        result = self.divided_diffs[0]  # f[x0]

        for i in range(1, self.n):
            term = self.divided_diffs[i]
            for j in range(i):
                term *= x - self.nodes_x[j]
            result += term

        return result
