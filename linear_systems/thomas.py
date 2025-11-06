"""
Задание 6: Решить СЛАУ методом прогонки
"""

from .base_linear_solver import BaseLinearSolver, SolutionResult


class ThomasSolver(BaseLinearSolver):
    @property
    def method_name(self) -> str:
        return "Thomas Algorithm (Tridiagonal Matrix Algorithm)"

    def solve(self, A: list[list[float]], b: list[float]) -> SolutionResult:
        try:
            self._validate_system(A, b)
            n = len(A)

            # Извлекаем диагонали из матрицы A
            a, main_diag, c = self._extract_diagonals(A)
            d = b.copy()

            # Прямая прогонка - вычисление прогоночных коэффициентов
            alpha = [0.0] * n
            beta = [0.0] * n

            # Первый узел
            alpha[0] = -c[0] / main_diag[0]
            beta[0] = d[0] / main_diag[0]

            # Промежуточные узлы
            for i in range(1, n - 1):
                denominator = main_diag[i] + a[i] * alpha[i - 1]
                alpha[i] = -c[i] / denominator
                beta[i] = (d[i] - a[i] * beta[i - 1]) / denominator

            # Последний узел
            denominator = main_diag[n - 1] + a[n - 1] * alpha[n - 2]
            beta[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 2]) / denominator

            # Обратная прогонка
            x = [0.0] * n
            x[n - 1] = beta[n - 1]

            for i in range(n - 2, -1, -1):
                x[i] = alpha[i] * x[i + 1] + beta[i]

            return SolutionResult(
                success=True, result=x, message="Tridiagonal system solved successfully"
            )

        except Exception as e:
            return SolutionResult(
                success=False,
                result=None,
                message=f"Error solving tridiagonal system: {str(e)}",
            )

    def _extract_diagonals(self, A: list[list[float]]) -> tuple:
        n = len(A)
        a = [0.0] * n  # нижняя диагональ (a[0] не используется)
        main_diag = [0.0] * n  # главная диагональ
        c = [0.0] * n  # верхняя диагональ (c[n-1] не используется)

        for i in range(n):
            main_diag[i] = A[i][i]

            if i > 0:
                a[i] = A[i][i - 1]  # элемент под главной диагональю
            if i < n - 1:
                c[i] = A[i][i + 1]  # элемент над главной диагональю

        return a, main_diag, c

    def _validate_system(self, A: list[list[float]], b: list[float]):
        super()._validate_system(A, b)
        n = len(A)

        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(A[i][j]) > 1e-10:
                    raise ValueError("Matrix is not tridiagonal")
