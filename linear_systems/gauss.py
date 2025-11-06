"""
Задание 5: Решить СЛАУ методом Гаусса (прямой и обратный ходы)
"""

from .base_linear_solver import BaseLinearSolver, SolutionResult


class GaussSolver(BaseLinearSolver):
    @property
    def method_name(self) -> str:
        return "Gaussian Elimination for solving linear systems"

    def solve(self, A: list[list[float]], b: list[float]) -> SolutionResult:
        try:
            self._validate_system(A, b)
            n = len(A)

            # Создаем расширенную матрицу
            Ab = [A[i] + [b[i]] for i in range(n)]

            # Прямой ход
            for i in range(n):
                # Поиск главного элемента
                max_row = max(range(i, n), key=lambda r: abs(Ab[r][i]))
                Ab[i], Ab[max_row] = Ab[max_row], Ab[i]

                pivot = Ab[i][i]
                if abs(pivot) < 1e-12:
                    return SolutionResult(False, None, message="Matrix is singular")

                # Нормализация
                for j in range(i, n + 1):
                    Ab[i][j] /= pivot

                # Обнуление столбца
                for k in range(i + 1, n):
                    factor = Ab[k][i]
                    for j in range(i, n + 1):
                        Ab[k][j] -= factor * Ab[i][j]

            # Обратный ход
            x = [0.0] * n
            for i in range(n - 1, -1, -1):
                x[i] = Ab[i][n]
                for j in range(i + 1, n):
                    x[i] -= Ab[i][j] * x[j]
                x[i] /= Ab[i][i]

            return SolutionResult(
                success=True, result=x, message="System solved successfully"
            )

        except Exception as e:
            return SolutionResult(
                success=False, result=None, message=f"Error solving system: {str(e)}"
            )
