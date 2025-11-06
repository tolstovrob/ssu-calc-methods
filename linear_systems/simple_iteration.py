"""
Задание 7: Решить СЛАУ методом простой (ехидной) итерации
"""

from .base_linear_solver import BaseLinearSolver, SolutionResult


class SimpleIterationSolver(BaseLinearSolver):
    def __init__(self, precision: float = 1e-6, max_iterations: int = 1000):
        super().__init__(precision, max_iterations)

    @property
    def method_name(self) -> str:
        return "Simple Iteration Method"

    def solve(self, A: list[list[float]], b: list[float]) -> SolutionResult:
        try:
            self._validate_system(A, b)
            n = len(A)

            alpha, beta = self._build_iteration_system(A, b)

            # Проверяем условие сходимости
            if not self._check_convergence(alpha):
                return SolutionResult(
                    success=False,
                    result=None,
                    message="Convergence condition not satisfied (‖α‖ ≥ 1)",
                )

            # Начальное приближение - нулевой вектор
            x = [0.0] * n
            iterations = 0
            error = float("inf")

            for iteration in range(self.max_iterations):
                x_new = [0.0] * n

                for i in range(n):
                    sum_term = 0.0
                    for j in range(n):
                        sum_term += alpha[i][j] * x[j]
                    x_new[i] = sum_term + beta[i]

                # Вычисляем погрешность
                error = self._calculate_error(x_new, x)
                iterations = iteration + 1

                if error < self.precision:
                    return SolutionResult(
                        success=True,
                        result=x_new,
                        error=error,
                        iterations=iterations,
                        message=f"Converged after {iterations} iterations",
                    )

                x = x_new

            return SolutionResult(
                success=False,
                result=x,
                error=error,
                iterations=iterations,
                message=f"Maximum iterations ({self.max_iterations}) reached",
            )

        except Exception as e:
            return SolutionResult(
                success=False,
                result=None,
                message=f"Error in simple iteration method: {str(e)}",
            )

    def _build_iteration_system(self, A: list[list[float]], b: list[float]) -> tuple:
        n = len(A)
        alpha = [[0.0] * n for _ in range(n)]
        beta = [0.0] * n

        for i in range(n):
            if abs(A[i][i]) < 1e-12:
                raise ValueError(f"Zero diagonal element a[{i}][{i}]")

            beta[i] = b[i] / A[i][i]

            for j in range(n):
                if i != j:
                    alpha[i][j] = -A[i][j] / A[i][i]
                else:
                    alpha[i][j] = 0.0  # диагональные элементы alpha равны 0

        return alpha, beta

    def _check_convergence(self, alpha: list[list[float]]) -> bool:
        n = len(alpha)

        norm_alpha = 0.0
        for i in range(n):
            row_sum = sum(abs(alpha[i][j]) for j in range(n))
            norm_alpha = max(norm_alpha, row_sum)

        return norm_alpha < 1.0

    def _calculate_error(self, x_new: list[float], x_old: list[float]) -> float:
        return max(abs(x_new[i] - x_old[i]) for i in range(len(x_new)))
