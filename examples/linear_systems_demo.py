"""
Демонстрация методов решения СЛАУ
"""

from io.data_handlers import DataHandler

from core.linear_systems.gauss import GaussSolver
from core.linear_systems.simple_iteration import SimpleIterationSolver
from core.linear_systems.thomas import ThomasSolver


def demo_linear_systems():
    """Демонстрация всех методов решения СЛАУ"""
    print("=== LINEAR SYSTEMS SOLVERS DEMO ===\n")

    # Метод Гаусса
    print("1. Gaussian Elimination:")
    A, b = DataHandler.read_linear_system()
    solver = GaussSolver()
    result = solver.solve(A, b)
    DataHandler.print_solution(result, solver.method_name)

    # Метод прогонки
    print("\n2. Thomas Algorithm:")
    a, main_diag, c, d = DataHandler.read_tridiagonal_system()
    n = len(main_diag)
    A_tridiag = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if i > 0:
            A_tridiag[i][i - 1] = a[i]
        A_tridiag[i][i] = main_diag[i]
        if i < n - 1:
            A_tridiag[i][i + 1] = c[i]

    solver = ThomasSolver()
    result = solver.solve(A_tridiag, d)
    DataHandler.print_solution(result, solver.method_name)

    # Метод простых итераций
    print("\n3. Simple Iteration Method:")
    A, b = DataHandler.read_linear_system()
    solver = SimpleIterationSolver(precision=1e-6, max_iterations=1000)
    result = solver.solve(A, b)
    DataHandler.print_solution(result, solver.method_name)


if __name__ == "__main__":
    demo_linear_systems()
