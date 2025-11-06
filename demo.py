from data import DataHandler
from interpolation.cubic_splines import CubicSplinesSolver
from interpolation.general_polynomial import GeneralPolynomialSolver
from interpolation.lagrange import LagrangeSolver
from interpolation.newton import NewtonSolver
from linear_systems.gauss import GaussSolver
from linear_systems.simple_iteration import SimpleIterationSolver
from linear_systems.thomas import ThomasSolver


def demo_interpolation():
    """Демонстрация всех методов интерполяции"""
    print("=== INTERPOLATION METHODS DEMO ===\n")

    xs, fs = DataHandler.read_interpolation_nodes()

    solvers = [
        LagrangeSolver(xs, fs),
        NewtonSolver(xs, fs),
        GeneralPolynomialSolver(xs, fs),
        CubicSplinesSolver(xs, fs),
    ]

    test_points = []
    for i in range(len(xs) - 1):
        test_points.append(xs[i])
        test_points.append((xs[i] + xs[i + 1]) / 2)
    test_points.append(xs[-1])

    for solver in solvers:
        result = solver.solve()
        DataHandler.print_solution(result, solver.method_name)

        if result.success:
            DataHandler.print_interpolation_comparison(solver, test_points)
        print("-" * 50)


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

    # Проверяем размерности
    if len(a) != n - 1:
        print(f"Lower diagonal should contain {n - 1} items, {len(a)} given")
        return

    if len(c) != n - 1:
        print(f"Upper diagonal should contain {n - 1} items, {len(a)} given")
        return

    if len(d) != n:
        print(f"Constant column should contain {n - 1} items, {len(a)} given")
        return

    A_tridiag = [[0.0] * n for _ in range(n)]
    for i in range(n):
        # Главная диагональ
        A_tridiag[i][i] = main_diag[i]
        # Верхняя диагональ (элементы над главной)
        if i < n - 1:
            A_tridiag[i][i + 1] = c[i]
        # Нижняя диагональ (элементы под главной) - ИСПРАВЛЕННЫЙ ИНДЕКС!
        if i > 0:
            A_tridiag[i][i - 1] = a[i - 1]  # БЫЛО: a[i], СТАЛО: a[i-1]

    solver = ThomasSolver()
    result = solver.solve(A_tridiag, d)
    DataHandler.print_solution(result, solver.method_name)

    # Метод простых итераций
    print("\n3. Simple Iteration Method:")
    A, b = DataHandler.read_linear_system()
    solver = SimpleIterationSolver(precision=1e-6, max_iterations=1000)
    result = solver.solve(A, b)
    DataHandler.print_solution(result, solver.method_name)
