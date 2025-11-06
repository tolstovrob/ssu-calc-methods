"""
Демонстрация методов интерполяции
"""

from io.data_handlers import DataHandler

from core.interpolation.cubic_splines import CubicSplinesSolver
from core.interpolation.general_polynomial import GeneralPolynomialSolver
from core.interpolation.lagrange import LagrangeSolver
from core.interpolation.newton import NewtonSolver


def demo_interpolation():
    """Демонстрация всех методов интерполяции"""
    print("=== INTERPOLATION METHODS DEMO ===\n")

    # Чтение данных
    xs, fs = DataHandler.read_interpolation_nodes()

    # Создание решателей
    solvers = [
        LagrangeSolver(xs, fs),
        NewtonSolver(xs, fs),
        GeneralPolynomialSolver(xs, fs),
        CubicSplinesSolver(xs, fs),
    ]

    # Тестовые точки
    test_points = []
    for i in range(len(xs) - 1):
        test_points.append(xs[i])
        test_points.append((xs[i] + xs[i + 1]) / 2)
    test_points.append(xs[-1])

    # Сравнение методов
    for solver in solvers:
        result = solver.solve()
        DataHandler.print_solution(result, solver.method_name)

        if result.success:
            DataHandler.print_interpolation_comparison(solver, test_points)
        print("-" * 50)


if __name__ == "__main__":
    demo_interpolation()
