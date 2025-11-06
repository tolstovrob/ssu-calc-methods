"""
Главный модуль для демонстрации всех численных методов
"""

from typing import Any

from core.interpolation.cubic_splines import CubicSplinesSolver
from core.interpolation.general_polynomial import GeneralPolynomialSolver
from core.interpolation.lagrange import LagrangeSolver
from core.interpolation.newton import NewtonSolver
from core.linear_systems.gauss import GaussSolver
from core.linear_systems.simple_iteration import SimpleIterationSolver
from core.linear_systems.thomas import ThomasSolver


class DataHandler:
    @staticmethod
    def read_interpolation_nodes() -> tuple[list[float], list[float]]:
        xs = list(map(float, input("Arguments of nodes: ").split()))
        fs = list(map(float, input("Values of nodes: ").split()))
        return xs, fs

    @staticmethod
    def read_linear_system() -> tuple[list[list[float]], list[float]]:
        n = int(input("Matrix dimension: "))

        print("Enter matrix A (row by row):")
        A = [list(map(float, input().split())) for _ in range(n)]

        print("Enter vector b:")
        b = list(map(float, input().split()))

        return A, b

    @staticmethod
    def read_tridiagonal_system() -> tuple[
        list[float], list[float], list[float], list[float]
    ]:
        _ = int(input("Matrix dimension: "))

        print("Enter lower diagonal (a):")
        a = list(map(float, input().split()))

        print("Enter main diagonal (b):")
        b = list(map(float, input().split()))

        print("Enter upper diagonal (c):")
        c = list(map(float, input().split()))

        print("Enter right side (d):")
        d = list(map(float, input().split()))

        return a, b, c, d

    @staticmethod
    def print_solution(result, method_name: str):
        print(f"\n{method_name} Results:")
        print(f"Success: {result.success}")
        if result.message:
            print(f"Message: {result.message}")
        if result.error is not None:
            print(f"Error: {result.error:.6e}")
        if result.iterations is not None:
            print(f"Iterations: {result.iterations}")
        if result.result is not None:
            print(f"Result: {result.result}")

    @staticmethod
    def print_matrix(matrix: list[list[Any]], precision: int = 4):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if isinstance(matrix[i][j], float):
                    print(f"{matrix[i][j]:.{precision}f}", end="\t")
                else:
                    print(f"{matrix[i][j]}", end="\t")
            print()

    @staticmethod
    def print_interpolation_comparison(solver, test_points: list[float]):
        print(f"\n{solver.method_name} - Evaluation at points:")
        for point in test_points:
            try:
                value = solver.evaluate(point)
                print(f"  f({point:.3f}) = {value:.6f}")
            except Exception as e:
                print(f"  f({point:.3f}) = Error: {e}")


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
        print(
            f"❌ Ошибка: нижняя диагональ должна содержать {n - 1} элементов, получено {len(a)}"
        )
        return
    if len(c) != n - 1:
        print(
            f"❌ Ошибка: верхняя диагональ должна содержать {n - 1} элементов, получено {len(c)}"
        )
        return
    if len(d) != n:
        print(
            f"❌ Ошибка: правая часть должна содержать {n} элементов, получено {len(d)}"
        )
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

    print("\nСформированная трехдиагональная матрица:")

    DataHandler.print_matrix(A_tridiag)

    solver = ThomasSolver()
    result = solver.solve(A_tridiag, d)
    DataHandler.print_solution(result, solver.method_name)

    # Метод простых итераций
    print("\n3. Simple Iteration Method:")
    A, b = DataHandler.read_linear_system()
    solver = SimpleIterationSolver(precision=1e-6, max_iterations=1000)
    result = solver.solve(A, b)
    DataHandler.print_solution(result, solver.method_name)


def main():
    """Главная функция для запуска демонстраций"""
    print("=" * 60)
    print("NUMERICAL METHODS DEMONSTRATION")
    print("=" * 60)

    while True:
        print("\nВыберите категорию методов:")
        print("1. Методы интерполяции")
        print("2. Методы решения СЛАУ")
        print("3. Выход")

        choice = input("\nВаш выбор (1-3): ").strip()

        if choice == "1":
            demo_interpolation()
        elif choice == "2":
            demo_linear_systems()
        elif choice == "3":
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
