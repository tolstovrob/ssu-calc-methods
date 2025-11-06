"""
Модуль printers.py содержит функции для красивого вывода данных
"""

from typing import Any


def print_matrix(matrix: list[list[Any]], precision: int = 4):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if isinstance(matrix[i][j], float):
                print(f"{matrix[i][j]:.{precision}f}", end="\t")
            else:
                print(f"{matrix[i][j]}", end="\t")
        print()


def print_interpolation_comparison(solver, test_points: list[float]):
    print(f"\n{solver.method_name} - Evaluation at points:")
    for point in test_points:
        try:
            value = solver.evaluate(point)
            print(f"  f({point:.3f}) = {value:.6f}")
        except Exception as e:
            print(f"  f({point:.3f}) = Error: {e}")
