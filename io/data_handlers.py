"""
Модуль data_handlers.py содержит функции для ввода и вывода данных
"""


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
