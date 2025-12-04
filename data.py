"""
Модуль data_handlers.py содержит функции для ввода и вывода данных
"""

from typing import Any
import numpy as np


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

    # ================== НОВЫЕ МЕТОДЫ ДЛЯ ОДУ ==================

    @staticmethod
    def read_ode_parameters() -> tuple[float, float, float, int]:
        """Чтение параметров для решения задачи Коши"""
        print("\nВведите параметры для решения задачи Коши:")
        x0 = float(input("Начальное значение x0: "))
        y0 = float(input(f"Начальное значение y({x0}) = y0: "))
        h = float(input("Шаг интегрирования h: "))
        n = int(input("Количество шагов n: "))
        
        return x0, y0, h, n

    @staticmethod
    def read_custom_ode_function():
        """Чтение пользовательской функции ОДУ"""
        print("\nВведите правую часть ОДУ в формате Python:")
        print("Пример: для y' = 2*x + y^2 введите: '2*x + y**2'")
        func_str = input("f(x, y) = ")
        
        # Создаем функцию из строки
        def custom_func(x, y):
            return eval(func_str, {"x": x, "y": y, "np": np, "sin": np.sin, 
                                  "cos": np.cos, "exp": np.exp, "log": np.log})
        
        return custom_func

    @staticmethod
    def print_ode_solution(result, method_name: str, show_details: bool = True):
        """Вывод результатов решения ОДУ"""
        print(f"\n{method_name} Results:")
        print(f"Success: {result.success}")
        if result.message:
            print(f"Message: {result.message}")
        if result.iterations is not None:
            print(f"Steps: {result.iterations}")
        
        if result.success and result.x_values is not None:
            print(f"\nNumber of points: {len(result.x_values)}")
            print(f"x range: [{result.x_values[0]:.3f}, {result.x_values[-1]:.3f}]")
            
            if show_details and len(result.x_values) <= 20:
                # Показываем детали только если точек немного
                DataHandler._print_ode_detailed_table(result)
            elif result.error_values is not None and result.exact_values is not None:
                # Иначе показываем только статистику
                max_error = np.max(result.error_values)
                mean_error = np.mean(result.error_values)
                print(f"Maximum error: {max_error:.6e}")
                print(f"Mean error: {mean_error:.6e}")

    @staticmethod
    def _print_ode_detailed_table(result):
        """Вспомогательный метод для детального вывода таблицы ОДУ"""
        if not result.success or result.x_values is None:
            return
        
        print("\nDetailed results:")
        print("-" * 130)
        
        if result.exact_values is not None and result.error_values is not None:
            # С точным решением
            print(f"{'x':>12} | {'y_numerical':>12} | {'y_exact':>12} | {'error':>12} |")
            print("-" * 130)
            for i in range(len(result.x_values)):
                x = result.x_values[i]
                y_num = result.y_values[i]
                y_exact = result.exact_values[i]
                error = result.error_values[i]
                print(f"{x:12.7f} | {y_num:12.7f} | {y_exact:12.7f} | {error:12.7f} |")
        else:
            # Без точного решения
            print(f"{'x':>12} | {'y_numerical':>12} |")
            print("-" * 130)
            for i in range(len(result.x_values)):
                x = result.x_values[i]
                y_num = result.y_values[i]
                print(f"{x:12.7f} | {y_num:12.7f} |")
        
        print("-" * 130)

    @staticmethod
    def print_ode_comparison(result_euler, result_improved):
        """Сравнение методов Эйлера и усовершенствованного Эйлера"""
        if not result_euler.success or not result_improved.success:
            print("Cannot compare: one or both methods failed")
            return
        
        print("\n" + "=" * 60)
        print("COMPARISON OF EULER METHODS")
        print("=" * 60)
        
        # Форматированный вывод как в вашем коде
        print("\nМетод Эйлера: ")
        print("-" * 130)
        print("x:      ", " ".join(f"{x:>10.7f}" for x in result_euler.x_values))
        print("y_M:    ", " ".join(f"{y:>10.7f}" for y in result_euler.y_values))
        
        if result_euler.exact_values is not None:
            print("y_T:    ", " ".join(f"{y:>10.7f}" for y in result_euler.exact_values))
        
        if result_euler.error_values is not None:
            print("Погрешн:", " ".join(f"{e:>10.7f}" for e in result_euler.error_values))
        print("-" * 130)

        print("\nУсовершенствованный метод Эйлера: ")
        print("-" * 130)
        print("x:      ", " ".join(f"{x:>10.7f}" for x in result_improved.x_values))
        print("y_M:    ", " ".join(f"{y:>10.7f}" for y in result_improved.y_values))
        
        if result_improved.exact_values is not None:
            print("y_T:    ", " ".join(f"{y:>10.7f}" for y in result_improved.exact_values))
        
        if result_improved.error_values is not None:
            print("Погрешн:", " ".join(f"{e:>10.7f}" for e in result_improved.error_values))
        print("-" * 130)
        
        # Статистика ошибок
        if result_euler.error_values is not None and result_improved.error_values is not None:
            print("\n" + "=" * 60)
            print("ERROR STATISTICS")
            print("=" * 60)
            
            euler_max = np.max(result_euler.error_values)
            euler_mean = np.mean(result_euler.error_values)
            improved_max = np.max(result_improved.error_values)
            improved_mean = np.mean(result_improved.error_values)
            
            print(f"{'Method':<25} {'Max Error':<15} {'Mean Error':<15}")
            print("-" * 55)
            print(f"{'Euler':<25} {euler_max:<15.7f} {euler_mean:<15.7f}")
            print(f"{'Improved Euler':<25} {improved_max:<15.7f} {improved_mean:<15.7f}")
            
            if euler_max > 0:
                improvement = (euler_max - improved_max) / euler_max * 100
                print(f"\nImprovement in max error: {improvement:.2f}%")

    @staticmethod
    def print_ode_summary_table(solver_results: list):
        """Сводная таблица для нескольких методов ОДУ"""
        if not solver_results:
            return
        
        print("\n" + "=" * 60)
        print("SUMMARY OF ODE SOLVERS")
        print("=" * 60)
        
        print(f"\n{'Method':<30} {'Success':<10} {'Steps':<10} {'Max Error':<15}")
        print("-" * 65)
        
        for solver, result in solver_results:
            if result.success and result.error_values is not None:
                max_error = np.max(result.error_values)
                print(f"{solver.method_name:<30} {'✓':<10} "
                      f"{result.iterations:<10} {max_error:<15.6e}")
            else:
                status = '✓' if result.success else '✗'
                print(f"{solver.method_name:<30} {status:<10} "
                      f"{result.iterations if result.iterations else '-':<10} "
                      f"{'-':<15}")