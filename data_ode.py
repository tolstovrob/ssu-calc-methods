"""
Демонстрация методов решения задач Коши для ОДУ первого порядка
"""

import numpy as np
from data import DataHandler
from ode.euler_ode import EulerODESolver
from ode.improved_euler_ode import ImprovedEulerODESolver


def demo_cauchy_problem():
    """Демонстрация методов решения задачи Коши"""
    print("=" * 60)
    print("CAUCHY PROBLEM FOR ODE - DEMONSTRATION")
    print("=" * 60)
    
    # Определяем тестовую задачу из вашего кода
    def f_test(x, y, V=5):
        """Функция правой части ОДУ: y' = 2Vx + Vx² - y"""
        return 2 * V * x + V * x**2 - y
    
    def exact_solution_test(x, V=5):
        """Точное решение: y(x) = Vx²"""
        return V * x**2
    
    print("\nТестовая задача:")
    print("  Уравнение: y' = 2Vx + Vx² - y")
    print("  Точное решение: y(x) = Vx²")
    print("  Параметр V = 5")
    
    # Параметры из вашего кода
    x0 = 1.0
    y0 = 5.0
    V = 5.0
    h = 0.001
    n = 10
    
    print(f"\nПараметры решения:")
    print(f"  Начальные условия: x0 = {x0}, y0 = {y0}")
    print(f"  Шаг интегрирования h = {h}")
    print(f"  Количество шагов n = {n}")
    
    # Создаем решатели
    euler_solver = EulerODESolver()
    improved_solver = ImprovedEulerODESolver()
    
    print("\n" + "=" * 60)
    print("1. МЕТОД ЭЙЛЕРА")
    print("=" * 60)
    
    # Решаем методом Эйлера
    euler_result = euler_solver.solve_with_exact(
        f=f_test,
        exact_func=exact_solution_test,
        x0=x0,
        y0=y0,
        h=h,
        n=n,
        V=V
    )
    
    DataHandler.print_ode_solution(euler_result, euler_solver.method_name)
    
    if euler_result.success:
        # Выводим таблицу в стиле вашего кода
        print("\nМетод Эйлера:")
        print("-" * 130)
        print("x:      ", " ".join(f"{x:>10.7f}" for x in euler_result.x_values))
        print("y_M:    ", " ".join(f"{y:>10.7f}" for y in euler_result.y_values))
        print("y_T:    ", " ".join(f"{y:>10.7f}" for y in euler_result.exact_values))
        print("Погрешн:", " ".join(f"{e:>10.7f}" for e in euler_result.error_values))
        print("-" * 130)
    
    print("\n" + "=" * 60)
    print("2. УСОВЕРШЕНСТВОВАННЫЙ МЕТОД ЭЙЛЕРА")
    print("=" * 60)
    
    # Решаем усовершенствованным методом Эйлера
    improved_result = improved_solver.solve_with_exact(
        f=f_test,
        exact_func=exact_solution_test,
        x0=x0,
        y0=y0,
        h=h,
        n=n,
        V=V
    )
    
    DataHandler.print_ode_solution(improved_result, improved_solver.method_name)
    
    if improved_result.success:
        # Выводим таблицу в стиле вашего кода
        print("\nУсовершенствованный метод Эйлера:")
        print("-" * 130)
        print("x:      ", " ".join(f"{x:>10.7f}" for x in improved_result.x_values))
        print("y_M:    ", " ".join(f"{y:>10.7f}" for y in improved_result.y_values))
        print("y_T:    ", " ".join(f"{y:>10.7f}" for y in improved_result.exact_values))
        print("Погрешн:", " ".join(f"{e:>10.7f}" for e in improved_result.error_values))
        print("-" * 130)
    
    # Сравнение методов
    if euler_result.success and improved_result.success:
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ ТОЧНОСТИ МЕТОДОВ")
        print("=" * 60)
        
        euler_max_error = np.max(euler_result.error_values)
        euler_mean_error = np.mean(euler_result.error_values)
        improved_max_error = np.max(improved_result.error_values)
        improved_mean_error = np.mean(improved_result.error_values)
        
        print(f"{'Метод':<30} {'Макс. погрешность':<20} {'Сред. погрешность':<20}")
        print("-" * 70)
        print(f"{'Метод Эйлера':<30} {euler_max_error:<20.7f} {euler_mean_error:<20.7f}")
        print(f"{'Ус. метод Эйлера':<30} {improved_max_error:<20.7f} {improved_mean_error:<20.7f}")
        
        # Вычисляем улучшение
        if euler_max_error > 0:
            improvement = (euler_max_error - improved_max_error) / euler_max_error * 100
            print(f"\nУлучшение максимальной погрешности: {improvement:.2f}%")


def demo_cauchy_problem_interactive():
    """Интерактивная демонстрация с вводом пользовательских параметров"""
    print("=" * 60)
    print("CAUCHY PROBLEM - INTERACTIVE MODE")
    print("=" * 60)
    
    # Запрос параметров у пользователя
    print("\nВведите параметры для задачи Коши:")
    try:
        x0 = float(input("Начальное значение x0: "))
        y0 = float(input(f"Начальное значение y({x0}) = y0: "))
        h = float(input("Шаг интегрирования h: "))
        n = int(input("Количество шагов n: "))
        
        print("\nВыберите тестовую задачу:")
        print("1. y' = 2Vx + Vx² - y, y(x) = Vx² (V=5)")
        print("2. y' = -y, y(x) = e^{-x}")
        print("3. y' = x + y, y(x) = e^x - x - 1")
        
        choice = input("\nВаш выбор (1-3): ").strip()
        
        if choice == "1":
            V = float(input("Введите параметр V (по умолчанию 5): ") or "5")
            def f(x, y, V=V):
                return 2 * V * x + V * x**2 - y
            def exact(x, V=V):
                return V * x**2
            params = {'V': V}
        elif choice == "2":
            def f(x, y):
                return -y
            def exact(x):
                return np.exp(-x)
            params = {}
        elif choice == "3":
            def f(x, y):
                return x + y
            def exact(x):
                return np.exp(x) - x - 1
            params = {}
        else:
            print("Неверный выбор, используется задача по умолчанию")
            V = 5
            def f(x, y, V=V):
                return 2 * V * x + V * x**2 - y
            def exact(x, V=V):
                return V * x**2
            params = {'V': V}
        
        # Решаем обоими методами
        euler_solver = EulerODESolver()
        improved_solver = ImprovedEulerODESolver()
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ")
        print("=" * 60)
        
        # Метод Эйлера
        euler_result = euler_solver.solve_with_exact(f, exact, x0, y0, h, n, **params)
        DataHandler.print_ode_solution(euler_result, "Euler Method")
        
        # Усовершенствованный метод Эйлера
        improved_result = improved_solver.solve_with_exact(f, exact, x0, y0, h, n, **params)
        DataHandler.print_ode_solution(improved_result, "Improved Euler Method")
        
        # Сравнение
        if euler_result.success and improved_result.success:
            DataHandler.print_ode_comparison_table(euler_result, "Euler Method Detailed")
            DataHandler.print_ode_comparison_table(improved_result, "Improved Euler Method Detailed")
        
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
    except Exception as e:
        print(f"Ошибка при решении: {e}")


if __name__ == "__main__":
    demo_cauchy_problem()