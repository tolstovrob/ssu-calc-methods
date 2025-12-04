#!/usr/bin/python3.13

"""
Главный модуль для демонстрации всех численных методов
"""

from demo import demo_interpolation, demo_linear_systems
from demo_ode import demo_cauchy_problem, demo_cauchy_problem_interactive


def main():
    """Главная функция для запуска демонстраций"""
    print("=" * 60)
    print("NUMERICAL METHODS DEMONSTRATION")
    print("=" * 60)

    while True:
        print("\nВыберите категорию методов:")
        print("1. Методы интерполяции")
        print("2. Методы решения СЛАУ")
        print("3. Методы решения ОДУ (задачи Коши)")
        print("4. Выход")

        choice = input("\nВаш выбор (1-4): ").strip()

        if choice == "1":
            demo_interpolation()
        elif choice == "2":
            demo_linear_systems()
        elif choice == "3":
            print("\nВыберите режим решения ОДУ:")
            print("  1. Демонстрация с тестовой задачей")
            print("  2. Интерактивный режим с выбором задачи")
            sub_choice = input("  Ваш выбор (1-2): ").strip()
            if sub_choice == "1":
                demo_cauchy_problem()
            elif sub_choice == "2":
                demo_cauchy_problem_interactive()
            else:
                print("Неверный выбор")
        elif choice == "4":
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()