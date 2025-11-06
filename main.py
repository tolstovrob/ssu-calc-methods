#!/usr/bin/python3.13

"""
Главный модуль для демонстрации всех численных методов
"""

from demo import demo_interpolation, demo_linear_systems


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
