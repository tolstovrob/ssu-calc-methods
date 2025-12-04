#!/usr/bin/env python3
"""
Модуль для построения графиков интерполяционных методов
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from interpolation.cubic_splines import CubicSplinesSolver
from interpolation.general_polynomial import GeneralPolynomialSolver
from interpolation.lagrange import LagrangeSolver
from interpolation.newton import NewtonSolver


def read_interpolation_nodes() -> tuple[List[float], List[float]]:
    """Чтение узлов интерполяции"""
    print("Введите узлы интерполяции:")
    xs = list(map(float, input("Аргументы x (через пробел): ").split()))
    fs = list(map(float, input("Значения f(x) (через пробел): ").split()))
    return xs, fs


def plot_interpolation_methods():
    """Построение графиков всех методов интерполяции в одних осях"""
    # Чтение данных
    xs, fs = read_interpolation_nodes()

    # Создание решателей
    solvers = [
        ("Полином Лагранжа", LagrangeSolver(xs, fs)),
        ("Полином Ньютона", NewtonSolver(xs, fs)),
        ("Общий полином", GeneralPolynomialSolver(xs, fs)),
        ("Кубические сплайны", CubicSplinesSolver(xs, fs)),
    ]

    # Решаем все методы
    successful_methods = []
    for name, solver in solvers:
        result = solver.solve()
        if result.success:
            successful_methods.append((name, solver))
            print(f"✅ {name} - успешно построен")
        else:
            print(f"❌ {name} - ошибка: {result.message}")

    if not successful_methods:
        print("❌ Ни один метод интерполяции не сработал")
        return

    # Настройка графика
    x_min, x_max = min(xs), max(xs)
    padding = (x_max - x_min) * 0.2  # 20% отступ по краям
    x_plot = np.linspace(x_min - padding, x_max + padding, 1000)

    # Создаем один график
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green", "orange", "purple"]
    line_styles = ["-", "--", "-.", ":", "-"]

    # Строим графики для каждого метода в одних осях
    for idx, (name, solver) in enumerate(successful_methods):
        # Вычисляем значения для графика
        y_plot = []
        for x in x_plot:
            try:
                y_val = solver.evaluate(x)
                y_plot.append(y_val)
            except (ValueError, ZeroDivisionError):
                y_plot.append(np.nan)

        # Строим график
        color = colors[idx % len(colors)]
        line_style = line_styles[idx % len(line_styles)]

        plt.plot(
            x_plot, y_plot, color=color, linestyle=line_style, linewidth=2, label=name
        )

    # Добавляем график x^3 + 1
    y_cubic = x_plot**3 + 1
    plt.plot(
        x_plot,
        y_cubic,
        color="purple",
        linestyle="-",
        linewidth=3,
        alpha=0.7,
        label="x³ + 1 (эталон)",
    )

    # Отмечаем узлы интерполяции
    plt.scatter(xs, fs, color="black", s=80, zorder=5, label="Узлы интерполяции")

    # Настройки графика
    plt.title(
        f"Сравнение методов интерполяции ({len(xs)} узлов)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("x", fontsize=14)
    plt.ylabel("f(x)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Выделяем область интерполяции
    plt.axvspan(x_min, x_max, alpha=0.1, color="gray", label="Область интерполяции")

    # Устанавливаем пределы
    plt.xlim(x_min - padding, x_max + padding)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_interpolation_methods()
