"""
Задание: Решение задачи Коши для ОДУ первого порядка методом Эйлера
"""

import numpy as np
from typing import Callable
from .base_ode_solver import BaseODESolver, ODEResult


class EulerODESolver(BaseODESolver):
    """Метод Эйлера для решения задачи Коши y' = f(x, y)"""
    
    def __init__(self, precision: float = 1e-6, max_steps: int = 10000):
        super().__init__(precision, max_steps)
    
    @property
    def method_name(self) -> str:
        return "Euler Method for ODE (Cauchy Problem)"
    
    def solve(self, f: Callable, x0: float, y0: float, 
              h: float, n: int, *args, **kwargs) -> ODEResult:
        """
        Решает задачу Коши y' = f(x, y) с начальным условием y(x0) = y0
        методом Эйлера
        
        Args:
            f: функция правой части ОДУ f(x, y, ...)
            x0: начальное значение x
            y0: начальное значение y(x0)
            h: шаг интегрирования
            n: количество шагов
            *args, **kwargs: дополнительные аргументы для функции f
            
        Returns:
            ODEResult с результатами решения
        """
        try:
            self._validate_parameters(x0, y0, h, n)
            
            # Инициализация массивов
            x = np.zeros(n + 1)
            y = np.zeros(n + 1)
            x[0] = x0
            y[0] = y0

            # Основной цикл метода Эйлера
            for i in range(n):
                x[i + 1] = x[i] + h
                y[i + 1] = y[i] + h * f(x[i], y[i], *args, **kwargs)
            
            return ODEResult(
                success=True,
                x_values=x,
                y_values=y,
                iterations=n,
                message=f"Euler method completed successfully with {n} steps, h={h}"
            )
            
        except Exception as e:
            return ODEResult(
                success=False,
                x_values=np.array([]),
                y_values=np.array([]),
                message=f"Error in Euler method: {str(e)}"
            )