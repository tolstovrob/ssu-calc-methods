"""
Модуль base_ode_solver.py описывает базовый класс для всех способов решения
задач Коши для обыкновенных дифференциальных уравнений (ОДУ) первого порядка
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np


@dataclass
class ODEResult:
    """Результат решения задачи Коши для ОДУ"""
    success: bool
    x_values: np.ndarray
    y_values: np.ndarray
    error_values: Optional[np.ndarray] = None
    exact_values: Optional[np.ndarray] = None
    iterations: Optional[int] = None
    message: str = ""


class BaseODESolver:
    """Базовый класс для всех решателей задач Коши для ОДУ первого порядка"""
    
    def __init__(self, precision: float = 1e-6, max_steps: int = 10000):
        self.precision = precision
        self.max_steps = max_steps
    
    @abstractmethod
    def solve(self, f: Callable, x0: float, y0: float, 
              h: float, n: int, *args, **kwargs) -> ODEResult: ...
    
    @property
    @abstractmethod
    def method_name(self) -> str: ...
    
    def solve_with_exact(self, f: Callable, exact_func: Callable,
                         x0: float, y0: float, h: float, n: int,
                         *args, **kwargs) -> ODEResult:
        """
        Решает ОДУ и вычисляет погрешность относительно точного решения
        
        Args:
            f: функция правой части ОДУ y' = f(x, y, ...)
            exact_func: функция точного решения y(x)
            x0: начальная точка
            y0: начальное значение y(x0)
            h: шаг интегрирования
            n: количество шагов
            *args, **kwargs: дополнительные параметры для f и exact_func
        """
        result = self.solve(f, x0, y0, h, n, *args, **kwargs)
        
        if not result.success:
            return result
        
        # Вычисляем точное решение и погрешность
        exact_vals = exact_func(result.x_values, *args, **kwargs)
        error_vals = np.abs(result.y_values - exact_vals)
        
        return ODEResult(
            success=True,
            x_values=result.x_values,
            y_values=result.y_values,
            error_values=error_vals,
            exact_values=exact_vals,
            iterations=n,
            message=result.message
        )
    
    def _validate_parameters(self, x0: float, y0: float, h: float, n: int):
        """Проверка корректности параметров"""
        if h <= 0:
            raise ValueError("Step size h must be positive")
        if n <= 0:
            raise ValueError("Number of steps n must be positive")
        if n > self.max_steps:
            raise ValueError(f"Number of steps {n} exceeds maximum {self.max_steps}")