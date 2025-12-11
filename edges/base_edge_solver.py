"""
Модуль base_edge_solver.py описывает базовый класс для всех способов решения
краевых задач. Следует для каждого способа унаследовать класс и сделать 
свою реализацию. На выходе всегда получится унифицированный объект класса 
результата со всей полезной информацией
"""

from abc import abstractmethod
from typing import Callable, Optional, Tuple

from base_solver import BaseSolver, SolutionResult


class BaseEdgeSolver(BaseSolver):
    def __init__(self, precision: float = 1e-6, max_iterations: int = 1000):
        super().__init__(precision, max_iterations)

    @abstractmethod
    def solve(
        self,
        equation: Callable[[float, float, float], float],  # f(x, y, y')
        boundary_conditions: Tuple[Tuple[float, float], Tuple[float, float]],  # ((x0, y0), (x1, y1))
        interval: Tuple[float, float],
        step_size: Optional[float] = None
    ) -> SolutionResult: ...

    def _validate_parameters(
        self,
        equation: Callable[[float, float, float], float],
        boundary_conditions: Tuple[Tuple[float, float], Tuple[float, float]],
        interval: Tuple[float, float],
        step_size: Optional[float] = None
    ):
        """
        Валидация входных параметров для решения краевой задачи.
        
        Args:
            equation: Функция правой части дифференциального уравнения y'' = f(x, y, y')
            boundary_conditions: Краевые условия в виде ((x0, y0), (x1, y1))
            interval: Интервал решения в виде (x_start, x_end)
            step_size: Шаг сетки (опционально)
        """
        if not callable(equation):
            raise TypeError("equation must be a callable function f(x, y, y_prime)")
        
        if len(boundary_conditions) != 2:
            raise ValueError("boundary_conditions must be a tuple of two tuples ((x0, y0), (x1, y1))")
        
        (x0, y0), (x1, y1) = boundary_conditions
        if not isinstance(x0, (int, float)) or not isinstance(y0, (int, float)):
            raise TypeError("x0 and y0 must be numeric values")
        if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)):
            raise TypeError("x1 and y1 must be numeric values")
        
        if len(interval) != 2:
            raise ValueError("interval must be a tuple (x_start, x_end)")
        
        x_start, x_end = interval
        if not isinstance(x_start, (int, float)) or not isinstance(x_end, (int, float)):
            raise TypeError("x_start and x_end must be numeric values")
        
        if x_start >= x_end:
            raise ValueError("x_start must be less than x_end")
        
        if step_size is not None:
            if not isinstance(step_size, (int, float)):
                raise TypeError("step_size must be a numeric value")
            if step_size <= 0:
                raise ValueError("step_size must be positive")
    
    @property
    @abstractmethod
    def method_name(self) -> str: ...