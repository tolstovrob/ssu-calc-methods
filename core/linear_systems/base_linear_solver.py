"""
Модуль base_linear_solver.py описывает базовый класс для всех способов решения
СЛАУ. Следует для каждого способа унаследовать класс и сделать свою реализацию.
На выходе всегда получится унифицированный объект класса результата со всей
полезной информацией
"""

from abc import abstractmethod

from core.base_solver import BaseSolver, SolutionResult


class BaseLinearSolver(BaseSolver):
    def __init__(self, precision: float = 1e-6, max_iterations: int = 1000):
        super().__init__(precision, max_iterations)

    @abstractmethod
    def solve(self, A: list[list[float]], b: list[float]) -> SolutionResult: ...

    def _validate_system(self, A: list[list[float]], b: list[float]):
        n = len(A)

        if n != len(b):
            raise ValueError("Matrix A and vector b must have same dimension")

        for row in A:
            if len(row) != n:
                raise ValueError("Matrix A must be square")
