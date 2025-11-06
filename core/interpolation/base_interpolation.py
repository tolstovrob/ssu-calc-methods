"""
Модуль base_interpolation.py описывает базовый класс для всех способов решения
задачи интерполяции. Следует для каждого способа унаследовать класс и сделать
свою реализацию. На выходе всегда получится унифицированный объект класса
результата со всей полезной информацией
"""

from abc import abstractmethod

from core.base_solver import BaseSolver, SolutionResult


class BaseInterpolationSolver(BaseSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__()
        self._validate_nodes(nodes_x, nodes_y)
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.n = len(nodes_x)

    def _validate_nodes(self, nodes_x: list[float], nodes_y: list[float]):
        if len(nodes_x) != len(nodes_y):
            raise ValueError("X and Y nodes must have same length")

        if len(nodes_x) < 2:
            raise ValueError("At least 2 nodes required")

        if len(nodes_x) != len(set(nodes_x)):
            raise ValueError("X nodes must be unique")

    @abstractmethod
    def evaluate(self, x: float) -> float: ...

    def solve(self) -> SolutionResult:
        try:
            return SolutionResult(
                success=True,
                result=self,
                message=f"{self.method_name} built successfully",
            )
        except Exception as e:
            return SolutionResult(
                success=False,
                result=None,
                message=f"Error building interpolator: {str(e)}",
            )
