"""
Модуль base_solver.py содержит описание базового класса для всех решателей
задач. Сам по себе не несёт смысловую нагрузку, однако является родителем для
конкретных решателей, которые его будут реализовывать.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SolutionResult:
    success: bool
    result: Any
    error: Optional[float] = None
    iterations: Optional[int] = None
    message: str = ""


class BaseSolver(ABC):
    def __init__(self, precision: float = 1e-6, max_iterations: int = 1000):
        self.precision = precision
        self.max_iterations = max_iterations

    @abstractmethod
    def solve(self, *args, **kwargs) -> SolutionResult: ...

    @property
    @abstractmethod
    def method_name(self) -> str: ...
