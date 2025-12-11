"""
Модуль edge_implementation.py реализует решение краевой задачи
с использованием базового класса BaseEdgeSolver
"""

from typing import Callable, Optional, Tuple, List
import math

# Импортируем созданные ранее классы
from base_solver import BaseSolver, SolutionResult
from base_edge_solver import BaseEdgeSolver


class ShootingEdgeSolver(BaseEdgeSolver):
    """Реализация метода стрельбы для решения краевой задачи"""
    
    @property
    def method_name(self) -> str:
        return "Shooting method for edge problem"
    
    def solve(
        self,
        equation: Callable[[float, float, float], float],
        boundary_conditions: Tuple[Tuple[float, float], Tuple[float, float]],
        interval: Tuple[float, float],
        step_size: Optional[float] = None,
        **kwargs
    ) -> SolutionResult:
        """
        Решение краевой задачи методом стрельбы.
        
        Args:
            equation: Функция правой части y'' = f(x, y, y')
            boundary_conditions: Краевые условия ((x0, y0), (x1, y1))
            interval: Интервал решения (x_start, x_end)
            step_size: Шаг сетки
            **kwargs: Дополнительные параметры
        
        Returns:
            SolutionResult с результатами вычислений
        """
        try:
            self._validate_parameters(equation, boundary_conditions, interval, step_size)
            
            # Извлекаем краевые условия
            (x0, y0), (x1, y1) = boundary_conditions
            x_start, x_end = interval
            
            # Если шаг не указан, вычисляем его
            if step_size is None:
                step_size = (x_end - x_start) / 100
            
            # Метод стрельбы - предполагаем начальный наклон y'(x0)
            initial_slope_guess = (y1 - y0) / (x1 - x0)
            
            # Решаем задачу Коши с начальным условием (x0, y0, y'=guess)
            result = self._solve_cauchy(
                equation, 
                (x0, y0, initial_slope_guess),
                (x_start, x_end),
                step_size,
                **kwargs
            )
            
            if not result.success:
                return result
            
            x_values, y_values = result.result
            
            # Корректируем методом половинного деления для удовлетворения правому краевому условию
            iterations = 0
            current_y_end = y_values[-1]
            
            while abs(current_y_end - y1) > self.precision and iterations < self.max_iterations:
                # Корректируем начальный наклон
                initial_slope_guess *= y1 / current_y_end if current_y_end != 0 else 1.1
                
                # Решаем снова
                result = self._solve_cauchy(
                    equation,
                    (x0, y0, initial_slope_guess),
                    (x_start, x_end),
                    step_size,
                    **kwargs
                )
                
                if not result.success:
                    return result
                
                x_values, y_values = result.result
                current_y_end = y_values[-1]
                iterations += 1
            
            return SolutionResult(
                success=True,
                result={'x': x_values, 'y': y_values, 'initial_slope': initial_slope_guess},
                iterations=iterations,
                error=abs(current_y_end - y1),
                message=f"Successfully solved using {self.method_name}"
            )
            
        except Exception as e:
            return SolutionResult(
                success=False,
                result=None,
                message=f"Error in {self.method_name}: {str(e)}"
            )
    
    def _solve_cauchy(
        self,
        equation: Callable[[float, float, float], float],
        initial_conditions: Tuple[float, float, float],  # (x0, y0, y0')
        interval: Tuple[float, float],
        step_size: float,
        **kwargs
    ) -> SolutionResult:
        """Вспомогательный метод для решения задачи Коши"""
        x0, y0, y_prime0 = initial_conditions
        x_start, x_end = interval
        
        n = int(math.ceil((x_end - x_start) / step_size))
        
        x_values = [x0]
        y_values = [y0]
        y_prime_values = [y_prime0]
        
        x_current = x0
        y_current = y0
        y_prime_current = y_prime0
        
        for i in range(n):
            if x_current >= x_end:
                break
            
            # Метод Рунге-Кутты 2-го порядка для системы ОДУ
            k1_y = y_prime_current
            k1_y_prime = equation(x_current, y_current, y_prime_current, **kwargs)
            
            k2_y = y_prime_current + step_size * k1_y_prime / 2
            k2_y_prime = equation(
                x_current + step_size / 2,
                y_current + step_size * k1_y / 2,
                y_prime_current + step_size * k1_y_prime / 2,
                **kwargs
            )
            
            # Обновляем значения
            y_next = y_current + step_size * k2_y
            y_prime_next = y_prime_current + step_size * k2_y_prime
            x_next = x_current + step_size
            
            # Корректируем шаг на последней итерации
            if x_next > x_end:
                step_size_corrected = x_end - x_current
                y_next = y_current + step_size_corrected * k2_y
                y_prime_next = y_prime_current + step_size_corrected * k2_y_prime
                x_next = x_end
            
            x_values.append(x_next)
            y_values.append(y_next)
            y_prime_values.append(y_prime_next)
            
            x_current = x_next
            y_current = y_next
            y_prime_current = y_prime_next
        
        return SolutionResult(
            success=True,
            result=(x_values, y_values),
            message="Cauchy problem solved"
        )


class FiniteDifferenceEdgeSolver(BaseEdgeSolver):
    """Реализация метода конечных разностей для решения краевой задачи"""
    
    @property
    def method_name(self) -> str:
        return "Finite difference method for edge problem"
    
    def solve(
        self,
        equation: Callable[[float, float, float], float],
        boundary_conditions: Tuple[Tuple[float, float], Tuple[float, float]],
        interval: Tuple[float, float],
        step_size: Optional[float] = None,
        **kwargs
    ) -> SolutionResult:
        """
        Решение краевой задачи методом конечных разностей.
        
        Args:
            equation: Функция правой части y'' = f(x, y, y')
            boundary_conditions: Краевые условия ((x0, y0), (x1, y1))
            interval: Интервал решения (x_start, x_end)
            step_size: Шаг сетки
            **kwargs: Дополнительные параметры
        
        Returns:
            SolutionResult с результатами вычислений
        """
        try:
            self._validate_parameters(equation, boundary_conditions, interval, step_size)
            
            # Извлекаем краевые условия
            (x0, y0), (x1, y1) = boundary_conditions
            x_start, x_end = interval
            
            # Если шаг не указан, вычисляем его
            if step_size is None:
                step_size = (x_end - x_start) / 100
            
            # Создаем сетку
            n = int(math.ceil((x_end - x_start) / step_size))
            x_values = [x_start + i * step_size for i in range(n + 1)]
            
            # Корректируем последнюю точку
            x_values[-1] = x_end
            
            # Инициализируем значения y
            y_values = [0.0] * (n + 1)
            y_values[0] = y0
            y_values[-1] = y1
            
            # Итерационный процесс (метод простых итераций)
            iterations = 0
            max_diff = float('inf')
            
            while max_diff > self.precision and iterations < self.max_iterations:
                max_diff = 0.0
                
                for i in range(1, n):
                    # Аппроксимация производных конечными разностями
                    y_prime = (y_values[i+1] - y_values[i-1]) / (2 * step_size)
                    y_double_prime = (y_values[i+1] - 2 * y_values[i] + y_values[i-1]) / (step_size ** 2)
                    
                    # Вычисляем новое значение
                    new_y = y_values[i] + 0.5 * (equation(x_values[i], y_values[i], y_prime, **kwargs) - y_double_prime)
                    
                    # Обновляем максимальное изменение
                    diff = abs(new_y - y_values[i])
                    if diff > max_diff:
                        max_diff = diff
                    
                    y_values[i] = new_y
                
                iterations += 1
            
            return SolutionResult(
                success=True,
                result={'x': x_values, 'y': y_values},
                iterations=iterations,
                error=max_diff,
                message=f"Successfully solved using {self.method_name}"
            )
            
        except Exception as e:
            return SolutionResult(
                success=False,
                result=None,
                message=f"Error in {self.method_name}: {str(e)}"
            )


# Пример конкретного уравнения
def edge_equation(x: float, y: float, y_prime: float, V: float = 5) -> float:
    """Пример уравнения: y'' = 2Vx + Vx² - y - y'"""
    return 2 * V * x + V * x**2 - y - y_prime


def exact_edge_solution(x: float, V: float = 5) -> float:
    """Точное решение примера краевой задачи"""
    return V * x**2 + math.sin(x)


def format_table_row(values: List[float], label: str, width: int = 12) -> str:
    """Форматирование строки таблицы"""
    formatted_values = " ".join(f"{v:>{width}.7f}" for v in values[:10])  # Показываем первые 10 значений
    if len(values) > 10:
        formatted_values += " ..."
    return f"{label:<10} {formatted_values}"


def print_edge_solution_results(
    solver_name: str,
    result: SolutionResult,
    exact_solution_func: Callable[[float], float],
    V: float = 5
):
    """Вывод результатов краевой задачи"""
    if not result.success:
        print(f"\n{solver_name} - Ошибка: {result.message}")
        return
    
    x_values = result.result['x']
    y_values = result.result['y']
    
    # Вычисляем точное решение и погрешности
    y_exact = [exact_solution_func(x, V) for x in x_values]
    errors = [abs(y_num - y_exact) for y_num, y_exact in zip(y_values, y_exact)]
    
    print(f"\n{solver_name}:")
    print("-" * 150)
    print(format_table_row(x_values, "x:"))
    print(format_table_row(y_values, "y_M:"))
    print(format_table_row(y_exact, "y_T:"))
    print(format_table_row(errors, "Погрешн:"))
    print("-" * 150)
    print(f"Итераций: {result.iterations}")
    if result.error is not None:
        print(f"Конечная ошибка: {result.error:.10f}")
    print(f"Сообщение: {result.message}")

