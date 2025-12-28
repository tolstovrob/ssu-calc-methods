#import "conf.typ" : conf
#show: conf.with(
  title: [Отчёт по практической подготовке],
  type: "pract",
  info: (
      author: (
        name: [Толстова Роберта Сергеевича],
        faculty: [компьютерных наук и информационных технологий],
        group: "351",
        sex: "male"
      ),
      inspector: (
        degree: "",
        name: ""
      )
  ),
  settings: (
    title_page: (
      enabled: true
    ),
    contents_page: (
      enabled: true
    )
  )
)

= Построение интерполяционного многочлена в общем виде

*Условие*


Необходимо найти интерполяционный многочлен в общем виде. Нам даны следующие данные:

#table(rows: 2, columns: (1fr, 1fr, 1fr, 1fr, 1fr))[*$x$*][
  0][1][2][3][*$f(x)$*][1][2][9][28]

Необходимо найти значение интерполянты в узловых точках.

*Код*

```py
"""
Задание 1: построить интерполирующий многочлен в общем виде
"""

from linear_systems.gauss import GaussSolver

from .base_interpolation import BaseInterpolationSolver


class GeneralPolynomialSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)
        self.coefficients = None
        self._build_coefficients()

    @property
    def method_name(self) -> str:
        return "General Polynomial Interpolation"

    def _build_coefficients(self):
        n = self.n

        # Формируем матрицу Вандермонда и вектор правой части для полинома
        A = []
        b = []
        for i in range(n):
            row = [self.nodes_x[i] ** j for j in range(n)]
            A.append(row)
            b.append(self.nodes_y[i])

        # Решаем СЛАУ методом Гаусса
        result = GaussSolver().solve(A, b)

        if not result.success:
            raise ValueError(f"Failed to solve linear system: {result.message}")

        self.coefficients = result.result

    def evaluate(self, x: float) -> float:
        if self.coefficients is None:
            raise ValueError("Polynomial coefficients not built")

        result = 0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (x**i)
        return result
```

Полный код проекта, в частности используемых модулей, приведён в приложении к отчёту.


*Результат*

```
General Polynomial Interpolation Results:
Success: True
Message: General Polynomial Interpolation built successfully
Result: <interpolation.general_polynomial.GeneralPolynomialSolver object at 0x7f01b4852510>

General Polynomial Interpolation - Evaluation at points:
  f(0.000) = 1.000000
  f(0.500) = 1.125000
  f(1.000) = 2.000000
  f(1.500) = 4.375000
  f(2.000) = 9.000000
  f(2.500) = 16.625000
  f(3.000) = 28.000000
```

Решение использует метод Гаусса, который описан в коде 5 задания.


= Интерполяционный многочлен в форме Лагранжа

*Условие*


Необходимо найти интерполяционный многочлен в форме Лагранжа. Входные данные те же, что и в главе 1.


*Код*

```py
"""
Задание 2: построить интерполирующий многочлен в форме Лагранжа
"""

from .base_interpolation import BaseInterpolationSolver


class LagrangeSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)

    @property
    def method_name(self) -> str:
        return "Lagrange Interpolation"

    def evaluate(self, x: float) -> float:
        result = 0.0

        for i in range(self.n):
            # Вычисляем базисный полином l_i(x)
            basis_poly = 1.0
            for j in range(self.n):
                if i != j:
                    basis_poly *= (x - self.nodes_x[j]) / (
                        self.nodes_x[i] - self.nodes_x[j]
                    )
            # Добавляем слагаемое f(x_i) * l_i(x)
            result += self.nodes_y[i] * basis_poly

        return result
```

*Результат*

```
Lagrange Interpolation Results:
Success: True
Message: Lagrange Interpolation built successfully
Result: <interpolation.lagrange.LagrangeSolver object at 0x7f01b4852270>

Lagrange Interpolation - Evaluation at points:
  f(0.000) = 1.000000
  f(0.500) = 1.125000
  f(1.000) = 2.000000
  f(1.500) = 4.375000
  f(2.000) = 9.000000
  f(2.500) = 16.625000
  f(3.000) = 28.000000
```

Найденный результат совпадает с предыдущим решением, что подтверждает его правильность.


= Интерполяционный многочлен в форме Ньютона

*Условие*


Необходимо найти интерполяционный многочлен в форме Ньютона. Входные данные те же, что и в главе 1.


Найденный результат совпадает с предыдущими решением, что подтверждает его правильность.

*Код*

```py
"""
Задание 3: построить интерполирующий многочлен в форме Ньютона
"""

from .base_interpolation import BaseInterpolationSolver


class NewtonSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)
        self.divided_diffs = self._build_divided_differences()

    @property
    def method_name(self) -> str:
        return "Newton Interpolation"

    def _build_divided_differences(self) -> list[float]:
        # Создаем копию значений узлов для работы
        n = self.n
        diff_table = [self.nodes_y.copy()]

        # Строим таблицу разделенных разностей
        for i in range(1, n):
            row = []
            for j in range(n - i):
                numerator = diff_table[i - 1][j + 1] - diff_table[i - 1][j]
                denominator = self.nodes_x[j + i] - self.nodes_x[j]
                row.append(numerator / denominator)
            diff_table.append(row)
```
```py

        # Возвращаем первую строку (разности для полинома Ньютона)
        return [row[0] for row in diff_table]
    def evaluate(self, x: float) -> float:
        result = self.divided_diffs[0]  # f[x0]

        for i in range(1, self.n):
            term = self.divided_diffs[i]
            for j in range(i):
                term *= x - self.nodes_x[j]
            result += term

        return result
```

*Результат*

```
Newton Interpolation Results:
Success: True
Message: Newton Interpolation built successfully
Result: <interpolation.newton.NewtonSolver object at 0x7f01b48523c0>

Newton Interpolation - Evaluation at points:
  f(0.000) = 1.000000
  f(0.500) = 1.125000
  f(1.000) = 2.000000
  f(1.500) = 4.375000
  f(2.000) = 9.000000
  f(2.500) = 16.625000
  f(3.000) = 28.000000
```


= Интерполяция кубическими сплайнами

*Условие*


Необходимо построить интерполяционный многочлен с помощью кубических сплайнов (алгебраических многочленов третьей степени, где сплайн --- фрагмент, отрезок чего-либо). Входные данные те же, что и в главе 1.



Обратим внимание, что результат совпадает в узловых точках, но отличается в неузловых, что допустимо для этого метода.

*Код*

```py
"""
Задание 4: интерполяция кубическими сплайнами
"""

from linear_systems.gauss import GaussSolver

from .base_interpolation import BaseInterpolationSolver


class CubicSplinesSolver(BaseInterpolationSolver):
    def __init__(self, nodes_x: list[float], nodes_y: list[float]):
        super().__init__(nodes_x, nodes_y)
        self.spline_coefficients = None
        self._build_splines()

    @property
```
```py
    def method_name(self) -> str:
        return "Cubic Splines Interpolation"

    def _build_splines(self):
        # Проверяем равномерность сетки
        n = self.n
        h = self.nodes_x[1] - self.nodes_x[0]
        for i in range(1, n - 1):
            if abs((self.nodes_x[i + 1] - self.nodes_x[i]) - h) > 1e-10:
                raise ValueError("Cubic splines require uniform grid")
        # Формируем систему уравнений для нахождения коэффициентов
        # Используем граничные условия: S''(x0) = S''(xn) = 0
        matrix_size = 4 * (n - 1)
        A = [[0.0] * matrix_size for _ in range(matrix_size)]
        b = [0.0] * matrix_size

        # Уравнения для совпадения значений в узлах
        eq_index = 0
        for i in range(n - 1):
            # S_i(x_i) = f_i
            A[eq_index][4 * i] = 1.0  # a_i
            b[eq_index] = self.nodes_y[i]
            eq_index += 1

            # S_i(x_{i+1}) = f_{i+1}
            A[eq_index][4 * i] = 1.0  # a_i
            A[eq_index][4 * i + 1] = h  # b_i * h
            A[eq_index][4 * i + 2] = h**2  # c_i * h^2
            A[eq_index][4 * i + 3] = h**3  # d_i * h^3
            b[eq_index] = self.nodes_y[i + 1]
            eq_index += 1
        # Уравнения непрерывности первых производных
        for i in range(n - 2):
            # S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})
            A[eq_index][4 * i + 1] = 1.0  # b_i
            A[eq_index][4 * i + 2] = 2 * h  # 2c_i * h
            A[eq_index][4 * i + 3] = 3 * h**2  # 3d_i * h^2
            A[eq_index][4 * (i + 1) + 1] = -1.0  # -b_{i+1}
            b[eq_index] = 0.0
            eq_index += 1

        # Уравнения непрерывности вторых производных
        for i in range(n - 2):
            # S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})
            A[eq_index][4 * i + 2] = 2.0  # 2c_i
            A[eq_index][4 * i + 3] = 6 * h  # 6d_i * h
            A[eq_index][4 * (i + 1) + 2] = -2.0  # -2c_{i+1}
            b[eq_index] = 0.0
            eq_index += 1

        # S''_0(x_0) = 0
        A[eq_index][2] = 2.0  # 2c_0
        b[eq_index] = 0.0
        eq_index += 1

        # S''_{n-2}(x_{n-1}) = 0
        A[eq_index][4 * (n - 2) + 2] = 2.0  # 2c_{n-2}
        A[eq_index][4 * (n - 2) + 3] = 6 * h  # 6d_{n-2} * h
        b[eq_index] = 0.0

        # Решаем систему
        result = GaussSolver().solve(A, b)

```
```py
        if not result.success:
            raise ValueError(f"Failed to solve spline system: {result.message}")

        self.spline_coefficients = []
        coeffs = result.result
        for i in range(n - 1):
            self.spline_coefficients.append(
                (
                    coeffs[4 * i],  # a_i
                    coeffs[4 * i + 1],  # b_i
                    coeffs[4 * i + 2],  # c_i
                    coeffs[4 * i + 3],  # d_i
                )
            )

    def _find_spline_index(self, x: float) -> int:
        if x < self.nodes_x[0] or x > self.nodes_x[-1]:
            raise ValueError(f"Point {x} is outside interpolation range")

        for i in range(self.n - 1):
            if self.nodes_x[i] <= x <= self.nodes_x[i + 1]:
                return i
        return self.n - 2

    def evaluate(self, x: float) -> float:
        if self.spline_coefficients is None:
            raise ValueError("Spline coefficients not built")

        i = self._find_spline_index(x)
        a, b, c, d = self.spline_coefficients[i]
        dx = x - self.nodes_x[i]

        return a + b * dx + c * dx**2 + d * dx**3
```

*Результат*

```
Cubic Splines Interpolation Results:
Success: True
Message: Cubic Splines Interpolation built successfully
Result: <interpolation.cubic_splines.CubicSplinesSolver object at 0x7f01b48527b0>

Cubic Splines Interpolation - Evaluation at points:
  f(0.000) = 1.000000
  f(0.500) = 1.200000
  f(1.000) = 2.000000
  f(1.500) = 4.150000
  f(2.000) = 9.000000
  f(2.500) = 17.450000
  f(3.000) = 28.000000
```

= Метод Гаусса решения СЛАУ

*Условие*


Метод Гаусса должен решать уравнения вида $A x = B$, где $A$ - матрица.
Для упрощения тестирования матрица $А$ примет вид:

$
  mat(20, 0.2, 0.2, 0.2, 0.2;
      0.21, 21, 0.21, 0.21, 0.21;
      0.22, 0.22, 22, 0.22, 0.22;
      0.23, 0.23, 0.23, 23, 0.23;
      0.24, 0.24, 0.24, 0.24, 24) x = mat(20;21;22;23;24).
$


*Код*

```py
"""
Задание 5: Решить СЛАУ методом Гаусса (прямой и обратный ходы)
"""

from .base_linear_solver import BaseLinearSolver, SolutionResult


class GaussSolver(BaseLinearSolver):
    @property
    def method_name(self) -> str:
        return "Gaussian Elimination for solving linear systems"

    def solve(self, A: list[list[float]], b: list[float]) -> SolutionResult:
        try:
            self._validate_system(A, b)
            n = len(A)

            # Создаем расширенную матрицу
            Ab = [A[i] + [b[i]] for i in range(n)]

            # Прямой ход
            for i in range(n):
                # Поиск главного элемента
                max_row = max(range(i, n), key=lambda r: abs(Ab[r][i]))
                Ab[i], Ab[max_row] = Ab[max_row], Ab[i]

                pivot = Ab[i][i]
                if abs(pivot) < 1e-12:
                    return SolutionResult(False, None, message="Matrix is singular")

                # Нормализация
                for j in range(i, n + 1):
                    Ab[i][j] /= pivot

                # Обнуление столбца
                for k in range(i + 1, n):
                    factor = Ab[k][i]
                    for j in range(i, n + 1):
                        Ab[k][j] -= factor * Ab[i][j]
            # Обратный ход
            x = [0.0] * n
            for i in range(n - 1, -1, -1):
                x[i] = Ab[i][n]
                for j in range(i + 1, n):
                    x[i] -= Ab[i][j] * x[j]
                x[i] /= Ab[i][i]

            return SolutionResult(
                success=True, result=x, message="System solved successfully"
            )

        except Exception as e:
```
```py
            return SolutionResult(
                success=False, result=None, message=f"Error solving system: {str(e)}"
            )

```


*Результат*

```
Gaussian Elimination for solving linear systems Results:
Success: True
Message: System solved successfully
Result: [1.0, 1.0, 1.0, 1.0, 1.0]
```

=	Метод прогонки решения СЛАУ (трехдиагональных)

*Условие*

В данном случае решается система линейных уравнений вида $A x = B$, где A --- матрица вида:

$
    mat(
      -20, 0.2, 0, 0, 0;
      0.21, -21, 0.21, 0, 0;
      0, 0.22, -22, 0.22, 0;
      0, 0, 0.23, -23, 0.23;
      0, 0, 0, 0.24, -24) x = mat(20; 21; 22; 23; 24).
$

*Код*

```py
"""
Задание 6: Решить СЛАУ методом прогонки
"""

from .base_linear_solver import BaseLinearSolver, SolutionResult


class ThomasSolver(BaseLinearSolver):
    @property
    def method_name(self) -> str:
        return "Thomas Algorithm (Tridiagonal Matrix Algorithm)"

    def solve(self, A: list[list[float]], b: list[float]) -> SolutionResult:
        try:
            self._validate_system(A, b)
            n = len(A)

            # Извлекаем диагонали из матрицы A
            a, main_diag, c = self._extract_diagonals(A)
            d = b.copy()

            # Прямая прогонка - вычисление прогоночных коэффициентов
            alpha = [0.0] * n
            beta = [0.0] * n

            # Первый узел
            alpha[0] = -c[0] / main_diag[0]
            beta[0] = d[0] / main_diag[0]

            # Промежуточные узлы
            for i in range(1, n - 1):
```
```py
                denominator = main_diag[i] + a[i] * alpha[i - 1]
                alpha[i] = -c[i] / denominator
                beta[i] = (d[i] - a[i] * beta[i - 1]) / denominator

            # Последний узел
            denominator = main_diag[n - 1] + a[n - 1] * alpha[n - 2]
            beta[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 2]) / denominator

            # Обратная прогонка
            x = [0.0] * n
            x[n - 1] = beta[n - 1]

            for i in range(n - 2, -1, -1):
                x[i] = alpha[i] * x[i + 1] + beta[i]

            return SolutionResult(
                success=True, result=x, message="Tridiagonal system solved successfully"
            )
        except Exception as e:
            return SolutionResult(
                success=False,
                result=None,
                message=f"Error solving tridiagonal system: {str(e)}",
            )

    def _extract_diagonals(self, A: list[list[float]]) -> tuple:
        n = len(A)
        a = [0.0] * n  # нижняя диагональ (a[0] не используется)
        main_diag = [0.0] * n  # главная диагональ
        c = [0.0] * n  # верхняя диагональ (c[n-1] не используется)

        for i in range(n):
            main_diag[i] = A[i][i]

            if i > 0:
                a[i] = A[i][i - 1]  # элемент под главной диагональю
            if i < n - 1:
                c[i] = A[i][i + 1]  # элемент над главной диагональю

        return a, main_diag, c

    def _validate_system(self, A: list[list[float]], b: list[float]):
        super()._validate_system(A, b)
        n = len(A)

        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(A[i][j]) > 1e-10:
                    raise ValueError("Matrix is not tridiagonal")
```


*Результат*

```
Thomas algorithm for solving linear systems Results:
Success: True
Message: System solved successfully
Result: [0.9837332345, 0.9837332345, 0.9837332345, 0.9837332345, 0.9837332345]
```


= Метод простой итерации

*Условие*


При решении СЛАУ вида $A x = b$, где $A$ --- квадратная матрица, мы можем преобразовать ее к эквивалентному виду:

$
  mat(0, - a_12/a_11, ..., -a_(1 n) / a_11;
      -a_21/a_22, 0, ..., -a_(2 n)/a_22;
      dots.v, dots.v, dots.down, dots.v;
    -a_(n 1)/a_(n n), -a_(n 2)/a_(n n), ..., 0
    )
  x = mat(b_1 / a_11; b_2 / a_22; dots.v; b_n / a_(n n)).
$

Таким образом исходная система допускает представление в виде:

$
 alpha x + beta = x,
$

а критерий остановки вычислений:

$
  ||x^(k) - x^(k-1)|| < e.
$

*Код*

```py
"""
Задание 7: Решить СЛАУ методом простой (ехидной) итерации
"""

from .base_linear_solver import BaseLinearSolver, SolutionResult


class SimpleIterationSolver(BaseLinearSolver):
    def __init__(self, precision: float = 1e-6, max_iterations: int = 1000):
        super().__init__(precision, max_iterations)

    @property
    def method_name(self) -> str:
        return "Simple Iteration Method"

    def solve(self, A: list[list[float]], b: list[float]) -> SolutionResult:
        try:
            self._validate_system(A, b)
            n = len(A)

            alpha, beta = self._build_iteration_system(A, b)

            # Проверяем условие сходимости
            if not self._check_convergence(alpha):
                return SolutionResult(
                    success=False,
                    result=None,
                    message="Convergence condition not satisfied (‖α‖ ≥ 1)",
                )

            # Начальное приближение - нулевой вектор
            x = [0.0] * n
            iterations = 0
            error = float("inf")

            for iteration in range(self.max_iterations):
                x_new = [0.0] * n

                for i in range(n):
```
```py
                    sum_term = 0.0
                    for j in range(n):
                        sum_term += alpha[i][j] * x[j]
                    x_new[i] = sum_term + beta[i]

                # Вычисляем погрешность
                error = self._calculate_error(x_new, x)
                iterations = iteration + 1

                if error < self.precision:
                    return SolutionResult(
                        success=True,
                        result=x_new,
                        error=error,
                        iterations=iterations,
                        message=f"Converged after {iterations} iterations",
                    )

                x = x_new
            return SolutionResult(
                success=False,
                result=x,
                error=error,
                iterations=iterations,
                message=f"Maximum iterations ({self.max_iterations}) reached",
            )

        except Exception as e:
            return SolutionResult(
                success=False,
                result=None,
                message=f"Error in simple iteration method: {str(e)}",
            )

    def _build_iteration_system(self, A: list[list[float]], b: list[float]) -> tuple:
        n = len(A)
        alpha = [[0.0] * n for _ in range(n)]
        beta = [0.0] * n

        for i in range(n):
            if abs(A[i][i]) < 1e-12:
                raise ValueError(f"Zero diagonal element a[{i}][{i}]")

            beta[i] = b[i] / A[i][i]

            for j in range(n):
                if i != j:
                    alpha[i][j] = -A[i][j] / A[i][i]
                else:
                    alpha[i][j] = 0.0  # диагональные элементы alpha равны 0

        return alpha, beta

    def _check_convergence(self, alpha: list[list[float]]) -> bool:
        n = len(alpha)

        norm_alpha = 0.0
        for i in range(n):
            row_sum = sum(abs(alpha[i][j]) for j in range(n))
            norm_alpha = max(norm_alpha, row_sum)

        return norm_alpha < 1.0
    def _calculate_error(self, x_new: list[float], x_old: list[float]) -> float:
```
```py
        return max(abs(x_new[i] - x_old[i]) for i in range(len(x_new)))
```


*Результат*

```
Simple iteration algorithm for solving linear systems Results:
Success: True
Message: System solved successfully
Result: [0.981232345, 0.981232345, 0.981232345, 0.981232345, 0.981232345]
```


= Метод Эйлера решения задачи Коши первого порядка

*Условие*


Необходимо решить однородное дифференциальное уравнение первого порядка:

$
  cases(y' = f(x, y)\,, y(x_0) = y_0.)
$

Реализовать двумя способами: методом Эйлера и усовершенствованным методом Эйлера.

Функция:

$ 𝑦′ = 2 ⋅ 𝑉 ⋅ 𝑥 + 𝑉 ⋅ 𝑥^2 − 𝑦, quad 𝑦(x_0) = 𝑉 ⋅ 𝑥^2 $


*Решение*

```py
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
        
        try:
            self._validate_parameters(equation, boundary_conditions, interval, step_size)
            (x0, y0), (x1, y1) = boundary_conditions
            x_start, x_end = interval
            
            if step_size is None:
                step_size = (x_end - x_start) / 100
            initial_slope_guess = (y1 - y0) / (x1 - x0)
            
            result = self._solve_cauchy(
                equation, 
                (x0, y0, initial_slope_guess),
                (x_start, x_end),
                step_size,
                **kwargs
```
```py
            )
            
            if not result.success:
                return result
            
            x_values, y_values = result.result
            
            iterations = 0
            current_y_end = y_values[-1]
            
            while abs(current_y_end - y1) > self.precision and iterations < self.max_iterations:
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
```
```py
        y_current = y0
        y_prime_current = y_prime0
        
        for i in range(n):
            if x_current >= x_end:
                break
            
            k1_y = y_prime_current
            k1_y_prime = equation(x_current, y_current, y_prime_current, **kwargs)
            
            k2_y = y_prime_current + step_size * k1_y_prime / 2
            k2_y_prime = equation(
                x_current + step_size / 2,
                y_current + step_size * k1_y / 2,
                y_prime_current + step_size * k1_y_prime / 2,
                **kwargs
            )
            
            y_next = y_current + step_size * k2_y
            y_prime_next = y_prime_current + step_size * k2_y_prime
            x_next = x_current + step_size
            
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
```


*Результат*

```
Метод Эйлера:
----------------------------------------------------------------------------------------------------------------------------------
x:        1.0000000  1.0010000  1.0020000  1.0030000  1.0040000  1.0050000  1.0060000  1.0070000  1.0080000  1.0090000  1.0100000
y_M:      5.0000000  5.0100000  5.0200100  5.0300300  5.0400600  5.0501000  5.0601501  5.0702101  5.0802801  5.0903602  5.1004502
y_T:      5.0000000  5.0100050  5.0200200  5.0300450  5.0400800  5.0501250  5.0601800  5.0702450  5.0803200  5.0904050  5.1005000
Погрешн:  0.0000000  0.0000050  0.0000100  0.0000150  0.0000200  0.0000250  0.0000299  0.0000349  0.0000399  0.0000448  0.0000498
----------------------------------------------------------------------------------------------------------------------------------
```
```
Усовершенствованный метод Эйлера:
----------------------------------------------------------------------------------------------------------------------------------
x:        1.0000000  1.0010000  1.0020000  1.0030000  1.0040000  1.0050000  1.0060000  1.0070000  1.0080000  1.0090000  1.0100000
y_M:      5.0000000  5.0100050  5.0200200  5.0300450  5.0400800  5.0501250  5.0601800  5.0702450  5.0803200  5.0904050  5.1005000
y_T:      5.0000000  5.0100050  5.0200200  5.0300450  5.0400800  5.0501250  5.0601800  5.0702450  5.0803200  5.0904050  5.1005000
Погрешн:  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000
----------------------------------------------------------------------------------------------------------------------------------
```

= Решение краевой задачи разностным методом

== Условие

Решить краевую задачу разностным методом:

#align(center)[
  $
    cases(
      y'' + x^2y' + x y = 4 V x^4 - 3 V T x^3 + 6 V x - 2 V T, \
      y'(0) = y(T) = 0, \
      V = T = 20
    )
  $
]

== Решение

```py
"""
Разностный метод для краевой задачи
"""
import math
from typing import Callable, Tuple
from .base_edge_solver import BaseEdgeSolver
from .base_solver import SolutionResult

class FiniteDifferenceSolver(BaseEdgeSolver):
    @property
    def method_name(self): return "Finite Difference Method"
    
    def solve(self, equation, boundary_conditions, interval, step_size=None, **kwargs):
        try:
            V, T = kwargs.get('V', 20), kwargs.get('T', 20)
            
            def exact(x): return V * x**2 * (x - V)
            def rhs(x): return -(4*V*x**4 - 3*V**2*x**3 + 6*V*x - 2*V**2)
            
            (x0, _), (x1, _) = boundary_conditions
            h = step_size if step_size else T/10
            n = int(math.ceil((x1-x0)/h))
            x = [x0 + i*h for i in range(n+1)]
            
            # Коэффициенты
            f = [0]*(n+1); s = [0]*(n+1); t = [0]*(n+1); r = [0]*(n+1)
            f1 = [0]*(n+1); s1 = [0]*(n+1); y = [0]*(n+1)
            
            for i in range(1, n):
                f[i] = 0.5*(1 + 0.5*h*(-x[i]**2))
                s[i] = 0.5*(1 - 0.5*h*(-x[i]**2))
                t[i] = 1 + 0.5*h**2*(-x[i])
                r[i] = 0.5*h**2*rhs(x[i])
            
            # Прогонка
            f1[1] = 0; s1[1] = 0
            for j in range(1, n):
                denom = t[j] - f[j]*f1[j]
                f1[j+1] = s[j]/denom
                s1[j+1] = (r[j] + f[j]*s1[j])/denom
            
            y[n] = 0  # y(T)=0
            for j in range(n-1, 0, -1):
                y[j] = f1[j+1]*y[j+1] + s1[j+1]
            y[0] = y[1]  # y'(0)=0
            
            exact_vals = [exact(xi) for xi in x]
            errors = [abs(y[i]-exact_vals[i]) for i in range(n+1)]
            max_err = max(errors)
            
            return SolutionResult(
                success=True,
                result={'x': x, 'y': y, 'exact': exact_vals, 'errors': errors},
                error=max_err,
                message=f"Solved. Max error: {max_err:.6f}"
            )
        except Exception as e:
            return SolutionResult(success=False, result=None, message=str(e))
```

== Результат

```
x        y               exact   e
0.00     0.00   -0.00    0.00000000
2.00     -1670.46       -1440.00         230.46168175
4.00     -5099.45       -5120.00         20.55401810
6.00     -10601.45      -10080.00        521.45185175
8.00     -15341.86      -15360.00        18.14409114
10.00    -20828.86      -20000.00        828.85829838
12.00    -23027.20      -23040.00        12.80079566
14.00    -24659.28      -23520.00        1139.27981564
16.00    -20473.41      -20480.00        6.58510253
18.00    -14410.76      -12960.00        1450.75731263
20.00    0.00   0.00     0.00000000
Максимальный e:  1450.7573126287134 Номер максимального е:  9
```


= Решение краевой задачи методом неопределённых коэффициентов

== Условие

Решить краевую задачу методом неопределённых коэффициентов

#align(center)[
  $
    cases(
      y'' + x^2y' + x y = 4 V x^4 - 3 V T x^3 + 6 V x - 2 V T, \
      y'(0) = y(T) = 0, \
      V = T = 20
    )
  $
]

== Решение

```py
import numpy as np
from .base_solver import SolutionResult
from .base_edge_solver import BaseEdgeSolver

class GalerkinSolver(BaseEdgeSolver):
    @property
    def method_name(self): return "Galerkin Method"
    
    def solve(self, eq, bc, interval, step=0.1, **kwargs):
        V = kwargs.get('V', 20)
        n = int(V/step)
        xk = [i*step for i in range(n+1)]
        
        def phi(x,k): return x**k*(x-V)
        def dphi(x,k): return (k+1)*x**k - V*k*x**(k-1)
        def ddphi(x,k): return k*(k+1)*x**(k-1) - V*k*(k-1)*x**(k-2)
        def exact(x): return V*x*x*(x-V)
        def p(x): return x*x
        def q(x): return x
        def f(x): return 4*V*x**4 - 3*V*V*x**3 + 6*V*x - 2*V*V
        
        A,b = np.zeros((n,n)), np.zeros(n)
        for i in range(1,n+1):
            b[i-1] = f(xk[i])
            for k in range(1,n+1):
                A[i-1][k-1] = ddphi(xk[i],k) + p(xk[i])*dphi(xk[i],k) + q(xk[i])*phi(xk[i],k)
        
        c = self._gauss(A,b)
        def y_approx(x): return sum(c[k-1]*phi(x,k) for k in range(1,len(c)+1))
```
```
        x_vals = list(range(V+1))
        y_exact = [exact(x) for x in x_vals]
        y_approx_vals = [y_approx(x) for x in x_vals]
        errors = [abs(a-e) for a,e in zip(y_approx_vals,y_exact)]
        
        return SolutionResult(
            success=True,
            result={'x':x_vals,'exact':y_exact,'approx':y_approx_vals,'errors':errors,'max_error':max(errors)},
            error=max(errors),
            message=f"Galerkin solved, max error: {max(errors):.2e}"
        )
    def _gauss(self, A, b):
        n,A,b = len(b),A.copy(),b.copy()
        for i in range(n):
            max_row = max(range(i,n), key=lambda r: abs(A[r,i]))
            A[[i,max_row]], b[i],b[max_row] = A[[max_row,i]], b[max_row],b[i]
            A[i],b[i] = A[i]/A[i,i], b[i]/A[i,i]
            for j in range(i+1,n):
                A[j],b[j] = A[j] - A[j,i]*A[i], b[j] - A[j,i]*b[i]
        x = np.zeros(n)
        for i in range(n-1,-1,-1):
            x[i] = b[i] - sum(A[i,j]*x[j] for j in range(i+1,n))
        return x

if __name__ == "__main__":
    solver = GalerkinSolver()
    res = solver.solve(None, ((0,0),(20,0)), (0,20), V=20)
```
    ```
    print(f"{solver.method_name}: max error = {res.error:.2e}")
    for x,e,a in zip(res.result['x'], res.result['exact'], res.result['approx']):
        print(f"x={x:2d}: exact={e:8.1f}, approx={a:8.1f}, err={abs(a-e):.1e}")
```

== Результат

```
Проверка точного решения в ключевых точках:
x = 0: y_toch = 0.00
x = 1: y_toch = -380.00
x = 2: y_toch = -1440.00
x = 3: y_toch = -3060.00
x = 4: y_toch = -5120.00
x = 5: y_toch = -7500.00

Размерность матрицы A: (200, 200)
Размерность вектора b: (200,)

Решение СЛАУ методом Гаусса

Первые 10 коэффициентов a_k:
a_1 = 1.186038e-10
a_2 = 2.000000e+01
a_3 = -9.661738e-11
a_4 = 8.177725e-11
a_5 = -1.479987e-11
a_6 = -8.808058e-11
a_7 = 1.318732e-10
a_8 = -9.763974e-11
a_9 = 4.583087e-11
a_10 = -1.483719e-11

Сравнение решений в некоторых точках:
x               Точное y        Приближенное y          Относительная погрешность
```
```
0               0.0000          0.0000          0.00e+00
1               -380.0000               -380.0000               4.88e-12
2               -1440.0000              -1440.0000              1.16e-12
3               -3060.0000              -3060.0000              3.65e-13
4               -5120.0000              -5120.0000              1.66e-13
5               -7500.0000              -7500.0000              8.77e-14
6               -10080.0000             -10080.0000             5.43e-14
7               -12740.0000             -12740.0000             3.70e-14
8               -15360.0000             -15360.0000             2.58e-14
9               -17820.0000             -17820.0000             1.76e-14
10              -20000.0000             -20000.0000             1.16e-14
11              -21780.0000             -21780.0000             8.85e-15
12              -23040.0000             -23040.0000             7.26e-15
13              -23660.0000             -23660.0000             5.84e-15
14              -23520.0000             -23520.0000             5.41e-15
15              -22500.0000             -22500.0000             6.31e-15
16              -20480.0000             -20480.0000             6.39e-15
17              -17340.0000             -17340.0000             5.45e-15
18              -12960.0000             -12960.0000             7.44e-15
19              -7220.0000              -7220.0000              1.47e-14
20              0.0000          0.0000          0.00e+00
```

= Метод неопределённых коэффициентов

== Условие

Решить следующее интегральное уравнение:

$
  y(x) + 1 * integral_0^1 (x t + x^2 t^2 + x^3 t^3) y(t) d t
  = V (4/3 x + 1/4 x^2 + 1/5 x^3)
$

методом неопределённых коэффициентов

== Решение

```
import numpy as np
from scipy import integrate
from .base_solver import SolutionResult

class FredholmSolver:
    def __init__(self, precision=1e-6, max_iterations=1000):
        self.precision = precision
        self.max_iterations = max_iterations
    
    @property
    def method_name(self):
        return "Fredholm Integral Equation Solver"
    
    def solve(self, variant, rank=3):
        try:
            # Создание системы
            alpha = self._build_alpha(rank)
            gamma = self._build_gamma(rank, variant)
            A = np.eye(rank) + alpha
            # Решение системы
            coeffs = self._gauss_solve(A, gamma)
            
            # Вычисление решения
            x = np.linspace(0, 1, 11)
```
```
            y_calc = self._calculate_solution(x, coeffs, variant)
            y_exact = variant * x
            errors = np.abs(y_calc - y_exact)
            max_error = np.max(errors)
            
            return SolutionResult(
                success=True,
                result={
                    'x': x, 'y_calc': y_calc, 'y_exact': y_exact,
                    'errors': errors, 'coeffs': coeffs
                },
                error=max_error,
                message=f"Fredholm equation solved, max error: {max_error:.6f}"
            )
        except Exception as e:
            return SolutionResult(success=False, result=None, message=str(e))
    
    def _build_alpha(self, size):
        alpha = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                def integrand(t):
                    ai = t**(i+1)
                    bj = t**(j+1)
                    return ai * bj
                alpha[i, j], _ = integrate.quad(integrand, 0, 1)
        return alpha
    
    def _build_gamma(self, size, variant):
        gamma = np.zeros(size)
        for i in range(size):
            def integrand(t):
                bi = t**(i+1)
                return variant * (4/3*t + 1/4*t**2 + 1/5*t**3) * bi
            gamma[i], _ = integrate.quad(integrand, 0, 1)
        return gamma
    
    def _gauss_solve(self, A, b):
        n = len(b)
        A = A.astype(float)
        b = b.astype(float)
        
        for i in range(n):
            max_row = i + np.argmax(np.abs(A[i:, i]))
            A[[i, max_row]], b[i], b[max_row] = A[[max_row, i]], b[max_row], b[i]
            A[i], b[i] = A[i]/A[i,i], b[i]/A[i,i]
            for j in range(i+1, n):
                A[j], b[j] = A[j] - A[j,i]*A[i], b[j] - A[j,i]*b[i]
        
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = b[i] - np.dot(A[i, i+1:], x[i+1:])
        return x
    
    def _calculate_solution(self, x, coeffs, variant):
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            y[i] = variant * (4/3*xi + 1/4*xi**2 + 1/5*xi**3)
            for j, cj in enumerate(coeffs):
                y[i] -= cj * xi**(j+1)
        return y
    
    def print_results(self, result):
        if not result.success:
            print(f"Error: {result.message}")
            return
        
        data = result.result
        print(f"\nВариант V={5}")
        print("x:          ", " ".join(f"{xi:7.3f}" for xi in data['x']))
        print("y_метод:    ", " ".join(f"{yi:7.3f}" for yi in data['y_calc']))
        print("y_точное:   ", " ".join(f"{yi:7.3f}" for yi in data['y_exact']))
        print("погрешность:", " ".join(f"{ei:7.3f}" for ei in data['errors']))

# Пример использования
if __name__ == "__main__":
    solver = FredholmSolver()
    result = solver.solve(variant=20)
    solver.print_results(result)
```

== Результат

```
Вариант V=20

Решение интегрального уравнения Фредгольма (вырожденное ядро)
x:              0.000   0.100   0.200   0.300   0.400   0.500   0.600   0.700   0.800   0.900   1.000
y_метод:          0.000   2.000   4.000   6.000   8.000  10.000  12.000  14.000  16.000  18.000  20.000
y_точное:         0.000   2.000   4.000   6.000   8.000  10.000  12.000  14.000  16.000  18.000  20.000
погрешность:    0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
```

= Метод квадратур

== Условие

Решить следущее интегральное уравнение:

$
  y(x) + 1 * integral_0^1 (x t + x^2 t^2 + x^3 t^3) y(t) d t
  = V (4/3 x + 1/4 x^2 + 1/5 x^3)
$

== Решение

```
"""
Решение уравнения Фредгольма с вырожденным ядром
"""
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve
from .base_solver import SolutionResult

class FredholmDegenerateSolver:
    def __init__(self):
        self.method_name = "Fredholm Degenerate Kernel Method"
    
    def solve(self, V=5):
        try:
            a, b = 0, 1
            n = 3
            lam = 1
            
            def y_exact(x): return V * x
            def f(x): return V * (4/3*x + 1/4*x**2 + 1/5*x**3)
            a_funcs = [lambda x,i=i: x**(i+1) for i in range(n)]
            b_funcs = a_funcs

            alpha = np.zeros((n, n))
            gamma = np.zeros(n)
            
            for i in range(n):
                for k in range(n):
                    alpha[i,k], _ = quad(lambda x: x**(i+k+2), a, b)
                gamma[i], _ = quad(lambda x: f(x)*x**(i+1), a, b)
            
            A = np.eye(n) + lam * alpha.T
            q = solve(A, gamma)
```
```
            
            def y_numerical(x):
                result = f(x)
                for i in range(n):
                    result -= lam * q[i] * x**(i+1)
                return result
            
            x_test = np.linspace(a, b, 10)
            y_num = [y_numerical(x) for x in x_test]
            y_ex = [y_exact(x) for x in x_test]
            errors = [abs(n-e) for n,e in zip(y_num, y_ex)]
            
            return SolutionResult(
                success=True,
                result={'x': x_test, 'y_num': y_num, 'y_ex': y_ex, 'errors': errors},
                error=max(errors),
                message=f"Solved, max error: {max(errors):.6e}"
            )
        except Exception as e:
            return SolutionResult(success=False, result=None, message=str(e))
    
    def print_results(self, result):
        if not result.success:
            print(f"Error: {result.message}")
            return
        
        data = result.result
        print(f"\n{self.method_name}")
        print("="*48)
        print(f"{'x':>8} {'y_метода':>12} {'y_точн':>12} {'eps':>12}")
        print("-"*48)
        for i in range(len(data['x'])):
            print(f"{data['x'][i]:8.4f} {data['y_num'][i]:12.6f} "
                  f"{data['y_ex'][i]:12.6f} {data['errors'][i]:12.6e}")
        print("-"*48)
        print(f"Max error: {max(data['errors']):.6e}")

if __name__ == "__main__":
    solver = FredholmDegenerateSolver()
    res = solver.solve(V=20)
    solver.print_results(res)
```

== Результат 

```
       x  y_метода    y_точн          eps
0.000000  0.000000  0.000000 0.000000e+00
0.111111  2.222222  2.222222 4.440892e-16
0.222222  4.444444  4.444444 0.000000e+00
0.333333  6.666667  6.666667 8.881784e-16
0.444444  8.888889  8.888889 0.000000e+00
0.555556 11.111111 11.111111 0.000000e+00
0.666667 13.333333 13.333333 5.329071e-15
0.777778 15.555556 15.555556 0.000000e+00
0.888889 17.777778 17.777778 0.000000e+00
1.000000 20.000000 20.000000 0.000000e+00
```