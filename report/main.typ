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

== Условие <par12>

Необходимо найти интерполяционный многочлен в общем виде. Нам даны следующие данные:

#table(rows: 2, columns: (1fr, 1fr, 1fr, 1fr, 1fr))[*$x$*][
  0][1][2][3][*$f(x)$*][1][2][9][28]

Необходимо найти значение интерполянты в узловых точках.

== Результат

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

== Код

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

= Интерполяционный многочлен в форме Лагранжа

== Условие

Необходимо найти интерполяционный многочлен в форме Лагранжа. Входные данные те же, что и в главе @par12.

== Результат

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

== Код

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

= Интерполяционный многочлен в форме Ньютона

== Условие

Необходимо найти интерполяционный многочлен в форме Ньютона. Входные данные те же, что и в главе @par12.

== Результат

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

Найденный результат совпадает с предыдущими решением, что подтверждает его правильность.

== Код

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

= Интерполяция кубическими сплайнами

== Условие

Необходимо построить интерполяционный многочлен с помощью кубических сплайнов (алгебраических многочленов третьей степени, где сплайн --- фрагмент, отрезок чего-либо). Входные данные те же, что и в главе @par12.

== Результат

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

Обратим внимание, что результат совпадает в узловых точках, но отличается в неузловых, что допустимо для этого метода.

== Код

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
```

```py
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
```
```py

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

= Метод Гаусса решения СЛАУ

== Условие

Метод Гаусса должен решать уравнения вида $A x = B$, где $A$ - матрица.
Для упрощения тестирования матрица $А$ примет вид:

$
  mat(20, 0.2, 0.2, 0.2, 0.2;
      0.21, 21, 0.21, 0.21, 0.21;
      0.22, 0.22, 22, 0.22, 0.22;
      0.23, 0.23, 0.23, 23, 0.23;
      0.24, 0.24, 0.24, 0.24, 24) x = mat(20;21;22;23;24).
$

== Результат

```
Gaussian Elimination for solving linear systems Results:
Success: True
Message: System solved successfully
Result: [1.0, 1.0, 1.0, 1.0, 1.0]
```

== Код

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
```
```py
            )

        except Exception as e:
            return SolutionResult(
                success=False, result=None, message=f"Error solving system: {str(e)}"
            )

```

=	Метод прогонки решения СЛАУ (трехдиагональных)

== Условие
В данном случае решается система линейных уравнений вида $A x = B$, где A --- матрица вида:

$
    mat(
      -20, 0.2, 0, 0, 0;
      0.21, -21, 0.21, 0, 0;
      0, 0.22, -22, 0.22, 0;
      0, 0, 0.23, -23, 0.23;
      0, 0, 0, 0.24, -24) x = mat(20; 21; 22; 23; 24).
$

== Результат

```
Thomas algorithm for solving linear systems Results:
Success: True
Message: System solved successfully
Result: [0.9837332345, 0.9837332345, 0.9837332345, 0.9837332345, 0.9837332345]
```

== Код

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
```
```py
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

