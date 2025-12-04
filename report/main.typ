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