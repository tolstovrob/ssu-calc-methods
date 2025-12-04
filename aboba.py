import numpy as np

def f(x, y, V=5):
    return 2 * V * x + V * x**2 - y

# Метод Эйлера 
def euler_method(x0, y0, h, n, V):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0

    for i in range(n):
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h * f(x[i], y[i], V)
    
    return x, y

# Усовершенствованный метод Эйлера 
def improved_euler_method(x0, y0, h, n, V):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0 

    for i in range(n):
        x[i + 1] = x[i] + h
        y_half = y[i] + (h / 2) * f(x[i], y[i], V)
        x_half = x[i] + h / 2
        y[i + 1] = y[i] + h * f(x_half, y_half, V)

    return x, y

def exact_solution(x, V=5):
    return V * x**2

# Параметры
x0 = 1
V = y0 = 5
h = 0.001 # шаг
n = 10

# Вычисление решений
x_euler, y_euler = euler_method(x0, y0, h, n, V)
x_improved, y_improved = improved_euler_method(x0, y0, h, n, V)
y_exact = exact_solution(x_euler)

# Вычисление погрешностей
error_euler = np.abs(y_euler - y_exact)
error_improved = np.abs(y_improved - y_exact)

print("\nМетод Эйлера: ")
print("-" * 130)
print("x:      ", " ".join(f"{x:>10.7f}" for x in x_euler))
print("y_M:    ", " ".join(f"{y:>10.7f}" for y in y_euler))
print("y_T:    ", " ".join(f"{y:>10.7f}" for y in y_exact))
print("Погрешн:", " ".join(f"{e:>10.7f}" for e in error_euler))
print("-" * 130)

print("\nУсовершенствованный метод Эйлера: ")
print("-" * 130)
print("x:      ", " ".join(f"{x:>10.7f}" for x in x_improved))
print("y_M:    ", " ".join(f"{y:>10.7f}" for y in y_improved))
print("y_T:    ", " ".join(f"{y:>10.7f}" for y in y_exact))
print("Погрешн:", " ".join(f"{e:>10.7f}" for e in error_improved))
print("-" * 130)