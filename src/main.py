import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# === 1. Символьне рівняння руху літака ===
m, v, F_thrust, C_d, rho, S, g, alpha = sp.symbols('m v F_thrust C_d rho S g alpha')  # Змінив a на alpha

# Сила опору повітря
F_drag = - (1 / 2) * C_d * rho * S * v ** 2
# Сила тяжіння
F_gravity = -m * g * sp.sin(alpha)
# Рівняння руху
eq = sp.Eq(m * sp.Derivative(v), F_thrust + F_drag + F_gravity)

# === 2. Підставляємо конкретні значення ===
values = {
    m: 3,  # маса літака (кг)
    F_thrust: 50,  # сила тяги (Н)
    C_d: 0.03,  # коефіцієнт опору
    rho: 1.225,  # густина повітря (кг/м³)
    S: 0.4,  # площа крила (м²)
    g: 9.81,  # прискорення вільного падіння (м/с²)
    alpha: 0,  # кут нахилу траєкторії (радіани)
    'v': 20  # початкова швидкість (м/с)
}

# Підставляємо значення в рівняння
eq_with_values = eq.subs(values)

# === 3. Виведення рівнянь у LaTeX форматі ===
print("📌 **Основне рівняння руху літака:**")
sp.pprint(eq, use_unicode=True)
print("\n📌 **Рівняння після підстановки значень:**")
sp.pprint(eq_with_values, use_unicode=True)

# === 4. Часова шкала та розрахунок швидкості ===
t = np.linspace(0, 30, 300)  # Час (від 0 до 30 секунд, 300 точок)
dt = t[1] - t[0]  # Крок часу
v = np.zeros_like(t)  # Масив для швидкості
a = np.zeros_like(t)  # Масив для прискорення
v[0] = values['v']  # Початкова швидкість

# === 5. Генеруємо випадкові повітряні потоки ===
wind_effect = np.random.uniform(-2, 2, size=len(t))  # Турбулентність в межах ±2 м/с²

# === 6. Чисельне інтегрування рівняння руху ===
alpha_value = float(values[alpha])  # Отримуємо числове значення кута

for i in range(1, len(t)):
    F_drag_val = -0.5 * values[C_d] * values[rho] * values[S] * v[i - 1] ** 2
    F_gravity_val = -values[m] * values[g] * np.sin(alpha_value)  # Використовуємо числове значення

    a[i] = (values[F_thrust] + F_drag_val + F_gravity_val) / values[m] + wind_effect[i]
    v[i] = v[i - 1] + a[i] * dt

# === 7. Візуалізація графіків ===
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# Графік швидкості
ax[0].plot(t, v, label="Швидкість літака", color="blue")
ax[0].set_xlabel("Час (с)")
ax[0].set_ylabel("Швидкість (м/с)")
ax[0].set_title("Зміна швидкості літака у часі (з урахуванням вітру)")
ax[0].legend()
ax[0].grid(True)

# Графік прискорення
ax[1].plot(t, a, label="Прискорення літака", color="red")
ax[1].set_xlabel("Час (с)")
ax[1].set_ylabel("Прискорення (м/с²)")
ax[1].set_title("Зміна прискорення у часі (ефект турбулентності)")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()