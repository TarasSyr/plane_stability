import numpy as np
import matplotlib.pyplot as plt

# Параметри
t = np.linspace(0, 60, 600)  # 60 секунд, 600 точок
v0 = 80  # початкова швидкість

# Базова кінематична модель
v_base = v0 - 2 * np.exp(-0.2 * t)

# Пориви вітру: шум + синусоїдальні збурення
np.random.seed(42)  # для стабільності
wind_pulses = 3 * (np.random.rand(len(t)) - 0.5)  # випадкові пориви [-1.5..+1.5]

# Додаткові повільні атмосферні коливання
wind_wave = 2 * np.sin(0.5 * t + np.random.rand())

# Повна швидкість з урахуванням вітру
v_total = v_base + wind_pulses + wind_wave

# Графік
plt.figure(figsize=(10, 5))
plt.plot(t, v_total, label="v(t) з вітром", color='blue')
plt.plot(t, v_base, label="Базова v(t)", linestyle='--', color='orange')
plt.xlabel("Час (с)")
plt.ylabel("Швидкість (м/с)")
plt.title("Кінематична модель швидкості літака з поривами вітру")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
