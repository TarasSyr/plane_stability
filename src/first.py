import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Генерація даних моделі
np.random.seed(42)
t = np.linspace(0, 60, 600)  # Час: 0-60 сек, 600 точок
v0 = 80  # Початкова швидкість
v_base = v0 - 2 * np.exp(-0.2 * t)  # Базова швидкість без вітру
wind_pulses = 3 * (np.random.rand(len(t)) - 0.5)  # Випадкові пориви
wind_wave = 2 * np.sin(0.5 * t + np.random.rand())  # Періодичний вітер
v_total = v_base + wind_pulses + wind_wave  # Швидкість з вітром
wind_combined = wind_pulses + wind_wave  # Сумарний вплив вітру

# 2. Початкові графіки швидкостей
plt.figure(figsize=(12, 8))

# Графік 1: Базова швидкість vs Швидкість з вітром
plt.subplot(3, 1, 1)
plt.plot(t, v_base, label='Базова швидкість (без вітру)', color='green', linestyle='--')
plt.plot(t, v_total, label='Швидкість з вітром', color='blue', alpha=0.7)
plt.title('Динаміка швидкості БПЛА')
plt.ylabel('Швидкість (м/с)')
plt.legend()
plt.grid(True)

# Графік 2: Вплив вітру
plt.subplot(3, 1, 2)
plt.plot(t, wind_combined, label='Сумарний вплив вітру', color='red', alpha=0.7)
plt.title('Вітер: випадкові пориви + періодичні коливання')
plt.ylabel('Швидкість вітру (м/с)')
plt.legend()
plt.grid(True)

# 3. Гістограма розподілу швидкості
plt.subplot(3, 1, 3)
n_bins = int(1 + 3.322 * np.log10(len(t)))  # Формула Стерджеса
plt.hist(v_total, bins=n_bins, density=True, alpha=0.7, color='purple', edgecolor='black')
plt.title('Гістограма розподілу швидкості з вітром')
plt.xlabel('Швидкість (м/с)')
plt.ylabel('Щільність ймовірності')
plt.grid(True)

plt.tight_layout()
plt.savefig(r'C:\Users\Taras\PycharmProjects\plane_stability\results\initial_analysis.png')
plt.show()

# 4. Q-Q графік для перевірки нормальності
plt.figure(figsize=(8, 6))
stats.probplot(v_total, plot=plt)
plt.title('Q-Q графік (нормальність розподілу)')
plt.grid(True)
plt.savefig(r'C:\Users\Taras\PycharmProjects\plane_stability\results\qq_plot.png')
plt.show()

# 5. Автокореляція для аналізу залежностей
def autocorrelation(x):
    result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    return result[result.size // 2:] / np.max(result)

acf = autocorrelation(v_total - np.mean(v_total))
plt.figure(figsize=(10, 4))
markerline, _, _ = plt.stem(acf[:50], linefmt='blue', markerfmt='o', basefmt='gray')
plt.title('Функція автокореляції (перші 50 лагів)')
plt.xlabel('Лаг')
plt.ylabel('Коефіцієнт кореляції')
plt.grid(True)
plt.savefig(r'C:\Users\Taras\PycharmProjects\plane_stability\results\autocorrelation.png')
plt.show()

# 6. Основні статистичні характеристики
mu = np.mean(v_total)
sigma = np.std(v_total, ddof=1)
variance = np.var(v_total, ddof=1)

stats_text = f"""
=== Статистичний аналіз ===
1. Основні метрики:
   - Середнє (μ): {mu:.2f} м/с
   - Дисперсія (σ²): {variance:.2f} (м/с)²
   - СКО (σ): {sigma:.2f} м/с
   - Медіана: {np.median(v_total):.2f} м/с
   - Мін/Макс: {np.min(v_total):.2f} / {np.max(v_total):.2f} м/с

2. Довірчі інтервали (95%):
   - Мат. очікування: [{mu - 1.96*sigma/np.sqrt(len(t)):.2f}; {mu + 1.96*sigma/np.sqrt(len(t)):.2f}]
   - Дисперсія: [{(len(t)-1)*variance/stats.chi2.ppf(0.975, len(t)-1):.2f}; {(len(t)-1)*variance/stats.chi2.ppf(0.025, len(t)-1):.2f}]
"""

# 7. Перевірка на нормальність (χ²-тест)
def chi2_test(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected = stats.norm.pdf(bin_centers, mu, sigma) * len(data) * (bin_edges[1]-bin_edges[0])
    chi2 = np.sum((hist - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2, bins-3)
    return chi2, p_value

chi2, p_value = chi2_test(v_total, n_bins)
stats_text += f"""
3. Перевірка нормальності (χ²):
   - χ² = {chi2:.2f} (p-value = {p_value:.3f})
   - Висновок: {'Нормальний розподіл (p > 0.05)' if p_value > 0.05 else 'Ненормальний розподіл'}
"""

# Збереження результатів
with open(r'C:\Users\Taras\PycharmProjects\plane_stability\results\results.txt', 'w', encoding='utf-8') as f:
    f.write(stats_text)

print("Аналіз завершено! Результати збережено у файли:")
print("- initial_analysis.png (графіки швидкостей та гістограма)")
print("- qq_plot.png (Q-Q графік)")
print("- autocorrelation.png (автокореляція)")
print("- results.txt (статистичні метрики)")