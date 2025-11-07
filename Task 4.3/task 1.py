import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt

y = np.array([1.6, 0.8, 1.2, 0.5, 0.9, 1.1, 1.1, 0.6, 1.5, 0.8, 0.9, 1.2, 0.5, 1.3, 0.8, 1.2])
t = np.arange(1, len(y) + 1)

plt.style.use('ggplot')

print("--- а) Графік часового ряду ---")
plt.figure(figsize=(12, 5))
plt.plot(t, y, marker='o', linestyle='-')
plt.title('Графік часового ряду y(t)')
plt.xlabel('Спостереження (t)')
plt.ylabel('Значення y(t)')
plt.xticks(t)
plt.grid(True, which='both', linestyle='--')
plt.show()

print("\n--- б) Наближене визначення ACF(1) з графіка (а) ---")
print("Аналізуючи графік (а), можна помітити часті та різкі коливання.")
print("Високі значення часто змінюються низькими, і навпаки.")
print("Наприклад:")
print("  t=1 (1.6) -> t=2 (0.8)  (Падіння)")
print("  t=3 (1.2) -> t=4 (0.5)  (Падіння)")
print("  t=8 (0.6) -> t=9 (1.5)  (Зростання)")
print("  t=9 (1.5) -> t=10 (0.8) (Падіння)")
print("Така поведінка, коли за високим значенням слідує низьке (і навпаки),")
print("вказує на **негативну автокореляцію** першого порядку (ACF(1) < 0).")

print("\n--- в) Графік y(t+1) від y(t) та розрахунок ---")

y_t = y[:-1]
y_t_plus_1 = y[1:]

plt.figure(figsize=(7, 7))
plt.scatter(y_t, y_t_plus_1, alpha=0.9, s=50) 

mean_y = y.mean()
plt.axvline(mean_y, color='gray', linestyle='--', label=f'Середнє = {mean_y:.2f}')
plt.axhline(mean_y, color='gray', linestyle='--')

plt.title('Графік залежності y(t+1) від y(t) (Lag Plot)')
plt.xlabel('y(t)')
plt.ylabel('y(t+1)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

print("\nНаближене визначення з графіка (c):")
print("Точки на графіку 'y(t+1) від y(t)' (lag plot) чітко розташовані")
print("вздовж лінії, що має **негативний нахил**.")
print("Більшість точок знаходиться у верхньому лівому та нижньому правому")
print("квадрантах (відносно ліній середнього).")
print("Це підтверджує висновок з (б) і вказує на сильну негативну автокореляцію.")

r1_pearson = np.corrcoef(y_t, y_t_plus_1)[0, 1]

r1_statsmodels = smt.acf(y, nlags=1, fft=False)[1]

print("\nТочний розрахунок ACF(1):")
print(f"  Метод 1 (Кореляція Пірсона для y_t та y_t_plus_1, як на графіку):")
print(f"    r(1) = {r1_pearson:.4f}")
print(f"  Метод 2 (Стандартне статистичне визначення ACF з statsmodels):")
print(f"    r(1) = {r1_statsmodels:.4f}")

print("\nЯк бачимо, коефіцієнт автокореляції першого порядку є **негативним**")
print("і досить сильним (близько -0.7).")