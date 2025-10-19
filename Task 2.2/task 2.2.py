import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt

n = 60
df = pd.DataFrame({
    "місдорецидиву": np.random.randint(1, 24, size=n),   # час до події (1–24 міс.)
    "Рецидив": np.random.choice([1, 2], size=n, p=[0.7, 0.3]),  # 2 = подія (рецидив), 1 = цензуровано
    "глісон_к": np.random.choice([6, 7, 8, 9], size=n),
    "ЕПР_к": np.random.choice([0, 1], size=n),           # 0 = ні, 1 = так
    "стадія": np.random.choice([2, 3, 4], size=n),
    "T_к": np.random.choice([2, 3, 4], size=n),
    "CD": np.random.choice([0, 1], size=n),
    "променева_к": np.random.choice([0, 1], size=n),
    "ОЕ_к": np.random.choice([0, 1], size=n)
})

print("Перші рядки датасету:")
print(df.head())

# === 2. Кодування категоріальних змінних (drop_first → уникаємо колінеарності) ===
df_encoded = pd.get_dummies(df, columns=["глісон_к", "стадія", "T_к"], drop_first=True)

# === 3. Модель пропорційних ризиків Кокса ===
cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_encoded, duration_col="місдорецидиву", event_col="Рецидив")

print("\n=== Результати регресії Кокса ===")
cph.print_summary()

# === 4. Криві виживаності (Kaplan-Meier) ===
kmf = KaplanMeierFitter()

plt.figure(figsize=(8, 6))
for stage in df["стадія"].unique():
    mask = df["стадія"] == stage
    kmf.fit(df["місдорецидиву"][mask], 
            event_observed=(df["Рецидив"][mask] == 2), 
            label=f"Стадія {stage}")
    kmf.plot_survival_function(ci_show=False)

plt.title("Криві виживаності за стадіями")
plt.xlabel("Місяці до рецидиву")
plt.ylabel("Ймовірність виживання")
plt.legend()
plt.show()

# === 5. Графік коефіцієнтів моделі ===
cph.plot()
plt.title("Коефіцієнти регресії Кокса")
plt.show()
