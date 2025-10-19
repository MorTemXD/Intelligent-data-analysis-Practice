import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import warnings

# Ігнорувати попередження, щоб зробити вивід чистішим
warnings.filterwarnings("ignore")

# --- Крок 1: Завантаження даних ---
file_path = 'medical_data_50_patients.csv'
try:
    df = pd.read_csv(file_path)
    print("--- Дані успішно завантажено ---")
    print(df.head())
except FileNotFoundError:
    print(f"Помилка: Файл '{file_path}' не знайдено.")
    exit() # Вихід, якщо файл не знайдено

# --- Крок 2: Вибір та візуалізація часового ряду (Завдання 1) ---
# Наприклад, беремо 'Цукор крові (ммоль/л)' як часовий ряд
# Важливо: Ряд сортується за 'Віком', а не за часом.
ts_col = 'Цукор крові (ммоль/л)'
series = df.sort_values(by='Вік')[ts_col].reset_index(drop=True)
time_index = np.arange(1, len(series) + 1) # Індекс для вісі X

plt.figure(figsize=(12, 5))
plt.plot(time_index, series, marker='o', linestyle='-', color='blue')
plt.title(f"Часовий ряд: {ts_col} (посортований за віком)", fontsize=14)
plt.xlabel("Пацієнти (по зростанню віку)")
plt.ylabel(ts_col)
plt.grid(True)
plt.savefig("plot_1_timeseries.png") # Збереження графіка
print("\n--- Графік 1 збережено: 'plot_1_timeseries.png' (Початковий ряд) ---")


# --- Крок 3: Перевірка стаціонарності (Завдання 1) ---
print("\n--- Крок 3: Перевірка стаціонарності (Тест Дікі-Фуллера) ---")
result = adfuller(series)
p_value = result[1]
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {p_value}")

if p_value > 0.05:
    print(f"Ряд НЕ є стаціонарним (p-value = {p_value:.4f}). Потрібне диференціювання.")
    is_stationary = False
    d = 1 # Встановлюємо порядок диференціювання d=1
else:
    print(f"Ряд є стаціонарним (p-value = {p_value:.4f}).")
    is_stationary = True
    d = 0 # Диференціювання не потрібне

# --- Крок 4: Приведення до стаціонарного вигляду (Завдання 2) ---
if not is_stationary:
    print("\n--- Крок 4: Застосування диференціювання (d=1) ---")
    diff_series = series.diff().dropna()
    # Повторна перевірка стаціонарності
    result_diff = adfuller(diff_series)
    p_value_diff = result_diff[1]
    print(f"p-value диференційованого ряду: {p_value_diff}")
    if p_value_diff <= 0.05:
        print("Диференційований ряд тепер є стаціонарним.")
        analysis_series = diff_series # Ряд для аналізу ACF/PACF
    else:
        print("Ряд все ще не стаціонарний. Можливо, потрібне подальше диференціювання.")
        analysis_series = diff_series
else:
    # Якщо ряд стаціонарний, аналізуємо його без змін
    analysis_series = series

# --- Крок 5: Підбір моделі (Графіки ACF/PACF) (Завдання 3) ---
print("\n--- Крок 5: Побудова ACF/PACF для підбору p та q ---")
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(analysis_series, lags=20, ax=ax[0], title='ACF для (диф.) ряду')
plot_pacf(analysis_series, lags=20, ax=ax[1], title='PACF для (диф.) ряду')
plt.savefig("plot_2_acf_pacf.png")
print("--- Графік 2 збережено: 'plot_2_acf_pacf.png' (ACF та PACF) ---")
print("Подивіться на ці графіки для вибору p (з PACF) та q (з ACF).")


# --- Крок 6: Знаходження параметрів (Навчання моделі) (Завдання 4) ---
# На основі ACF/PACF ви маєте обрати p та q.
# Для прикладу, як і у вашому коді, візьмемо p=1, q=1.
# d було визначено на Кроці 3 (0 або 1).
p = 1
q = 1

print(f"\n--- Крок 6: Побудова моделі ARIMA(p={p}, d={d}, q={q}) ---")
try:
    # Застосовуємо модель до ОРИГІНАЛЬНОГО ряду 'series'
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
except Exception as e:
    # Обробка помилок, якщо модель не може бути навчена
    print(f"Помилка при навчанні моделі: {e}")
    print("Спроба з простішими параметрами (наприклад, (0,d,0))")
    model = ARIMA(series, order=(0, d, 0))
    model_fit = model.fit()
    print(model_fit.summary())


# --- Крок 7: Установлення адекватності моделі (Завдання 5) ---
print("\n--- Крок 7: Перевірка адекватності моделі (Аналіз залишків) ---")
residuals = model_fit.resid

plt.figure(figsize=(12, 5))
plt.plot(residuals)
plt.title("Залишки моделі ARIMA")
plt.grid(True)
plt.savefig("plot_3_residuals.png")
print("--- Графік 3 збережено: 'plot_3_residuals.png' (Залишки) ---")

plt.figure(figsize=(12, 5))
plot_acf(residuals, lags=20, title="ACF залишків")
plt.savefig("plot_4_residuals_acf.png")
print("--- Графік 4 збережено: 'plot_4_residuals_acf.png' (ACF Залишків) ---")
print("Якщо на ACF залишків немає значущих сплесків, модель адекватна.")

# Оцінка точності
try:
    # Порівнюємо диференційований ряд з залишками (які теж є "диференційованими")
    # Треба зрізати перший 'd' елемент із залишків, щоб збіглася довжина
    if d > 0:
        mse = mean_squared_error(analysis_series, residuals[d:])
        print(f"MSE залишків: {mse:.4f}")
    else:
        mse = mean_squared_error(series, residuals)
        print(f"MSE залишків: {mse:.4f}")
except Exception as e:
    print(f"Не вдалося розрахувати MSE: {e}")

# ==============================================================================
# --- Крок 8: Прогнозування (Завдання 6) ---
# ЦЕЙ БЛОК БУЛО ДОДАНО
# ==============================================================================
print("\n--- Крок 8: Прогнозування на 10 періодів уперед ---")
n_forecast = 10
# Використовуємо .get_forecast() для отримання прогнозу та довірчих інтервалів
forecast_result = model_fit.get_forecast(steps=n_forecast)

# Отримуємо самі прогнозні значення
forecast = forecast_result.predicted_mean
# Отримуємо довірчі інтервали
conf_int = forecast_result.conf_int(alpha=0.05) # 95% довірчий інтервал

print("\nПрогнозні значення:")
print(forecast)

print("\nДовірчі інтервали:")
print(conf_int)

# Візуалізація прогнозу
plt.figure(figsize=(14, 7))
# Історичні дані
plt.plot(time_index, series, label='Історичні дані', marker='o')

# Індекс для прогнозу
# Він починається одразу після останнього індексу 'time_index'
last_index = time_index[-1]
forecast_index = np.arange(last_index + 1, last_index + 1 + n_forecast)

# Прогнозні значення
plt.plot(forecast_index, forecast, label='Прогноз', color='red', linestyle='--', marker='o')

# Довірчі інтервали
plt.fill_between(forecast_index,
                 conf_int.iloc[:, 0], # Нижня межа
                 conf_int.iloc[:, 1], # Верхня межа
                 color='red', alpha=0.1, label='95% Довірчий інтервал')

plt.title(f"Прогноз для: {ts_col}")
plt.xlabel("Пацієнти (по зростанню віку) + Прогноз")
plt.ylabel(ts_col)
plt.legend()
plt.grid(True)
plt.savefig("plot_5_forecast.png")
print("--- Графік 5 збережено: 'plot_5_forecast.png' (Прогноз) ---")
print("\n--- Скрипт виконано успішно ---")