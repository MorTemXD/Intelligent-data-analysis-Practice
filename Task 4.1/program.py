import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy.signal import savgol_filter

# Ігнорувати попередження
import warnings
warnings.filterwarnings("ignore")

# --- Крок 1: Завантаження даних ---
file_path = 'medical_data_50_patients.csv'
try:
    df = pd.read_csv(file_path)
    print("--- Дані успішно завантажено ---")
    print(df.head())
except FileNotFoundError:
    print(f"Помилка: Файл '{file_path}' не знайдено.")
    exit()

# --- НОВИЙ КРОК 1.1: Дослідження структури часового ряду (тренд та періодичні коливання) ---
print("\n--- Новий Крок 1.1: Дослідження структури часового ряду ---")
ts_col = 'Цукор крові (ммоль/л)'
series = df.sort_values(by='Вік')[ts_col].reset_index(drop=True)
time_index = np.arange(1, len(series) + 1)

# Декомпозиція часового ряду для виявлення тренду, сезонності та залишків
decomposition = seasonal_decompose(series, model='additive', period=5)  # Припускаємо період 5 для прикладу
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(series, label='Оригінальний ряд')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Тренд')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Сезонність')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Залишки')
plt.legend(loc='upper left')
plt.suptitle('Декомпозиція часового ряду: Цукор крові')
plt.tight_layout()
plt.show()
print("--- Графік 1.1 виведено: Декомпозиція часового ряду ---")

# --- НОВИЙ КРОК 1.2: Згладжування та фільтрація часового ряду ---
print("\n--- Новий Крок 1.2: Згладжування часового ряду (Savitzky-Golay фільтр) ---")
smoothed_series = savgol_filter(series, window_length=5, polyorder=2)  # Застосовуємо фільтр Savitzky-Golay
plt.figure(figsize=(12, 5))
plt.plot(time_index, series, label='Оригінальний ряд', alpha=0.5)
plt.plot(time_index, smoothed_series, label='Згладжений ряд (Savitzky-Golay)', color='red')
plt.title(f"Згладжений часовий ряд: {ts_col}")
plt.xlabel("Пацієнти (по зростанню віку)")
plt.ylabel(ts_col)
plt.legend()
plt.grid(True)
plt.show()
print("--- Графік 1.2 виведено: Згладжений ряд ---")

# --- Крок 2: Вибір та візуалізація часового ряду ---
plt.figure(figsize=(12, 5))
plt.plot(time_index, series, marker='o', linestyle='-', color='blue')
plt.title(f"Часовий ряд: {ts_col} (посортований за віком)", fontsize=14)
plt.xlabel("Пацієнти (по зростанню віку)")
plt.ylabel(ts_col)
plt.grid(True)
plt.show()
print("\n--- Графік 1 виведено: Початковий ряд ---")

# --- Крок 3: Перевірка стаціонарності ---
print("\n--- Крок 3: Перевірка стаціонарності (Тест Дікі-Фуллера) ---")
result = adfuller(series)
p_value = result[1]
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {p_value}")

if p_value > 0.05:
    print(f"Ряд НЕ є стаціонарним (p-value = {p_value:.4f}). Потрібне диференціювання.")
    is_stationary = False
    d = 1
else:
    print(f"Ряд є стаціонарним (p-value = {p_value:.4f}).")
    is_stationary = True
    d = 0

# --- Крок 4: Приведення до стаціонарного вигляду ---
if not is_stationary:
    print("\n--- Крок 4: Застосування диференціювання (d=1) ---")
    diff_series = series.diff().dropna()
    result_diff = adfuller(diff_series)
    p_value_diff = result_diff[1]
    print(f"p-value диференційованого ряду: {p_value_diff}")
    if p_value_diff <= 0.05:
        print("Диференційований ряд тепер є стаціонарним.")
        analysis_series = diff_series
    else:
        print("Ряд все ще не стаціонарний.")
        analysis_series = diff_series
else:
    analysis_series = series

# --- Крок 5: Підбір моделі (Графіки ACF/PACF) ---
print("\n--- Крок 5: Побудова ACF/PACF для підбору p та q ---")
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(analysis_series, lags=20, ax=ax[0], title='ACF для (диф.) ряду')
plot_pacf(analysis_series, lags=20, ax=ax[1], title='PACF для (диф.) ряду')
plt.show()
print("--- Графік 2 виведено: ACF та PACF ---")

# --- Крок 6: Побудова моделі ARIMA ---
p = 1
q = 1
print(f"\n--- Крок 6: Побудова моделі ARIMA(p={p}, d={d}, q={q}) ---")
try:
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
except Exception as e:
    print(f"Помилка при навчанні моделі: {e}")
    model = ARIMA(series, order=(0, d, 0))
    model_fit = model.fit()
    print(model_fit.summary())

# --- Крок 7: Перевірка адекватності моделі ---
print("\n--- Крок 7: Перевірка адекватності моделі (Аналіз залишків) ---")
residuals = model_fit.resid
plt.figure(figsize=(12, 5))
plt.plot(residuals)
plt.title("Залишки моделі ARIMA")
plt.grid(True)
plt.show()
print("--- Графік 3 виведено: Залишки ---")

plt.figure(figsize=(12, 5))
plot_acf(residuals, lags=20, title="ACF залишків")
plt.show()
print("--- Графік 4 виведено: ACF Залишків ---")

if d > 0:
    mse = mean_squared_error(analysis_series, residuals[d:])
    print(f"MSE залишків: {mse:.4f}")
else:
    mse = mean_squared_error(series, residuals)
    print(f"MSE залишків: {mse:.4f}")

# --- Крок 8: Прогнозування ---
print("\n--- Крок 8: Прогнозування на 10 періодів уперед ---")
n_forecast = 10
forecast_result = model_fit.get_forecast(steps=n_forecast)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

print("\nПрогнозні значення:")
print(forecast)
print("\nДовірчі інтервали:")
print(conf_int)

plt.figure(figsize=(14, 7))
plt.plot(time_index, series, label='Історичні дані', marker='o')
last_index = time_index[-1]
forecast_index = np.arange(last_index + 1, last_index + 1 + n_forecast)
plt.plot(forecast_index, forecast, label='Прогноз', color='red', linestyle='--', marker='o')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1, label='95% Довірчий інтервал')
plt.title(f"Прогноз для: {ts_col}")
plt.xlabel("Пацієнти (по зростанню віку) + Прогноз")
plt.ylabel(ts_col)
plt.legend()
plt.grid(True)
plt.show()
print("--- Графік 5 виведено: Прогноз ---")

# --- НОВИЙ КРОК 8.1: Дослідження причинно-наслідкових взаємозв’язків (Кореляційний аналіз) ---
print("\n--- Новий Крок 8.1: Кореляційний аналіз ---")
# Вибираємо кілька змінних для аналізу кореляцій
corr_features = ['Цукор крові (ммоль/л)', 'Холестерин (мг/дл)', 'BMI', 'Вік', 'Пульс (уд/хв)']
corr_matrix = df[corr_features].corr()

# Візуалізація кореляційної матриці
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Кореляційна матриця для медичних показників")
plt.show()
print("--- Графік 8.1 виведено: Кореляційна матриця ---")
print("\nКореляційна матриця:")
print(corr_matrix)

# --- Крок 9: Класифікація (Прогнозування діабету) ---
print("\n--- Крок 9: Класифікація (Прогнозування діабету) ---")
df['Діабет'] = df['Хронічні захворювання'].apply(lambda x: 1 if x == 'Діабет' else 0)
features = ['Вік', 'Цукор крові (ммоль/л)', 'Холестерин (мг/дл)', 'BMI']
X = df[features]
y = df['Діабет']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print("\nРезультати класифікації:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print("\nМатриця помилок:")
print(confusion_matrix(y_test, y_pred))

# --- Крок 10: Кластеризація ---
print("\n--- Крок 10: Кластеризація ---")
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Кластер'] = clusters

sil_score = silhouette_score(X_scaled, clusters)
db_score = davies_bouldin_score(X_scaled, clusters)
print(f"Силуетний коефіцієнт: {sil_score:.4f}")
print(f"Індекс Дэвіса-Болдіна: {db_score:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X['Цукор крові (ммоль/л)'], X['BMI'], c=clusters, cmap='viridis')
plt.title("Кластеризація пацієнтів")
plt.xlabel("Цукор крові (ммоль/л)")
plt.ylabel("BMI")
plt.grid(True)
plt.show()
print("--- Графік 6 виведено: Кластери ---")

print("\n--- Скрипт виконано успішно ---")