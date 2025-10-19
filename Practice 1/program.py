from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Крок 1: Завантаження даних ---
file_path = 'medical_data_50_patients.csv'
df = pd.read_csv(file_path)
print("--- Дані успішно завантажено ---\n")
print(df.head(), "\n")

# --- Крок 2: Створення статистичного файлу ---
numeric_cols = ['Вік', 'Цукор крові (ммоль/л)', 'Холестерин (мг/дл)',
                'Пульс (уд/хв)', 'Температура (°C)', 'Вага (кг)', 'Зріст (см)', 'BMI']

stats_df = df[numeric_cols].describe().transpose()
stats_df.to_csv("statistical_summary_for_STATISTICA.csv", sep=';', encoding='utf-8')
print("--- Статистичний файл збережено для STATISTICA ---\n")

# --- Функція для runs test ---
def runs_test(series):
    runs = 0
    prev = None
    for value in series:
        if prev is None:
            prev = value
            continue
        current = 1 if value > prev else -1 if value < prev else 0
        if current != 0:
            if current != prev:
                runs += 1
            prev = current
    return runs

# --- Крок 3: Побудова графіків для всіх числових показників ---
window = 5  # для moving average

for col in numeric_cols:
    print(f"--- Аналіз: {col} ---")
    
    # Сортування за віком для часових рядів
    df_sorted = df.sort_values(by='Вік')
    series = df_sorted[col].reset_index(drop=True)
    time_index = np.arange(1, len(series)+1)
    
    # --- Графік оригінал + MA ---
    smoothed_series = series.rolling(window=window, min_periods=1).mean()
    
    plt.figure(figsize=(12,5))
    plt.plot(time_index, series, marker='o', linestyle='-', alpha=0.5, label='Оригінал')
    plt.plot(time_index, smoothed_series, color='red', linewidth=2, label=f'Згладжений (MA{window})')
    plt.title(f"Часовий ряд: {col}", fontsize=14)
    plt.xlabel("Пацієнти (по зростанню віку)")
    plt.ylabel(col)
    plt.grid(True)
    plt.legend()
    plt.show()  # Відображення графіку
    plt.close()
    
    # --- Висновок по графіку ---
    if smoothed_series.iloc[-1] > smoothed_series.iloc[0]:
        print(f"Висновок за графіком: спостерігається зростаючий тренд {col} з віком.\n")
    elif smoothed_series.iloc[-1] < smoothed_series.iloc[0]:
        print(f"Висновок за графіком: спостерігається спадний тренд {col} з віком.\n")
    else:
        print(f"Висновок за графіком: тенденція {col} стабільна, значних змін не спостерігається.\n")
    
    # --- Критерій серій (Крок 4) ---
    series_runs = runs_test(series)
    print(f"Кількість серій для {col}: {series_runs}")
    if series_runs < len(series) / 2:
        print(f"Висновок за критерієм серій: тенденція {col} в часовому ряду присутня.\n")
    else:
        print(f"Висновок за критерієм серій: тенденція {col} в часовому ряду відсутня.\n")
    print("------------------------------------------------------\n")

# --- Крок 5: Розкласти динамічний ряд на складові (Python) ---
print("\n--- Початок Кроку 5: Розкладання рядів ---")

for col in numeric_cols:
    print(f"  Аналіз розкладання для: {col}")
    
    series = df_sorted[col].reset_index(drop=True)
    time_index = np.arange(1, len(series)+1)

    # 1. Тренд (T): Використовуємо те ж ковзне середнє, що й у Кроці 3
    trend = series.rolling(window=window, min_periods=1).mean()
    
    # 2. Випадкова компонента (R): Адитивна модель Y = T + R
    residual = series - trend
    
    print(f"  Висновок (Крок 5): Ряд '{col}' розкладено на 'Тренд' (MA{window}) та 'Залишок'.")
    
    # Побудова графіку розкладання
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f'Розкладання ряду: {col}', fontsize=16, y=1.02)
        
        # 1. Оригінал
        series.plot(ax=ax1, label='Оригінал (Y)', color='blue')
        ax1.set_ylabel('Оригінал')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 2. Тренд
        trend.plot(ax=ax2, label=f'Тренд (T = MA{window})', color='red')
        ax2.set_ylabel('Тренд')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # 3. Випадкова компонента (Залишок)
        residual.plot(ax=ax3, label='Залишок (R = Y - T)', color='green', marker='o', markersize=4, linestyle='None')
        ax3.set_ylabel('Залишок')
        ax3.axhline(0, color='grey', linestyle='--')
        ax3.set_xlabel("Пацієнти (по зростанню віку)")
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()  # Відображення графіку
        plt.close(fig)

    except Exception as e:
        print(f"  Не вдалося побудувати графік розкладання для {col}: {e}")

print("--- Крок 5 завершено ---")

# --- Крок 6: Побудувати автокореляційну функцію (ACF) ---
print("\n--- Початок Кроку 6: Аналіз ACF залишків ---")

for col in numeric_cols:
    print(f"  Аналіз ACF для: {col}")

    # Повторно розраховуємо залишки (щоб бути впевненими)
    series = df_sorted[col].reset_index(drop=True)
    trend = series.rolling(window=window, min_periods=1).mean()
    residual = series - trend
    
    # Видаляємо NaN на початку ряду залишків
    cleaned_residual = residual.dropna()
    
    if not cleaned_residual.empty:
        try:
            fig_acf = plt.figure(figsize=(10, 5))
            # Використовуємо імпортовану функцію
            plot_acf(cleaned_residual, ax=fig_acf.gca(), lags=len(cleaned_residual)//2 - 1,
                     title=f'ACF випадкової компоненти (Залишків) для {col}')
            
            plt.show()  # Відображення графіку
            plt.close(fig_acf)

            print(f"  Висновок (Крок 6): ACF для '{col}' побудовано. Проаналізуйте графік, щоб оцінити коректність розкладення.")
            
        except Exception as e:
            print(f"  Не вдалося побудувати ACF для {col}: {e}")
    else:
        print(f"  Не вдалося побудувати ACF для {col}: ряд залишків порожній.")

print("--- Крок 6 завершено ---")
print("\n--- Повний аналіз (Кроки 1-6) завершено ---")