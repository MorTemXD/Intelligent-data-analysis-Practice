import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# Набір 1: Сильна позитивна кореляція
data1 = {
    "Ліпопротеїни": [150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
    "Гемоглобін": [130, 135, 140, 145, 150, 155, 160, 165, 170, 175]
}

# Набір 2: Слабка негативна кореляція
data2 = {
    "Ліпопротеїни": [150, 180, 210, 170, 200, 220, 160, 190, 205, 175],
    "Гемоглобін": [140, 138, 135, 142, 136, 133, 143, 137, 134, 141]
}

# Набір 3: Відсутність кореляції
data3 = {
    "Ліпопротеїни": [150, 180, 210, 170, 200, 220, 160, 190, 205, 175],
    "Гемоглобін": [130, 145, 135, 150, 140, 130, 145, 135, 150, 140]
}

datasets = [data1, data2, data3]
dataset_names = ["Сильна позитивна кореляція", "Слабка негативна кореляція", "Відсутність кореляції"]

# === ФУНКЦІЯ ДЛЯ ІНТЕРПРЕТАЦІЇ ===
def interpret_correlation(r, p):
    strength = abs(r)
    
    if strength >= 0.9:
        strength_desc = "дуже високий"
    elif strength >= 0.75:
        strength_desc = "високий"
    elif strength >= 0.50:
        strength_desc = "помірний"
    elif strength >= 0.25:
        strength_desc = "слабкий"
    elif strength >= 0.10:
        strength_desc = "дуже слабкий"
    else:
        strength_desc = "практично відсутній"
    
    direction = "позитивний" if r > 0 else "негативний"
    significance = "значущий" if p < 0.05 else "не значущий"
    
    return strength_desc, direction, significance

# === ФУНКЦІЯ ДЛЯ ПЕРЕВІРКИ НОРМАЛЬНОСТІ ===
def check_normality(data, alpha=0.05):
    statistic, p_value = stats.shapiro(data)
    return p_value > alpha, statistic, p_value

# === АНАЛІЗ КОЖНОГО НАБОРУ ДАНИХ ===
results = []

for i, (data, name) in enumerate(zip(datasets, dataset_names)):
    df = pd.DataFrame(data)
    x = df["Ліпопротеїни"]
    y = df["Гемоглобін"]
    
    # Базова статистика
    x_stats = {
        'mean': np.mean(x),
        'std': np.std(x, ddof=1),
        'min': np.min(x),
        'max': np.max(x)
    }
    
    y_stats = {
        'mean': np.mean(y),
        'std': np.std(y, ddof=1),
        'min': np.min(y),
        'max': np.max(y)
    }
    
    # Перевірка нормальності
    x_normal, x_shapiro_stat, x_shapiro_p = check_normality(x)
    y_normal, y_shapiro_stat, y_shapiro_p = check_normality(y)
    
    # Розрахунок коефіцієнтів кореляції
    pearson_corr, pearson_p = stats.pearsonr(x, y)
    spearman_corr, spearman_p = stats.spearmanr(x, y)
    
    # Коефіцієнт детермінації
    r_squared = pearson_corr ** 2
    
    # Довірчі інтервали для кореляції Пірсона
    n = len(x)
    z = np.arctanh(pearson_corr)
    z_se = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * z_se
    z_upper = z + 1.96 * z_se
    pearson_ci_lower = np.tanh(z_lower)
    pearson_ci_upper = np.tanh(z_upper)
    
    # Збереження результатів
    results.append({
        'name': name,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'pearson_ci': (pearson_ci_lower, pearson_ci_upper),
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'r_squared': r_squared,
        'x_normal': x_normal,
        'y_normal': y_normal,
        'x_stats': x_stats,
        'y_stats': y_stats,
        'n': n
    })
    
    # Створення графіків
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{name}\n', fontsize=16, fontweight='bold')
    
    # Діаграма 1: Просте розсіювання
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, color='blue', alpha=0.7, s=60)
    plt.xlabel('Ліпопротеїни')
    plt.ylabel('Гемоглобін')
    plt.title('Діаграма розсіювання')
    plt.grid(True, alpha=0.3)
    
    # Діаграма 2: З лінією регресії
    plt.subplot(1, 3, 2)
    plt.scatter(x, y, color='red', alpha=0.7, s=60)
    z_coef = np.polyfit(x, y, 1)
    p_func = np.poly1d(z_coef)
    plt.plot(x, p_func(x), "r--", alpha=0.8, linewidth=2)
    plt.xlabel('Ліпопротеїни')
    plt.ylabel('Гемоглобін')
    plt.title(f'З лінією регресії\nПірсона: r = {pearson_corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Діаграма 3: Рангове розсіювання
    plt.subplot(1, 3, 3)
    rank_x = pd.Series(x).rank()
    rank_y = pd.Series(y).rank()
    plt.scatter(rank_x, rank_y, color='green', alpha=0.7, s=60)
    plt.xlabel('Ранг ліпопротеїнів')
    plt.ylabel('Ранг гемоглобіну')
    plt.title(f'Рангове розсіювання\nСпірмена: ρ = {spearman_corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Вивід результатів в консоль
    print("=" * 80)
    print(f"АНАЛІЗ НАБОРУ ДАНИХ {i+1}: {name.upper()}")
    print("=" * 80)
    
    # Базова статистика
    print("\n БАЗОВА СТАТИСТИКА:")
    print(f"• Розмір вибірки: n = {n}")
    print(f"• Ліпопротеїни: M = {x_stats['mean']:.1f} ± {x_stats['std']:.1f} (min={x_stats['min']}, max={x_stats['max']})")
    print(f"• Гемоглобін: M = {y_stats['mean']:.1f} ± {y_stats['std']:.1f} (min={y_stats['min']}, max={y_stats['max']})")
    
    # Перевірка нормальності
    print(f"\n ПЕРЕВІРКА НОРМАЛЬНОСТІ (Шапіро-Вілка, α=0.05):")
    print(f"• Ліпопротеїни: W = {x_shapiro_stat:.3f}, p = {x_shapiro_p:.3f} - {'нормальний' if x_normal else 'не нормальний'}")
    print(f"• Гемоглобін: W = {y_shapiro_stat:.3f}, p = {y_shapiro_p:.3f} - {'нормальний' if y_normal else 'не нормальний'}")
    
    # Кореляційний аналіз
    print(f"\n КОРЕЛЯЦІЙНИЙ АНАЛІЗ:")
    print(f"• Коефіцієнт Пірсона: r = {pearson_corr:.4f}, p = {pearson_p:.4f}")
    print(f"• 95% ДІ для Пірсона: [{pearson_ci_lower:.4f}, {pearson_ci_upper:.4f}]")
    print(f"• Коефіцієнт Спірмена: ρ = {spearman_corr:.4f}, p = {spearman_p:.4f}")
    print(f"• Коефіцієнт детермінації: R² = {r_squared:.4f} ({r_squared*100:.1f}% дисперсії)")
    
    # Інтерпретація
    strength_desc, direction, significance = interpret_correlation(pearson_corr, pearson_p)
    print(f"\n ІНТЕРПРЕТАЦІЯ (Пірсон):")
    print(f"• Сила зв'язку: {strength_desc} (r = {pearson_corr:.4f})")
    print(f"• Напрямок: {direction}")
    print(f"• Статистична значущість: {significance} (p = {pearson_p:.4f})")
    
    # Рекомендації
    print(f"\n РЕКОМЕНДАЦІЇ:")
    if x_normal and y_normal:
        print("• Обидві змінні нормально розподілені - Пірсон підходить")
    else:
        print("• Одна або обидві змінні не нормальні - Спірмен кращий вибір")
    
    if abs(pearson_corr - spearman_corr) < 0.05:
        print("• Методи узгоджуються: різниця незначна (< 0.05)")
    elif abs(pearson_corr - spearman_corr) < 0.1:
        print("• Методи частково узгоджуються: різниця помірна (0.05-0.10)")
    else:
        print("• Методи не узгоджуються: різниця значна (> 0.10)")
    
    # Практичне значення
    if r_squared >= 0.5:
        print("• Практичне значення: ВИСОКЕ (R² ≥ 50%)")
    elif r_squared >= 0.25:
        print("• Практичне значення: ПОМІРНЕ (25% ≤ R² < 50%)")
    elif r_squared >= 0.1:
        print("• Практичне значення: СЛАБКЕ (10% ≤ R² < 25%)")
    else:
        print("• Практичне значення: НЕЗНАЧНЕ (R² < 10%)")
    
    print("\n" + "-" * 80 + "\n")

# === ЗАГАЛЬНИЙ ВИСНОВОК ===
print("\n ПОРІВНЯЛЬНА ТАБЛИЦЯ:")
print("Набір даних".ljust(30) + "Пірсон (r)".ljust(15) + "Спірмен (ρ)".ljust(15) + "R²".ljust(12) + "p-значення".ljust(15) + "Статус")
print("-" * 90)

for result in results:
    status = "Значущий" if result['pearson_p'] < 0.05 else "Не значущий"
    print(f"{result['name']:<30}{result['pearson_corr']:>7.4f}{result['spearman_corr']:>14.4f}{result['r_squared']:>11.4f}{result['pearson_p']:>14.4f}  {status}")

print("\n ДЕТАЛЬНИЙ АНАЛІЗ:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['name']}:")
    print(f"   • 95% ДІ кореляції: [{result['pearson_ci'][0]:.4f}, {result['pearson_ci'][1]:.4f}]")
    print(f"   • Нормальність: Ліпопротеїни - {'так' if result['x_normal'] else 'ні'}, Гемоглобін - {'так' if result['y_normal'] else 'ні'}")
    print(f"   • Пояснена дисперсія: {result['r_squared']*100:.1f}%")

print(f"\n СТАТИСТИКА ВИБІРОК:")
total_n = sum([result['n'] for result in results])
print(f"• Загальний обсяг даних: {total_n} спостережень")
print(f"• Середній розмір вибірки: {total_n/len(results):.1f} на набір")

print("=" * 100)