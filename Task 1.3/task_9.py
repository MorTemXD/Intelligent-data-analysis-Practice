import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS)
    Реалізація моделі нейро-нечіткої мережі
    """
    
    def __init__(self, n_inputs=2, n_mfs=3, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs  # кількість функцій належності на кожен вхід
        self.learning_rate = learning_rate
        
        # Ініціалізація параметрів функцій належності (гаусові функції)
        self.centers = np.random.uniform(-1, 1, (n_inputs, n_mfs))
        self.sigmas = np.random.uniform(0.5, 1.5, (n_inputs, n_mfs))
        
        # Ініціалізація параметрів правил (лінійні функції)
        self.consequent_params = np.random.uniform(-1, 1, (n_mfs**n_inputs, n_inputs + 1))
        
        # Історія тренування
        self.loss_history = []
    
    def gaussian_mf(self, x, center, sigma):
        """Гаусова функція належності"""
        return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    def forward(self, X):
        """Прямий прохід через мережу"""
        batch_size = X.shape[0]
        
        # Шар 1: Функції належності
        membership_values = np.zeros((batch_size, self.n_inputs, self.n_mfs))
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                membership_values[:, i, j] = self.gaussian_mf(
                    X[:, i], self.centers[i, j], self.sigmas[i, j]
                )
        
        # Шар 2: Активація правил (T-norm - добуток)
        rule_activations = np.ones((batch_size, self.n_mfs ** self.n_inputs))
        
        # Генерація всіх можливих комбінацій правил
        rule_combinations = np.indices([self.n_mfs] * self.n_inputs).reshape(self.n_inputs, -1).T
        
        for rule_idx, combination in enumerate(rule_combinations):
            for input_idx, mf_idx in enumerate(combination):
                rule_activations[:, rule_idx] *= membership_values[:, input_idx, mf_idx]
        
        # Шар 3: Нормалізація рівнів активації правил
        rule_sums = np.sum(rule_activations, axis=1, keepdims=True)
        rule_sums[rule_sums == 0] = 1e-10  # уникнення ділення на нуль
        normalized_activations = rule_activations / rule_sums
        
        # Шар 4: Виходи правил
        rule_outputs = np.zeros((batch_size, self.n_mfs ** self.n_inputs))
        for rule_idx in range(self.n_mfs ** self.n_inputs):
            # Лінійна функція: w0 + w1*x1 + w2*x2 + ...
            linear_combination = self.consequent_params[rule_idx, 0]  # bias
            for input_idx in range(self.n_inputs):
                linear_combination += (self.consequent_params[rule_idx, input_idx + 1] * 
                                     X[:, input_idx])
            rule_outputs[:, rule_idx] = normalized_activations[:, rule_idx] * linear_combination
        
        # Шар 5: Агрегація виходів
        final_output = np.sum(rule_outputs, axis=1)
        
        return final_output, membership_values, rule_activations, normalized_activations, rule_outputs
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Тренування мережі методом зворотного поширення помилки"""
        for epoch in range(epochs):
            # Прямий прохід
            y_pred, membership, rule_act, norm_act, rule_out = self.forward(X)
            
            # Обчислення помилки
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            
            # Зворотний прохід - градієнтний спуск
            batch_size = X.shape[0]
            
            # Градієнти для параметрів правил
            grad_consequent = np.zeros_like(self.consequent_params)
            
            for rule_idx in range(self.n_mfs ** self.n_inputs):
                # Градієнт для кожного параметра правила
                grad_consequent[rule_idx, 0] = np.mean(error * norm_act[:, rule_idx])  # bias
                for input_idx in range(self.n_inputs):
                    grad_consequent[rule_idx, input_idx + 1] = np.mean(
                        error * norm_act[:, rule_idx] * X[:, input_idx]
                    )
            
            # Оновлення параметрів правил
            self.consequent_params -= self.learning_rate * grad_consequent
            
            # Градієнти для параметрів функцій належності (спрощена версія)
            grad_centers = np.zeros_like(self.centers)
            grad_sigmas = np.zeros_like(self.sigmas)
            
            # Генерація комбінацій правил
            rule_combinations = np.indices([self.n_mfs] * self.n_inputs).reshape(self.n_inputs, -1).T
            
            for i in range(self.n_inputs):
                for j in range(self.n_mfs):
                    for rule_idx, combination in enumerate(rule_combinations):
                        if combination[i] == j:
                            # Спрощений градієнт для центрів та сигм
                            for k in range(batch_size):
                                x_val = X[k, i]
                                center = self.centers[i, j]
                                sigma = self.sigmas[i, j]
                                
                                # Похідна гаусової функції
                                d_mu_d_center = membership[k, i, j] * (x_val - center) / (sigma ** 2)
                                d_mu_d_sigma = membership[k, i, j] * ((x_val - center) ** 2) / (sigma ** 3)
                                
                                # Вплив на вихід правила
                                rule_grad = error[k] * (rule_out[k, rule_idx] - y_pred[k]) / np.sum(rule_act[k])
                                
                                grad_centers[i, j] += rule_grad * d_mu_d_center
                                grad_sigmas[i, j] += rule_grad * d_mu_d_sigma
            
            # Оновлення параметрів функцій належності
            self.centers -= self.learning_rate * grad_centers / batch_size
            self.sigmas -= self.learning_rate * grad_sigmas / batch_size
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Епоха {epoch}, Втрати: {loss:.6f}")
    
    def predict(self, X):
        """Прогнозування"""
        y_pred, _, _, _, _ = self.forward(X)
        return y_pred
    
    def plot_membership_functions(self):
        """Візуалізація функцій належності"""
        fig, axes = plt.subplots(1, self.n_inputs, figsize=(5*self.n_inputs, 4))
        if self.n_inputs == 1:
            axes = [axes]
        
        x_range = np.linspace(-3, 3, 100)
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                y_vals = self.gaussian_mf(x_range, self.centers[i, j], self.sigmas[i, j])
                axes[i].plot(x_range, y_vals, label=f'MF {j+1}')
            axes[i].set_title(f'Вхід {i+1}')
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        plt.show()

# Демонстрація роботи ANFIS
def demo_anfis():
    """Демонстраційний приклад використання ANFIS"""
    
    # Генерація синтетичних даних
    print("Генерація даних...")
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
    
    # Нормалізація даних
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Розділення на тренувальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Створення та тренування ANFIS
    print("\nСтворення ANFIS мережі...")
    anfis = ANFIS(n_inputs=2, n_mfs=3, learning_rate=0.01)
    
    print("Початок тренування...")
    anfis.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Прогнозування на тестовій вибірці
    y_pred = anfis.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nСередньоквадратична помилка на тесті: {mse:.6f}")
    
    # Візуалізація результатів
    plt.figure(figsize=(15, 5))
    
    # Графік 1: Функції належності
    plt.subplot(1, 3, 1)
    anfis.plot_membership_functions()
    
    # Графік 2: Історія тренування
    plt.subplot(1, 3, 2)
    plt.plot(anfis.loss_history)
    plt.title('Історія тренування')
    plt.xlabel('Епоха')
    plt.ylabel('Втрати')
    plt.grid(True)
    
    # Графік 3: Прогнозування vs Фактичні значення
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Фактичні значення')
    plt.ylabel('Прогнозовані значення')
    plt.title('Прогнозування vs Фактичні значення')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return anfis

# Приклад використання для конкретної задачі
def custom_problem_example():
    """Приклад використання ANFIS для нелінійної функції"""
    
    # Створення нелінійної функції: z = sin(x) * cos(y)
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.linspace(-np.pi, np.pi, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    # Підготовка даних
    X_data = np.column_stack([X.ravel(), Y.ravel()])
    y_data = Z.ravel()
    
    # Нормалізація
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_data = scaler_X.fit_transform(X_data)
    y_data = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
    
    # Тренування ANFIS
    anfis = ANFIS(n_inputs=2, n_mfs=4, learning_rate=0.005)
    anfis.train(X_data, y_data, epochs=500, verbose=True)
    
    # Прогнозування
    Z_pred = anfis.predict(X_data)
    Z_pred = Z_pred.reshape(X.shape)
    
    # Візуалізація
    fig = plt.figure(figsize=(12, 4))
    
    # Оригінальна функція
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('Оригінальна функція')
    
    # Прогнозована функція
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, scaler_y.inverse_transform(Z_pred.reshape(-1, 1)).reshape(X.shape), 
                     cmap='viridis', alpha=0.8)
    ax2.set_title('Прогноз ANFIS')
    
    # Похибка
    ax3 = fig.add_subplot(133, projection='3d')
    error = np.abs(Z - scaler_y.inverse_transform(Z_pred.reshape(-1, 1)).reshape(X.shape))
    ax3.plot_surface(X, Y, error, cmap='hot', alpha=0.8)
    ax3.set_title('Абсолютна похибка')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Нейро-нечітка мережа ANFIS")
    print("=" * 50)
    
    # Запуск демонстрації
    model = demo_anfis()
    
    print("\n" + "=" * 50)
    print("Демонстрація для нелінійної функції...")
    custom_problem_example()