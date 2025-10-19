
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. K-MEANS АЛГОРИТМ
# =============================================
def kmeans_pp_init(X, k, rng):
    """Ініціалізація центроїдів методом k-means++"""
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]))
    # Перший центроїд випадково
    idx = rng.integers(0, n)
    centroids[0] = X[idx]
    # Решта за k-means++
    for i in range(1, k):
        d2 = np.min(np.sum((X[:, None, :] - centroids[None, :i, :])**2, axis=2), axis=1)
        probs = d2 / d2.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
    return centroids

def kmeans(X, k, max_iter=100, tol=1e-4, n_init=10, rng=None):
    """Основний алгоритм k-means з кількома ініціалізаціями"""
    if rng is None:
        rng = np.random.default_rng()
    
    best_inertia = np.inf
    best_labels = None
    best_centroids = None
    
    for init in range(n_init):
        C = kmeans_pp_init(X, k, rng)
        prev_C = None
        
        for iteration in range(max_iter):
            # Призначення точок до кластерів
            d2 = np.sum((X[:, None, :] - C[None, :, :])**2, axis=2)
            labels = np.argmin(d2, axis=1)
            
            # Оновлення центроїдів
            C_new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) 
                            else C[j] for j in range(k)])
            
            # Перевірка збіжності
            shift = np.linalg.norm(C_new - C)
            C = C_new
            if shift < tol:
                break
        
        # Обчислення inertia (SSE)
        inertia = np.sum((X - C[labels])**2)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = C.copy()
    
    return best_centroids, best_labels, best_inertia

# =============================================
# 2. SOM (КАРТИ КОХОНЕНА)
# =============================================
class SOM:
    """Self-Organizing Map (Карта Кохонена)"""
    
    def __init__(self, m, n, dim, lr=0.5, sigma=None, rng=None):
        self.m, self.n, self.dim = m, n, dim
        self.lr0 = lr
        self.sigma0 = sigma if sigma is not None else max(m, n) / 2
        self.rng = rng or np.random.default_rng()
        
        # Ініціалізація ваг
        self.W = self.rng.normal(0, 1, size=(m, n, dim))
        
        # Координати решітки
        self.grid_y, self.grid_x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
        
        self.bmus = None
        
    def _bmu(self, x):
        """Знаходження Best Matching Unit"""
        d2 = np.sum((self.W - x)**2, axis=2)
        iy, ix = np.unravel_index(np.argmin(d2), (self.m, self.n))
        return iy, ix
    
    def fit(self, X, epochs=20, verbose=True):
        """Навчання SOM"""
        T = epochs * len(X)
        t = 0
        
        for epoch in range(epochs):
            self.rng.shuffle(X)
            for x in X:
                # Спадаючі параметри
                lr = self.lr0 * np.exp(-t / T)
                sigma = self.sigma0 * np.exp(-t / T)
                
                # Знаходження BMU
                iy, ix = self._bmu(x)
                
                # Гаусівська сусідська функція
                d2 = (self.grid_y - iy)**2 + (self.grid_x - ix)**2
                h = np.exp(-d2 / (2 * (sigma**2) + 1e-8))
                
                # Оновлення ваг
                self.W += lr * h[..., None] * (x - self.W)
                t += 1
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, lr={lr:.4f}, sigma={sigma:.4f}")
    
    def transform(self, X):
        """Проекція даних на карту"""
        coords = np.array([self._bmu(x) for x in X])
        return coords
    
    def get_heatmap(self, X):
        """Теплова карта активації нейронів"""
        coords = self.transform(X)
        heat = np.zeros((self.m, self.n), dtype=int)
        for iy, ix in coords:
            heat[iy, ix] += 1
        return heat

# =============================================
# 3. MУРAШИНИЙ АЛГОРИТМ (ACO) ДЛЯ TSP
# =============================================
class AntColonyTSP:
    """Ant Colony Optimization для задачі Комівояжера"""
    
    def __init__(self, coords, alpha=1.0, beta=5.0, rho=0.5, Q=100.0):
        self.n_cities = len(coords)
        self.coords = np.array(coords)
        
        # Евклідові відстані
        self.D = np.sqrt(((self.coords[:, None, :] - self.coords[None, :, :])**2).sum(axis=2)) + 1e-9
        self.eta = 1.0 / self.D  # Евристика
        
        # Параметри ACO
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Феромони
        self.tau = np.full((self.n_cities, self.n_cities), 1.0)
        
    def tour_length(self, tour):
        """Довжина туру"""
        return sum(self.D[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))
    
    def solve(self, n_ants=None, n_iters=150, rng=None, elitist=True):
        """Розв'язання TSP"""
        if n_ants is None:
            n_ants = self.n_cities
        if rng is None:
            rng = np.random.default_rng()
        
        best_len = np.inf
        best_tour = None
        
        for it in range(n_iters):
            all_tours = []
            all_lens = []
            
            # Кожна мураха будує тур
            for a in range(n_ants):
                start = rng.integers(0, self.n_cities)
                unvisited = set(range(self.n_cities))
                unvisited.remove(start)
                tour = [start]
                current = start
                
                while unvisited:
                    J = list(unvisited)
                    # Ймовірності переходу
                    numerators = (self.tau[current, J]**self.alpha) * (self.eta[current, J]**self.beta)
                    p = numerators / numerators.sum()
                    nxt = rng.choice(J, p=p)
                    tour.append(nxt)
                    unvisited.remove(nxt)
                    current = nxt
                
                L = self.tour_length(tour)
                all_tours.append(tour)
                all_lens.append(L)
            
            # Оновлення феромонів
            self.tau *= (1.0 - self.rho)
            
            # Elitist strategy: підсилюємо найкращий тур
            if elitist:
                idx_best = np.argmin(all_lens)
                iter_best_tour = all_tours[idx_best]
                iter_best_len = all_lens[idx_best]
                deposit = self.Q / iter_best_len
                
                for i in range(self.n_cities):
                    u = iter_best_tour[i]
                    v = iter_best_tour[(i + 1) % self.n_cities]
                    self.tau[u, v] += deposit
                    self.tau[v, u] += deposit
            else:
                # Класичний Ant System: всі мурахи
                for tour, L in zip(all_tours, all_lens):
                    deposit = self.Q / L
                    for i in range(self.n_cities):
                        u = tour[i]
                        v = tour[(i + 1) % self.n_cities]
                        self.tau[u, v] += deposit
                        self.tau[v, u] += deposit
            
            # Оновлення глобального найкращого
            current_best_len = min(all_lens)
            if current_best_len < best_len:
                best_len = current_best_len
                best_tour = all_tours[np.argmin(all_lens)].copy()
                
                if it % 25 == 0:
                    print(f"Iter {it}: Best length = {best_len:.2f}")
        
        return best_tour, best_len

# =============================================
# ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ДАНИХ
# =============================================
def generate_synthetic_data(n_clusters=3, n_samples=200, rng=None):
    """Генерація синтетичних 2D даних з кластерами"""
    if rng is None:
        rng = np.random.default_rng(42)
    
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    X = []
    
    for i, center in enumerate(centers[:n_clusters]):
        cluster = rng.normal(center, [0.8, 0.9, 0.7][i], size=(n_samples, 2))
        X.append(cluster)
    
    return np.vstack(X)

# =============================================
# ВІЗУАЛІЗАЦІЯ
# =============================================
def plot_kmeans_results(X, centroids, labels, inertia):
    """Візуалізація результатів k-means"""
    plt.figure(figsize=(8, 6))
    
    # Розсіювання точок з кластерами
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidth=3)
    plt.title(f'K-means кластери (SSE: {inertia:.2f})')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()

def plot_som_results(som, X, heat):
    """Візуалізація SOM карти"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Теплова карта активації
    im1 = axes[0].imshow(heat, cmap='hot', interpolation='nearest')
    axes[0].set_title('Теплова карта активації')
    plt.colorbar(im1, ax=axes[0])
    
    # Ваги нейронів (перша координата)
    axes[1].imshow(som.W[:, :, 0], cmap='viridis')
    axes[1].set_title('Ваги нейронів (X1)')
    
    # Вага нейронів (друга координата)
    axes[2].imshow(som.W[:, :, 1], cmap='plasma')
    axes[2].set_title('Ваги нейронів (X2)')
    
    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

def plot_tsp_results(coords, best_tour, best_len):
    """Візуалізація результатів TSP"""
    tour_coords = coords[best_tour + [best_tour[0]]]  # Закритий цикл
    
    plt.figure(figsize=(10, 8))
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', color='blue', markersize=8)
    plt.plot(tour_coords[0, 0], tour_coords[0, 1], 'ro', markersize=12, label='Старт')
    for i, city in enumerate(best_tour):
        plt.annotate(str(city), (coords[city, 0], coords[city, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title(f'TSP Результат: довжина = {best_len:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# =============================================
# ДЕМОНСТРАЦІЯ РОБОТИ
# =============================================
def main():
    """Головна функція демонстрації"""
    print("=== ДЕМОНСТРАЦІЯ АЛГОРИТМІВ КЛАСТЕРНОГО АНАЛІЗУ ===\n")
    
    # Генерація даних
    rng = np.random.default_rng(42)
    X = generate_synthetic_data(n_clusters=3, n_samples=200, rng=rng)
    print(f"Згенеровано {X.shape[0]} точок у 2D просторі\n")
    
    # 1. K-MEANS
    print("1. K-MEANS КЛАСТЕРИЗАЦІЯ")
    centroids, labels, inertia = kmeans(X, k=3, n_init=10, rng=rng)
    print(f"Центроїди:\n{centroids}")
    print(f"Inertia (SSE): {inertia:.2f}")
    print(f"Silhouette Score: {silhouette_score(X, labels):.3f}\n")
    
    plot_kmeans_results(X, centroids, labels, inertia)
    
    # 2. SOM
    print("2. SOM (КАРТА КОХОНЕНА)")
    som = SOM(m=10, n=10, dim=2, lr=0.5, rng=rng)
    som.fit(X, epochs=20)
    heat = som.get_heatmap(X)
    
    hotspots = np.unravel_index(np.argsort(heat.ravel())[::-1][:5], heat.shape)
    print("Top-5 найактивніших нейронів:")
    for i, (iy, ix) in enumerate(zip(*hotspots)):
        print(f"  ({iy}, {ix}): {heat[iy, ix]} активацій")
    
    plot_som_results(som, X, heat)
    
    # 3. TSP (ACO)
    print("\n3. MУРAШИНИЙ АЛГОРИТМ ДЛЯ TSP")
    n_cities = 15
    coords = rng.random((n_cities, 2)) * 100
    aco = AntColonyTSP(coords)
    best_tour, best_len = aco.solve(n_ants=20, n_iters=200, rng=rng)
    
    print(f"Найкраща довжина туру: {best_len:.2f}")
    print(f"Оптимальний тур: {best_tour}")
    
    plot_tsp_results(coords, best_tour, best_len)
    
    print("\n=== ДЕМОНСТРАЦІЯ ЗАВЕРШЕНА ===")

if __name__ == "__main__":
    main()