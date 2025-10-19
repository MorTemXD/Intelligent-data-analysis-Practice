import numpy as np
import random

# ===================================================================
# 1. Спряжені градієнти з розширеним логуванням
# ===================================================================
def print_progress(history, freq=5, method_name="CG"):
    print(f"\n=== Деталізація ітерацій для {method_name} ===")
    print("-" * 80)
    print(f"{'Ітер':<4} | {'x':<18} | {'f(x)':<9} | {'||grad||':<11} | {'Δf':<10}")
    print("-" * 80)
    prev_f = history[0][1]
    for i, (x, fval, normg) in enumerate(history):
        if i % freq == 0 or i == len(history) - 1 or i <= 3:
            delta = f"{(prev_f - fval):+.2e}" if i else "----------"
            vec = "[" + ", ".join(f"{xi:+.3f}" for xi in x) + "]"
            print(f"{i:<4} | {vec:<18} | {fval:<9.6f} | {normg:<11.2e} | {delta:<10}")
            prev_f = fval
    print("-" * 80)

def f_rosen(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosen(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

def cg_fletcher_reeves(f, grad_f, x0, max_iter=100, tol=1e-8, freq=5):
    x = x0.copy()
    g = grad_f(x)
    d = -g
    history = [(x.copy(), f(x), np.linalg.norm(g))]
    for k in range(max_iter):
        alpha = 1.0
        while f(x + alpha * d) > f(x) + 1e-4 * alpha * np.dot(g, d):
            alpha *= 0.5
            if alpha < 1e-8:
                break
        x_new = x + alpha * d
        g_new = grad_f(x_new)
        beta = np.dot(g_new, g_new) / np.dot(g, g)
        d = -g_new + beta * d
        x, g = x_new, g_new
        history.append((x.copy(), f(x), np.linalg.norm(g)))
        if np.linalg.norm(g) < tol:
            break
    print_progress(history, freq, "Fletcher-Reeves")
    print(f"Фінал: x={x}, f(x)={f(x):.6e}, ітерацій={len(history) - 1}\n")
    return x, f(x)

def cg_polak_ribiere(f, grad_f, x0, max_iter=100, tol=1e-8, freq=5):
    x = x0.copy()
    g = grad_f(x)
    d = -g
    history = [(x.copy(), f(x), np.linalg.norm(g))]
    for k in range(max_iter):
        alpha = 1.0
        while f(x + alpha * d) > f(x) + 1e-4 * alpha * np.dot(g, d):
            alpha *= 0.5
            if alpha < 1e-8:
                break
        x_new = x + alpha * d
        g_new = grad_f(x_new)
        y = g_new - g
        beta = max(0, np.dot(g_new, y) / np.dot(g, g))
        d = -g_new + beta * d
        x, g = x_new, g_new
        history.append((x.copy(), f(x), np.linalg.norm(g)))
        if np.linalg.norm(g) < tol:
            break
    print_progress(history, freq, "Polak-Ribière")
    print(f"Фінал: x={x}, f(x)={f(x):.6e}, ітерацій={len(history) - 1}\n")
    return x, f(x)

# ===================================================================
# 2. Метод Заутендайка (з фіксаціями)
# ===================================================================
def zoutendijk(f, grad_f, constraints, x0, max_iter=100, tol=1e-6, eps=1e-6):
    x = np.copy(x0)
    history = [(x.copy(), f(x), np.linalg.norm(grad_f(x)))]
    for k in range(max_iter):
        g = grad_f(x)
        active = [i for i, gi in enumerate(constraints) if gi(x) >= -eps]
        d = -g
        for i in active:
            grad_c = numerical_grad(constraints[i], x)
            norm_grad_c = np.linalg.norm(grad_c) ** 2 + 1e-10
            d -= np.dot(d, grad_c) * grad_c / norm_grad_c
        if np.linalg.norm(d) < 1e-10:
            history.append((x.copy(), f(x), np.linalg.norm(grad_f(x))))
            break
        d = d / np.linalg.norm(d)  # Нормалізація напряму
        step = 1.0
        for i, gi in enumerate(constraints):
            if np.dot(numerical_grad(gi, x), d) >= 0:
                continue
            t = 0.0
            while gi(x + t * d) > eps and t < step:
                t += 0.01
            step = min(step, t)
        if step < 1e-10:
            history.append((x.copy(), f(x), np.linalg.norm(grad_f(x))))
            break
        x_new = x + step * d
        history.append((x_new.copy(), f(x_new), np.linalg.norm(grad_f(x_new))))
        if np.linalg.norm(grad_f(x_new)) < tol or np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    print_progress(history, 5, "ЗАУТЕНДАЙК")
    print(f"Фінал: x={x}, f(x)={f(x):.6e}, ітерацій={len(history) - 1}\n")
    for i, gi in enumerate(constraints):
        print(f"Обмеження {i + 1}: g_i(x_final)={gi(x):+.4e} (<=0)")
    return x, f(x)

def numerical_grad(f, x, h=1e-8):
    g = np.zeros(len(x))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = h
        g[i] = (f(x + dx) - f(x - dx)) / (2 * h)
    return g

# ===================================================================
# 3. Градієнтний спуск (з безпекою)
# ===================================================================
def steepest_descent(f, grad_f, x0, lr=0.00005, max_iter=200, tol=1e-8):
    x = x0.copy()
    history = [(x.copy(), f(x), np.linalg.norm(grad_f(x)))]
    for i in range(max_iter):
        g = grad_f(x)
        g = np.clip(g, -1000, 1000)
        x_new = x - lr * g
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            print("Error: overflow!")
            break
        history.append((x_new.copy(), f(x_new), np.linalg.norm(grad_f(x_new))))
        if np.linalg.norm(grad_f(x_new)) < tol:
            break
        x = x_new
    print_progress(history, 10, "Steepest Descent (fixed)")
    print(f"Фінал: x={x}, f(x)={f(x):.6e}, ітер={len(history) - 1}")
    return x, f(x)

# ===================================================================
# 4. Еволюційний алгоритм
# ===================================================================
def evolutionary_algorithm(fitness_func, bounds, pop_size=20, generations=40):
    log = []
    n_dim = len(bounds) if hasattr(bounds[0], '__len__') else 2
    bound = bounds if hasattr(bounds[0], '__len__') else [bounds] * n_dim
    pop = np.array([np.random.uniform(bound[i][0], bound[i][1], n_dim) for i in range(n_dim) for _ in range(pop_size // n_dim)])
    pop = pop[:pop_size]
    for gen in range(generations + 1):
        fits = np.array([fitness_func(x) for x in pop])
        log.append({"gen": gen, "min": np.min(fits), "avg": np.mean(fits), "max": np.max(fits)})
        if gen == generations:
            break
        indices = [np.argmin(fits[np.random.choice(pop_size, 3, replace=False)]) for _ in range(pop_size)]
        parents = pop[indices]
        children = []
        for i in range(0, pop_size, 2):
            alpha = np.random.rand()
            child1 = alpha * parents[i] + (1 - alpha) * parents[(i + 1) % pop_size]
            child2 = alpha * parents[(i + 1) % pop_size] + (1 - alpha) * parents[i]
            children += [child1, child2]
        pop = np.array(children[:pop_size])
        for i in range(pop_size):
            if np.random.rand() < 0.1:
                pop[i] += np.random.normal(scale=0.1, size=n_dim)
                for j, (l, r) in enumerate(bound):
                    pop[i][j] = np.clip(pop[i][j], l, r)
    print("\n=== Хід еволюційного алгоритму ===")
    print("Gen |   min   |   avg   |   max")
    print("---------------------------------")
    for stat in log[::max(1, len(log) // 8)] + [log[-1]]:
        print(f"{stat['gen']:<3} | {stat['min']:7.4f} | {stat['avg']:7.4f} | {stat['max']:7.4f}")
    best_idx = np.argmin([fitness_func(x) for x in pop])
    print(f"Фінал: min={log[-1]['min']:.4f}, avg={log[-1]['avg']:.4f}, найкращий індивід: {pop[best_idx]}")
    return pop[best_idx]

# ===================================================================
# 5. Оператори відбору
# ===================================================================
def selection_demo(n=100):
    pop = [np.random.rand(2) * 2 - 1 for _ in range(n)]
    fits = [np.sum(x**2) for x in pop]
    # Турнір
    selected = [pop[np.argmin([fits[i] for i in np.random.choice(n, 3)])] for _ in range(n)]
    print("\nТурнірний відбір: середня якість =", np.mean([np.sum(x**2) for x in selected]))
    # Ранжування
    indices = np.argsort(fits)
    ranks = np.arange(n)
    probs = (n - ranks) / np.sum(n - ranks)
    selected = [pop[np.random.choice(range(n), p=probs)] for _ in range(n)]
    print("Ранжування: середня якість =", np.mean([np.sum(x**2) for x in selected]))
    # Пропорційний
    inv = 1 / (np.array(fits) + 1e-4)
    probs = inv / np.sum(inv)
    selected = [pop[np.random.choice(range(n), p=probs)] for _ in range(n)]
    print("Пропорційний: середня якість =", np.mean([np.sum(x**2) for x in selected]))

# ===================================================================
# 6. Мурашина колонія (TSP)
# ===================================================================
def ant_colony_demo():
    print("\n=== Ant Colony Optimization (TSP) ===")
    dmat = np.random.randint(10, 100, (6, 6))
    np.fill_diagonal(dmat, 0)
    dmat = (dmat + dmat.T) / 2  # Симетрична матриця
    pheromones = np.ones_like(dmat, dtype=float) * 0.1
    n_ants = 5
    n_iter = 10
    best_length = np.inf
    best_path = None
    for it in range(n_iter + 1):
        paths, lengths = [], []
        for ant in range(n_ants):
            path = [0]
            unvisited = list(range(1, 6))
            for _ in range(5):
                probs = [pheromones[path[-1]][j] / (dmat[path[-1]][j] + 1e-10) for j in unvisited]
                probs = np.array(probs) / np.sum(probs)
                next_city = np.random.choice(unvisited, p=probs)
                path.append(next_city)
                unvisited.remove(next_city)
            length = sum(dmat[path[i], path[(i + 1) % 6]] for i in range(6))
            if length < best_length:
                best_length = length
                best_path = path
            paths.append(path)
            lengths.append(length)
        print(f"Ітерація {it:2}: кращий шлях {min(lengths):.1f}, середній={np.mean(lengths):.1f}")
        pheromones *= 0.95  # Випаровування
        for path, length in zip(paths, lengths):
            for i in range(6):
                pheromones[path[i], path[(i + 1) % 6]] += 1 / (length + 1e-10)
                pheromones[path[(i + 1) % 6], path[i]] += 1 / (length + 1e-10)
    print(f"Фінальний кращий шлях: {best_path}, довжина = {best_length:.1f}")

# ===================================================================
# Головний запуск
# ===================================================================
if __name__ == "__main__":
    np.random.seed(42)  # Для відтворюваності
    print("== Спряжені градієнти ==")
    x0 = np.array([-1.2, 1.0])
    cg_fletcher_reeves(f_rosen, grad_rosen, x0, max_iter=25, freq=5)
    cg_polak_ribiere(f_rosen, grad_rosen, x0, max_iter=25, freq=5)
    print("\n== Метод Заутендайка ==")
    f2 = lambda x: x[0]**2 + x[1]**2
    grad2 = lambda x: np.array([2 * x[0], 2 * x[1]])
    con2 = [lambda x: x[0] + x[1] - 1, lambda x: -x[0], lambda x: -x[1]]
    zoutendijk(f2, grad2, con2, np.array([0.5, 0.5]))
    print("\n== Градієнтний спуск (fixed) ==")
    steepest_descent(f_rosen, grad_rosen, x0, lr=0.00005)
    print("\n== Еволюційний алгоритм ==")
    evolutionary_algorithm(lambda x: (x[0]**2 + x[1]**4), (-2, 2), pop_size=20, generations=40)
    print("\n== Оператори відбору ==")
    selection_demo(100)
    print("\n== Мурашина колонія ==")
    ant_colony_demo()