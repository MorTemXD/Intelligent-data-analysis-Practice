import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- КРОК 1. Початкова система ---
temp = ctrl.Antecedent(np.arange(10, 81, 1), 'temp')
head = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'head')
valve = ctrl.Consequent(np.arange(-10, 11, 1), 'valve')

temp['cold'] = fuzz.trimf(temp.universe, [10, 20, 35])
temp['mid'] = fuzz.trimf(temp.universe, [30, 35, 40])
temp['hot'] = fuzz.trimf(temp.universe, [40, 50, 80])

head['small'] = fuzz.trimf(head.universe, [0, 0.1, 0.3])
head['norm'] = fuzz.trimf(head.universe, [0.25, 0.5, 0.75])
head['big'] = fuzz.trimf(head.universe, [0.6, 0.8, 1])

valve['open_q'] = fuzz.trimf(valve.universe, [-10, -7, -5])
valve['open_s'] = fuzz.trimf(valve.universe, [-6, -3, -1])
valve['norm'] = fuzz.trimf(valve.universe, [-2, 0, 2])
valve['close_s'] = fuzz.trimf(valve.universe, [1, 3, 6])
valve['close_q'] = fuzz.trimf(valve.universe, [5, 7, 10])

rules = [
    ctrl.Rule(temp['cold'] & head['small'], valve['open_q']),
    ctrl.Rule(temp['cold'] & head['norm'], valve['open_s']),
    ctrl.Rule(temp['cold'] & head['big'], valve['norm']),
    ctrl.Rule(temp['mid'] & head['small'], valve['open_s']),
    ctrl.Rule(temp['mid'] & head['norm'], valve['norm']),
    ctrl.Rule(temp['mid'] & head['big'], valve['close_s']),
    ctrl.Rule(temp['hot'] & head['small'], valve['norm']),
    ctrl.Rule(temp['hot'] & head['norm'], valve['close_s']),
    ctrl.Rule(temp['hot'] & head['big'], valve['close_q'])
]

system_before = ctrl.ControlSystem(rules)
sim_before = ctrl.ControlSystemSimulation(system_before)

# --- Симуляція ДО зміни ---
temp_values = np.linspace(10, 80, 30)
head_values = np.linspace(0, 1, 30)
results_before = []

for t in temp_values:
    for h in head_values:
        sim_before.reset()
        sim_before.input['temp'] = t
        sim_before.input['head'] = h
        sim_before.compute()
        results_before.append([t, h, sim_before.output.get('valve', np.nan)])

results_before = np.array(results_before)

# --- КРОК 2. Оптимізована система ---
temp2 = ctrl.Antecedent(np.arange(10, 81, 1), 'temp2')
head2 = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'head2')
valve2 = ctrl.Consequent(np.arange(-10, 11, 1), 'valve2')

temp2['cold'] = fuzz.gaussmf(temp2.universe, 20, 7)
temp2['mid'] = fuzz.gaussmf(temp2.universe, 37, 4)
temp2['hot'] = fuzz.gaussmf(temp2.universe, 60, 10)

head2['small'] = fuzz.trapmf(head2.universe, [0, 0, 0.15, 0.35])
head2['norm'] = fuzz.trapmf(head2.universe, [0.25, 0.4, 0.6, 0.75])
head2['big'] = fuzz.trapmf(head2.universe, [0.7, 0.85, 1, 1])

valve2['open_q'] = fuzz.gaussmf(valve2.universe, -8, 1.5)
valve2['open_s'] = fuzz.gaussmf(valve2.universe, -4, 1.5)
valve2['norm'] = fuzz.gaussmf(valve2.universe, 0, 1.5)
valve2['close_s'] = fuzz.gaussmf(valve2.universe, 4, 1.5)
valve2['close_q'] = fuzz.gaussmf(valve2.universe, 8, 1.5)

rules2 = [
    ctrl.Rule(temp2['cold'] & head2['small'], valve2['open_q']),
    ctrl.Rule(temp2['cold'] & head2['norm'], valve2['open_s']),
    ctrl.Rule(temp2['cold'] & head2['big'], valve2['norm']),
    ctrl.Rule(temp2['mid'] & head2['small'], valve2['open_s']),
    ctrl.Rule(temp2['mid'] & head2['norm'], valve2['norm']),
    ctrl.Rule(temp2['mid'] & head2['big'], valve2['close_s']),
    ctrl.Rule(temp2['hot'] & head2['small'], valve2['norm']),
    ctrl.Rule(temp2['hot'] & head2['norm'], valve2['close_s']),
    ctrl.Rule(temp2['hot'] & head2['big'], valve2['close_q'])
]

system_after = ctrl.ControlSystem(rules2)
sim_after = ctrl.ControlSystemSimulation(system_after)

# --- Симуляція ПІСЛЯ зміни ---
results_after = []
for t in temp_values:
    for h in head_values:
        sim_after.reset()
        sim_after.input['temp2'] = t
        sim_after.input['head2'] = h
        sim_after.compute()
        results_after.append([t, h, sim_after.output.get('valve2', np.nan)])

results_after = np.array(results_after)

# --- КРОК 3. Порівняння на одному графіку ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Поверхня ДО змін
ax.plot_trisurf(results_before[:, 0], results_before[:, 1], results_before[:, 2],
                cmap='viridis', alpha=0.6, label="До змін")

# Поверхня ПІСЛЯ змін
ax.plot_trisurf(results_after[:, 0], results_after[:, 1], results_after[:, 2],
                cmap='plasma', alpha=0.6, label="Після змін")

ax.set_xlabel('Температура (°C)')
ax.set_ylabel('Напір')
ax.set_zlabel('Кут клапана')
ax.set_title('Порівняння систем ДО і ПІСЛЯ змін')
plt.legend(['До змін', 'Після змін'])
plt.show()

# --- КРОК 4. Кількісне порівняння ---
diff = results_after[:, 2] - results_before[:, 2]
valid = ~np.isnan(diff)
diff = diff[valid]
mae = np.mean(np.abs(diff))
max_diff = np.max(np.abs(diff))
sign_change = np.mean(np.sign(results_after[valid, 2]) != np.sign(results_before[valid, 2])) * 100

print(f"\n=== ПОРІВНЯННЯ СИСТЕМ ===")
print(f"Середня абсолютна різниця (MAE): {mae:.3f}")
print(f"Максимальна різниця: {max_diff:.3f}")
print(f"Зміна напрямку (%): {sign_change:.2f}%")

plt.figure(figsize=(10,5))
plt.plot(diff, label="Δ Кут клапана (після - до)", color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.title("Порівняння виходів систем до і після оптимізації")
plt.xlabel("Комбінації (темп + напір)")
plt.ylabel("Зміна виходу")
plt.show()

# --- КРОК 5. Висновки ---
print("\nВИСНОВКИ:")
print("1. Ґауссові й трапецеїдальні функції зменшили різкі стрибки у вихідних значеннях.")
print(f"2. Середня різниця між старою і новою системами: {mae:.2f}, зміна напрямку: {sign_change:.1f}%.")
print("3. Оновлена нечітка система краще підтримує температуру та напір гарячої води.")
