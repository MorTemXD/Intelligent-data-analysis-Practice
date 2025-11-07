import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz
from skfuzzy import control as ctrl

print("Крок 1-2: Будуємо еталонну поверхню...")

def target_function(x1, x2):
    return (x1**2) * np.sin(x2 - 1)

n = 15
x1_domain = np.linspace(-7, 3, n)
x2_domain = np.linspace(-4.4, 1.7, n)
X1, X2 = np.meshgrid(x1_domain, x2_domain)

Y_target = target_function(X1, X2)

fig = plt.figure(figsize=(20, 7))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X1, X2, Y_target, cmap='viridis')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.set_title('Еталонна поверхня (Target)')
ax1.view_init(elev=20., azim=-120)


print("Крок 3-10: Будуємо систему Сугено вручну...")

x1_universe = np.arange(-7, 3.1, 0.1)
x2_universe = np.arange(-4.4, 1.71, 0.1)

x1_lo_params = [-11, -7, -2]
x1_mid_params = [-6, -2, 2]
x1_hi_params = [-2, 3, 8]

x1_lo_mf = fuzz.trimf(x1_universe, x1_lo_params)
x1_mid_mf = fuzz.trimf(x1_universe, x1_mid_params)
x1_hi_mf = fuzz.trimf(x1_universe, x1_hi_params)

x2_lo_params = [-6.8, -4.4, -1.35]
x2_mid_params = [-3.79, -1.35, 1.09]
x2_hi_params = [-1.35, 1.7, 4.75]

x2_lo_mf = fuzz.trimf(x2_universe, x2_lo_params)
x2_mid_mf = fuzz.trimf(x2_universe, x2_mid_params)
x2_hi_mf = fuzz.trimf(x2_universe, x2_hi_params)

Y_sugeno = np.zeros_like(Y_target)

for i in range(n):
    for j in range(n):
        x1_val = X1[i, j]
        x2_val = X2[i, j]
        
        mem_x1_lo = fuzz.interp_membership(x1_universe, x1_lo_mf, x1_val)
        mem_x1_mid = fuzz.interp_membership(x1_universe, x1_mid_mf, x1_val)
        mem_x1_hi = fuzz.interp_membership(x1_universe, x1_hi_mf, x1_val)
        
        mem_x2_lo = fuzz.interp_membership(x2_universe, x2_lo_mf, x2_val)
        mem_x2_mid = fuzz.interp_membership(x2_universe, x2_mid_mf, x2_val)
        mem_x2_hi = fuzz.interp_membership(x2_universe, x2_hi_mf, x2_val)
        
        alpha_R1 = np.fmin(mem_x1_mid, mem_x2_mid)
        alpha_R2 = np.fmin(mem_x1_hi, mem_x2_hi)
        alpha_R3 = np.fmin(mem_x1_hi, mem_x2_lo)
        alpha_R4 = np.fmin(mem_x1_lo, mem_x2_mid)
        alpha_R5 = np.fmin(mem_x1_lo, mem_x2_lo)
        alpha_R6 = np.fmin(mem_x1_lo, mem_x2_hi)
        alpha_R7 = np.fmin(mem_x1_mid, mem_x2_lo)
        alpha_R8 = np.fmin(mem_x1_mid, mem_x2_hi)
        alpha_R9 = np.fmin(mem_x1_hi, mem_x2_mid)
        
        z_R1 = 0
        z_R2 = 2*x1_val + 2*x2_val + 1
        z_R3 = 4*x1_val - x2_val
        z_R4 = 8*x1_val + 2*x2_val + 8
        z_R5 = 50
        z_R6 = 50
        z_R7 = 0
        z_R8 = 0
        z_R9 = 0
        
        numerator = (alpha_R1 * z_R1) + (alpha_R2 * z_R2) + (alpha_R3 * z_R3) + \
                    (alpha_R4 * z_R4) + (alpha_R5 * z_R5) + (alpha_R6 * z_R6) + \
                    (alpha_R7 * z_R7) + (alpha_R8 * z_R8) + (alpha_R9 * z_R9)
                    
        denominator = alpha_R1 + alpha_R2 + alpha_R3 + alpha_R4 + \
                      alpha_R5 + alpha_R6 + alpha_R7 + alpha_R8 + alpha_R9
        
        if denominator == 0:
            Y_sugeno[i, j] = 0
        else:
            Y_sugeno[i, j] = numerator / denominator

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X1, X2, Y_sugeno, cmap='plasma')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title('Поверхня Сугено (Ручна реалізація)')
ax2.view_init(elev=20., azim=-120)


print("Крок 11: Будуємо систему Мамдані...")

x1_m = ctrl.Antecedent(np.arange(-7, 3.1, 0.1), 'x1')
x2_m = ctrl.Antecedent(np.arange(-4.4, 1.71, 0.1), 'x2')

y_m = ctrl.Consequent(np.arange(-50, 51, 1), 'y_mamdani')

x1_m['lo'] = fuzz.trimf(x1_m.universe, [-11, -7, -2])
x1_m['mid'] = fuzz.trimf(x1_m.universe, [-6, -2, 2])
x1_m['hi'] = fuzz.trimf(x1_m.universe, [-2, 3, 8])

x2_m['lo'] = fuzz.trimf(x2_m.universe, [-6.8, -4.4, -1.35])
x2_m['mid'] = fuzz.trimf(x2_m.universe, [-3.79, -1.35, 1.09])
x2_m['hi'] = fuzz.trimf(x2_m.universe, [-1.35, 1.7, 4.75])

y_m['NB'] = fuzz.trimf(y_m.universe, [-50, -50, -25])
y_m['NS'] = fuzz.trimf(y_m.universe, [-50, -25, 0])
y_m['ZE'] = fuzz.trimf(y_m.universe, [-25, 0, 25])
y_m['PS'] = fuzz.trimf(y_m.universe, [0, 25, 50])
y_m['PB'] = fuzz.trimf(y_m.universe, [25, 50, 50])

rule1_m = ctrl.Rule(x1_m['lo'] & x2_m['lo'], y_m['PB'])
rule2_m = ctrl.Rule(x1_m['lo'] & x2_m['mid'], y_m['NB'])
rule3_m = ctrl.Rule(x1_m['lo'] & x2_m['hi'], y_m['PB'])
rule4_m = ctrl.Rule(x1_m['mid'] & x2_m['lo'], y_m['PS'])
rule5_m = ctrl.Rule(x1_m['mid'] & x2_m['mid'], y_m['ZE'])
rule6_m = ctrl.Rule(x1_m['mid'] & x2_m['hi'], y_m['PS'])
rule7_m = ctrl.Rule(x1_m['hi'] & x2_m['lo'], y_m['NS'])
rule8_m = ctrl.Rule(x1_m['hi'] & x2_m['mid'], y_m['NS'])
rule9_m = ctrl.Rule(x1_m['hi'] & x2_m['hi'], y_m['PS'])

mamdani_ctrl_sys = ctrl.ControlSystem([rule1_m, rule2_m, rule3_m, rule4_m, rule5_m, rule6_m, rule7_m, rule8_m, rule9_m])
mamdani_sim = ctrl.ControlSystemSimulation(mamdani_ctrl_sys)

Y_mamdani = np.zeros_like(Y_target)
for i in range(n):
    for j in range(n):
        mamdani_sim.input['x1'] = X1[i, j]
        mamdani_sim.input['x2'] = X2[i, j]
        mamdani_sim.compute()
        Y_mamdani[i, j] = mamdani_sim.output['y_mamdani']

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X1, X2, Y_mamdani, cmap='cividis')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title('Поверхня Мамдані (skfuzzy)')
ax3.view_init(elev=20., azim=-120)

print("Крок 12: Відображення результатів для порівняння...")
plt.tight_layout()
plt.show()

print("Код виконано.")