import numpy as np
import matplotlib.pyplot as plt

N = 100
delta = 0.5
sigma = 2.0
y0 = 10

tau_max = 30

epsilon = np.random.normal(loc=0.0, scale=sigma, size=N)

y_series = np.zeros(N)
y_series[0] = y0

for t in range(1, N):
    y_series[t] = delta + y_series[t-1] + epsilon[t]
    
y_last = y_series[-1]
t_last = N - 1

tau_steps = np.arange(1, tau_max + 1)
forecast_t = t_last + tau_steps

forecast_series = y_last + tau_steps * delta

forecast_std = np.sqrt(tau_steps) * sigma

conf_interval_upper = forecast_series + 1.96 * forecast_std
conf_interval_lower = forecast_series - 1.96 * forecast_std

plt.figure(figsize=(14, 7))
plt.style.use('ggplot')

plt.plot(np.arange(N), y_series, label='Історичні дані (Y_t)', color='blue')

plt.plot(t_last, y_last, 'ro', markersize=8, label='Останнє значення (Y_t)')

plt.plot(forecast_t, forecast_series, 'r--', 
         label=f'Прогноз (Y_t + τ*δ) при δ={delta}')

plt.fill_between(forecast_t, conf_interval_lower, conf_interval_upper, 
                 color='red', alpha=0.15, 
                 label=f'95% довірчий інтервал (MSE = τ*σ²)')

plt.title('Симуляція та прогноз для випадкового блукання з трендом')
plt.xlabel('Час (t)')
plt.ylabel('Значення (Y_t)')
plt.legend()
plt.grid(True)
plt.show()