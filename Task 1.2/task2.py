import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

# Task 1: Класифікація часових рядів
print("=== Запуск задачі 1: Класифікація часових рядів ===")
np.random.seed(1)
n = 300
trend = np.linspace(10, 20, n)
season = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 2, n)
y = trend + season + noise
adf_stat, p_value, _, _, _, _ = adfuller(y)
print(f"ADF статистика: {adf_stat:.3f}, p-value: {p_value:.3f}")
if p_value > 0.05:
    print("Ряд нестаціонарний (не відхиляємо H0)")
else:
    print("Ряд стаціонарний (відхиляємо H0)")

# Task 2: Коваріаційна матриця
print("\n=== Запуск задачі 2: Коваріаційна матриця ===")
X = np.array([[2.5, 3.0, 5.0],
              [1.8, 2.5, 6.2],
              [3.2, 4.1, 4.8],
              [2.9, 3.8, 5.5],
              [1.5, 2.0, 7.0]])
S = np.cov(X, rowvar=False, bias=False)
print("Коваріаційна матриця S:")
print(S)
print("Симетрична? ", np.allclose(S, S.T))

def covariance_demo():
    X = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 0.0],
                  [3.0, 4.0, 5.0],
                  [4.0, 3.0, 2.0],
                  [5.0, 6.0, 7.0]])
    S = np.cov(X, rowvar=False, bias=False)
    print("Додаткова коваріаційна матриця (demo):")
    print(S)
    print("Симетрична? ", np.allclose(S, S.T))

covariance_demo()

# Task 3: Лінійний дискримінантний аналіз (LDA)
print("\n=== Запуск задачі 3: Лінійний дискримінантний аналіз (LDA) ===")
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
cv_scores = cross_val_score(lda, X, y, cv=5, scoring='accuracy')
print("LDA класифікаційний звіт (тест):")
print(classification_report(y_test, y_pred))
print("Матриця помилок (тест):")
print(confusion_matrix(y_test, y_pred))
print(f"Середня точність CV: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Task 4: Відмінність між LDA та QDA
print("\n=== Запуск задачі 4: Відмінність між LDA та QDA ===")
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
lda_acc = accuracy_score(y_test, lda_pred)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
qda_acc = accuracy_score(y_test, qda_pred)
cv_lda_scores = cross_val_score(lda, X, y, cv=5, scoring='accuracy')
cv_qda_scores = cross_val_score(qda, X, y, cv=5, scoring='accuracy')
print(f"LDA точність (тест): {lda_acc:.3f}, CV: {cv_lda_scores.mean():.3f}")
print(f"QDA точність (тест): {qda_acc:.3f}, CV: {cv_qda_scores.mean():.3f}")

# Task 5: Грунтовний аналіз можливостей LDA і QDA
print("\n=== Запуск задачі 5: Грунтовний аналіз можливостей LDA і QDA ===")
wine = load_wine()
X, y = wine.data, wine.target
pipe_lda = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
])
pipe_rda = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5))  # RDA with fixed shrinkage
])
cv_lda_scores = cross_val_score(pipe_lda, X, y, cv=5, scoring='accuracy')
cv_rda_scores = cross_val_score(pipe_rda, X, y, cv=5, scoring='accuracy')
print(f"LDA CV: {cv_lda_scores.mean():.3f} (+/- {cv_lda_scores.std() * 2:.3f})")
print(f"RDA (shrinkage=0.5) CV: {cv_rda_scores.mean():.3f} (+/- {cv_rda_scores.std() * 2:.3f})")

# Task 6: Можливості пакету Statistics Toolbox (Python-аналог)
print("\n=== Запуск задачі 6: Можливості пакету Statistics Toolbox (Python-аналог) ===")
np.random.seed(42)
data = np.random.normal(5, 2, 200)
print(f"Mean: {np.mean(data):.3f}, Std: {np.std(data):.3f}")
print(f"KS-test: {stats.kstest(data, 'norm', args=(5,2))[:2]}")
group1 = np.random.normal(10, 1, 30)
group2 = np.random.normal(12, 1, 30)
f_stat, p_val = stats.f_oneway(group1, group2)
print(f"ANOVA F: {f_stat:.3f}, p: {p_val:.3f}")
X_cluster = np.random.rand(200, 3)
kmeans = KMeans(n_clusters=4, random_state=42).fit(X_cluster)
print(f"Inertia: {kmeans.inertia_:.3f}")
pca = PCA(n_components=2)
pca.fit(X_cluster)
print(f"Explained variance: {pca.explained_variance_ratio_}")
ts = np.cumsum(np.random.normal(1, 0.5, 300))  # Increased to 300 points
arima = ARIMA(ts, order=(1,1,1)).fit()  # Changed order to (1,1,1)
print(f"AIC: {arima.aic:.2f}")

# Task 7: Призначення і синтаксис функцій gscatter і classify (Python-аналоги)
print("\n=== Запуск задачі 7: Призначення і синтаксис функцій gscatter і classify (Python-аналоги) ===")
np.random.seed(42)
x = np.random.normal(0, 1, 90)
y = np.random.normal(0, 1, 90)
g = np.repeat([0, 1, 2], 30)
colors = ['c', 'm', 'y']
for i in range(3):
    plt.scatter(x[g == i], y[g == i], c=colors[i], label=f'Група {i}')
plt.legend()
plt.title('Групована scatter (аналог gscatter)')
plt.show()
x0 = np.random.normal(0, 1, (60, 2))
x1 = np.random.normal(3, 1, (60, 2))
training = np.vstack([x0, x1])
group = np.array([0] * 60 + [1] * 60)
sample = np.random.normal(2, 1, (15, 2))
lda = LinearDiscriminantAnalysis()
lda.fit(training, group)
pred = lda.predict(sample)
print(f"Передбачені класи: {pred}")
mean0 = np.mean(training[group == 0], axis=0)
cov0 = np.cov(training[group == 0].T)
dist = [mahalanobis(s, mean0, cov0) for s in sample]
print(f"Mahalanobis відстані (перші 3): {dist[:3]}")

# Приклад інтелектуальної задачі: Кластеризація клієнтів банку
print("\n=== Запуск прикладу інтелектуальної задачі: Кластеризація клієнтів банку ===")
np.random.seed(42)
age = np.random.randint(18, 70, 200)
income = np.random.normal(50000, 20000, 200)
spending = np.random.normal(50, 20, 200) + (income / 1000)
X = np.column_stack([age, income, spending])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Елбоу метод')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, labels)
print(f"Силуетний коефіцієнт: {sil_score:.3f}")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('Кластери клієнтів')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()