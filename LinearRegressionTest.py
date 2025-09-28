import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Genero datos sinteticos

np.random.seed(42)
n_samples = 100
n_features = 5

# Creao matriz X cond caracteristicas correlacionadas 

X = np.random.randn(n_samples, n_features)

X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
X[:, 2] = X[:, 0] + 0.3 * np.random.randn(n_samples)

# Coeficientes reales

true_coefs = np.array([1.5, -2.0, 0.5, -1.0, 0.8])

# Caluclar y = X * coeficientes + ruido

y = X.dot(true_coefs) + np.random.randn(n_samples) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear instancia del modelo
ols_model = LinearRegression()

# Entrenear el modelo

ols_model.fit(X_train, y_train)

# Predicciones 
y_pred_ols = ols_model.predict(X_test)

# Calcular metricas

mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)


alphas = [0.01, 0.1, 1.0, 10.0]
ridge_coefs = None

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    
    if alpha == 1.0:
        ridge_coefs = ridge_model.coef_
    
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    
    print(f"Alpha = {alpha}: MSE = {mse_ridge:.4f}")
    
plt.figure(figsize=(10,6))

x_pos = np.arange(n_features)

plt.bar(x_pos - 0.2, ols_model.coef_, width=0.4, label="OLS", color="blue", alpha=0.7)
plt.bar(x_pos + 0.2, ridge_coefs, width=0.4, label="Ridge (α=1.0)", color="red", alpha=0.7)

plt.xlabel("Índice del coeficiente")
plt.ylabel("Valor del coeficiente")
plt.title("Comparacion de coeficientes: OLS vs Ridge")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()