import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Genero datos sinteticos

np.random.seed(42)
n_samples = 50
n_features = 5

# Creao matriz X cond caracteristicas correlacionadas 

X_simple = 2 * np.random.rand(n_samples, 1)



# Coeficientes reales

true_coefs = np.array([1.5, -2.0, 0.5, -1.0, 0.8])

# Caluclar y = X * coeficientes + ruido

y_simple = 4 + 3 * X_simple[:, 0] + np.random.randn(n_samples)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=42
)


# Crear instancia del modelo
ols_model = LinearRegression()

# Entrenear el modelo

ols_model.fit(X_train_s, y_train_s)

# Predicciones 
y_pred_ols = ols_model.predict(X_test_s)

# Calcular metricas

mse_ols = mean_squared_error(y_test_s, y_pred_ols)
r2_ols = r2_score(y_test_s, y_pred_ols)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train_s)


alphas = [0.01, 0.1, 1.0, 10.0]
ridge_coefs = None

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_s, y_train_s)
    
    if alpha == 1.0:
        ridge_coefs = ridge_model.coef_
    
    y_pred_ridge = ridge_model.predict(X_test_s)
    mse_ridge = mean_squared_error(y_test_s, y_pred_ridge)
    
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

# Crear figura con 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- SUBPLOT 1: OLS ---
ax1.scatter(X_train_s, y_train_s, color='blue', alpha=0.5, label='Datos entrenamiento')
ax1.scatter(X_test_s, y_test_s, color='green', alpha=0.5, label='Datos prueba')

# LA RECTA DE REGRESIÓN OLS
X_line = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
y_pred_line_ols = ols_model.predict(X_line)
ax1.plot(X_line, y_pred_line_ols, color='red', linewidth=2, 
         label=f'Recta OLS: y = {ols_model.intercept_:.2f} + {ols_model.coef_[0]:.2f}x')

ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Regresión Lineal OLS - LA RECTA QUE MEJOR SE AJUSTA')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- SUBPLOT 2: Ridge ---
ax2.scatter(X_train_s, y_train_s, color='blue', alpha=0.5, label='Datos entrenamiento')
ax2.scatter(X_test_s, y_test_s, color='green', alpha=0.5, label='Datos prueba')

# LA RECTA DE REGRESIÓN RIDGE
y_pred_line_ridge = ridge.predict(X_line)
ax2.plot(X_line, y_pred_line_ridge, color='orange', linewidth=2,
         label=f'Recta Ridge: y = {ridge.intercept_:.2f} + {ridge.coef_[0]:.2f}x')

ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.set_title('Regresión Ridge - TAMBIÉN ES UNA RECTA')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()