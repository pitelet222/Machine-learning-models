import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

# Cargo el database de House prices

df = pd.read_csv("data/train.csv")

# Preparar los datos
y = df["SalePrice"].copy()
# High-impact numerical features
numerical_features = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", 
    "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF"
]

# High-impact categorical features
categorical_features = [
    "MSZoning", "Neighborhood", "BldgType", "HouseStyle", 
    "ExterQual", "Foundation", "BsmtQual", "HeatingQC", 
    "CentralAir", "KitchenQual", "GarageType", "SaleType"
]
# Separate numerical and categorical data
X_numerical = df[numerical_features].copy()
X_categorical = df[categorical_features].copy()

# Clean numerical features
X_numerical = X_numerical.fillna(X_numerical.median())

# Clean categorical features (fill with most frequent value)
for col in categorical_features:
    X_categorical[col] = X_categorical[col].fillna(X_categorical[col].mode()[0])

# Create preprocessor for categorical features
categorical_preprocessor = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', list(range(len(numerical_features)))),
        ('cat', categorical_preprocessor, list(range(len(categorical_features))))
    ]
)

# Combine numerical and categorical data
X_combined = pd.concat([X_numerical, X_categorical], axis=1)

# Apply preprocessing
X = preprocessor.fit_transform(X_combined)

# Get feature names after preprocessing
numerical_feature_names = numerical_features
categorical_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out())
all_feature_names = numerical_feature_names + categorical_feature_names

print(f"Total features after encoding: {len(all_feature_names)}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features encoded to: {len(categorical_feature_names)} columns")

# Ensure y matches X after any potential row drops
y = y[X_combined.index]

n_features = X.shape[1]

# Caluclar y = X * coeficientes + ruido

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Crear instancia del modelo
ols_model = LinearRegression()

# Entrenear el modelo

ols_model.fit(X_train, y_train)

# Predicciones 
y_pred_ols = ols_model.predict(X_test)

# Calcular metricas

mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)


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

x_pos = np.arange(len(ols_model.coef_))

plt.bar(x_pos - 0.2, ols_model.coef_, width=0.4, label="OLS", color="blue", alpha=0.7)
plt.bar(x_pos + 0.2, ridge_coefs, width=0.4, label="Ridge (α=1.0)", color="red", alpha=0.7)

plt.xlabel("Índice del coeficiente")
plt.ylabel("Valor del coeficiente")
plt.title("Comparacion de coeficientes: OLS vs Ridge")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Crear figura con 2 subplots
fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax_text = plt.subplot(2, 1, 2)

# --- SUBPLOT 1: OLS ---
# Scatter plot: Predicciones vs Valores reales
y_pred_train = ols_model.predict(X_train)
ax1.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Entrenamiento')
ax1.scatter(y_test, y_pred_ols, color='green', alpha=0.5, label='Prueba')

# Línea diagonal perfecta
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción perfecta')

ax1.set_xlabel('Valores reales ($)')
ax1.set_ylabel('Predicciones ($)')
ax1.set_ylabel('Precio de venta ($)')
ax2.set_ylabel('Precio de venta ($)')

# Calcular métricas para mostrar
mse_ols_test = mean_squared_error(y_test, y_pred_ols)
r2_ols_test = r2_score(y_test, y_pred_ols)
y_pred_test_ridge = ridge.predict(X_test)
mse_ridge_test = mean_squared_error(y_test, y_pred_test_ridge)
r2_ridge_test = r2_score(y_test, y_pred_test_ridge)

# --- SUBPLOT 2: Ridge ---
# Scatter plot: Predicciones vs Valores reales para Ridge
y_pred_train_ridge = ridge.predict(X_train)
y_pred_test_ridge = ridge.predict(X_test)
ax2.scatter(y_train, y_pred_train_ridge, color='blue', alpha=0.5, label='Entrenamiento')
ax2.scatter(y_test, y_pred_test_ridge, color='green', alpha=0.5, label='Prueba')

# Línea diagonal perfecta
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción perfecta')

ax2.set_xlabel('Valores reales ($)')
ax2.set_ylabel('Predicciones ($)')
ax2.set_title(f'Ridge - {len(all_feature_names)} features\nMSE: {mse_ridge_test:.0f}, R²: {r2_ridge_test:.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# Configure the text subplot (Step 6)
ax_text.axis('off')  # Remove axes from text area

# Show summary instead of all features (too many to display)
features_summary = f"Features: {len(numerical_features)} numerical + {len(categorical_features)} categorical\n"
features_summary += f"Total after encoding: {len(all_feature_names)} features\n"
features_summary += f"Categorical: {', '.join(categorical_features[:3])}..."

ax_text.text(0.5, 0.5, features_summary, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.show()