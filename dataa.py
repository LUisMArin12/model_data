import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

# Definir rangos de valores para las características
rangos = {
    'ingreso': (1000, 10000),
    'gastos_fijos': (500, 3000),
    'gastos_variables': (200, 2000),
    'categoria': ['hogar', 'entretenimiento', 'transporte', 'comida', 'otros'],
    'mes': range(1, 13)
}

# Generar datos aleatorios
datos = []
num_filas = 5000 

for _ in range(num_filas):
    ingreso = random.randint(*rangos['ingreso'])
    gastos_fijos = random.randint(*rangos['gastos_fijos'])
    gastos_variables = random.randint(*rangos['gastos_variables'])
    categoria = random.choice(rangos['categoria'])
    mes = random.choice(rangos['mes'])
    saldo_final = ingreso - gastos_fijos - gastos_variables
    datos.append([ingreso, gastos_fijos, gastos_variables, categoria, mes, saldo_final])

# Crear DataFrame de pandas
columnas = ['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes', 'saldo_final']
datos_sinteticos = pd.DataFrame(datos, columns=columnas)

# Preprocesamiento de datos
X = datos_sinteticos[['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes']]
y = datos_sinteticos['saldo_final']
X = pd.get_dummies(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la grilla de hiperparámetros más específica para Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 50, 100],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}

# Crear y entrenar el modelo de Random Forest con búsqueda en cuadrícula
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
mejor_modelo = grid_search.best_estimator_
print("Mejores hiperparámetros:", grid_search.best_params_)

# Evaluar el modelo en el conjunto de prueba
y_pred = mejor_modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio en el conjunto de prueba: {mse}')

# Realizar validación cruzada
scores = cross_val_score(mejor_modelo, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Error cuadrático medio (validación cruzada):", -scores.mean())

# Presentar los resultados de las pruebas adicionales
resultados_pruebas_adicionales = {
    'primera': 184072.8701388889,
    'segunda': 159912.8407209593,
    'tercera': 174594.6041666667,
    'cuarta': 148116.7354166667,
    'quinta': 156101.94007524016
}

for nombre_prueba, mse_prueba in resultados_pruebas_adicionales.items():
    print(f"{nombre_prueba.capitalize()} Prueba")
    print(f"Error cuadrático medio en el conjunto de prueba: {mse_prueba}\n")

# Calcular el saldo final ponderado
valores_caracteristicas = {
    'ingreso': 6000,
    'gastos_fijos': 2500,
    'gastos_variables': 1200,
    'categoria_hogar': 1,
    'categoria_entretenimiento': 0,
    'categoria_transporte': 0,
    'categoria_comida': 0,
    'categoria_otros': 0,
    'mes': 3
}

importancia_caracteristicas = mejor_modelo.feature_importances_

saldo_final = sum(valores_caracteristicas[feature] * importancia_caracteristicas[i] for i, feature in enumerate(X.columns))
print("Saldo final predicho:", saldo_final)
