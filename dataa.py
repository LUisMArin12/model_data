# import numpy as np
# import random
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split, GridSearchCV
# import pandas as pd

# # Definir rangos de valores para las características
# rangos = {
#     'ingreso': (1000, 10000),
#     'gastos_fijos': (500, 3000),
#     'gastos_variables': (200, 2000),
#     'categoria': ['hogar', 'entretenimiento', 'transporte', 'comida', 'otros'],
#     'mes': range(1, 13)
# }

# # Generar datos aleatorios
# datos = []
# num_filas = 5000 

# for _ in range(num_filas):
#     ingreso = random.randint(*rangos['ingreso'])
#     gastos_fijos = random.randint(*rangos['gastos_fijos'])
#     gastos_variables = random.randint(*rangos['gastos_variables'])
#     categoria = random.choice(rangos['categoria'])
#     mes = random.choice(rangos['mes'])
#     saldo_final = ingreso - gastos_fijos - gastos_variables
#     datos.append([ingreso, gastos_fijos, gastos_variables, categoria, mes, saldo_final])

# # Crear DataFrame de pandas
# columnas = ['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes', 'saldo_final']
# datos_sinteticos = pd.DataFrame(datos, columns=columnas)

# # Preprocesamiento de datos
# X = datos_sinteticos[['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes']]
# y = datos_sinteticos['saldo_final']
# X = pd.get_dummies(X)

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definir la grilla de hiperparámetros más específica para Random Forest
# param_grid = {
#     'n_estimators': [100, 200, 300],  
#     'max_depth': [None, 50, 100],  
#     'min_samples_split': [2, 5, 10],  
#     'min_samples_leaf': [1, 2, 4]  
# }

# # Crear y entrenar el modelo de Random Forest con búsqueda en cuadrícula
# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Obtener el mejor modelo
# mejor_modelo = grid_search.best_estimator_
# print("Mejores hiperparámetros:", grid_search.best_params_)

# # Evaluar el modelo en el conjunto de prueba
# y_pred = mejor_modelo.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Error cuadrático medio en el conjunto de prueba: {mse}')

# # Calcular el saldo final ponderado
# valores_caracteristicas = {
#     'ingreso': 6000,
#     'gastos_fijos': 2500,
#     'gastos_variables': 1200,
#     'categoria_hogar': 1,
#     'categoria_entretenimiento': 0,
#     'categoria_transporte': 0,
#     'categoria_comida': 0,
#     'categoria_otros': 0,
#     'mes': 3
# }

# importancia_caracteristicas = mejor_modelo.feature_importances_

# saldo_final = sum(valores_caracteristicas[feature] * importancia_caracteristicas[i] for i, feature in enumerate(X.columns))
# print("Saldo final predicho:", saldo_final)

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Datos específicos
datos_especificos = [
    [6000, 2500, 1200, 'hogar', 3],  # Ejemplo de datos específicos
    [8000, 2000, 1500, 'comida', 5],  # Ejemplo de datos específicos
    # Agrega más filas de datos específicos si es necesario
]

# Añadir datos aleatorios adicionales
num_filas_adicionales = 98
for _ in range(num_filas_adicionales):
    ingreso = np.random.randint(1000, 10000)
    gastos_fijos = np.random.randint(500, 3000)
    gastos_variables = np.random.randint(200, 2000)
    categoria = np.random.choice(['hogar', 'entretenimiento', 'transporte', 'comida', 'otros'])
    mes = np.random.randint(1, 13)
    datos_especificos.append([ingreso, gastos_fijos, gastos_variables, categoria, mes])

# Crear DataFrame de pandas con los datos específicos
columnas = ['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes']
datos_sinteticos = pd.DataFrame(datos_especificos, columns=columnas)

# Calcular el saldo final como la diferencia entre el ingreso y los gastos
datos_sinteticos['saldo_final'] = datos_sinteticos['ingreso'] - datos_sinteticos['gastos_fijos'] - datos_sinteticos['gastos_variables']

# Preprocesamiento de datos
X = datos_sinteticos[['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes']]
y = datos_sinteticos['saldo_final']

# Codificación one-hot para la característica 'categoria'
X = pd.get_dummies(X)

# Definir la grilla de hiperparámetros más específica para Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 50, 100],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}

# Crear y entrenar el modelo de Random Forest con búsqueda en cuadrícula
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

# Obtener el mejor modelo
mejor_modelo = grid_search.best_estimator_
print("Mejores hiperparámetros:", grid_search.best_params_)

# Calcular el saldo final predicho
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

saldo_final_predicho = mejor_modelo.predict([list(valores_caracteristicas.values())])[0]
print("Saldo final predicho:", saldo_final_predicho)
