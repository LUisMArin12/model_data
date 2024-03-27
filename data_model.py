# import pandas as pd
# import numpy as np
# import random
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error

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
# num_filas = 100  # Número de filas de datos a generar

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

# # Crear escenarios hipotéticos
# escenarios = [
#     [6000, 2500, 1200, 'hogar', 3, 2300],
#     [8000, 3000, 1500, 'entretenimiento', 6, 3500],
#     [4000, 1800, 800, 'transporte', 9, 1400],
#     [10000, 2000, 2500, 'comida', 12, 5500],
#     [3000, 1200, 600, 'otros', 7, 1200]
# ]

# # Agregar escenarios hipotéticos al DataFrame
# datos_sinteticos = pd.concat([datos_sinteticos, pd.DataFrame(escenarios, columns=columnas)], ignore_index=True)

# # Preprocesamiento de datos
# X = datos_sinteticos[['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes']]
# y = datos_sinteticos['saldo_final']
# X = pd.get_dummies(X)

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definir la grilla de hiperparámetros a explorar
# param_grid = {
#     'n_estimators': [100, 200, 300],  # Número de árboles en el bosque
#     'max_depth': [None, 5, 10, 15],
#     'min_samples_leaf': [1, 2, 4, 8],
#     'min_samples_split': [2, 5, 10, 20]
# }

# # Crear e entrenar el modelo de bosque aleatorio con búsqueda de hiperparámetros
# modelo = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# # Obtener el mejor modelo
# mejor_modelo = grid_search.best_estimator_

# # Evaluar el mejor modelo en el conjunto de prueba
# y_pred = mejor_modelo.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Error cuadrático medio del mejor modelo: {mse}')

# # Crear nuevos datos de entrada
# nuevos_datos = pd.DataFrame({
#     'ingreso': [5000],
#     'gastos_fijos': [2000],
#     'gastos_variables': [1000],
#     'categoria_hogar': [1],  # Cambiar 'categoria' por 'categoria_hogar' para coincidir con la codificación one-hot
#     'mes': [3]
# })

# # Realizar la predicción con el mejor modelo
# prediccion = mejor_modelo.predict(nuevos_datos)
# print(f'Predicción de saldo final con el mejor modelo: {prediccion[0]}')


import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import parallel_backend  # Importar joblib directamente

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
num_filas = 1000  # Número de filas de datos a generar

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

# Crear escenarios hipotéticos
esc = [
    [6000, 2500, 1200, 'hogar', 3, 2300],
    [8000, 3000, 1500, 'entretenimiento', 6, 3500],
    [4000, 1800, 800, 'transporte', 9, 1400],
    [10000, 2000, 2500, 'comida', 12, 5500],
    [3000, 1200, 600, 'otros', 7, 1200]
]

# Agregar escenarios hipotéticos al DataFrame
datos_sinteticos = pd.concat([datos_sinteticos, pd.DataFrame(esc, columns=columnas)], ignore_index=True)

# Preprocesamiento de datos
X = datos_sinteticos[['ingreso', 'gastos_fijos', 'gastos_variables', 'categoria', 'mes']]
y = datos_sinteticos['saldo_final']
X = pd.get_dummies(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la grilla de hiperparámetros a explorar
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árboles en el bosque
    'max_depth': [None, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10, 20]
}

# Crear y entrenar el modelo de bosque aleatorio con búsqueda de hiperparámetros
modelo = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)  # Añadir n_jobs=-1 para paralelizar el ajuste
with parallel_backend('threading'):  # Asegurar la paralelización con joblib
    grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
mejor_modelo = grid_search.best_estimator_

# Evaluar el mejor modelo en el conjunto de prueba
y_pred = mejor_modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio del mejor modelo: {mse}')

# Crear datos de entrada para realizar una predicción
nuevos_datos = pd.DataFrame({
    'ingreso': [5000],
    'gastos_fijos': [2000],
    'gastos_variables': [1000],
    'categoria': ['hogar'],  # Mantener 'categoria' en lugar de 'categoria_hogar'
    'mes': [8]
})

# Preprocesamiento de datos para los nuevos datos
nuevos_datos = pd.get_dummies(nuevos_datos, columns=['categoria'])

# Asegurarse de que todas las categorías del conjunto de entrenamiento estén presentes en los nuevos datos
for categoria_entrenamiento in X_train.columns:
    if categoria_entrenamiento not in nuevos_datos.columns:
        nuevos_datos[categoria_entrenamiento] = 0

# Reordenar las columnas en el mismo orden que se utilizó durante el entrenamiento
nuevos_datos = nuevos_datos[X_train.columns]

# Realizar la predicción con el mejor modelo
prediccion = mejor_modelo.predict(nuevos_datos)
print(f'Predicción de saldo final con el mejor modelo: {prediccion[0]}')


