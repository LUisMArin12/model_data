# model_data
Modelo de Machine Learning

1.	Se importan las bibliotecas necesarias: numpy para operaciones numéricas, pandas para trabajar con datos tabulares y RandomForestRegressor y GridSearchCV de sklearn.ensemble y sklearn.model_selection, respectivamente, para el modelo de Random Forest y la búsqueda de hiperparámetros en cuadrícula.
2.	Se definen algunos datos específicos y se generan datos adicionales aleatorios para crear un conjunto de datos sintéticos. Estos datos incluyen características como ingreso, gastos fijos, gastos variables, categoría y mes.
3.	Se crea un DataFrame de pandas (datos sintéticos) utilizando los datos generados, con columnas etiquetadas según las características.
4.	Se calcula una nueva característica llamada saldo final, que representa la diferencia entre los ingresos y los gastos fijos y variables.
5.	Se realiza el preprocesamiento de datos dividiendo el DataFrame en características (X) y la variable objetivo (y).
6.	Se aplica la codificación one-hot a la característica categórica 'categoria' en las características X.
7.	Se define una grilla de hiperparámetros (param_grid) para el modelo de Random Forest, que incluye opciones para el número de estimadores, la profundidad máxima del árbol, el número mínimo de muestras requeridas para dividir un nodo y el número mínimo de muestras requeridas para estar en un nodo hoja.
8.	Se crea un objeto GridSearchCV que utilizará validación cruzada para buscar la mejor combinación de hiperparámetros dentro de la grilla definida, optimizando la métrica de puntuación de error cuadrático medio negativo.
9.	Se ajusta el objeto grid_search a los datos de entrada (X, y)
10.	Se obtiene el mejor modelo (mejor_modelo) seleccionado por la búsqueda en cuadrícula y se imprimen los mejores hiperparámetros encontrados.
11.	Se definen los valores de las características para un caso específico de predicción (valores_caracteristicas), incluyendo las características codificadas one-hot para la categoría.
12.	Se calcula la importancia de las características del mejor modelo.
13.	Se utiliza el mejor modelo para predecir el saldo final para el caso específico proporcionado (saldo_final_predicho) y se imprime el resultado.










