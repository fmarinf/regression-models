## Análisis de serie temporal en Rust

El código es un modelo de predicción de serie temporal en Rust que utiliza dos modelos de regresión diferentes para predecir los valores futuros de un conjunto de datos. En primer lugar, se carga el archivo CSV que contiene los datos de la serie temporal. Luego, se divide en conjuntos de entrenamiento y prueba.

El primer modelo de regresión que se utiliza es una regresión lineal simple. Se entrena en el conjunto de entrenamiento y se utiliza para predecir los valores futuros en el conjunto de prueba. A continuación, se entrena un segundo modelo de regresión basado en una red neuronal recurrente (RNN). Este modelo se entrena en las secuencias de datos del conjunto de entrenamiento y se utiliza para predecir los valores futuros en el conjunto de prueba.

Finalmente, se utiliza la biblioteca Plotters para crear una visualización de los datos originales y las predicciones de ambos modelos de regresión. Los resultados se muestran en una sola gráfica, lo que permite comparar fácilmente las predicciones de ambos modelos.

En resumen, este código es una implementación de modelos de regresión lineal simple y de red neuronal recurrente para la predicción de serie temporal en Rust, con una visualización de los resultados utilizando la biblioteca Plotters.
