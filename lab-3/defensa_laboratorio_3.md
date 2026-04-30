# Defensa del Laboratorio 3: Regresión

Este documento está diseñado para ayudarte a entender la lógica implementada en cada uno de los notebooks del laboratorio y prepararte para cualquier pregunta técnica que te puedan hacer durante la defensa.

## Consideraciones Generales y Ejecución

*   **¿Cómo ejecutar?:** Todos los ejercicios fueron desarrollados en **Jupyter Notebook**. Para ejecutarlos, asegúrate de tener tu entorno virtual activo (`source .venv/bin/activate` en la terminal) con las librerías `pandas`, `numpy`, `scikit-learn` y `matplotlib` instaladas. Simplemente abre cada archivo `.ipynb` desde VSCode o ejecutando `jupyter notebook` en la terminal y corre las celdas en orden (Shift + Enter).
*   **Librería Principal:** Todo el modelado predictivo se apoya en `scikit-learn`, usando principalmente la clase `LinearRegression`.

---

## 1. Predicción de Ventas (`prediccionVentas.ipynb`)

### ¿Qué hace el código?
El objetivo es predecir las ventas para el año 2026 basándose en el registro histórico anual de ventas.

### Lógica y conceptos para la defensa:
*   **Modelo Utilizado:** **Regresión Lineal Simple.** Es "simple" porque tenemos una sola variable independiente (el Año) y una variable dependiente (las Ventas).
*   **Extracción de características:** Separamos el DataFrame en `X` (Año) y `y` (Ventas). Es importante recordar que Scikit-Learn espera que `X` sea una matriz bidimensional (un array de 2D, por eso se extrae como `df[['Año']]`).
*   **Entrenamiento:** La función `modelo.fit(X, y)` encuentra matemáticamente la mejor línea recta que minimiza los errores cuadrados (diferencia entre los puntos reales y la línea).
*   **Coeficiente e Intercepto:** 
    *   `modelo.coef_`: Es la **pendiente** de la recta. Nos dice cuánto aumentan las ventas en promedio por cada año que pasa.
    *   `modelo.intercept_`: Es el **intercepto**, es decir, el valor teórico que tendrían las ventas si el año fuera 0 (el punto donde la recta cruza el eje Y).
*   **Predicción:** Se genera creando un array con el valor `[[2026]]` y pasándolo a `modelo.predict()`.

---

## 2. Predicción de Función Polinómica (`prediccionPolinomio.ipynb`)

### ¿Qué hace el código?
Ajusta un modelo a un conjunto de puntos que claramente no siguen un comportamiento de línea recta, y predice el valor para puntos específicos ($X = 0, 1.5, 3, 5$).

### Lógica y conceptos para la defensa:
*   **El Problema Lineal:** Una regresión lineal clásica (`y = mx + b`) daría resultados terribles aquí porque los puntos forman curvas.
*   **Modelo Utilizado:** **Regresión Polinómica (Grado 3).** Aunque se llama polinómica, técnicamente es un caso especial de la Regresión Lineal Múltiple. 
*   **¿Cómo funciona internamente?:** Usamos `PolynomialFeatures(degree=3)`. Esta herramienta toma nuestra única característica $X$ y crea nuevas características sintéticas sumándole exponentes: $X^1, X^2, X^3$. 
*   **Entrenamiento:** Luego de hacer esa transformación geométrica de los datos (`fit_transform(X)`), le pasamos esas nuevas características al mismo modelo clásico `LinearRegression`. El modelo ahora no dibuja una línea, sino que calcula los pesos para la ecuación: $y = w_0 + w_1x + w_2x^2 + w_3x^3$.
*   **Cuidado con el Grado:** Si el profesor te pregunta por qué elegiste Grado 3: Observando la dispersión inicial de los datos, presentan múltiples curvaturas. Un grado 2 (parábola) no es suficiente, y grados muy altos (como 15 o 20) generarían **sobreajuste (Overfitting)**, donde la curva pasa exactamente por los puntos pero falla miserablemente en predecir datos nuevos.

---

## 3. Predicción Precio del Bitcoin (`prediccionPrecioBitcoin.ipynb`)

### ¿Qué hace el código?
Toma un histórico de precios diarios de Bitcoin y ajusta un modelo para predecir precios en los días 23, 24 y 25 de abril del 2026. Calcula también un porcentaje de desviación contra el "precio real".

### Lógica y conceptos para la defensa:
*   **Transformación de la Fecha:** Las fechas (como strings o tipo Date) no pueden introducirse directamente a un modelo matemático. La solución aplicada fue convertir la fecha a un número continuo (**Ordinal**), que representa un conteo ininterrumpido de días desde el año 1. De esta forma, el modelo ve el tiempo fluyendo como una secuencia de números enteros y puede trazar una tendencia.
*   **Limpieza de Datos:** Hubo que transformar la columna `Price` eliminando las comas separadoras de miles (ej: "78,185.1" a "78185.1") para que Python pudiera tratarlos numéricamente como flotantes.
*   **Porcentaje de Desviación (%dv):** Implementamos estrictamente la fórmula provista en el PDF: `| valor_real - valor_predicho | / valor_real * 100`. Esto mide el "error" relativo de nuestra predicción frente a la realidad.
*   **Importante para la entrega:** En el dataset `datosBitcoin.csv`, el último día registrado es el 22/04/2026. Como se te pide calcular el %dv para los días 23, 24 y 25, **tienes que ingresar manualmente en el código los valores de `precios_reales`** buscando en internet cómo cerró el Bitcoin esos días, para que las matemáticas den de forma correcta. En el código los dejé con unos valores aproximados para que pudiera compilar la gráfica, pero asegúrate de colocar los valores justos en la variable `precios_reales` antes de empaquetar el .tar.gz para la entrega.
