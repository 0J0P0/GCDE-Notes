# Apuntes AA

## 01

### Learning
Un sistema _(vivo o no)_ aprende si usa experiencia pasada para mejorar el rendimiento del futuro.

- experiencia pasada = data.

- mejorar el rendimiento del futuro = mejorar predicciones.

Proceso de ecnontrar y ajsutar buenos modelos que expliquen los datos finitos observados y tambiñen pueda predecir nuevos datos.

#### Supervised Learning
Usando data etiquetada para aprender a predecir etiquetas desconocidas.

- Regresión: predicción de valores continuos reales.
- Clasificación: predicción de valores discretos (categorías).

#### Unsupervised Learning
No so necesarias etiquetas para aprender.

- Clustering: agrupar datos en grupos. Descubrir grupos homogéneos en los datos.
- Dimensionality Reduction: reducir la dimensionalidad de los datos. Descubrir subespacios de los datos que contienen la mayor parte de la información.

### Data

Una muestra aleatoria coleeccionada del problema que queremos modelar, con un conjunto de atributos y sus respuestas correspondientes.

- Las filas son ejemplos y las columnas son los atributos describiendo los ejemplos.
    - Los atributos pueden ser continuos o discretos.

### Models

Descripción de como se generan y comportan los datos de manera general. 

Artefacto que nos permite hacer predicciones sobre nuevos datos y describir la relación entre los atributos y la respuesta. Puede ser entendido como un mecanismo de comprensión de los datos con habilidades de predicción.

#### Modelo como una función
La función mappea los ejempos de entrenamiento a sus respuestas.

$$f: R^d \rightarrow {C_1, ..., C_k}$$
$$f: R^d \rightarrow R$$

#### Modelo como distribución de probabilidad
Si se asume que lso datos vienen de un proceso estóstico, puede ser util permitir al modelo representar/quantificar la incertidumbre en sus predicciones.


### Data preprocessing

Cada problema requiere un enfoque diferente en lo que respecta al preprocesamiento de los datos. Tiene un gran impacto en el rendimiento del modelo.

### Complexity

Como se puede restringir la complejidad del modelo:
- Regularización: penalizar modelos complejos.
- Aportar más datos.
- Reducir el espacio de hipótesis. (Ej: reducir el número de atributos, o restricciones en la forma de la función)

#### Complexity control

Se debe aplicar un control de "fitting" para evitar que el modelo se ajuste demasiado a los datos de entrenamiento. Esto asegura que el modelo generalice bien a nuevos datos.

$$\text{true error} \leq \text{training error} + \text{complexity(f)}$$

### Evaluation

#### Loss function

Función que mide la diferencia entre la predicción y el valor real. Se usa para evaluar el rendimiento del modelo. Mide que tan "lejos" está la predicción del valor real.

$$ L(y, \hat{y})$$

#### True Error

Error que se comete al usar el modelo en nuevos datos. (Generalization error, expected error)

$$\text{true error} = \mathbb{E}_{x,y} [L(y, f(x))]$$

No se puede calcular, porque  se modela sobre un conjunto de datos finito. Pero se puede aproximar usando el error de entrenamiento.

#### Empirical Error

(Training error)

Asumiendo que los datos son independientes e idénticamente distribuidos (i.i.d.):

$$\text{true error} \approx \text{empirical error} = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))$$

**Un error de training bajo no implica un error de generalización bajo.**

Minimizar excessivamente el error de entrenamiento puede llevar a un modelo que no generaliza bien. (Overfitting)

La manera natural de arrercar el error de entrenamiento es reducir la complejidad del modelo. Pero esto puede llevar a un error de generalización alto.

#### Regularization Error

$$\text{regularization error} = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i)) + \lambda |f|$$

#### Validation Set

Dividir los datos en dos conjuntos: training y validation. Usar el conjunto de entrenamiento para entrenar el modelo y el conjunto de validación para evaluar el modelo.

Una mejor estimación de _true error_ es el error de validación. El proceso de _learning_ no tiene acceso a los datos de validación.

El error de generalización se puede estimar mediante el computo del error de validación.

#### Resampling Methods

Diferentes formas de dividir los datos en conjuntos de entrenamiento y validación para mejorar la estimación del error de generalización.


## 02

### Optimization view

Definir una función de error que mida la diferencia entre la predicción y el valor real. Minimizar la función de error para encontrar los parámetros del modelo que mejor se ajustan a los datos.

Usar los valores de los parámetros para predecir nuevos datos.

#### Least Squares univariate linear regression

$$ \hat{y} = \theta_0 + \theta_1 x$$

El metodo de minimos cuadrados es un metodo de optimizacion que minimiza la suma de los cuadrados de las diferencias entre los valores observados y los valores predichos.

Minimiza la función de error (loss), que es la suma de los cuadrados de las diferencias entre los valores observados y los valores predichos.

$$\text{loss}(\theta_1, \theta_2) = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Para minimizar la función de error, se debe calcular la derivada de la función de error con respecto a cada uno de los parámetros y establecerla en cero.

Siempre hay una columna de unos en la matriz de diseño, para que el término de sesgo $\theta_0$ no se anule.

$$\hat{y} = X \theta, \ \theta =[ \theta_0, \theta_1]^T$$

$$\theta = (X^T X)^{-1} X^T y$$

#### SVD (Singular Value Decomposition)

Para invertir la matriz $X^T X$, se puede usar la descomposición en valores singulares (SVD).

$$A = U \Sigma V^T$$

Con lo que se puede calcular $\theta$ como:

$$\theta = V \Sigma^{-1} U^T y$$

### Probabilistic view

#### Maximum Likelihood Estimation

Se asume que los datos vienen de un proceso estadístico. Se puede modelar el proceso como una distribución de probabilidad.

L es una funcion de los parametros, asumiendo que los datos son independientes e identicamente distribuidos (i.i.d.).

$$ \hat{y} = \theta_0 + \theta_1 x + \epsilon$$

$$\epsilon \sim N(0, \sigma^2)$$

$$y_i \sim N(\hat{y}_i \theta, \sigma^2)$$

La verosimilitud del parametro $\theta$ es:

$$L(\theta, \sigma^2) = \prod_{i=1}^n p(y_i | x_i; \theta, \sigma^2)$$

Se busca el valor de $\theta$ que maximiza la verosimilitud.

$$\hat{\theta} = \arg \max_{\theta} \log L(\theta, \sigma^2)$$

Diferenciando la función de verosimilitud con respecto a $\theta$ y estableciendo el resultado en cero, se obtiene la estimación de máxima verosimilitud.

$$\hat{\theta}_{ML} = (X^T X)^{-1} X^T y$$

$$\hat{\sigma}^2_{ML} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^n (y_i - x_i \hat{\theta}_{ML})^2$$

#### Bias-Variance decomposition

Asumiendo que los datos son i.i.d., se puede descomponer el error de generalización en tres componentes:

$$\text{true error} = \mathbb{E}_{x,y} [L(y, f(x))] = \mathbb{E}_{x,y} [(y - \hat{y})^2] = \text{bias}(\hat{y})^2 + \text{variance}(\hat{y}) + \text{irreducible error}$$

- Bias: error de sesgo. Mide que tan lejos está la predicción del valor real.

- Variance: error de varianza. Mide que tan lejos están las predicciones de un modelo entrenado con diferentes conjuntos de datos.

- Irreducible error: error irreducible. No se puede reducir. Mide la cantidad de ruido en los datos.


Modelos con alta varianza tienden a sobreajustar los datos de entrenamiento.
- El modelo es estable si la varianza es baja. (No varía mucho con los datos de entrenamiento)

Modelos con alto sesgo tienden a subajustar los datos de entrenamiento.
- Un modelo es flexible si el sesgo es bajo. (Se ajusta bien a los datos de entrenamiento)


### Regularization by Maximum Posterior Estimation

Posteriors are a way to combine prior knowledge with data. The prior is a distribution over the parameters, and the posterior is a distribution over the parameters given the data.

$$ P(\theta | X) = \frac{P(X | \theta) P(\theta)}{P(X)}$$

La distribución posterior es una combinación de la distribución a priori y la distribución de verosimilitud.

Noción de que tipo de distribución de parámetros es más probable dada la evidencia.

$$\hat{\theta}_{MAP} = \arg \max_{\theta} \log P(\theta | X) = \arg \max_{\theta} \log P(X | \theta) P(\theta)$$

pues

$$P(\theta | X) \propto P(X | \theta) P(\theta)$$

### Hyperparameters

Los hiperparámetros son parámetros que no se pueden aprender directamente de los datos. Se deben especificar antes de entrenar el modelo.

$\lambda$ es un hiperparámetro que controla la complejidad del modelo. Entre más grande sea $\lambda$, más simple será el modelo y menores serán los pesos de los parámetros.

#### Hyperparameter tuning

El proceso de ajustar los hiperparámetros para obtener el mejor rendimiento del modelo.

##### Cross-validation

Dividir los datos en conjuntos de entrenamiento y validación. Entrenar el modelo con los datos de entrenamiento y evaluarlo con los datos de validación.

1. Decidir los valores de los hiperparámetros.
2. Particionar los datos en K conjuntos
3. Repetir K veces:
    1. Entrenar el modelo con K-1 conjuntos de train. Con todos los valores de los hiperparámetros.
    2. Evaluar el modelo con el conjunto restante (k).
    3. Calcular el error de validación.
4. Promediar los errores de validación.


##### Leave-one-out cross-validation

Dividir los datos en conjuntos de entrenamiento y validación. Entrenar el modelo con los datos de entrenamiento y evaluarlo con los datos de validación.

El conjunto de validación es un siempre el mismo.

### Ridge Regression

Asumiendo una distribución a priori Gaussiana, se obtiene la regresión de Ridge.

$$\hat{\theta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y$$

Que minimiza la el cuadrado de la norma L2 de los parámetros y the sum of the squared error.

### Lasso Regression

Minimize the sum of the squared error and the L1 norm of the parameters.

$$\hat{\theta}_{Lasso} = \arg \min_{\theta} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\theta_j|$$

Surge de asumir una distribución a priori Laplaciana para los parámetros.





