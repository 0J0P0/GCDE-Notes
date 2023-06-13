# Apuntes AA

- [Apuntes AA](#apuntes-aa)
  - [01](#01)
    - [Learning](#learning)
      - [Supervised Learning](#supervised-learning)
      - [Unsupervised Learning](#unsupervised-learning)
    - [Data](#data)
    - [Models](#models)
      - [Modelo como una función](#modelo-como-una-función)
      - [Modelo como distribución de probabilidad](#modelo-como-distribución-de-probabilidad)
    - [Data preprocessing](#data-preprocessing)
    - [Complexity](#complexity)
      - [Complexity control](#complexity-control)
    - [Evaluation](#evaluation)
      - [Loss function](#loss-function)
      - [True Error](#true-error)
      - [Empirical Error](#empirical-error)
      - [Regularization Error](#regularization-error)
      - [Validation Set](#validation-set)
      - [Resampling Methods](#resampling-methods)
  - [02](#02)
    - [Optimization view](#optimization-view)
      - [Least Squares univariate linear regression](#least-squares-univariate-linear-regression)
      - [SVD (Singular Value Decomposition)](#svd-singular-value-decomposition)
    - [Probabilistic view](#probabilistic-view)
      - [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
      - [Bias-Variance decomposition](#bias-variance-decomposition)
    - [Regularization by Maximum Posterior Estimation](#regularization-by-maximum-posterior-estimation)
    - [Hyperparameters](#hyperparameters)
      - [Hyperparameter tuning](#hyperparameter-tuning)
        - [Cross-validation](#cross-validation)
        - [Leave-one-out cross-validation](#leave-one-out-cross-validation)
    - [Ridge Regression](#ridge-regression)
    - [Lasso Regression](#lasso-regression)
  - [03](#03)
    - [Clustering](#clustering)
    - [K-means](#k-means)
    - [K-means++](#k-means-1)
    - [Indice Calisnki-Harabasz](#indice-calisnki-harabasz)
    - [Gaussian Mixture Models](#gaussian-mixture-models)
      - [Expectation-Maximization](#expectation-maximization)
  - [04](#04)
    - [Discriminative vs Generative](#discriminative-vs-generative)
      - [LDA/QDA](#ldaqda)
      - [Naive Bayes](#naive-bayes)
      - [Perceptron](#perceptron)
      - [Logistic Regression](#logistic-regression)
  - [05](#05)
    - [K-Nearest Neighbors](#k-nearest-neighbors)
  - [06](#06)
    - [Regression tree](#regression-tree)
    - [Classification tree](#classification-tree)
      - [Gini](#gini)
    - [Random forest](#random-forest)
  - [07](#07)
    - [Boosting](#boosting)
      - [AdaBoost classifier](#adaboost-classifier)
      - [Additive Boosting](#additive-boosting)
      - [Gradient boosting](#gradient-boosting)


## 01

### Learning
Un sistema _(vivo o no)_ aprende si usa experiencia pasada para mejorar el rendimiento del futuro.

- experiencia pasada = data.

- mejorar el rendimiento del futuro = mejorar predicciones.

Proceso de ecnontrar y ajsutar buenos modelos que expliquen los datos finitos observados y tambien puedan predecir nuevos datos.

#### Supervised Learning
Usando data etiquetada para aprender a predecir etiquetas desconocidas.

- Regresión: predicción de valores continuos reales.
- Clasificación: predicción de valores discretos (categorías).

#### Unsupervised Learning
No son necesarias etiquetas para aprender.

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
Si se asume que los datos vienen de un proceso estóstico, puede ser util permitir al modelo representar/quantificar la incertidumbre en sus predicciones.


### Data preprocessing

Cada problema requiere un enfoque diferente en lo que respecta al preprocesamiento de los datos. Tiene un gran impacto en el rendimiento del modelo.

### Complexity

Como se puede restringir la complejidad del modelo:
- Regularización: penalizar modelos complejos.
- Aportar más datos.
- Reducir el espacio de hipótesis. (Ej: reducir el número de atributos, o restricciones en la forma de la función)

#### Complexity control

Se debe aplicar un control de "fitting" para evitar que el modelo no se ajuste demasiado a los datos de entrenamiento. Esto asegura que el modelo generalice bien a nuevos datos.

$$\text{true error} \leq \text{training error} + \text{complexity(f)}$$

### Evaluation

#### Loss function

Función que mide la diferencia entre la predicción y el valor real. Se usa para evaluar el rendimiento del modelo. Mide que tan "lejos" está la predicción del valor real.

$$L(y, \hat{y})$$

#### True Error

Error que se comete al usar el modelo en nuevos datos. (Generalization error, expected error)

$$\text{true error} = \mathbb{E}_{x,y} [L(y, f(x))]$$

No se puede calcular, porque  se modela sobre un conjunto de datos finito. Pero se puede aproximar usando el error de entrenamiento.

#### Empirical Error

Asumiendo que los datos son independientes e idénticamente distribuidos (i.i.d.):

$$\text{true error} \approx \text{empirical error} = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))$$

**Un error de training bajo no implica un error de generalización bajo.**

Minimizar excessivamente el error de entrenamiento puede llevar a un modelo que no generaliza bien. (Overfitting)

La manera natural de arreglar el error de entrenamiento es reducir la complejidad del modelo. Pero esto puede llevar a un error de generalización alto.

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

$$\hat{y} = \theta_0 + \theta_1 x$$

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

$$\hat{y} = \theta_0 + \theta_1 x + \epsilon$$

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

$$P(\theta | X) = \frac{P(X | \theta) P(\theta)}{P(X)}$$

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

- La funcion de regresion lineal es el mejor predictor posible, en el sentido que conseguiria sesgo cero y varianza minima.

### Lasso Regression

Minimize the sum of the squared error and the L1 norm of the parameters.

$$\hat{\theta}_{Lasso} = \arg \min_{\theta} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\theta_j|$$

Surge de asumir una distribución a priori Laplaciana para los parámetros.


## 03

### Clustering

Particionar la muestra de datos en grupos de modo que las observaciones del mismo cluster tienda a ser mas similar que las observaciones en diferentes clusters.

- Los elementos estan mas relacionados a elementos cercanos que a elementos lejanos.

- El numero de veces que se puede particionar un conjunto de N elementos en K grupos es $S_{N,K} = \frac{1}{K!} \sum_{k=1}^K (-1)^{k} {K \choose k} (K-k)^N$

### K-means

- Cada cluster esta representado por un centroide. $\mu_k$ es el centroide del cluster $k$.
- Cada observación pertenece a un cluster. $r_{nk} = 1$ si la observación $n$ pertenece al cluster $k$.

Se espera minimizar la distancia entre las observaciones y los centroides de sus clusters.

$$
\min_{\mu, r} \sum_{n=1}^N \sum_{k=1}^K r_{nk} ||x_n - \mu_k||^2
$$

Es un problema NP. Alternativa propuesta solo encuentra un minimo local.

- Para centroides fijos, la solución es asignar cada observación al centroide más cercano.

$$
r_{nk} = \begin{cases}
1 & \text{if } k = \arg \min_j ||x_n - \mu_j||^2 \\
0 & \text{otherwise}
\end{cases}
$$

- Para asignaciones fijas, la solución es mover cada centroide al centroide de las observaciones asignadas a él. (A partir de la derivada de la funcion de error igualada a 0)

$$
\mu_k = \frac{\sum_{n=1}^N r_{nk} x_n}{\sum_{n=1}^N r_{nk}}
$$

```python
# N = numero de observaciones
# K = numero de clusters
mu = random(K)
while not converged:
    for n in range(N):
        for k in range(K):
            r[n,k] = 1 if k == argmin_j ||x_n - mu_j||^2 else 0
    for k in range(K):
        mu[k] = sum_n r[n,k] * x_n / sum_n r[n,k]
```
**Ventajas:**
- Implementación simple
- Escalable
- Converge a un mínimo local
- Rapido, incluso para grandes conjuntos de datos
- Fácil de interpretar

**Desventajas:**
- El número de clusters debe ser especificado
- No funciona bien con clusters de diferentes tamaños y densidades
- No funciona bien con datos de alta dimensionalidad
- No funciona bien con datos con outliers
- Asignacion de clusters fija
- Convergencia a un mínimo local
- Sensible a la inicialización de los centroides

### K-means++

Heuristica para inicializar los centroides.

1. Seleccionar un centroide aleatorio de los datos.
2. Para cada observación, calcular la distancia al centroide más cercano.
3. Seleccionar un nuevo centroide de acuerdo a la probabilidad de que sea seleccionado proporcional a la distancia del centroide más cercano.
4. Repetir 2 y 3 hasta que se tengan K centroides.
5. Ejecutar K-means con los centroides iniciales.


### Indice Calisnki-Harabasz

$$
CH = \frac{N-K}{K-1} \frac{\sum_{k=1}^K n_k ||\mu_k - \mu||^2}{\sum_{k=1}^K \sum_{n=1}^{n_k} ||x_n - \mu_k||^2}
$$

Mide el ratio entre la distancia de separacion entre centros de clusters (suma de las distancia de los centroides para la media total) y la compactabilidad de los clusters (suma de las distancias de los puntos a sus centroides en el denominador).

Cantidades normalizadas por $K-1$ y $N-K$ para evitar que el indice crezca con el numero de clusters.

### Gaussian Mixture Models

Modelo probabilistico que asume que los datos provienen de una mezcla de distribuciones gaussianas.

Construida usando una suma convexa de suma de gaussianas.

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

- Cada $\mathcal{N}$ es una componente del modelo.
- $\pi_k$ es la probabilidad de que una observación pertenezca a la componente $k$. Coeficientes de mezcla.

#### Expectation-Maximization

Algoritmo para encontrar los parámetros del modelo.

1. Inicializar los parámetros del modelo. $\mu_k, \Sigma_k, \pi_k$
2. **Expectation**: Calcular la probabilidad de que cada observación pertenezca a cada componente del modelo. $p(z_{nk} = 1 | x_n, \theta)$.
   1. $\gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}$
3. **Maximization**: Actualizar los parámetros del modelo mediante los estimadores ML.
   1. $\mu_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n$
   2. $\Sigma_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k)(x_n - \mu_k)^T$
   3. $\pi_k = \frac{N_k}{N}$

Inicialización de los parámetros con kmeans++

1. $\mu_k = \text{centroides de kmeans++}$
2. $\Sigma_k = \text{matriz de covarianza de cada cluster}$
3. $\pi_k = \frac{N_k}{N}$

## 04

**Regiones de decision**: Las regiones de decision son las regiones del espacio donde los puntos tienen la misma clase.

**Fronteras de decision**: Son los puntos en la frontera entre regiones de decision.

**Clasificadores lineales**: Tienen fronteras de decision representadas por hiperplanos de dimension $d-1$.

- Se busca minimizar el "error esperado" de clasificacion de manera similar a la regresion lineal.

Minimizar la perdida esperada prediciendo la clase para la cual se minimiza la perdida esperada.

$$
\hat{y} = \argmin_y E[L(y, y')] = \argmin_y \sum_{y'} L(y, y') p(y' | x) = \argmax_y p(y | x)
$$

**Bayes classifier**: Clasificador optimo que minimiza el error esperado de clasificacion. Bajo la 0-1 loss, el clasificador bayesiano es el que asigna la clase más probable a cada observación. No es realizable a la práctica, pues no se conoce $p(y,x)$.

- Se intenta aproximar mediante $p(y | x)$ del conjunto de entrenamiento.

- El clasificador de Bayes es el mejor posible, independientemente de la distribucion de los datos, bajo la 0-1 loss.

### Discriminative vs Generative

**Discriminative**: Modela la distribucion condicional $p(y | x)$. Aprende la frontera de decision directamente en base a la minimizacion de la perdida esperada.

- **Ventajas:**
  - No hay necesidad de conocer la distribucion de los datos.

- **Desventajas:**
  - Incapas de generar nuevas muestras.

**Generative**: Modela la distribucion conjunta $p(x, y)$. Aprende la distribucion de los datos y la frontera de decision se obtiene a partir de la distribucion de los datos.

- Complicado cuando no se conoce la distribucion de los datos o hay muy pocos datos.
- **Ventajas:**
  - Capacidad de generar nuevas muestras.  Permite también generar nuevas muestras basadas en el modelo ya que se conoce la distribucion de los datos.
  - Capturan la estructura de los datos al modelar la distribucion conjunta.

- **Desventajas:**
  - Mayor complejidad computacional.
  - Mayor dependencia de los datos de entrenamiento. Se requiere un conjunto de entrenamiento más grande para obtener buenos resultados.

#### LDA/QDA

- Modelo generativo.
- Resultado de implementar una clasificacion bayesiana asumiendo que las distribuciones condicionadas de las clases son gaussianas.

$$
p(x | y = k) = \mathcal{N}(x | \mu_k, \Sigma)
$$

- Minimizar la variabilidad intra-clase y maximizar la variabilidad inter-clase. Maximizar la distancia entre medias de las distribuciones de las clases y minimizar la varianza de las distribuciones de las clases.


Si se asume que las distribuciones a priori son $p(y = k) = \pi_k$, entonces la funcion discriminante es:

$$
g_k(x) = \log (p(y=k)p(x | y=k)) = \log \pi_k - \frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x - \mu_k) - \frac{1}{2} \log |\Sigma| + \text{C}
$$

El primer termino es el logaritmo de la probabilidad a priori de la clase $k$.

El segundo termino es la distancia de Mahalanobis entre el punto $x$ y el centroide de la clase $k$. La distancia de Mahalanobis es una medida de distancia que tiene en cuenta la correlacion entre las variables.

- La frontera de decision es el conjunto de puntos $x$ tales que $g_k(x) = g_j(x)$.

- La funcion $g_k$ es la funcion cuadratica discriminiante (QDA) y el clasificador correspondiente es implemnetado prediciendo la clase $k$ para la cual $g_k(x)$ es maxima.
- Que corresponde a escoger la clase con una probibilidad a posteriori maxima.

- Fronteras de desicion cuadraticas, porque la frontera de decision es un paraboloide.

Asumiendo que la matriz de covarianza es la misma $\Sigma$, la funcion discriminante se simplifica a:

$$
g_k(x) = \log \pi_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + x^T \Sigma^{-1} \mu_k
$$

- La frontera de decision es el conjunto de puntos $x$ tales que $g_k(x) = g_j(x)$.
- Metodo lineal porque la frontera de decision es un hiperplano. Las fronteras son lineales.

En ambos casos es fundamental escoger la forma de la matriz de covarianza (diagonal, isotrópica, etc). En el caso de LDA, si la matriz de covarianza es diagonal y con varianzas iguales, entonces se usa una distáncia euclidea ponderada. 

Se puede usar regularización (RDA) añadiendo continuidad entre la matriz de covarianza conjunta (LDA) y la matriz de covarianza de cada clase (QDA). También evita problemas de singularidad, en posibles casos de datos escasos.

$$\hat\Sigma_k(\alpha)=\alpha\hat\Sigma_k+(1-\alpha)\hat\Sigma$$

- Para conocer los distintos parametros de la distribucion conjunta, se puede usar el metodo de maxima verosimilitud.

$$
\hat{\pi}_k = \frac{n_k}{n} \quad \hat{\mu}_k = \frac{1}{n_k} \sum_{i:y_i = k} x_i \quad \hat{\Sigma}_k = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu}_{y_i})(x_i - \hat{\mu}_{y_i})^T
$$

#### Naive Bayes

- Modelo generativo.
- Asumiendo que las variables son independientes, la distribucion conjunta se puede escribir como:

$$
p(x, y) = p(y) \prod_{j=1}^d p(x_j | y)
$$

En general no es cierto pero puede ser una buena aproximacion para muchos casos. Los parámetros se estiman con la frecuencia de cada clase y categoria en los datos. 

Nuevamente se escoge la clase que maximiza $g_k(x)$.

$$
g_k(x) = \log \pi_k + \sum_{j=1}^d \log p(x_j | y=k)
$$

Puede ser necesario usar Laplace smoothing cuando tenemos sparse data con categorias con frecuencia 0 en la muestra.

- Si se asume que asume que todas las variables siguen una distribucion gaussiana, entonces Gaussian Naive Bayes es equivalente a QDA con matrices de covarianza diagonal.

#### Perceptron

- Modelo discriminativo.
- No probabilistico para datos linealmente separables.
- Algoritmo online. Actualiza los pesos en cada iteracion. Recibe una observacion de training a la vez y actualiza los pesos en base a la observacion.

$$
\hat{y} = \text{sign}(\sum_{j=1}^d w_j x_j) = \text{sign}(w^T x) = \begin{cases}
1 & \text{si } w^T x > 0 \\
-1 & \text{si } w^T x \leq 0
\end{cases}
$$

```
w = 0
while not converged:
    for i in range(N):
        # predicts positive class if w^T x_i > 0
        # predicts negative class if w^T x_i <= 0

        # else, on a mistake (y_i != y_hat_i)
          w = w + y_i * x_i
          # update w on positive or negative mistake (depending on y_i)
```

La actualizacion de $w$ en ambos casos se acerca en 1 a la solucion optima.

$$
w_{t+1}^T x_i = w_t^T x_i + y_i x_i^T x_i = w_t^T x_i + y_i ||x_i||^2 = w_t^T x_i + y_i
$$

al haber escalado para tener norma euclidiana 1.

El numero de errores que comete Perceptron es acotado por:

$$
\frac{1}{\gamma^2}, \ \gamma = \min_{x} w^T x
$$

- $\gamma$ es la distancia del hiperplano a la clase más cercana. Similar a la distancia de Margen en SVM. Aunque en SVM se maximiza el margen de separación.

- **Ventajas:**
  - Simplicidad y eficiencia computacional. Especialmente para datos linealmente separables.
  - Interpretacion intuitiva de los pesos. Explica la importancia de cada variable.

- **Desventajas:**
  - No probabilistico. No se puede interpretar la salida como una probabilidad.
  - No es robusto a outliers.
  - No es robusto a datos no linealmente separables.
  - No es robusto a datos con clases desbalanceadas.


Una mejor alternativa es usar la funcion *sigmoid* (mapea a $[0,1]$ para que el resultado pueda ser interpretado como probabilidad) o *logistic*, que es una version suavizada de la funcion signo.

#### Logistic Regression

- Propiedad simetrica: $\sigma(-x) = 1 - \sigma(x)$.
- Diferenciable
- Probabilistico
- Modelo discriminativo
- Su inverso es la funcion *logit*.

## 05

### K-Nearest Neighbors

Se usa la vecindad local para estimar la probabilidad de que una observación pertenezca a una clase.

Para un nuevo ejemplo $x$
1. Calcular la distancia/similitud con todos los ejemplos del conjunto de entrenamiento.
2. Seleccionar los $k$ ejemplos más cercanos.
   1. Segun la mayoria de votos. (Classification)
   2. Voto basado en la distancia. (Ponderar por el inverso de la distancia o el inverso del cuadrado de la distancia) (Classification)
   3. Media de los valores de los $k$ ejemplos más cercanos. (Regression)
   4. Media ponderada por la distancia de los $k$ ejemplos más cercanos. (Regression)
3. Emititr una prediccion con la combinacion de las clases de los $k$ ejemplos más cercanos.

- Predicciones lentas sobretodo si tenemos un dataset grande.
- Valores de k muy bajos pueden llevar a overfitting.
  - Sensible a ruido y outliers.
- Valores de k muy altos pueden llevar a underfitting.
- Maldicion de dimensionalidad. A medida que aumenta la dimensionalidad, la distancia entre los puntos se vuelve cada vez más similar.
- Estandarizar los datos es importante.

Cuando $k=1$, las regiones de decision corresponden a la union de celulas de voronoi. Se usa la misma observacion para predecir su clase.

## 06 

**Metodos de ensamble**: Combinar varios modelos para obtener un modelo más robusto.
- Los modelos deben ser independientes.

Cuando los modelos base son independientes, el error del modelo ensamblado es menor que el error de los modelos base.

### Regression tree

Particiona el espacio de variables de entrada en regiones rectangulares. Las predicciones son constantes en cada region, pueden ser calculadas como la media de los valores de entrenamiento en cada region.

### Classification tree

- Cada nodo interno es una pregunta sobre una caracteristica.
- Cada rama es una respuesta a la pregunta.
- Cada hoja son predicciones constantes.

Problema NP-completo. Se usa greedy search para encontrar una solución aproximada.

Como definir el nodo inicial? 
- Medir la pureza (nivel de mezcla de observaciones en diferentes clases). Gini, Entropia, etc.
  - La pureza de un nodo es la suma ponderada de las impurezas de las hojas.
  - La ponderacion es la fraccion de observaciones que pertenecen a cada hoja entre todas las observaciones del nodo.

$$
x, x_0 = \argmin_{x, x_0} \frac{S_{x\leq x_0}}{S} Gini(S_{x\leq x_0}) + \frac{S_{x > x_0}}{S} Gini(S_{x > x_0})
$$

- x es la variable
- $x_0$ es el punto de corte. (En el caso de variables categoricas, $x_0$ es una categoria)

#### Gini

El indice gini es una medida de impureza de una distribucion de probabilidad. 

$$
G = \sum_{k=1}^K \hat{p}_{k} (1 - \hat{p}_{k}) = 1 - \sum_{k=1}^K \hat{p}_{k}^2
$$

Donde $\hat{p}_{k}$ es la fracción de observaciones que pertenecen a la clase $k$ en el nodo.

- Entre mas pura la distribucion, menor el indice de gini.
- Entre mas uniforme la distribucion, mayor el indice de gini.


### Random forest

Para reducir la variancia de un estimador, se puede entrenar varios estimadores y promediar sus predicciones.

**Bagging**: Entrenar varios modelos base con diferentes subconjuntos de datos de entrenamiento. Promediar las predicciones de los modelos base.
- Hacer la predicciones mas robustas y mas precisas.
  - Entre mas modelos base, la varianza se reduce.
- Ideal para cuando los modelos base tienen alta varianza

- Las repericiones estan permitidas.
- Las muestras que no han sido escogidas son puestas en un set de validacion OOB. (Out of bag)
- El error OOB es una estimacion del error de generalizacion. No hay necesidad de realizar validacion cruzada.

Los arboles de decision son muy sensibles a los datos de entrenamiento. Pequeños cambios en los datos de entrenamiento pueden llevar a arboles muy diferentes. Por lo tanto sufren de alta varianza. Candidatos perfectos para bagging. En el caso de Random Forest también se usa un subconjunto diferente de las features en cada split/nodo de los arboles.

Pseudo-codigo:

```python
T_b = []
for b in range(B)
  # B boostrap samples (with replacement)
  X_b, y_b = bootstrap(X, y)

  # Train a decision tree on X_b, y_b. Each node by:
    # Select m variables at random from X_b
    # Pick best variable/split-point to split on (Gini/MSE)
    # Split the node
    # Repeat until the leaves are pure or until the leaves contain a minimum number of training examples
  tree = DecisionTree()
  tree.fit(X_b, y_b)

  T_b.append(tree)
```

**Ventajas**:
- No requiere validacion cruzada (OBB error rate)
- No requiere estandarizacion de los datos.
- Facil paralelizacion.
- Metodos para problemas de clasificacion desbalanceados.

**Desventajas**:
- Muchos trees pueden dificultar la interpretacion.


## 07

### Boosting

Boosting es un meta-algortimo de aprendizaje automatico que reduce el sesgo y la varianza de los estimadores base en un contexto de aprendizaje supervisado.

- Tipo de algoritmo de ensamble.

- Combinar varios clasificadores debiles (base) para obtener un clasificador fuerte. Combinación lineal ponderada de los clasificadores debiles en funcion de la exactitud de sus predicciones.

#### AdaBoost classifier

$$
H(x) = \hat{y} = \text{sign}(\sum_{t=1}^T \alpha_t h_t(x))
$$

- $h_t(x)$ es el clasificador debil (predictor base) de la iteracion $t$. La idea es entrenar el clasificador debil para minimizar el error de una version ponderada del dataset de entrenamiento.
- $\alpha_t$ es el peso del clasificador debil de la iteracion $t$ (ponderacion de la exactitud de sus predicciones).
- $T$ es el numero de iteraciones.

Se busca mejorar el modelo base en cada iteración. En cada iteración se le da mas peso a las observaciones que fueron mal clasificadas en la iteración anterior para corregirla. Se le da menos peso a las observaciones que fueron bien clasificadas en la iteración anterior.

```python
# Inicializar pesos
D_1 = [1/N, ..., 1/N] # N observaciones

for t in range(T):
    # Entrenar clasificador debil
    h_t = train(X, y, D_t)
    
    # Calcular error
    e_t = sum(D_t * (h_t != y)) / sum(D_t)
    
    # Calcular peso
    alpha_t = 1/2 * log((1 - e_t) / e_t)
    
    # Actualizar pesos
    # Z_t es un factor de normalizacion para que los pesos sumen 1 (sean una distribucion)
    D_t+1 = D_t * exp(-alpha_t * y * h_t(X)) / Z_t
```

- Una forma de entrenar el clasificador debil $h_t$ es formar los arboles de decision de cada feature y escoger el que tenga el menor Gini index.
  - Los predictores lineales pueden ser Decision Stumps (arboles de decision con un solo nodo interno), con un orden secuencial.

- Una forma de actualizar el conjunto de entrenamiento es seleccionar el mismo numero total de muestras $N$ del conjunto de entrenamiento original, pero con reemplazo. Las muestras que fueron seleccionadas varias veces tienen mayor peso $D_t$. Nuevamente se asignan pesos uniformes a las muestras.

**Desventajas:**
- Sensible a ruido y outliers
- Trabajo secuencial


#### Additive Boosting

Encontrar una función $F(x)$ que minimice el error cuadratico medio. Tal que $F(x) \approx y$.

- Modelo aditivo

$$
F(x) = \sum_{t=1}^T f_t(x)
$$

- $f_t(x)$ es el clasificador debil (predictor base) de la iteracion $t$.

Los residuos son las diferencias entre las predicciones y los valores reales.

$$
r_t = y - f_{t}(x)
$$

Mejorar el predictor $f_t$ en cada iteración agregando otro predictor y el resultado sea el valor observado del target. Que $f_{t+1}$ se aproxime a $r_t$.

- Sofisticando los predictores base mediante una suma.


```python	
# Inicializar F_0
f_0 = 0
F_t = f_0
for t in range(1, T):
    # Generar algoritmo base nuevo
    # beta_m = coeficiente del modelo base
    (beta_m, gamma_m) = arg_min(sum(L(y, F_t(x) - beta * h(x, gamma))))

    # Actualizar F_t
    F_t = F_t + beta_m * h(x, gamma_m)
```

- Definir $L(.)$ y $h(.)$.
- Típicamente $L(.)$ es el error cuadratico medio y $h(.)$ es un arbol de decision.


#### Gradient boosting

Gradient boosting es una generalizacion de boosting que permite optimizar cualquier funcion de perdida diferenciable.

$$
f_k := f_{k-1} + \gamma_k \nabla L(y, f_{k-1}(x))
$$

$$\nabla L(y, f_{k-1}(x)) = \frac{\partial L(y, f_{k-1}(x))}{\partial f_{k-1}(x)}$$

- Suma de modelos: paralelismo entre gradient descent y boosting para regresion.
- $\gamma_k$ es el learning rate. En este caso sera una exploracion lineal.


En el caso de usar el error cuadratico medio como funcion de perdida, la derivada es:

$$
\nabla L(y, f_{k-1}(x)) = y - f_{k-1}(x)
$$