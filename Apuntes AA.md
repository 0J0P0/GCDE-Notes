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
      - [Gini](#gini)
    - [Random forest](#random-forest)


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

$$L(y, \hat{y})$$

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

```
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
3. Seleccionar un nuevo centroide de acuerdo a la probabilidad de que sea seleccionado proporcional a la distancia al centroide más cercano.
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

**Bayes classifier**: Clasificador optimo que minimiza el error esperado de clasificacion. Bajo la 0-1 loss, el clasificador bayesiano es el que asigna la clase más probable a cada observación.

Minimizar la perdida esperada prediciendo la clase para la cual se minimiza la perdida esperada.

$$
\hat{y} = \argmin_y E[L(y, y')] = \argmin_y \sum_{y'} L(y, y') p(y' | x) = \argmax_y p(y | x)
$$

### Discriminative vs Generative

**Discriminative**: Modela la distribucion condicional $p(y | x)$. Aprende la frontera de decision directamente en base a la minimizacion de la perdida esperada.

**Generative**: Modela la distribucion conjunta $p(x, y)$. Aprende la distribucion de los datos y la frontera de decision se obtiene a partir de la distribucion de los datos.

- El umbral de decision se puede obtener a partir de la distribucion de los datos.
- Complicado cuando no se conoce la distribucion de los datos o hay muy pocos datos.

#### LDA/QDA

- Modelo generativo.
- Resultado de implementar una clasificacion bayesiana asumiendo que las distribuciones de las clases son gaussianas.
- Minimizar la variabilidad intra-clase y maximizar la variabilidad inter-clase. Maximizar la media de las distribuciones de las clases y minimizar la varianza de las distribuciones de las clases.

$$
p(x | y = k) = \mathcal{N}(x | \mu_k, \Sigma)
$$

Si se asume que las distribuciones a priori son $p(y = k) = \pi_k$, entonces la funcion discriminante es:

$$
g_k(x) = \log \pi_k - \frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x - \mu_k) + C
$$

El primer termino es el logaritmo de la probabilidad a priori de la clase $k$.

El segundo termino es la distancia de Mahalanobis entre el punto $x$ y el centroide de la clase $k$. La distancia de Mahalanobis es una medida de distancia que tiene en cuenta la correlacion entre las variables.

- La frontera de decision es el conjunto de puntos $x$ tales que $g_k(x) = g_j(x)$.

- La funcion $g_k$ es la funcion cuadratica discriminiante (QDA) y el clasificador correspondiente es implemnetado prediciendo la clase $k$ para la cual $g_k(x)$ es maxima.
- Que corresponde a escger la clase con una probibilidad a posteriori maxima.

- Fronteras de desicion cuadraticas, porque la frontera de decision es un paraboloide.

Asumiendo que la matriz de covarianza es la misma $\Sigma$, la funcion discriminante se simplifica a:

$$
g_k(x) = \log \pi_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + x^T \Sigma^{-1} \mu_k
$$

- La frontera de decision es el conjunto de puntos $x$ tales que $g_k(x) = g_j(x)$.
- Metodo lineal porque la frontera de decision es un hiperplano. Las fronteras son lineales.

#### Naive Bayes

- Modelo generativo.
- Asumiendo que las variables son independientes, la distribucion conjunta se puede escribir como:

$$
p(x, y) = p(y) \prod_{j=1}^d p(x_j | y)
$$

En general no es cierto pero puede ser una buena aproximacion para muchos casos.

#### Perceptron

- Modelo discriminativo.
- No probabilistico para datos linealmente separables.

$$
y = \text{sign}(\sum_{j=1}^d w_j x_j) = \text{sign}(w^T x) = \begin{cases}
1 & \text{si } w^T x > 0 \\
-1 & \text{si } w^T x \leq 0
\end{cases}
$$

```
w = 0
while not converged:
    for i in range(N):
        # predicts positive class if w^T x_i > 0

        # else, on a mistake:
        if y_i * w^T x_i <= 0:
            w = w + y_i * x_i
            # update w on positive or negative mistake (depending on y_i)
```

La actualizacion de $w$ en ambos casos se hacerca en 1 a la solucion optima.

$$
w_{t+1}^T x_i = w_t^T x_i + y_i x_i^T x_i = w_t^T x_i + y_i ||x_i||^2 = w_t^T x_i + y_i
$$

al haber escalado para tener norma euclidiana 1.

- Un amejor alternativa es usar la funcion *sigmoid* (mapea a [0,1] para que el resultado pueda ser interpretado como probabilidad) o *logistic*, que es una version suavizada de la funcion signo.

#### Logistic Regression

- Propiedad simetrica: $\sigma(-x) = 1 - \sigma(x)$.
- Diferenciable
- Probabilistico
- Modelo discriminativo
- Su inverso es la funcion *logit*.

## 05

### K-Nearest Neighbors

Se usa la vecinada local para estimar la probabilidad de que una observación pertenezca a una clase.

Para un nueva¡o ejemplo $x$
1. Calcular la distancia/similitud con todos los ejemplos del conjunto de entrenamiento.
2. Seleccionar los $k$ ejemplos más cercanos.
3. Emititr una prediccion con la combinacion de las clases de los $k$ ejemplos más cercanos.

- Predicciones lentas sobretodo si tenemos un dataset grande.
- Valores de k muy bajor pueden llevar a overfitting.
- Valores de k muy altos pueden llevar a underfitting.
- Maldicion de dimensionalidad. A medida que aumenta la dimensionalidad, la distancia entre los puntos se vuelve cada vez más similar.
- Estandarizar los datos es importante.

Cuando $k=1$, las regiones de decision corresponden a la union de celulas de voronoi.

## 06 

**Metodos de ensamble**: Combinar varios modelos para obtener un modelo más robusto.

Cuando los modelos base son independientes, el error del modelo ensamblado es menor que el error de los modelos base.

### Regression tree

Particiona el espacio de variables de entrada en regiones rectangulares. Las predicciones son constantes en cada region, pueden ser calculadas como la media de los valores de entrenamiento en cada region.

- Cada nodo interno es una pregunta sobre una caracteristica.
- Cada rama es una respuesta a la pregunta.
- Cada hoja es una predicción.
- Cada hoja son predicciones constantes.

Problema NP-completo. Se usa greedy search para encontrar una solución aproximada.

#### Gini

$$
G = \sum_{k=1}^K \hat{p}_{k} (1 - \hat{p}_{k})
$$

Donde $\hat{p}_{k}$ es la fracción de observaciones de la clase $k$.

- Entre mas pura la distribucion, menor el indice de gini. Pura hace referencia a que la distribucion es 0 o 1.
- Entre mas uniforme la distribucion, mayor el indice de gini.


### Random forest

Para reducir la variancia de un estimador, se puede entrenar varios estimadores y promediar sus predicciones.

**Bagging**: Entrenar varios modelos base con diferentes subconjuntos de datos de entrenamiento. Promediar las predicciones de los modelos base.
- Hacer la predicciones mas robustas y mas precisas.

Los arboles de decision son muy sensibles a los datos de entrenamiento. Pequeños cambios en los datos de entrenamiento pueden llevar a arboles muy diferentes. Por lo tanto sufren de alta varianza. Candidatos perfectos para bagging.


- Ideal para cuando los modelos base tienen alta varianza

**Boosting**: Entrenar varios modelos base con diferentes subconjuntos de datos de entrenamiento. 
- Las repericiones estan permitidas.
- Las muestras que no han sido escogidas son puestas en un set de validacion OOB.
- El error OOB es una estimacion del error de generalizacion. No hay necesidad de realizar validacion cruzada.

**Ventajas**:
- No requiere validacion cruzada.
- No requiere estandarizacion de los datos.
- Facil paralelizacion.
- Metodos para problemas de clasificacion desbalanceados.

**Desventajas**:
- Muchos trees pueden dificultar la interpretacion.
