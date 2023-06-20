# Introduction to Audio and Image Processing

- [Introduction to Audio and Image Processing](#introduction-to-audio-and-image-processing)
- [1. Statistical Signal Modeling](#1-statistical-signal-modeling)
  - [1.1. Introduction](#11-introduction)
    - [1.1.b Random Variable](#11b-random-variable)
  - [1.2. Modelling of memoryless processes](#12-modelling-of-memoryless-processes)
    - [1.2.a Sample wise operators](#12a-sample-wise-operators)
    - [1.2.b Quantization](#12b-quantization)
    - [1.2.c Video](#12c-video)
  - [1.3 Discrete Stochastic Processes](#13-discrete-stochastic-processes)
- [2. Estimation Theory](#2-estimation-theory)
  - [2.1 Introduction](#21-introduction)
  - [2.2 C-R Bound and Efficient estimators](#22-c-r-bound-and-efficient-estimators)
  - [2.3 ML and MAP Estimation](#23-ml-and-map-estimation)
- [3. Optimal and Adaptative Filtering](#3-optimal-and-adaptative-filtering)
  - [Wiener-Hopf filter](#wiener-hopf-filter)
    - [Minimum MSE prediction](#minimum-mse-prediction)
  - [Linear prediction](#linear-prediction)
    - [Linear prediction coding](#linear-prediction-coding)
    - [Linear prediction coding of speech signals](#linear-prediction-coding-of-speech-signals)
  - [Adaptive filters](#adaptive-filters)
    - [Steepest descent algorithm](#steepest-descent-algorithm)
    - [Convergence analysis](#convergence-analysis)
    - [LMS algorithm](#lms-algorithm)

# 1. Statistical Signal Modeling

## 1.1. Introduction

### 1.1.b Random Variable

**Random variable**: 
- Is a variable whose value is a random phenomenon. It is a function that maps a sample space to a real line.
- An assigment to a variable of the results of an experiment performed multiple times.

**Moments of a random variable**:

- **Mean** (first order moment): $\mu = E[X] = \sum_{x \in \Omega} x P(X=x)$

- **Variance** (second order moment): $\sigma^2 = E[(X-\mu)^2]
- The **variance** measures the dispersion of the random variable around its mean.

$$\sigma_X^2 = E[(X-\mu_X)^2] = E[X^2] - E[X]^2$$

**Processing of random variables**:

$$Y = g(X)$$

- $E[Y] = E[g(X)]$

- $Var[Y] = E[(g(X)-E[g(X)])^2] = E[g(X)^2] - E[g(X)]^2$

$$f_Y(y) = \bigg | \frac{dg(x)}{dx} \bigg |^{-1} f_X(x)$$

For adding two random variables (independent): $Z = Y + X$

$$f_Z(z) = f_Y(z) * f_X(z)$$

**Covariance**:

To compare two random variables to understand their relationship. Covariance measures its joint variability.

$$Cov(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] = E[XY] - E[X]E[Y]$$

- Sign: $Cov(X,Y) > 0$ shows the positive tendency in the linear relationship between $X$ and $Y$.

**Correlation**:

Measures the joint variability of two random variables, regardless of their mean.

$$Corr(X,Y) = E[XY]$$

## 1.2. Modelling of memoryless processes

### 1.2.a Sample wise operators

Memoryless processes assume that every sample of the process is independent of its neighbor samples.

- Only take into account the _sample values_ (sample wise operators). Not their index or neighbors.

- Only take into account the pixel values. Process the same manner all the pixels with the same value.

**Transformation**:

Transformation or _mapping_ of the range of values of the input signal onto the range of values of the output signal. 

**Range transform operators**:

Without taking into account the specificities of the image.

- **Grey level mapping** different segments of the range are expanded or compressed.
    - **Expanded**: the image is brighter. When the derivative of the trasnformation is greater than 1.
    - **Compressed**: the image is darker. When the derivative of the trasnformation is less than 1.

- **Contrast mapping** exapand a range of the input image onto the whole range of the output image.
    - **Clipping**: a set of values of the input image are mapped onto a single value of the output image. Non-reversible since it is not a bijection.

- **Negative mapping**: invert the range of the input image. Do not change contrast.

- **Binarization mapping** map the input image onto a binary image. Using a threshold.

- **Logarithmic mapping**: compress the input image range.

- **Power-$\gamma$ mapping**: powers larger than 1 make the shadows darker, while powers smaller than 1 make dark regions lighter.

$$
s = c r^{\gamma}
$$

- **Pseudo-color mapping**: map the input image onto a color image.

Implementation in pseudocode:

```python
def mapping(input_image, output_image, mapping_function):
    for i in range(input_image.height):
        for j in range(input_image.width):
            output_image[i,j] = mapping_function(input_image[i,j])
```

More efficient implementatio (using a dictionary):

```python
def mapping_table(input_image):
    MT = {}
    for p in range(MAX_PIXEL_VALUE):
        MT[p] = mapping_function(p)
    return MT

def mapping(input_image, output_image, MT):
    for i in range(input_image.height):
        for j in range(input_image.width):
            output_image[i,j] = MT[input_image[i,j]]
```

**Histogram-based operators**:

Adapting the operator to the image statistics.

- **Dark image**: Grey level values concatenated in the lowest range of the histogram.

- **Bright image**: Grey level values concatenated in the highest range of the histogram.

- **Contrast image**: Grey level values distributed in the whole range of the histogram.
    - Low contrast: the histogram is concentrated in a small range of the image.
    - High contrast: the histogram is distributed in the whole range of the image.

Discrete function that stores for each possible image value ($r_k$), the number of ocurrences of that value in the image ($n_k$).

$$h(r_k) = n_k \ \forall k \in \{0,1,...,L-1\}, \  n = \sum_{k=0}^{L-1} n_k$$

- The normalizaed histogram is an estimation of the probability density function of a random variable associated to the image.

```python
def histogram(input_image):
    H = [0] * MAX_PIXEL_VALUE
    for i in range(input_image.height):
        for j in range(input_image.width):
            H[input_image[i,j]] += 1
    return H
```

> A separate historgram for each color component is not recommended, since it does not represent the joint distribution of the color components. It can produce artificial color effects.

**Histogram equalization**:

Images with a low contrast (small dynamic range) can be improved by equalizing the histogram (sretching the histogram to the whole range of the image).

- Aim to produce a flat histogram of the output image $y$ (uniform distribution of the grey levels). So the probability density function of the random variable associated to the image is uniform.

$$f_Y(y) = 1 \implies \bigg | \frac{dg(x)}{dx} \bigg | =  f_X(x) \implies y = g(x) = \int_{-\infty}^x f_X(x) dx$$

Adapted to the continuous case:

$$s = T(r) = \int_{-\infty}^r f_X(x) dx \iff s_k = T(r_k) = \sum_{i=0}^{k-1} p(r_i) = \sum_{i=0}^{k} \frac{n_i}{n}$$

after scaling the values to the range of the output image.

$$t_k = \texttt{round} ( (L-1) s_k )$$

Some colors may be lost in the process. The histogram equalization is not reversible.

### 1.2.b Quantization

Quantization is a sample wise operator. It is a transformation of the input image into a new image with a lower number of grey levels. (Truncation, rounding, storage, etc.)

- Non-reversible transformation.

A quantizer $q(x)$ performs a mapping $q: \mathbb{R} \rightarrow \mathbb{C}$.
- **Classification**: $i = \alpha(x), \ i = {1, ..., N}$
- **Representation**: $y_i = \beta(i), \ \mathbb{C} = \{y_1, ..., y_N\} \subset \mathbb{R}$

Therefor:

$$q(x) = \beta(\alpha(x))$$

**Mid-rise quantizer**:

Levels $y_i$ are centered around the midpoints of the intervals $[\Delta_{i-1}, \Delta_i], \ i = 1, ..., L$. Loss of one level. L odd.

$$\Delta_i = \frac{1}{L} i, \ L = 2^B, \ B = \#bits$$

**Mid-tread quantizer**:

Levels $y_i$ are centered around the midpoints of the intervals $[\Delta_{i-1}, \Delta_i], \ i = 1, ..., L$. Symmetric around 0. Loss of one level. L even. Mid-tread quantizer is best for noice suppression.

$$\Delta_i = \frac{1}{L} i, \ L = 2^B-1, \ B = \#bits$$

**Quantization error**:

The difference between the original image and the quantized image.

$$e[n] = x[n] - q(x[n])$$

**Distortion measure**:

$d(x, q(x))$ is a measure that quantifies the cost of substtituting the original value x by its representation q(x).

$$D(q) = E[d(X, q(X))] = \int_{-\infty}^{\infty} d(x, q(x)) f_X(x) dx = \sum_{i=0}^{L-1} \int_{\Delta_{i}}^{\Delta_{i-1}} d(x, y_i) f_X(x) dx$$

- **Mean square error**: average of the square of the difference between the original image and the quantized image. $ MSE = \frac{1}{N} \sum_{n=1}^N |e[n]|^2 $

### 1.2.c Video

**Still background**: the background is static and the foreground is moving.

**Variable background**: the background is moving and the foreground is also moving. (single or multiple gaussian distributions)

## 1.3 Discrete Stochastic Processes

A set of signals that can be analysed as a result of the same experiment, whose samples are random variables that keep some order (time, space, etc.).

- $\textbf{X[n]} = \{X[n,1], ..., X[n,i]\}$ is the random process. 
- $ X[n_0] $ is a random variable.
- $ X[n_0, i_0] $ is a deterministic value.

**Mean**:

$$m_X[n] = E[X[n]] = \int_{-\infty}^{\infty} x f_X(x;n) dx$$

**Instantaneous power**: deterministic function that measures the average power mof the process at each time instant.

$$P_X[n] = E[X[n]^2]$$

**Variance of a random process**: deterministic function that measures the variance of a process at each time instant.

$$\sigma_X[n]^2 = E[(X[n] - m_X[n])^2] = P_X[n] - m_X[n]^2$$

**Auto-correlation function**: deterministic 2D function that measures the correlation between two random variables defined at two samples

$$r[n_1, n_2] = E[X[n_1] X[n_2]]$$

$$r_{XY}[n_1, n_2] = E[X[n_1]Y[n_2]]$$

**Auto-covaariance function**:  deterministic 2D function that measures the similarity between two random variables defined at two samples

$$c[n_1, n_2] = E[(X[n_1] - m_X[n_1]) (X[n_2] - m_X[n_2])]$$

$$c_{XY}[n_1, n_2] = E[(X[n_1] - m_X[n_1]) (Y[n_2] - m_Y[n_2])]$$

**Independent random processes**: $X[n]$ and $Y[n]$ are independent if $f_{XY}(x,y) = f_X(x)f_Y(y)$

**Uncorrelated random processes**: $X[n]$ and $Y[n]$ are uncorrelated if $c_{XY}[n_1, n_2] = 0 \implies r_{XY}[n_1, n_2] = m_X[n_1] m_Y[n_2]$

**Orthogonal random processes**: $X[n]$ and $Y[n]$ are orthogonal if $r_{XY}[n_1, n_2] = 0$


**Wide-sense stationary (WSS)**: the statistical properties of the process do not change over time. The mean, variance and auto-correlation function are constant over time.

**Cyclotomic stationary (CSS)**: the statistical properties of the process do not change over time. The mean, variance and auto-correlation function are periodic over time.

**Linear Filtering of a Random Process**:

Given a stochastic process $X[n]$ and a linear filter $h[n]$:

$$Y[n] = \sum_{k=-\infty}^{\infty} h[k] X[n-k]$$

the output process $Y[n]$ is a linear function of the input process $X[n]$ and a stochastic process as well.

**Matrix notation**:

$$y[n] = h[n]^T \cdot x[n]$$

filter vector $h$ contains the $N$ coefficients of the filter.

# 2. Estimation Theory

## 2.1 Introduction

Given an N-point data set which depends on an unknown parameter, we wish to determine it based on the data.

The dependence of the available data with respect to the parameters is captured by the model that is proposed.

We can propose different estimators. We need to assess the performance of the estimators.

The estimated value depends on:
- The available realization
- The selected window

Thus the estimation is a random variable.

**Bias**: 
The bias of an estimator is the difference
between the expected value of the estimator
and the true value of the parameter being estimated.

**MVU**: 
The unbiased constrain is desirable and, among all unbiased estimators, that of minimum variance is preferred (Minimum Variance Unbiased: MVU)

**Consistent**: An estimator is consistent if, as the number of samples increases, the resulting sequence
of estimates converges to the true value of the parameter.

$$\lim_{N \to \infty} E[\hat{\theta}] = \theta, \ \lim_{N \to \infty} \sigma^2[\hat{\theta}] = 0$$

**MSE**: If the estimator is biased, the dispersion of the estimations with respect to the actual value to be estimated ($\theta$) is not the variance but the Mean Square Error of the estimator MSE.

## 2.2 C-R Bound and Efficient estimators

**Cramer-Rao Bound (CRB)**: The CRB is a lower bound on the variance of the estimator. It is a function of the Fisher information matrix.

- Determines the minimum possible variance for any unbiased estimator.

There exists a lower bound of the variance of the whole set of unbiased estimators of a parameter $\theta$.

$$\sigma^2[\hat{\theta}] \geq \frac{1}{I(\theta)}$$

where $I(\theta)$ is the Fisher information matrix.

$$\frac{\partial^2 \ln f(x_i;\theta)}{\partial \theta} = \kappa(\theta) (\hat{\theta} - \theta)$$

The more informative the set of samples, the sharper the likelihood function $\ln f(x; \theta)$.


## 2.3 ML and MAP Estimation

**ML**

Properties of the ML estimator:
1. Asymptotically unbiased (and in a large number of cases, unbiased).

2. Asymptotically efficient (when N increases, its variance attains CRLB).

3. Efficiency: When there exists an efficient estimator, it is the ML estimator.

4. Gaussian for N large: it is characterized by its mean and variance.

5. Invariance: The ML estimator of a function of $\alpha$ parameter $\alpha = g(\theta)$ can be obtained as: $\hat{\alpha} = g(\hat{\theta})$


**MAP**

A Bayesian estimator models the parameter we are attempting to estimate as a realization of a random variable, instead of as a constant unknown parameter.

Increasing number of samples: The conditional probability $f(x|\theta)$ will be sharper around $\theta_0$. In this case, if the information provided by $f(\theta)$ is correct,  both estimators tend to be the same.

# 3. Optimal and Adaptative Filtering

Given a set of data from an observed noisy process $x[x]$ and a desired target process $d[n]$ that we want to estimate, produce an estimated of the target process $y[n]$ by a linear time-invariant filter $h[n]$.

## Wiener-Hopf filter

- Assume known stationary signal and correlation.

The Wiener-Hopf filter is the optimal linear filter for a stationary random process $x[n]$.

- Observed noisy process $x[n] = a[n] + b[n]$ (observations).
- Desired target process we want to estimate $d[n] = a'[n] + b[n]$ (reference).
- Estimated of the target process $y[n]$ (output of the filter).
- $r_{aa'} \neq 0$ (only two parts correlated)

**System identification**: identify a given sistem (model).
- Noisy reference signal $d[n]$.
- Noise-free observation signal $x[n]$.

**System inversion**: estimate a system and apply its inverse to the signal.
- Noisy observation signal $x[n]$.
- Noise-free reference signal $d[n]$.
- Train data to generate an inverse convolution of the system.

**Signal prediction**
- Observation and reference samples of the same noisy process.

**Signal cancelation**: compare the primary signal $d[n]$ with the interference $x[n]$. The clean signal is $e[n]$. The signal that we want to obtain is in $d[n]$.
- Noisy observations with interferences.
- Noisy interferences as reference signal.

### Minimum MSE prediction
- Used as a optimization criteria.
- Mathematically tractable
- Useful for real solution applications

The Wiener-Hopf filter minimizes the MSE between the desired signal and the output of the filter.

$$\min_{h} E[e[n]^2] = \min_{h} E[(d[n] - y[n])^2] = \min_{h} E[(d[n] - h^T x[n])^2]$$

which implies that:

$$\nabla_h E[e[n]^2] = 0 \implies \nabla_h E[(d[n] - h^T x[n])^2] = 2E[(d[n]-h^T x[n]) \cdot (-x[n])] = E[e[n] x[n]] = 0$$

and $x[n]$ and $d[n]$ with zero mean, with error orthogonal to the observations.

Under $h_{opt}$, the solution of the Wiener-Hopf equation:

- $E[y[n]e[n]] = 0$ (orthogonality principle)

- $E[d[n]^2] \geq E[e[n]^2]$ (minimum MSE)

- If $E[x[n]d[n]] = 0 \implies E[y[n]^2] = 0$ (if the observations are uncorrelated with the desired signal, the variance of the estimation is zero)

- $\epsilon = E[e[n]^2] = r_d[0] - r_{dx}^T h_{opt}$ (minimum variance of the error = minimum MSE)

From $e[n]$ and $E[e[n]x[n]] = 0$ (or by differentiating $E[e[n]^2]$ with respect to $h$):

$$E[e[n] x[n]] = E[(d[n] - h^T x[n]) x[n]] = \\
E[d[n] x[n]] - E[h^T x[n] x[n]] = \\
r_{dx} - R_x h = 0$$

therefore:

$$h_{opt} = R_x^{-1} r_{dx}$$

For any filter, the MSE can be expressed as:

$$E[e[n]^2] = \epsilon + (h_{opt} - h)^T R_x (h_{opt} - h)$$

where $\epsilon = r_d[0] - r_{dx}^T h_{opt}$ is the variance of the error.


##  Linear prediction

The Wiener-hopf filter in the context of _forward prediction_.

- Reference: $d[n] = s[n]$
- Observations: $\underline{x}[n] = \underline{s}[n-1]$
- Estimation: $y[n] = \hat{s}[n]$

### Linear prediction coding

**Coding gain**: $G = \frac{E[s^2[n]]}{E[e^2[n]]}$

The decoder recieves the filter that has been used for prediction and the prediction errror.

By quantizing: $e[n]_q = e[n] + \epsilon_q[n]$

###  Linear prediction coding of speech signals

- Sound signal makes the vocal cords vibrate.

## Adaptive filters

- The Wiener-Hopf filter is not optimal for a non-stationary process.
  - Not fixed system. Time variant.
  - The Wiener-Hopf filter should adapt to the statistical variations of the process based on the study of the error.

**Speed of convergence**: It measures the capability of the algorithm to bring the adaptive  solution to the optimal one, independently of the  initial conditions. It is a transient‐phase property.

**Misadjustment**: It measures the stability of the reached  solution, once convergence is achieved. It is due to the randomness of the input data. It is a steady‐state property.

- An observation signal $x[n]$ with *low correlation* implies a *faster convergence*. The level curves tend to form a circle.
- An observation signal $x[n]$ with *high correlation* implies a *slower convergence*. The level curves tend to form an ellipse.

### Steepest descent algorithm

$$
\lim_{k \to \infty} \hat{h}[k] = h_{opt} \\
\lim_{k \to \infty} E[e^2[k]] = \epsilon
$$

The steepest descent algorithm is the gradient descent algorithm for the MSE.

$$
\hat{h}[k+1] = \hat{h}[k] - \frac{1}{2} \mu \nabla_{\hat{h}} E[e^2[k]] \\
$$

where $\mu$ is the step size and determines the speed of convergence towards the optimum.

The gradient of the error surface: $\nabla_{\hat{h}} E[e^2[k]] = -2 r_{dx} + 2 R_x \hat{h}[k]$

$$
\hat{h}[k+1] = \hat{h}[k] + \mu (r_{dx} - R_x \hat{h}[k])
$$

- When the level curves of the objective function tend to form a circle, the steepest descent algorithm converges faster, since the gradient points towards the optimum.

### Convergence analysis

The convergence of the steepest descent algorithm depends on the eigenvalues of the correlation matrix $R_x$.

- The correlation matrix $R_x$ is symmetric and semi-definite positive. $\lambda_i \geq 0$.

**Range of converegence:** $\mu \in (0, \frac{2}{\lambda_{max}})$

- $\mu = \frac{2}{\lambda_{max}}$: the algorithm converges in one iteration.

$$
\mu < \frac{2}{\lambda_{max}}, \\
\lambda_{max} \leq \sum \lambda_i = \text{trace} (R_x)
$$

- An increase of $r_x[0]$ implies an increas of $\lambda_{max}$ and a decrease of the range of convergence.

**Speed of convergence:**
- The speed of convergence is proportional to the dispertion of the eigenvalues. 

$$
N_{iter} \propto \frac{\lambda_{max}}{\lambda_{min}}\ln{\delta}
$$

- $\delta$ measures the distance of the current filter to the optimal filter.
  - A higher power $r_x[0]$ implies a lower eigenvalue dispertion.
  - In the limit, where $r_x[0] \to \infty$, the eigenvalue dispertion approaches to $1$, for the min and max eigenvalues tend to be infinite.
  - Low correlation $\implies$ low eigenvalue dispertion $\implies$ fast convergence.
  - High correlation $\implies$ high eigenvalue dispertion $\implies$ slow convergence.
  
  - In the case of a **white noise** signal, all the eigenvalues of the correlation matrix are equal to the variance of the signal. The eigenvalue dispertion is the minimum possible $1$.

- Eigenvalue dispertion affects the speed of convergence but not the misadjustment.

### LMS algorithm

The LMS algorithm is a stochastic approximation of the steepest descent algorithm. Since the correlation matrix and cross-correlation vector are unknown, they are estimated by the instantaneous values of the observations.

$$
\hat{h}[n+1] = \hat{h}[n] + \mu e[n] x[n]
$$

where $e[n] = d[n] - \hat{h}^T[n] x[n]$ is the instantaneous value of the error. 

- The gradient of the error surface is estimated by the instantaneous value of the error. Mimic a desired filter by finding the filter coefficients that relate to producing the least mean square of the error signal 

- Only one iteration can be done per sample. Thus the index $n$ is the same as the observation index.

- It is a stochastic gradient descent method in that the filter is only adapted based on the error at the current time.

- The correlation matrix $R_x$ is estimated by the instantaneous value of the observation vector.

$$
M \approx \frac{\mu}{2} \sum_{i=1}^{N} \lambda_i = \frac{\mu}{2} N r_x[0], \\
\mu << \frac{2}{N r_x[0]}
$$

- The misadjustment is proportional to the step size $\mu$.

- Increasing the power of the signal increases the misadjustment.