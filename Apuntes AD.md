# AD - Data Analysis

- [AD - Data Analysis](#ad---data-analysis)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
    - [What is PCA?](#what-is-pca)
    - [Objective](#objective)
    - [Steps of PCA](#steps-of-pca)
    - [Theory of PCA](#theory-of-pca)
    - [Scree plot](#scree-plot)
    - [Standardization of variables](#standardization-of-variables)
  - [Factor Analysis](#factor-analysis)
    - [What is Factor Analysis?](#what-is-factor-analysis)
    - [Variance of the factors](#variance-of-the-factors)
    - [Fractor Analysis by using PCA](#fractor-analysis-by-using-pca)
      - [Communality](#communality)
  - [Rotation](#rotation)
    - [Orthogonal rotation](#orthogonal-rotation)
    - [Oblique rotation](#oblique-rotation)
  - [Multidimensional Scaling](#multidimensional-scaling)
    - [What is Multidimensional Scaling?](#what-is-multidimensional-scaling)
    - [Algorithm](#algorithm)
  - [Correspondence Analysis](#correspondence-analysis)
    - [What is Correspondence Analysis?](#what-is-correspondence-analysis)
      - [Contingency table](#contingency-table)
    - [Test of independency](#test-of-independency)
    - [Profiles](#profiles)


## Principal Component Analysis (PCA)

### What is PCA?

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. 

- The number of principal components is less than or equal to the number of original variables.

- The first principal component accounts for as much of the variability in the data as possible, and each succeeding component accounts for as much of the remaining variability as possible. 

The transformation is defined in such a way that the first principal component has the largest possible variance, and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing $n$ observations) are an uncorrelated orthogonal basis set. 

- PCA is sensitive to the relative scaling of the original variables.

### Objective

- Dimension reduction: by considering correlation between variables. Lower dimensionality of the data while retaining as much information as possible. 
  - This makes visualization easier. It gives a better understanding of the data.

- To construct new uncorrelated variables called **principal components**.


### Steps of PCA

1. Check for correlation between variables. If there is a high correlation between variables, then PCA is a good choice. If there is no correlation, then PCA is not a good choice.

2. Center the data. This is done by subtracting the mean of each variable from each observation of that variable. This is done to make sure that the variables are centered around zero.

3. Calculate the covariance matrix $S$. This is done by multiplying the centered data matrix by its transpose. The covariance matrix is a square matrix with the number of rows and columns equal to the number of variables.

4. Calculate the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors are the principal components. The eigenvalues are the variances of the principal components.

5. Choose the most important components from the ones calculated in the previous step. This is done by choosing the components that have the highest eigenvalues. The number of components chosen depends on the amount of information that needs to be retained. The more components chosen, the more information is retained.

6. Project the data onto the principal components. This is done by multiplying the centered data matrix by the eigenvectors.

### Theory of PCA

The first principal component $Z_1$ is the linear combination of the original variables that accounts for the largest amount of variance in the data. The second principal component $Z_2$ is the linear combination of the original variables that accounts for the second largest amount of variance in the data, and is orthogonal to $Z_1$.

The $i^{th}$ principal component $Z_i$ is the linear combination of the original variables that accounts for the $i^{th}$ largest amount of variance in the data, and is orthogonal to all the previous principal components.

$$Z_1 = u_{11} x_1 + u_{12} x_2 + ... + u_{1p} x_p$$

where $Z_i$ is the $i^{th}$ principal component, $u_1, u_2, ..., u_p$ are the eigenvectors (loading vectors) of the covariance matrix (orthonormals), and $x_1, x_2, ..., x_p$ are the original variables.

The loading vector $u_1$ defines a direction in the feature space along which the data vary the most.

In matrix notation:

$$Z = X^cU$$

where $Z$ is the matrix of principal  $(n\times p)$, $X^c$ is the centered data matrix, and $U$ is the matrix of eigenvectors.

The variance of the $i^{th}$ principal component is the $i^{th}$ eigenvalue of the covariance matrix $S$ from SD (spectral descomposition).

$$ S = U D_{\lambda} U^T $$

where $S$ is the covariance matrix of the centered data, $D_{\lambda}$ is the diagonal matrix of eigenvalues, and $U$ is the matrix of eigenvectors.

This comes from the fact that,

$$ \text{Var}(Z) = \frac{1}{n-1}Z^TZ = ... = D_{\lambda} $$

(this can be obtained also from singular value decomposition (SVD) of the centered data matrix since there is no need for a square matrix).

The total variance of the PC is the sum of the eigenvalues of the covariance matrix.

$$ \text{Total var}(Z) = \sum_{i=1}^p \lambda_i = \sum_{i=1}^p s^2_{ii} = \text{trace}(S) $$

The proportion of variance explained by the $i^{th}$ principal component is the ratio of the $i^{th}$ eigenvalue to the sum of all the eigenvalues.

$$ \frac{\lambda_i}{\sum_{i=1}^p \lambda_i} $$

in order to decide the number of components to keep, we need to decide the proportion of variance we want to keep. This can be done by looking the magnitude of the eigenvalues or by the *scree plot*.

### Scree plot

The scree plot is a plot of the eigenvalues of the covariance matrix. The eigenvalues are plotted in descending order. 

$$ P_j = \frac{\lambda_j}{\sum_{i=1}^p \lambda_i} $$

The number of principal components to keep is the number of eigenvalues that are above the horizontal line. The horizontal line is the eigenvalue that is equal to the average of the eigenvalues.

### Standardization of variables

Standardization is a method of transforming variables so that they have a mean of zero and a standard deviation of one. This is done to make sure that the variables are on the same scale.

- If the $x_i$ variables are standardized (zero mean), then the covariance matrix $S$ is the correlation matrix $R$.
- Standardization is necessary when the units of measurement of the variables are different.

Correlation between variables is not affected by standardization, they are independent of measurement units. The eigenvalues are affected by standardization because the covariance matrix is affected by standardization.



## Factor Analysis

### What is Factor Analysis?

Factor analysis is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors. 

- The observed variables are assumed to be linear combinations of the factors plus random error.
- The factors are assumed to be uncorrelated.
- The factors are assumed to be normally distributed. 
- The random error is assumed to be normally distributed.


$$ X_i = a_{i1} F_1 + a_{i2} F_2 + ... + a_{ip} F_p + \epsilon_i $$

where $X_i$ is the $i^{th}$ standardized observation of the $p$ variables, $F_1, F_2, ..., F_p$ are the $p$ uncorrelated factors (mean zero and unit variance), $a_{ij}$ are the factor loadings, and $\epsilon_i$ is the random error (zero mean).

$$ \epsilon_i \sim N(0, \Sigma) $$

$$ X = aF + \epsilon $$

where $X$ is the matrix of observations $(n\times p)$, $F$ is the matrix of factors $(n\times p)$, $a$ is the matrix of factor loadings $(p\times p)$, and $\epsilon$ is the matrix of random error $(n\times p)$.

### Variance of the factors

The variance of $X_i$ is the sum of the variances of the factors and the variances of the random error.

$$ \text{Var}(X_i) = a_{i1}^2 \text{Var}(F_1) + a_{i2}^2 \text{Var}(F_2) + ... + a_{ip}^2 \text{Var}(F_p) + \text{Var}(\epsilon_i) $$

### Fractor Analysis by using PCA

If $m$ components are selected from the PCA, the random error term corresponds to the remaining $p-m$ components (linear combination).

$$ X_i = u_{1i}Z_1 + u_{2i}Z_2 + ... + u_{mi}Z_m + \epsilon_i $$

where $Z_1, Z_2, ..., Z_m$ are the $m$ principal components (since the eigenvectors are orthonormal), and $u_{1i}, u_{2i}, ..., u_{mi}$ are the eigenvecotrs of the covariance matrix.

To trasnform the PC to factors, we need to standarize the variables. So that 

$$ a_{ij} = u_{ij}\sqrt{\lambda_j} \implies F_i = \frac{Z_i}{\sqrt{\lambda_i}} $$

where $\lambda_i$ is the $i^{th}$ eigenvalue of the covariance matrix.

#### Communality

The communality is the proportion of the variance of the $i^{th}$ variable that is explained by the factors. The communality is the sum of the squared factor loadings.

$$ c_i = \sum_{j=1}^p a_{ij}^2 $$

The higher the communality, the more the variable is explained by the factors.

## Rotation

### Orthogonal rotation

Orthogonal rotation is a method of transforming the factor loadings so that the factors are uncorrelated.

### Oblique rotation

Oblique rotation is a method of transforming the factor loadings so that the factors are correlated.

## Multidimensional Scaling

### What is Multidimensional Scaling?

Multidimensional scaling (MDS) is a statistical technique for reducing the dimensionality of a data set by representing the data as points in a geometric space. The geometric space is defined by the distances between the points. 

Similarity and distance are inversevely related. The closer the points are, the more similar they are. The farther the points are, the less similar they are.

Similarity: $s(x, y)$

- For categorical variables it can be defined as
$$ s(x, y) = \alpha / k, \ \ \alpha = \sum 1_{x_i = y_i} $$
  - where $k$ is the number of categories.

Distance: $d(x, y)$

$$ d^2(x, y) = s(x, x) + s(y, y) - 2s(x, y) $$

**Similarity matrix**: $Q = X^TX$, where the columns of $X$ have zero mean and are orthogonal to each other. If $X$ is not centered, then $(I - 11)X$ is used as a centered transformation of $X$.

since $Q_{ij}$ is the dot product of the $i^{th}$ and $j^{th}$ row of $X$. If both elements are the same, the dot product is the square of the element ($\cos(\alpha_{ij}) = 1$). If the elements are different, the dot product close to zero.

**Distance matrix**: $D$

$$ d_{ij}^2 = s_{ii} + s_{jj} - 2s_{ij} = \sum_{k=1}^p x_{ik}^2 + \sum_{k=1}^p x_{ik}^2 - 2\sum_{k=1}^p x_{ik}x_{jk} = \sum_{k=1}^p (x_{ik} - x_{jk})^2$$

### Algorithm

1. Compute the distance matrix $D$ and $D^2$.
2. Define the centering matrix $P = I - \frac{1}{n}11^T$.
3. Build the cross product matrix $Q = -\frac{1}{2}PD^2P$.
4. Build $X = V_p \Lambda^{1/2}_p$, where $V$ is the matrix of eigenvectors of $Q$ and $\Lambda$ is the diagonal matrix of eigenvalues of $Q$. Using the $p$ greatest eigenvalues.

## Correspondence Analysis

### What is Correspondence Analysis?

Correspondence analysis is a statistical technique for displaying the rows and columns of a contingency table in a low-dimensional space. The rows and columns are represented as points in a geometric space. The geometric space is defined by the correspondence between the rows and columns.

- To understand relationships between categorical variables.
  - Visualize associations between categorical variables.
- Understand the contingency table as a data matrix.

#### Contingency table

A contingency table is a table showing the categories of one variable in rows and another in columns.

Each cell $x_{ij}$ is the number of observations that fall into the $i^{th}$ row category and the $j^{th}$ column category.

- Margin: row or column total $x_{i.}$ or $x_{.j}$
  - Marginal probability: $f_{i.} = \frac{x_{i.}}{n}$ or $f_{.j} = \frac{x_{.j}}{n}$

- Frecuency of the cell: $f_{ij} = \frac{x_{ij}}{n}$ (joint probability)
  - If independent, $f_{ij} = f_{i.}f_{.j}$

- Conditional probability: $f_{j|i} = \frac{f_{ij}}{f_{i.}}$ or $f_{i|j} = \frac{f_{ij}}{f_{.j}}$

### Test of independency

The null hypothesis is that the two categorical variables are independent.

- Chi square test of independence

$$ \chi^2 = \sum_{i, j}^{I, J} \frac{(x_{ij} - \hat{x}_{ij})^2}{\hat{x}_{ij}} = n \sum_{i, j} \frac{(n f_{ij} - n f_{i.}f_{.j})^2}{n f_{i.}f_{.j}} = n \phi^2 \sim \chi^2_{(I-1)(J-1)}$$

- where $\hat{x}_{ij} = \frac{x_{i.}x_{.j}}{n}$ is the expected value of the cell $x_{ij}$ under the null hypothesis.

- $\phi^2$ is called link. The deviation of the link is associated with the desviation of observed from expected values.

### Profiles

Row profiles are found as $f_{j|i} = \frac{f_{ij}}{f_{i.}}$ and column profiles are found as $f_{i|j} = \frac{f_{ij}}{f_{.j}}$.






