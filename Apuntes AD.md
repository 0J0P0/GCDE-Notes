# AD - Data Analysis

## Principal Component Analysis (PCA)

### What is PCA?

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. 

- The number of principal components is less than or equal to the number of original variables.

- The first principal component accounts for as much of the variability in the data as possible, and each succeeding component accounts for as much of the remaining variability as possible. 

The transformation is defined in such a way that the first principal component has the largest possible variance, and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing $n$ observations) are an uncorrelated orthogonal basis set. 

- PCA is sensitive to the relative scaling of the original variables.

### Objective

Dimension reduction. Done by considering correlation between variables. 

To construct new uncorrelated variables called **principal components**.

Lower dimensionality of the data while retaining as much information as possible. This makes visualization easier. It gives a better understanding of the data.

### Steps of PCA

1. Check for correlation between variables. If there is a high correlation between variables, then PCA is a good choice. If there is no correlation, then PCA is not a good choice.

2. Center the data. This is done by subtracting the mean of each variable from each observation of that variable. This is done to make sure that the variables are centered around zero.

3. Calculate the covariance matrix. This is done by multiplying the centered data matrix by its transpose. The covariance matrix is a square matrix with the number of rows and columns equal to the number of variables.

4. Calculate the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors are the principal components. The eigenvalues are the variances of the principal components.

5. Choose the most important components from the ones calculated in the previous step. This is done by choosing the components that have the highest eigenvalues. The number of components chosen depends on the amount of information that needs to be retained. The more components chosen, the more information is retained.

6. Project the data onto the principal components. This is done by multiplying the centered data matrix by the eigenvectors.

### Theory of PCA

The first principal component $Z_1$ is the linear combination of the original variables that accounts for the largest amount of variance in the data. The second principal component $Z_2$ is the linear combination of the original variables that accounts for the second largest amount of variance in the data, and is orthogonal to $Z_1$.

The third principal component $Z_3$ is the linear combination of the original variables that accounts for the third largest amount of variance in the data, and is orthogonal to both $Z_1$ and $Z_2$, and so on. The $i^{th}$ principal component $Z_i$ is the linear combination of the original variables that accounts for the $i^{th}$ largest amount of variance in the data, and is orthogonal to all the previous principal components.

$$Z_1 = u_{11} x_1 + u_{12} x_2 + ... + u_{1p} x_p$$

where $Z_i$ is the $i^{th}$ principal component, $u_1, u_2, ..., u_p$ are the eigenvectors of the covariance matrix (orthonormal), and $x_1, x_2, ..., x_p$ are the original variables.

Each $Z_i$ has a individual score for each observation. $Z_i$ is uncorrelateed with all the other PC since they are orthogonal to each other.

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

$$ \text{Var}(Z) = \sum_{i=1}^p \lambda_i = \sum_{i=1}^p s^2_{ii} = \text{trace}(S) $$

The proportion of variance explained by the $i^{th}$ principal component is the ratio of the $i^{th}$ eigenvalue to the sum of all the eigenvalues.

$$ \frac{\lambda_i}{\sum_{i=1}^p \lambda_i} $$

in order to decide the number of components to keep, we need to decide the proportion of variance we want to keep. This can be done by looking the magnitude of the eigenvalues or by the *scree plot*.

### Scree plot

The scree plot is a plot of the eigenvalues of the covariance matrix. The eigenvalues are plotted in descending order. 

$$ P_j = \frac{\lambda_j}{\sum_{i=1}^p \lambda_i} $$

The number of principal components to keep is the number of eigenvalues that are above the horizontal line. The horizontal line is the eigenvalue that is equal to the average of the eigenvalues. The average of the eigenvalues is the average of the variances of the principal components.

### Standardization of variables

Standardization is a method of transforming variables so that they have a mean of zero and a standard deviation of one. This is done to make sure that the variables are on the same scale.

Correlation between variables is not affected by standardization, they are independent of measurement units. The eigenvalues are affected by standardization because the covariance matrix is affected by standardization.


## Factor Analysis

### What is Factor Analysis?

Factor analysis is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors. Factor analysis is used to describe the underlying structure of observed variables in terms of a smaller number of unobserved variables called factors. The factors are linear combinations of the observed variables. The factors are assumed to be uncorrelated. The observed variables are assumed to be linear combinations of the factors plus random error. The factors are assumed to be uncorrelated with the random error. The factors are assumed to be normally distributed. The random error is assumed to be normally distributed.

Based on a model where the variables can be written as a linear combination of factors plus random error. The factors are assumed to be uncorrelated with the random error. The factors are assumed to be normally distributed. The random error is assumed to be normally distributed.

$$ X_i = a_{i1} F_1 + a_{i2} F_2 + ... + a_{ip} F_p + \epsilon_i $$

where $X_i$ is the $i^{th}$ observation of the $p$ variables, $F_1, F_2, ..., F_p$ are the $p$ uncorrelated factors (mean zero and unit variance), $a_{ij}$ are the factor loadings, and $\epsilon_i$ is the random error (mean zero).

$$ \epsilon_i \sim N(0, \Sigma) $$

$$ X = aF + \epsilon $$

where $X$ is the matrix of observations $(n\times p)$, $F$ is the matrix of factors $(n\times p)$, $a$ is the matrix of factor loadings $(p\times p)$, and $\epsilon$ is the matrix of random error $(n\times p)$.

### Variance of the factors

The variance of the factors is the sum of the variances of the factors and the variances of the random error.

$$ \text{Var}(F) = \text{Var}(F) + \text{Var}(\epsilon) $$

The variance of $X_i$ is the sum of the variances of the factors and the variances of the random error.

$$ \text{Var}(X_i) = a_{i1}^2 \text{Var}(F_1) + a_{i2}^2 \text{Var}(F_2) + ... + a_{ip}^2 \text{Var}(F_p) + \text{Var}(\epsilon_i) $$

### Fractor Analysis by using PCA

To trasnform the pc to factors, we need to standarize the variables. So that 

$$ a_{ij} = u_{ij}\sqrt{\lambda_j} $$

where $u_{ij}$ is the $i^{th}$ eigenvector of the covariance matrix, and $\lambda_j$ is the $j^{th}$ eigenvalue of the covariance matrix.

...

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

Distance: $d(x, y)$

$$ d^2(x, y) = s(x, x) + s(y, y) - 2s(x, y) $$

**Similarity matrix**: $Q = X^TX$

since $Q_{ij}$ is the dot product of the $i^{th}$ and $j^{th}$ row of $X$. If both elements are the same, the dot product is the square of the element. If the elements are different, the dot product close to zero.

**Distance matrix**: $D$

$$ D_{ij} = \sqrt{Q_{ii} + Q_{jj} - 2Q_{ij}} $$

since,

$$ d_{ij}^2 = s_{ii} + s_{jj} - 2s_{ij} = \sum_{k=1}^p x_{ik}^2 + \sum_{k=1}^p x_{ik}^2 - 2\sum_{k=1}^p x_{ik}x_{jk} = \sum_{k=1}^p (x_{ik} - x_{jk})^2$$