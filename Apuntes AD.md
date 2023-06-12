# AD - Data Analysis

> If `pvalue < 0.05` then we reject the null hypothesis. If `pvalue > 0.05` then we accept the null hypothesis.

> Covariance matrix $S = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^T$ (square and symmetric matrix). Or $S = \frac{1}{n} X^T P X$ where $P$ is the centering matrix.
> When dividing by $n-1$ instead of $n$ we get an unbiased estimator of the covariance matrix.

> Standarisazion of variables implies that the covariance matrix is the correlation matrix.

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
      - [Cloud of profiles $N\_I$ or $N\_J$](#cloud-of-profiles-n_i-or-n_j)
    - [Intertia](#intertia)
    - [Representation quality](#representation-quality)
  - [Multiple Correspondence Analysis](#multiple-correspondence-analysis)
    - [Studying individuals](#studying-individuals)
    - [Relationship to CA](#relationship-to-ca)
  - [Cluster Analysis](#cluster-analysis)
    - [Hierarchical clustering](#hierarchical-clustering)
      - [Single nearest neighbor](#single-nearest-neighbor)
      - [Single farthest neighbor](#single-farthest-neighbor)
      - [Average linkage](#average-linkage)
      - [Centroide distance](#centroide-distance)
      - [Ward's criterion](#wards-criterion)
    - [Elbow method](#elbow-method)
    - [Pseudo F index](#pseudo-f-index)
    - [Silhouette index](#silhouette-index)
  - [Linear Discriminant Analysis](#linear-discriminant-analysis)
    - [LDA vs PCA](#lda-vs-pca)


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

3. Calculate the covariance matrix $S$. This is done by multiplying the centered data matrix by its transpose and dividing by the number of observations. The covariance matrix is a square matrix with the variances of the variables along the diagonal and the covariances between each pair of variables in the off-diagonal elements.

4. Calculate the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors are the principal components. The eigenvalues are the variances of the principal components.

5. Choose the most important components from the ones calculated in the previous step. This is done by choosing the components that have the highest eigenvalues. The number of components chosen depends on the amount of information that needs to be retained. The more components chosen, the more information is retained.

6. Project the data onto the principal components. This is done by multiplying the centered data matrix by the eigenvectors. We obtain the **scores**. 

### Theory of PCA

The first principal component $Z_1$ is the linear combination of the original variables that accounts for the largest amount of variance in the data. The second principal component $Z_2$ is the linear combination of the original variables that accounts for the second largest amount of variance in the data, and is orthogonal to $Z_1$.

The $i^{th}$ principal component $Z_i$ is the linear combination of the original variables that accounts for the $i^{th}$ largest amount of variance in the data, and is orthogonal to all the previous principal components.

$$Z_1 = u_{11} x_1 + u_{12} x_2 + ... + u_{1p} x_p$$

where $Z_i$ is the $i^{th}$ principal component, $u_1, u_2, ..., u_p$ are the eigenvectors (loading vectors) of the covariance matrix (orthonormals), and $x_1, x_2, ..., x_p$ are the original variables.

The loading vector $u_1$ defines a direction in the feature space along which the data vary the most.

In matrix notation:

$$Z = X^cU$$

where $Z$ is the matrix of principal $(n\times p)$, $X^c$ is the centered data matrix, and $U$ is the matrix of eigenvectors.

The variance of the $i^{th}$ principal component is the $i^{th}$ eigenvalue of the covariance matrix $S$ from SD (spectral descomposition).

$$S = U D_{\lambda} U^T$$

where $S$ is the covariance matrix of the centered data, $D_{\lambda}$ is the diagonal matrix of eigenvalues, and $U$ is the matrix of eigenvectors.

This comes from the fact that,

$$\text{Var}(Z) = \frac{1}{n-1}Z^TZ = ... = D_{\lambda}$$

(this can be obtained also from singular value decomposition (SVD) of the centered data matrix since there is no need for a square matrix).

The total variance of the PC is the sum of the eigenvalues of the covariance matrix.

$$\text{Total var}(Z) = \sum_{i=1}^p \lambda_i = \sum_{i=1}^p s^2_{ii} = \text{trace}(S)$$

The proportion of variance explained by the $i^{th}$ principal component is the ratio of the $i^{th}$ eigenvalue to the sum of all the eigenvalues.

$$\frac{\lambda_i}{\sum_{i=1}^p \lambda_i}$$

in order to decide the number of components to keep, we need to decide the proportion of variance we want to keep ($80\%$). This can be done by looking the magnitude of the eigenvalues or by the *scree plot*. Alternatively the dimensions for which $\lambda_i > \bar\lambda$ can be kept, if the variables have beens standardized $\bar\lambda=1$.

### Scree plot

The scree plot is a plot of the eigenvalues of the covariance matrix. The eigenvalues are plotted in descending order. 

$$P_j = \frac{\lambda_j}{\sum_{i=1}^p \lambda_i}$$

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


$$X_i = a_{i1} F_1 + a_{i2} F_2 + ... + a_{ip} F_p + \epsilon_i$$

where $X_i$ is the $i^{th}$ standardized observation of the $p$ variables, $F_1, F_2, ..., F_p$ are the $p$ uncorrelated factors (mean zero and unit variance), $a_{ij}$ are the factor loadings, and $\epsilon_i$ is the random error (zero mean).

$$\epsilon_i \sim N(0, \Sigma)$$

$$X = aF + \epsilon$$

where $X$ is the matrix of observations $(n\times p)$, $F$ is the matrix of factors $(n\times p)$, $a$ is the matrix of factor loadings $(p\times p)$, and $\epsilon$ is the matrix of random error $(n\times p)$.

### Variance of the factors

The variance of $X_i$ is the sum of the variances of the factors and the variances of the random error.

$$\text{Var}(X_i) = a_{i1}^2 \text{Var}(F_1) + a_{i2}^2 \text{Var}(F_2) + ... + a_{ip}^2 \text{Var}(F_p) + \text{Var}(\epsilon_i)$$

### Fractor Analysis by using PCA

If $m$ components are selected from the PCA, the random error term corresponds to the remaining $p-m$ components (linear combination).

$$X_i = u_{1i}Z_1 + u_{2i}Z_2 + ... + u_{mi}Z_m + \epsilon_i$$

where $Z_1, Z_2, ..., Z_m$ are the $m$ principal components (since the eigenvectors are orthonormal), and $u_{1i}, u_{2i}, ..., u_{mi}$ are the eigenvecotrs of the covariance matrix.

To transform the PC to factors, we need to standarize the variables. So that 

$$a_{ij} = u_{ij}\sqrt{\lambda_j} \implies F_i = \frac{Z_i}{\sqrt{\lambda_i}}$$

where $\lambda_i$ is the $i^{th}$ eigenvalue of the covariance matrix.

#### Communality

The communality is the proportion of the variance of the $i^{th}$ variable that is explained by the factors. The communality is the sum of the squared factor loadings.

$$c_i = \sum_{j=1}^p a_{ij}^2$$

The higher the communality, the more the variable is explained by the factors.

## Rotation

### Orthogonal rotation

Orthogonal rotation is a method of transforming the factor loadings so that the factors are uncorrelated.

### Oblique rotation

Oblique rotation is a method of transforming the factor loadings so that the factors are correlated.

## Multidimensional Scaling

### What is Multidimensional Scaling?

Multidimensional scaling (MDS) is a statistical technique for reducing the dimensionality of a data set by representing the data as points in a geometric space. The geometric space is defined by the distances between the points.  

- Is a generalization of PCA. Instead of using the covariance matrix, it uses the similarity matrix $Q=XX^T$.

Similarity and distance are inversevely related. The closer the points are, the more similar they are. The farther the points are, the less similar they are.

Similarity: $s(x, y)$

- For categorical variables it can be defined as
$$s(x, y) = \alpha / k, \ \ \alpha = \sum 1_{x_i = y_i}$$
  - where $k$ is the number of categories.

Distance: $d(x, y)$

$$d^2(x, y) = s(x, x) + s(y, y) - 2s(x, y)$$

**Similarity matrix**: $Q = XX^T$ (cross-product matrix), where the columns of $X$ have zero mean and are orthogonal to each other. If $X$ is not centered, then $(I - \frac{1}{n} 1^T1)X$ is used as a centered transformation of $X$.  Note the difference between $Q=XX^T$ and $S=\frac{1}{n}X^TX$.


since $Q_{ij}$ is the dot product of the $i^{th}$ and $j^{th}$ row of $X$. If both elements are the same, the dot product is the square of the element ($\cos(\alpha_{ij}) = 1$). If the elements are different, the dot product close to zero.

**Distance matrix**: $D$

Since the covariance matrix is a similarity matrix $q_{ij} = s_{ij}$, the distance matrix can be computed as

$$d_{ij}^2 = s_{ii} + s_{jj} - 2s_{ij} = \sum_{k=1}^p x_{ik}^2 + \sum_{k=1}^p x_{ik}^2 - 2\sum_{k=1}^p x_{ik}x_{jk} = \sum_{k=1}^p (x_{ik} - x_{jk})^2$$

### Algorithm

1. Compute the distance matrix $D$ and $D^2$.
2. Define the centering matrix $P = I - \frac{1}{n}11^T$.
3. Build the cross product matrix $Q = -\frac{1}{2}PD^2P$.
4. Build $X = V_p \Lambda^{1/2}_p$, where $V$ is the matrix of eigenvectors of $Q$ and $\Lambda$ is the diagonal matrix of eigenvalues of $Q$. Using the $p$ greatest eigenvalues.

Recompute $X$ from $D$ and $Q$.

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

The null hypothesis is that the rows and columns of the contengency table are independent (the two categorical variables are independent).

- Chi square test of independence
$$\chi^2 = \sum_{i,j}\left(\frac{real_{ij}-expected_{ij}}{expected_{ij}}\right)$$

$$\chi^2 = \sum_{i, j}^{I, J} \frac{(x_{ij} - \hat{x}_{ij})^2}{\hat{x}_{ij}} = n \sum_{i, j} \frac{(f_{ij} - f_{i.}f_{.j})^2}{f_{i.}f_{.j}} = n\phi^2 \sim \chi^2_{(I-1)(J-1)}$$
  

- where $\hat{x}_{ij} = \frac{x_{i.}x_{.j}}{n}$ is the expected value of the cell $x_{ij}$ under the null hypothesis.

- $\phi^2$ is called link. The desviation of the link is associated with the desviation of observed from expected values (under independence).

  - When independence, the profiles are the same as the mean profiles.
  - The cloud has 0 inertia.
  - The further the data from independence, the higher the inertia and the more the profiles spread from the mean profiles.
  - As will be later seen, this interpreation matches the physical interpretation of the inertia of the cloud of points.

### Profiles

Row profiles are found as $f_{j|i} = \frac{f_{ij}}{f_{i.}}$ $R_I=(f_{1|i},\dots,f_{J|i})$ and column profiles are found as $f_{i|j} = \frac{f_{ij}}{f_{.j}}$. Note that the distance measures weight each dimension by $1/f_{·j}$

- **Row profile** shows the distribution of the row variable across the different categories of the column variable

- The **average profile** $G_I$ is the average of the row profiles. It is the center of the cloud of row profiles.

#### Cloud of profiles $N_I$ or $N_J$

The bigger the distance, the higher variance between the point of the plots.

**Distance**:

$d_{i,l}^2 = \sum_{j=1}^J \frac{1}{f_{.j}} (\frac{f_{ij}}{fi.} - \frac{f_{lj}}{f_{l.}})^2$ (row profiles)

$d_{j,l}^2 = \sum_{i=1}^I \frac{1}{f_{i.}} (\frac{f_{ij}}{f_{.j}} - \frac{f_{il}}{f_{.l}})^2$ (column profiles)

### Intertia

The inertia is the sum of the squared distances between the points and the center of the cloud of points $G_I$.

- The farther the point in the cloud from the center, the less similar the profile is to the median profile.

$$Inertia(N_I/G_I) = \sum_{i=1}^I Inertia(I/G_I) = \sum_{i=1}Î f_{i.} d_{i, G_I}^2 = \phi^2$$

$$Inertia(N_J/G_J) = \sum_{j=1}^J Inertia(J/G_J) = \sum_{j=1}^J f_{.j} d_{j, G_J}^2 = \phi^2$$

$$
\sum_{k=1}^K \lambda_k = Inertia(N_I/G_I) = Inertia(N_J/G_J) = \phi^2
$$

### Representation quality
We can measure, similary lo in PCA:
- Quality of point $i$: 
 $$qlt_s(i)=\frac{inertia of i in rank r}{total inertia of i}=cos^2(O_i,OH_i^s)$$
- Contribution:
  $$ctr_s(i)=\frac{inertia of i on rank s}{inertia of N_I on rank s}$$

Supplementary elements can be added to the analysisby using the barycentric properties.
```r
res.ca <- CA(df)

# Eigenvalues
res.ca$eig 
# Row profiles
res.ca$row
# Coordinates of the row profiles
res.ca$row$coord
# Cloud of row profiles
plot.CA(res.ca, choix = "row", invisible = "col")
```

## Multiple Correspondence Analysis

To visualize relationships between categories of J number of qualitative variables for I number of individuals.

**Indicator matrix**: Rows represent individuals and columns represent dummy variables for each category of the qualitative variables.

- $I_{ij} = 1$ if the individual $i$ is in the category $j$ of the qualitative variable.

**Burt matrix**: Rows represent categories of the qualitative variables and columns represent categories of the qualitative variables. All possible pairs of categories.

- $B_{ij}$ is the number of time the categorical pairs $i,j$ appear together.
- Symmetric matrix.

$$
\lambda^{B} = (\lambda^{I})^2
$$

### Studying individuals

The distance between two categories is measured based on the number of individuals they have in common.

The fewer individuals they have in common, the further they are.

$$
d_{ij}^2 = C \sum_{k=1}^K \frac{(x_{ik}-x_{jk})^2}{I_k}
$$

- $C$ is a constant.
- $I_k$ is the number of individuals in the category $k$.
- $x_{ik}$ is 1 if the individual $i$ is in the category $k$ and 0 otherwise.

### Relationship to CA
By setting $C=I/J$ wwe can obtain the distnace between row and column profiles equivalent to CA. 

The inertia of a category $K$ is given by $\frac{1}{J}\left(1-\frac{I} {I_k}\right)$ and increases when the category is Rare.

The inertia of a variable is given by $\frac{K_j-1}{J}$.

The total inertia of the cloud of categories is $K/J-1$.

As in CA, the barycentric property applies.



## Cluster Analysis

- To group individuals
- To reduce dimensionality

### Hierarchical clustering

- In hierarchical clustering, the clusters are organized as a tree. Denoted as a dendrogram.

**Agglomeration**: Start with each individual in its own cluster and then merge clusters until all individuals are in the same cluster.
- Close groups are merged first.

**Division**: Start with all individuals in the same cluster and then split clusters until each individual is in its own cluster.
- Distant groups are split first.

#### Single nearest neighbor

$$
d(C_1, C_2) = \min_{i \in C_1, j \in C_2} d_{x_i, x_j}
$$

#### Single farthest neighbor

$$
d(C_1, C_2) = \max_{i \in C_1, j \in C_2} d_{x_i, x_j}
$$

#### Average linkage

$$
d(C_1, C_2) = \frac{1}{|C_1||C_2|} \sum_{i \in C_1, j \in C_2} d_{x_i, x_j}
$$

#### Centroide distance

$$
d(C_1, C_2) = d_{\bar{x}_{C_1}, \bar{x}_{C_2}}
$$

#### Ward's criterion

Minimize the variance of the clusters and maximize the variance between clusters.

$$
SS_A = \sum_{i=1}^{|C_1|} (x_i - \bar{x_A})^2
$$

$$
SS_B = \sum_{i=1}^{|C_2|} (x_i - \bar{x_B})^2
$$

$$
SS_{A \cup B} = \sum_{i=1}^{|C_1| + |C_2|} (x_i - \bar{x}_{A \cup B})^2, \ \ \bar{x}_{A \cup B} = \frac{|C_1| \bar{x_A} + |C_2| \bar{x_B}}{|C_1| + |C_2|}
$$

$$
\min \Delta = \min SS_{A \cup B} - (SS_A + SS_B)
$$

```r
?hclust # hiercal clustering
fit <- hclust(d, method="single") 
plot(fit,main="Dendrogram of Single Linkage") # Dendogram
```

### Elbow method

- To determine the number of clusters.
- Total within sum of squares (TWSS) per number of clusters.
- The optimal number of clusters is the number of clusters after which the TWSS starts to decrease in a linear fashion.

```r
k4$withinss
# SSQ within each cluster
k4$totss
# SSQ total
k4$tot.withinss
# Total SSQ within clusters
k4$betweenss + k4$tot.withinss
# SSQ total
```

### Pseudo F index

Defines the ratio of between cluster sum of squares (BSS) to the within cluster sum of squares (WSS)

$$
F = \frac{BSS/(K-1)}{WSS/(N-K)}
$$

- $K$ is the number of clusters.
- $N$ is the number of individuals.

```r	
aux<-c()
for (i in 2:10){
  k<-kmeans(usa[,-5],centers=i,nstart=25)
  aux[i]<-((k$betweenss)*(nrow(usa)-i))/((k$tot.withinss)*(i-1))
}
plot(aux, xlab="Number of Clusters", ylab="Pseudo-F", type="l", main="Pseudo F Index")
which.max(aux) # max value is selected
```

### Silhouette index

The Silhouette index is a measure used to assess the quality of clustering in data analysis.

Measures the closeness of each point in a cluster to the points in the other clusters.

The Silhouette index is calculated for each data point and ranges from -1 to 1. A higher value indicates that the data point is well-matched to its own cluster and poorly-matched to neighboring clusters, while a lower value suggests that the data point may be assigned to the wrong cluster.

The number of clusters that maximizes the Silhoutte index is chosen as the optimum number of clusters.

- For each data point, calculate two distances:
  - a) Average distance to all other data points in the same cluster (cohesion).
  - b) Average distance to all data points in the nearest neighboring cluster (separation).

- Compute the silhouette coefficient for each data point using the formula:
silhouette coefficient = (separation - cohesion) / max(separation, cohesion)

- Calculate the average silhouette coefficient across all data points to obtain the Silhouette index for the clustering algorithm.

```r
res <- fastkmed(d, 6)
# Depending on the number of clusters, the silhouette index may be different. The distances may be different.
silhouette <- sil(d, res$medoid, res$cluster)
silhouette$result
silhouette$plot
```

## Linear Discriminant Analysis

- The difference with clustering is that in LDA we know how many groups we want to find. We want to find the best linear combination of variables that separates the groups.

- To classify individuals into groups by a linear combination of quantitative variables.

**Assumptions**:
- The distribution of a quantitative variable is normal in each category $k$: $X_k \sim N(\mu_k, \Sigma_k)$
- The covariance matrices are equal: $\Sigma_1 = \Sigma_2 = ... = \Sigma_k = \Sigma$

The exponent of the multivariate normal distribution is:

$$
-\frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x - \mu_k)
$$

which refers to the Mahalanobis distance between the observation and the mean of the category k.

- The bigger the Mahalanobis distance $D_k$, the less likely the observation belongs to the category $k$. This is because the observation is far from the mean of the category $k$.

**ML estimators:**
- $\hat{\mu}_k = \frac{1}{n} \sum_{i=1}^n x_i$
- $\hat{\Sigma}_k = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$

The probability function that an observation comes from a given category is:

$$
f_k(x) = P(X = x | Y = k) = \frac{1}{(2 \pi)^{p/2} |\Sigma|^{1/2}} e^{-\frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x - \mu_k)}
$$

Using Bayes' theorem, the probability that an observation belongs to a given category is (posterior probability):

$$
P(Y = k | X = x) = \frac{P(X = x | Y = k) P(Y = k)}{P(X = x)} = \frac{f_k(x) \pi_k}{\sum_{l=1}^K f_l(x) \pi_l}
$$

where $P(Y = k)$ is the prior probability of category k and $P(X = x)$ is the marginal probability of the observation, which is the sum of $P(X = x | Y = k) P(Y = k)$ over all categories.

For two categories, the probability that an observation belongs to category 1 is:

$$
P(Y = 1 | X = x) = \frac{f_1(x) \pi_1}{f_1(x) \pi_1 + f_2(x) \pi_2} = \frac{1}{1 + \frac{\pi_1}{\pi_2} e^{-\frac{1}{2}(D_2^2 - D_1^2)}}
$$

- The greater the distance $D_1$ with respect to $D_2$, the greater the denominator and the greater the probability that the observation belongs to category 2. The opposite is true for category 1. Assuming that $\frac{\pi_1}{\pi_2} = 1$.

- Similarly, if $\pi_1 f_1(x) < \pi_2 f_2(x)$, then the observation is more likely to belong to category 2. The opposite is true for category 1.

The maximization of this probability is equivalent to the minimization of the Mahalanobis distance and leads to the Bayes classifier.

The objective is to maximize the ratio of the between-group variance to the within-group variance.

$$
\frac{\hat{\mu_1} - \hat{\mu_2}}{s_1^2 + s_2^2}
$$

where $\hat{\mu_k}$ is the mean of the observations in category k and $s_k^2$ is the variance of the observations in category k.

The misclassifications are denoted by $P(1|2)$ and $P(2|1)$ in the case of two categories. Maximizing the ratio of the between-group variance to the within-group variance is equivalent to minimizing the misclassification rate.

**Classification Rule**

$$
\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k), \ \ \delta_k(x) \approx \log(\pi_k f_k(x))
$$

- The observation is assigned to the category k for which $\delta_k(x)$ is the largest.

**Correct classification rate**

|          | Predicted 1 | Predicted 2 |
| -------- | ----------- | ----------- |
| Actual 1 | $n_{11}$    | $n_{12}$    |
| Actual 2 | $n_{21}$    | $n_{22}$    |


$$
CRR = \frac{n_{11} + n_{22}}{n_{11} + n_{12} + n_{21} + n_{22}}
$$

```r	
skullda<-lda(type~., data=skulls)
# Prior probabilities initially computed from the sample sizes.

# The group means of the variables are the centroids of the groups.
# The greater the difference between the group means of each variable, the greater the difference and the better the separation of the groups with respect to that variable.

# The number of discriminant functions is equal to the number of groups minus one. The number of groups is equal to the number of levels of the response variable.
```

```r	
# Prediction Accuracy p1^2+p^2
pa<-skullda$prior[1]^2 + skullda$prior[2]^2
# Prediction Accuracy should be greater than 1/levels of the response variable. This would mean that the model is better than random guessing.
```

### LDA vs PCA

- Supervision:
  - PCA is an unsupervised technique, meaning it does not consider class labels during its computations.
  - LDA is a supervised technique that utilizes class labels to maximize the separability between different classes.
- Objective:
  - The objective of PCA is to maximize the variance in the dataset, capturing the directions (principal components) that explain the most variability.
  - The objective of LDA is to maximize the class separability by finding linear combinations of features that maximize the between-class distance and minimize the within-class distance.

- Use case:
  - PCA is often used for exploratory data analysis, visualization, noise reduction, and data compression. It helps identify the most significant features or patterns in the data.
  - LDA is primarily used for classification tasks and feature extraction when there is a clear distinction between classes. It seeks to enhance the separability of different classes in a dataset.