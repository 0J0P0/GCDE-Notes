# Constrained optimization

- [Constrained optimization](#constrained-optimization)
  - [First order conditions](#first-order-conditions)
    - [Active constrains](#active-constrains)
    - [Feasibility](#feasibility)
    - [Descent direction](#descent-direction)
  - [Feasible points](#feasible-points)
  - [KKT conditions](#kkt-conditions)
    - [Second order conditions](#second-order-conditions)

$$\min f(x) \\ s.t. \ \ \ h_i(x) = 0 \\  \ \ \  \ \ \  \ \ \ g_j(x) \leq 0$$

## First order conditions

### Active constrains

- Constaraisn $h_i$ are always active at feasible points.

- $g_i = 0$ are active at the optimal point and $g_i < 0$ are inactive.

### Feasibility

Direction $d$ preserves feasibility if $\nabla h(x + d) = 0$ (up to first-order).

$$\nabla h(x)^T d = 0, \ \ h(x + d) = 0 = h(x) + \nabla h(x)^T d$$

Direction $d$ preserves feasibility if $\nabla g(x + d) \approx g(x) + \nabla g(x)^T d \leq 0$ (up to first-order). (or $\geq$)

### Descent direction

Direction decreseas $f$ if $\nabla f(x)^T d < 0$ (up to first-order).

$$\nabla f(x)^T d < 0, \ \ f(x + d) = f(x) + \nabla f(x)^T d  \implies f(x + d) < f(x)$$


**For $g$ active**: If at $x$ there is a direction $d$ at the intersection of the hyper-planes $\nabla g(x)^Td \leq 0$, $\nabla f(x)^T d < 0$, then $x$ is not a minimizer.

- there is no $d$ in the intersection iif:
$$\nabla f(x) + \lambda \nabla g(x) = 0, \ \ \lambda \geq 0$$

**General**: 

$$\nabla_x L(x, \lambda, \mu) = 0 \\ 
\mu \geq 0, \ \lambda \geq 0, \\
\mu_i g_i(x) = 0$$

where the last condition is the complementary condition. thus when $g_i$ is inactive, then $\mu_i = 0$. And when $g_i$ is active, then $\mu_i \geq 0$.

## Feasible points

A feasible point is regular iff:

$$[\nabla h_i(x), \nabla g_j(x)] \ \ \text{are linearly independent} \ \ \forall i, j$$

If x is a regular point, then the tanget plane T is equal to M, where M i:

$$M = \{d \in \mathbb{R}^n \ | \ \nabla h_i(x)^T d = 0, \ \nabla g_j(x)^T d \leq 0 \ \forall i, j\}$$

The plane T is the set of directions that preserve feasibility and decrease f.

## KKT conditions

if x is a regular point and a local minimizer, then there exists $\lambda^*$ and $\mu^*$ such that:

1. $h(x) = 0, g(x) \leq 0$
2. $\nabla_x L(x, \lambda^\*, \mu^\*) = 0$
3. $\mu^\* \geq 0, \ \lambda^\* \geq 0$ and $\mu^\*\_i g_i(x) = 0$ if ($g_i(x) < 0$) then $\mu^\*_i = 0$.


### Second order conditions

**Necessary condition**:

$$\nabla^2_x L(x, \lambda, \mu) \succeq 0$$

with $x$ as a local minimizer and a regular point.

**Sufficient condition**:

$$\nabla^2_x L(x, \lambda, \mu) \succ 0$$

$$
d^T \nabla^2_x L(x, \lambda, \mu) d \succ 0, \ \ \forall c \in M \\
M = \{d \in \mathbb{R}^n \ | \ \nabla h_i(x)^T d = 0, \ \nabla g_j(x)^T d = 0 \ \forall i, j \}
$$


no need for $x$ to be a regular point. $x$ becomes a strict local minimizer.

If we are dealing with a convex problem, the KKT are necessary and sufficient conditions for optimality.

## Duality

From a primal problem, we can derive a dual problem. The dual problem is a lower bound of the primal problem. If the primal problem is convex, then the dual problem is a tight lower bound.

$$
q(\lambda, \mu) = \min_x L(x, \lambda, \mu) \\
\max_{\lambda, \mu} q(\lambda, \mu) \\
s.t. \ \ \ \mu \geq 0
$$

### Concavity

The dual function $q(\lambda, \mu)$ is concave.

### Weak duality theorem

Let $x$ be a feasible point of the primal problem and $(\lambda, \mu)$ be a feasible point of the dual problem. Then:

$$q(\lambda, \mu) \leq f(x)$$

The distnace between the dual and primal is the duality gap.

### Strong duality theorem

If the primal problem is convex and has a feasible point, then the dual problem has a feasible point and the duality gap is zero.

$$q(\lambda, \mu) = f(x)$$

Strong dfuality is satisfied by LP and QP problems. And most of convex problems.

### Wolfe duality theorem

If $f, h_i, g_j$ are convex and differentiable, a necessary and sufficient condition of optimality of the dual function is 

$$\nabla L(x, \lambda, \mu) = 0$$

and thus the dual function can be rewritten as:

$$
\max \ \ \ L(x, \lambda, \mu) \\
s.t. \ \ \ \nabla L(x, \lambda, \mu) = 0 \\
\mu \geq 0
$$

### Dual linear optimization
Primal problem:
$$
\min c^T x \\
s.t. \ \ \ Ax = b \ [\lambda] \\
x \geq 0
$$

Dual problem:
$$
\max b^T \lambda \\
s.t. \ \ \ A^T \lambda = c \\
$$

### Dual quadratic optimization










