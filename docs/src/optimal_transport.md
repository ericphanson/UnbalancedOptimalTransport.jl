# Optimal transport

Optimal transport is a theory that provides a way to describe distances between
positive measures in terms of distances on their underlying domains. The
following is a short description of the task; for a more thorough review, see
[Wikipedia](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)),
or some of the [Other references](@ref).


In this package, we only consider finite sets, and so a "positive measure" is
simply a collection of non-negative numbers. In the case that these numbers sum
to one, we can consider them as probabilities, although we don't make that
restriction here. Let $X$ and $Y$ be finite sets (the only kind considered in
this package), and $a$ and $b$ be functions,

```math
a : X \to \mathbb{R}_+ \quad \text{and} \quad b: Y \to \mathbb{R}_+.
```

We assume $a(x) > 0$ for each $x \in X$ and likewise $b(y) > 0$ for $y \in Y$;
zero elements can simply be removed from $X$ or $Y$.

The associated *optimal transport problem* is the following. We start with $a(x)$
"mass" of material at each point $x \in X$, and wish to end up with $b(y)$ mass
of material at each point $y \in Y$, but there is some cost $c(x,y)$ associated
to moving a unit of mass from $x$ to $y$.

For each pair $(x,y) \in X\times Y$, we aim to decide how much mass to move from
$x$ to $y$ by choosing a *coupling*, a function $π : X \times Y \to
\mathbb{R}_+$, which minimizes the cost:

```math
\begin{aligned}
\operatorname{OT}(a,b) := \text{minimize} \quad & \sum_{x\in X, y \in Y} π(x,y) c(x,y)\\
\text{such that} \quad & a(x) = \sum_{y \in Y} \pi(x,y) \quad \forall x \in X\\
& b(y) = \sum_{x \in X} \pi(x,y) \quad, \forall y \in Y\\
& \pi(x,y) \geq 0 \quad \forall x\in X,\, \forall y \in Y.
\end{aligned}
```

The constraint $a(x) = \sum_{y \in Y} \pi(x,y)$ means that $a(x)$ units of mass
was transported from location $x$, and $b(y) = \sum_{x \in X} \pi(x,y)$ means
that $b(y)$ units of mass ended up at location $y$, as desired.

The problem as written, however, has no solution (is *infeasible*), however, if
$\sum_{x\in X} a(x) \neq \sum_{y \in Y}b(y)$, as can be verified by the fact
that if a solution $\pi$ existed, then the constraints imply that

```math
\sum_{y \in Y} b(y) = \sum_{x\in X, y \in Y} \pi(x,y) = \sum_{x \in X} a(x).
```

which is a contradiction. That's because our problem as posed does not allow the
creation or destruction of mass.

To resolve this, following [[SFVTP19](@ref)], we remove the requirements that
$a(x) = \sum_{y \in Y} \pi(x,y)$ and $b(y) = \sum_{x \in X} \pi(x,y)$, and
instead add a penalty that grows the more those constraints are violated (i.e.
imposing "soft" constraints instead of "hard" constraints).

We model these penalties as a $\varphi$-divergence $D_\varphi$, which are
defined in terms of a function $\varphi : \mathbb{R}_+ \to \mathbb{R}_+$ as

```math
D_\varphi(u \| v) := \sum_{z \in Z} \varphi( \frac{u(z)}{v(z)} ) b(z)
```

where $Z$ is a finite set, and $u : Z \to \mathbb{R}_+$ and $v : Z \to
\mathbb{R}_+$ as functions with $v(z) > 0$ for each $z \in Z$. See
[Divergences](@ref) for the list of divergences implemented in this package.

With that modification, we have the *unbalanced optimal transport problem*,

```math
\begin{aligned}
\operatorname{OT}(D_\varphi, a,b) := \text{minimize} \quad &
\sum_{x\in X, y \in Y} π(x,y) c(x,y) + D_\varphi(\pi_1 \| a) + D_\varphi(\pi_2 \| b)\\
\text{such that} \quad & \pi(x,y) \geq 0 \quad \forall x\in X,\, \forall y \in Y.
\end{aligned}
```

where $\pi_1$ is defined by $\pi_1(x) = \sum_{y \in Y} \pi(x,y)$ for each $x \in
X$, and likewise $\pi_2(y)=\sum_{x \in X} \pi(x,y)$ for $y \in Y$.

## Entropic regularization

We are faced with a problem: $\operatorname{OT}(D_\varphi, a,b)$ can be
difficult to compute, especially when $X$ and $Y$ are large. It turns out that
$\operatorname{OT}(D_\varphi, a,b)$ can be described as the limit of a sequence
of problems that are easier to solve.

For $\varepsilon \geq 0$, consider the *entropically-regularized unbalanced
optimal transport problem*,

```math
\begin{aligned}
\operatorname{OT}(D_\varphi, a,b, \varepsilon) := \text{minimize} \quad &
    \sum_{x\in X, y \in Y} π(x,y) c(x,y) + D_\varphi(\pi_1 \| a)
    + D_\varphi(\pi_2 \| b) + \varepsilon \operatorname{KL}(\pi \| a \otimes b)\\
\text{such that} \quad & \pi(x,y) \geq 0 \quad \forall x\in X,\, \forall y \in Y.
\end{aligned}
```

where $\operatorname{KL}$ is the *Kullback Leibler divergence* (also called the
*relative entropy*), which in fact is the $\varphi$-divergence with $\varphi(x)
= x\log x - x + 1$, and $a \otimes b : X \times Y \to \mathbb{R}_+$ is defined
by $(a \otimes b)(x,y) := a(x)b(y)$.

Entropic regularization refers to the additional term $\varepsilon
\operatorname{KL}(\pi \| a \otimes b)$. Adding this term yields several
advantages; see e.g. Chapter 4 of [PC18] in [Other references](@ref) for
entropic regularization in optimal transport in general, and [[SFVTP19](@ref)]
in particular for the unbalanced case. As concerns this package, however, the
most important advantage is that $\operatorname{OT}(D_\varphi, a,b,
\varepsilon)$ can be computed quickly by an iterative scheme called *Sinkhorn's
algorithm*, which is implemented in the function [`unbalanced_sinkhorn!`](@ref),
based on Algorithm 1 of [[SFVTP19](@ref)]. This is used to compute the quantity
$\operatorname{OT}(D_\varphi, a,b, \varepsilon)$ in the function [`OT!`](@ref).

Note that this package also implements the optimization problem form of
$\operatorname{OT}(D_\varphi, a, b, \varepsilon)$, using the modelling language
[Convex.jl](https://github.com/JuliaOpt/Convex.jl) and the solvers
[SCS.jl](https://github.com/JuliaOpt/SCS.jl) and
[ECOS.jl](https://github.com/JuliaOpt/ECOS.jl), in `test/Convex_formulation.jl`,
in order to test the implementation of Sinkhorn's algorithm in
[`unbalanced_sinkhorn!`](@ref). This functionality is contained in the tests
rather than the package itself in order to avoid the extra dependencies, and
since solving the optimization problem is much less practical than using
Sinkhorn's algorithm.

One problem raised by the regularization is that

```math
\operatorname{OT}(D_\varphi,a,a,\varepsilon) \neq 0
```

in general. This is not ideal when one wishes to consider $(a,b) \mapsto
\operatorname{OT}(D_\varphi,a,b,\varepsilon)$ as a measure of distance between
$a$ and $b$. To remedy this, one defines the *Sinkhorn divergence*

```math
S(D_\varphi,a,b,\varepsilon) := \operatorname{OT}(D_\varphi,a,b,\varepsilon)
    - \frac{1}{2}\operatorname{OT}(D_\varphi,a,a,\varepsilon)
    - \frac{1}{2}\operatorname{OT}(D_\varphi,b, b,\varepsilon)
    + \frac{\epsilon}{2}(m(a) - m(b))^2
```

where e.g. $m(a) = \sum_{x\in X} a(x)$ is the total mass of $a$ (this is Def. 6
of [[SFVTP19](@ref)]). This quantity is computed by [`sinkhorn_divergence!`](@ref).


When $D_\varphi$ is chosen as proportional to $\operatorname{KL}$, this quantity
has the following useful properties, shown in Theorem 4 of [[SFVTP19](@ref)]:

* Positive-definiteness: $S(D_\varphi,a,b,\varepsilon) \geq 0$ with equality if
  and only if $a=b$
* Convexity in both arguments: $a \mapsto S(D_\varphi,a,b,\varepsilon)$ is
  convex, and $b \mapsto S(D_\varphi,a,b,\varepsilon)$ is convex.

## Dual potentials

The quantity $\operatorname{OT}(D_\varphi, a,b, \varepsilon)$ can be
equivalently described by the so-called *dual problem*,

```math
\begin{aligned}
\operatorname{OT}(D_\varphi, a,b, \varepsilon) = \text{maximize} \quad
& -\sum_{x\in X} a(x) \varphi^*(-f(x)) -\sum_{y\in Y} b(y) \varphi^*(-g(x))\\
    & \quad - \varepsilon \sum_{x\in X, y \in Y} a(x)b(y)
    \left(\exp( \frac{f(x) + g(y) - c(x,y)}{\varepsilon} ) -1\right) \\
\text{where} \quad & f : X \to \mathbb{R}, \, g : Y \to \mathbb{R}
\end{aligned}
```

where $\varphi^*: \mathbb{R} \to \mathbb{R}$ is the Legendre transform of
$\varphi$, defined as

```math
\varphi^*(z) = \sup_{x \geq 0} xz - \varphi(x).
```

The functions $f$ and $g$ are called the *dual potentials* (in
[[SFVTP19](@ref)], at least). Sinkhorn's algorithm as implemented in
[`unbalanced_sinkhorn!`](@ref) computes the optimal dual potentials, namely
those which achieve the maximum in the optimization problem above.


## References
### SFVTP19

[SFVTP19] Séjourné, T., Feydy, J., Vialard, F.-X., Trouvé, A., Peyré, G., 2019.
*Sinkhorn Divergences for Unbalanced Optimal Transport*.
[arXiv:1910.12958](https://arxiv.org/abs/1910.12958).

### Other references

[PC18] Peyré, G., Cuturi, M., 2018. *Computational Optimal Transport*.
[arXiv:1803.00567](https://arxiv.org/abs/1803.00567).

[Villani09] Villani, C., 2009. *Optimal Transport: Old and New*, Grundlehren der
mathematischen Wissenschaften. Springer-Verlag, Berlin Heidelberg.
