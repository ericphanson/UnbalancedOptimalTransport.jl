```@meta
CurrentModule = UnbalancedOptimalTransport
```

# UnbalancedOptimalTransport

This package provides an MIT license, dependency-free implementation of
Algorithm 1 of "Sinkhorn Divergences for Unbalanced Optimal Transport"
[[SFVTP19](@ref)] in [`unbalanced_sinkhorn!`](@ref), in a generic and extensible
way. This is used to compute Sinkhorn divergences via
[`sinkhorn_divergence!`](@ref).

See [Optimal transport](@ref) for some background and a mathematical description
of the quantities computed by this package, [Public API](@ref) for a description
of the functions provided, and below for a quick tutorial.

While the code is generic, it is not currently written to dispatch to BLAS or
non-scalar GPU operations, although such contributions would be welcomed.

## Quick tutorial

```@repl 1
using UnbalancedOptimalTransport, Plots

X = 1:4; # a set
a_weights = [0.5, 1.0, 1.0, 0.5]; # weights on X
Y = 3:5; # another set
b_weights = [0.5, 0.75, 0.5]; # weights on Y
plot(bar(X, a_weights, label="a"), bar(Y, b_weights, label="b"),
        xlims = (0, 6), ylims=(0, 1.2), legend=:topleft);
savefig("histograms.svg"); nothing # hide
```

![](histograms.svg)

We wish to move the `a` histogram to the `b` histogram with the least total
cost, however we will clearly need to remove some mass as well. We choose the
[`KL`](@ref) divergence to penalize mass destruction.

```@repl 1
a = DiscreteMeasure(a_weights, X);
b = DiscreteMeasure(b_weights, Y);
cost = (x, y) -> abs(x - y)
ϵ = 0.01 # small regularization
D = UnbalancedOptimalTransport.KL(1.0)
SD = sinkhorn_divergence!(D, cost, a, b, ϵ)
```

The number `SD` provides us with a distance between the `a` and `b` histograms,
as computed by the (unbalanced) Sinkhorn divergence. We can also compute the
"optimal coupling" which shows us how to move between the histograms.

```@repl 1
π = optimal_coupling!(D, cost, a, b, ϵ)
```

Here, `π[x,y]` represents how much mass we should move from `x` to `y`. Since
the sum of the rows is less than the corresponding entries of `a_weight`, some
of the mass is destroyed. To understand this better, let us lower the penalty
for mass destruction and mass creation to almost nothing, and see how the
optimal coupling changes.

```@repl 1
D = UnbalancedOptimalTransport.KL(0.01)
π = optimal_coupling!(D, cost, a, b, ϵ)
```

We can see that the only non-tiny entries of `π` are the `(3,1)` and `(4,2)`,
corresponding to `(x=3, y=3)`, as the first element of $Y$ is $3$. We see then
with this choice of divergence and cost, we don't really move any mass, and just
create and destroy as needed. On the other hand, let us see what happens when
there is a high penalty for mass creation and destruction:

```@repl 1
D = UnbalancedOptimalTransport.KL(1000.0)
sinkhorn_divergence!(D, cost, a, b, ϵ)
π = optimal_coupling!(D, cost, a, b, ϵ)
```

We see warnings about the maximum number of
iterations being exceeded, so let's increase that parameter and try again. Note
that warnings can be disabled by passing `warn=false` as a keyword argument.

```@repl 1
sinkhorn_divergence!(D, cost, a, b, ϵ; max_iters = 10^6)
π = optimal_coupling!(D, cost, a, b, ϵ; max_iters = 10^6)
```

We see that now we move some mass from each each element of `X` to each element
of `Y`, to try to avoid needing to create or destroy mass, except no mass is
moved from `x=4` to `y=3`, presumably because it's better to move it to `y=4`
and `y=5` instead.
