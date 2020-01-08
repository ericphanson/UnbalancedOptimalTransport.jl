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
SD = sinkhorn_divergence!(UnbalancedOptimalTransport.KL(1.0), a, b, ϵ; C = cost)
```

The number `SD` provides us with a distance between the `a` and `b` histograms,
as computed by the (unbalanced) Sinkhorn divergence.
