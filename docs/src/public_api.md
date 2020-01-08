# Public API

This package provides one type, [`DiscreteMeasure`](@ref), which describes a
measure on a finite set for use in Sinkhorn's algorithm and the related
functions. The first step to computing e.g. the Sinkhorn divergence
([`sinkhorn_divergence!`](@ref)) is to construct `DiscreteMeasure`'s describing
the quantities of interest.

```@docs
DiscreteMeasure
```

## Functions

This package provides two functions which act on [`DiscreteMeasure`](@ref)'s
to calculate quantities of interest:

```@docs
OT!
sinkhorn_divergence!
```

## Sinkhorn's algorithm

The above functions rely on [`unbalanced_sinkhorn!`](@ref) which uses Sinkhorn's
algorithm to calculate the optimal [Dual potentials](@ref).

```@docs
unbalanced_sinkhorn!
```

## Divergences

As described on the page [Optimal transport](@ref), we use $\varphi$-divergences
as penalty terms for mass creation and destruction. This package implements four
such divergences, which are all described in Section 2.4 of [[SFVTP19](@ref)],
and are listed below. To add your own divergences, see the [`AbstractDivergence`
interface](@ref) section.

```@docs
UnbalancedOptimalTransport.KL
UnbalancedOptimalTransport.TV
UnbalancedOptimalTransport.Balanced
UnbalancedOptimalTransport.RG
```

### `AbstractDivergence` interface

To add a divergence `MyDivergence`, create a `struct`

```julia
struct MyDivergence <: UnbalancedOptimalTransport.AbstractDivergence end
```

and implement a method for [`UnbalancedOptimalTransport.aprox`](@ref) and
[`UnbalancedOptimalTransport.φstar`](@ref). Optionally, one can also implement a method
for [`UnbalancedOptimalTransport.initialize_dual_potential!`](@ref) and
[`sinkhorn_divergence!`](@ref), as a specialized implementation may obtain
better performance.

```@docs
UnbalancedOptimalTransport.AbstractDivergence
UnbalancedOptimalTransport.aprox
UnbalancedOptimalTransport.φstar
UnbalancedOptimalTransport.initialize_dual_potential!
```

## Utilities

[`sinkhorn_divergence!`](@ref) uses the following function, which may be
specialized to improve performance.

```@docs
UnbalancedOptimalTransport.fdot
```
