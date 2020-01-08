"""
    initialize_dual_potential!(::AbstractDivergence, a::DiscreteMeasure) -> Nothing

Apply an initialization for the dual potential, for use in
[`unbalanced_sinkhorn!`](@ref); falls back to zeroing out the dual potential.
Specialized implementations can improve performance, but should not affect
correctness.
"""
function initialize_dual_potential!(::AbstractDivergence, a::DiscreteMeasure)
    a.dual_potential .= 0
end

"""
    function unbalanced_sinkhorn!(
        D::AbstractDivergence,
        a::DiscreteMeasure,
        b::DiscreteMeasure,
        ϵ = 1e-1;
        C = (x, y) -> norm(x - y),
        tol = 1e-5,
        max_iters = 10^5,
        warn::Bool = true,
    ) -> NamedTuple

Implements algorithm 1 of [[SFVTP19](@ref)]. The `dual_potential` fields of `a`
and `b` are updated to hold the optimal dual potentials. The `density`,
`log_density`, and `set` fields are not modified. The parameters are

* `D`: the [`AbstractDivergence`](@ref) used for measuring the cost of
  creating or destroying mass
* `ϵ`: the regularization parameter
* `C`: a function from `a.set` × `b.set` to real numbers; should satisfy
  `C(x,y) = C(y,x)` and `C(x,x)=0` when applicable
* `tol`: the convergence tolerance
* `max_iters`: the maximum number of iterations to perform.
* `warn`: whether or not to warn when the maximum number of iterations is reached.

Returns a NamedTuple of the number of iterations performed (`iters`), and the
maximum residual (`max_residual`), which is the maximum infinity norm difference
between consecutive iterates of the dual potentials, at the end of the process.
If `max_iters` is not reached, iteration stops when the `max_residual` falls
below `tol`.

"""
function unbalanced_sinkhorn!(
    D::AbstractDivergence,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    C = (x, y) -> norm(x - y),
    tol = 1e-5,
    max_iters = 10^5,
    warn::Bool = true,
)
    if D isa Balanced && warn && sum(a.density) ≉ sum(b.density)
        @warn "Should have `sum(a.density) ≈ sum(b.density)` for `D==Balanced()`"
        error("$(sum(a.density) - sum(b.density))")
    end

    initialize_dual_potential!(D, a)
    initialize_dual_potential!(D, b)

    x = a.set
    f = a.dual_potential
    tmp_f = a.cache

    y = b.set
    g = b.dual_potential
    tmp_g = b.cache

    max_residual = Inf
    iters = 0
    while iters < max_iters && max_residual > tol
        iters += 1
        max_residual = 0.0
        @inbounds for j in eachindex(y)
            for i in eachindex(a.log_density, f, tmp_f, x)
                tmp_f[i] = a.log_density[i] + (f[i] - C(x[i], y[j])) / ϵ
            end
            new_g = -ϵ * logsumexp!(tmp_f)
            new_g = -aprox(D, ϵ, -new_g)
            diff = abs(g[j] - new_g)
            if diff > max_residual
                max_residual = diff
            end
            g[j] = new_g
        end
        @inbounds for j in eachindex(x)
            for i in eachindex(b.log_density, g, tmp_g, y)
                tmp_g[i] = b.log_density[i] + (g[i] - C(x[j], y[i])) / ϵ
            end
            new_f = -ϵ * logsumexp!(tmp_g)
            new_f = -aprox(D, ϵ, -new_f)
            diff = abs(f[j] - new_f)
            if diff > max_residual
                max_residual = diff
            end
            f[j] = new_f
        end
    end

    iters == max_iters && error("$(D), $(eltype(a))")
    if warn && iters == max_iters
        @warn "Maximum iterations ($max_iters) reached" max_residual
    end

    return (iters = iters, max_residual = max_residual)
end

"""
    function OT!(
        D::AbstractDivergence,
        a::DiscreteMeasure,
        b::DiscreteMeasure,
        ϵ = 1e-1;
        C = (x, y) -> norm(x - y),
        kwargs...,
    ) -> Number

Computes the optimal transport cost between `a` and `b`, using
[`unbalanced_sinkhorn!`](@ref); see that function for the meaning of the
parameters and the keyword arguments. Implements Equation (15) of
[[SFVTP19](@ref)].
"""
function OT!(
    D::AbstractDivergence,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    C = (x, y) -> norm(x - y),
    kwargs...,
)
    unbalanced_sinkhorn!(D, a, b, ϵ; C = C, kwargs...)

    f = a.dual_potential
    g = b.dual_potential
    x = a.set
    y = b.set

    T = promote_type(eltype(a), eltype(b))
    _nφstar = q -> -φstar(D, -q)

    t1 = fdot(_nφstar, a.density, f)
    t2 = fdot(_nφstar, b.density, g)
    t3 = zero(T)
    for i in eachindex(x), j in eachindex(y)
        t3 -=
            ϵ *
            a.density[i] *
            b.density[j] *
            (exp((f[i] + g[j] - C(x[i], y[j])) / ϵ) - one(T))
    end
    return t1 + t2 + t3
end

# Generic method
function _sinkhorn_divergence!(
    D::AbstractDivergence,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ;
    kwargs...,
)
    OT_aa = OT!(D, a, a, ϵ; kwargs...)
    OT_bb = OT!(D, b, b, ϵ; kwargs...)
    OT_ab = OT!(D, a, b, ϵ; kwargs...)

    OT_ab + (-OT_aa - OT_bb + ϵ * (sum(a.density) - sum(b.density))^2) / 2
end

"""
    sinkhorn_divergence!(
        D::AbstractDivergence,
        a::DiscreteMeasure,
        b::DiscreteMeasure,
        ϵ = 1e-1;
        kwargs...,
    ) -> Number

Computes the unbalanced sinkhorn divergence between `a` and `b` as defined in
Def. 6 of [[SFVTP19](@ref)], using [`unbalanced_sinkhorn!`](@ref); see that
function for the meaning of the parameters and the keyword arguments. Sets the
optimal `dual_potential`'s of `a` and `b`.
"""
sinkhorn_divergence!(
    D::AbstractDivergence,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    kwargs...,
) = _sinkhorn_divergence!(D, a, b, ϵ; kwargs...)

"""
    function optimal_coupling(
        a::DiscreteMeasure,
        b::DiscreteMeasure,
        ϵ=1e-1;
        C = (x, y) -> norm(x - y),
    ) -> Matrix

Computes the optimal coupling between `a` and `b` using the optimal dual
potentials, the regularization parameter `ϵ`, and the cost function `C`. One of
[`unbalanced_sinkhorn!`](@ref) or [`OT!`](@ref) or
[`sinkhorn_divergence!`](@ref) must be called first to set the optimal dual
potentials, with the same choice of `ϵ` and `C`. This function implements Prop.
6 of [[SFVTP19](@ref)].
"""
function optimal_coupling(
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    C = (x, y) -> norm(x - y),
)
    f = a.dual_potential
    g = b.dual_potential
    x = a.set
    y = b.set
    return [
        exp((f[i] + g[j] - C(x[i], y[j])) / ϵ)
        for i in eachindex(a.density), j in eachindex(b.density)
    ]
end
