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
        C,
        a::DiscreteMeasure,
        b::DiscreteMeasure,
        ϵ = 1e-1;
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
* `C`: either a function from `a.set` × `b.set` to real numbers; should satisfy
  `C(x,y) = C(y,x)` and `C(x,x)=0` when applicable, or a precomputed cost matrix, see [`cost_matrix`](@ref)
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
    C::AbstractMatrix,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    tol = 1e-5,
    max_iters = 10^5,
    warn::Bool = true,
)
    if D isa Balanced && warn && sum(a.density) ≉ sum(b.density)
        @warn "Should have `sum(a.density) ≈ sum(b.density)` for `D==Balanced()`"
    end

    initialize_dual_potential!(D, a)
    initialize_dual_potential!(D, b)

    f = a.dual_potential
    tmp_f = a.cache

    g = b.dual_potential
    tmp_g = b.cache

    max_residual = Inf
    iters = 0
    while iters < max_iters && max_residual > tol
        iters += 1
        max_residual = 0.0
        @inbounds for j in eachindex(g)
            for i in eachindex(a.log_density, f, tmp_f)
                tmp_f[i] = a.log_density[i] + (f[i] - C[i, j]) / ϵ
            end
            new_g = -ϵ * logsumexp!(tmp_f)
            new_g = -aprox(D, ϵ, -new_g)
            diff = abs(g[j] - new_g)
            if diff > max_residual
                max_residual = diff
            end
            g[j] = new_g
        end
        @inbounds for j in eachindex(f)
            for i in eachindex(b.log_density, g, tmp_g)
                tmp_g[i] = b.log_density[i] + (g[i] - C[j, i]) / ϵ
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

    if warn && iters == max_iters
        @warn "Maximum iterations ($max_iters) reached" max_residual
    end

    return (iters = iters, max_residual = max_residual)
end

"""
    function OT!(
        D::AbstractDivergence,
        C,
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
    C::AbstractMatrix,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    kwargs...,
)

    unbalanced_sinkhorn!(D, C, a, b, ϵ; kwargs...)

    f = a.dual_potential
    g = b.dual_potential

    T = promote_type(eltype(a), eltype(b))
    _nφstar = q -> -φstar(D, -q)

    t1 = fdot(_nφstar, a.density, f)
    t2 = fdot(_nφstar, b.density, g)
    t3 = zero(T)
    for i in eachindex(f), j in eachindex(g)
        t3 -=
            ϵ *
            a.density[i] *
            b.density[j] *
            (exp((f[i] + g[j] - C[i, j]) / ϵ) - one(T))
    end
    return t1 + t2 + t3
end

# Generic method
function _sinkhorn_divergence!(
    D::AbstractDivergence,
    C,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ;
    kwargs...,
)
    OT_aa = OT!(D, C, a, a, ϵ; kwargs...)
    OT_bb = OT!(D, C, b, b, ϵ; kwargs...)
    OT_ab = OT!(D, C, a, b, ϵ; kwargs...)

    OT_ab + (-OT_aa - OT_bb + ϵ * (sum(a.density) - sum(b.density))^2) / 2
end

"""
    sinkhorn_divergence!(
        D::AbstractDivergence,
        C,
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
    C,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    kwargs...,
) = _sinkhorn_divergence!(D, C, a, b, ϵ; kwargs...)

sinkhorn_divergence!(
    D::AbstractDivergence,
    C::AbstractMatrix,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    kwargs...,
) = throw(ArgumentError("Must pass a cost function `C`, not a cost matrix."))

"""
    function optimal_coupling!(
        D::AbstractDivergence,
        C,
        a::DiscreteMeasure,
        b::DiscreteMeasure,
        ϵ = 1e-1;
        dual_potentials_populated::Bool = false,
        kwargs...) -> Matrix

Computes the optimal coupling between `a` and `b` using the optimal dual
potentials, the regularization parameter `ϵ`, and the cost function `C`.

If `dual_potentials_populated = false`, [`unbalanced_sinkhorn!`](@ref) is
called to populate the dual potentials of `a` and `b`, using the divergence `D`.
If `dual_potentials_populated = true`, one of [`unbalanced_sinkhorn!`](@ref)
or [`OT!`](@ref) or [`sinkhorn_divergence!`](@ref) must be called first to
set the optimal dual potentials, with the same choice of `ϵ` and `C`. In this case,
`a` and `b` are not mutated.

This function implements Prop.
6 of [[SFVTP19](@ref)].
"""
function optimal_coupling!(
    D::AbstractDivergence,
    C::AbstractMatrix,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    dual_potentials_populated::Bool = false,
    kwargs...,
)
    if !dual_potentials_populated
        unbalanced_sinkhorn!(D, C, a, b, ϵ; kwargs...)
    end

    f = a.dual_potential
    g = b.dual_potential
    return [
        exp((f[i] + g[j] - C[i, j]) / ϵ) * a.density[i] * b.density[j]
        for i in eachindex(a.density), j in eachindex(b.density)
    ]
end


# Provide default choice of norm as cost function
for fun in [:unbalanced_sinkhorn!, :OT!, :sinkhorn_divergence!, :optimal_coupling!]
    @eval begin
        $fun(
            D::AbstractDivergence,
            a::DiscreteMeasure,
            b::DiscreteMeasure,
            args...;
            kwargs...
        ) = $fun(D, (x,y)->norm(x-y), a, b, args...; kwargs...)
    end
end

# If a function is provided, precompute the cost matrix. sinkhorn_divergence is omitted since it requires three different cost matrices.
for fun in [:unbalanced_sinkhorn!, :OT!, :optimal_coupling!]
    @eval begin
        $fun(
            D::AbstractDivergence,
            C, # Not restricting to Function to allow callable structs etc.
            a::DiscreteMeasure,
            b::DiscreteMeasure,
            args...;
            kwargs...
        ) = $fun(D, cost_matrix(C, a, b), a, b, args...; kwargs...)
    end
end
