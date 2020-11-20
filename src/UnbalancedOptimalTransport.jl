"""
This package provides an MIT license, dependency-free implementation of
Algorithm 1 of "Sinkhorn Divergences for Unbalanced Optimal Transport" [SFVTP19
(http://arxiv.org/abs/1910.12958)], which allows calculation of the optimal
transport and Sinkhorn divergence between two positive measures (with possibly
different total mass), where mass creation and destruction is penalized by one
of several possible φ-divergences.
"""
module UnbalancedOptimalTransport

using LinearAlgebra: norm

export DiscreteMeasure, SignedMeasure,
    OT!, optimal_coupling!, sinkhorn_divergence!, unbalanced_sinkhorn!, cost_matrix

struct DiscreteMeasure{P,LP,S,T}
    density::P
    log_density::LP
    set::S
    dual_potential::Vector{T}
    cache::Vector{T}
end

"""
    DiscreteMeasure(density, [log_density], set) -> DiscreteMeasure

Construct a `DiscreteMeasure` object for use in [`unbalanced_sinkhorn!`](@ref)
and related functions.

* `density` should be strictly positive; zero elements should instead be removed
  from `set`
* `log_density` should be equal to `log.(density)` and can be omitted
  (in which case its computed automatically)
* `set` is a collection so that `density[i]` is the probability of the element
  `set[i]` occurring (where `i ∈ eachindex(density, set)`).

"""
function DiscreteMeasure(density::P, log_density::LP, set::S) where {P,LP,S}
    T = eltype(density)
    n = length(density)
    n == length(log_density) ||
    throw(ArgumentError("`density`, `log_density`, and `set` (if supplied) must have equal length"))
    set !== nothing &&
    length(set) != n &&
    throw(ArgumentError("`density`, `log_density`, and `set` must have equal length"))
    dual_potential = zeros(T, n)
    cache = similar(dual_potential)
    DiscreteMeasure{P,LP,S,T}(density, log_density, set, dual_potential, cache)
end

DiscreteMeasure(density, set = nothing) = DiscreteMeasure(density, log.(density), set)

Base.eltype(::DiscreteMeasure{P,LP,S,T}) where {P,LP,S,T} = T
Base.length(a::DiscreteMeasure) = length(a.density)

include("divergences.jl")
include("sinkhorn.jl")
include("optimized_methods.jl")
include("utilities.jl")

struct SignedMeasure{P, N}
    pos::DiscreteMeasure{P}
    neg::DiscreteMeasure{N}
end

function SignedMeasure(density)
    pos_inds = findall(>(0), density)
    pos = DiscreteMeasure(density[pos_inds], pos_inds)
    neg_inds = findall(<(0), density)
    neg = DiscreteMeasure(-1 * density[neg_inds], neg_inds)
    return SignedMeasure(pos, neg)
end

function Base.:(+)(a::DiscreteMeasure, b::DiscreteMeasure)
    T_set = promote_type(eltype(a.set), eltype(b.set))
    T_density =  promote_type(eltype(a.density), eltype(b.density))
    set_to_density = Dict{T_set, T_density}()

    for (i, x) in enumerate(a.set)
        v = get(set_to_density, x, zero(T_density))
        set_to_density[x] = v + a.density[i]
    end

    for (i, x) in enumerate(b.set)
        v = get(set_to_density, x, zero(T_density))
        set_to_density[x] = v + b.density[i]
    end
    
    set = collect(keys(set_to_density))
    density = collect(values(set_to_density))
    return DiscreteMeasure(density, set)
end

export make_measures
function make_measures(a::SignedMeasure, b::SignedMeasure)
    return a.pos + b.neg, b.pos + a.neg
end

end # module
