using Convex, LinearAlgebra
using UnbalancedOptimalTransport: AbstractDivergence, KL, TV, RG, Balanced

"""
    divergence(D::AbstractDivergence, a, b)

Computes the the divergence `D` between `a` and `b`, where `a` can be a
Convex.jl expression, and `b` is a constant.
"""
function divergence end

lg(x) = x <= 0 ? zero(x) : log(x)

# need to type-pirate a method for non-Convex.jl objects
Convex.entropy(a::AbstractArray) = sum(x -> -x * lg(x), a)

function divergence(::KL{ρ}, a, b) where {ρ}
    ρ * (-entropy(a) - dot(a, lg.(b)) + sum(b - a))
end

function divergence(::TV{ρ}, a, b) where {ρ}
    ρ * norm(a - b, 1)
end

# Computes the objective of the unbalanced regularized optimal transport problem
# This is a separate method so it can be reused for testing `optimal_coupling`.
function objective(
    π,
    D::AbstractDivergence,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ;
    C = (x, y) -> norm(x - y),
)
    π_1 = sum(π, dims = 2)
    π_2 = vec(sum(π, dims = 1))

    # instantiate the cost matrix
    C_matrix = cost_matrix(C, a, b)

    obj = dot(C_matrix, π) + ϵ * divergence(KL(), vec(π), kron(b.density, a.density))

    if (D isa KL) || (D isa TV)
        obj += divergence(D, π_1, a.density) + divergence(D, π_2, b.density)
    end

    return obj
end

function OT_convex(
    D::AbstractDivergence,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ;
    solver,
    verbose = false,
    C = (x, y) -> norm(x - y),
)
    ϵ >= 0 || throw(DomainError(ϵ, "Need ϵ >= 0"))

    # coupling and its marginals
    π = Variable(length(a), length(b))
    π_1 = sum(π, dims = 2)
    π_2 = vec(sum(π, dims = 1))

    obj = objective(π, D, a, b, ϵ; C = C)

    if D isa Balanced
        constraints = [π >= 0, π_1 == a.density, π_2 == b.density]
    elseif D isa RG
        l, u = params(D)
        constraints = [
            π >= 0,
            l * a.density <= π_1,
            π_1 <= u * a.density,
            l * b.density <= π_2,
            π_2 <= u * b.density,
        ]
    else
        constraints = [π >= 0]
    end

    problem = minimize(obj, constraints)

    solve!(problem, solver; verbose = verbose)

    return (optimal_value = problem.optval, optimal_coupling = evaluate(π))
end

params(::RG{l,u}) where {l,u} = (l, u)
