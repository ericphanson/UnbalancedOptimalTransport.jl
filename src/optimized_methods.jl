# The formulas for `initialize_dual_potential!` are in Section 6.1.2 of SFVTP19.
function initialize_dual_potential!(::KL{ρ}, a::DiscreteMeasure) where {ρ}
    c = -ρ * log(length(a.log_density))
    a.dual_potential .= c
end

function initialize_dual_potential!(::TV{ρ}, a::DiscreteMeasure) where {ρ}
    c = -ρ * sign(log(length(a.log_density)))
    a.dual_potential .= c
end

# Specialized implementation for the KL-divergence
# Prop. 12 of SFVTP19.
function sinkhorn_divergence!(
    D::KL{ρ},
    C,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    kwargs...,
) where {ρ}
    f = a.dual_potential
    g = b.dual_potential
    scaled_exp = z -> exp(-z / ρ)
    unbalanced_sinkhorn!(D, C, a, a, ϵ; kwargs...)
    term_aa = fdot(scaled_exp, a.density, f)
    unbalanced_sinkhorn!(D, C, b, b, ϵ; kwargs...)
    term_bb = fdot(scaled_exp, b.density, g)
    unbalanced_sinkhorn!(D, C, a, b, ϵ; kwargs...)
    term_ab = fdot(scaled_exp, a.density, f) + fdot(scaled_exp, b.density, g)
    return -(ρ + ϵ / 2) * (term_ab - term_aa - term_bb)
end

# Needed to avoid ambiguity errors
sinkhorn_divergence!(
    D::KL{ρ},
    C::AbstractMatrix,
    a::DiscreteMeasure,
    b::DiscreteMeasure,
    ϵ = 1e-1;
    kwargs...,
) where {ρ} = throw(ArgumentError("Must pass a cost function `C`, not a cost matrix."))
