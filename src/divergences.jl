"""
    abstract type AbstractDivergence

An abstract type representing Csiszár φ-divergences. Subtypes should implement
[`φstar`](@ref) and [`aprox`](@ref), and optionally can implement
[`initialize_dual_potential!`](@ref) and/or [`sinkhorn_divergence!`](@ref).
"""
abstract type AbstractDivergence end

"""
    φstar(::AbstractDivergence, q::Number) -> Number

The Legendre conjugate of the function `φ` associated to the divergence.
"""
function φstar end

"""
    aprox(::AbstractDivergence, ϵ::Number, x::Number) -> Number

The anisotropic proximity operator defined in Def. 2 of [[SFVTP19](@ref)].
"""
function aprox end


# The following formulas for `φstar` and `aprox` are found in Section 2.4 of SFVTP19.

"""
    Balanced <: AbstractDivergence

Represents the divergence `Dᵩ(a|b)` which is zero if `a==b` and infinite
otherwise. Generalized by [`RG`](@ref).
"""
struct Balanced <: AbstractDivergence end
φstar(::Balanced, q) = q
aprox(::Balanced, ϵ, x) = x

"""
    KL{ρ} <: AbstractDivergence

Represents the divergence `ρ*KL(a|b)`, where `KL` is the Kullback-Leibler
divergence. The parameter `ρ` is simply a scaling.
"""
struct KL{ρ} <: AbstractDivergence end
KL(ρ::Number) = KL{ρ}()
KL() = KL(1.0)

φstar(::KL{ρ}, q) where {ρ} = ρ * (exp(q / ρ) - 1)

aprox(::KL{ρ}, ϵ, x) where {ρ} = inv(one(ρ) + ϵ / ρ) * x


"""
    RG{l,u} <: AbstractDivergence

Represents the divergence `Dᵩ(a|b)` which is zero if `l*b .<= a .<= u*b` and
infinite otherwise. Equivalent to [`Balanced`](@ref) when `l == u`.
"""
struct RG{l,u} <: AbstractDivergence end

function RG(l::Number, u::Number)
    l <= u || throw(DomainError(u - l, "Need l <= u"))
    RG{l,u}()
end

φstar(::RG{a,b}, q) where {a,b} = max(a * q, b * q)

function aprox(::RG{a,b}, ϵ, x) where {a,b}
    if (s = x - ϵ * log(a)) < 0
        s
    elseif (t = x - ϵ * log(b)) > 0
        t
    else
        0
    end
end


"""
    TV{ρ} <: AbstractDivergence

Represents the divergence `ρ*TV(u,v) = ρ*norm(u-v,1)`, where `TV` is twice the
total variation distance. The parameter `ρ` is simply a scaling.
"""
struct TV{ρ} <: AbstractDivergence end
TV(ρ::Number) = TV{ρ}()
TV() = TV(1.0)

φstar(::TV{ρ}, q) where {ρ} =
    q <= ρ ? max(-ρ, q) : throw(DomainError(q, "Must have q <= ρ"))

function aprox(::TV{ρ}, ϵ, x) where {ρ}
    if -ρ <= x <= ρ
        x
    elseif x < -ρ
        -ρ
    else
        ρ
    end
end
