using UnbalancedOptimalTransport
using UnbalancedOptimalTransport: KL, TV, RG, Balanced
using Test
using SCS, ECOS
using StaticArrays
using Convex: MOI

include("Convex_formulation.jl")

const TEST_DIVERGENCES = (KL(), KL(2.0), KL(100.0), TV(4.0), RG(0.5, 1.5), Balanced())

function rand_measure(
    n,
    d;
    scale = 100,
    static = false,
    prob_type = Float64,
    set_type = Float64,
)
    if static
        make_vector =
            () -> d == 1 ? Scalar{set_type}(randn()) : SVector{d,set_type}(randn(d))
    else
        make_vector = () -> d == 1 ? [set_type(randn())] : set_type.(randn(d))
    end
    DiscreteMeasure(rand(prob_type, n), [scale * make_vector() for _ = 1:n])
end

function balanced_measures(n1, n2, d; kwargs...)
    a = rand_measure(n1, d; kwargs...)
    b = rand_measure(n2, d; kwargs...)
    b = DiscreteMeasure(sum(a.density) * b.density / sum(b.density), b.set)
    return a, b
end

@testset "UnbalancedOptimalTransport.jl" begin

    # We check the implementation of the Sinkhorn algorithm against its definition as an optimization problem.
    @testset "regularized OT! with balancing via $D, ϵ=$ϵ, same=$same" for D in TEST_DIVERGENCES, ϵ ∈ (0.1, 1.0), same in (false, true)
        # We choose very small test problems, since solving the convex problem is relatively slow
        # Ensure the problem is feasible
        if D isa Balanced
            a, b = balanced_measures(4, 3, 1; scale = 100.0)
        else
            a = rand_measure(4, 1; scale = 100.0)
            b = rand_measure(3, 1; scale = 100.0)
        end

        if D isa RG
            sc = rand() + 0.5
            b = DiscreteMeasure(sc * sum(a.density) * b.density / sum(b.density), b.set)
        end

        # Test if aliasing is a problem
        if same
            # b = deepcopy(a)
            b = a
        end

        # ECOS natively supports the exponential cone we need for the KL divergence
        if D isa TV
            solver = MOI.OptimizerWithAttributes(
                SCS.Optimizer,
                "verbose" => 0,
                "eps_rel" => 1e-6,
                "eps_abs" => 1e-6,
                "max_iters" => 20_000
            )
        else
            solver = MOI.OptimizerWithAttributes(
                ECOS.Optimizer,
                "verbose" => 0,
                "eps_rel" => 1e-6,
                "eps_abs" => 1e-6,
            )
        end

        result = OT_convex(D, a, b, ϵ; solver = solver, verbose = true)

        @test OT!(D, a, b, ϵ) ≈ result.optimal_value rtol = 1e-3 atol = 1e-3

        if same
            # Also test against ourselves but `deepcopy`'ied
            @test OT!(D, a, b, ϵ) ≈ OT!(D, a, deepcopy(b), ϵ) rtol = 1e-3 atol = 1e-3
        end

        coupling_sinkhorn = optimal_coupling!(D, a, b, ϵ)
        @test result.optimal_coupling ≈ coupling_sinkhorn rtol = 1e-3 atol = 1e-3
    end

    @testset "Optimal coupling" begin
        for ϵ ∈ (0.1, 1.0), D in TEST_DIVERGENCES
            if D isa Balanced
                a, b = balanced_measures(4, 3, 1; scale = 100.0)
            else
                a = rand_measure(4, 1; scale = 100.0)
                b = rand_measure(3, 1; scale = 100.0)
            end

            if D isa RG
                sc = rand() + 0.5
                b = DiscreteMeasure(sc * sum(a.density) * b.density / sum(b.density), b.set)
            end

            aa = deepcopy(a)
            bb = deepcopy(b)
            OT_val = OT!(D, aa, bb, ϵ)

            # Try both with `dual_potentials_populated` true and false
            π2 = optimal_coupling!(D, aa, bb, ϵ; dual_potentials_populated = true)
            π = optimal_coupling!(D, a, b, ϵ)
            @test π ≈ π2

            obj = objective(π, D, a, b, ϵ)
            @test OT_val ≈ obj rtol = 1e-3
        end
    end

    @testset "Allocations" begin
        a = rand_measure(2, 2; static = true)
        b = rand_measure(2, 2; static = true)
        C = cost_matrix(a, b)
        OT!(KL(), C, a, b) # compile

        a = rand_measure(100, 2; static = true)
        b = rand_measure(80, 2; static = true)
        C = cost_matrix(a, b)
        @test @allocated(OT!(KL(), C, a, b)) <= 500

        a = rand_measure(1000, 2; static = true)
        b = rand_measure(800, 2; static = true)
        C = cost_matrix(a, b)
        @test @allocated(OT!(KL(), C, a, b)) <= 500
    end

    @testset "Prop. 12: Optimized KL-Sinkhorn divergence method" begin
        for ρ ∈ (0.5, 2.0),
            ϵ ∈ (0.1, 1.0, 10.0),
            C ∈ ((x, y) -> norm(x - y, 3.0), (x, y) -> norm(x - y))

            a = rand_measure(100, 2; static = true)
            b = rand_measure(80, 2; static = true)
            @test sinkhorn_divergence!(KL(ρ), C, a, b, ϵ; tol = 1e-6) ≈
                  UnbalancedOptimalTransport.generic_sinkhorn_divergence!(
                KL(ρ),
                C,
                a,
                b,
                ϵ;
                tol = 1e-6,
            ) rtol = 1e-4

            # Test default argument for `C`
            @test sinkhorn_divergence!(KL(ρ), a, b, ϵ; tol = 1e-6) ≈
                  UnbalancedOptimalTransport.generic_sinkhorn_divergence!(
                KL(ρ),
                (x, y) -> norm(x - y),
                a,
                b,
                ϵ;
                tol = 1e-6,
            ) rtol = 1e-4
        end
    end

    @testset "BigFloats" begin
        ϵ = 1.0
        ϵ_big = big(ϵ)
        for D in TEST_DIVERGENCES
            if D isa Balanced
                a, b = balanced_measures(8, 9, 1; static = true, scale = 2.0)
            else
                a = rand_measure(8, 1; static = true, scale = 2.0)
                b = rand_measure(9, 1; static = true, scale = 2.0)
            end
            a_set_big = [Scalar{BigFloat}(x.data) for x in a.set]
            a_big = DiscreteMeasure(big.(a.density), a_set_big)
            b_set_big = [Scalar{BigFloat}(x.data) for x in b.set]
            b_big = DiscreteMeasure(big.(b.density), b_set_big)
            if D isa Balanced
                b_big = DiscreteMeasure(
                    sum(a_big.density) * b_big.density / sum(b_big.density),
                    b_big.set,
                )
            end

            sd = @inferred(sinkhorn_divergence!(D, a, b, ϵ; max_iters = 10^6))
            sd_big =
                @inferred(sinkhorn_divergence!(D, a_big, b_big, ϵ_big; max_iters = 10^6))
            @test sd_big isa BigFloat
            @test sd ≈ sd_big
        end
    end

    @testset "Lemma 11: equality of dual potentials" begin
        for D in (KL(1.0), KL(2.0)), ϵ in (0.1, 1.0, 10.0)
            a = rand_measure(30, 3; static = true)
            b = DiscreteMeasure(copy(a.density), copy(a.set))
            unbalanced_sinkhorn!(D, a, b, ϵ; tol = 1e-7, max_iters = 10^6)
            @test a.dual_potential ≈ b.dual_potential rtol = 1e-4
        end
    end

    @testset "Theorem 4: positive definiteness and (separate) convexity of Sinkhorn divergence" begin
        for D in (KL(1.0), KL(2.0)), ϵ in (0.1, 1.0, 10.0)
            a_set = rand_measure(30, 3; static = true).set
            a_1 = DiscreteMeasure(rand(30), copy(a_set))
            a_2 = DiscreteMeasure(rand(30), copy(a_set))

            b_set = rand_measure(25, 3; static = true).set
            b_1 = DiscreteMeasure(rand(25), copy(b_set))
            b_2 = DiscreteMeasure(rand(25), copy(b_set))

            λ = rand()
            a = DiscreteMeasure(λ .* a_1.density .+ (1 - λ) .* a_2.density, copy(a_set))
            b = DiscreteMeasure(λ .* b_1.density .+ (1 - λ) .* b_2.density, copy(b_set))

            sd_b1 = sinkhorn_divergence!(D, a, b_1, ϵ; tol = 1e-5)
            sd_b2 = sinkhorn_divergence!(D, a, b_2, ϵ; tol = 1e-5)
            sd_a1 = sinkhorn_divergence!(D, a_1, b, ϵ; tol = 1e-5)
            sd_a2 = sinkhorn_divergence!(D, a_2, b, ϵ; tol = 1e-5)
            sd = sinkhorn_divergence!(D, a, b, ϵ; tol = 1e-5)
            @test all(x -> x > 0, (sd_b1, sd_b2, sd_a1, sd_a2, sd)) # positive definite
            @test λ * sd_a1 + (1 - λ) * sd_a2 >= sd # convexity in a
            @test λ * sd_b1 + (1 - λ) * sd_b2 >= sd # convexity in b

            @test sinkhorn_divergence!(D, a, a, ϵ; tol = 1e-5) ≈ 0 atol = 1e-4
            @test sinkhorn_divergence!(D, b, b, ϵ; tol = 1e-5) ≈ 0 atol = 1e-4
        end
    end

    @testset "eltype" begin
        for T in (Float32, Float64, BigFloat)
            a = rand_measure(4, 2; prob_type = T)
            @test eltype(a) == T
        end
    end

    @testset "Warnings and errors" begin
        a, b = balanced_measures(4, 3, 2)
        a = DiscreteMeasure(a.density * 2, a.set) # unbalanced
        @test_logs (:warn, r"for `D==Balanced\(\)`") unbalanced_sinkhorn!(
            Balanced(),
            a,
            b;
            tol = 10.0,
        )

        a = rand_measure(100, 2; static = true)
        b = rand_measure(90, 2; static = true)
        @test_logs (:warn, r"Maximum iterations") unbalanced_sinkhorn!(
            TV(),
            a,
            b,
            max_iters = 1,
        )

        @test_throws ArgumentError DiscreteMeasure(rand(5), rand(4), rand(5))
        @test_throws ArgumentError DiscreteMeasure(rand(5), rand(5), rand(4))
        @test_throws ArgumentError DiscreteMeasure(rand(4), rand(5), rand(5))
        @test_throws ArgumentError sinkhorn_divergence!(
            KL(1),
            randn(100, 90),
            a,
            b,
            ϵ = 1e-1,
        )
    end
end
