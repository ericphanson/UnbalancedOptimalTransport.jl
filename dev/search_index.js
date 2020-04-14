var documenterSearchIndex = {"docs":
[{"location":"public_api/#Public-API-1","page":"Public API","title":"Public API","text":"","category":"section"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"This package provides one type, DiscreteMeasure, which describes a measure on a finite set for use in Sinkhorn's algorithm and the related functions. The first step to computing e.g. the Sinkhorn divergence (sinkhorn_divergence!) is to construct DiscreteMeasure's describing the quantities of interest.","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"DiscreteMeasure","category":"page"},{"location":"public_api/#UnbalancedOptimalTransport.DiscreteMeasure","page":"Public API","title":"UnbalancedOptimalTransport.DiscreteMeasure","text":"DiscreteMeasure(density, [log_density], set) -> DiscreteMeasure\n\nConstruct a DiscreteMeasure object for use in unbalanced_sinkhorn! and related functions.\n\ndensity should be strictly positive; zero elements should instead be removed from set\nlog_density should be equal to log.(density) and can be omitted (in which case its computed automatically)\nset is a collection so that density[i] is the probability of the element set[i] occurring (where i ∈ eachindex(density, set)).\n\n\n\n\n\n","category":"type"},{"location":"public_api/#Functions-1","page":"Public API","title":"Functions","text":"","category":"section"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"This package provides three functions which act on DiscreteMeasure's to calculate quantities of interest:","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"OT!\nsinkhorn_divergence!\noptimal_coupling!","category":"page"},{"location":"public_api/#UnbalancedOptimalTransport.OT!","page":"Public API","title":"UnbalancedOptimalTransport.OT!","text":"function OT!(\n    D::AbstractDivergence,\n    C,\n    a::DiscreteMeasure,\n    b::DiscreteMeasure,\n    ϵ = 1e-1;\n    C = (x, y) -> norm(x - y),\n    kwargs...,\n) -> Number\n\nComputes the optimal transport cost between a and b, using unbalanced_sinkhorn!; see that function for the meaning of the parameters and the keyword arguments. Implements Equation (15) of [SFVTP19].\n\n\n\n\n\n","category":"function"},{"location":"public_api/#UnbalancedOptimalTransport.sinkhorn_divergence!","page":"Public API","title":"UnbalancedOptimalTransport.sinkhorn_divergence!","text":"sinkhorn_divergence!(\n    D::AbstractDivergence,\n    C,\n    a::DiscreteMeasure,\n    b::DiscreteMeasure,\n    ϵ = 1e-1;\n    kwargs...,\n) -> Number\n\nComputes the unbalanced sinkhorn divergence between a and b as defined in Def. 6 of [SFVTP19], using unbalanced_sinkhorn!; see that function for the meaning of the parameters and the keyword arguments. Sets the optimal dual_potential's of a and b.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#UnbalancedOptimalTransport.optimal_coupling!","page":"Public API","title":"UnbalancedOptimalTransport.optimal_coupling!","text":"function optimal_coupling!(\n    D::AbstractDivergence,\n    C,\n    a::DiscreteMeasure,\n    b::DiscreteMeasure,\n    ϵ = 1e-1;\n    dual_potentials_populated::Bool = false,\n    kwargs...) -> Matrix\n\nComputes the optimal coupling between a and b using the optimal dual potentials, the regularization parameter ϵ, and the cost function C.\n\nIf dual_potentials_populated = false, unbalanced_sinkhorn! is called to populate the dual potentials of a and b, using the divergence D. If dual_potentials_populated = true, one of unbalanced_sinkhorn! or OT! or sinkhorn_divergence! must be called first to set the optimal dual potentials, with the same choice of ϵ and C. In this case, a and b are not mutated.\n\nThis function implements Prop. 6 of [SFVTP19].\n\n\n\n\n\n","category":"function"},{"location":"public_api/#Sinkhorn's-algorithm-1","page":"Public API","title":"Sinkhorn's algorithm","text":"","category":"section"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"The above functions rely on unbalanced_sinkhorn! which uses Sinkhorn's algorithm to calculate the optimal Dual potentials.","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"unbalanced_sinkhorn!","category":"page"},{"location":"public_api/#UnbalancedOptimalTransport.unbalanced_sinkhorn!","page":"Public API","title":"UnbalancedOptimalTransport.unbalanced_sinkhorn!","text":"function unbalanced_sinkhorn!(\n    D::AbstractDivergence,\n    C,\n    a::DiscreteMeasure,\n    b::DiscreteMeasure,\n    ϵ = 1e-1;\n    tol = 1e-5,\n    max_iters = 10^5,\n    warn::Bool = true,\n) -> NamedTuple\n\nImplements algorithm 1 of [SFVTP19]. The dual_potential fields of a and b are updated to hold the optimal dual potentials. The density, log_density, and set fields are not modified. The parameters are\n\nD: the AbstractDivergence used for measuring the cost of creating or destroying mass\nϵ: the regularization parameter\nC: either a function from a.set × b.set to real numbers; should satisfy C(x,y) = C(y,x) and C(x,x)=0 when applicable, or a precomputed cost matrix  as generated by e.g. cost_matrix\ntol: the convergence tolerance\nmax_iters: the maximum number of iterations to perform.\nwarn: whether or not to warn when the maximum number of iterations is reached.\n\nReturns a NamedTuple of the number of iterations performed (iters), and the maximum residual (max_residual), which is the maximum infinity norm difference between consecutive iterates of the dual potentials, at the end of the process. If max_iters is not reached, iteration stops when the max_residual falls below tol.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#Divergences-1","page":"Public API","title":"Divergences","text":"","category":"section"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"As described on the page Optimal transport, we use varphi-divergences as penalty terms for mass creation and destruction. This package implements four such divergences, which are all described in Section 2.4 of [SFVTP19], and are listed below. To add your own divergences, see the AbstractDivergence interface section.","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"UnbalancedOptimalTransport.KL\nUnbalancedOptimalTransport.TV\nUnbalancedOptimalTransport.Balanced\nUnbalancedOptimalTransport.RG","category":"page"},{"location":"public_api/#UnbalancedOptimalTransport.KL","page":"Public API","title":"UnbalancedOptimalTransport.KL","text":"KL{ρ} <: AbstractDivergence\n\nRepresents the divergence ρ*KL(a|b), where KL is the Kullback-Leibler divergence. The parameter ρ is simply a scaling.\n\n\n\n\n\n","category":"type"},{"location":"public_api/#UnbalancedOptimalTransport.TV","page":"Public API","title":"UnbalancedOptimalTransport.TV","text":"TV{ρ} <: AbstractDivergence\n\nRepresents the divergence ρ*TV(u,v) = ρ*norm(u-v,1), where TV is twice the total variation distance. The parameter ρ is simply a scaling.\n\n\n\n\n\n","category":"type"},{"location":"public_api/#UnbalancedOptimalTransport.Balanced","page":"Public API","title":"UnbalancedOptimalTransport.Balanced","text":"Balanced <: AbstractDivergence\n\nRepresents the divergence Dᵩ(a|b) which is zero if a==b and infinite otherwise. Generalized by RG.\n\n\n\n\n\n","category":"type"},{"location":"public_api/#UnbalancedOptimalTransport.RG","page":"Public API","title":"UnbalancedOptimalTransport.RG","text":"RG{l,u} <: AbstractDivergence\n\nRepresents the divergence Dᵩ(a|b) which is zero if l*b .<= a .<= u*b and infinite otherwise. Equivalent to Balanced when l == u.\n\n\n\n\n\n","category":"type"},{"location":"public_api/#AbstractDivergence-interface-1","page":"Public API","title":"AbstractDivergence interface","text":"","category":"section"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"To add a divergence MyDivergence, create a struct","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"struct MyDivergence <: UnbalancedOptimalTransport.AbstractDivergence end","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"and implement a method for UnbalancedOptimalTransport.aprox and UnbalancedOptimalTransport.φstar. Optionally, one can also implement a method for UnbalancedOptimalTransport.initialize_dual_potential! and sinkhorn_divergence!, as a specialized implementation may obtain better performance.","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"UnbalancedOptimalTransport.AbstractDivergence\nUnbalancedOptimalTransport.aprox\nUnbalancedOptimalTransport.φstar\nUnbalancedOptimalTransport.initialize_dual_potential!","category":"page"},{"location":"public_api/#UnbalancedOptimalTransport.AbstractDivergence","page":"Public API","title":"UnbalancedOptimalTransport.AbstractDivergence","text":"abstract type AbstractDivergence\n\nAn abstract type representing Csiszár φ-divergences. Subtypes should implement φstar and aprox, and optionally can implement initialize_dual_potential! and/or sinkhorn_divergence!.\n\n\n\n\n\n","category":"type"},{"location":"public_api/#UnbalancedOptimalTransport.aprox","page":"Public API","title":"UnbalancedOptimalTransport.aprox","text":"aprox(::AbstractDivergence, ϵ::Number, x::Number) -> Number\n\nThe anisotropic proximity operator defined in Def. 2 of [SFVTP19].\n\n\n\n\n\n","category":"function"},{"location":"public_api/#UnbalancedOptimalTransport.φstar","page":"Public API","title":"UnbalancedOptimalTransport.φstar","text":"φstar(::AbstractDivergence, q::Number) -> Number\n\nThe Legendre conjugate of the function φ associated to the divergence.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#UnbalancedOptimalTransport.initialize_dual_potential!","page":"Public API","title":"UnbalancedOptimalTransport.initialize_dual_potential!","text":"initialize_dual_potential!(::AbstractDivergence, a::DiscreteMeasure) -> Nothing\n\nApply an initialization for the dual potential, for use in unbalanced_sinkhorn!; falls back to zeroing out the dual potential. Specialized implementations can improve performance, but should not affect correctness.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#Utilities-1","page":"Public API","title":"Utilities","text":"","category":"section"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"sinkhorn_divergence! uses the following function, which may be specialized to improve performance.","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"UnbalancedOptimalTransport.fdot","category":"page"},{"location":"public_api/#UnbalancedOptimalTransport.fdot","page":"Public API","title":"UnbalancedOptimalTransport.fdot","text":"fdot(f, u, v) -> Number\n\nA generic, allocation-free implementation of dot(u, f.(v)). It may be faster to provide a specialized method to dispatch to BLAS or so forth.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"The following function cost_matrix is used in unbalanced_sinkhorn!, OT!, and optimal_coupling to precompute the costs given a cost function.","category":"page"},{"location":"public_api/#","page":"Public API","title":"Public API","text":"UnbalancedOptimalTransport.cost_matrix","category":"page"},{"location":"public_api/#UnbalancedOptimalTransport.cost_matrix","page":"Public API","title":"UnbalancedOptimalTransport.cost_matrix","text":"cost_matrix([C,] a, b) -> Matrix\n\nPrecompute the cost matrix given a cost function C. If no function C is provided, the default is C(x,y) = norm(x-y).\n\n\n\n\n\n","category":"function"},{"location":"optimal_transport/#Optimal-transport-1","page":"Optimal transport","title":"Optimal transport","text":"","category":"section"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"Optimal transport is a theory that provides a way to describe distances between positive measures in terms of distances on their underlying domains. The following is a short description of the task; for a more thorough review, see Wikipedia, or some of the Other references.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"In this package, we only consider finite sets, and so a \"positive measure\" is simply a collection of non-negative numbers. In the case that these numbers sum to one, we can consider them as probabilities, although we don't make that restriction here. Let X and Y be finite sets (the only kind considered in this package), and a and b be functions,","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"a  X to mathbbR_+ quad textand quad b Y to mathbbR_+","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"We assume a(x)  0 for each x in X and likewise b(y)  0 for y in Y; zero elements can simply be removed from X or Y.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"The associated optimal transport problem is the following. We start with a(x) \"mass\" of material at each point x in X, and wish to end up with b(y) mass of material at each point y in Y, but there is some cost c(xy) associated to moving a unit of mass from x to y.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"For each pair (xy) in Xtimes Y, we aim to decide how much mass to move from x to y by choosing a coupling, a function π  X times Y to mathbbR_+, which minimizes the cost:","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"beginaligned\noperatornameOT(ab) = textminimize quad  sum_xin X y in Y π(xy) c(xy)\ntextsuch that quad  a(x) = sum_y in Y pi(xy) quad forall x in X\n b(y) = sum_x in X pi(xy) quad forall y in Y\n pi(xy) geq 0 quad forall xin X forall y in Y\nendaligned","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"The constraint a(x) = sum_y in Y pi(xy) means that a(x) units of mass was transported from location x, and b(y) = sum_x in X pi(xy) means that b(y) units of mass ended up at location y, as desired.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"The problem as written, however, has no solution (is infeasible), however, if sum_xin X a(x) neq sum_y in Yb(y), as can be verified by the fact that if a solution pi existed, then the constraints imply that","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"sum_y in Y b(y) = sum_xin X y in Y pi(xy) = sum_x in X a(x)","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"which is a contradiction. That's because our problem as posed does not allow the creation or destruction of mass.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"To resolve this, following [SFVTP19], we remove the requirements that a(x) = sum_y in Y pi(xy) and b(y) = sum_x in X pi(xy), and instead add a penalty that grows the more those constraints are violated (i.e. imposing \"soft\" constraints instead of \"hard\" constraints).","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"We model these penalties as a varphi-divergence D_varphi, which are defined in terms of a function varphi  mathbbR_+ to mathbbR_+ as","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"D_varphi(u  v) = sum_z in Z varphileft( fracu(z)v(z) right) b(z)","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"where Z is a finite set, and u  Z to mathbbR_+ and v  Z to mathbbR_+ as functions with v(z)  0 for each z in Z. See Divergences for the list of divergences implemented in this package.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"With that modification, we have the unbalanced optimal transport problem,","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"beginaligned\noperatornameOT(D_varphi ab) = textminimize quad \nsum_xin X y in Y π(xy) c(xy) + D_varphi(pi_1  a) + D_varphi(pi_2  b)\ntextsuch that quad  pi(xy) geq 0 quad forall xin X forall y in Y\nendaligned","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"where pi_1 is defined by pi_1(x) = sum_y in Y pi(xy) for each x in X, and likewise pi_2(y)=sum_x in X pi(xy) for y in Y.","category":"page"},{"location":"optimal_transport/#Entropic-regularization-1","page":"Optimal transport","title":"Entropic regularization","text":"","category":"section"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"We are faced with a problem: operatornameOT(D_varphi ab) can be difficult to compute, especially when X and Y are large. It turns out that operatornameOT(D_varphi ab) can be described as the limit of a sequence of problems that are easier to solve.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"For varepsilon geq 0, consider the entropically-regularized unbalanced optimal transport problem,","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"beginaligned\noperatornameOT(D_varphi ab varepsilon) = textminimize quad \n    sum_xin X y in Y π(xy) c(xy) + D_varphi(pi_1  a)\n    + D_varphi(pi_2  b) + varepsilon operatornameKL(pi  a otimes b)\ntextsuch that quad  pi(xy) geq 0 quad forall xin X forall y in Y\nendaligned","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"where operatornameKL is the Kullback Leibler divergence (also called the relative entropy), which in fact is the varphi-divergence with varphi(x) = xlog x - x + 1, and a otimes b  X times Y to mathbbR_+ is defined by (a otimes b)(xy) = a(x)b(y).","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"Entropic regularization refers to the additional term varepsilon operatornameKL(pi  a otimes b). Adding this term yields several advantages; see e.g. Chapter 4 of [PC18] in Other references for entropic regularization in optimal transport in general, and [SFVTP19] in particular for the unbalanced case. As concerns this package, however, the most important advantage is that operatornameOT(D_varphi ab varepsilon) can be computed quickly by an iterative scheme called Sinkhorn's algorithm, which is implemented in the function unbalanced_sinkhorn!, based on Algorithm 1 of [SFVTP19]. This is used to compute the quantity operatornameOT(D_varphi ab varepsilon) in the function OT!.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"Note that this package also implements the optimization problem form of operatornameOT(D_varphi a b varepsilon), using the modelling language Convex.jl and the solvers SCS.jl and ECOS.jl, in test/Convex_formulation.jl, in order to test the implementation of Sinkhorn's algorithm in unbalanced_sinkhorn!. This functionality is contained in the tests rather than the package itself in order to avoid the extra dependencies, and since solving the optimization problem is much less practical than using Sinkhorn's algorithm.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"One problem raised by the regularization is that","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"operatornameOT(D_varphiaavarepsilon) neq 0","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"in general. This is not ideal when one wishes to consider (ab) mapsto operatornameOT(D_varphiabvarepsilon) as a measure of distance between a and b. To remedy this, one defines the Sinkhorn divergence","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"S(D_varphiabvarepsilon) = operatornameOT(D_varphiabvarepsilon)\n    - frac12operatornameOT(D_varphiaavarepsilon)\n    - frac12operatornameOT(D_varphib bvarepsilon)\n    + fracepsilon2(m(a) - m(b))^2","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"where e.g. m(a) = sum_xin X a(x) is the total mass of a (this is Def. 6 of [SFVTP19]). This quantity is computed by sinkhorn_divergence!.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"When D_varphi is chosen as proportional to operatornameKL, this quantity has the following useful properties, shown in Theorem 4 of [SFVTP19]:","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"Positive-definiteness: S(D_varphiabvarepsilon) geq 0 with equality if and only if a=b\nConvexity in both arguments: a mapsto S(D_varphiabvarepsilon) is convex, and b mapsto S(D_varphiabvarepsilon) is convex.","category":"page"},{"location":"optimal_transport/#Dual-potentials-1","page":"Optimal transport","title":"Dual potentials","text":"","category":"section"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"The quantity operatornameOT(D_varphi ab varepsilon) can be equivalently described by the so-called dual problem,","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"beginaligned\noperatornameOT(D_varphi ab varepsilon) = textmaximize quad\n -sum_xin X a(x) varphi^*(-f(x)) -sum_yin Y b(y) varphi^*(-g(x))\n     quad - varepsilon sum_xin X y in Y a(x)b(y)\n    left(exp( fracf(x) + g(y) - c(xy)varepsilon ) -1right) \ntextwhere quad  f  X to mathbbR  g  Y to mathbbR\nendaligned","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"where varphi^* mathbbR to mathbbR is the Legendre transform of varphi, defined as","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"varphi^*(z) = sup_x geq 0 xz - varphi(x)","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"The functions f and g are called the dual potentials (in [SFVTP19], at least). Sinkhorn's algorithm as implemented in unbalanced_sinkhorn! computes the optimal dual potentials, namely those which achieve the maximum in the optimization problem above.","category":"page"},{"location":"optimal_transport/#References-1","page":"Optimal transport","title":"References","text":"","category":"section"},{"location":"optimal_transport/#SFVTP19-1","page":"Optimal transport","title":"SFVTP19","text":"","category":"section"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"[SFVTP19] Séjourné, T., Feydy, J., Vialard, F.-X., Trouvé, A., Peyré, G., 2019. Sinkhorn Divergences for Unbalanced Optimal Transport. arXiv:1910.12958.","category":"page"},{"location":"optimal_transport/#Other-references-1","page":"Optimal transport","title":"Other references","text":"","category":"section"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"[PC18] Peyré, G., Cuturi, M., 2018. Computational Optimal Transport. arXiv:1803.00567.","category":"page"},{"location":"optimal_transport/#","page":"Optimal transport","title":"Optimal transport","text":"[Villani09] Villani, C., 2009. Optimal Transport: Old and New, Grundlehren der mathematischen Wissenschaften. Springer-Verlag, Berlin Heidelberg.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"CurrentModule = UnbalancedOptimalTransport","category":"page"},{"location":"#UnbalancedOptimalTransport-1","page":"Home","title":"UnbalancedOptimalTransport","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This package provides an MIT license, dependency-free implementation of Algorithm 1 of \"Sinkhorn Divergences for Unbalanced Optimal Transport\" [SFVTP19] in unbalanced_sinkhorn!, in a generic and extensible way. This is used to compute Sinkhorn divergences via sinkhorn_divergence!.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"See Optimal transport for some background and a mathematical description of the quantities computed by this package, Public API for a description of the functions provided, and below for a quick tutorial.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"While the code is generic, it is not currently written to dispatch to BLAS or non-scalar GPU operations, although such contributions would be welcomed.","category":"page"},{"location":"#Quick-tutorial-1","page":"Home","title":"Quick tutorial","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"using UnbalancedOptimalTransport, Plots\n\nX = 1:4; # a set\na_weights = [0.5, 1.0, 1.0, 0.5]; # weights on X\nY = 3:5; # another set\nb_weights = [0.5, 0.75, 0.5]; # weights on Y\nplot(bar(X, a_weights, label=\"a\"), bar(Y, b_weights, label=\"b\"),\n        xlims = (0, 6), ylims=(0, 1.2), legend=:topleft);\nsavefig(\"histograms.svg\"); nothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We wish to move the a histogram to the b histogram with the least total cost, however we will clearly need to remove some mass as well. We choose the KL divergence to penalize mass destruction.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"a = DiscreteMeasure(a_weights, X);\nb = DiscreteMeasure(b_weights, Y);\ncost = (x, y) -> abs(x - y)\nϵ = 0.01 # small regularization\nD = UnbalancedOptimalTransport.KL(1.0)\nSD = sinkhorn_divergence!(D, cost, a, b, ϵ)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The number SD provides us with a distance between the a and b histograms, as computed by the (unbalanced) Sinkhorn divergence. We can also compute the \"optimal coupling\" which shows us how to move between the histograms.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"π = optimal_coupling!(D, cost, a, b, ϵ)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Here, π[x,y] represents how much mass we should move from x to y. Since the sum of the rows is less than the corresponding entries of a_weight, some of the mass is destroyed. To understand this better, let us lower the penalty for mass destruction and mass creation to almost nothing, and see how the optimal coupling changes.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"D = UnbalancedOptimalTransport.KL(0.01)\nπ = optimal_coupling!(D, cost, a, b, ϵ)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We can see that the only non-tiny entries of π are the (3,1) and (4,2), corresponding to (x=3, y=3), as the first element of Y is 3. We see then with this choice of divergence and cost, we don't really move any mass, and just create and destroy as needed. On the other hand, let us see what happens when there is a high penalty for mass creation and destruction:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"D = UnbalancedOptimalTransport.KL(1000.0)\nsinkhorn_divergence!(D, cost, a, b, ϵ)\nπ = optimal_coupling!(D, cost, a, b, ϵ)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We see warnings about the maximum number of iterations being exceeded, so let's increase that parameter and try again. Note that warnings can be disabled by passing warn=false as a keyword argument.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"sinkhorn_divergence!(D, cost, a, b, ϵ; max_iters = 10^6)\nπ = optimal_coupling!(D, cost, a, b, ϵ; max_iters = 10^6)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We see that now we move some mass from each each element of X to each element of Y, to try to avoid needing to create or destroy mass, except no mass is moved from x=4 to y=3, presumably because it's better to move it to y=4 and y=5 instead.","category":"page"}]
}