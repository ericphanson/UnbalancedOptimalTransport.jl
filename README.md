# UnbalancedOptimalTransport

[![CI](https://github.com/ericphanson/UnbalancedOptimalTransport.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/ericphanson/UnbalancedOptimalTransport.jl/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/ericphanson/UnbalancedOptimalTransport.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ericphanson/UnbalancedOptimalTransport.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ericphanson.github.io/UnbalancedOptimalTransport.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ericphanson.github.io/UnbalancedOptimalTransport.jl/dev)

This package provides an MIT license, dependency-free implementation of
Algorithm 1 of "Sinkhorn Divergences for Unbalanced Optimal Transport"
[[SFVTP19](http://arxiv.org/abs/1910.12958)], which allows calculation of the
optimal transport and Sinkhorn divergence between two positive measures (with
possibly different total mass), where mass creation and destruction is penalized
by one of several possible φ-divergences.

See the documentation for a quick tutorial as well as a mathematical description
of the quantities computed by this package.

While the code is generic, it is not currently written to dispatch to BLAS or
non-scalar GPU operations, although such contributions would be welcomed.

This package was motivated by a desire to compare bitmaps corresponding to
printed strings in
[VisualStringDistances.jl](https://github.com/ericphanson/VisualStringDistances.jl).

# Related packages

<https://github.com/JuliaOptimalTransport/OptimalTransport.jl> provides many optimal
transport methods, including the unbalanced case treated here. See also
[#9](https://github.com/ericphanson/UnbalancedOptimalTransport.jl/issues/9).

Other packages (warning, this list may become outdated):

* <https://github.com/mirkobunse/EarthMoversDistance.jl> (wrapper of C library)
* <https://github.com/mark-fangzhou-xie/JOT-Julia-Optimal-Transport> (balanced
  only, I think)
* <https://github.com/niladridas/OT_Julia> (balanced only, I think)
* <https://www.numerical-tours.com/julia/> (tutorials)
* <https://github.com/lchizat/optimal-transport> (not maintained)
* <https://github.com/baggepinnen/SpectralDistances.jl> (uses this package for the unbalanced case!)
* <https://github.com/devmotion/OptimalTransport.jl> (balanced only; exact algorithm via the network simplex method)
