using UnbalancedOptimalTransport
using Documenter

previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

makedocs(;
    modules = [UnbalancedOptimalTransport],
    authors = "Eric P. Hanson",
    repo = "https://github.com/ericphanson/UnbalancedOptimalTransport.jl/blob/{commit}{path}#L{line}",
    sitename = "UnbalancedOptimalTransport.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ericphanson.github.io/UnbalancedOptimalTransport.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Optimal transport" => "optimal_transport.md",
        "Public API" => "public_api.md",
    ],
)

deploydocs(; repo = "github.com/ericphanson/UnbalancedOptimalTransport.jl")

ENV["GKSwstype"] = previous_GKSwstype;
