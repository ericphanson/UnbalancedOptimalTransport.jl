using UnbalancedOptimalTransport
using Documenter

makedocs(;
    modules=[UnbalancedOptimalTransport],
    authors="Eric P. Hanson",
    repo="https://github.com/ericphanson/UnbalancedOptimalTransport.jl/blob/{commit}{path}#L{line}",
    sitename="UnbalancedOptimalTransport.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ericphanson.github.io/UnbalancedOptimalTransport.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ericphanson/UnbalancedOptimalTransport.jl",
)
