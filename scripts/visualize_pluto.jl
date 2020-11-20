### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 535d9692-29b3-11eb-1ef4-1f3c7f559481
using UnbalancedOptimalTransport

# ╔═╡ 5bad3192-29b3-11eb-343d-a77cb1f89144
using UnbalancedOptimalTransport: KL, Balanced

# ╔═╡ 8c677950-2b48-11eb-1843-356af0ab079d
using Markdown, Random

# ╔═╡ 7ff472ae-29b3-11eb-267b-59058f5f5e85
using LinearAlgebra

# ╔═╡ f5497004-29b3-11eb-3ead-5766e13c274a
using LinearAlgebra: dot

# ╔═╡ 56ae2176-29b6-11eb-1c7f-39cd7ff91f75
using Test

# ╔═╡ 955137ba-29b6-11eb-1efc-1b919d6dfbde
using UnicodePlots

# ╔═╡ 84af5abc-29b7-11eb-38d8-d747d7309c91
using AbstractPlotting, CairoMakie

# ╔═╡ fdabe84e-29b8-11eb-1556-157a0847c121
using AbstractPlotting.MakieLayout

# ╔═╡ 3ec9e128-2b35-11eb-3632-75687fc22283
using PlutoUI

# ╔═╡ 988bf8e4-2b54-11eb-0a90-5baaf35eaa4c
md"# Optimal transport"

# ╔═╡ ba877b44-2b54-11eb-0a06-ed34fe010d50


# ╔═╡ 3e2f024c-2b54-11eb-38c3-494488526f66
md"# Signed transport: special case 1"

# ╔═╡ be58708c-2b26-11eb-2be3-e5a5b6d06324
md"""
Consider the task of transporting vector `a` which has equal amounts of positive and negative mass to the zero vector `a`. That means we want to rearrange the mass in `b` until it all cancels out. It costs $1$ to move $1$ unit of (positive or negative) mass one step to the left or right, and we wish to minimize the cost of this rearrangement.
"""

# ╔═╡ c728610c-2b49-11eb-1397-c91c9caf48bf
md"""
Recall that by solving the optimal transport problem we obtain the optimal coupling, which is a matrix whose $(i,j)$th entry tells us how much mass to move from site $i$ to site $j$,

We can see that to transport $a$ to $b$, we could instead look at how to transport $a_+$ to $a_-$: the optimal coupling for that problem tells us how to move the mass in $a_+$ to obtain $a_-$, but if we apply it to $a$, we end up at $b$, because we've exactly moved the positive mass of $a$ onto the negative mass of $a$, cancelling it out!
"""

# ╔═╡ 64d9a390-29ba-11eb-211d-8ff252146626
@bind t PlutoUI.Clock()

# ╔═╡ 55fd407c-2b55-11eb-25a8-95b498153d2a
md"# Signed transport: special case 2"

# ╔═╡ 63e2bbd6-2b55-11eb-0634-7f7bc7411750
md"Let's consider a slightly more general case. Instead of $b$ being zero, let's chose a pair of non-zero signed vectors."

# ╔═╡ c7cd855e-2b58-11eb-3363-21d8f7299e05
md"We can inspect their decomposition into positive and negative parts, where $a = a_+ - a_-$ and $b = b_+ - b_-$:"

# ╔═╡ 19b0f808-2b59-11eb-3afd-3184c03c2ae6
md"Just as in the previous case, we can reformulate this as transport problem between positive vectors. This time, we wish to transport the mass from $a_+ + b_-$ to $b_+ + a_-$.

Why? If we have a coupling which can rearrange $a_+ + b_-$ into $b_+ + a_-$, then the coupling gives a map

$$v \mapsto v + (b_+ + a_-) - (a_+ + b_-)$$

So if we apply it to $a = a_+ - a_-$, then we get out

$$a \mapsto a_+ - a_- + (b_+ + a_-) - (a_+ + b_-) = b_+ - b_-$$

which is just $b$, as desired.

Thus, we simply need to solve the optimal transport problem between these two:
"

# ╔═╡ 01efdcba-2b5a-11eb-0fee-bbb86fdab23f
md"Luckily, we've chosen $a$ and $b$ so that $a_+ + b_-$ and $b_+ + a_-$ have the same total mass, just as in the previous case. That means we can solve the optimal transport problem and recover a solution:"

# ╔═╡ 227c4dec-2b5a-11eb-077f-cd1a91657013
@bind t2 PlutoUI.Clock()

# ╔═╡ 9e8f69e4-2b54-11eb-0c5c-353c741d5417
md"# Unbalanced positive transport"

# ╔═╡ b473839c-2b54-11eb-33cd-91b352c48226


# ╔═╡ a48a36c2-2b54-11eb-1a08-859efec67f94
md"# Signed transport: general case"

# ╔═╡ b5044648-2b54-11eb-0c07-757bba7a0b1d


# ╔═╡ 71e2a2e8-2b58-11eb-07eb-b522b7bea98c
md"# Choices" 

# ╔═╡ 5bfc5448-29b3-11eb-1941-d9bff0f425c6
#D = KL(10.0)
D = Balanced()

# ╔═╡ c14e7db6-29b4-11eb-1687-315faa512117
ϵ = 0.001

# ╔═╡ 5baf5922-29b3-11eb-2a67-ef89bd104db8
d = 10

# ╔═╡ 5bc23a06-29b3-11eb-062c-393396ede7f2
b = zeros(d)

# ╔═╡ f17a5ce2-2b55-11eb-37a2-4d56fad8b4b3
a2 = 10*(rand(d) - rand(d))

# ╔═╡ 3975431a-2b54-11eb-0c7a-2b5d7878de97
md"# Code"

# ╔═╡ a38e7f08-29bc-11eb-01f3-256e6c2313f9
function rand_normalized(d)
	a = rand(d)
	return a / sum(a)
end

# ╔═╡ 5bc343b0-29b3-11eb-3e5a-df86461a783d
a = 10 .* (rand_normalized(d) - rand_normalized(d))

# ╔═╡ 5bcbfe6a-29b3-11eb-2e7a-1bab02e6e094
am = SignedMeasure(a)

# ╔═╡ 5bd681a0-29b3-11eb-0ca5-630541f87c60
bm = SignedMeasure(b)

# ╔═╡ 5c3c6c04-29b3-11eb-2142-e344978fb8e0
function extended_vec(density, set, d)
    v = zeros(d)
    for i = eachindex(set, density)
        v[set[i]] = density[i]
    end
    return v
end

# ╔═╡ 5c480da2-29b3-11eb-1e71-fb742ec0fc67
extended_vec(dm::DiscreteMeasure, d) = extended_vec(dm.density, dm.set, d)

# ╔═╡ 5c537656-29b3-11eb-2ed4-0d0bd0650fab
function extended_coupling_matrix(coupling_matrix, set1, d1, set2, d2)
    m = zeros(d1, d2)
    for i in axes(coupling_matrix, 1), j in axes(coupling_matrix, 2)
        m[set1[i], set2[j]] = coupling_matrix[i,j]
    end
    return m
end

# ╔═╡ 1ed870a8-29bb-11eb-3523-adcb5c6d9923
convex_combination(x,y,t) = (1 .- t) .* x .+ t .* y

# ╔═╡ 169e540c-2b32-11eb-1172-47c12c009bd0
convex_combination(x,y) = t -> convex_combination(x,y,t)

# ╔═╡ 214cf31e-2b27-11eb-1fbd-d50cfd38a1b7
function make_list(a, a_name, b, b_name, ab_conv)
	
	list = [(a, a_name)]
	
	if !(D isa Balanced)
		push!(list, (ab_conv(0), "Create/destroy mass"))
	end
	
	append!(list, [(ab_conv(0.25), "Transport 25%"), (ab_conv(0.5), "Transport 50%"), (ab_conv(0.75), "Transport 75%"),  (ab_conv(1), "Transport 100%")])
	
	if !(D isa Balanced)
		push!(list, (b, "Create/destroy mass"))
	end
	
	push!(list, (b, b_name))
	return list
end

# ╔═╡ 4a32067a-2b31-11eb-28c5-1519242d8284
function get_ylims(list)
	exs = extrema.(first.(list))
	m = minimum(first, exs)
	M = maximum(last, exs)
	range = M - m
	lower_buffer = m ≈ 0 ? range / 20 : range / 10
	upper_buffer = range / 10
	return (m - lower_buffer, M + upper_buffer)
end

# ╔═╡ 4e690274-2b23-11eb-2907-5546c62231d3
begin
	maybe_lift(f, x::Observable) = lift(f, x)
	maybe_lift(f, x) = f(x)
end

# ╔═╡ b98a4cca-29d4-11eb-2e05-812d10497850
FONT = "Julia Mono"

# ╔═╡ 507cb14e-2b1b-11eb-36b3-a134271c47ef
plus = '₊'

# ╔═╡ 5e3f5d40-2b1b-11eb-18e4-5557c5b69a29
minus = '₋'

# ╔═╡ e86d4eb0-2b44-11eb-0ea2-c579b998317f
Markdown.parse("Optimal transport requires positive masses. However, we can decompose our signed vectors into non-negative vectors, e.g. `a = a$plus - a$minus`:")

# ╔═╡ 0001bfe2-2b49-11eb-1607-259b06c569e6
pos(v) =  max.(v, 0)

# ╔═╡ 4d1fe84e-2b49-11eb-230f-aba60a20c75a
pos(v::SignedMeasure) = v.pos

# ╔═╡ 0f81c860-2b49-11eb-3113-031161588326
neg(v) = -min.(v, 0)

# ╔═╡ 53468264-2b49-11eb-3235-d166a7cf2b54
neg(v::SignedMeasure) = v.neg

# ╔═╡ 71e3f3a6-2b55-11eb-2d2a-d5a604785ddf
b2 = shuffle!([sum(pos(a2)) * rand_normalized(d ÷ 2); -sum(neg(a2)) * rand_normalized(d ÷ 2)])

# ╔═╡ 3e6886d6-2b2a-11eb-1c5f-c786380bdc51
extended_vec(sm::SignedMeasure, d) = extended_vec(pos(sm), d) - extended_vec(neg(sm), d)

# ╔═╡ 32731c08-2b49-11eb-3358-bf9fc08f6004
extended_vec(am.pos,d) ≈ pos(a)

# ╔═╡ 41fffad2-2b49-11eb-2601-81a7941b0474
extended_vec(am.neg,d) ≈ neg(a)

# ╔═╡ c532b756-2b28-11eb-2d2f-c1ca7e2046e1
function get_transitions(D::UnbalancedOptimalTransport.AbstractDivergence, ϵ, x::DiscreteMeasure, y::DiscreteMeasure, d)
	
	C = [norm(x.set[i] - y.set[j]) for i = eachindex(x.set), j = eachindex(y.set)]
	
	coupling = optimal_coupling!(D, C, x, y, ϵ)

	x_ext = extended_vec(x, d)
	y_ext = extended_vec(y, d)
	coupling_ext = extended_coupling_matrix(coupling, x.set, d, y.set, d)
	μ_ext = vec(sum(coupling_ext, dims=2))
	ν_ext = vec(sum(coupling_ext, dims=1))
	
	return x_ext, μ_ext, ν_ext, y_ext
end

# ╔═╡ e8185fae-2b29-11eb-1b03-d35e774da123
function get_transitions(D::UnbalancedOptimalTransport.AbstractDivergence, ϵ, a::SignedMeasure, b::SignedMeasure, d)
	x, y = make_measures(a, b)
	
	x_ext, μ_ext, ν_ext, y_ext = get_transitions(D, ϵ, x, y, d)
	
	a_ext = extended_vec(a, d)
	b_ext = extended_vec(b, d)
	
	return a_ext, a_ext - x_ext + μ_ext,  b_ext - y_ext + ν_ext, b_ext
end

# ╔═╡ 35e9872a-29b9-11eb-18c6-fd932571d10b
colors = AbstractPlotting.current_default_theme()[:palette].color[]

# ╔═╡ bd50e5fa-2b44-11eb-318f-cb0171004c46
default_color = colors[1]

# ╔═╡ b5ef9b9a-2b20-11eb-2c94-5db769d419ad
function barplot_with_title!(scene, data, title; color = default_color, kwargs...)
	layout = GridLayout()
	layout[1, 1] = t = LText(scene, title, textsize = 30, font = FONT, color = (:black, 0.25))
	t.tellwidth = false

	layout[2, 1] = ax = LAxis(scene)
	AbstractPlotting.barplot!(ax, maybe_lift(x -> 1:length(x), data), data; color=color, kwargs...)
	#ax.aspect = AxisAspect(2)
	return (layout, ax)
end


# ╔═╡ 91998cc8-2b44-11eb-3e6a-090dd9a3debf
let
	scene, layout = layoutscene()
	l1, ax1 = barplot_with_title!(scene, a, "a")
	l2, ax2 = barplot_with_title!(scene, b, "b")
	linkyaxes!(ax1, ax2)

	layout[1, 1:2] = [l1, l2]
	scene
end

# ╔═╡ 3d1d0440-2b58-11eb-12b0-fbdf499e54ae
let
	scene, layout = layoutscene()
	l1, ax1 = barplot_with_title!(scene, a2, "a", color = colors[2])
	l2, ax2 = barplot_with_title!(scene, b2, "b", color = colors[2])
	linkyaxes!(ax1, ax2)

	layout[1, 1:2] = [l1, l2]
	scene
end

# ╔═╡ 02d85880-2b59-11eb-204c-cfe9f0e198b2
let
	scene, layout = layoutscene()
	l1, ax1 = barplot_with_title!(scene, pos(a2) + neg(b2), "a$plus + b$minus", color = colors[2])
	l2, ax2 = barplot_with_title!(scene, pos(b2) + neg(a2), "b$plus + a$minus", color = colors[2])
	linkyaxes!(ax1, ax2)

	layout[1, 1:2] = [l1, l2]
	scene
end

# ╔═╡ ddc87348-2b21-11eb-1079-0f662e8190f5
function barplot_pos_neg!(scene, v, name; ylims = (minimum(v)-1, maximum(v)+1), kwargs...)
	layout = GridLayout()	
	l1, ax1 = barplot_with_title!(scene, v, name; ylims, kwargs...)
	l2, ax2 = barplot_with_title!(scene, pos(v), "$name$plus"; ylims, kwargs...)
	l3, ax3 = barplot_with_title!(scene, neg(v), "$name$minus"; ylims, kwargs...)
	linkyaxes!(ax1, ax2, ax3)
	layout[1, 1:3] = [l1, l2, l3]
	return layout
end

# ╔═╡ 9627c1f0-29d3-11eb-2871-1ba3530f91f4
let
	scene, layout = layoutscene()
	layout[1,1] = barplot_pos_neg!(scene, a, "a")
	scene
end

# ╔═╡ aa72c7a0-2b58-11eb-1969-05213e08dd22
let
	scene, layout = layoutscene()
	l1 = barplot_pos_neg!(scene, a2, "a", color = colors[2])
	l2 = barplot_pos_neg!(scene, b2, "b", color = colors[2])
	layout[1:2, 1] = [l1, l2]
	scene
end

# ╔═╡ 14d4d918-2b2b-11eb-31a8-051e0c39ce3f
function _transition_plot!(scene, list, bar_vals, title_text; kwargs...)
	layout = GridLayout()
	ylims = get_ylims(list)
	l1, ax1 = barplot_with_title!(scene, list[1][1],  list[1][2]; kwargs...)
	l2, ax2 = barplot_with_title!(scene, bar_vals, title_text; kwargs...)
	l3, ax3 = barplot_with_title!(scene, list[end][1], list[end][2]; kwargs...)
	ylims!(ax1, ylims)
	ylims!(ax2, ylims)
	ylims!(ax3, ylims)
	# linkyaxes!(ax1, ax2, ax3)	
	layout[1, 1:3] = [l1, l2, l3]
	return layout
end

# ╔═╡ 87722ef4-2b2a-11eb-1ffe-03fd75e12532
function transport_plot!(scene, a_name, b_name, transitions; repeat = false, kwargs...)
	
	a, a0, b0, b = transitions
	
	list = make_list(a, a_name, b, b_name, convex_combination(a0, b0))
	@info length(list)

	title_text = Node(" ")
	bar_vals = Node(a)
	
	 up = function(t)
		if repeat
			vals, title = list[mod1(t, end)]
		else
			vals, title = list[min(t, end)]
		end
		bar_vals[] = vals
		title_text[] = title
		return nothing
	end
			
	layout = GridLayout()

	layout[1, 1:3] = _transition_plot!(scene, list, bar_vals, title_text; kwargs...)
	return layout, up
end

# ╔═╡ 7cb2ec38-2b34-11eb-0131-03c09f0f3581
function transport_anim_both!(scene, a, aname, b, bname; ab_name1 ="$aname$plus + $bname$minus", ab_name2 = "$bname$plus + $aname$minus",  kwargs...)
	layout = GridLayout()
	
	x, y = make_measures(a,b)
	l1, up1 = transport_plot!(scene, ab_name1, ab_name2, get_transitions(D, ϵ, x, y, d); kwargs...)
	
	layout[1,1] = l1
	
	l2, up2 = transport_plot!(scene, "a", "b", get_transitions(D, ϵ, a, b, d); kwargs...)
	layout[2,1] = l2
	
	up = t -> (up1(t); up2(t); nothing)
	
	return layout, up
end

# ╔═╡ 3e51a966-2b34-11eb-035b-ddca47de3459
signed_transport_scene, up = let
	scene, layout = layoutscene()
	l, u = transport_anim_both!(scene, SignedMeasure(a), "a", SignedMeasure(b), "b"; ab_name1 = "a$plus", ab_name2 = "a$minus")
	layout[1, 1] = l
	scene, u
end

# ╔═╡ de79c2a2-29bf-11eb-0ba7-5574d8c69216
begin
	up(t)
	signed_transport_scene
end

# ╔═╡ 4a71ec26-2b5a-11eb-2ad8-0bb517c71847
signed_transport_scene2, up2 = let
	scene, layout = layoutscene()
	l, u = transport_anim_both!(scene, SignedMeasure(a2), "a", SignedMeasure(b2), "b"; color = colors[2])
	layout[1, 1] = l
	scene, u
end

# ╔═╡ 2962c1d6-2b5a-11eb-025b-23d28581febd
begin
	up2(t2)
	signed_transport_scene2
end

# ╔═╡ 5c626206-29b3-11eb-2cae-275536bd1ab6
lg(x) = x <= 0 ? zero(x) : log(x)

# ╔═╡ 5c6f5238-29b3-11eb-1962-65fe06d73f08
entropy(a::AbstractArray) = sum(x -> -x * lg(x), a)

# ╔═╡ 5c7b5d92-29b3-11eb-2c0b-01e0dbf269a7
function divergence(::KL{ρ}, a, b) where {ρ}
    ρ * (-entropy(a) - dot(a, lg.(b)) + sum(b - a))
end

# ╔═╡ d750b1d6-2b37-11eb-1fa1-4fad6904a144
t0 = Ref(0)

# ╔═╡ c16ae7b0-2b37-11eb-271f-791043d98a3e
@bind click PlutoUI.Button()

# ╔═╡ 0dcffaec-2b35-11eb-1329-197b99f7c08f
begin
	click
	up(t0[] += 1)
	t0val = t0[]
	signed_transport_scene
end

# ╔═╡ f7c242ea-2b37-11eb-0389-07bef2f1e86c
mod1(t0val, D isa Balanced ? 6 : 8)

# ╔═╡ Cell order:
# ╟─988bf8e4-2b54-11eb-0a90-5baaf35eaa4c
# ╠═ba877b44-2b54-11eb-0a06-ed34fe010d50
# ╟─3e2f024c-2b54-11eb-38c3-494488526f66
# ╟─be58708c-2b26-11eb-2be3-e5a5b6d06324
# ╟─91998cc8-2b44-11eb-3e6a-090dd9a3debf
# ╟─e86d4eb0-2b44-11eb-0ea2-c579b998317f
# ╟─9627c1f0-29d3-11eb-2871-1ba3530f91f4
# ╟─c728610c-2b49-11eb-1397-c91c9caf48bf
# ╠═64d9a390-29ba-11eb-211d-8ff252146626
# ╟─de79c2a2-29bf-11eb-0ba7-5574d8c69216
# ╟─55fd407c-2b55-11eb-25a8-95b498153d2a
# ╟─63e2bbd6-2b55-11eb-0634-7f7bc7411750
# ╟─3d1d0440-2b58-11eb-12b0-fbdf499e54ae
# ╟─c7cd855e-2b58-11eb-3363-21d8f7299e05
# ╟─aa72c7a0-2b58-11eb-1969-05213e08dd22
# ╟─19b0f808-2b59-11eb-3afd-3184c03c2ae6
# ╟─02d85880-2b59-11eb-204c-cfe9f0e198b2
# ╟─01efdcba-2b5a-11eb-0fee-bbb86fdab23f
# ╟─227c4dec-2b5a-11eb-077f-cd1a91657013
# ╟─2962c1d6-2b5a-11eb-025b-23d28581febd
# ╠═9e8f69e4-2b54-11eb-0c5c-353c741d5417
# ╠═b473839c-2b54-11eb-33cd-91b352c48226
# ╠═a48a36c2-2b54-11eb-1a08-859efec67f94
# ╠═b5044648-2b54-11eb-0c07-757bba7a0b1d
# ╟─71e2a2e8-2b58-11eb-07eb-b522b7bea98c
# ╠═5bfc5448-29b3-11eb-1941-d9bff0f425c6
# ╠═c14e7db6-29b4-11eb-1687-315faa512117
# ╟─5baf5922-29b3-11eb-2a67-ef89bd104db8
# ╟─5bc23a06-29b3-11eb-062c-393396ede7f2
# ╟─5bc343b0-29b3-11eb-3e5a-df86461a783d
# ╟─f17a5ce2-2b55-11eb-37a2-4d56fad8b4b3
# ╟─71e3f3a6-2b55-11eb-2d2a-d5a604785ddf
# ╟─3975431a-2b54-11eb-0c7a-2b5d7878de97
# ╠═535d9692-29b3-11eb-1ef4-1f3c7f559481
# ╠═5bad3192-29b3-11eb-343d-a77cb1f89144
# ╠═8c677950-2b48-11eb-1843-356af0ab079d
# ╠═7ff472ae-29b3-11eb-267b-59058f5f5e85
# ╠═f5497004-29b3-11eb-3ead-5766e13c274a
# ╠═56ae2176-29b6-11eb-1c7f-39cd7ff91f75
# ╠═955137ba-29b6-11eb-1efc-1b919d6dfbde
# ╠═84af5abc-29b7-11eb-38d8-d747d7309c91
# ╠═fdabe84e-29b8-11eb-1556-157a0847c121
# ╠═a38e7f08-29bc-11eb-01f3-256e6c2313f9
# ╠═5bcbfe6a-29b3-11eb-2e7a-1bab02e6e094
# ╠═32731c08-2b49-11eb-3358-bf9fc08f6004
# ╠═41fffad2-2b49-11eb-2601-81a7941b0474
# ╠═5bd681a0-29b3-11eb-0ca5-630541f87c60
# ╠═3e51a966-2b34-11eb-035b-ddca47de3459
# ╠═4a71ec26-2b5a-11eb-2ad8-0bb517c71847
# ╠═c532b756-2b28-11eb-2d2f-c1ca7e2046e1
# ╠═e8185fae-2b29-11eb-1b03-d35e774da123
# ╠═5c3c6c04-29b3-11eb-2142-e344978fb8e0
# ╠═5c480da2-29b3-11eb-1e71-fb742ec0fc67
# ╠═3e6886d6-2b2a-11eb-1c5f-c786380bdc51
# ╠═5c537656-29b3-11eb-2ed4-0d0bd0650fab
# ╠═1ed870a8-29bb-11eb-3523-adcb5c6d9923
# ╠═169e540c-2b32-11eb-1172-47c12c009bd0
# ╠═214cf31e-2b27-11eb-1fbd-d50cfd38a1b7
# ╠═4a32067a-2b31-11eb-28c5-1519242d8284
# ╠═87722ef4-2b2a-11eb-1ffe-03fd75e12532
# ╠═7cb2ec38-2b34-11eb-0131-03c09f0f3581
# ╠═4e690274-2b23-11eb-2907-5546c62231d3
# ╠═b98a4cca-29d4-11eb-2e05-812d10497850
# ╠═507cb14e-2b1b-11eb-36b3-a134271c47ef
# ╠═5e3f5d40-2b1b-11eb-18e4-5557c5b69a29
# ╠═bd50e5fa-2b44-11eb-318f-cb0171004c46
# ╠═b5ef9b9a-2b20-11eb-2c94-5db769d419ad
# ╠═0001bfe2-2b49-11eb-1607-259b06c569e6
# ╠═4d1fe84e-2b49-11eb-230f-aba60a20c75a
# ╠═0f81c860-2b49-11eb-3113-031161588326
# ╠═53468264-2b49-11eb-3235-d166a7cf2b54
# ╠═ddc87348-2b21-11eb-1079-0f662e8190f5
# ╠═14d4d918-2b2b-11eb-31a8-051e0c39ce3f
# ╠═3ec9e128-2b35-11eb-3632-75687fc22283
# ╠═35e9872a-29b9-11eb-18c6-fd932571d10b
# ╠═5c626206-29b3-11eb-2cae-275536bd1ab6
# ╠═5c6f5238-29b3-11eb-1962-65fe06d73f08
# ╠═5c7b5d92-29b3-11eb-2c0b-01e0dbf269a7
# ╠═d750b1d6-2b37-11eb-1fa1-4fad6904a144
# ╠═c16ae7b0-2b37-11eb-271f-791043d98a3e
# ╟─f7c242ea-2b37-11eb-0389-07bef2f1e86c
# ╠═0dcffaec-2b35-11eb-1329-197b99f7c08f
