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

# ╔═╡ 5baf5922-29b3-11eb-2a67-ef89bd104db8
d = 10

# ╔═╡ 5bc23a06-29b3-11eb-062c-393396ede7f2
a = zeros(d)

# ╔═╡ a38e7f08-29bc-11eb-01f3-256e6c2313f9
rand_unit(d) = (a =rand(d); a / sum(a) )

# ╔═╡ 5bc343b0-29b3-11eb-3e5a-df86461a783d
b = 10 .* ( rand_unit(d) - rand_unit(d))

# ╔═╡ b8e4fd28-29bc-11eb-0b85-bfbbf8a3c3d4
sum(b)

# ╔═╡ 5bcbfe6a-29b3-11eb-2e7a-1bab02e6e094
am = SignedMeasure(a)

# ╔═╡ 5bd681a0-29b3-11eb-0ca5-630541f87c60
bm = SignedMeasure(b)

# ╔═╡ 5bfc5448-29b3-11eb-1941-d9bff0f425c6
#D = KL(10.0)
D = Balanced()

# ╔═╡ c14e7db6-29b4-11eb-1687-315faa512117
ϵ = 0.001

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

# ╔═╡ 3e6886d6-2b2a-11eb-1c5f-c786380bdc51
extended_vec(sm::SignedMeasure, d) = extended_vec(sm.pos, d) - extended_vec(sm.neg, d)

# ╔═╡ e8185fae-2b29-11eb-1b03-d35e774da123
function get_transitions(D::UnbalancedOptimalTransport.AbstractDivergence, ϵ, a::SignedMeasure, b::SignedMeasure, d)
	x, y = make_measures(a, b)
	
	x_ext, μ_ext, ν_ext, y_ext = get_transitions(D, ϵ, x, y, d)
	
	a_ext = extended_vec(a, d)
	b_ext = extended_vec(b, d)
	
	return a_ext, a_ext - x_ext + μ_ext,  b_ext - y_ext + ν_ext, b_ext
end

# ╔═╡ 5c537656-29b3-11eb-2ed4-0d0bd0650fab
function extended_coupling_matrix(coupling_matrix, set1, d1, set2, d2)
    m = zeros(d1, d2)
    for i in axes(coupling_matrix, 1), j in axes(coupling_matrix, 2)
        m[set1[i], set2[j]] = coupling_matrix[i,j]
    end
    return m
end

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

# ╔═╡ 5c0ba9fc-29b3-11eb-398e-ddd48ee94fb6
get_transitions(D, ϵ, am, bm, d)

# ╔═╡ 993992be-2b37-11eb-0b1c-3ff64225a3c8
t1, t2, t3, t4 = get_transitions(D, ϵ, make_measures(am, bm)..., d)

# ╔═╡ af2f3dee-2b37-11eb-05ac-25320f758490
t1

# ╔═╡ a5391d50-2b37-11eb-34f6-2f5bc3e5588c
t4

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

# ╔═╡ d750b1d6-2b37-11eb-1fa1-4fad6904a144
t0 = Ref(0)

# ╔═╡ c16ae7b0-2b37-11eb-271f-791043d98a3e
@bind click PlutoUI.Button()

# ╔═╡ 345b8e68-2b36-11eb-02ec-f90efaf891aa
extrema(extended_vec(bm, d))

# ╔═╡ b98a4cca-29d4-11eb-2e05-812d10497850
FONT = "Julia Mono"

# ╔═╡ 507cb14e-2b1b-11eb-36b3-a134271c47ef
plus = '₊'

# ╔═╡ 5e3f5d40-2b1b-11eb-18e4-5557c5b69a29
minus = '₋'

# ╔═╡ be58708c-2b26-11eb-2be3-e5a5b6d06324
md"""
Consider the task of optimal transporting the zero vector to a vector which has equal amounts of positive and negative mass.
"""

# ╔═╡ 4e690274-2b23-11eb-2907-5546c62231d3
begin
	maybe_lift(f, x::Observable) = lift(f, x)
	maybe_lift(f, x) = f(x)
end

# ╔═╡ b5ef9b9a-2b20-11eb-2c94-5db769d419ad
function barplot_with_title!(scene, data, title; kwargs...)
	layout = GridLayout()
	layout[1, 1] = t = LText(scene, title, textsize = 30, font = FONT, color = (:black, 0.25))
	t.tellwidth = false

	layout[2, 1] = ax = LAxis(scene)
	AbstractPlotting.barplot!(ax, maybe_lift(x -> 1:length(x), data), data; kwargs...)
	return (layout, ax)
end


# ╔═╡ ddc87348-2b21-11eb-1079-0f662e8190f5
function barplot_pos_neg!(scene, v, name; ylims = (minimum(v)-1, maximum(v)+1), kwargs...)
	layout = GridLayout()	
	l1, ax1 = barplot_with_title!(scene, v, name; ylims, kwargs...)
	l2, ax2 = barplot_with_title!(scene, max.(v, 0), "$name$minus"; ylims, kwargs...)
	l3, ax3 = barplot_with_title!(scene, -min.(v, 0), "$name$plus"; ylims, kwargs...)
	linkyaxes!(ax1, ax2, ax3)
	layout[1, 1:3] = [l1, l2, l3]
	return layout
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
function transport_plot!(scene, a_name, b_name, transitions; kwargs...)
	
	a, a0, b0, b = transitions
	
	list = make_list(a, a_name, b, b_name, convex_combination(a0, b0))
	@info length(list)

	title_text = Node(" ")
	bar_vals = Node(a)
	
	 up = function(t)
		vals, title = list[mod1(t, end)]
		bar_vals[] = vals
		title_text[] = title
		return nothing
	end
			
	layout = GridLayout()

	layout[1, 1:3] = _transition_plot!(scene, list, bar_vals, title_text; kwargs...)
	return layout, up
end

# ╔═╡ 7cb2ec38-2b34-11eb-0131-03c09f0f3581
function transport_anim_both!(scene, a, aname, b, bname; kwargs...)
	layout = GridLayout()
	
	x, y = make_measures(a,b)
	l1, up1 = transport_plot!(scene, "$aname$plus + $bname$minus", "$bname$plus + $aname$minus", get_transitions(D, ϵ, x, y, d); kwargs...)
	
	layout[1,1] = l1
	
	l2, up2 = transport_plot!(scene, "a", "b", get_transitions(D, ϵ, a, b, d); kwargs...)
	layout[2,1] = l2
	
	up = t -> (up1(t); up2(t); nothing)
	
	return layout, up
end

# ╔═╡ 64d9a390-29ba-11eb-211d-8ff252146626
@bind t PlutoUI.Clock()

# ╔═╡ 14c2f17e-29bb-11eb-3e39-d99c9eb1f1ef
t

# ╔═╡ 35e9872a-29b9-11eb-18c6-fd932571d10b
colors = AbstractPlotting.current_default_theme()[:palette].color[]

# ╔═╡ 3e51a966-2b34-11eb-035b-ddca47de3459
signed_transport_scene, up = let
	scene, layout = layoutscene()
	l, u = transport_anim_both!(scene, am, "a", bm, "b"; color = colors[3])
	layout[1, 1] = l
	scene, u
end

# ╔═╡ 0dcffaec-2b35-11eb-1329-197b99f7c08f
begin
	click
	up(t0[] += 1)
	t0val = t0[]
	signed_transport_scene
end

# ╔═╡ f7c242ea-2b37-11eb-0389-07bef2f1e86c
mod1(t0val, D isa Balanced ? 6 : 8)

# ╔═╡ de79c2a2-29bf-11eb-0ba7-5574d8c69216
begin
	up(t)
	signed_transport_scene
end

# ╔═╡ 9627c1f0-29d3-11eb-2871-1ba3530f91f4
initial_final_scene = let
	scene, layout = layoutscene()
	layout[1,1] = barplot_pos_neg!(scene, a, "a"; color = colors[1])
	layout[2,1] = barplot_pos_neg!(scene, b, "b"; color = colors[1])
	scene
end

# ╔═╡ 5c626206-29b3-11eb-2cae-275536bd1ab6
lg(x) = x <= 0 ? zero(x) : log(x)

# ╔═╡ 5c6f5238-29b3-11eb-1962-65fe06d73f08
entropy(a::AbstractArray) = sum(x -> -x * lg(x), a)

# ╔═╡ 5c7b5d92-29b3-11eb-2c0b-01e0dbf269a7
function divergence(::KL{ρ}, a, b) where {ρ}
    ρ * (-entropy(a) - dot(a, lg.(b)) + sum(b - a))
end

# ╔═╡ Cell order:
# ╠═535d9692-29b3-11eb-1ef4-1f3c7f559481
# ╠═5bad3192-29b3-11eb-343d-a77cb1f89144
# ╠═7ff472ae-29b3-11eb-267b-59058f5f5e85
# ╠═f5497004-29b3-11eb-3ead-5766e13c274a
# ╠═56ae2176-29b6-11eb-1c7f-39cd7ff91f75
# ╠═955137ba-29b6-11eb-1efc-1b919d6dfbde
# ╠═84af5abc-29b7-11eb-38d8-d747d7309c91
# ╠═fdabe84e-29b8-11eb-1556-157a0847c121
# ╠═5baf5922-29b3-11eb-2a67-ef89bd104db8
# ╠═5bc23a06-29b3-11eb-062c-393396ede7f2
# ╠═a38e7f08-29bc-11eb-01f3-256e6c2313f9
# ╠═5bc343b0-29b3-11eb-3e5a-df86461a783d
# ╠═b8e4fd28-29bc-11eb-0b85-bfbbf8a3c3d4
# ╠═5bcbfe6a-29b3-11eb-2e7a-1bab02e6e094
# ╠═5bd681a0-29b3-11eb-0ca5-630541f87c60
# ╠═5bfc5448-29b3-11eb-1941-d9bff0f425c6
# ╠═c14e7db6-29b4-11eb-1687-315faa512117
# ╠═5c0ba9fc-29b3-11eb-398e-ddd48ee94fb6
# ╠═c532b756-2b28-11eb-2d2f-c1ca7e2046e1
# ╠═e8185fae-2b29-11eb-1b03-d35e774da123
# ╠═993992be-2b37-11eb-0b1c-3ff64225a3c8
# ╠═af2f3dee-2b37-11eb-05ac-25320f758490
# ╠═a5391d50-2b37-11eb-34f6-2f5bc3e5588c
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
# ╠═3e51a966-2b34-11eb-035b-ddca47de3459
# ╠═d750b1d6-2b37-11eb-1fa1-4fad6904a144
# ╠═c16ae7b0-2b37-11eb-271f-791043d98a3e
# ╟─f7c242ea-2b37-11eb-0389-07bef2f1e86c
# ╠═0dcffaec-2b35-11eb-1329-197b99f7c08f
# ╠═345b8e68-2b36-11eb-02ec-f90efaf891aa
# ╠═b98a4cca-29d4-11eb-2e05-812d10497850
# ╠═507cb14e-2b1b-11eb-36b3-a134271c47ef
# ╠═5e3f5d40-2b1b-11eb-18e4-5557c5b69a29
# ╠═be58708c-2b26-11eb-2be3-e5a5b6d06324
# ╠═9627c1f0-29d3-11eb-2871-1ba3530f91f4
# ╠═4e690274-2b23-11eb-2907-5546c62231d3
# ╠═b5ef9b9a-2b20-11eb-2c94-5db769d419ad
# ╠═ddc87348-2b21-11eb-1079-0f662e8190f5
# ╠═14d4d918-2b2b-11eb-31a8-051e0c39ce3f
# ╠═3ec9e128-2b35-11eb-3632-75687fc22283
# ╠═64d9a390-29ba-11eb-211d-8ff252146626
# ╠═14c2f17e-29bb-11eb-3e39-d99c9eb1f1ef
# ╠═de79c2a2-29bf-11eb-0ba7-5574d8c69216
# ╠═35e9872a-29b9-11eb-18c6-fd932571d10b
# ╠═5c626206-29b3-11eb-2cae-275536bd1ab6
# ╠═5c6f5238-29b3-11eb-1962-65fe06d73f08
# ╠═5c7b5d92-29b3-11eb-2c0b-01e0dbf269a7
