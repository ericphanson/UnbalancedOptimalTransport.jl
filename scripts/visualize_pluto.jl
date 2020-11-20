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
using UnbalancedOptimalTransport: KL

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

# ╔═╡ 6893fa88-29ba-11eb-28a0-b38baa194632
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

# ╔═╡ 5be87ed2-29b3-11eb-32e4-7f6093c1ed0c
x, y = make_measures(am, bm)

# ╔═╡ 9c05d88c-29b5-11eb-314f-ade1d0f49cea
C = [norm(x.set[i] - y.set[j]) for i = eachindex(x.set), j = eachindex(y.set)]

# ╔═╡ 5bfc5448-29b3-11eb-1941-d9bff0f425c6
D = KL(10.0)

# ╔═╡ c14e7db6-29b4-11eb-1687-315faa512117
ϵ = 0.01

# ╔═╡ 5c0ba9fc-29b3-11eb-398e-ddd48ee94fb6
coupling = optimal_coupling!(D, C, x, y, ϵ)

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

# ╔═╡ 5c626206-29b3-11eb-2cae-275536bd1ab6
lg(x) = x <= 0 ? zero(x) : log(x)

# ╔═╡ 5c6f5238-29b3-11eb-1962-65fe06d73f08
entropy(a::AbstractArray) = sum(x -> -x * lg(x), a)

# ╔═╡ 5c7b5d92-29b3-11eb-2c0b-01e0dbf269a7
function divergence(::KL{ρ}, a, b) where {ρ}
    ρ * (-entropy(a) - dot(a, lg.(b)) + sum(b - a))
end

# ╔═╡ 5c82d9d4-29b3-11eb-238b-abf06cf8c071
coupling_ext = extended_coupling_matrix(coupling, x.set, d, y.set, d)

# ╔═╡ 5c96fb42-29b3-11eb-3d19-37ca1f531e54
x_ext = extended_vec(x, d)

# ╔═╡ 3b838770-29b5-11eb-3e83-0d503da5ba4c
μ = vec(sum(coupling, dims=2))

# ╔═╡ 633b9244-29b5-11eb-21c0-d75199c68e31
ν = vec(sum(coupling, dims=1))

# ╔═╡ 5ca7ac44-29b3-11eb-2338-5d4885228fcc
μ_ext = vec(sum(coupling_ext, dims=2))

# ╔═╡ 40fc7b9e-29b5-11eb-0274-7f750fc3e298
extended_vec(μ, x.set,d) ≈ μ_ext

# ╔═╡ 5cb739de-29b3-11eb-2788-43ac65193ffd
ν_ext = vec(sum(coupling_ext, dims=1))

# ╔═╡ 5cbea318-29b3-11eb-3e61-7336bfb06ccc
y_ext = extended_vec(y, d)

# ╔═╡ 1386e1b6-29b4-11eb-0873-a30b6f04c673
C_ext = [ norm(i-j) for i = 1:d, j = 1:d]

# ╔═╡ 76744004-29b3-11eb-0c64-77aaeacad85e
cost_unbal1 = divergence(D, μ_ext, x_ext)

# ╔═╡ 323e34f8-29b5-11eb-1aa5-07f50fa23a45
divergence(D, μ, x.density)

# ╔═╡ 20fc7f02-29b4-11eb-000c-e9e3b10da84c
cost_transport = dot(C_ext, coupling_ext)

# ╔═╡ f2700c48-29b4-11eb-0c51-477938b2182d
dot(C, coupling)

# ╔═╡ 000a3dde-29b4-11eb-39cc-853693304093
cost_unbal2 = divergence(D, ν_ext, y_ext)

# ╔═╡ 6a7d543c-29b5-11eb-1f05-5d89621c05cf
divergence(D, ν, y.density)

# ╔═╡ 381db1f6-29b4-11eb-3e0c-c57c48820e76
cost_reg =  ϵ * divergence(KL(), vec(coupling_ext), kron(y_ext, x_ext))

# ╔═╡ d980f67a-29b4-11eb-0541-2dd4bd8e8dbb
total_cost = cost_unbal1 + cost_transport + cost_unbal2 + cost_reg

# ╔═╡ e2144ff0-29b4-11eb-020c-2dd0fd2ee6ad
@test total_cost ≈ OT!(D, x, y, ϵ; tol=1e-5) atol=1e-3

# ╔═╡ aa9dfc18-29b9-11eb-1fdf-25f9e0a979fb
bar_vals = Node(x_ext)

# ╔═╡ d92da068-29c0-11eb-12cb-f96655832b8a
bar_vals2 = Node(x_ext)

# ╔═╡ 004866c6-29bd-11eb-30c1-d7aecfa0fca3
title_text = Node("title")

# ╔═╡ dbb425fa-29c0-11eb-2ef6-d7afbae3b5fb
title_text2 = Node("title")

# ╔═╡ 1ed870a8-29bb-11eb-3523-adcb5c6d9923
begin
	convex_combination(x,y,t) = t .* y .+ (1 .- t) .* x
	convex_combination(x,y) = t -> convex_combination(x,y,t)
end

# ╔═╡ 793ed690-29bb-11eb-051d-e15da51965fe
μν_conv = convex_combination(μ_ext, ν_ext)

# ╔═╡ b997c4ca-29ba-11eb-0fe9-89101801ff09
#list = [x_ext, μ_ext, μν_conv(0.25), μν_conv(0.5), μν_conv(0.75),  ν_ext, y_ext]

# ╔═╡ 79e5e934-29bc-11eb-2893-3127d9778fc6
ab_conv = convex_combination(a - x_ext + μ_ext,  b - y_ext + ν_ext)

# ╔═╡ 1471492c-29bc-11eb-3c15-bb1793756268
list = [(a, "a"), (ab_conv(0), "Create/destroy mass"), (ab_conv(0.25), "Transport 25%"), (ab_conv(0.5), "Transport 50%"), (ab_conv(0.75), "Transport 75%"),  (ab_conv(1), "Transport 100%"), (b, "Create/destroy mass"), (b, "b")]

# ╔═╡ d63617ec-29ba-11eb-3fed-812bc2a2782b
ex = let
	exs = extrema.(first.(list))
	minimum(first.(exs)) - 1, maximum(last.(exs)) + 1
end

# ╔═╡ b98a4cca-29d4-11eb-2e05-812d10497850
FONT = "Julia Mono"

# ╔═╡ 507cb14e-2b1b-11eb-36b3-a134271c47ef
plus = '₊'

# ╔═╡ 5e3f5d40-2b1b-11eb-18e4-5557c5b69a29
minus = '₋'

# ╔═╡ 5c672632-29c0-11eb-1d7b-dd324cb662a4
list2 = [(x_ext, "a$plus + b$minus"), (μν_conv(0), "Create/destroy mass"), (μν_conv(0.25), "Transport 25%"), (μν_conv(0.5), "Transport 50%"), (μν_conv(0.75), "Transport 75%"),  (μν_conv(1), "Transport 100%"), (y_ext, "Create/destroy mass"), (y_ext, "b$plus + a$minus")]

# ╔═╡ 2a5875bc-29c1-11eb-20d0-47f8c5e3a5da
ex2 = let
	exs = extrema.(first.(list2))
	minimum(first.(exs)) - .2, maximum(last.(exs)) + 1
end

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
function barplot_pos_neg!(scene, v, name; kwargs...)
	layout = GridLayout()
	ylims = (minimum(v)-1, maximum(v)+1)
	
	l1, ax1 = barplot_with_title!(scene, v, name; ylims, kwargs...)
	l2, ax2 = barplot_with_title!(scene, max.(v, 0), "$name$minus"; ylims, kwargs...)
	l3, ax3 = barplot_with_title!(scene, -min.(v, 0), "$name$plus"; ylims, kwargs...)
	linkyaxes!(ax1, ax2, ax3)
	layout[1, 1:3] = [l1, l2, l3]
	return layout
end

# ╔═╡ 64d9a390-29ba-11eb-211d-8ff252146626
@bind t PlutoUI.Clock()

# ╔═╡ 14c2f17e-29bb-11eb-3e39-d99c9eb1f1ef
t

# ╔═╡ e552abf4-29c5-11eb-2c7e-33ba6bd2a502
ts = 1:max(length(list), length(list2))

# ╔═╡ 35e9872a-29b9-11eb-18c6-fd932571d10b
colors = AbstractPlotting.current_default_theme()[:palette].color[]

# ╔═╡ 9627c1f0-29d3-11eb-2871-1ba3530f91f4
initial_final_scene = let
	scene, layout = layoutscene()
	layout[1,1] = barplot_pos_neg!(scene, a, "a"; color = colors[1])
	layout[2,1] = barplot_pos_neg!(scene, b, "b"; color = colors[1])
	scene
end

# ╔═╡ 8d29425c-2b22-11eb-1c7a-07a74ee372c3
joint_scene = let
	kwargs = (; color = colors[1])
	scene, layout = layoutscene()

	l11, ax11 = barplot_with_title!(scene, list2[1][1],  list2[1][2]; kwargs...)
	l12, ax12 = barplot_with_title!(scene, bar_vals2, title_text2; kwargs...)
	l13, ax13 = barplot_with_title!(scene, list2[end][1], list2[end][2]; kwargs...)
	linkyaxes!(ax11, ax12, ax13)
	layout[1, 1:3] = [l11, l12, l13]
	
	l21, ax21 = barplot_with_title!(scene, list[1][1],  list[1][2]; kwargs...)
	l22, ax22 = barplot_with_title!(scene, bar_vals, title_text; kwargs...)
	l23, ax23 = barplot_with_title!(scene, list[end][1], list[end][2]; kwargs...)
	linkyaxes!(ax21, ax22, ax23)
	layout[2, 1:3] = [l21, l22, l23]
	scene
end

# ╔═╡ de79c2a2-29bf-11eb-0ba7-5574d8c69216
let
	vals, title = list[min(end, t)]
	bar_vals[] = vals
	title_text[] = title
	
	vals2, title2 = list2[min(end, t)]
	bar_vals2[] = vals2
	title_text2[] = title2
	joint_scene
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
# ╠═5be87ed2-29b3-11eb-32e4-7f6093c1ed0c
# ╠═9c05d88c-29b5-11eb-314f-ade1d0f49cea
# ╠═5bfc5448-29b3-11eb-1941-d9bff0f425c6
# ╠═c14e7db6-29b4-11eb-1687-315faa512117
# ╠═5c0ba9fc-29b3-11eb-398e-ddd48ee94fb6
# ╠═5c3c6c04-29b3-11eb-2142-e344978fb8e0
# ╠═5c480da2-29b3-11eb-1e71-fb742ec0fc67
# ╠═5c537656-29b3-11eb-2ed4-0d0bd0650fab
# ╠═5c626206-29b3-11eb-2cae-275536bd1ab6
# ╠═5c6f5238-29b3-11eb-1962-65fe06d73f08
# ╠═5c7b5d92-29b3-11eb-2c0b-01e0dbf269a7
# ╠═5c82d9d4-29b3-11eb-238b-abf06cf8c071
# ╠═5c96fb42-29b3-11eb-3d19-37ca1f531e54
# ╠═3b838770-29b5-11eb-3e83-0d503da5ba4c
# ╠═633b9244-29b5-11eb-21c0-d75199c68e31
# ╠═40fc7b9e-29b5-11eb-0274-7f750fc3e298
# ╠═5ca7ac44-29b3-11eb-2338-5d4885228fcc
# ╠═5cb739de-29b3-11eb-2788-43ac65193ffd
# ╠═5cbea318-29b3-11eb-3e61-7336bfb06ccc
# ╠═1386e1b6-29b4-11eb-0873-a30b6f04c673
# ╠═76744004-29b3-11eb-0c64-77aaeacad85e
# ╠═323e34f8-29b5-11eb-1aa5-07f50fa23a45
# ╠═20fc7f02-29b4-11eb-000c-e9e3b10da84c
# ╠═f2700c48-29b4-11eb-0c51-477938b2182d
# ╠═000a3dde-29b4-11eb-39cc-853693304093
# ╠═6a7d543c-29b5-11eb-1f05-5d89621c05cf
# ╠═381db1f6-29b4-11eb-3e0c-c57c48820e76
# ╠═d980f67a-29b4-11eb-0541-2dd4bd8e8dbb
# ╠═e2144ff0-29b4-11eb-020c-2dd0fd2ee6ad
# ╠═aa9dfc18-29b9-11eb-1fdf-25f9e0a979fb
# ╠═d92da068-29c0-11eb-12cb-f96655832b8a
# ╠═004866c6-29bd-11eb-30c1-d7aecfa0fca3
# ╠═dbb425fa-29c0-11eb-2ef6-d7afbae3b5fb
# ╠═6893fa88-29ba-11eb-28a0-b38baa194632
# ╠═1ed870a8-29bb-11eb-3523-adcb5c6d9923
# ╠═793ed690-29bb-11eb-051d-e15da51965fe
# ╠═b997c4ca-29ba-11eb-0fe9-89101801ff09
# ╠═79e5e934-29bc-11eb-2893-3127d9778fc6
# ╠═1471492c-29bc-11eb-3c15-bb1793756268
# ╠═5c672632-29c0-11eb-1d7b-dd324cb662a4
# ╠═2a5875bc-29c1-11eb-20d0-47f8c5e3a5da
# ╠═d63617ec-29ba-11eb-3fed-812bc2a2782b
# ╠═b98a4cca-29d4-11eb-2e05-812d10497850
# ╠═507cb14e-2b1b-11eb-36b3-a134271c47ef
# ╠═5e3f5d40-2b1b-11eb-18e4-5557c5b69a29
# ╠═9627c1f0-29d3-11eb-2871-1ba3530f91f4
# ╠═4e690274-2b23-11eb-2907-5546c62231d3
# ╠═b5ef9b9a-2b20-11eb-2c94-5db769d419ad
# ╠═ddc87348-2b21-11eb-1079-0f662e8190f5
# ╠═64d9a390-29ba-11eb-211d-8ff252146626
# ╠═14c2f17e-29bb-11eb-3e39-d99c9eb1f1ef
# ╠═de79c2a2-29bf-11eb-0ba7-5574d8c69216
# ╠═e552abf4-29c5-11eb-2c7e-33ba6bd2a502
# ╠═8d29425c-2b22-11eb-1c7a-07a74ee372c3
# ╠═35e9872a-29b9-11eb-18c6-fd932571d10b
