#!/usr/bin/env julia

# PFTM vs JW depth study (Julia)
# Mirrors the Python logic: parity schedulers + depth model + linear fits.
#
# Dependencies:
#   ] add DataFrames Statistics Printf
# Optional plotting:
#   ] add Plots

using Statistics
using Printf
using DataFrames

# ---------- Schedulers (parallel 2q-layer accounting) ----------

"""
    jw_chain_layers(prefix_len::Int) -> Vector{Vector{Tuple{Int,Int}}}

JW parity ladder into target i: all CNOTs share the same target ⇒ serial in 2q depth.
Returns a vector of layers; each layer is a vector of (ctrl, tgt) pairs.
The pairs are placeholders (structural only).
"""
function jw_chain_layers(prefix_len::Int)
    return [[(0, 1)] for _ in 1:prefix_len]
end

"""
    pftm_tree_layers(prefix_len::Int) -> Vector{Vector{Tuple{Int,Int}}}

PFTM binary reduction on prefix 0..i-1.
Level 0: (0→1), (2→3), ...
Level 1: (1→3), (5→7), ...
Level L: combine rightmost of left half → rightmost of whole block.
"""
function pftm_tree_layers(prefix_len::Int)
    layers = Vector{Vector{Tuple{Int,Int}}}()
    if prefix_len <= 1
        return layers
    end
    block = 2
    while block <= prefix_len
        half = block ÷ 2
        layer = Vector{Tuple{Int,Int}}()
        start = 0
        while start + block - 1 < prefix_len
            ctrl = start + half - 1
            tgt  = start + block - 1
            push!(layer, (ctrl, tgt))
            start += block
        end
        if !isempty(layer)
            push!(layers, layer)
        end
        block *= 2
    end
    return layers
end

"""
    depth_from_layers(layers; include_rotation=true) -> Int

Depth = compute + (local rotation) + uncompute.
We count the local Q_i^+ decomposition as a constant 2 single-qubit layers (H/Rz/H compressed).
"""
function depth_from_layers(layers; include_rotation::Bool=true)
    compute = length(layers)
    uncompute = length(layers)
    rot = include_rotation ? 2 : 0
    return compute + rot + uncompute
end

"""
    worst_case_depths(n_vals::AbstractVector{Int}) -> DataFrame

Measures worst-case term a_{n-1} for each n.
"""
function worst_case_depths(n_vals::AbstractVector{Int})
    rows = Vector{NamedTuple}()
    for n in n_vals
        i = n - 1
        jw_layers   = jw_chain_layers(i)
        pftm_layers = pftm_tree_layers(i)
        push!(rows, (
            n_qubits = n,
            prefix_len = i,
            jw_compute_layers = length(jw_layers),
            pftm_compute_layers = length(pftm_layers),
            jw_total_depth = depth_from_layers(jw_layers),
            pftm_total_depth = depth_from_layers(pftm_layers),
        ))
    end
    return DataFrame(rows)
end

# ---------- Depth study ----------

n_vals = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
df = worst_case_depths(n_vals)

# ---------- Fits ----------
# PFTM depth ~ a * log2(n) + b
x_log = log2.(Float64.(df.n_qubits))
y_p   = Float64.(df.pftm_total_depth)

# JW depth ~ a * n + b
x_lin = Float64.(df.n_qubits)
y_jw  = Float64.(df.jw_total_depth)

# simple linear regression via normal equations: minimize ||Aβ - y||_2
# where A = [x 1]
function linfit(x::Vector{Float64}, y::Vector{Float64})
    A = hcat(x, ones(length(x)))
    β = A \ y
    yhat = A * β
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = 1 - ss_res / ss_tot
    (slope = β[1], intercept = β[2], r2 = r2, yhat = yhat)
end

fit_p = linfit(x_log, y_p)
fit_j = linfit(x_lin, y_jw)

# ---------- Output ----------

println("\nDepth comparison (worst-case a_{n-1}):")
show(df, allrows=true, allcols=true)
println()

@printf "\nFits:\n"
@printf "PFTM depth ≈ %.3f * log2(n) + %.3f (R^2 = %.4f)\n" fit_p.slope fit_p.intercept fit_p.r2
@printf " JW  depth ≈ %.3f * n       + %.3f (R^2 = %.4f)\n" fit_j.slope fit_j.intercept fit_j.r2

# ---------- Optional plotting ----------
# using Plots
# default(fmt = :png)
# p1 = scatter(x_log, y_p, label="PFTM depth", xlabel="log2(n)", ylabel="depth",
#              title="PFTM depth scales ~ log2(n)")
# plot!(sort(x_log), fit_p.slope .* sort(x_log) .+ fit_p.intercept, label="fit")
# savefig("pftm_depth_vs_log2n.png")
#
# p2 = scatter(x_lin, y_jw, label="JW depth", xlabel="n", ylabel="depth",
#              title="JW depth scales ~ n")
# plot!(sort(x_lin), fit_j.slope .* sort(x_lin) .+ fit_j.intercept, label="fit")
# savefig("jw_depth_vs_n.png")
#
# println("\nSaved plots: pftm_depth_vs_log2n.png, jw_depth_vs_n.png")

