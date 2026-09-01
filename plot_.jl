# plot_.jl — Poincaré 磁面图 + 诊断曲线（GLMakie）
# 用法: julia --project=. plot_.jl data/run_A.jld2
# 输出: data/run_A_poincare.png, data/run_A_diag.png
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays, JLD2, GLMakie
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/operators.jl")
include("src/solovev.jl")
include("src/relaxation.jl")
include("src/fieldline.jl")

length(ARGS) == 1 || error("用法: julia --project=. plot_.jl data/run_A.jld2")
file = ARGS[1]
dat = load(file)
n, p, q_star, ε = dat["n"], dat["p"], dat["q_star"], dat["eps"]
κ = haskey(dat, "kappa") ? dat["kappa"] : 1.0
δ = haskey(dat, "delta") ? dat["delta"] : 0.0
b = dat["b"]

println("重建算子 (n=$n, p=$p, ε=$ε, κ=$κ, δ=$δ) ...")
ops = setup_operators((n, n, 1), (p, p, 1), torus_mapping(ε, κ, δ))
_, pcoef, _, _, _ = force_velocity(ops, b)   # 最终态压强（Leray 副产品，V³）

# 单元索引/局部坐标约定自检
let mp = torus_mapping(ε, κ, δ)
    pts = Mantis.Points.CartesianPoints(([0.3], [0.2], [0.7]))
    x2 = Mantis.Geometry.evaluate(ops.geom, 2, pts)   # 单元 2 = r 方向第二个
    @assert maximum(abs.(vec(x2) .- mp.mapping([1.3 / n, 0.2 / n, 0.7]))) < 1e-12
end

println("网格求值与场重构 ...")
fl = reconstruct_bhat(ops, b, ε, κ, δ)
rg, tg = fl.rg, fl.tg
Pphys = scalar_from_3form(ops, pcoef, ε, κ, δ, rg, tg, fl.R, fl.W)
println("  谐和系数 c = $(fl.c), ∫∫curlz = $(fl.curlz_int)")

Br_f, Bt_f, Bz_f = bilinear(rg, tg, fl.Br), bilinear(rg, tg, fl.Bt), bilinear(rg, tg, fl.Bz)
P_f = bilinear(rg, tg, Pphys)
RZ(r, t) = poloidal_position(ε, κ, δ, r, t)

R_ax, z_ax, _ = find_axis(Pphys, rg, tg, ε, κ, δ)
println("  磁轴 ≈ (R,z) = ($R_ax, $z_ax)")

println("场线追踪 ...")
Rs, Zs, Pc, Ic, iotas = Float64[], Float64[], Float64[], Float64[], Float64[]
for r0 in range(0.10, 0.96; length=28)
    pts, ι = trace_line(Br_f, Bt_f, Bz_f, RZ, R_ax, z_ax, r0)
    isempty(pts) && continue
    mean_orbit_radius(pts, RZ, R_ax, z_ax) > 5e-3 || continue  # 剔除坐在磁轴上的退化种子
    push!(iotas, abs(ι))
    for u in pts
        R, z = RZ(u[1], u[2])
        push!(Rs, R); push!(Zs, z)
        push!(Pc, P_f(u[1], u[2])); push!(Ic, abs(ι))
    end
end
println("  iota 范围: ", extrema(filter(isfinite, iotas)))

# ---------- Poincaré 图 ----------
bd = [poloidal_position(ε, κ, δ, 1.0, t) for t in range(0, 1; length=400)]
bdR, bdZ = first.(bd), last.(bd)
fig = Figure(size=(1250, 620))
ax1 = Axis(fig[1, 1]; xlabel="R", ylabel="z", title="Poincare (pressure)",
    aspect=DataAspect())
sc1 = scatter!(ax1, Rs, Zs; color=Pc, colormap=:plasma, markersize=2)
lines!(ax1, bdR, bdZ; color=:gray)
Colorbar(fig[1, 2], sc1; label="p")
ax2 = Axis(fig[1, 3]; xlabel="R", ylabel="z", title="Poincare (iota)",
    aspect=DataAspect())
sc2 = scatter!(ax2, Rs, Zs; color=Ic, colormap=:turbo, markersize=2)
lines!(ax2, bdR, bdZ; color=:gray)
Colorbar(fig[1, 4], sc2; label="iota")
out1 = replace(file, ".jld2" => "_poincare.png")
save(out1, fig)
println("已保存 $out1")

# ---------- 诊断曲线 ----------
force, energy, divB = dat["force"], dat["energy"], dat["divB"]
hit, hel, flux = dat["helicity_iter"], dat["helicity"], dat["flux"]
fig2 = Figure(size=(1100, 700))
ax = Axis(fig2[1, 1]; yscale=log10, xlabel="iter",
    title="force residual |JxB-grad p|/|grad p|")
lines!(ax, 1:length(force), force)
ax = Axis(fig2[1, 2]; xlabel="iter", title="energy")
lines!(ax, 1:length(energy), energy)
ax = Axis(fig2[2, 1]; yscale=log10, xlabel="iter", title="|div B|")
lines!(ax, 1:length(divB), max.(divB, 1e-20))
ax = Axis(fig2[2, 2]; yscale=log10, xlabel="iter", title="conservation drift")
lines!(ax, hit, max.(abs.(flux .- flux[1]) ./ abs(flux[1]), 1e-17); label="toroidal flux (rel)")
lines!(ax, hit, max.(abs.(hel .- hel[1]), 1e-17); label="helicity (abs)")
axislegend(ax; position=:lt)
out2 = replace(file, ".jld2" => "_diag.png")
save(out2, fig2)
println("已保存 $out2")
