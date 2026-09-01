# plot3d.jl — 仿星器（非轴对称）Poincaré 图 + 诊断曲线
# 用法: julia --project=. plot3d.jl data/run_C.jld2
# 输出: data/run_C_poincare.png, data/run_C_diag.png
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

length(ARGS) == 1 || error("用法: julia --project=. plot3d.jl data/run_C.jld2")
file = ARGS[1]
dat = load(file)
n, p, ε, κ = dat["n"], dat["p"], dat["eps"], dat["kappa"]
nfp, nzeta = dat["nfp"], dat["nzeta"]
b = dat["b"]
dat["geometry"] == "stellarator" || error("$file 不是仿星器算例，请用 plot_.jl")

println("重建算子 (n=$n, p=$p, nζ=$nzeta, ε=$ε, κ=$κ, n_fp=$nfp) ...")
mapping = stellarator_mapping(ε, κ, nfp)
ops = setup_operators((n, n, nzeta), (p, p, p), mapping)

println("三维场重构 ...")
fl = reconstruct_bhat3(ops, b)
let (num, ref) = fl.flux_check
    println("  环向通量对照: ∫∫∫B̂_ζ = $num vs 装配值 $ref " *
            "(相对差 $(abs(num - ref) / abs(ref)))")
    abs(num - ref) / abs(ref) < 1e-3 || @warn "2-形式求值与装配的环向通量不一致"
end
Br_f = trilinear(fl.rg, fl.tg, fl.zg, fl.Br)
Bt_f = trilinear(fl.rg, fl.tg, fl.zg, fl.Bt)
Bz_f = trilinear(fl.rg, fl.tg, fl.zg, fl.Bz)

# 截面：ζ = 0 与 ζ = 1/(2 n_fp)（半个场周期，形状差别最大的两个截面）
sections = (0.0, 1 / (2 * nfp))
RZ(r, t, z) = stellarator_position(ε, κ, nfp, r, t, z)

println("场线追踪（每条 300 圈，8 个子截面）...")
const NSUB = 8
seeds = range(0.06, 0.95; length=26)

function trace_at(ζ0, r0; transits=300)
    sh(f) = (r, t, z) -> f(r, t, z + ζ0)
    return trace_line3(sh(Br_f), sh(Bt_f), sh(Bz_f), r0, 0.0;
        transits=transits, nsub=NSUB)
end
# 子截面 m（1..NSUB）对应的真实环向角与该处的 (R,z)
sub_zeta(ζ0, m) = ζ0 + m / NSUB
RZsub_at(ζ0) = (r, t, m) -> RZ(r, t, sub_zeta(ζ0, m))

data = map(sections) do ζ0
    # 磁轴参考点：贴近轴的一条场线在各子截面上的平均位置
    # （3D 中磁轴是闭合曲线，每个子截面上的参考点不同）。
    # ι 是绕数，只要参考点落在所有磁面内部，取得粗一点也不影响结果。
    axpts = SVector{2, Float64}[]
    for rseed in (0.03, 0.05, 0.08, 0.12)
        pts, _ = trace_at(ζ0, rseed; transits=60)
        length(pts) > 10NSUB && (axpts = pts; break)
    end
    isempty(axpts) && error("无法定位磁轴参考点（ζ=$ζ0）")
    axis = map(1:NSUB) do m
        sel = [axpts[i] for i in eachindex(axpts) if mod1(i, NSUB) == m]
        isempty(sel) && return (1.0, 0.0)
        rs = [RZsub_at(ζ0)(u[1], u[2], m) for u in sel]
        (sum(first, rs) / length(rs), sum(last, rs) / length(rs))
    end
    println("  ζ=$(round(ζ0, digits=3)) 磁轴 ≈ (R,z) = " *
            "($(round(axis[NSUB][1], digits=4)), $(round(axis[NSUB][2], digits=4)))")

    Rs, Zs, Is = Float64[], Float64[], Float64[]
    iotas = Float64[]
    for r0 in seeds
        pts, _ = trace_at(ζ0, r0)
        length(pts) > 4NSUB || continue
        # 剔除坐在磁轴上的退化种子：绕数无定义，会给出 ι≈0 污染色标
        mean_orbit_radius3(pts, RZsub_at(ζ0), axis, NSUB) > 5e-3 || continue
        ι = abs(iota_from_track(pts, RZsub_at(ζ0), axis, NSUB))
        isfinite(ι) || continue
        push!(iotas, ι)
        for i in NSUB:NSUB:length(pts)          # 只取整圈处的点作 Poincaré 截面
            R, z = RZ(pts[i][1], pts[i][2], ζ0)
            push!(Rs, R); push!(Zs, z); push!(Is, ι)
        end
    end
    println("    $(length(Rs)) 个打点, iota ∈ $(round.(extrema(iotas), sigdigits=4))")
    return (Rs, Zs, Is)
end

# ---------- Poincaré 图 ----------
fig = Figure(size=(1250, 640))
for (i, ζ0) in enumerate(sections)
    Rs, Zs, Is = data[i]
    ax = Axis(fig[1, 2i - 1]; xlabel="R", ylabel="z", aspect=DataAspect(),
        title="Poincare at zeta = $(round(ζ0, digits=3))")
    sc = scatter!(ax, Rs, Zs; color=Is, colormap=:turbo, markersize=2)
    bd = [RZ(1.0, t, ζ0) for t in range(0, 1; length=400)]
    lines!(ax, first.(bd), last.(bd); color=:gray)
    Colorbar(fig[1, 2i], sc; label="iota")
end
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
