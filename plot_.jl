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

length(ARGS) == 1 || error("用法: julia --project=. plot_.jl data/run_A.jld2")
file = ARGS[1]
dat = load(file)
n, p, q_star, ε = dat["n"], dat["p"], dat["q_star"], dat["eps"]
b = dat["b"]

println("重建算子 (n=$n, p=$p) ...")
ops = setup_operators((n, n, 1), (p, p, 1), circular_torus_mapping(ε))
_, pcoef, _, _, _ = force_velocity(ops, b)   # 最终态压强（Leray 副产品，V³）

# ---------- 单元索引/局部坐标约定自检 ----------
let nel_r = ops.nel[1], nel_t = ops.nel[2]
    pts = Mantis.Points.CartesianPoints(([0.3], [0.2], [0.7]))
    x = Mantis.Geometry.evaluate(ops.geom, 1, pts)           # 单元 1 = 左下角
    Φref = circular_torus_mapping(ε).mapping([0.3 / nel_r, 0.2 / nel_t, 0.7])
    @assert maximum(abs.(vec(x) .- Φref)) < 1e-12 "单元 1 局部坐标约定不符"
    x2 = Mantis.Geometry.evaluate(ops.geom, 2, pts)          # 单元 2 = r 方向第二个
    Φref2 = circular_torus_mapping(ε).mapping([1.3 / nel_r, 0.2 / nel_t, 0.7])
    @assert maximum(abs.(vec(x2) .- Φref2)) < 1e-12 "单元线性索引应 r 最快"
end

# ---------- (r,θ) 网格求值（ζ=0 截面，轴对称）----------
# 返回逻辑 pullback 分量（2-形式为循环序 proxy (B̂r, B̂θ, B̂ζ)；3-形式为 p̂ = p·detJ）
function eval_grid(ops, k, coeffs; nr=241, nt=289, rmin=0.0)
    field = form_field(ops, k, coeffs)
    nel_r, nel_t = ops.nel[1], ops.nel[2]
    rg = collect(range(rmin, 1.0; length=nr))
    tg = collect(range(0.0, 1.0; length=nt))
    elr = [min(floor(Int, r * nel_r) + 1, nel_r) for r in rg]
    xir = [rg[i] * nel_r - (elr[i] - 1) for i in 1:nr]
    elt = [min(floor(Int, t * nel_t) + 1, nel_t) for t in tg]
    xit = [tg[j] * nel_t - (elt[j] - 1) for j in 1:nt]
    ncomp = k == 2 ? 3 : 1
    out = [zeros(nr, nt) for _ in 1:ncomp]
    for et in 1:nel_t, er in 1:nel_r
        is = findall(==(er), elr)
        js = findall(==(et), elt)
        (isempty(is) || isempty(js)) && continue
        pts = Mantis.Points.CartesianPoints((xir[is], xit[js], [0.5]))
        eid = er + nel_r * (et - 1)
        vals, _ = Mantis.Forms.evaluate(field, eid, pts)
        for c in 1:ncomp
            out[c][is, js] .= reshape(vals[c][:, 1], length(is), length(js))
        end
    end
    return rg, tg, out
end

println("网格求值 ...")
rg, tg, Bhat = eval_grid(ops, 2, b)
rgp, tgp, Phat = eval_grid(ops, 3, pcoef; rmin=1e-3)
detJ(r, t) = 4π^2 * ε^2 * r * (1 + ε * r * cospi(2t))
Pphys = [Phat[1][i, j] / detJ(rgp[i], tgp[j]) for i in 1:length(rgp), j in 1:length(tgp)]

# ---------- 双线性插值（θ 周期已由网格含端点 1.0 覆盖）----------
function bilinear(rg, tg, F)
    nr, nt = length(rg), length(tg)
    dr, dtt = rg[2] - rg[1], tg[2] - tg[1]
    r0 = rg[1]
    return function (r, t)
        tm = mod(t, 1.0)
        ir = clamp(floor(Int, (r - r0) / dr) + 1, 1, nr - 1)
        it = clamp(floor(Int, tm / dtt) + 1, 1, nt - 1)
        fr = clamp((r - rg[ir]) / dr, 0.0, 1.0)
        ft = (tm - tg[it]) / dtt
        return (1 - fr) * (1 - ft) * F[ir, it] + fr * (1 - ft) * F[ir + 1, it] +
               (1 - fr) * ft * F[ir, it + 1] + fr * ft * F[ir + 1, it + 1]
    end
end
Br = bilinear(rg, tg, Bhat[1])
Bt = bilinear(rg, tg, Bhat[2])
Bz = bilinear(rg, tg, Bhat[3])
Pf = bilinear(rgp, tgp, Pphys)

# ---------- 场线追踪：dr/dζ = B̂r/B̂ζ, dθ/dζ = B̂θ/B̂ζ（RK4 定步长）----------
function trace_line(r0; transits=400, steps_per=64)
    f(u) = (bz = Bz(u[1], u[2]);
    SVector(Br(u[1], u[2]) / bz, Bt(u[1], u[2]) / bz))
    h = 1.0 / steps_per
    u = SVector(r0, 0.0)
    pts = SVector{2, Float64}[]
    θtot = 0.0
    for _ in 1:transits
        for _ in 1:steps_per
            k1 = f(u); k2 = f(u + h / 2 * k1)
            k3 = f(u + h / 2 * k2); k4 = f(u + h * k3)
            unew = u + h / 6 * (k1 + 2k2 + 2k3 + k4)
            (0.002 < unew[1] < 0.999) || return pts, θtot / max(length(pts), 1)
            u = unew
        end
        push!(pts, u)
    end
    return pts, (u[2] - 0.0) / transits   # ι = Δθ/Δζ（圈/圈）
end

println("场线追踪 ...")
seeds = range(0.08, 0.96; length=28)
Rs = Float64[]; Zs = Float64[]; Pc = Float64[]; Ic = Float64[]
iotas = Float64[]
for r0 in seeds
    pts, ι = trace_line(r0)
    push!(iotas, abs(ι))
    for u in pts
        push!(Rs, 1 + ε * u[1] * cospi(2 * u[2]))
        push!(Zs, ε * u[1] * sinpi(2 * u[2]))
        push!(Pc, Pf(u[1], u[2]))
        push!(Ic, abs(ι))
    end
end
println("  ι 范围: ", extrema(iotas))

# ---------- Poincaré 图 ----------
tt = range(0, 2π; length=200)
fig = Figure(size=(1250, 620))
ax1 = Axis(fig[1, 1]; xlabel="R", ylabel="z", title="Poincaré (按压强着色)",
    aspect=DataAspect())
sc1 = scatter!(ax1, Rs, Zs; color=Pc, colormap=:plasma, markersize=2)
lines!(ax1, 1 .+ ε .* cos.(tt), ε .* sin.(tt); color=:gray)
Colorbar(fig[1, 2], sc1; label="p")
ax2 = Axis(fig[1, 3]; xlabel="R", ylabel="z", title="Poincaré (按 ι 着色)",
    aspect=DataAspect())
sc2 = scatter!(ax2, Rs, Zs; color=Ic, colormap=:turbo, markersize=2)
lines!(ax2, 1 .+ ε .* cos.(tt), ε .* sin.(tt); color=:gray)
Colorbar(fig[1, 4], sc2; label="ι")
out1 = replace(file, ".jld2" => "_poincare.png")
save(out1, fig)
println("已保存 $out1")

# ---------- 诊断曲线 ----------
force, energy, divB = dat["force"], dat["energy"], dat["divB"]
hit, hel, flux = dat["helicity_iter"], dat["helicity"], dat["flux"]
fig2 = Figure(size=(1100, 700))
ax = Axis(fig2[1, 1]; yscale=log10, xlabel="iter", title="‖J×B−∇p‖/‖∇p‖")
lines!(ax, 1:length(force), force)
ax = Axis(fig2[1, 2]; xlabel="iter", title="能量 ½‖B‖²")
lines!(ax, 1:length(energy), energy)
ax = Axis(fig2[2, 1]; yscale=log10, xlabel="iter", title="‖div B‖")
lines!(ax, 1:length(divB), max.(divB, 1e-20))
ax = Axis(fig2[2, 2]; yscale=log10, xlabel="iter", title="守恒量相对漂移")
lines!(ax, hit, max.(abs.(flux .- flux[1]) ./ abs(flux[1]), 1e-16); label="环向通量")
lines!(ax, hit, max.(abs.(hel .- hel[1]), 1e-16); label="螺旋度(绝对)")
axislegend(ax; position=:lt)
out2 = replace(file, ".jld2" => "_diag.png")
save(out2, fig2)
println("已保存 $out2")
