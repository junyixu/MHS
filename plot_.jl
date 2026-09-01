# plot_.jl — Poincaré 磁面图 + 诊断曲线（GLMakie）
# 用法: julia --project=. plot_.jl data/run_A.jld2
# 输出: data/run_A_poincare.png, data/run_A_diag.png
#
# 场重构走矢势路线（Mantis v0.6.0 的 2-形式 FormField 求值分量混合，见 plan.md）：
#   Â = A 的逻辑 1-形式分量（求值约定已验证: num_k = h_k·Â_k）
#   轴对称下 B̂r = ∂θÂζ, B̂θ = −∂rÂζ（场线 = Âζ 等值线，哈密顿结构）
#   B̂ζ = ∂rÂθ − ∂θÂr + c·V̂ζ，V̂ζ = −2πε²r/R̂（真空场解析），c 由环向通量匹配
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
κ = haskey(dat, "kappa") ? dat["kappa"] : 1.0
δ = haskey(dat, "delta") ? dat["delta"] : 0.0
b = dat["b"]

println("重建算子 (n=$n, p=$p, ε=$ε, κ=$κ, δ=$δ) ...")
ops = setup_operators((n, n, 1), (p, p, 1), torus_mapping(ε, κ, δ))
_, pcoef, _, _, _ = force_velocity(ops, b)   # 最终态压强（Leray 副产品，V³）
a = vector_potential(ops, b)

# ---------- 单元索引/局部坐标约定自检 ----------
let mp = torus_mapping(ε, κ, δ)
    pts = Mantis.Points.CartesianPoints(([0.3], [0.2], [0.7]))
    x2 = Mantis.Geometry.evaluate(ops.geom, 2, pts)   # 单元 2 = r 方向第二个
    Φref = mp.mapping([1.3 / n, 0.2 / n, 0.7])
    @assert maximum(abs.(vec(x2) .- Φref)) < 1e-12 "单元线性索引应 r 最快"
end

# ---------- (r,θ) 网格求值（ζ 任意，轴对称）----------
function eval_grid(ops, k, coeffs; nr=241, nt=289)
    field = form_field(ops, k, coeffs)
    nel_r, nel_t = ops.nel[1], ops.nel[2]
    rg = collect(range(0.0, 1.0; length=nr))
    tg = collect(range(0.0, 1.0; length=nt))
    elr = [min(floor(Int, r * nel_r) + 1, nel_r) for r in rg]
    xir = [rg[i] * nel_r - (elr[i] - 1) for i in 1:nr]
    elt = [min(floor(Int, t * nel_t) + 1, nel_t) for t in tg]
    xit = [tg[j] * nel_t - (elt[j] - 1) for j in 1:nt]
    ncomp = k == 1 ? 3 : 1
    out = [zeros(nr, nt) for _ in 1:ncomp]
    for et in 1:nel_t, er in 1:nel_r
        is = findall(==(er), elr)
        js = findall(==(et), elt)
        (isempty(is) || isempty(js)) && continue
        pts = Mantis.Points.CartesianPoints((xir[is], xit[js], [0.5]))
        vals, _ = Mantis.Forms.evaluate(field, er + nel_r * (et - 1), pts)
        for c in 1:ncomp
            out[c][is, js] .= reshape(vals[c][:, 1], length(is), length(js))
        end
    end
    return rg, tg, out
end

println("网格求值与场重构 ...")
rg, tg, Anum = eval_grid(ops, 1, a)
# canonical → 全局逻辑分量: Â_k = num_k / h_k, h = (1/n, 1/n, 1)
Ar, At, Az = Anum[1] .* ops.nel[1], Anum[2] .* ops.nel[2], Anum[3]

# 中心差分（r 边界单侧；θ 周期，首末列相同）
function ddr(F, rg)
    G = similar(F)
    h = rg[2] - rg[1]
    G[2:(end - 1), :] .= (F[3:end, :] .- F[1:(end - 2), :]) ./ (2h)
    G[1, :] .= (F[2, :] .- F[1, :]) ./ h
    G[end, :] .= (F[end, :] .- F[end - 1, :]) ./ h
    return G
end
function ddt(F, tg)
    G = similar(F)
    h = tg[2] - tg[1]
    G[:, 2:(end - 1)] .= (F[:, 3:end] .- F[:, 1:(end - 2)]) ./ (2h)
    # θ 周期：列 1 与列 end 是同一点
    G[:, 1] .= (F[:, 2] .- F[:, end - 1]) ./ (2h)
    G[:, end] .= G[:, 1]
    return G
end

Bhat_r = ddt(Az, tg)
Bhat_t = -ddr(Az, rg)
Rhat = [poloidal_position(ε, κ, δ, r, t)[1] for r in rg, t in tg]
Wθ = [poloidal_jacobian(ε, κ, δ, t) for t in tg]      # W = f g′ − f′ g
# 真空场 (1/R)e_φ 的 2-形式拉回 ζ-分量: V̂ζ = det(DΦ)/(2πR²) = r W(θ)/R̂
Vz = [rg[i] * Wθ[j] / Rhat[i, j] for i in eachindex(rg), j in eachindex(tg)]
curlz = ddr(At, rg) .- ddt(Ar, tg)
# c 由环向通量匹配：∫∫B̂ζ dr dθ = −toroidal_flux；∫∫curlz = 0（边界项为零）
trap(F) = sum((F[1:(end - 1), 1:(end - 1)] .+ F[2:end, 1:(end - 1)] .+
               F[1:(end - 1), 2:end] .+ F[2:end, 2:end]) ./ 4) *
          (rg[2] - rg[1]) * (tg[2] - tg[1])
c_harm = (-toroidal_flux(ops, b) - trap(curlz)) / trap(Vz)
Bhat_z = curlz .+ c_harm .* Vz
println("  谐和系数 c = $c_harm, ∫∫curlz = $(trap(curlz))")

# 压强（3-形式求值约定: num = p·detJ/(n_r·n_θ)），det DΦ = 2π r R W(θ)
_, _, Pnum = eval_grid(ops, 3, pcoef)
detJg = [2π * max(rg[i], 1e-6) * Rhat[i, j] * Wθ[j]
         for i in eachindex(rg), j in eachindex(tg)]
Pphys = Pnum[1] .* (ops.nel[1] * ops.nel[2]) ./ detJg
Pphys[1, :] .= Pphys[2, :]   # 轴上 0/0，用邻行外推

# ---------- 双线性插值（θ 周期） ----------
function bilinear(rg, tg, F)
    nr, nt = length(rg), length(tg)
    dr, dtt = rg[2] - rg[1], tg[2] - tg[1]
    return function (r, t)
        tm = mod(t, 1.0)
        ir = clamp(floor(Int, r / dr) + 1, 1, nr - 1)
        it = clamp(floor(Int, tm / dtt) + 1, 1, nt - 1)
        fr = clamp((r - rg[ir]) / dr, 0.0, 1.0)
        ft = (tm - tg[it]) / dtt
        return (1 - fr) * (1 - ft) * F[ir, it] + fr * (1 - ft) * F[ir + 1, it] +
               (1 - fr) * ft * F[ir, it + 1] + fr * ft * F[ir + 1, it + 1]
    end
end
Br_f = bilinear(rg, tg, Bhat_r)
Bt_f = bilinear(rg, tg, Bhat_t)
Bz_f = bilinear(rg, tg, Bhat_z)
P_f = bilinear(rg, tg, Pphys)

# ---------- 场线追踪：dr/dζ = B̂r/B̂ζ, dθ/dζ = B̂θ/B̂ζ（RK4 定步长）----------
# 磁轴（Shafranov 位移处）：压强极大点。ι 用绕磁轴的几何极向角累积
# （坐标轴 r=0 在 (R,z)=(1,0)，内侧磁面不包围它，逻辑 θ̂ 不能用来数极向圈数）
iax = argmax(Pphys)
R_ax, z_ax = poloidal_position(ε, κ, δ, rg[iax[1]], tg[iax[2]])
println("  磁轴 ≈ (R,z) = ($R_ax, $z_ax)")

RZ(u) = poloidal_position(ε, κ, δ, u[1], u[2])

function trace_line(r0; transits=400, steps_per=64)
    f(u) = (bz = Bz_f(u[1], u[2]);
    SVector(Br_f(u[1], u[2]) / bz, Bt_f(u[1], u[2]) / bz))
    h = 1.0 / steps_per
    u = SVector(r0, 0.0)
    pts = SVector{2, Float64}[]
    R0, z0 = RZ(u)
    ang_prev = atan(z0 - z_ax, R0 - R_ax)
    ang_tot = 0.0
    for _ in 1:transits
        for _ in 1:steps_per
            k1 = f(u); k2 = f(u + h / 2 * k1)
            k3 = f(u + h / 2 * k2); k4 = f(u + h * k3)
            unew = u + h / 6 * (k1 + 2k2 + 2k3 + k4)
            (0.002 < unew[1] < 0.999) ||
                return pts, length(pts) > 0 ? ang_tot / (2π * length(pts)) : NaN
            u = unew
            Rc, zc = RZ(u)
            ang = atan(zc - z_ax, Rc - R_ax)
            dα = ang - ang_prev
            ang_tot += dα - 2π * round(dα / (2π))   # wrap 到 (−π, π]
            ang_prev = ang
        end
        push!(pts, u)
    end
    return pts, ang_tot / (2π * transits)   # ι = 极向圈数/环向圈数
end

println("场线追踪 ...")
seeds = range(0.10, 0.96; length=28)
Rs, Zs, Pc, Ic = Float64[], Float64[], Float64[], Float64[]
iotas = Float64[]
for r0 in seeds
    pts, ι = trace_line(r0)
    isempty(pts) && continue
    # 坐在磁轴上的种子退化（轨道半径 ~0，极向角无意义），跳过
    orbit_r = sum(hypot(RZ(u)[1] - R_ax, RZ(u)[2] - z_ax) for u in pts) / length(pts)
    orbit_r > 5e-3 || continue
    push!(iotas, abs(ι))
    for u in pts
        R, z = RZ(u)
        push!(Rs, R); push!(Zs, z)
        push!(Pc, P_f(u[1], u[2]))
        push!(Ic, abs(ι))
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
ax = Axis(fig2[1, 1]; yscale=log10, xlabel="iter", title="force residual |JxB-grad p|/|grad p|")
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
