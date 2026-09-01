# sanity4.jl — Phase 5 验证：D 形截面映射与复形
# 1) Jacobian 有限差分对照、det > 0（映射非退化）
# 2) 圆截面退化一致性（κ=1, δ=0 与 circular 完全相同）
# 3) 体积对照解析值 2π·(截面面积)，截面面积由边界曲线 Green 公式独立算
# 4) 约束复形性质（DD=0、Leray、A-解）+ Solov'ev 初值力残差
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/operators.jl")
include("src/solovev.jl")
include("src/relaxation.jl")

# ITER 位形
const εI, κI, δI = 0.33, 1.7, 0.33
const q_star = 1.57

# --- 1. Jacobian 有限差分 + det > 0 ---
println("--- Jacobian 校验 (ε,κ,δ)=($εI,$κI,$δI) ---")
mp = torus_mapping(εI, κI, δI)
let h = 1e-6, Id = Matrix(I, 3, 3), worst_err = 0.0, mindet = Inf
    for _ in 1:200
        x = [0.05 + 0.95 * rand(), rand(), rand()]
        J = mp.dmapping(x)
        Jfd = hcat(((mp.mapping(x .+ h .* Id[:, j]) .- mp.mapping(x .- h .* Id[:, j])) ./ (2h)
                    for j in 1:3)...)
        worst_err = max(worst_err, maximum(abs.(J .- Jfd)))
        mindet = min(mindet, det(J))
    end
    println("  最大 FD 误差 = $worst_err, 最小 det = $mindet")
    @assert worst_err < 1e-6 "Jacobian 与有限差分不符"
    @assert mindet > 0 "det DΦ 必须处处为正（映射退化）"
end

# det DΦ = 2π r R W(θ) 的解析公式校验
let x = [0.6, 0.37, 0.21]
    dnum = det(mp.dmapping(x))
    R, _ = poloidal_position(εI, κI, δI, x[1], x[2])
    dana = 2π * x[1] * R * poloidal_jacobian(εI, κI, δI, x[2])
    println("  det 公式: 数值 $dnum vs 解析 $dana, 相对差 = ", abs(dnum - dana) / abs(dnum))
    @assert abs(dnum - dana) / abs(dnum) < 1e-12
end

# --- 2. 圆截面退化一致性 ---
let mc = circular_torus_mapping(1 / 3), mg = torus_mapping(1 / 3, 1.0, 0.0)
    e = maximum(maximum(abs.(mc.mapping(x) .- mg.mapping(x))) +
                maximum(abs.(mc.dmapping(x) .- mg.dmapping(x)))
                for x in ([0.3, 0.2, 0.7], [0.9, 0.55, 0.13]))
    println("  圆截面退化一致性: 最大差 = $e")
    @assert e < 1e-14
end

# --- 3. 体积对照 ---
# 旋转体精确体积（Green 公式）: V = ∬2πR dA = π∮R² dz，沿 r=1 边界曲线积分。
# 注意不能用「2π·截面面积」——那是 Pappus 定理取形心 R_c=1 的近似，
# D 形因三角形变形心并不在 R=1（实测差 2.7%）。
function revolution_volume(ε, κ, δ; N=400000)
    V = 0.0
    for i in 0:(N - 1)
        R0, z0 = poloidal_position(ε, κ, δ, 1.0, i / N)
        R1, z1 = poloidal_position(ε, κ, δ, 1.0, (i + 1) / N)
        V += 0.5 * (R0^2 + R1^2) * (z1 - z0)      # 梯形法逼近 ∮R²dz
    end
    return abs(π * V)
end
for (ε, κ, δ) in ((1 / 3, 1.0, 0.0), (εI, κI, δI))
    ops = setup_operators((6, 12, 1), (3, 3, 1), torus_mapping(ε, κ, δ))
    vol = sum(ops.raw.M0)                          # B 样条单位分解 ⇒ ∫1 dV
    vex = revolution_volume(ε, κ, δ)
    println("  (ε,κ,δ)=($ε,$κ,$δ): 装配 = $vol, 解析 = $vex, " *
            "相对误差 = $(abs(vol - vex) / vex)")
    @assert abs(vol - vex) / vex < 1e-8 "体积不符，映射或求积有误"
end

# --- 4. D 形上的复形与初值 ---
println("--- D 形复形与 Solov'ev 初值 ---")
for nel in ((6, 12, 1), (10, 20, 1))
    ops = setup_operators(nel, (3, 3, 1), torus_mapping(εI, κI, δI))
    a0 = randn(size(ops.M1, 1))
    p0 = randn(size(ops.M0, 1))
    r_cg = norm(ops.Kcurl * strong_grad(ops, p0)) / norm(ops.Kgrad * p0)
    r_dc = norm(ops.Kdiv * strong_curl(ops, a0)) / norm(ops.Kcurl * a0)
    w = randn(size(ops.M2, 1))
    v, _, _ = leray(ops, w)
    h = ops.hharm
    hn = mnorm(ops.M2, h)
    # κ̄ = κ 使 B₀·n|∂Ω = O(ε²)（论文 V.D.b）
    b = solovev_initial_field(ops, q_star; κ̄=κI)
    vf, _, σp, _, _ = force_velocity(ops, b)
    println("  nel=$nel: DD=0 ($r_cg, $r_dc), Leray = ",
        norm(ops.Kdiv * v) / norm(ops.Kdiv * w))
    println("    谐和场 ‖div h‖/‖h‖ = ", norm(ops.Kdiv * h) / hn,
        ", ‖~curl h‖/‖h‖ = ", mnorm(ops.M1, weak_curl(ops, h)) / hn)
    println("    初始 ‖div B‖/‖B‖ = ", norm(ops.Kdiv * b) / mnorm(ops.M2, b),
        ", 力残差 = ", mnorm(ops.M2, vf) / mnorm(ops.M2, σp),
        ", 环向通量 = ", toroidal_flux(ops, b))
    @assert max(r_cg, r_dc) < 1e-9
    @assert norm(ops.Kdiv * v) / norm(ops.Kdiv * w) < 1e-10
    @assert norm(ops.Kdiv * b) / mnorm(ops.M2, b) < 1e-10
end
println("Phase 5 几何/复形检查通过 ✓")
