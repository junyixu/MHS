# sanity6.jl — Phase 7 验证：3D 旋转椭圆仿星器几何与 n_ζ>1 的复形
# 1) Jacobian 有限差分、det>0、det 解析公式（4π²ε² ν(ζ)ν(ζ+1/2) r R）
# 2) 旋转椭圆判据：R、Z 半轴反相（a ≡ b 就退化成波纹环，不是仿星器）
# 3) 退化一致性：κ=1 ⇒ ν≡1 ⇒ 与圆截面环完全相同
# 4) 体积：V = 2π²ε² ∫₀¹ν(ζ)ν(ζ+1/2)dζ —— 用轻量 ∑M₀ 对照
# 5) n_ζ>1 复形：DD=0、Leray、谐和场、Solov'ev 初值 + 计时
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays, Printf
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/operators.jl")
include("src/solovev.jl")
include("src/relaxation.jl")

const εS, κS, nfpS = 0.33, 1.2, 3      # 论文 V.F 的仿星器参数
const q_star = 1.57

println("--- 1. Jacobian 校验 (ε,κ,n_fp)=($εS,$κS,$nfpS) ---")
mp = stellarator_mapping(εS, κS, nfpS)
let h = 1e-6, Id = Matrix(I, 3, 3), worst = 0.0, mindet = Inf
    for _ in 1:300
        x = [0.05 + 0.95 * rand(), rand(), rand()]
        J = mp.dmapping(x)
        Jfd = hcat(((mp.mapping(x .+ h .* Id[:, j]) .- mp.mapping(x .- h .* Id[:, j])) ./ (2h)
                    for j in 1:3)...)
        worst = max(worst, maximum(abs.(J .- Jfd)))
        mindet = min(mindet, det(J))
    end
    println("  最大 FD 误差 = $worst, 最小 det = $mindet")
    @assert worst < 1e-6 "Jacobian 与有限差分不符"
    @assert mindet > 0 "det DΦ 必须处处为正"
end
# det = 4π²ε² ν(ζ)ν(ζ+1/2) r R
let x = [0.63, 0.29, 0.41]
    a, b = stellarator_nu(κS, nfpS, x[3]), stellarator_nu(κS, nfpS, x[3] + 0.5)
    R, _ = stellarator_position(εS, κS, nfpS, x[1], x[2], x[3])
    dana = 4π^2 * εS^2 * a * b * x[1] * R
    dnum = det(mp.dmapping(x))
    println("  det 公式: 数值 $dnum vs 解析 $dana, 相对差 = ", abs(dnum - dana) / abs(dnum))
    @assert abs(dnum - dana) / abs(dnum) < 1e-12
end

println("--- 2. 旋转椭圆判据（R、Z 半轴必须反相）---")
# n_fp 为奇数时 ν(ζ+1/2) = 2 − ν(ζ)：R 半轴最大处 Z 半轴最小 ⇒ 截面转 90°
let ν(ζ) = stellarator_nu(κS, nfpS, ζ)
    for ζ in (0.0, 1 / (2 * nfpS), 0.137)
        @printf("  ζ=%.4f: (a,b) = (%.4f, %.4f)\n", ζ, ν(ζ), ν(ζ + 0.5))
        @assert abs(ν(ζ) + ν(ζ + 0.5) - 2) < 1e-14 "n_fp 奇数时须有 b = 2 − a"
    end
    @assert abs(ν(0.0) - ν(0.5)) > 0.1 "a ≡ b ⇒ 截面恒为圆，是波纹环不是仿星器"
end

println("--- 3. 退化一致性（κ=1 ⇒ ν≡1 ⇒ 圆截面环）---")
let ms = stellarator_mapping(1 / 3, 1.0, 3), mt = torus_mapping(1 / 3, 1.0, 0.0)
    e = maximum(maximum(abs.(ms.mapping(x) .- mt.mapping(x))) +
                maximum(abs.(ms.dmapping(x) .- mt.dmapping(x)))
                for x in ([0.3, 0.2, 0.7], [0.9, 0.55, 0.13], [0.5, 0.8, 0.44]))
    println("  最大差 = $e")
    @assert e < 1e-14
end

println("--- 4. 体积 ---")
# V = ∫∫∫ det = 4π²ε² ∫₀¹ a(ζ)b(ζ)dζ · ∫∫ rR dr dθ，而 ∫∫ rR dr dθ = 1/2
# ∫₀¹ν(ζ)ν(ζ+1/2)dζ = 1 + (−1)^{n_fp} A²/2，A = 1−κ
stellarator_volume(ε, κ, n_fp) = 2π^2 * ε^2 * (1 + (-1)^n_fp * (1 - κ)^2 / 2)
# 被积函数含 cos²，**非多项式**，Gauss 求积不可能精确（环截面那边是多项式，故能到 1e-11）。
# 且误差对 n_ζ **不单调**——取决于单元边界与 cos 周期的对齐。
# 隔离变量的正确做法：固定网格、加求积点，误差掉到机器精度即证明残差纯属求积。
# 体积只需 ∑M₀（B 样条单位分解），用轻量装配，不做整套算子与分解。
let vex = stellarator_volume(εS, κS, nfpS)
    volerr(nz, qx) = abs(sum(assemble_mass0((6, 12, nz), (3, 3, 3), mp; qextra=qx)) - vex) / vex
    for nz in (4, 6, 8)
        @printf("  n_ζ=%d, qextra=3: 相对误差 %.3e\n", nz, volerr(nz, 3))
        @assert volerr(nz, 3) < 1e-6 "体积误差过大，映射可能有误"
    end
    e = [volerr(6, qx) for qx in (3, 6, 9)]
    for (qx, err) in zip((3, 6, 9), e)
        @printf("  n_ζ=6, qextra=%d: 相对误差 %.3e\n", qx, err)
    end
    @assert e[end] < 1e-12 "加求积点后仍不收敛 ⇒ 残差不是求积误差，映射或体积公式有误"
end

println("--- 5. n_ζ>1 复形性质与 Solov'ev 初值 ---")
for nel in ((6, 12, 4), (12, 12, 6))       # 后者是论文 V.F 的分辨率
    t_setup = @elapsed ops = setup_operators(nel, (3, 3, 3),
        stellarator_mapping(εS, κS, nfpS))
    ndof = size.(ops.E, 1)
    a0, p0 = randn(size(ops.M1, 1)), randn(size(ops.M0, 1))
    r_cg = norm(ops.Kcurl * strong_grad(ops, p0)) / norm(ops.Kgrad * p0)
    r_dc = norm(ops.Kdiv * strong_curl(ops, a0)) / norm(ops.Kcurl * a0)
    w = randn(size(ops.M2, 1))
    v, _, _ = leray(ops, w)
    h = ops.hharm
    hn = mnorm(ops.M2, h)
    b = solovev_initial_field(ops, q_star; κ̄=1.0)   # 论文 V.F：κ̄ = 1.0
    t_force = @elapsed (vf, _, σp, _, _) = force_velocity(ops, b)
    println("  nel=$nel  自由度 $ndof   (装配 $(round(t_setup, digits=1)) s)")
    @printf("    DD=0: %.2e / %.2e,  Leray %.2e\n", r_cg, r_dc,
        norm(ops.Kdiv * v) / norm(ops.Kdiv * w))
    @printf("    谐和场 ‖div h‖/‖h‖ = %.2e, ‖~curl h‖/‖h‖ = %.2e\n",
        norm(ops.Kdiv * h) / hn, mnorm(ops.M1, weak_curl(ops, h)) / hn)
    @printf("    初值 ‖divB‖/‖B‖ = %.2e, 力残差 = %.4e, 环向通量 = %.6f\n",
        norm(ops.Kdiv * b) / mnorm(ops.M2, b),
        mnorm(ops.M2, vf) / mnorm(ops.M2, σp), toroidal_flux(ops, b))
    @printf("    单次力评估 %.3f s ⇒ 1000 步约 %.1f 分钟\n",
        t_force, t_force * 1000 / 60)
    @assert max(r_cg, r_dc) < 1e-9
    @assert norm(ops.Kdiv * v) / norm(ops.Kdiv * w) < 1e-10
    @assert norm(ops.Kdiv * b) / mnorm(ops.M2, b) < 1e-10
end
println("Phase 7 几何/复形检查通过 ✓")
