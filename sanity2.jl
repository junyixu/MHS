# sanity2.jl — Phase 0.2 原型验证
# 1) Solov'ev 矢势 A 的 L² 投影，B = curl A ⇒ div B = 0 到机器精度
# 2) 独立 Gauss/梯形求积对照 ∫|B|²，锚定 M2 的度规正确性
# 3) 叉乘载荷 (J×H, Λ²)、(v×H, Λ¹) 的装配与交叉一致性
# 4) 物理锚点：Solov'ev 力平衡 ‖J×H − ∇p‖/‖∇p‖ 随分辨率下降
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/solovev.jl")

const ε = 1 / 3
const q_star = 1.57
const τ = solovev_tau(q_star)

# 独立求积（Simpson×梯形，纯 Base）计算 ∫|B⁰|² dV
function energy_reference(; nr=401, nθ=400)
    tot = 0.0
    for (i, r) in enumerate(range(0, 1; length=nr))
        wr = (i == 1 || i == nr) ? 1.0 : (i % 2 == 0 ? 4.0 : 2.0)  # Simpson
        for θ in range(0, 1 - 1 / nθ; length=nθ)                    # 周期梯形
            R = 1 + ε * r * cospi(2θ)
            x1, x3 = R, ε * r * sinpi(2θ)
            B = solovev_B(x1, 0.0, x3, τ)
            tot += wr * (B[1]^2 + B[2]^2 + B[3]^2) * 4π^2 * ε^2 * r * R / nθ
        end
    end
    return tot * (1 / (nr - 1)) / 3
end

E_ref = energy_reference()
println("参考能量 ∫|B⁰|² = $E_ref")

function run_case(nel, deg)
    mapping = circular_torus_mapping(ε)
    X = build_de_rham_complex(nel, deg, mapping)
    dΩ = build_quadrature(X, deg)
    ops = assemble_operators(X, dΩ)
    geom = Mantis.Forms.get_geometry(X[1])
    M1f, M2f = lu(ops.M1), lu(ops.M2)

    # --- Solov'ev: A 投影 → B = curl A ---
    Ah = Mantis.Assemblers.solve_L2_projection(X[2], solovev_A_field(q_star, geom), dΩ)
    a = Ah.coefficients
    b = M2f \ Vector(ops.Kcurl * a)
    div_rel = norm(ops.Kdiv * b) / norm(ops.Kcurl * a)
    println("  ‖div B‖ 相对残差 = $div_rel")
    @assert div_rel < 1e-12

    E_h = dot(b, ops.M2 * b)
    println("  能量 bᵀM2b = $E_h, 相对误差 vs 独立求积 = $(abs(E_h - E_ref) / E_ref)")

    # --- 叉乘载荷一致性（随机离散场） ---
    Bh = Mantis.Forms.build_form_field(X[3], b; label="B")
    αh = Mantis.Forms.build_form_field(X[2], randn(size(ops.M1, 1)); label="α")
    βh = Mantis.Forms.build_form_field(X[2], randn(size(ops.M1, 1)); label="β")
    γh = Mantis.Forms.build_form_field(X[3], randn(size(ops.M2, 1)); label="γ")

    load_JxH(J, H) = assemble_load(X[3] ∧ ★(J ∧ H), X[3], dΩ)   # 2-形式载荷
    load_vxH(v2, H) = assemble_load(X[2] ∧ (★(v2) ∧ H), X[2], dΩ) # 1-形式载荷

    Lαβ = load_JxH(αh, βh)
    @assert norm(load_JxH(αh, αh)) / (norm(Lαβ) + eps()) < 1e-12 "α×α ≠ 0"
    @assert norm(Lαβ + load_JxH(βh, αh)) / norm(Lαβ) < 1e-12 "反对称性失败"
    s1 = dot(γh.coefficients, Lαβ)                       # (γ, α×β)
    s2 = integrate_scalar(★(γh) ∧ αh ∧ βh, dΩ)           # ∫ ★γ∧α∧β
    s3 = dot(αh.coefficients, load_vxH(γh, βh))          # (α, γ×β) = -(γ, α×β)
    println("  叉乘一致性: s1=$s1, s2=$s2, s3=$s3")
    @assert abs(s1 - s2) / abs(s1) < 1e-10 "载荷装配 vs 标量积分不一致"
    @assert abs(s3 + s1) / abs(s1) < 1e-10 "三重楔积恒等式失败"

    # --- 物理锚点: Π²(J⁰×B⁰) ≈ grad Π⁰p⁰（解析 J、H，避开边界条件问题） ---
    # 弱 curl 需要 V₀ 空间（Phase 2 引入 BC 后再测离散 J 版本）
    jxh = M2f \ load_JxH(solovev_J_field(geom), solovev_B1_field(q_star, geom))

    ph = Mantis.Assemblers.solve_L2_projection(X[1], solovev_p_field(geom), dΩ)
    gp = M1f \ Vector(ops.Kgrad * ph.coefficients)
    gp_norm2 = dot(gp, ops.M1 * gp)
    err2 = dot(jxh, ops.M2 * jxh) - 2 * dot(gp, ops.S12 * jxh) + gp_norm2
    force_rel = sqrt(max(err2, 0.0)) / sqrt(gp_norm2)
    println("  力平衡 ‖J×H − ∇p‖/‖∇p‖ = $force_rel")
    return force_rel
end

println("--- 分辨率 (4,8,1), p=(3,3,1) ---")
f1 = run_case((4, 8, 1), (3, 3, 1))
println("--- 分辨率 (8,16,1), p=(3,3,1) ---")
f2 = run_case((8, 16, 1), (3, 3, 1))
println("力残差收敛比 (期望 >2): ", f1 / f2)
@assert f2 < f1 "力残差未随分辨率下降"
println("Phase 0.2 全部通过 ✓")
