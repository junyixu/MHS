# sanity3.jl — Phase 2 验证（E-提取矩阵版：r=1 BC + 最小相容轴约束）
# 1) 约束后复形闭合性: grad(V⁰₀)⊂V¹₀, curl(V¹₀)⊂V²₀, DD=0
# 2) Leray 投影强无散（含 ∫p 增广，λ=0）
# 3) 谐和 2-形式质量（轴约束恢复 𝔥₀² 一维）+ A-解诊断
# 4) Solov'ev 初始化管线 + 初始力残差
# 5) 单次力评估计时
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/operators.jl")
include("src/solovev.jl")
include("src/relaxation.jl")

const ε = 1 / 3
const q_star = 1.57

# full 向量到 span(Eᵀ) 的正交投影残差（E 行互不相交 ⇒ EEᵀ 对角）
function span_residual(E, vfull)
    D = vec(sum(E .^ 2; dims=2))
    return norm(vfull - E' * ((E * vfull) ./ D)) / norm(vfull)
end

function run_case(nel, deg)
    ops = setup_operators(nel, deg, circular_torus_mapping(ε))

    # --- 1. 约束后复形闭合性 ---
    a0 = randn(size(ops.M1, 1))
    c_full = ops.raw.M2 \ Vector(ops.raw.Kcurl * scatter(ops, 1, a0))
    println("  curl(V¹₀)⊂V²₀ 残差 = ", span_residual(ops.E[3], c_full))
    @assert span_residual(ops.E[3], c_full) < 1e-12

    p0 = randn(size(ops.M0, 1))
    g_full = ops.raw.M1 \ Vector(ops.raw.Kgrad * scatter(ops, 0, p0))
    println("  grad(V⁰₀)⊂V¹₀ 残差 = ", span_residual(ops.E[2], g_full))
    @assert span_residual(ops.E[2], g_full) < 1e-12

    r_cg = norm(ops.Kcurl * strong_grad(ops, p0)) / norm(ops.Kgrad * p0)
    r_dc = norm(ops.Kdiv * strong_curl(ops, a0)) / norm(ops.Kcurl * a0)
    println("  约束 DD=0: curl∘grad = $r_cg, div∘curl = $r_dc")
    @assert max(r_cg, r_dc) < 1e-10

    # --- 2. Leray 强无散 ---
    w = randn(size(ops.M2, 1))
    v, p, σ = leray(ops, w)
    println("  ‖div ΠLeray w‖/‖div w‖ = ", norm(ops.Kdiv * v) / norm(ops.Kdiv * w))
    @assert norm(ops.Kdiv * v) / norm(ops.Kdiv * w) < 1e-10

    # --- 3. 谐和场 + A-解 ---
    h = ops.hharm
    hn = mnorm(ops.M2, h)
    println("  谐和场: ‖h‖ = $hn, ‖div h‖/‖h‖ = ", norm(ops.Kdiv * h) / hn,
        ", ‖~curl h‖/‖h‖ = ", mnorm(ops.M1, weak_curl(ops, h)) / hn)
    let btest = randn(size(ops.M2, 1)), n0 = size(ops.M0, 1)
        AV = [-ops.M0 ops.Kgrad'; ops.Kgrad ops.Kcc]
        rhs = [zeros(n0); ops.Kcurl' * btest]
        sol = ops.avec_fact \ rhs
        println("  A-解: 残差 = ", norm(AV * sol - rhs) / norm(rhs),
            ", ‖~curl(b−curlA)‖/‖~curl b‖ = ",
            mnorm(ops.M1, weak_curl(ops, btest - strong_curl(ops, sol[(n0 + 1):end]))) /
            mnorm(ops.M1, weak_curl(ops, btest)))
    end

    # --- 4. Solov'ev 初始化 + 力残差 ---
    b = solovev_initial_field(ops, q_star)
    println("  初始 ‖div B‖/‖B‖ = ", norm(ops.Kdiv * b) / mnorm(ops.M2, b))
    @assert norm(ops.Kdiv * b) / mnorm(ops.M2, b) < 1e-10

    t = @elapsed begin
        vf, pp, σp, _, _ = force_velocity(ops, b)
    end
    force_rel = mnorm(ops.M2, vf) / mnorm(ops.M2, σp)
    println("  初始力残差 ‖J×H−∇p‖/‖∇p‖ = $force_rel   (单次力评估 $t s)")

    th = @elapsed H = helicity(ops, b)
    println("  螺旋度 H = $H   ($th s), 环向通量 = ", toroidal_flux(ops, b))
    return force_rel, H
end

println("--- (4,8,1), p=(3,3,1) ---")
f1, H1 = run_case((4, 8, 1), (3, 3, 1))
println("--- (8,16,1), p=(3,3,1) ---")
f2, H2 = run_case((8, 16, 1), (3, 3, 1))
println("初始力残差: $(f1) → $(f2); 螺旋度: $(H1) → $(H2)")
println("Phase 2(E) 全部通过 ✓")
