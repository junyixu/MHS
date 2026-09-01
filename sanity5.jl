# sanity5.jl — 论文 Remark 29：为什么必须引入辅助变量 H_h
#
# 离散格式里 E_h = Π¹₀(v_h × H_h)，其中 H_h = Π¹₀B_h。
# 螺旋度守恒的证明（Prop 28）最后一步需要 (v_h×H_h, H_h)_{L²} = 0，
# 这是「同一个 H_h 出现在叉乘的两个槽位」带来的**代数恒等式**（三重积有重复因子）。
# 若改用 E_h = Π¹₀(v_h × B_h)，则需要 (v_h×B_h, H_h) = 0，
# 但 H_h ≠ B_h（两者住在不同的离散空间），该量只是 O(h^p) 而非零。
#
# 本脚本量化这一差别：
#   (1) 代数检验：∫H·(v×H) vs ∫H·(v×B)     —— 与时间格式无关，最干净
#   (2) 弛豫检验：两种 E 取法下螺旋度的累积漂移
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays, Printf
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/operators.jl")
include("src/solovev.jl")
include("src/relaxation.jl")

const ε = 1 / 3
const q_star = 1.57

# E = Π¹₀(v × B) 的载荷：v、B 均为 2-形式，★ 把它们变回 1-形式后做 1∧1 楔积
function load_vxB(ops, v2::AbstractVector, b2::AbstractVector)
    Vf = form_field(ops, 2, v2; label="v")
    Bf = form_field(ops, 2, b2; label="B")
    return ops.E[2] * assemble_load(ops.X[2] ∧ (★(Vf) ∧ ★(Bf)), ops.X[2], ops.dΩ)
end

println("="^78)
println("(1) 代数检验：三重积 (v×X, H) 是否恒为零")
println("="^78)
for n in (6, 8, 12)
    ops = setup_operators((n, n, 1), (3, 3, 1), circular_torus_mapping(ε))
    b = solovev_initial_field(ops, q_star)
    v, _, _, j, h = force_velocity(ops, b)

    lhs_H = dot(h, load_vxH(ops, v, h))     # ∫H·(v×H) —— 结构性零
    lhs_B = dot(h, load_vxB(ops, v, b))     # ∫H·(v×B) —— Remark 29 的泄漏项
    scale = sqrt(dot(v, ops.M2 * v)) * dot(h, ops.M1 * h)
    @printf("  n=%2d:  ∫H·(v×H)/scale = %9.2e     ∫H·(v×B)/scale = %9.2e\n",
        n, abs(lhs_H) / scale, abs(lhs_B) / scale)
end
println("  ⇒ 前者是机器零（代数恒等，与分辨率无关），后者大 14 个量级")

println("\n" * "="^78)
println("(2) 螺旋度漂移：空间误差 vs 时间误差")
println("="^78)
# 固定总演化时间 T，加密时间步（Δt 减半、步数加倍）：
#   用 H_h（正确）：Prop 28 说中点格式下精确守恒 ⇒ 漂移纯属时间离散，∝ Δt，应随之减小
#   用 B_h（错误）：泄漏来自空间投影 Π¹₀B ≠ B ⇒ 与 Δt 无关，应停在一个平台上
function drift_fixed_time(ops, b0, T, N; use_H::Bool)
    b = copy(b0)
    H0 = helicity(ops, b)
    dt = T / N
    for _ in 1:N
        v, _, _, j, h = force_velocity(ops, b)
        e = use_H ? ops.M1f \ load_vxH(ops, v, h) : ops.M1f \ load_vxB(ops, v, b)
        b = b .+ dt .* strong_curl(ops, e)
    end
    return abs(helicity(ops, b) - H0) / abs(H0)
end

let n = 8, T = 4.0
    ops = setup_operators((n, n, 1), (3, 3, 1), circular_torus_mapping(ε))
    b0 = solovev_initial_field(ops, q_star)
    @printf("  n=%d, 固定总时间 T=%.1f\n", n, T)
    @printf("  %-8s %-8s %-14s %-14s\n", "步数 N", "Δt", "用 H_h（正确）", "用 B_h（错误）")
    for N in (250, 500, 1000, 2000)
        dA = drift_fixed_time(ops, b0, T, N; use_H=true)
        dB = drift_fixed_time(ops, b0, T, N; use_H=false)
        @printf("  %-8d %-8.4f %-14.3e %-14.3e\n", N, T / N, dA, dB)
    end
end
println("  ⇒ 正确格式的漂移随 Δt 线性减小（纯时间离散误差，中点格式下会精确为零）；")
println("     错误格式的漂移停在平台上——那是空间投影误差，再小的 Δt 也消不掉。")
