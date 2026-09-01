# sanity.jl — Phase 0 原型验证
# 1) (θ,ζ) 周期、n_ζ=1 的 3D 张量积 de Rham 复形 + 圆截面环形映射
# 2) 验证 Jacobian（有限差分对照）、det>0、环体积
# 3) 验证 D_{k+1}∘D_k = 0（curl∘grad、div∘curl 到求解器精度）
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays
using Mantis.Forms: d, ★, ∧, ∫

const FS = Mantis.FunctionSpaces
const Geo = Mantis.Geometry

# ---------------- 参数 ----------------
nel = (3, 6, 1)   # (n_r, n_θ, n_ζ)
deg = (2, 2, 1)   # (p_r, p_θ, p_ζ)
const ε = 1 / 3   # 反环径比（小半径），R₀ = 1

# ---------------- 圆截面环形映射 ----------------
# Φ(r,θ,ζ) = (R̂ cos2πζ, -R̂ sin2πζ, ε r sin2πθ),  R̂ = 1 + ε r cos2πθ
# 取 -sin 使 det DΦ = 4π²ε² r R̂ > 0
function torus_map(x::AbstractVector)
    r, θ, ζ = x[1], x[2], x[3]
    R = 1 + ε * r * cospi(2θ)
    return [R * cospi(2ζ), -R * sinpi(2ζ), ε * r * sinpi(2θ)]
end

function torus_dmap(x::AbstractVector)
    r, θ, ζ = x[1], x[2], x[3]
    c, s = cospi(2θ), sinpi(2θ)
    C, S = cospi(2ζ), sinpi(2ζ)
    R = 1 + ε * r * c
    # J[i,j] = ∂Φᵢ/∂x̂ⱼ，SMatrix 按列填充
    return SMatrix{3, 3}(
        ε * c * C, -ε * c * S, ε * s,                       # ∂/∂r
        -2π * ε * r * s * C, 2π * ε * r * s * S, 2π * ε * r * c,  # ∂/∂θ
        -2π * R * S, -2π * R * C, 0.0,                      # ∂/∂ζ
    )
end

mapping = Geo.Mapping((3, 3), torus_map, torus_dmap)

# 有限差分验证 Jacobian
let h = 1e-6, rng = [(0.3, 0.1, 0.7), (0.9, 0.6, 0.2), (0.5, 0.25, 0.0)]
    for pt in rng
        x = collect(pt)
        J = torus_dmap(x)
        Jfd = hcat((
            (torus_map(x .+ h .* Matrix(I, 3, 3)[:, j]) .- torus_map(x .- h .* Matrix(I, 3, 3)[:, j])) ./ (2h)
            for j in 1:3
        )...)
        err = maximum(abs.(J .- Jfd))
        detJ = det(J)
        println("Jacobian FD check @ $pt: err = $err, det = $detJ")
        @assert err < 1e-6 "Jacobian 与有限差分不符"
        @assert detJ > 0 "det DΦ 应为正"
    end
end

# ---------------- 一维样条空间（θ, ζ 周期） ----------------
B = FS.create_dim_wise_bspline_spaces(
    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), nel, FS.Bernstein.(deg), deg .- 1
)
S0 = (
    B[1],                                     # r: clamped
    FS.GTBSplineSpace((B[2],), [deg[2] - 1]), # θ: 周期
    FS.GTBSplineSpace((B[3],), [deg[3] - 1]), # ζ: 周期
)
S1 = map(FS.get_derivative_space, S0)
println("一维 0-form 空间维数 (r,θ,ζ): ", FS.get_num_basis.(S0))
println("一维 1-form 空间维数 (r,θ,ζ): ", FS.get_num_basis.(S1))
@assert FS.get_num_basis(S0[3]) == 1 "n_ζ=1 周期样条应只有 1 个自由度（轴对称）"

# ---------------- 3D de Rham 复形 ----------------
# k-form 分量方向组合与 Mantis 参考实现一致（proxy 顺序）
function build_complex(S0, S1, mapping)
    fem = (S0, S1)
    form_spaces = ntuple(4) do kp1
        k = kp1 - 1
        combos = Mantis.Forms.get_basis_index_combinations(3, k)
        comps = ntuple(length(combos)) do c
            idxs = ones(Int, 3)
            idxs[combos[c]] .= 2
            FS.TensorProductSpace(ntuple(dim -> fem[idxs[dim]][dim], 3), mapping)
        end
        space = length(comps) == 1 ? comps[1] : FS.DirectSumSpace(comps)
        Mantis.Forms.FormSpace(k, space, "ω_$k")
    end
    return form_spaces
end

X = build_complex(S0, S1, mapping)
for k in 0:3
    println("V^$k 自由度数: ", Mantis.Forms.get_num_basis(X[k + 1]))
end

# ---------------- 求积与装配 ----------------
geom = Mantis.Forms.get_geometry(X[1])
qrule = Mantis.Quadrature.tensor_product_rule(deg .+ 3, Mantis.Quadrature.gauss_legendre)
dΩ = Mantis.Quadrature.StandardQuadrature(qrule, Geo.get_num_elements(geom))

function assemble_bilinear(expr_builder, Xtest, Xtrial, dΩ)
    wfi = Mantis.Assemblers.WeakFormInputs(Xtest, Xtrial)
    v = Mantis.Assemblers.get_test_form(wfi)
    u = Mantis.Assemblers.get_trial_form(wfi)
    wf = Mantis.Assemblers.WeakForm(((expr_builder(v, u),),), ((0,),), wfi)
    A, _ = Mantis.Assemblers.assemble(wf)
    return A
end

println("装配质量矩阵与混合矩阵...")
M = ntuple(k -> assemble_bilinear((v, u) -> ∫(v ∧ ★(u), dΩ), X[k], X[k], dΩ), 4)
K = ntuple(k -> assemble_bilinear((v, u) -> ∫(v ∧ ★(d(u)), dΩ), X[k + 1], X[k], dΩ), 3)
for k in 1:4
    println("M$(k-1): size = ", size(M[k]), ", nnz = ", nnz(M[k]))
end

# 体积检查: B 样条单位分解 ⇒ sum(M0) = ∫ 1 dV = 2π²ε²
vol = sum(M[1])
vol_exact = 2 * π^2 * ε^2
println("体积: 装配 = $vol, 精确 = $vol_exact, 相对误差 = $(abs(vol - vol_exact) / vol_exact)")

# ---------------- DD = 0 验证 ----------------
# 强微分实现为 Dₖ = Mₖ₊₁⁻¹ Kₖ（与 MRX 论文一致）；验证 Kₖ₊₁ Mₖ₊₁⁻¹ Kₖ ≈ 0
Mfact = ntuple(k -> lu(M[k]), 4)
for k in 1:2
    u = randn(size(M[k], 1))
    du = Mfact[k + 1] \ Vector(K[k] * u)      # dᵏu 的系数
    r = K[k + 1] * du                          # ∫ v^{k+2} ∧ ★(d(dᵏu))
    rel = norm(r) / (norm(K[k + 1], Inf) * norm(du) + eps())
    name = k == 1 ? "curl∘grad" : "div∘curl"
    println("$name: ‖K$(k+1)·D$(k-1)u‖ 相对残差 = $rel")
    @assert rel < 1e-10 "$name 不为零！"
end

println("Phase 0.1 全部通过 ✓")
