# (θ,ζ) 周期的 3D 张量积 B 样条 de Rham 复形与算子装配
const FS = Mantis.FunctionSpaces

# 逻辑域 [0,1]³，r 方向 clamped，θ/ζ 方向周期（GTBSplineSpace 单 patch）
function build_de_rham_complex(nel::NTuple{3, Int}, deg::NTuple{3, Int}, mapping)
    B = FS.create_dim_wise_bspline_spaces(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), nel, FS.Bernstein.(deg), deg .- 1
    )
    S0 = (
        B[1],
        FS.GTBSplineSpace((B[2],), [deg[2] - 1]),
        FS.GTBSplineSpace((B[3],), [deg[3] - 1]),
    )
    S1 = map(FS.get_derivative_space, S0)
    fem = (S0, S1)
    return ntuple(4) do kp1
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
end

function build_quadrature(X, deg::NTuple{3, Int}; extra::Int=3)
    geom = Mantis.Forms.get_geometry(X[1])
    qrule = Mantis.Quadrature.tensor_product_rule(
        deg .+ extra, Mantis.Quadrature.gauss_legendre
    )
    return Mantis.Quadrature.StandardQuadrature(
        qrule, Mantis.Geometry.get_num_elements(geom)
    )
end

function assemble_bilinear(expr_builder, Xtest, Xtrial, dΩ)
    wfi = Mantis.Assemblers.WeakFormInputs(Xtest, Xtrial)
    v = Mantis.Assemblers.get_test_form(wfi)
    u = Mantis.Assemblers.get_trial_form(wfi)
    wf = Mantis.Assemblers.WeakForm(((expr_builder(v, u),),), ((0,),), wfi)
    A, _ = Mantis.Assemblers.assemble(wf)
    return A
end

# 载荷向量：expr 为 rank-1 表达式（含一个 FormSpace 测试槽），逐单元累加
function assemble_load(expr, Xtest, dΩ)
    out = zeros(Mantis.Forms.get_num_basis(Xtest))
    intg = ∫(expr, dΩ)
    for e in 1:Mantis.Quadrature.get_num_base_elements(dΩ)
        vals, idxs = Mantis.Forms.evaluate(intg, e)
        for (k, gid) in enumerate(idxs[1])
            out[gid] += vals[k]
        end
    end
    return out
end

# rank-0 表达式的标量积分 ∫_Ω expr
function integrate_scalar(expr, dΩ)
    intg = ∫(expr, dΩ)
    nelem = Mantis.Quadrature.get_num_base_elements(dΩ)
    return sum(first(Mantis.Forms.evaluate(intg, e))[1] for e in 1:nelem)
end

# 复形的全部常用矩阵:
#   Mk    : k-形式质量矩阵
#   Kgrad : (Λ¹ᵢ, grad Λ⁰ⱼ)   Kcurl : (Λ²ᵢ, curl Λ¹ⱼ)   Kdiv : (Λ³ᵢ, div Λ²ⱼ)
#   S12   : ∫ Λ¹ᵢ ∧ Λ²ⱼ = (Λ¹ᵢ, Λ²ⱼ)_proxy   （1↔2 形式 L² 配对，Π¹ 投影用）
# 强微分系数 = Mₖ₊₁⁻¹ Kₖ（样条复形中精确）；弱微分 = Mₖ⁻¹ Kₖᵀ
function assemble_operators(X, dΩ)
    mass(Xk) = assemble_bilinear((v, u) -> ∫(v ∧ ★(u), dΩ), Xk, Xk, dΩ)
    mixed(Xte, Xtr) = assemble_bilinear((v, u) -> ∫(v ∧ ★(d(u)), dΩ), Xte, Xtr, dΩ)
    return (
        M0=mass(X[1]), M1=mass(X[2]), M2=mass(X[3]), M3=mass(X[4]),
        Kgrad=mixed(X[2], X[1]), Kcurl=mixed(X[3], X[2]), Kdiv=mixed(X[4], X[3]),
        S12=assemble_bilinear((v, u) -> ∫(v ∧ u, dΩ), X[2], X[3], dΩ),
        Kcc=assemble_bilinear((v, u) -> ∫(d(v) ∧ ★(d(u)), dΩ), X[2], X[2], dΩ),
    )
end
