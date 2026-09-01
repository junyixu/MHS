# 算子层：提取矩阵（r=1 本质 BC + 最小相容轴约束）、缓存分解、
# Leray 投影、谐和场、螺旋度
#
# 约定：所有演化量在「约化坐标」下（reduced = E · full）；full = Eᵀ · reduced。
# E_k 行互不相交（选择行或轴环黏合行），E Eᵀ 为对角。
#
# 约束设计（"C⁻ 级最小 polar"，恰好封住轴丝泄漏且保持复形在系数级精确闭合）：
#   r=1 迹（本质 BC）：值携带 r-因子 (S⁰ᵣ) 的分量去掉末 r-指标
#     —— V⁰ 全部、V¹ 的 θ/ζ 分量、V² 的 [θζ] 分量
#   r=0 轴：V⁰ 轴环黏合（θ-无关）；V¹ θ-分量轴环=0、ζ-分量轴环黏合；
#     V² [θζ]-分量轴环=0；其余不约束
#   闭合性依据：样条方向导数 = 相邻系数差分；黏合环差分恒零、零环导数恒零
#   效果：div 相容性恢复（Leray 需 ∫p 增广，λ=0 精确）、𝔥₀² 恢复 1 维、
#     curl(V¹₀) 不再能经轴丝携带环向通量 ⇒ 环向通量被动力学精确保持

function component_extraction(dm::NTuple{3, Int}, drop_last_r::Bool, axis::Symbol)
    nr, nt, nz = dm
    li = LinearIndices(dm)
    rows, cols, vals = Int[], Int[], Float64[]
    nrow = 0
    if axis == :glue
        for jz in 1:nz
            nrow += 1
            for jt in 1:nt
                push!(rows, nrow); push!(cols, li[1, jt, jz]); push!(vals, 1.0)
            end
        end
    end
    r_lo = axis == :none ? 1 : 2
    r_hi = drop_last_r ? nr - 1 : nr
    for jz in 1:nz, jt in 1:nt, ir in r_lo:r_hi
        nrow += 1
        push!(rows, nrow); push!(cols, li[ir, jt, jz]); push!(vals, 1.0)
    end
    return sparse(rows, cols, vals, nrow, prod(dm))
end

function extraction_matrices(nel::NTuple{3, Int}, deg::NTuple{3, Int})
    n0 = (nel[1] + deg[1], nel[2], nel[3])   # (clamped, 周期, 周期) 0-form 一维维数
    n1 = (n0[1] - 1, nel[2], nel[3])
    dims(idxs) = ntuple(dir -> dir in idxs ? n1[dir] : n0[dir], 3)
    # (1-form 方向组合, drop_last_r, 轴约束) —— 与 build_de_rham_complex 分量顺序一致
    spec = (
        (((), true, :glue),),
        (((1,), false, :none), ((2,), true, :zero), ((3,), true, :glue)),
        (((2, 3), true, :zero), ((1, 3), false, :none), ((1, 2), false, :none)),
        (((1, 2, 3), false, :none),),
    )
    return ntuple(4) do kp1
        blocks = [component_extraction(dims(s[1]), s[2], s[3]) for s in spec[kp1]]
        length(blocks) == 1 ? blocks[1] : blockdiag(blocks...)
    end
end

scatter(ops, k::Int, v::AbstractVector) = ops.E[k + 1]' * v

form_field(ops, k::Int, v::AbstractVector; label="f") =
    Mantis.Forms.build_form_field(ops.X[k + 1], Vector(scatter(ops, k, v)); label=label)

# 叉乘载荷（约化坐标）：J,H 为 V¹ 系数，v2 为 V² 系数
function load_JxH(ops, j::AbstractVector, h::AbstractVector)
    Jf = form_field(ops, 1, j; label="J")
    Hf = form_field(ops, 1, h; label="H")
    return ops.E[3] * assemble_load(ops.X[3] ∧ ★(Jf ∧ Hf), ops.X[3], ops.dΩ)
end

function load_vxH(ops, v2::AbstractVector, h::AbstractVector)
    Vf = form_field(ops, 2, v2; label="v")
    Hf = form_field(ops, 1, h; label="H")
    return ops.E[2] * assemble_load(ops.X[2] ∧ (★(Vf) ∧ Hf), ops.X[2], ops.dΩ)
end

# Leray 投影：v = w − ~grad p，div v = 0（强，机器精度）
# 鞍点 [M2 K_divᵀ; K_div 0] + ∫p=0 增广（轴约束恢复 div 相容性 ⇒ p 有常数
# 零空间；相容性也保证增广乘子 λ = 0 精确成立）
function leray(ops, w::AbstractVector)
    n2, n3 = size(ops.M2, 1), size(ops.M3, 1)
    sol = ops.leray_fact \ [zeros(n2); -(ops.Kdiv * w); 0.0]
    σ = sol[1:n2]                          # σ = −~grad p
    p = -sol[(n2 + 1):(n2 + n3)]           # 鞍点解出的是 −p，翻转为物理压强
    return w + σ, p, σ
end

# 螺旋度用 A-解（k=1 Hodge-Laplace）：
# [−M0 K_gradᵀ; K_grad K_cc][q; A] = [0; K_curlᵀ b]，精确给出 range(curl) 上的
# L² 投影 curl A（q = 0、Coulomb 规范自动成立）
function vector_potential(ops, b::AbstractVector)
    n0 = size(ops.M0, 1)
    sol = ops.avec_fact \ [zeros(n0); ops.Kcurl' * b]
    return sol[(n0 + 1):end]
end

# 广义螺旋度 H = ∫A·(B + Π^𝔥B)，Π^𝔥B = (h,B)_{M2}/(h,h)_{M2}·h（论文 Prop. 17）
function helicity(ops, b::AbstractVector)
    a = vector_potential(ops, b)
    c = dot(ops.hharm, ops.M2 * b) * ops.hharm_normsq_inv
    return dot(a, ops.S12 * (b + c * ops.hharm))
end

# 环向通量（∝ (B, e_φ/R)；差一个 ±2π 因子与定向，用于守恒监控）
toroidal_flux(ops, b::AbstractVector) = dot(ops.fluxw, b) / (2π)

function setup_operators(nel::NTuple{3, Int}, deg::NTuple{3, Int}, mapping; qextra::Int=3)
    X = build_de_rham_complex(nel, deg, mapping)
    dΩ = build_quadrature(X, deg; extra=qextra)
    raw = assemble_operators(X, dΩ)
    E = extraction_matrices(nel, deg)
    nfull = ntuple(k -> Mantis.Forms.get_num_basis(X[k]), 4)
    @assert all(size.(E, 2) .== nfull)

    E0, E1, E2, E3 = E
    M0, M1 = E0 * raw.M0 * E0', E1 * raw.M1 * E1'
    M2, M3 = E2 * raw.M2 * E2', E3 * raw.M3 * E3'
    Kgrad, Kcurl, Kdiv = E1 * raw.Kgrad * E0', E2 * raw.Kcurl * E1', E3 * raw.Kdiv * E2'
    S12, Kcc = E1 * raw.S12 * E2', E1 * raw.Kcc * E1'

    n2, n3 = size(M2, 1), size(M3, 1)
    m3 = sparse(reshape(vec(sum(M3; dims=1)), :, 1))   # ∫p 约束权重
    leray_fact = lu([
        M2 Kdiv' spzeros(n2, 1)
        Kdiv spzeros(n3, n3) m3
        spzeros(1, n2) m3' spzeros(1, 1)
    ])
    avec_fact = lu([
        -M0 Kgrad'
        Kgrad Kcc
    ])

    ops = (
        X=X, dΩ=dΩ, geom=Mantis.Forms.get_geometry(X[1]),
        nel=nel, deg=deg, E=E, nfull=nfull,
        M0=M0, M1=M1, M2=M2, M3=M3,
        M0f=cholesky(Symmetric(M0)), M1f=cholesky(Symmetric(M1)),
        M2f=cholesky(Symmetric(M2)), M3f=cholesky(Symmetric(M3)),
        Kgrad=Kgrad, Kcurl=Kcurl, Kdiv=Kdiv, S12=S12, Kcc=Kcc,
        leray_fact=leray_fact, avec_fact=avec_fact,
        raw=raw,
        fluxw=Float64[], hharm=Float64[], hharm_normsq_inv=0.0,
    )
    fluxw = ops.E[3] * vacuum_toroidal_load(ops)
    ops = merge(ops, (fluxw=fluxw,))
    # 谐和 2-形式（𝔥₀² 一维，环向真空场方向）：ΠLeray 后去掉 curl 部分
    w = ops.M2f \ fluxw
    v, _, _ = leray(ops, w)
    h = v - (ops.M2f \ (Kcurl * vector_potential(ops, v)))
    return merge(ops, (hharm=h, hharm_normsq_inv=1.0 / dot(h, M2 * h)))
end

# Π²₀(e_φ/R) 的载荷（解析 1/R e_φ 场；(Λ², f¹) 的 proxy 配对 = Λ² ∧ f）
function vacuum_toroidal_load(ops)
    function expr(x::Matrix{Float64})
        n = size(x, 1)
        v1, v2, v3 = ntuple(_ -> Vector{Float64}(undef, n), 3)
        for i in 1:n
            R2 = x[i, 1]^2 + x[i, 2]^2
            v1[i] = -x[i, 2] / R2
            v2[i] = x[i, 1] / R2
            v3[i] = 0.0
        end
        return [v1, v2, v3]
    end
    f = Mantis.Forms.AnalyticalFormField(1, expr, ops.geom, "e_φ/R")
    return assemble_load(ops.X[3] ∧ f, ops.X[3], ops.dΩ)
end

# 强微分（约化坐标；样条复形中精确）
strong_grad(ops, p) = ops.M1f \ (ops.Kgrad * p)
strong_curl(ops, a) = ops.M2f \ (ops.Kcurl * a)
strong_div(ops, b) = ops.M3f \ (ops.Kdiv * b)
weak_curl(ops, b) = ops.M1f \ (ops.Kcurl' * b)   # ~curl: V²₀ → V¹₀
proj_21(ops, b) = ops.M1f \ (ops.S12 * b)        # Π¹₀: V²₀ → V¹₀
