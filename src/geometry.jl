# 环形映射 Φ: [0,1]³ → ℝ³，逻辑坐标 (r,θ,ζ)，大半径 R₀ = 1
#
# 论文 Eq. (2)/(7) 的 D 形截面：poloidal 边界曲线（Eq. 7）
#   Γ(t) = (1 + ε cos(t + arcsin(δ) sin t),  ε κ sin t),   t = 2πθ
# 相对几何中心 (R,z)=(1,0) 做线性径向缩放，故
#   R(r,θ) = 1 + r f(θ),   f(θ) = ε cos(2πθ + d sin 2πθ),  d = arcsin δ
#   Z(r,θ) =     r g(θ),   g(θ) = ε κ sin 2πθ
#   Φ = (R cos2πζ, −R sin2πζ, Z)      （取 −sin 使 det DΦ > 0）
# det DΦ = 2π r R (f g′ − f′ g)；圆截面 (κ=1, δ=0) 退化为 4π²ε² r R。
#
# ε 反环径比（minor/major），κ 拉长比（circle→ellipse），δ 三角形变（ellipse→D）
function torus_mapping(ε::Float64, κ::Float64=1.0, δ::Float64=0.0)
    d = asin(δ)
    f(θ) = ε * cos(2π * θ + d * sinpi(2θ))
    g(θ) = ε * κ * sinpi(2θ)
    df(θ) = -2π * ε * sin(2π * θ + d * sinpi(2θ)) * (1 + d * cospi(2θ))
    dg(θ) = 2π * ε * κ * cospi(2θ)

    function Φ(x::AbstractVector)
        r, θ, ζ = x[1], x[2], x[3]
        R = 1 + r * f(θ)
        return [R * cospi(2ζ), -R * sinpi(2ζ), r * g(θ)]
    end
    function DΦ(x::AbstractVector)
        r, θ, ζ = x[1], x[2], x[3]
        C, S = cospi(2ζ), sinpi(2ζ)
        R = 1 + r * f(θ)
        fθ, gθ, dfθ, dgθ = f(θ), g(θ), df(θ), dg(θ)
        # J[i,j] = ∂Φᵢ/∂x̂ⱼ，SMatrix 按列填充
        return SMatrix{3, 3}(
            fθ * C, -fθ * S, gθ,                       # ∂/∂r
            r * dfθ * C, -r * dfθ * S, r * dgθ,        # ∂/∂θ
            -2π * R * S, -2π * R * C, 0.0,             # ∂/∂ζ
        )
    end
    return Mantis.Geometry.Mapping((3, 3), Φ, DΦ)
end

circular_torus_mapping(ε::Float64) = torus_mapping(ε)

# 3D 仿星器映射（论文 Eq. 3）：截面半径随环向角调制，破坏轴对称
#   R(r,θ,ζ) = 1 + r ε a(ζ) cos2πθ,   Z(r,θ,ζ) = r ε b(ζ) sin2πθ
#   ν(ζ) = 1 + (1 − κ) cos(2 n_fp π ζ)，论文 Eq. (3)：a(ζ) = ν(ζ)、b(ζ) = ν(ζ+1/2)
#   det DΦ = 4π² ε² a(ζ) b(ζ) r R > 0
# **半周期偏移是关键**：R、Z 半轴反相 ⇒ 截面是随 ζ 转 90° 的椭圆（旋转椭圆仿星器）。
#   若误取 a = b = ν，截面恒为圆、只有半径脉动，那是波纹环——没有螺旋成形、
#   没有真空旋转变换，根本不是仿星器（见 PITFALLS #9）。
#   κ = 1 ⇒ ν ≡ 1 ⇒ 退化为圆截面环，可作自检。
function stellarator_mapping(ε::Float64, κ::Float64, n_fp::Int)
    amp = 1 - κ
    ω = 2 * n_fp * π
    ν(ζ) = 1 + amp * cos(ω * ζ)
    dν(ζ) = -amp * ω * sin(ω * ζ)
    a, da = ν, dν                        # R 半轴
    b, db = (ζ -> ν(ζ + 0.5)), (ζ -> dν(ζ + 0.5))   # Z 半轴：ν(ζ+1/2)

    function Φ(x::AbstractVector)
        r, θ, ζ = x[1], x[2], x[3]
        R = 1 + r * ε * a(ζ) * cospi(2θ)
        return [R * cospi(2ζ), -R * sinpi(2ζ), r * ε * b(ζ) * sinpi(2θ)]
    end
    function DΦ(x::AbstractVector)
        r, θ, ζ = x[1], x[2], x[3]
        c, s = cospi(2θ), sinpi(2θ)
        C, S = cospi(2ζ), sinpi(2ζ)
        aζ, bζ, daζ, dbζ = a(ζ), b(ζ), da(ζ), db(ζ)
        R = 1 + r * ε * aζ * c
        Rr, Zr = ε * aζ * c, ε * bζ * s
        Rθ, Zθ = -2π * r * ε * aζ * s, 2π * r * ε * bζ * c
        Rζ, Zζ = r * ε * daζ * c, r * ε * dbζ * s
        # J[i,j] = ∂Φᵢ/∂x̂ⱼ，SMatrix 按列填充
        return SMatrix{3, 3}(
            Rr * C, -Rr * S, Zr,                         # ∂/∂r
            Rθ * C, -Rθ * S, Zθ,                         # ∂/∂θ
            Rζ * C - 2π * R * S, -Rζ * S - 2π * R * C, Zζ,   # ∂/∂ζ
        )
    end
    return Mantis.Geometry.Mapping((3, 3), Φ, DΦ)
end

# 仿星器截面调制因子与其在 (r,θ,ζ) 处的物理位置（画图/验证用）
stellarator_nu(κ::Float64, n_fp::Int, ζ) = 1 + (1 - κ) * cos(2 * n_fp * π * ζ)
function stellarator_position(ε::Float64, κ::Float64, n_fp::Int, r, θ, ζ)
    a, b = stellarator_nu(κ, n_fp, ζ), stellarator_nu(κ, n_fp, ζ + 0.5)
    return (1 + r * ε * a * cospi(2θ), r * ε * b * sinpi(2θ))
end

# --- 画图侧辅助（与上面的 f, g 定义一致）---

# 逻辑 (r,θ) → 物理 (R,z)
function poloidal_position(ε::Float64, κ::Float64, δ::Float64, r, θ)
    dd = asin(δ)
    return (1 + r * ε * cos(2π * θ + dd * sinpi(2θ)), r * ε * κ * sinpi(2θ))
end

# 极向 Jacobian 因子 W(θ) = f g′ − f′ g，满足 det DΦ = 2π r R W(θ)
# 圆截面退化为常数 2πε²
function poloidal_jacobian(ε::Float64, κ::Float64, δ::Float64, θ)
    dd = asin(δ)
    ψ = 2π * θ + dd * sinpi(2θ)
    f = ε * cos(ψ)
    g = ε * κ * sinpi(2θ)
    df = -2π * ε * sin(ψ) * (1 + dd * cospi(2θ))
    dg = 2π * ε * κ * cospi(2θ)
    return f * dg - df * g
end
