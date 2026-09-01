# 圆截面环形映射 Φ: [0,1]³ → ℝ³，逻辑坐标 (r,θ,ζ)，R₀ = 1
# Φ = (R̂ cos2πζ, -R̂ sin2πζ, ε r sin2πθ)，R̂ = 1 + ε r cos2πθ
# 取 -sin 使 det DΦ = 4π²ε² r R̂ > 0
function circular_torus_mapping(ε::Float64)
    function Φ(x::AbstractVector)
        r, θ, ζ = x[1], x[2], x[3]
        R = 1 + ε * r * cospi(2θ)
        return [R * cospi(2ζ), -R * sinpi(2ζ), ε * r * sinpi(2θ)]
    end
    function DΦ(x::AbstractVector)
        r, θ, ζ = x[1], x[2], x[3]
        c, s = cospi(2θ), sinpi(2θ)
        C, S = cospi(2ζ), sinpi(2ζ)
        R = 1 + ε * r * c
        # J[i,j] = ∂Φᵢ/∂x̂ⱼ，SMatrix 按列填充
        return SMatrix{3, 3}(
            ε * c * C, -ε * c * S, ε * s,
            -2π * ε * r * s * C, 2π * ε * r * s * S, 2π * ε * r * c,
            -2π * R * S, -2π * R * C, 0.0,
        )
    end
    return Mantis.Geometry.Mapping((3, 3), Φ, DΦ)
end
