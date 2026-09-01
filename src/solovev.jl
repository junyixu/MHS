# Solov'ev 平衡（论文 Eq. 8），R₀ = 1，单位制 μ₀ = 1
#   ψ(R,z) = -1/2 ( κ̄²/4 (R²-1)² + R²z² )
#   B⁰ = Rz e_R + (τ/R) e_φ - (κ̄²/2 (R²-1) + z²) e_z，τ = q* κ̄(κ̄²+1)/(κ̄+1)
#   A⁰ = (ψ/R) e_φ - τ ln(R) e_z，满足 curl A⁰ = B⁰
#   p⁰ = (κ̄²+1) ψ，满足 J⁰×B⁰ = grad p⁰
solovev_tau(q_star; κ̄=1.0) = q_star * κ̄ * (κ̄^2 + 1) / (κ̄ + 1)

solovev_ψ(R, z; κ̄=1.0) = -0.5 * (κ̄^2 / 4 * (R^2 - 1)^2 + R^2 * z^2)

function solovev_B(x1, x2, z, τ; κ̄=1.0)
    R2 = x1^2 + x2^2
    R = sqrt(R2)
    BR = R * z
    Bz = -(κ̄^2 / 2 * (R2 - 1) + z^2)
    # e_R = (x1, x2, 0)/R, e_φ = (-x2, x1, 0)/R
    return (
        BR * x1 / R - τ * x2 / R2,
        BR * x2 / R + τ * x1 / R2,
        Bz,
    )
end

# 矢势 A⁰（1-形式，物理 proxy 分量）
function solovev_A_field(q_star, geom; κ̄=1.0)
    τ = solovev_tau(q_star; κ̄=κ̄)
    function expr(x::Matrix{Float64})
        n = size(x, 1)
        A1, A2, A3 = ntuple(_ -> Vector{Float64}(undef, n), 3)
        for i in 1:n
            x1, x2, z = x[i, 1], x[i, 2], x[i, 3]
            R2 = x1^2 + x2^2
            Aφ_over_R = solovev_ψ(sqrt(R2), z; κ̄=κ̄) / R2
            A1[i] = -Aφ_over_R * x2
            A2[i] = Aφ_over_R * x1
            A3[i] = -τ * log(R2) / 2
        end
        return [A1, A2, A3]
    end
    return Mantis.Forms.AnalyticalFormField(1, expr, geom, "A⁰")
end

# B⁰ 作为 1-形式（proxy 同矢量；H = Π¹B 的解析原型）
function solovev_B1_field(q_star, geom; κ̄=1.0)
    τ = solovev_tau(q_star; κ̄=κ̄)
    function expr(x::Matrix{Float64})
        n = size(x, 1)
        B1, B2, B3 = ntuple(_ -> Vector{Float64}(undef, n), 3)
        for i in 1:n
            B1[i], B2[i], B3[i] = solovev_B(x[i, 1], x[i, 2], x[i, 3], τ; κ̄=κ̄)
        end
        return [B1, B2, B3]
    end
    return Mantis.Forms.AnalyticalFormField(1, expr, geom, "B⁰₁")
end

# 电流 J⁰ = curl B⁰ = (κ̄²+1) R e_φ（1-形式）
function solovev_J_field(geom; κ̄=1.0)
    function expr(x::Matrix{Float64})
        n = size(x, 1)
        J1, J2, J3 = ntuple(_ -> Vector{Float64}(undef, n), 3)
        for i in 1:n
            J1[i] = -(κ̄^2 + 1) * x[i, 2]
            J2[i] = (κ̄^2 + 1) * x[i, 1]
            J3[i] = 0.0
        end
        return [J1, J2, J3]
    end
    return Mantis.Forms.AnalyticalFormField(1, expr, geom, "J⁰")
end

# 压强 p⁰（0-形式）
function solovev_p_field(geom; κ̄=1.0)
    function expr(x::Matrix{Float64})
        n = size(x, 1)
        p = Vector{Float64}(undef, n)
        for i in 1:n
            R = sqrt(x[i, 1]^2 + x[i, 2]^2)
            p[i] = (κ̄^2 + 1) * solovev_ψ(R, x[i, 3]; κ̄=κ̄)
        end
        return [p]
    end
    return Mantis.Forms.AnalyticalFormField(0, expr, geom, "p⁰")
end
