# 磁弛豫（理想 η=0，γ=0 即 𝒜 = Π^Leray）
# 两种时间步进（与 mrx 一致）：
#  :explicit_ls — 显式步 + 解析线搜索 δt = ‖v‖²/‖curl E‖²（能量沿方向的精确
#                 一维极小；mrx 默认，论文算例配置 descent_method: gradient）
#  :midpoint    — 论文 Eq. 6 中点 Picard + 自适应 δt（结构保持性证明所用格式）

mnorm(M, v) = sqrt(dot(v, M * v))

# Solov'ev 初始化管线：A 投影(全空间) → 强 curl → 限制到 V²₀ → Leray 清理 → 归一化
# 返回强无散（机器精度）、B·n=0 的初始系数
function solovev_initial_field(ops, q_star)
    Ah = Mantis.Assemblers.solve_L2_projection(
        ops.X[2], solovev_A_field(q_star, ops.geom), ops.dΩ)
    b_full = ops.raw.M2 \ Vector(ops.raw.Kcurl * Ah.coefficients)
    b = Vector(ops.M2f \ (ops.E[3] * (ops.raw.M2 * b_full)))
    b, _, _ = leray(ops, b)
    return b ./ mnorm(ops.M2, b)
end

# 力评估：b 的系数 → (v, p, σ, j, h)
#   J = ~curl B, H = Π¹₀B, w = Π²₀(J×H), v = Π^Leray w (= J×B−∇p 的 Riesz 表示)
function force_velocity(ops, b)
    j = weak_curl(ops, b)
    h = proj_21(ops, b)
    w = ops.M2f \ load_JxH(ops, j, h)
    v, p, σ = leray(ops, w)
    return v, p, σ, j, h
end

# 单步中点 Picard（朴素不动点 y ← f(y)，与 mrx 实现一致）
function picard_step(ops, b, dt, tol, kmax)
    y = copy(b)
    resid = Inf
    local v, σ
    for k in 1:kmax
        bh = 0.5 .* (b .+ y)
        v, p, σ, j, h = force_velocity(ops, bh)
        e = ops.M1f \ load_vxH(ops, v, h)
        f = b .+ dt .* strong_curl(ops, e)
        resid = mnorm(ops.M2, f .- y)
        y = f
        if resid < tol || !isfinite(resid)
            return y, k, resid, v, σ
        end
    end
    return y, kmax, resid, v, σ
end

function relax(ops, b0; dt0, N_iter, scheme=:explicit_ls, tol=1e-9,
    kstar=4, kmax=20, eps_dt=0.01, helicity_every=50, print_every=100)
    b = copy(b0)
    dt = dt0
    hist = (
        energy=Float64[], force=Float64[], divB=Float64[], dt=Float64[],
        picard=Int[], helicity_iter=Int[], helicity=Float64[], flux=Float64[],
    )
    n = 0
    while n < N_iter
        local v, σ, k
        if scheme == :explicit_ls
            v, p, σ, j, h = force_velocity(ops, b)
            e = ops.M1f \ load_vxH(ops, v, h)
            db = strong_curl(ops, e)
            db2 = dot(db, ops.M2 * db)
            db2 > 0 || break
            dt = -dot(b, ops.M2 * db) / db2   # 能量沿 db 的一维精确极小
            b = b .+ dt .* db
            k = 1
        elseif scheme == :midpoint
            y, k, resid, v, σ = picard_step(ops, b, dt, tol, kmax)
            if (k == kmax && resid > tol) || !isfinite(resid)
                dt /= 2                       # 未收敛：减 δt 重试本步
                continue
            end
            b = y
            dt = k > kstar ? dt / (1 + eps_dt)^2 : dt * (1 + eps_dt)
        else
            error("未知 scheme: $scheme")
        end
        n += 1
        push!(hist.energy, 0.5 * dot(b, ops.M2 * b))
        push!(hist.force, mnorm(ops.M2, v) / mnorm(ops.M2, σ))
        push!(hist.divB, norm(ops.Kdiv * b))
        push!(hist.dt, dt)
        push!(hist.picard, k)
        if n % helicity_every == 0 || n == 1
            push!(hist.helicity_iter, n)
            push!(hist.helicity, helicity(ops, b))
            push!(hist.flux, toroidal_flux(ops, b))
        end
        if n % print_every == 0 || n == 1
            println("iter $n: force = $(hist.force[end]), energy = $(hist.energy[end]), " *
                    "dt = $dt, picard = $k")
            flush(stdout)
        end
    end
    return b, hist
end
