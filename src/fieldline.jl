# 场重构、磁面标签与场线追踪（plot_.jl 与 validate.jl 共用）
#
# 走矢势路线：Mantis v0.6.0 的 2-形式 FormField 求值分量互相混合（见 PITFALLS.md §3），
# 而 1-/3-形式约定正常。故解 A 后由 Â 的差分重构 B̂。
#
# 轴对称下 Âζ 就是 2π×极向磁通函数：Âζ = A·∂ζΦ = 2πR·A_φ = 2πψ，
# 故磁面 = Âζ 等值线，且 ψ_h = Âζ/2π 可直接与解析 ψ 比较。

# ---------- 网格求值（k-形式系数 → (r,θ) 网格上的 canonical 分量）----------
function eval_grid(ops, k, coeffs; nr=241, nt=289)
    field = form_field(ops, k, coeffs)
    nel_r, nel_t = ops.nel[1], ops.nel[2]
    rg = collect(range(0.0, 1.0; length=nr))
    tg = collect(range(0.0, 1.0; length=nt))
    elr = [min(floor(Int, r * nel_r) + 1, nel_r) for r in rg]
    xir = [rg[i] * nel_r - (elr[i] - 1) for i in 1:nr]
    elt = [min(floor(Int, t * nel_t) + 1, nel_t) for t in tg]
    xit = [tg[j] * nel_t - (elt[j] - 1) for j in 1:nt]
    ncomp = k == 1 ? 3 : 1
    out = [zeros(nr, nt) for _ in 1:ncomp]
    for et in 1:nel_t, er in 1:nel_r
        is = findall(==(er), elr)
        js = findall(==(et), elt)
        (isempty(is) || isempty(js)) && continue
        pts = Mantis.Points.CartesianPoints((xir[is], xit[js], [0.5]))
        vals, _ = Mantis.Forms.evaluate(field, er + nel_r * (et - 1), pts)
        for c in 1:ncomp
            out[c][is, js] .= reshape(vals[c][:, 1], length(is), length(js))
        end
    end
    return rg, tg, out
end

# ---------- 网格差分（r 边界单侧；θ 周期，首末列为同一点）----------
function ddr(F, rg)
    G = similar(F)
    h = rg[2] - rg[1]
    G[2:(end - 1), :] .= (F[3:end, :] .- F[1:(end - 2), :]) ./ (2h)
    G[1, :] .= (F[2, :] .- F[1, :]) ./ h
    G[end, :] .= (F[end, :] .- F[end - 1, :]) ./ h
    return G
end
function ddt(F, tg)
    G = similar(F)
    h = tg[2] - tg[1]
    G[:, 2:(end - 1)] .= (F[:, 3:end] .- F[:, 1:(end - 2)]) ./ (2h)
    G[:, 1] .= (F[:, 2] .- F[:, end - 1]) ./ (2h)
    G[:, end] .= G[:, 1]
    return G
end

# ---------- 双线性插值（θ 周期）----------
function bilinear(rg, tg, F)
    nr, nt = length(rg), length(tg)
    dr, dtt = rg[2] - rg[1], tg[2] - tg[1]
    return function (r, t)
        tm = mod(t, 1.0)
        ir = clamp(floor(Int, r / dr) + 1, 1, nr - 1)
        it = clamp(floor(Int, tm / dtt) + 1, 1, nt - 1)
        fr = clamp((r - rg[ir]) / dr, 0.0, 1.0)
        ft = (tm - tg[it]) / dtt
        return (1 - fr) * (1 - ft) * F[ir, it] + fr * (1 - ft) * F[ir + 1, it] +
               (1 - fr) * ft * F[ir, it + 1] + fr * ft * F[ir + 1, it + 1]
    end
end

trapz2(F, rg, tg) =
    sum((F[1:(end - 1), 1:(end - 1)] .+ F[2:end, 1:(end - 1)] .+
         F[1:(end - 1), 2:end] .+ F[2:end, 2:end]) ./ 4) *
    (rg[2] - rg[1]) * (tg[2] - tg[1])

# ---------- 从 2-形式系数 b 重构逻辑分量 B̂ ----------
# 返回 NamedTuple: rg, tg, Br, Bt, Bz（B̂ 分量矩阵）, Az（=2πψ_h）, R（R̂ 网格）, W（W(θ)）
function reconstruct_bhat(ops, b, ε, κ, δ; nr=241, nt=289)
    a = vector_potential(ops, b)
    rg, tg, Anum = eval_grid(ops, 1, a; nr=nr, nt=nt)
    # canonical → 全局逻辑分量: Â_k = num_k / h_k,  h = (1/n_r, 1/n_θ, 1)
    Ar, At, Az = Anum[1] .* ops.nel[1], Anum[2] .* ops.nel[2], Anum[3]

    R = [poloidal_position(ε, κ, δ, r, t)[1] for r in rg, t in tg]
    W = [poloidal_jacobian(ε, κ, δ, t) for t in tg]
    # 真空场 (1/R)e_φ 的 2-形式拉回 ζ-分量: V̂ζ = det(DΦ)/(2πR²) = r W(θ)/R̂
    Vz = [rg[i] * W[j] / R[i, j] for i in eachindex(rg), j in eachindex(tg)]
    curlz = ddr(At, rg) .- ddt(Ar, tg)
    # 谐和系数由环向通量匹配；∫∫curlz = 0（边界项为零）故 c 良定
    c = (-toroidal_flux(ops, b) - trapz2(curlz, rg, tg)) / trapz2(Vz, rg, tg)

    return (rg=rg, tg=tg,
        Br=ddt(Az, tg), Bt=-ddr(Az, rg), Bz=curlz .+ c .* Vz,
        Az=Az, R=R, W=W, c=c, curlz_int=trapz2(curlz, rg, tg))
end

# ---------- 3-形式系数 → 物理标量（p = p̂·n_r·n_θ/det DΦ）----------
function scalar_from_3form(ops, coeffs, ε, κ, δ, rg, tg, R, W)
    _, _, num = eval_grid(ops, 3, coeffs; nr=length(rg), nt=length(tg))
    detJ = [2π * max(rg[i], 1e-6) * R[i, j] * W[j]
            for i in eachindex(rg), j in eachindex(tg)]
    F = num[1] .* (ops.nel[1] * ops.nel[2]) ./ detJ
    F[1, :] .= F[2, :]        # 轴上 0/0，用邻行外推
    return F
end

# ---------- 解析 Solov'ev 场的逻辑分量 B̂ = det(J)·J⁻¹·B ----------
function solovev_bhat_grid(ε, κ, δ, q_star, rg, tg; κ̄=1.0)
    mp = torus_mapping(ε, κ, δ)
    τ = solovev_tau(q_star; κ̄=κ̄)
    Br = zeros(length(rg), length(tg))
    Bt, Bz = similar(Br), similar(Br)
    for (i, r) in enumerate(rg), (j, t) in enumerate(tg)
        x̂ = [r, t, 0.0]
        J = mp.dmapping(x̂)
        x = mp.mapping(x̂)
        B = collect(solovev_B(x[1], x[2], x[3], τ; κ̄=κ̄))
        v = r < 1e-9 ? zeros(3) : det(J) * (J \ B)   # 轴上 J 奇异
        Br[i, j], Bt[i, j], Bz[i, j] = v
    end
    return Br, Bt, Bz
end

# ‖B₀‖_{L²(Ω)}：∫|B₀|² det(DΦ) dr dθ dζ，det = 2π r R W(θ)
function solovev_L2norm(ε, κ, δ, q_star; κ̄=1.0, nr=801, nt=800)
    τ = solovev_tau(q_star; κ̄=κ̄)
    tot = 0.0
    for (i, r) in enumerate(range(0, 1; length=nr))
        wr = (i == 1 || i == nr) ? 1.0 : (i % 2 == 0 ? 4.0 : 2.0)   # Simpson
        for t in range(0, 1 - 1 / nt; length=nt)                     # 周期梯形
            R, Z = poloidal_position(ε, κ, δ, r, t)
            B = solovev_B(R, 0.0, Z, τ; κ̄=κ̄)
            tot += wr * (B[1]^2 + B[2]^2 + B[3]^2) *
                   2π * r * R * poloidal_jacobian(ε, κ, δ, t) / nt
        end
    end
    return sqrt(tot / (nr - 1) / 3)
end

# ---------- 磁轴定位（压强/磁通极值点，抛物线细化）----------
function find_axis(F, rg, tg, ε, κ, δ; maximum_of=true)
    idx = maximum_of ? argmax(F) : argmin(F)
    return poloidal_position(ε, κ, δ, rg[idx[1]], tg[idx[2]])..., idx
end

# ---------- 场线追踪 ----------
# dr/dζ = B̂r/B̂ζ, dθ/dζ = B̂θ/B̂ζ（RK4 定步长），ι 由绕磁轴的极向角累积
# （逻辑 θ̂ 不可用：坐标轴 r=0 与磁轴因 Shafranov 位移不重合，见 PITFALLS.md §6）
function trace_line(Br_f, Bt_f, Bz_f, RZ, R_ax, z_ax, r0;
    transits=400, steps_per=64, rmin=0.002, rmax=0.999)
    f(u) = (bz = Bz_f(u[1], u[2]);
    SVector(Br_f(u[1], u[2]) / bz, Bt_f(u[1], u[2]) / bz))
    h = 1.0 / steps_per
    u = SVector(r0, 0.0)
    pts = SVector{2, Float64}[]
    R0, z0 = RZ(u[1], u[2])
    ang_prev = atan(z0 - z_ax, R0 - R_ax)
    ang_tot = 0.0
    for _ in 1:transits
        for _ in 1:steps_per
            k1 = f(u); k2 = f(u + h / 2 * k1)
            k3 = f(u + h / 2 * k2); k4 = f(u + h * k3)
            un = u + h / 6 * (k1 + 2k2 + 2k3 + k4)
            (rmin < un[1] < rmax) && all(isfinite, un) ||
                return pts, isempty(pts) ? NaN : ang_tot / (2π * length(pts))
            u = un
            Rc, zc = RZ(u[1], u[2])
            ang = atan(zc - z_ax, Rc - R_ax)
            dα = ang - ang_prev
            ang_tot += dα - 2π * round(dα / (2π))   # wrap 到 (−π, π]
            ang_prev = ang
        end
        push!(pts, u)
    end
    return pts, ang_tot / (2π * transits)
end

############################################################################################
#                        3D（非轴对称）场重构与场线追踪                                    #
############################################################################################
# 2D 那边走矢势路线，是因为当时误判「2-形式求值分量互相混合」（真正坏的是 3D rank-2
# 的解析 AnalyticalFormField 参考侧，见 PITFALLS #3）。2-形式求值其实是好的，
# 所以 3D 直接求值 b，不需要矢势、也不需要把谐和场近似成解析真空场
# —— 后者在仿星器上根本不成立（壁面不是回转面，cos∠ 只有 0.82）。

# k-形式系数 → (r,θ,ζ) 三维网格上的 canonical 分量
function eval_grid3(ops, k, coeffs; nr=97, nt=97, nz=49)
    field = form_field(ops, k, coeffs)
    ne = ops.nel
    rg, tg, zg = collect(range(0, 1; length=nr)), collect(range(0, 1; length=nt)),
    collect(range(0, 1; length=nz))
    loc(g, n) = (el = [min(floor(Int, x * n) + 1, n) for x in g];
    (el, [g[i] * n - (el[i] - 1) for i in eachindex(g)]))
    elr, xir = loc(rg, ne[1])
    elt, xit = loc(tg, ne[2])
    elz, xiz = loc(zg, ne[3])
    ncomp = (k == 1 || k == 2) ? 3 : 1
    out = [zeros(nr, nt, nz) for _ in 1:ncomp]
    for ez in 1:ne[3], et in 1:ne[2], er in 1:ne[1]
        is, js, ks = findall(==(er), elr), findall(==(et), elt), findall(==(ez), elz)
        (isempty(is) || isempty(js) || isempty(ks)) && continue
        pts = Mantis.Points.CartesianPoints((xir[is], xit[js], xiz[ks]))
        eid = er + ne[1] * ((et - 1) + ne[2] * (ez - 1))
        vals, _ = Mantis.Forms.evaluate(field, eid, pts)
        for c in 1:ncomp
            out[c][is, js, ks] .= reshape(vals[c][:, 1], length(is), length(js), length(ks))
        end
    end
    return rg, tg, zg, out
end

trapz3(F, rg, tg, zg) = sum(
    (F[1:(end - 1), 1:(end - 1), 1:(end - 1)] .+ F[2:end, 1:(end - 1), 1:(end - 1)] .+
     F[1:(end - 1), 2:end, 1:(end - 1)] .+ F[1:(end - 1), 1:(end - 1), 2:end] .+
     F[2:end, 2:end, 1:(end - 1)] .+ F[2:end, 1:(end - 1), 2:end] .+
     F[1:(end - 1), 2:end, 2:end] .+ F[2:end, 2:end, 2:end]) ./ 8) *
                       (rg[2] - rg[1]) * (tg[2] - tg[1]) * (zg[2] - zg[1])

# 2-形式系数 → 逻辑分量 B̂（canonical → 全局：分量 [jk] 乘单元测度 n_j·n_k）
# flux_check 是与独立装配的环向通量的对照：∫∫∫B̂_ζ 应等于 −toroidal_flux
function reconstruct_bhat3(ops, b; nr=97, nt=97, nz=49)
    rg, tg, zg, Bn = eval_grid3(ops, 2, b; nr=nr, nt=nt, nz=nz)
    n1, n2, n3 = ops.nel
    Bz = Bn[3] .* (n1 * n2)
    return (rg=rg, tg=tg, zg=zg,
        Br=Bn[1] .* (n2 * n3), Bt=Bn[2] .* (n1 * n3), Bz=Bz,
        flux_check=(trapz3(Bz, rg, tg, zg), -toroidal_flux(ops, b)))
end

# 三线性插值（θ、ζ 周期）
function trilinear(rg, tg, zg, F)
    nr, nt, nz = length(rg), length(tg), length(zg)
    dr, dt, dz = rg[2] - rg[1], tg[2] - tg[1], zg[2] - zg[1]
    return function (r, t, z)
        tm, zm = mod(t, 1.0), mod(z, 1.0)
        i = clamp(floor(Int, r / dr) + 1, 1, nr - 1)
        j = clamp(floor(Int, tm / dt) + 1, 1, nt - 1)
        k = clamp(floor(Int, zm / dz) + 1, 1, nz - 1)
        fr = clamp((r - rg[i]) / dr, 0.0, 1.0)
        ft, fz = (tm - tg[j]) / dt, (zm - zg[k]) / dz
        v = 0.0
        for (di, wi) in ((0, 1 - fr), (1, fr)), (dj, wj) in ((0, 1 - ft), (1, ft)),
            (dk, wk) in ((0, 1 - fz), (1, fz))

            v += wi * wj * wk * F[i + di, j + dj, k + dk]
        end
        return v
    end
end

# 3D 场线追踪：dr/dζ = B̂r/B̂ζ, dθ/dζ = B̂θ/B̂ζ
# 每 steps_per/nsub 步记录一次（nsub=1 即每整圈一个 Poincaré 点）。
# 算 ι 需要 nsub>1：单圈的极向角增量可能超过 π，只按整圈采样会发生混叠。
function trace_line3(Br_f, Bt_f, Bz_f, r0, θ0;
    transits=400, steps_per=96, nsub=1, rmin=0.002, rmax=0.999)
    f(u, z) = (bz = Bz_f(u[1], u[2], z);
    SVector(Br_f(u[1], u[2], z) / bz, Bt_f(u[1], u[2], z) / bz))
    h = 1.0 / steps_per
    stride = max(steps_per ÷ nsub, 1)
    u, ζ = SVector(r0, θ0), 0.0
    pts = SVector{2, Float64}[]        # 每 stride 步一个（含子截面）
    step = 0
    for _ in 1:(transits * steps_per)
        k1 = f(u, ζ); k2 = f(u + h / 2 * k1, ζ + h / 2)
        k3 = f(u + h / 2 * k2, ζ + h / 2); k4 = f(u + h * k3, ζ + h)
        un = u + h / 6 * (k1 + 2k2 + 2k3 + k4)
        (rmin < un[1] < rmax) && all(isfinite, un) || return pts, false
        u, ζ = un, ζ + h
        step += 1
        step % stride == 0 && push!(pts, u)
    end
    return pts, true
end

# 由子采样轨迹算 ι：绕参考轴（各子截面上的磁轴位置）累积极向角
# axis_pts[m] = 第 m 个子截面上的磁轴 (R,z)
function iota_from_track(pts, RZsub, axis_pts, nsub)
    isempty(pts) && return NaN
    ang_tot, prev = 0.0, NaN
    for (i, u) in enumerate(pts)
        m = mod1(i, nsub)
        R, z = RZsub(u[1], u[2], m)
        α = atan(z - axis_pts[m][2], R - axis_pts[m][1])
        if !isnan(prev)
            dα = α - prev
            ang_tot += dα - 2π * round(dα / (2π))
        end
        prev = α
    end
    return ang_tot / (2π * (length(pts) - 1) / nsub)   # 极向圈数 / 环向圈数
end

# 平均轨道半径（用于剔除坐在磁轴上的退化种子线）
mean_orbit_radius(pts, RZ, R_ax, z_ax) =
    isempty(pts) ? 0.0 :
    sum(hypot(RZ(u[1], u[2])[1] - R_ax, RZ(u[1], u[2])[2] - z_ax) for u in pts) / length(pts)

# 3D 版：参考点逐子截面不同（磁轴是闭合曲线）
mean_orbit_radius3(pts, RZsub, axis_pts, nsub) =
    isempty(pts) ? 0.0 :
    sum(enumerate(pts)) do (i, u)
        m = mod1(i, nsub)
        R, z = RZsub(u[1], u[2], m)
        hypot(R - axis_pts[m][1], z - axis_pts[m][2])
    end / length(pts)
