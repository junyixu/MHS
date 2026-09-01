# validate.jl — 定量验证：算子收敛阶 + 与解析 Solov'ev 平衡的对比 + p/ι 剖面
# 用法: julia --project=. validate.jl
# 输出: data/validate_profiles.png, 终端收敛表
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays, Printf, JLD2, GLMakie
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/operators.jl")
include("src/solovev.jl")
include("src/relaxation.jl")
include("src/fieldline.jl")

mkpath("data")

############################################################################################
# 第 1 部分：离散算子在精确 Solov'ev 解上的收敛阶
############################################################################################
# Solov'ev 场满足 J⁰×B⁰ = grad p⁰ 精确成立，故
#   res(n) = ‖Π²(J⁰×B⁰) − grad Π⁰p⁰‖ / ‖grad Π⁰p⁰‖
# 纯粹度量离散化误差（投影、叉乘装配、微分算子），应以样条阶收敛。
# 用解析 J⁰、B⁰（1-形式，Mantis 约定正常）在全空间上做，避开边界条件问题。
function operator_residual(nel, deg, ε, κ, δ, q_star; κ̄=1.0)
    ops = setup_operators(nel, deg, torus_mapping(ε, κ, δ))
    geom, dΩ, X = ops.geom, ops.dΩ, ops.X
    Jf = solovev_J_field(geom; κ̄=κ̄)
    Bf = solovev_B1_field(q_star, geom; κ̄=κ̄)
    jxb = ops.raw.M2 \ assemble_load(X[3] ∧ ★(Jf ∧ Bf), X[3], dΩ)
    ph = Mantis.Assemblers.solve_L2_projection(X[1], solovev_p_field(geom; κ̄=κ̄), dΩ)
    gp = ops.raw.M1 \ Vector(ops.raw.Kgrad * ph.coefficients)
    gp2 = dot(gp, ops.raw.M1 * gp)
    err2 = dot(jxb, ops.raw.M2 * jxb) - 2 * dot(gp, ops.raw.S12 * jxb) + gp2
    return sqrt(max(err2, 0.0)) / sqrt(gp2)
end

println("="^78)
println("第 1 部分：离散算子在精确 Solov'ev 解上的收敛（p=3，期望阶 ≥ 3）")
println("="^78)
for (name, ε, κ, δ, κ̄) in (("圆截面 (ε=1/3)", 1 / 3, 1.0, 0.0, 1.0),
    ("ITER D 形 (ε=.33,κ=1.7,δ=.33)", 0.33, 1.7, 0.33, 1.7))
    println("\n$name")
    @printf("  %-6s %-14s %s\n", "n", "力残差", "观测阶")
    prev_n, prev_r = 0, 0.0
    for n in (4, 6, 8, 12, 16)
        r = operator_residual((n, 2n, 1), (3, 3, 1), ε, κ, δ, 1.57; κ̄=κ̄)
        ord = prev_n == 0 ? NaN : log(prev_r / r) / log(n / prev_n)
        @printf("  %-6d %-14.4e %s\n", n, r, isnan(ord) ? "—" : @sprintf("%.2f", ord))
        prev_n, prev_r = n, r
    end
end

############################################################################################
# 第 2 部分：弛豫解 vs 解析 Solov'ev
############################################################################################
println("\n" * "="^78)
println("第 2 部分：弛豫解与解析 Solov'ev 场的差异")
println("="^78)

# ⟨b, B₀⟩：b 为 V²₀ 系数，B₀ 为解析 1-形式（proxy 配对 = Λ² ∧ B₀）
inner_with_analytic(ops, b, q_star; κ̄=1.0) = dot(b,
    ops.E[3] * assemble_load(ops.X[3] ∧ solovev_B1_field(q_star, ops.geom; κ̄=κ̄),
        ops.X[3], ops.dΩ))

results = Dict{String, Any}()
for (tag, file) in (("A", "data/run_A.jld2"), ("B", "data/run_B.jld2"))
    isfile(file) || (println("\n跳过 $file（不存在）"); continue)
    dat = load(file)
    n, p, q_star, ε = dat["n"], dat["p"], dat["q_star"], dat["eps"]
    κ = haskey(dat, "kappa") ? dat["kappa"] : 1.0
    δ = haskey(dat, "delta") ? dat["delta"] : 0.0
    κ̄ = κ                       # 初值取 κ̄=κ 使 B₀·n|∂Ω = O(ε²)
    b = dat["b"]

    ops = setup_operators((n, n, 1), (p, p, 1), torus_mapping(ε, κ, δ))
    _, pcoef, _, _, _ = force_velocity(ops, b)
    B0n = solovev_L2norm(ε, κ, δ, q_star; κ̄=κ̄)
    bn = sqrt(dot(b, ops.M2 * b))
    cosang = inner_with_analytic(ops, b, q_star; κ̄=κ̄) / (bn * B0n)
    shape_err = sqrt(max(2 - 2 * cosang, 0.0))    # ‖b̂ − B̂₀‖，两者均单位化

    fl = reconstruct_bhat(ops, b, ε, κ, δ)
    rg, tg = fl.rg, fl.tg
    Pph = scalar_from_3form(ops, pcoef, ε, κ, δ, rg, tg, fl.R, fl.W)
    ψh = fl.Az ./ (2π)                            # Âζ = 2πψ
    ψa = [solovev_ψ(poloidal_position(ε, κ, δ, r, t)...; κ̄=κ̄) for r in rg, t in tg]

    R_ax, z_ax, iax = find_axis(Pph, rg, tg, ε, κ, δ)
    # 壁面是否为磁面：r=1 上 ψh 的变化 / 整体变化幅度
    wall_var = (maximum(ψh[end, :]) - minimum(ψh[end, :])) /
               (maximum(ψh) - minimum(ψh))
    wall_var_a = (maximum(ψa[end, :]) - minimum(ψa[end, :])) /
                 (maximum(ψa) - minimum(ψa))

    println("\n--- run_$tag  (ε=$ε, κ=$κ, δ=$δ, n=$n, p=$p) ---")
    @printf("  形状差异 ‖B/‖B‖ − B₀/‖B₀‖‖        = %.4e   (夹角余弦 %.8f)\n",
        shape_err, cosang)
    @printf("  磁轴 R:  弛豫 %.5f   解析 Solov'ev %.5f   (Shafranov 位移 %.5f)\n",
        R_ax, 1.0, R_ax - 1.0)
    @printf("  壁面作为磁面 max-min(ψ)/range:  弛豫 %.3e   解析 %.3e\n",
        wall_var, wall_var_a)
    @printf("  最终力残差 %.4e,  ‖div B‖ %.3e\n", dat["force"][end], dat["divB"][end])

    results[tag] = (ops=ops, b=b, fl=fl, Pph=Pph, ψh=ψh, ψa=ψa,
        ε=ε, κ=κ, δ=δ, κ̄=κ̄, q_star=q_star, R_ax=R_ax, z_ax=z_ax, B0n=B0n, bn=bn)
end

############################################################################################
# 第 3 部分：p / ι 剖面（弛豫 vs 解析 Solov'ev）
############################################################################################
println("\n" * "="^78)
println("第 3 部分：p / ι 剖面")
println("="^78)

# 沿一条磁面追踪得到 ι；磁面标签用归一化极向磁通 s = (ψ−ψ_ax)/(ψ_wall−ψ_ax) ∈ [0,1]
function profile(Brm, Btm, Bzm, ψm, Pm, rg, tg, ε, κ, δ, R_ax, z_ax;
    seeds=range(0.06, 0.97; length=34), transits=300)
    Br_f, Bt_f, Bz_f = bilinear(rg, tg, Brm), bilinear(rg, tg, Btm), bilinear(rg, tg, Bzm)
    ψ_f = bilinear(rg, tg, ψm)
    P_f = Pm === nothing ? nothing : bilinear(rg, tg, Pm)
    RZ(r, t) = poloidal_position(ε, κ, δ, r, t)
    # ψ 在磁轴处取极值：取离壁面值更远的那个极值（对 ψ 的符号约定稳健）
    ψ_wall = sum(ψ_f(1.0, t) for t in tg) / length(tg)
    ψ_lo, ψ_hi = extrema(ψm)
    ψ_ax = abs(ψ_hi - ψ_wall) > abs(ψ_lo - ψ_wall) ? ψ_hi : ψ_lo
    s, ι, pv = Float64[], Float64[], Float64[]
    for r0 in seeds
        pts, iot = trace_line(Br_f, Bt_f, Bz_f, RZ, R_ax, z_ax, r0; transits=transits)
        (isempty(pts) || !isfinite(iot)) && continue
        # 场线中途跑出计算域 ⇒ ι 不可靠（解析 Solov'ev 场在壁面附近就是如此，
        # 因为壁面对它不是磁面）。只保留完整跑满 transits 圈的磁面。
        length(pts) == transits || continue
        mean_orbit_radius(pts, RZ, R_ax, z_ax) > 5e-3 || continue
        push!(s, (ψ_f(r0, 0.0) - ψ_ax) / (ψ_wall - ψ_ax))
        push!(ι, abs(iot))
        P_f === nothing || push!(pv, P_f(r0, 0.0))
    end
    return s, ι, pv, ψ_ax, ψ_wall
end

# 单调序列上的线性插值
function interp1(xs, ys, x)
    k = searchsortedfirst(xs, x)
    k <= 1 && return ys[1]
    k > length(xs) && return ys[end]
    w = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
    return (1 - w) * ys[k - 1] + w * ys[k]
end

function build_profiles(results)
    prof = Dict{String, Any}()
    for tag in ("A", "B")
        haskey(results, tag) || continue
        R = results[tag]
        rg, tg = R.fl.rg, R.fl.tg

        println("\nrun_$tag: 追踪弛豫场 ...")
        sh, ιh, ph, _, _ = profile(R.fl.Br, R.fl.Bt, R.fl.Bz, R.ψh, R.Pph, rg, tg,
            R.ε, R.κ, R.δ, R.R_ax, R.z_ax)
        println("  弛豫: $(length(sh)) 条磁面, ι ∈ $(round.(extrema(ιh), sigdigits=4))")

        println("  追踪解析 Solov'ev 场 ...")
        aBr, aBt, aBz = solovev_bhat_grid(R.ε, R.κ, R.δ, R.q_star, rg, tg; κ̄=R.κ̄)
        # 解析场的磁轴在 (R,z)=(1,0)（ψ 的极大点，无 Shafranov 位移）
        sa, ιa, _, ψax_a, ψwl_a = profile(aBr, aBt, aBz, R.ψa, nothing, rg, tg,
            R.ε, R.κ, R.δ, 1.0, 0.0)
        println("  解析: $(length(sa)) 条磁面, ι ∈ $(round.(extrema(ιa), sigdigits=4))")

        oh, oa = sortperm(sh), sortperm(sa)
        sh, ιh, ph = sh[oh], ιh[oh], ph[oh] .* R.B0n^2   # b 已单位化 ⇒ p 乘 ‖B₀‖²
        sa, ιa = sa[oa], ιa[oa]
        # 解析压强 p⁰ = (κ̄²+1)ψ；s 是归一化 ψ，故 p⁰ 在 s 上恰为直线
        pa = [(R.κ̄^2 + 1) * (ψax_a + s * (ψwl_a - ψax_a)) for s in sa]
        # 两者的 p 都有任意加性常数（Leray 乘子钉 ∫p=0；Solov'ev 定义如此），
        # 统一平移到最外侧共同磁面处为零后再比较
        s_ref = min(sh[end], sa[end])
        ph = ph .- interp1(sh, ph, s_ref)
        pa = pa .- interp1(sa, pa, s_ref)

        # ι 在共同 s 区间上的定量差异
        lo, hi = max(sh[1], sa[1]), min(sh[end], sa[end])
        grid = range(lo, hi; length=50)
        rel = [abs(interp1(sh, ιh, x) - interp1(sa, ιa, x)) / interp1(sa, ιa, x)
               for x in grid]
        @printf("  ι 相对差异 (s∈[%.2f,%.2f]): 平均 %.2f%%, 最大 %.2f%%\n",
            lo, hi, 100 * sum(rel) / length(rel), 100 * maximum(rel))
        dp = [abs(interp1(sh, ph, x) - interp1(sa, pa, x)) for x in grid]
        pscale = maximum(abs, pa)
        @printf("  p 相对差异 (按 |p| 峰值 %.3e 归一): 平均 %.2f%%, 最大 %.2f%%\n",
            pscale, 100 * sum(dp) / length(dp) / pscale, 100 * maximum(dp) / pscale)

        prof[tag] = (sh=sh, ιh=ιh, ph=ph, sa=sa, ιa=ιa, pa=pa)
    end
    return prof
end

prof = build_profiles(results)

############################################################################################
# 剖面图
############################################################################################
function draw_profiles(prof)
    fig = Figure(size=(1250, 820))
    titles = Dict("A" => "circular (eps=1/3)",
        "B" => "ITER D-shape (eps=.33, kappa=1.7, delta=.33)")
    row = 0
    for tag in ("A", "B")
        haskey(prof, tag) || continue
        row += 1
        P = prof[tag]
        ax = Axis(fig[row, 1]; xlabel="s = normalised poloidal flux", ylabel="iota",
            title="$(titles[tag]) - rotational transform")
        lines!(ax, P.sa, P.ιa; color=:gray40, linewidth=2, linestyle=:dash,
            label="analytic Solovev")
        scatterlines!(ax, P.sh, P.ιh; color=:crimson, markersize=6, label="relaxed")
        axislegend(ax; position=:lb)

        ax2 = Axis(fig[row, 2]; xlabel="s = normalised poloidal flux",
            ylabel="p - p(edge)", title="$(titles[tag]) - pressure")
        lines!(ax2, P.sa, P.pa; color=:gray40, linewidth=2, linestyle=:dash,
            label="analytic Solovev")
        scatterlines!(ax2, P.sh, P.ph; color=:royalblue, markersize=6, label="relaxed")
        axislegend(ax2; position=:rt)
    end
    return fig
end

out = "data/validate_profiles.png"
save(out, draw_profiles(prof))
println("\n已保存 $out")
