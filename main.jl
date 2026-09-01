# MRX 磁弛豫复现（Mantis.jl）：轴对称托卡马克平衡磁面
# 用法: julia --project=. main.jl parameter_A.jl
import Mantis
using LinearAlgebra, SparseArrays, StaticArrays, JLD2
using Mantis.Forms: d, ★, ∧, ∫

include("src/mantis_fixes.jl")
include("src/geometry.jl")
include("src/complex.jl")
include("src/operators.jl")
include("src/solovev.jl")
include("src/relaxation.jl")

length(ARGS) == 1 || error("用法: julia --project=. main.jl parameter_A.jl")

# 几何默认值（圆截面、轴对称）；参数文件可覆盖
geometry = :torus
aspect, elong, tri = 1 / 3, 1.0, 0.0    # ε 反环径比, κ 拉长比, δ 三角形变
nfp, nzeta = 3, 1                        # 环向场周期数, 环向单元数
kbar = nothing                           # Solov'ev 初值的 κ̄（默认见下）
include(joinpath(pwd(), ARGS[1]))        # 定义 n, p, q_star, dt0, N_iter [, 几何]

# 硬编码：η=0、γ=0
tag = replace(splitext(basename(ARGS[1]))[1], "parameter_" => "")
outfile = joinpath("data", "run_$tag.jld2")
mkpath("data")

if geometry === :stellarator
    nzeta > 1 || error("仿星器需要 nzeta > 1，否则退化为轴对称")
    mapping = stellarator_mapping(aspect, elong, nfp)
    geom_desc = "仿星器 (ε,κ,n_fp)=($aspect,$elong,$nfp)"
    κ̄ = kbar === nothing ? 1.0 : kbar          # 论文 V.F 取 κ̄=1.0
else
    mapping = torus_mapping(aspect, elong, tri)
    geom_desc = "环 (ε,κ,δ)=($aspect,$elong,$tri)"
    κ̄ = kbar === nothing ? elong : kbar        # κ̄=κ 使 B₀·n=O(ε²)（论文 V.D.b）
end
# n_ζ=1 时环向必须用 1 阶（单元周期样条 ⇒ 1 个常数自由度 = 轴对称）
nel, deg = (n, n, nzeta), (p, p, nzeta == 1 ? 1 : p)

println("复形装配: nel=$nel, deg=$deg, q*=$q_star, $geom_desc ...")
t_setup = @elapsed ops = setup_operators(nel, deg, mapping)
println("  自由度: V⁰₀..V³ = ", size.(ops.E, 1), "  ($t_setup s)")

b0 = solovev_initial_field(ops, q_star; κ̄=κ̄)
println("初始: ‖div B‖ = $(norm(ops.Kdiv * b0)), 环向通量 = $(toroidal_flux(ops, b0))")

t_relax = @elapsed b, hist = relax(ops, b0; dt0=dt0, N_iter=N_iter)
println("弛豫完成 ($t_relax s): 力残差 $(hist.force[1]) → $(hist.force[end])")

jldsave(outfile;
    b=b, b0=b0, n=n, p=p, q_star=q_star, dt0=dt0, N_iter=N_iter,
    eps=aspect, kappa=elong, delta=tri,
    geometry=String(geometry), nfp=nfp, nzeta=nzeta, kbar=κ̄,
    energy=hist.energy, force=hist.force, divB=hist.divB, dt=hist.dt,
    picard=hist.picard, helicity_iter=hist.helicity_iter,
    helicity=hist.helicity, flux=hist.flux,
)
println("数据已写入 $outfile")
