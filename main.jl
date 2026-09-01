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
include(joinpath(pwd(), ARGS[1]))   # 定义 n, p, q_star, dt0, N_iter

# 硬编码（跑通后再参数化）：圆截面 ε=1/3、n_ζ=1、η=0、γ=0
const ε = 1 / 3
tag = replace(splitext(basename(ARGS[1]))[1], "parameter_" => "")
outfile = joinpath("data", "run_$tag.jld2")
mkpath("data")

println("复形装配: nel=($n,$n,1), deg=($p,$p,1), q*=$q_star ...")
t_setup = @elapsed ops = setup_operators((n, n, 1), (p, p, 1), circular_torus_mapping(ε))
println("  自由度: V⁰₀..V³ = ", size.(ops.E, 1), "  ($t_setup s)")

b0 = solovev_initial_field(ops, q_star)
println("初始: ‖div B‖ = $(norm(ops.Kdiv * b0)), 环向通量 = $(toroidal_flux(ops, b0))")

t_relax = @elapsed b, hist = relax(ops, b0; dt0=dt0, N_iter=N_iter)
println("弛豫完成 ($t_relax s): 力残差 $(hist.force[1]) → $(hist.force[end])")

jldsave(outfile;
    b=b, b0=b0, n=n, p=p, q_star=q_star, dt0=dt0, N_iter=N_iter, eps=ε,
    energy=hist.energy, force=hist.force, divB=hist.divB, dt=hist.dt,
    picard=hist.picard, helicity_iter=hist.helicity_iter,
    helicity=hist.helicity, flux=hist.flux,
)
println("数据已写入 $outfile")
