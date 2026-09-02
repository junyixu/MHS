# MHS — 用 FEEC 复现 MRX 磁弛豫方法

用 Julia 的 [Mantis.jl](https://mantisfem.github.io/Mantis.jl/stable/)（B 样条 FEEC 有限元框架）复现
Blickhan, Stratton & Kaptanoglu (2025), *MRX: A differentiable 3D MHD equilibrium solver without
nested flux surfaces* ([arXiv:2510.26986](https://arxiv.org/abs/2510.26986)) 中的**磁弛豫方法**，求解托卡马克与仿星器的三维 MHD 平衡，并画出磁面结构。

三个算例：圆截面托卡马克、ITER D 形截面、三周期旋转椭圆仿星器。仿星器算例复现出论文
Fig. 12 的五岛链。

---

## 方法

平衡不是直接解 Grad–Shafranov，而是让一个初始场沿**理想**（$\eta = 0$）演化到能量极小：

$$
\frac{\partial B}{\partial t} = \nabla\times(u\times B),
\qquad
u = \mathcal{A}\,\bigl(J\times B - \nabla p\bigr),
\qquad
\mathcal{A} = \Pi^{\mathrm{Leray}} .
$$

理想演化让磁通与螺旋度成为运动不变量，磁能沿轨迹单调下降，不动点即
$J\times B = \nabla p$。压力不是输入，而是 Leray 投影的 Lagrange 乘子（$-p$）——
这正是 MRX 不需要嵌套磁面假设的原因：磁岛和随机层可以自然出现。

离散化取三维张量积样条 de Rham 复形，$\theta,\zeta$ 方向用周期样条：

$$
V_h^0 \xrightarrow{\ \operatorname{grad}\ } V_h^1
      \xrightarrow{\ \operatorname{curl}\ } V_h^2
      \xrightarrow{\ \operatorname{div}\ } V_h^3 ,
\qquad
\operatorname{curl}\circ\operatorname{grad} = 0,\quad
\operatorname{div}\circ\operatorname{curl} = 0 .
$$

$B$ 取为 2-形式，于是 $\nabla\cdot B = 0$ 在**系数层面**恒等成立（实测 $\lesssim 10^{-15}$，
且不随迭代增长），不需要任何散度清理或罚项。

几条实现上的选择：

- **时间步进**：显式最速下降 + 解析线搜索
  $\delta t = -\langle b,\dot b\rangle_{M_2}\big/\lVert\dot b\rVert^2_{M_2}$（能量沿下降方向的精确一维极小）；
  另备论文 Eq. (6) 的中点 Picard 格式。
- **磁轴 $r=0$ 奇性**：用最小相容轴约束（$V^0$ 轴环黏合、$V^1$ 的 $\theta$ 分量置零、
  $V^2$ 的 $[\theta\zeta]$ 分量置零）。这恰好封住轴上电流丝的环向通量泄漏——
  不加约束时弛豫会经由它卸载通量，造成 2000 步内 8% 的假能量下降。
- **性能**：弛豫过程中所有矩阵不变，稀疏分解只做一次，每步只做三角回代。
- **几何**：逻辑矩形 $(r,\theta,\zeta)\in[0,1]^3$ 经解析映射到物理环，不用三角网格。
  D 形截面取论文 Eq. (7)，仿星器取论文 Eq. (3)。

---

## 算例与结果

### 1. 圆截面托卡马克

$\varepsilon = 1/3$，$q^\ast = 1.57$，$(n_r,n_\theta) = (8,8)$，$p = 3$，5000 步。
左：压力着色；右：旋转变换 $\iota$ 着色。$\iota$ 从轴上 $0.78$ 单调降到边界 $0.55$。

[圆截面 Poincaré]
<img width="2500" height="1240" alt="run_A_poincare" src="https://github.com/user-attachments/assets/15630c91-8ae7-484f-9d78-0f61260feff3" />

### 2. ITER D 形截面

论文 V.D 参数：$\varepsilon = 0.33$、拉长比 $\kappa = 1.7$、三角形变 $\delta = 0.33$，$q^\ast = 1.57$。
磁面被拉长并向内挤出 D 形，$\iota$ 从 $0.56$ 降到 $0.33$。

[D 形 Poincaré]
<img width="2500" height="1240" alt="run_B_poincare" src="https://github.com/user-attachments/assets/e011894e-602e-4730-9557-e593411437b7" />

### 3. 三维旋转椭圆仿星器

论文 V.F 参数：$\varepsilon = 0.33$、$\kappa = 1.2$、$n_{fp} = 3$、$\bar\kappa = 1.0$、$q^\ast = 1.57$，
$(n_r,n_\theta,n_\zeta) = (12,12,6)$，$p = 3$，4000 步。

边界由论文 Eq. (3) 给出，是**随环向角旋转的椭圆**：

$$
R = 1 + r\,\varepsilon\,\nu(\zeta)\cos 2\pi\theta,
\qquad
Z = r\,\varepsilon\,\nu(\zeta + \tfrac{1}{2})\sin 2\pi\theta,
\qquad
\nu(\zeta) = 1 + (1-\kappa)\cos\bigl(2 n_{fp}\pi\zeta\bigr).
$$

$Z$ 里那个半周期偏移不能丢：$n_{fp}$ 为奇数时 $\nu(\zeta+\tfrac12) = 2 - \nu(\zeta)$，
两个半轴反相，截面才会随 $\zeta$ 转 $90^\circ$。写成 $a = b = \nu$ 得到的是没有螺旋成形的
波纹环，根本不产生真空旋转变换。

两个环向截面（$\zeta = 0$ 与 $\zeta = 1/6$）的 Poincaré 图。左图 $\iota = 3/5$ 处清晰可见
**五岛链**，$\iota = 3/4$ 处还有一条四岛链：

[仿星器 Poincaré]
<img width="2500" height="1280" alt="run_C_poincare" src="https://github.com/user-attachments/assets/2c21f6dd-b154-4f86-8fb6-a107eb86f073" />

诊断曲线——力残差、能量、$\lVert\nabla\cdot B\rVert$、守恒量漂移：

[仿星器诊断]
<img width="2200" height="1400" alt="run_C_diag" src="https://github.com/user-attachments/assets/d7285563-095b-4e6d-8762-ac498301685b" />



| 量 | 实测 |
| --- | --- |
| 力残差 $\lVert J\times B-\nabla p\rVert/\lVert\nabla p\rVert$ | $8.95\times10^{-3}\to8.53\times10^{-4}$（单调） |
| 磁能 | 单调下降，相对降幅 $1.22\times10^{-6}$ |
| $\lVert\nabla\cdot B\rVert$ | $\le 5.18\times10^{-16}$ |
| 环向通量漂移 | $2.41\times10^{-16}$ |
| 螺旋度漂移 | $1.01\times10^{-9}$（显式格式的 $O(\delta t)$ 误差） |
| $\iota$ 范围（$\zeta=0$） | $0.600$ – $0.797$ |
| $\iota$ 范围（$\zeta=1/6$） | $0.585$ – $0.795$ |

论文用同样的分辨率跑 $2.5\times10^4$ 步，力残差到 $7.4\times10^{-6}$；这里跑 4000 步（约 65 分钟），
$\iota$ 剖面与岛链结构已经收敛——它们比力残差收敛得快得多。要继续压残差就调大
`parameter_C.jl` 里的 `N_iter`。

### 定量验证

`validate.jl` 做网格收敛阶测试，并把弛豫结果与 Solov'ev 解析解（论文 Eq. 8）逐点比较：

[Solov'ev 对比]
<img width="2500" height="1640" alt="validate_profiles" src="https://github.com/user-attachments/assets/aaa7219a-b0d9-4e37-af59-ef89184db746" />


---

## 运行

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. main.jl parameter_A.jl     # 圆截面    → data/run_A.jld2
julia --project=. main.jl parameter_B.jl     # ITER D 形 → data/run_B.jld2
julia --project=. main.jl parameter_C.jl     # 3D 仿星器 → data/run_C.jld2

julia --project=. plot_.jl  data/run_A.jld2  # 轴对称算例画图
julia --project=. plot3d.jl data/run_C.jld2  # 仿星器画图（非轴对称）
julia --project=. validate.jl                # 收敛阶 + Solov'ev 定量对比
```

参数文件只暴露必要的几个量：`n`（径向/极向分辨率）、`p`（样条阶数）、
`q_star`（环向场强）、`dt0`（初始步长）、`N_iter`（迭代数），外加几何参数。

各阶段的验证脚本：`sanity.jl`（复形与映射）、`sanity2.jl`（Solov'ev 投影）、
`sanity3.jl`（算子层）、`sanity4.jl`（D 形几何）、`sanity5.jl`（Remark 29 螺旋度）、
`sanity6.jl`（仿星器几何与 $n_\zeta>1$ 复形）。

依赖 Julia 1.12、Mantis.jl、GLMakie、JLD2、StaticArrays；`Manifest.toml` 已锁版本。

---

## 代码结构

```
main.jl                     入口：装配 → Solov'ev 初值 → 弛豫 → 存 jld2
parameter_{A,B,C}.jl        三个算例的参数
plot_.jl / plot3d.jl        轴对称 / 非轴对称的 Poincaré 与诊断图
validate.jl                 收敛阶与 Solov'ev 对比
sanity*.jl                  各阶段验证脚本

src/mantis_fixes.jl         Mantis v0.6.0 三维 Wedge 1∧1 的热修复（必须最先 include）
src/geometry.jl             torus_mapping(ε,κ,δ)、stellarator_mapping(ε,κ,n_fp)
src/complex.jl              三维周期 de Rham 复形、矩阵与载荷装配
src/operators.jl            边界条件 / 轴约束的提取矩阵、缓存分解、Leray、谐和场、螺旋度
src/solovev.jl              Solov'ev 解析初值（论文 Eq. 8）
src/relaxation.jl           显式线搜索 / 中点 Picard，诊断量内联
src/fieldline.jl            场重构、磁面标签、场线追踪
```

分支按阶段推进，全部保留：`circular`（圆截面）→ `dshape`（D 形）→ `validation`（定量验证）
→ `stellarator`（三维仿星器），已合并进 `main`。

## 参考

- T. Blickhan, J. Stratton, A. A. Kaptanoglu,
  *MRX: A differentiable 3D MHD equilibrium solver without nested flux surfaces*,
  [arXiv:2510.26986](https://arxiv.org/abs/2510.26986) (2025).
  参考实现（Python/JAX）见论文对应的 `mrx` 包。
- Mantis.jl 文档：<https://mantisfem.github.io/Mantis.jl/stable/>
