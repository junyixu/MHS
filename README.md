# MHS — MRX magnetic relaxation with FEEC

A Julia reproduction of the **magnetic relaxation** method of Blickhan, Stratton & Kaptanoglu
(2025), *MRX: A differentiable 3D MHD equilibrium solver without nested flux surfaces*
([arXiv:2510.26986](https://arxiv.org/abs/2510.26986)), built on
[Mantis.jl](https://mantisfem.github.io/Mantis.jl/stable/), a B-spline FEEC finite element
framework. It computes three-dimensional MHD equilibria for tokamaks and stellarators and plots
the resulting flux-surface structure.

Three cases are included: a circular-cross-section tokamak, an ITER D-shaped cross-section, and a
three-field-period rotating-ellipse stellarator. The stellarator case reproduces the five-island
chain of Fig. 12 of the paper.

---

## Method

The equilibrium is not obtained by solving Grad–Shafranov directly. Instead, an initial field is
evolved under **ideal** ($\eta = 0$) dynamics towards an energy minimum:

$$
\frac{\partial B}{\partial t} = \nabla\times(u\times B),
\qquad
u = \mathcal{A}\,\bigl(J\times B - \nabla p\bigr),
\qquad
\mathcal{A} = \Pi^{\mathrm{Leray}} .
$$

Ideal evolution makes magnetic flux and helicity invariants of the motion, the magnetic energy
decreases monotonically along the trajectory, and the fixed point is exactly $J\times B = \nabla p$.
Pressure is not an input: it is the Lagrange multiplier ($-p$) of the Leray projection. That is
precisely why MRX needs no nested-flux-surface assumption — islands and stochastic layers are free
to appear.

The discretisation is a three-dimensional tensor-product spline de Rham complex, periodic in
$\theta$ and $\zeta$:

$$
V_h^0 \xrightarrow{\ \mathrm{grad}\ } V_h^1
      \xrightarrow{\ \mathrm{curl}\ } V_h^2
      \xrightarrow{\ \mathrm{div}\ } V_h^3 ,
\qquad
\mathrm{curl}\circ\mathrm{grad} = 0,\quad
\mathrm{div}\circ\mathrm{curl} = 0 .
$$

Taking $B$ as a 2-form makes $\nabla\cdot B = 0$ hold identically **at the coefficient level**
(measured at $\lesssim 10^{-15}$, with no growth over the iteration), so no divergence cleaning or
penalty term is required.

A few implementation choices worth naming:

- **Time stepping**: explicit steepest descent with an analytic line search,
  $\delta t = -\langle b,\dot b\rangle_{M_2}\big/\lVert\dot b\rVert^2_{M_2}$ — the exact
  one-dimensional minimiser of the energy along the descent direction. The midpoint Picard scheme
  of Eq. (6) of the paper is also available.
- **Axis singularity at $r=0$**: handled by the minimal consistent axis constraint ($V^0$ glued
  into a single ring on the axis, the $\theta$ component of $V^1$ set to zero, the $[\theta\zeta]$
  component of $V^2$ set to zero). This seals the toroidal-flux leak through the on-axis current
  filament; without it, relaxation drains flux through the axis and fakes an 8% energy drop within
  2000 steps.
- **Performance**: every matrix is constant throughout the relaxation, so the sparse factorisations
  are computed once and each step costs only a triangular solve.
- **Geometry**: the logical cube $(r,\theta,\zeta)\in[0,1]^3$ is mapped analytically onto the
  physical torus, with no triangular mesh. The D-shape follows Eq. (7) of the paper, the
  stellarator Eq. (3).

---

## Cases and results

### 1. Circular cross-section tokamak

$\varepsilon = 1/3$, $q^\ast = 1.57$, $(n_r,n_\theta) = (8,8)$, $p = 3$, 5000 steps.
Left: coloured by pressure; right: coloured by rotational transform $\iota$. The profile decreases
monotonically from $0.78$ on axis to $0.55$ at the boundary.

<img width="2500" height="1240" alt="Circular cross-section Poincaré plot" src="https://github.com/user-attachments/assets/15630c91-8ae7-484f-9d78-0f61260feff3" />

### 2. ITER D-shaped cross-section

Parameters from Sec. V.D of the paper: $\varepsilon = 0.33$, elongation $\kappa = 1.7$,
triangularity $\delta = 0.33$, $q^\ast = 1.57$. The flux surfaces are elongated and pushed inward
into a D-shape, and $\iota$ falls from $0.56$ to $0.33$.

<img width="2500" height="1240" alt="D-shaped Poincaré plot" src="https://github.com/user-attachments/assets/e011894e-602e-4730-9557-e593411437b7" />

### 3. Three-dimensional rotating-ellipse stellarator

Parameters from Sec. V.F of the paper: $\varepsilon = 0.33$, $\kappa = 1.2$, $n_{fp} = 3$,
$\bar\kappa = 1.0$, $q^\ast = 1.57$, with $(n_r,n_\theta,n_\zeta) = (12,12,6)$, $p = 3$ and
4000 steps.

The boundary is given by Eq. (3) of the paper — an **ellipse that rotates with the toroidal angle**:

$$
R = 1 + r\,\varepsilon\,\nu(\zeta)\cos 2\pi\theta,
\qquad
Z = r\,\varepsilon\,\nu(\zeta + \tfrac{1}{2})\sin 2\pi\theta,
\qquad
\nu(\zeta) = 1 + (1-\kappa)\cos\bigl(2 n_{fp}\pi\zeta\bigr).
$$

The half-period offset in $Z$ is essential: for odd $n_{fp}$ it gives
$\nu(\zeta+\tfrac12) = 2 - \nu(\zeta)$, so the two semi-axes are in antiphase and the cross-section
turns by $90^\circ$ with $\zeta$. Writing $a = b = \nu$ instead produces a corrugated torus with no
helical shaping, which generates no vacuum rotational transform at all.

Poincaré plots at two toroidal cuts ($\zeta = 0$ and $\zeta = 1/6$). The **five-island chain** at
$\iota = 3/5$ is clearly visible in the left panel, along with a four-island chain at $\iota = 3/4$:

<img width="2500" height="1280" alt="Stellarator Poincaré plot" src="https://github.com/user-attachments/assets/2c21f6dd-b154-4f86-8fb6-a107eb86f073" />

Diagnostics — force residual, energy, $\lVert\nabla\cdot B\rVert$, and drift of the invariants:

<img width="2200" height="1400" alt="Stellarator diagnostics" src="https://github.com/user-attachments/assets/d7285563-095b-4e6d-8762-ac498301685b" />

| Quantity | Measured |
| --- | --- |
| Force residual $\lVert J\times B-\nabla p\rVert/\lVert\nabla p\rVert$ | $8.95\times10^{-3}\to8.53\times10^{-4}$ (monotone) |
| Magnetic energy | monotonically decreasing, relative drop $1.22\times10^{-6}$ |
| $\lVert\nabla\cdot B\rVert$ | $\le 5.18\times10^{-16}$ |
| Toroidal flux drift | $2.41\times10^{-16}$ |
| Helicity drift | $1.01\times10^{-9}$ (the $O(\delta t)$ error of the explicit scheme) |
| $\iota$ range ($\zeta=0$) | $0.600$ – $0.797$ |
| $\iota$ range ($\zeta=1/6$) | $0.585$ – $0.795$ |

The paper runs $2.5\times10^4$ steps at the same resolution and reaches a force residual of
$7.4\times10^{-6}$. The 4000 steps here (about 65 minutes) are already enough for the $\iota$
profile and the island structure to converge — both converge much faster than the force residual.
Increase `N_iter` in `parameter_C.jl` to push the residual down further.

### Quantitative validation

`validate.jl` runs a grid convergence study and compares the relaxed solution pointwise against the
analytic Solov'ev equilibrium (Eq. 8 of the paper):

<img width="2500" height="1640" alt="Comparison against the analytic Solov'ev solution" src="https://github.com/user-attachments/assets/aaa7219a-b0d9-4e37-af59-ef89184db746" />

---

## Running

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. main.jl parameter_A.jl     # circular       → data/run_A.jld2
julia --project=. main.jl parameter_B.jl     # ITER D-shape   → data/run_B.jld2
julia --project=. main.jl parameter_C.jl     # 3D stellarator → data/run_C.jld2

julia --project=. plot_.jl  data/run_A.jld2  # plots for the axisymmetric cases
julia --project=. plot3d.jl data/run_C.jld2  # plots for the stellarator (non-axisymmetric)
julia --project=. validate.jl                # convergence order + Solov'ev comparison
```

The parameter files expose only what matters: `n` (radial/poloidal resolution), `p` (spline
degree), `q_star` (toroidal field strength), `dt0` (initial step size), `N_iter` (iteration count),
plus the geometry parameters.

Stage-by-stage verification scripts: `sanity.jl` (complex and mapping), `sanity2.jl` (Solov'ev
projection), `sanity3.jl` (operator layer), `sanity4.jl` (D-shaped geometry), `sanity5.jl`
(Remark 29 helicity), `sanity6.jl` (stellarator geometry and the $n_\zeta>1$ complex).

Requires Julia 1.12, Mantis.jl, GLMakie, JLD2 and StaticArrays; versions are pinned in
`Manifest.toml`.

---

## Repository layout

```
main.jl                     entry point: assembly → Solov'ev initial data → relaxation → jld2
parameter_{A,B,C}.jl        parameters for the three cases
plot_.jl / plot3d.jl        Poincaré and diagnostic plots, axisymmetric / non-axisymmetric
validate.jl                 convergence order and Solov'ev comparison
sanity*.jl                  stage-by-stage verification scripts

src/mantis_fixes.jl         hotfix for the 3D Wedge 1∧1 in Mantis v0.6.0 (must be included first)
src/geometry.jl             torus_mapping(ε,κ,δ), stellarator_mapping(ε,κ,n_fp)
src/complex.jl              3D periodic de Rham complex, matrix and load assembly
src/operators.jl            extraction matrices for BCs / axis constraints, cached factorisations,
                            Leray projection, harmonic fields, helicity
src/solovev.jl              analytic Solov'ev initial data (Eq. 8 of the paper)
src/relaxation.jl           explicit line search / midpoint Picard, with inline diagnostics
src/fieldline.jl            field reconstruction, flux-surface labels, field-line tracing
```

## References

- T. Blickhan, J. Stratton, A. A. Kaptanoglu,
  *MRX: A differentiable 3D MHD equilibrium solver without nested flux surfaces*,
  [arXiv:2510.26986](https://arxiv.org/abs/2510.26986) (2025).
  The reference implementation (Python/JAX) is the `mrx` package accompanying the paper.
- Mantis.jl documentation: <https://mantisfem.github.io/Mantis.jl/stable/>
