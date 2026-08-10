# Learning to Solve Parametric Mixed-Integer Optimal Control Problems via Differentiable Predictive Control

Code to reproduce the simulation results in the paper *Learning to Solve Parametric Mixed-Integer Optimal Control Problems via Differentiable Predictive Control*.

**Paper:** [arXiv:2506.19646](https://arxiv.org/abs/2506.19646)

## Overview

We learn explicit neural control policies that approximate the solution of a parametric **mixed-integer optimal control problem (MI-OCP)** for a conceptual energy storage system with both continuous and integer actuators. Instead of solving a mixed-integer program (MIP) online at every step, a policy network maps the current state and disturbance preview directly to control actions, enabling fast inference with near-optimal performance.

Three differentiable rounding strategies are used to handle integer decisions during training, and an imitation-learning baseline is trained on MIP solver labels:

| Approach | Script | Integer handling |
|---|---|---|
| Baseline (optimal solution) | `_1_CPLEX.py` | Exact MIP (CPLEX) solved online |
| DPC — Sigmoid STE | `_2_sigmoid.py` | Sigmoid straight-through estimator |
| DPC — Gumbel-softmax | `_3_softmax.py` | Gumbel-softmax relaxation |
| DPC — Learnable threshold | `_4_learnable_threshold.py` | Learned rounding threshold |
| Imitation learning | `_6` + `_7` | Supervised on CPLEX labels |

All trained policies are compared against the optimal CPLEX solution on cost, mean inference time, and relative suboptimality margin.

## Results

**Closed-loop trajectories** — DPC policy (blue) vs. exact optimal MIP solution (red dashed): states track the setpoints while respecting bounds and minimizing actuaction cost.

![Closed-loop trajectories](plots/img/closed_loop.png)

**Phase portrait** — state-space trajectories from 20 initial conditions converging to the setpoint, DPC vs. optimal.

![Phase portrait](plots/img/phase_portrait.png)

## System

A linear time-invariant second-order thermal energy system sampled at $T = 300$ s, simulated over ~6.5 days (1872 steps). The discrete-time dynamics are

$$x_{k+1} = A\,x_k + B_\mathrm{u}\,u_k + B_\delta\,\delta_k + E\,d_k,$$

with

$$
A = \begin{bmatrix} \alpha_1 & \nu \\ 0 & \alpha_2 - \nu \end{bmatrix}, \quad
B_\mathrm{u} = \begin{bmatrix} b_1 & 0 \\ 0 & b_2 \end{bmatrix}, \quad
B_\delta = \begin{bmatrix} 0 \\ b_3 \end{bmatrix}, \quad
E = \begin{bmatrix} -b_4 & 0 \\ 0 & -b_5 \end{bmatrix},
$$

where $\alpha_1 = 0.9983$, $\alpha_2 = 0.9966$ are the storage dissipation rates, $\nu = 0.001$ the transfer rate from $x_2$ to $x_1$, $b_1 = b_2 = 0.075$ the heat-pump efficiencies, $b_3 = 0.0825$ the heating-rod efficiency, and $b_4 = b_5 = 0.0833$ the disturbance-coupling coefficients.

- **States** $x_1, \; x_2$ [kWh] — thermal storage levels, bounded to `[0, 8.4]` and `[0, 3.6]`
- **Inputs** $u_1, \; u_2$ [kW] continuous (heat pumps), $\delta_1 \in \{0, 1, 2, 3\}$ [kW] integer (active heating rods)
- **Disturbances** $d_1, \; d_2$ [kW] — known energy consumption
- **Setpoint** $r_1 = 4.2, \; r_2 = 1.8$ kWh
- **Horizon lengths** $N \in \{10, 15, 20, 25, 30, 35, 40\}$

## Requirements

- Python 3.10+
- PyTorch, NumPy, SciPy, Matplotlib, tqdm
- [Neuromancer](https://github.com/pnnl/neuromancer) 1.5.6 (policy training & closed-loop simulation)
- [CVXPY](https://www.cvxpy.org/) with a MIP solver — **IBM CPLEX** (used in the paper) or **Gurobi**
- LaTeX (`pdflatex`) for PGF figure export

## Workflow

Scripts are numbered in execution order; run them from the repository root.

1. **Optimal baselines** — solve the MI-OCP online with CPLEX for 20 initial conditions.
   `python _1_CPLEX.py -ic 0` (single) or `./run_1_CPLEX.sh` (all, in parallel) → `CPLEX_inference_data/`
   Note: running the script in parallel requires having more than 20 CPU threads available.

2. **Train DPC policies** — one script per rounding method, all horizons.
   `python _2_sigmoid.py`, `_3_softmax.py`, `_4_learnable_threshold.py` → `training_outputs/<method>/models/`

3. **Imitation learning** *(optional)* — generate MIP labels then train.
   `./run_7_imitation_learning_data_generation.sh` → `imitation_learning_data/`, then `python _7_imitation_learning_policy_synthesis.py`

4. **Evaluate** — roll out all policies and compute cost, mean inference time, and suboptimality vs. CPLEX.
   `python _5_test_models.py` (or `./run_5_test_models.sh`) → `simulation_data/`

5. **Plot** — regenerate paper figures.
   `python _8_plots.py` → `plots/` (`fig1v2.pdf`, `phase_plotv2_large.pdf`, plus `.pgf` for LaTeX)

`_0_generate_synthetic_disturbance_data.py` (optional) regenerates the synthetic training disturbances in `training_data/`.

## Repository layout

```
_0 … _8 *.py        # pipeline scripts (data → train → evaluate → plot)
run_*.sh            # parallel/background launchers
utils/              # rounding helpers, initial conditions, system preview
training_data/      # synthetic disturbance windows for training
loads_matrix.mat    # test disturbance trajectories for evaluation
CPLEX_inference_data/   # exact optimal rollouts
imitation_learning_data/  # MIP-labeled training data for imitation learning
training_outputs/   # saved DPC policies
simulation_data/    # closed-loop rollouts & metrics
plots/              # generated figures
```

## Notes

- CPLEX must be installed; online MIP runs can take up to ~2 hours per instance.
- Edit the `env=` interpreter path in the `run_*.sh` scripts to match your environment.
- Training uses CUDA when available; evaluation defaults to CPU.
- Different random seeds may yield slightly different numbers than reported in the paper.

## Citation

```bibtex
@article{boldocky2025learning,
  title={Learning to solve parametric mixed-integer optimal control problems via differentiable predictive control},
  author={Boldock{\'y}, J{\'a}n and Javan, Shahriar Dadras and Gulan, Martin and M{\"o}nnigmann, Martin and Drgo{\v{n}}a, J{\'a}n},
  journal={arXiv preprint arXiv:2506.19646},
  year={2025}
}
```
