# Learning to Solve Parametric Mixed-Integer Optimal Control Problems via Differentiable Predictive Control

Code to reproduce the simulation results in the paper *Learning to Solve Parametric Mixed-Integer Optimal Control Problems via Differentiable Predictive Control*.

**Paper:** [arXiv:2506.19646](https://arxiv.org/abs/2506.19646)

## Overview

We learn explicit neural control policies that approximate the solution of a parametric **mixed-integer optimal control problem (MI-OCP)** for an energy storage system with both continuous and integer actuators. Instead of solving a mixed-integer program (MIP) online at every step, a policy network maps the current state and disturbance preview directly to control actions, enabling fast inference with near-optimal performance.

Three differentiable rounding strategies are used to handle integer decisions during training, and an imitation-learning baseline is trained on MIP solver labels:

| Approach | Script | Integer handling |
|---|---|---|
| DPC — Sigmoid STE | `_2_sigmoid.py` | Sigmoid straight-through estimator |
| DPC — Gumbel-softmax | `_3_softmax.py` | Gumbel-softmax relaxation |
| DPC — Learnable threshold | `_4_learnable_threshold.py` | Learned rounding threshold |
| Imitation learning | `_6` + `_7` | Supervised on CPLEX labels |
| Optimal baseline | `_1_CPLEX.py` | Exact MIP (CPLEX) solved online |

All trained policies are compared against the exact CPLEX solution on cost, mean inference time, and relative suboptimality.

## Results

**Closed-loop trajectories** — DPC policy (blue) vs. exact optimal MIP solution (red dashed): states track their setpoints while respecting bounds, and the integer input `δ1` is reproduced faithfully.

![Closed-loop trajectories](plots/img/closed_loop.png)

**Phase portrait** — state-space trajectories from 20 initial conditions converging to the setpoint, DPC vs. optimal.

![Phase portrait](plots/img/phase_portrait.png)

## System

A two-state storage model sampled at 5-minute intervals, simulated over ~6.5 days (1873 steps).

- **States** `x1, x2` [kWh] — storage levels, bounded to `[0, 8.4]` and `[0, 3.6]`
- **Inputs** `u1, u2` [kW] continuous, $\delta \in \{0, 1, 2, 3\}$ [kW] integer
- **Disturbances** `d1, d2` [kW] — exogenous loads
- **Setpoint** `(4.2, 1.8)` kWh
- **Horizons** `N ∈ {10, 15, 20, 25, 30, 35, 40}`

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
imitation_learning_data/  # MIP-labeled training data
training_outputs/   # saved policy checkpoints
simulation_data/    # closed-loop rollouts & metrics
plots/              # generated figures
```

## Notes

- CPLEX (or Gurobi) must be installed and licensed; online MIP runs can take up to ~2 hours per instance.
- Edit the `env=` interpreter path in the `run_*.sh` scripts to match your environment.
- Training uses CUDA when available; evaluation defaults to CPU.
- Different random seeds may yield slightly different numbers than reported in the paper.

## Citation

```bibtex
@article{mi-dpc-2025,
  title   = {Learning to Solve Parametric Mixed-Integer Optimal Control Problems via Differentiable Predictive Control},
  journal = {arXiv preprint arXiv:2506.19646},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.19646}
}
```
