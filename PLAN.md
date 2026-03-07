# DREAMPlace + BeyondPPA + MPC — Complete Implementation Reference

> **Who this is for:**
> Any developer, researcher, or AI agent picking up this project cold.
> Read cover-to-cover and you will understand the full problem, the design, every
> file's role, the mathematical formulation, how to build the system from scratch,
> and how to run both operational modes.
> **No additional context is required.**

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [The Problem Being Solved](#2-the-problem-being-solved)
3. [Two Operational Modes (and nothing in between)](#3-two-operational-modes-and-nothing-in-between)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Architecture Overview](#5-architecture-overview)
6. [Repository Structure](#6-repository-structure)
7. [Build & Installation (macOS Apple Silicon)](#7-build--installation-macos-apple-silicon)
8. [Running Experiments](#8-running-experiments)
9. [Implementation Reference — New Files](#9-implementation-reference--new-files)
10. [Implementation Reference — Modified Files](#10-implementation-reference--modified-files)
11. [Build System Details](#11-build-system-details)
12. [Parameters Reference](#12-parameters-reference)
13. [Data Flow Walkthrough](#13-data-flow-walkthrough)
14. [Expected Output & Verification](#14-expected-output--verification)
15. [Design Decisions & Trade-offs](#15-design-decisions--trade-offs)
16. [Known Limitations & Future Work](#16-known-limitations--future-work)

---

## 1. What This Project Is

**DREAMPlace** is an open-source, GPU-accelerated VLSI placement tool built on
PyTorch. It treats chip placement as a differentiable optimisation problem and
uses gradient descent (specifically Nesterov acceleration) to minimise a
placement objective.

This project adds the **BeyondPPA + MPC** integration on top of DREAMPlace:

- **BeyondPPA** (MLCAD 2025 paper): four differentiable reliability-penalty terms
  that discourage placement patterns known to cause post-route manufacturing and
  reliability defects.
- **MPC** (Model Predictive Control): an online receding-horizon controller that
  adaptively schedules the penalty weights *during* placement, so the solver
  balances reliability penalties against wirelength and density automatically.

This is **research code** — the primary benchmark is ISPD 2005 `adaptec1`.

---

## 2. The Problem Being Solved

### DREAMPlace's default objective

```
J = Σ_net HPWL(net)  +  λ · Σ_bin DensityPenalty(bin)
```

This minimises half-perimeter wirelength (HPWL) while spreading cells to avoid
density overflow. It produces legal, compact placements — but it leaves four
classes of **post-route reliability defects** unaddressed:

| # | Defect | Post-Route Consequence |
|---|--------|----------------------|
| 1 | Uneven macro density | IR-drop hotspots, localised thermal stress |
| 2 | Macros near IO ports | Signal integrity violations, noise coupling |
| 3 | Unaligned macros | Clock-tree synthesis difficulty, clock skew |
| 4 | Narrow gaps between macros (notches) | EM risk, routing congestion |

### The BeyondPPA fix

Add four differentiable penalty terms — one per defect class — to the objective.
Each term provides gradients that push macros toward safer positions *while*
gradient descent is running, so no separate post-processing step is needed.

### The MPC problem

Naively fixing the penalty weights leads to either:
- weights too small → penalties ignored, defects persist
- weights too large → wirelength degrades, overflow stalls

MPC solves this by treating the placement loop as a dynamical system and
adaptively tuning all penalty weights (and the density weight) at runtime to
track a target overflow trajectory while suppressing reliability metrics.

---

## 3. Two Operational Modes (and nothing in between)

The project deliberately exposes exactly **two modes**:

| Mode | Config flag | What runs |
|------|------------|-----------|
| **Baseline** | `beyond_ppa_flag=0` | Stock DREAMPlace (HPWL + density) |
| **BeyondPPA + MPC** | `beyond_ppa_flag=1`, `mpc_flag=1` | DREAMPlace + all 4 reliability penalties + adaptive MPC weight scheduling |

There is **no intermediate** "BeyondPPA without MPC" config file.
Setting `mpc_flag=1` while `beyond_ppa_flag=0` is a **configuration error** and
prints a warning:

```
WARNING: mpc_flag=1 has no effect when beyond_ppa_flag=0.
         Set beyond_ppa_flag=1 to enable BeyondPPA+MPC reliability mode.
```

---

## 4. Mathematical Formulation

### 4.1 Global placement objective

```
J(pos) = HPWL(pos)
       + λ_density · DensityPenalty(pos)
       + ramp(t) · [
           λ_d · MacroDensityUniformity(pos)   -- IR-drop
         + λ_io · IOKeepout(pos)                -- noise coupling
         + λ_a · MacroGridAlign(pos)            -- clock-tree
         + λ_n · MacroNotch(pos)                -- EM / congestion
         ]
```

where `pos` is the flat tensor `[x_0..x_N, y_0..y_N]` for all physical nodes.

### 4.2 Penalty term definitions

**MacroDensityUniformity** — variance of macro area density over a coarse grid:

```
grid[b] = Σ_i  bilinear_weight(macro_i, bin_b) × area_i / bin_area
Loss     = Var(grid)
```

Bilinear scatter makes the gradient smooth (no hard bin boundaries).

**IOKeepout** — ReLU² penalty for macros inside the keepout radius:

```
Loss = Σ_{i ∈ macros, j ∈ IO_ports}  ReLU(d_keepout − ‖center_i − pos_j‖)²
```

**MacroGridAlign** — periodic sin² potential pulling macros toward grid anchors:

```
Loss = Σ_{i ∈ macros}  [ sin²(π·x_i / pitch_x) + sin²(π·y_i / pitch_y) ]
```

Energy is exactly 0 at grid multiples, maximal halfway between.

**MacroNotch** — ReLU² penalty for narrow inter-macro gaps:

```
dx_gap(i,j) = max(0,  |cx_i − cx_j| − (w_i + w_j)/2 )
dy_gap(i,j) = max(0,  |cy_i − cy_j| − (h_i + h_j)/2 )
d_ij        = dx_gap + dy_gap    (Manhattan edge-to-edge gap)
Loss        = Σ_{i < j}  ReLU(d_thresh − d_ij)²
```

### 4.3 Overflow gating and linear ramp

BeyondPPA terms are **inactive** until:
```
current_overflow  <  beyond_ppa_gate_overflow  (default 0.20)
```

After activation, a linear ramp prevents a sudden objective shock:
```
ramp(t) = min(1,  (t − t_activate) / ramp_iters)
```

### 4.4 Log-space weight representation

All feature weights are stored in log-space:
```
log_w = [log(λ_d), log(λ_io), log(λ_a), log(λ_n)]
actual weight λ_i = exp(log_w[i])
```

MPC emits log-space increments `Δlog(λ)`. Weight update:
```
log_w[i]  +=  EMA_ALPHA × clip(Δlog_w[i], −MAX_DELTA, +MAX_DELTA)

EMA_ALPHA = 0.20   (80% history, 20% new signal)
MAX_DELTA = 0.095  (≈ ±10% change per call in natural-log scale)
```

### 4.5 MPC formulation

**State** `s ∈ ℝ⁶`:
```
s = [overflow,  Δhpwl_norm,  density_raw,  io_raw,  align_raw,  notch_raw]
```

**Control** `u ∈ ℝ⁵` (all log-space increments):
```
u = [Δlog(λ_density_weight),  Δlog(λ_d),  Δlog(λ_io),  Δlog(λ_a),  Δlog(λ_n)]
```

**Linear dynamics** (fitted online by ridge regression):
```
s_{t+1} ≈ A · s_t  +  B · u_t  +  c
```

**Receding-horizon cost** over H steps:
```
min_{u_0..u_{H-1}}  Σ_{k=0}^{H-1}  [ (s_k − s_ref)ᵀ Q (s_k − s_ref)
                                      +  u_kᵀ R u_k ]
```

Reference: `s_ref = [stop_overflow, 0, 0, 0, 0, 0]`
Solved by `scipy.optimize.minimize` with SLSQP and box constraints.

**Density-weight anchor** (prevents runaway):
```
dw_log_offset  =  clip(dw_log_offset + Δlog(λ_dw),  −log(10),  +log(10))
density_weight  =  density_weight_base × exp(dw_log_offset)
```

The base is recorded once at placement start; the offset is bounded to ±10×.

---

## 5. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│  DREAMPlace Global Placement Loop  (NonLinearPlace.py)                │
│                                                                       │
│  ┌─ Lgamma loop (gamma reduction) ───────────────────────────────┐   │
│  │  ┌─ Llambda loop (density weight update) ──────────────────┐  │   │
│  │  │  ┌─ Lsub loop (Nesterov gradient step) ──────────────┐  │  │   │
│  │  │  │                                                    │  │  │   │
│  │  │  │  pos  ──►  PlaceObj.obj_fn()  ──►  loss           │  │  │   │
│  │  │  │                 │                                  │  │  │   │
│  │  │  │          ┌──────┴──────┐                           │  │  │   │
│  │  │  │          │             │                           │  │  │   │
│  │  │  │     HPWL + λ·Density  BeyondPPAObj.forward()      │  │  │   │
│  │  │  │                       (4 ops, gated+ramped)        │  │  │   │
│  │  │  │                                                    │  │  │   │
│  │  │  └────────────────────────────────────────────────────┘  │  │   │
│  │  │                                                           │  │   │
│  │  │  Every mpc_interval Llambda steps:                        │  │   │
│  │  │    MPCController.record() → .fit() → .step()             │  │   │
│  │  │    → update density_weight + update BeyondPPA log_w      │  │   │
│  │  │                                                           │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

**Placement data layout** (`pos` tensor):
```
pos = [x_0 .. x_{N_mov-1}, x_{N_mov} .. x_{N_phys-1}, | y_0 .. y_{N_phys-1}]
       ← movable nodes →     ← fixed terminals →     ← y section, same layout →

N_mov  = num_movable_nodes   (cells + macros that the optimizer moves)
N_phys = num_physical_nodes  (N_mov + fixed IO terminals; excludes fillers)
```

**Macro detection** (DREAMPlace PlaceDB heuristic, not new code):
```python
movable_macro_mask[i] = True
    if node_area[i] > 10 × mean_cell_area
   AND node_size_y[i] > 2 × row_height
```

---

## 6. Repository Structure

```
DREAMPlace/
├── run_mac.sh                    ← ENTRY POINT for all build/run operations
├── Dockerfile.mac                ← ARM64 Ubuntu 22.04 container spec
├── PLAN.md                       ← THIS FILE
│
├── dreamplace/
│   ├── Placer.py                 ← top-level entry: loads params, calls NonLinearPlace
│   ├── Params.py                 ← parameter loader (reads params.json schema)
│   ├── params.json               ← parameter schema + defaults (17 new params added)
│   ├── BasicPlace.py             ← base class: data loading, legalization, I/O
│   ├── NonLinearPlace.py         ← MODIFIED: three-loop optimizer + MPC integration
│   ├── PlaceObj.py               ← MODIFIED: objective function (adds BeyondPPA)
│   ├── EvalMetrics.py            ← MODIFIED: adds bppa_* metric fields + logging
│   ├── BeyondPPAObj.py           ← NEW: combined reliability objective module
│   ├── MPCController.py          ← NEW: online MPC weight scheduler
│   │
│   └── ops/
│       ├── CMakeLists.txt        ← MODIFIED: registers 4 new subdirectories
│       ├── macro_density/
│       │   ├── macro_density.py  ← NEW: MacroDensityUniformityOp
│       │   └── CMakeLists.txt    ← NEW: pure-Python install (no C++ compilation)
│       ├── io_keepout/
│       │   ├── io_keepout.py     ← NEW: IOKeepoutOp
│       │   └── CMakeLists.txt    ← NEW
│       ├── macro_align/
│       │   ├── macro_align.py    ← NEW: MacroGridAlignOp
│       │   └── CMakeLists.txt    ← NEW
│       └── macro_notch/
│           ├── macro_notch.py    ← NEW: MacroNotchOp
│           └── CMakeLists.txt    ← NEW
│
└── test/
    └── ispd2005/
        ├── adaptec1.json         ← baseline mode config (existing)
        └── adaptec1_mpc.json     ← BeyondPPA+MPC mode config (NEW)
```

---

## 7. Build & Installation (macOS Apple Silicon)

All build operations go through `./run_mac.sh`. **Never run cmake or make
directly** — the script handles Docker, submodule checks, ABI detection, and
GPU-override automatically.

### Prerequisites

- **Docker Desktop** for Mac with Apple Silicon support (≥4.x recommended)
- Allocate **≥8 GB RAM** to Docker Desktop in Settings → Resources → Memory
- A clone of this repo with git history (not a ZIP download):
  ```bash
  git clone --recursive https://github.com/<your-fork>/DREAMPlace.git
  ```

### First-time setup (three commands)

```bash
# 1. Build the Docker image  (~10-20 min, downloads Ubuntu 22.04 + PyTorch)
./run_mac.sh build-image

# 2. Compile and install DREAMPlace  (~10-30 min)
#    Uses MAKE_JOBS=2 by default to avoid OOM.
#    Override: DREAMPLACE_MAKE_JOBS=4 ./run_mac.sh install
./run_mac.sh install

# 3. Download benchmark data  (~5-30 min, several GB)
./run_mac.sh download-benchmarks
```

After these three commands, `install/` contains the fully installed DREAMPlace
and benchmark data in `install/benchmarks/`.

### What `build-image` does

```
docker build --platform linux/arm64 -f Dockerfile.mac -t dreamplace-mac:latest .
```

The Dockerfile (`Dockerfile.mac`) installs:

| Package | Source | Notes |
|---------|--------|-------|
| Ubuntu 22.04 (ARM64) | Docker Hub | Base image |
| GCC 11, CMake 3.22 | `apt` | Build toolchain |
| Bison 3.8, Flex, Boost 1.74 | `apt` | Parser / library deps |
| Python 3.10 + pip | `apt` | Runtime |
| PyTorch CPU-only (linux_aarch64) | PyTorch WHL index | `torch.cuda.is_available() == False` |
| SciPy ≥ 1.1.0 | pip | Required for MPC SLSQP solver |
| NumPy ≥ 1.15.4, matplotlib, cairocffi | pip | DREAMPlace runtime deps |
| torch_optimizer 0.3.0, ncg_optimizer 0.2.2 | pip | Additional optimizers |

CPU-only PyTorch causes CMake to detect `TORCH_ENABLE_CUDA=0` automatically,
skipping all CUDA kernel compilation.

### What `install` does (inside Docker)

```bash
# Step 1: Initialize git submodules (pybind11, Limbo, OpenTimer, …)
git submodule update --init --recursive --jobs 4

# Step 2: Detect PyTorch C++ ABI (critical — mismatch causes linker errors)
TORCH_CXX_ABI=$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')

# Step 3: CMake configure
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/DREAMPlace/install \
  -DCMAKE_CXX_ABI=${TORCH_CXX_ABI} \
  -DENABLE_HETEROSTA=OFF \
  -DPython_EXECUTABLE=$(which python3)

# Step 4: Compile
make -j${MAKE_JOBS}      # MAKE_JOBS defaults to 2; override with DREAMPLACE_MAKE_JOBS

# Step 5: Install
make install             # copies all Python files + compiled .so to install/
```

The `install/` directory contains a fully self-contained copy of DREAMPlace
(Python modules + compiled C++ extensions + benchmark test configs).

### What `make install` does for the 4 new ops

The new ops are **pure Python** — no C++ compilation occurs. Each op's
`CMakeLists.txt` uses:
```cmake
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
```
This copies `macro_density.py`, `io_keepout.py`, etc. into
`install/dreamplace/ops/<op_name>/`.

### Reinstall (keep image, rebuild DREAMPlace)

```bash
./run_mac.sh reinstall
```

This wipes `build/` and `install/`, then runs `install` again.

### Full reset

```bash
./run_mac.sh nuke
./run_mac.sh build-image
./run_mac.sh install
```

### Status check

```bash
./run_mac.sh status
```

Prints: Docker image state, build/ existence, install/ + Placer.py existence,
submodule status.

### Interactive debug shell

```bash
./run_mac.sh shell
# Opens bash inside the container at /DREAMPlace
# Useful for: python3 -c "import dreamplace; ...", inspecting logs, etc.
```

---

## 8. Running Experiments

Both modes are run via `./run_mac.sh run <config.json>`. The script:
1. Checks the Docker image exists (auto-builds if not)
2. Checks `install/` exists
3. Creates a temporary copy of the config with `gpu` forced to `0`
4. Runs `python3 dreamplace/Placer.py <tmp_config>` from inside `install/`
5. Deletes the temporary config

The GPU override is transparent — you never need to manually edit JSON files.

### Mode 1: Baseline DREAMPlace

```bash
./run_mac.sh run test/ispd2005/adaptec1.json
```

Config: `test/ispd2005/adaptec1.json` (existing, unmodified stock file)
What runs: HPWL + density penalty only. BeyondPPA and MPC are disabled
(`beyond_ppa_flag` and `mpc_flag` default to 0).

### Mode 2: BeyondPPA + MPC

```bash
./run_mac.sh run test/ispd2005/adaptec1_mpc.json
```

Config: `test/ispd2005/adaptec1_mpc.json`

```json
{
  "aux_input": "benchmarks/ispd2005/adaptec1/adaptec1.aux",
  "gpu": 1,
  "num_bins_x": 512,
  "num_bins_y": 512,
  "global_place_stages": [
    {
      "num_bins_x": 512,
      "num_bins_y": 512,
      "iteration": 1000,
      "learning_rate": 0.01,
      "wirelength": "weighted_average",
      "optimizer": "nesterov",
      "Llambda_density_weight_iteration": 1,
      "Lsub_iteration": 1
    }
  ],
  "target_density": 1.0,
  "density_weight": 8e-5,
  "gamma": 4.0,
  "random_seed": 1000,
  "scale_factor": 1.0,
  "ignore_net_degree": 100,
  "enable_fillers": 1,
  "gp_noise_ratio": 0.025,
  "global_place_flag": 1,
  "legalize_flag": 1,
  "detailed_place_flag": 1,
  "detailed_place_engine": "",
  "detailed_place_command": "",
  "stop_overflow": 0.07,
  "dtype": "float32",
  "plot_flag": 0,
  "random_center_init_flag": 1,
  "gift_init_flag": 0,
  "sort_nets_by_degree": 0,
  "num_threads": 8,
  "deterministic_flag": 1,
  "beyond_ppa_flag": 1,
  "beyond_ppa_gate_overflow": 0.20,
  "mpc_flag": 1,
  "mpc_interval": 50
}
```

Note: `gpu=1` in the file is automatically overridden to `gpu=0` by `run_mac.sh`.
All BeyondPPA and MPC parameters not listed use their `params.json` defaults
(see Section 12).

---

## 9. Implementation Reference — New Files

### 9.1 `dreamplace/ops/macro_density/macro_density.py`

**Class:** `MacroDensityUniformityOp(nn.Module)`

**Purpose:** Measure how unevenly macros are distributed across the chip.
Penalise clustering (high variance → IR-drop hotspots).

**Constructor parameters:**
```python
MacroDensityUniformityOp(
    macro_mask,          # BoolTensor [N_movable] — True for macro nodes
    macro_size_x,        # float tensor [N_macro] — widths
    macro_size_y,        # float tensor [N_macro] — heights
    num_movable_nodes,   # int
    num_physical_nodes,  # int — used to index y-section of pos tensor
    num_bins_x,          # int — grid resolution (default 32)
    num_bins_y,          # int — grid resolution (default 32)
    xl, xh, yl, yh       # float — layout bounding box
)
```

**`forward(pos)` logic:**
1. Extract macro x, y from pos (using macro_mask and correct index ranges)
2. Compute fractional bin coordinates `fx, fy`
3. Compute per-macro area weight (capped at 1.0 per bin)
4. Compute four bilinear corner weights `w00, w10, w01, w11`
5. Scatter-add weights into a flat `[num_bins_x * num_bins_y]` density vector
   using out-of-place `scatter_add` (safe for autograd, no in-place mutation)
6. Return `density.var()` — the variance is the cost

**Key implementation note:** Uses `scatter_add` (not `scatter_add_`) to keep the
graph differentiable through the scatter operation.

---

### 9.2 `dreamplace/ops/io_keepout/io_keepout.py`

**Class:** `IOKeepoutOp(nn.Module)`

**Purpose:** Penalise macro centres that are closer than `keepout_dist` to any
fixed IO terminal. Prevents noise coupling and signal integrity violations.

**Constructor parameters:**
```python
IOKeepoutOp(
    macro_mask,          # BoolTensor [N_movable]
    io_x, io_y,          # float tensors — fixed IO terminal positions
    keepout_dist,        # float — minimum allowed distance (layout units)
    num_movable_nodes,   # int
    num_physical_nodes   # int
)
```

IO positions stored as `[N_io, 2]` buffer; never updated (terminals are fixed).

**`forward(pos)` logic:**
1. Extract macro centres (x, y) from pos
2. Stack into `[N_macro, 2]` tensor
3. Compute pairwise L2 distances: `torch.cdist(macro_centers, io_pos)` → `[N_macro, N_io]`
4. Compute violations: `viol = ReLU(keepout_dist − dist)`
5. Return `viol.pow(2).sum()`

**Complexity:** O(N_macro × N_IO) — small for ISPD benchmarks.

---

### 9.3 `dreamplace/ops/macro_align/macro_align.py`

**Class:** `MacroGridAlignOp(nn.Module)`

**Purpose:** Guide macros toward a virtual placement grid using a smooth periodic
potential. Aligned macros improve clock-tree synthesis regularity.

**Constructor parameters:**
```python
MacroGridAlignOp(
    macro_mask,          # BoolTensor [N_movable]
    pitch_x, pitch_y,    # float — grid pitch in layout units (= N × row_height)
    num_movable_nodes,   # int
    num_physical_nodes   # int
)
```

**`forward(pos)` logic:**
```python
loss_x = torch.sin(π * x / pitch_x).pow(2)
loss_y = torch.sin(π * y / pitch_y).pow(2)
return (loss_x + loss_y).sum()
```

Energy profile: 0 at grid multiples, 1.0 at half-pitch, smooth everywhere.
Gradients point toward the nearest grid line in both axes independently.

---

### 9.4 `dreamplace/ops/macro_notch/macro_notch.py`

**Class:** `MacroNotchOp(nn.Module)`

**Purpose:** Penalise narrow (notch) gaps between adjacent macro pairs.
Narrow gaps concentrate routing demand, raise EM risk, and disrupt the power
grid.

**Constructor parameters:**
```python
MacroNotchOp(
    macro_mask,          # BoolTensor [N_movable]
    macro_size_x,        # float tensor [N_macro]
    macro_size_y,        # float tensor [N_macro]
    notch_thresh,        # float — minimum allowed edge-to-edge gap (layout units)
    num_movable_nodes,   # int
    num_physical_nodes,  # int
    prune=False          # bool — enable cheap spatial pre-filter (see §15)
)
```

**`forward(pos)` logic:**
1. Extract macro centres (x, y)
2. Compute N×N pairwise centre differences (absolute values)
3. Compute half-sum of widths/heights
4. Compute Manhattan edge-to-edge gap:
   ```python
   dx_gap = relu(|cx_i - cx_j| - (w_i + w_j)/2)
   dy_gap = relu(|cy_i - cy_j| - (h_i + h_j)/2)
   d_ij = dx_gap + dy_gap
   ```
5. Optional prune: if `self.prune`, add large constant to d_ij for pairs with
   L2 centre distance > 2×notch_thresh (suppresses their penalty)
6. Apply upper-triangle mask (avoid double-counting and self-pairs)
7. Return `relu(notch_thresh - d_ij[ut_mask]).pow(2).sum()`

**Complexity:** O(N_macro²) — dense, but ISPD benchmarks have < 200 macros, so
this is fast in practice.

**Prune warning:** `prune=True` uses L2 centre distance as filter but d_ij uses
Manhattan gaps — the filter may incorrectly include/exclude corner pairs.
Disabled by default until a correctness-compatible filter is validated.

---

### 9.5 `dreamplace/BeyondPPAObj.py`

**Class:** `BeyondPPAObj(nn.Module)`

**Purpose:** Wrapper that owns all four ops, applies overflow gating, linear
ramp, and log-space weight management. This is the single object PlaceObj
interacts with.

**Constants:**
```python
FEAT_NAMES = ('density', 'io', 'align', 'notch')
EMA_ALPHA  = 0.20    # in update_weights_log()
MAX_DELTA  = 0.095   # in update_weights_log()
```

**Constructor:** Takes `(params, placedb, data_collections)`.
- Reads `beyond_ppa_gate_overflow` (default 0.3), `beyond_ppa_ramp_iters` (default 50)
- Identifies macros using `placedb.movable_macro_mask`
- Extracts IO terminal positions from `placedb.node_x/y[num_movable:]`
- Converts geometry params to layout units (multiplies by `row_height`)
- Instantiates all four ops
- Initialises `_log_w` from `params.beyond_ppa_weights` (default [1,1,1,1])

**Key methods:**

`check_and_enable(overflow)` — call once per Llambda iteration:
```python
if not self.enabled and overflow < self.gate_overflow:
    self.enabled = True
    self._ramp_step = 0
```

`_ramp_scale()` — returns linear ramp 0→1 over `ramp_iters` steps after enable.

`update_weights_log(delta_log)` — called by MPC after each planning step:
```python
for i, d in enumerate(delta_log):      # delta_log is u_opt[1:5]
    d_clipped = clip(d, -MAX_DELTA, MAX_DELTA)
    self._log_w[i] += EMA_ALPHA * d_clipped
```

`forward(pos)` — always computes raw costs (even when disabled, for MPC
observation); returns `(total_cost, raw_dict)`:
```python
# When disabled: total_cost is a zero tensor with no grad; raw costs tracked
# When enabled:  total_cost = ramp × Σ_i  λ_i × raw_cost_i
```

The `total_cost` scalar is added to the placement objective in `PlaceObj.obj_fn`.
The `raw_dict` is stored in `PlaceObj._bppa_raw` and propagated to EvalMetrics
and the MPC state vector.

---

### 9.6 `dreamplace/MPCController.py`

**Class:** `MPCController`

**Purpose:** Online receding-horizon controller. Every `mpc_interval` Llambda
iterations: records one transition → refits linear dynamics → runs SLSQP →
applies optimal control.

**Constants:**
```python
STATE_DIM    = 6
CONTROL_DIM  = 5
MAX_DELTA    = 0.095
MIN_HISTORY  = 15    # transitions needed before first fit
MIN_EXCITATION = 1e-6  # min control std to trust the fitted model
```

**Constructor** (`params`):
- `H` = `mpc_horizon` (default 5) — planning steps ahead
- `interval` = `mpc_interval` (default 50) — Llambda iters between MPC calls
- `hist_sz` = `mpc_history_size` (default 200) — rolling buffer size
- `ridge` = `mpc_ridge` (default 1e-3) — ridge regression coefficient
- `Q` = diagonal from `mpc_q_weights` (default [10, 1, 2, 3, 3, 3])
- `R` = diagonal from `mpc_r_weights` (default [0.1, 0.1, 0.1, 0.1, 0.1])
- `s_ref = [stop_overflow, 0, 0, 0, 0, 0]` — target reference state
- `_dw_log_offset_max = log(10)` — density weight bounded to ±10× base

**`record(s_t, u_t, s_next)`** — appends `(s_t, u_t, s_next)` to rolling buffer.
Pops oldest when buffer exceeds `hist_sz`.

**`fit()`** — ridge regression on current buffer:
```python
X = [s_t | u_t | 1]  for each recorded transition   # [n, S+C+1]
Y = [s_next]                                          # [n, S]
XtX += ridge × I
theta = solve(XtX, X.T @ Y)   # [S+C+1, S]
A = theta[:S, :].T             # [S, S]
B = theta[S:S+C, :].T          # [S, C]
c = theta[-1, :]               # [S]
```
Only updates model if `n >= MIN_HISTORY`. Measures excitation as mean std of
recent control signals.

**`_normalize(s)`** — fixed scale factors for each state dimension:
```python
scales = [1.0, 0.05, 1.0, 1e4, 1e2, 1e4]
s_norm = s / (scales + 1e-12)
```

**`step(current_state, current_u)`** — SLSQP optimisation:
```python
# If model not ready: return zeros (no change)
# Normalize state and s_ref
# Unroll H steps under linear dynamics
# Minimize Σ (s_k - s_ref_n)ᵀQ(…) + u_kᵀR u_k
# Box constraints: u ∈ [-MAX_DELTA, MAX_DELTA]^{H×C}
# Return first H control actions clipped to [-MAX_DELTA, MAX_DELTA]
```

**`apply_density_weight_delta(base, delta_log)`**:
```python
self._dw_log_offset = clip(self._dw_log_offset + delta_log,
                           -log(10), +log(10))
return base * exp(self._dw_log_offset)
```

---

## 10. Implementation Reference — Modified Files

### 10.1 `dreamplace/PlaceObj.py`

**What changed:** BeyondPPA objective instantiation and integration into `obj_fn`.

**In `__init__`** (lines 276–280):
```python
# BeyondPPA reliability objective (single instance, owned here)
self.beyond_ppa_obj = None
if getattr(params, 'beyond_ppa_flag', 0):
    from dreamplace.BeyondPPAObj import BeyondPPAObj
    self.beyond_ppa_obj = BeyondPPAObj(params, placedb, data_collections)
```

**In `obj_fn(pos)`** (lines 326–331):
```python
# BeyondPPA human-inspired reliability terms (gated by overflow)
if self.beyond_ppa_obj is not None:
    bppa_cost, self._bppa_raw = self.beyond_ppa_obj(pos)
    result = result + bppa_cost
else:
    self._bppa_raw = None
```

`result` already contains `HPWL + λ·density`; the BeyondPPA cost is added on
top. When BeyondPPA is disabled (`beyond_ppa_obj is None`) or not yet activated
(gated), `bppa_cost` is a zero tensor and does not affect gradients.

`self._bppa_raw` is the raw breakdown dict — persisted across the forward pass
so NonLinearPlace can pass it to EvalMetrics and the MPC state vector.

---

### 10.2 `dreamplace/NonLinearPlace.py`

**What changed:** MPC controller lifecycle, BeyondPPA gate integration, and
mpc_flag safety warning.

**MPC initialisation** (lines 125–140, inside `cmd_global_place`, once per stage):
```python
# Anchor density_weight for MPC
_density_weight_base = float(params.density_weight)

_mpc = None
_mpc_prev_state = None
_mpc_current_u  = None
if getattr(params, 'mpc_flag', 0) and getattr(params, 'beyond_ppa_flag', 0):
    import scipy.optimize           # validate import early
    from dreamplace.MPCController import MPCController
    _mpc = MPCController(params)
    _mpc_current_u = [0.0] * MPCController.CONTROL_DIM
elif getattr(params, 'mpc_flag', 0) and not getattr(params, 'beyond_ppa_flag', 0):
    logging.warning(
        "mpc_flag=1 has no effect when beyond_ppa_flag=0. "
        "Set beyond_ppa_flag=1 to enable BeyondPPA+MPC reliability mode.")
```

**BeyondPPA gate + MPC update** (lines 729–797, inside Llambda loop after density
weight update):

```python
# ── BeyondPPA gate + MPC weight update ───────────────
cur_llambda_metric = Llambda_metrics[-1][-1]
cur_overflow_scalar = float(cur_llambda_metric.overflow.mean() ...)

# Gate: activate BeyondPPA features when overflow is low enough
if model.beyond_ppa_obj is not None:
    model.beyond_ppa_obj.check_and_enable(cur_overflow_scalar)

# MPC: re-plan every mpc_interval Llambda iterations
if (_mpc is not None
        and model.beyond_ppa_obj is not None
        and Llambda_flat_iteration % _mpc.interval == 0):

    # Compute Δhpwl_norm from last two metrics
    delta_hpwl_norm = (h_cur - h_prev) / max(h_prev, 1.0)

    # Build state from raw BeyondPPA metrics
    current_state = [
        cur_overflow_scalar,
        delta_hpwl_norm,
        raw.get('density', 0.0),
        raw.get('io',      0.0),
        raw.get('align',   0.0),
        raw.get('notch',   0.0),
    ]

    if _mpc_prev_state is not None:
        _mpc.record(_mpc_prev_state, _mpc_current_u, current_state)
        _mpc.fit()

    _mpc_prev_state = current_state
    u_opt = _mpc.step(current_state, _mpc_current_u)
    _mpc_current_u = u_opt.tolist()

    # Apply density-weight delta (anchored)
    new_dw = _mpc.apply_density_weight_delta(
        _density_weight_base, float(u_opt[0]))
    with torch.no_grad():
        model.density_weight.fill_(new_dw)

    # Apply feature weight deltas in log-space
    if model.beyond_ppa_obj is not None:
        model.beyond_ppa_obj.update_weights_log(u_opt[1:5])

    logging.info("MPC step iter=%d  u=%s  dw=%.3E  bppa_w=%s" % (...))
```

**Key timing:** The MPC block runs *after* the Llambda density-weight update but
*before* the Llambda stop criterion check. This ensures MPC's density weight
override is used for the next Lsub sub-loop.

---

### 10.3 `dreamplace/EvalMetrics.py`

**What changed:** Added four BeyondPPA metric fields and their logging/evaluation.

**In `__init__`** (lines 39–43):
```python
# BeyondPPA reliability metrics (raw, unweighted)
self.bppa_density = None
self.bppa_io      = None
self.bppa_align   = None
self.bppa_notch   = None
```

**In `__str__`** (lines 94–97):
```python
if self.bppa_density is not None:
    content += ", BPPA[dens=%.3E io=%.3E align=%.3E notch=%.3E]" % (
        self.bppa_density, self.bppa_io,
        self.bppa_align,   self.bppa_notch)
```

**In `evaluate()`** (lines 150–157):
```python
if "beyond_ppa" in ops:
    _, raw = ops["beyond_ppa"](var)
    self.bppa_density = float(raw['density'])
    self.bppa_io      = float(raw['io'])
    self.bppa_align   = float(raw['align'])
    self.bppa_notch   = float(raw['notch'])
```

---

### 10.4 `dreamplace/params.json`

**What changed:** 17 new parameter entries added at the end of the JSON object
(lines 309–376).

Full list with defaults: see Section 12.

---

### 10.5 `dreamplace/ops/CMakeLists.txt`

**What changed:** Four `add_subdirectory` calls added at the end (lines 39–43):

```cmake
# BeyondPPA reliability feature ops (pure Python, no C++ compilation)
add_subdirectory(io_keepout)
add_subdirectory(macro_align)
add_subdirectory(macro_notch)
add_subdirectory(macro_density)
```

Each new op's `CMakeLists.txt` is identical in structure:
```cmake
set(OP_NAME macro_density)   # (or io_keepout, macro_align, macro_notch)

# Pure Python op — no C++ compilation needed.
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
```

---

## 11. Build System Details

### How DREAMPlace's CMake works

The top-level `CMakeLists.txt`:
1. Detects PyTorch installation via `find_package(Torch)`
2. Checks `torch.cuda.is_available()` — sets `TORCH_ENABLE_CUDA` accordingly
3. For each op subdirectory that has C++ sources: compiles a pybind11 extension
   module (`.so`) using `add_custom_command`/`add_custom_target`
4. `make install` copies all Python files and compiled `.so` files to
   `CMAKE_INSTALL_PREFIX/dreamplace/ops/<op_name>/`

### Why the new ops require no C++

All four new ops are implemented in pure Python (PyTorch `nn.Module`). PyTorch's
autograd differentiates through `scatter_add`, `cdist`, `sin`, `relu`, and
standard tensor ops natively. No C++/CUDA kernels are needed.

This means:
- No pybind11 binding code
- No `.cu` CUDA files
- No ABI compatibility issues
- `make install` simply copies `.py` files

### ABI detection (important for Apple Silicon)

```bash
TORCH_CXX_ABI=$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
cmake .. -DCMAKE_CXX_ABI=${TORCH_CXX_ABI} ...
```

PyTorch 2.x on ARM64 Ubuntu 22.04 uses the new C++11 ABI (`_GLIBCXX_USE_CXX11_ABI=1`).
The original DREAMPlace CMakeLists defaults to ABI=0 (old ABI). Without this
auto-detect step, all C++ extension modules produce undefined-symbol linker errors.
`run_mac.sh install` handles this automatically.

### Submodule initialisation

DREAMPlace uses git submodules for: `pybind11`, `Limbo` (parser), `OpenTimer`,
and others. The script checks for `thirdparty/pybind11/CMakeLists.txt` on both
the host and inside Docker to detect missing submodule content reliably.

---

## 12. Parameters Reference

All parameters belong to the `dreamplace/params.json` schema.
DREAMPlace's `Params.py` reads this file and sets attributes on the `params`
object. All new parameters are accessed via `getattr(params, 'name', default)`
to be backwards-compatible with configs that don't mention them.

### BeyondPPA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beyond_ppa_flag` | int (0/1) | `0` | Master enable for BeyondPPA reliability features |
| `beyond_ppa_weights` | list[float] | `[1.0, 1.0, 1.0, 1.0]` | Initial weights [density, io, align, notch] |
| `beyond_ppa_gate_overflow` | float | `0.3` | BeyondPPA activates when `overflow < this` |
| `beyond_ppa_ramp_iters` | int | `50` | Gradient steps to ramp from 0→1 after activation |
| `beyond_ppa_macro_bins` | int | `32` | Coarse grid size (N×N) for density measurement |
| `beyond_ppa_prune_notch` | int (0/1) | `0` | Enable notch pair pre-filter (disabled by default) |
| `io_keepout_distance` | float | `10` | Keepout radius from IO ports, in units of `row_height` |
| `grid_alignment_pitch_x` | float | `8` | Grid pitch X, in units of `row_height` |
| `grid_alignment_pitch_y` | float | `8` | Grid pitch Y, in units of `row_height` |
| `notch_threshold` | float | `5` | Min allowed edge-to-edge gap, in units of `row_height` |

### MPC Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mpc_flag` | int (0/1) | `0` | Enable MPC (requires `beyond_ppa_flag=1`) |
| `mpc_interval` | int | `50` | Llambda iterations between MPC planning steps |
| `mpc_horizon` | int | `5` | MPC prediction horizon H |
| `mpc_history_size` | int | `200` | Max transitions in rolling dynamics buffer |
| `mpc_ridge` | float | `1e-3` | Ridge regression regularisation coefficient |
| `mpc_q_weights` | list[float] | `[10, 1, 2, 3, 3, 3]` | State cost Q diagonal [overflow, Δhpwl, density, io, align, notch] |
| `mpc_r_weights` | list[float] | `[0.1, 0.1, 0.1, 0.1, 0.1]` | Control cost R diagonal [dw, density, io, align, notch] |

### Cost weight interpretation

The Q weights in `mpc_q_weights` penalise deviations of each state component
from its reference value. Higher Q → MPC prioritises reducing that component.
The default weighting (`overflow=10, io/align/notch=3`) says:
- Overflow convergence is the primary objective
- Reliability metrics have moderate cost
- HPWL change has low cost (optimiser already handles wirelength)

The R weights in `mpc_r_weights` penalise large control actions.
All default to 0.1 (moderate cost), preventing aggressive weight swings.

---

## 13. Data Flow Walkthrough

This section traces what happens at runtime when both flags are enabled.

### Startup

```
Placer.py
  → loads params from adaptec1_mpc.json
  → instantiates NonLinearPlace
      → NonLinearPlace.__init__
          → BasicPlace.__init__ (loads PlaceDB, creates data tensors)
```

### Global placement stage entry

```
NonLinearPlace.cmd_global_place()
  → constructs PlaceObj (PlaceObj.__init__)
      → if beyond_ppa_flag: BeyondPPAObj(params, placedb, data_collections)
          → identifies macro_mask from placedb.movable_macro_mask
          → creates MacroDensityUniformityOp, IOKeepoutOp,
                    MacroGridAlignOp, MacroNotchOp
  → records _density_weight_base = params.density_weight
  → if mpc_flag and beyond_ppa_flag:
      → creates MPCController(params)
      → _mpc_current_u = [0.0] * 5
```

### Each Lgamma step

Gamma (smoothing parameter) is reduced. PlaceObj is rebuilt with updated gamma.

### Each Llambda step

```
Llambda iteration:
  → Lsub loop: multiple Nesterov gradient steps
      → each step calls PlaceObj.obj_fn(pos):
          → HPWL = wirelength_op(pos)
          → Density = density_op(pos)
          → if beyond_ppa_obj:
              → bppa_cost, _bppa_raw = beyond_ppa_obj.forward(pos)
                  → density_op(pos), io_op(pos), align_op(pos), notch_op(pos)
                  → if enabled: return ramp × Σ λ_i × raw_i
                  → if not enabled: return 0 (but raw still computed)
              → result += bppa_cost
          → backward() → update pos via Nesterov

  → EvalMetrics.evaluate(pos)
      → overflow_op → overflow, max_density
      → if "beyond_ppa" in ops: → bppa_density, bppa_io, bppa_align, bppa_notch

  → density_weight update (DREAMPlace default logic)

  → BeyondPPA gate check:
      → beyond_ppa_obj.check_and_enable(overflow_scalar)
      → if newly enabled: logs "BeyondPPAObj: activated (overflow=…)"

  → MPC step (if Llambda_flat_iteration % mpc_interval == 0):
      → build current_state from overflow, Δhpwl, bppa_*
      → _mpc.record(prev_state, prev_u, current_state)
      → _mpc.fit()  (ridge regression, skips if < 15 samples)
      → u_opt = _mpc.step(current_state, current_u)
          → if not fitted or low excitation: return zeros
          → else: SLSQP over H steps → first action
      → new_dw = _mpc.apply_density_weight_delta(base, u_opt[0])
      → model.density_weight.fill_(new_dw)
      → beyond_ppa_obj.update_weights_log(u_opt[1:5])
      → logs: "MPC step iter=… u=[…] dw=… bppa_w=[…]"
```

### After global placement

Legalization and detailed placement run as normal (DREAMPlace default).
BeyondPPA and MPC have no role after global placement — they are global
placement tools only.

---

## 14. Expected Output & Verification

### Baseline run (`adaptec1.json`)

```
[INFO]  iter  100, (  0,  0,  0), Obj 1.234E+09, WL 4.56E+08, Density 8.90E+08, ...
[INFO]  iter  200, ...  Overflow 1.23E-01 ...
...
[INFO]  HPWL = 2.34E+07
[INFO]  Placed in X.XX seconds
```

No BeyondPPA or MPC lines will appear.

### BeyondPPA+MPC run (`adaptec1_mpc.json`)

**BeyondPPA activation** (happens when overflow drops below 0.20):
```
[INFO]  BeyondPPAObj: 42 macro nodes identified out of 210 movable
[INFO]  BeyondPPAObj: initial weights density=1.000 io=1.000 align=1.000 notch=1.000
...
[INFO]  BeyondPPAObj: activated (overflow=0.1950 < gate=0.2000)
```

**Per-iteration metrics** (includes BPPA breakdown after activation):
```
[INFO]  iter  350, (  0,  5,  0), Obj 2.34E+09, ..., Overflow 1.89E-01, ...
         BPPA[dens=3.45E-03 io=1.23E+04 align=5.67E+02 notch=8.90E+03], time 1.234ms
```

**MPC planning steps** (every 50 Llambda iterations after activation):
```
[INFO]  MPC step iter=402  u=[0.012, -0.031, 0.045, 0.023, -0.067]
         dw=8.123E-05  bppa_w=[1.034, 0.970, 1.046, 0.935]
```

**What to verify:**
1. BeyondPPA activates (the "activated" log line appears)
2. BPPA raw metrics trend downward over time after activation
3. MPC planning log lines appear every ~50 iterations after activation
4. MPC density weight `dw` varies around the base value (8e-5)
5. BeyondPPA feature weights `bppa_w` vary but stay near 1.0 (EMA + clipping)
6. Final HPWL is not dramatically worse than baseline (< 5-10% degradation is
   acceptable; reliability improvements cost some wirelength)
7. No crashes, no `NaN` in objective values

---

## 15. Design Decisions & Trade-offs

| Decision | Choice Made | Alternative Considered | Reason |
|----------|------------|----------------------|--------|
| Op implementation | Pure Python (PyTorch) | C++/CUDA kernels | ISPD macro counts (< 200) make Python fast enough; no compilation overhead |
| Weight representation | Log-space (`_log_w`) | Linear weights | Log-space ensures weights stay positive; increments are proportionally bounded |
| EMA smoothing | EMA_ALPHA=0.20 | Direct update | Prevents MPC from making oscillatory weight swings |
| MPC dynamics model | Online linear (ridge regression) | Offline neural model | No pretraining data available; ridge gives stable solution with small history |
| MPC solver | SLSQP (scipy) | QP solver, LQR | SLSQP handles box constraints natively; no extra dependencies |
| Density-weight anchoring | log-offset from base, bounded ±log(10) | Multiplicative | Prevents unbounded compounding across many intervals |
| BeyondPPA gating | Overflow threshold + linear ramp | Immediate activation | Prevents objective shock during early (high overflow) placement phase |
| Notch metric | Manhattan edge-to-edge gap | L2 gap | Manhattan aligns with row-based layout routing; cheaper to compute |
| Prune flag | Disabled by default | Enabled | L2 centre-distance filter ≠ Manhattan gap filter; corner cases incorrect |
| Two modes only | baseline vs full BeyondPPA+MPC | Three modes (+ BeyondPPA without MPC) | Cleaner interface; MPC is cheap; no reason to run BeyondPPA without adaptive tuning |

---

## 16. Known Limitations & Future Work

### Current limitations

1. **CPU-only on Apple Silicon**: The Docker image uses CPU-only PyTorch. The 4 new
   Python ops work on GPU too (no device-specific code), but the build system is
   configured for CPU. To use GPU, rebuild with a CUDA-enabled PyTorch wheel.

2. **Notch pruning bug**: `beyond_ppa_prune_notch=1` uses L2 centre distance to
   filter pairs but the penalty uses Manhattan gaps. Corner pairs can be wrongly
   included or excluded. Keep `beyond_ppa_prune_notch=0` (default) until fixed.

3. **Linear dynamics assumption**: MPC assumes `s_{t+1} ≈ A·s_t + B·u_t + c`.
   The true placement dynamics are nonlinear. The model is re-fitted online but
   may be a poor predictor during rapid overflow change phases.

4. **No rule-based fallback**: If MPC returns zeros (model not fitted or low
   excitation), BeyondPPA weights remain at their initial values for that
   interval. A heuristic fallback (e.g., increase weights proportional to metric
   magnitude) would improve early-phase reliability.

5. **Fixed normalisation scales in `_normalize`**: The MPCController normalises
   states with hardcoded scales `[1.0, 0.05, 1.0, 1e4, 1e2, 1e4]`. These are
   design-time estimates for ISPD 2005-class benchmarks and may be wrong for
   other designs.

6. **MPC not active before overflow gate**: Even if `mpc_flag=1`, MPC does not
   schedule weights until BeyondPPA is activated. During the early overflow-
   reduction phase, density weight follows DREAMPlace's default schedule only.

### Future work

- **Benchmark on ISPD 2015, ICCAD 2019**: Measure HPWL delta, overflow delta,
  and post-route reliability metrics vs. baseline.
- **Validated notch pruning**: Implement a correct spatial filter using cell-
  bbox overlap check instead of L2 centre distance.
- **Nonlinear dynamics**: Replace linear model with a small neural network fitted
  online (e.g., single hidden layer, low sample complexity).
- **GPU support**: Test with a CUDA-enabled Docker build.
- **Timing-aware feature**: Add a 5th BeyondPPA feature for critical-path macro
  proximity (the paper's Feature 1 is HPWL, already in DREAMPlace; Feature 5
  would be timing-driven).
- **Hyperparameter sweep**: Q, R, mpc_horizon, mpc_interval are all manually
  tuned for adaptec1. An automated sweep on the full ISPD suite would improve
  generalisability.

---

*End of implementation reference.*
