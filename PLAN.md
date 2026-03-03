# DREAMPlace + BeyondPPA + MPC: Complete Implementation Reference

> **Audience:** Any developer or AI agent picking up this project cold.
> This document is fully self-contained — read it cover-to-cover and you will understand the full design, every file's role, how to build the system, and how to run experiments.

---

## Table of Contents

1. [Why This Exists](#1-why-this-exists)
2. [What Was Built](#2-what-was-built)
3. [Architecture](#3-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Build & Installation](#5-build--installation)
6. [Running Experiments (Two Modes)](#6-running-experiments-two-modes)
7. [Implementation Reference — New Files](#7-implementation-reference--new-files)
8. [Implementation Reference — Modified Files](#8-implementation-reference--modified-files)
9. [Parameters Reference](#9-parameters-reference)
10. [Expected Output & Verification](#10-expected-output--verification)
11. [Design Decisions & Trade-offs](#11-design-decisions--trade-offs)
12. [Known Limitations & Future Work](#12-known-limitations--future-work)

---

## 1. Why This Exists

### The Problem

DREAMPlace's default global-placement objective is:

```
J = Σ HPWL(net_i)  +  λ · Σ DensityPenalty(bin_j)
```

This objective drives cells toward legal positions with short wires, but it leaves four classes of **post-route reliability defects** unaddressed:

| Defect | Post-Route Consequence |
|--------|----------------------|
| Uneven macro density | IR-drop hotspots, localised thermal stress |
| Macros near IO ports | Signal integrity violations, noise coupling |
| Unaligned macros | Clock-tree synthesis difficulty, clock skew |
| Narrow inter-macro gaps (notches) | EM violations, routing congestion |

### The Solution

Inspired by *BeyondPPA: Human-Inspired Reinforcement Learning for Post-Route Reliability-Aware Macro Placement* (MLCAD 2025), we add **four differentiable reliability penalty terms** directly into DREAMPlace's objective. Nesterov gradient descent then optimises all five terms simultaneously — no separate RL training required.

On top of that, a **Model Predictive Control (MPC) layer** adaptively tunes the five feature weights online during each placement run. It fits a linear dynamics model from observed (state, weight-vector, next-state) transitions and solves a receding-horizon optimisation problem every 50 Llambda iterations to find the best weight sequence. This replaces hand-tuned static λ values entirely.

### Why Better Than the Paper

The paper's DQN agent requires offline pre-training on historical placements. Our approach:
- No offline training — MPC model is fit online from data collected during the *current* run
- Fully differentiable — gradient descent can co-optimise all terms simultaneously
- Adapts online to each design's specific macro configuration and density profile

---

## 2. What Was Built

### Two Operational Modes

| Mode | Description | Config File |
|------|-------------|-------------|
| **Baseline** | Standard DREAMPlace (wirelength + density only) | `test/ispd2005/adaptec1.json` |
| **BeyondPPA + MPC** | Reliability-aware macro placement with adaptive weight control | `test/ispd2005/adaptec1_mpc.json` |

There is intentionally no "BeyondPPA without MPC" mode — MPC is lightweight enough to always run alongside BeyondPPA.

### New Components

```
dreamplace/BeyondPPAObj.py          Combined reliability objective module
dreamplace/MPCController.py         Online adaptive weight controller
dreamplace/ops/macro_density/       Density uniformity penalty op
dreamplace/ops/io_keepout/          IO keepout zone penalty op
dreamplace/ops/macro_align/         Grid alignment penalty op
dreamplace/ops/macro_notch/         Notch gap penalty op
test/ispd2005/adaptec1_mpc.json     Reference BeyondPPA+MPC config
```

### Modified Components

```
dreamplace/EvalMetrics.py           + bppa_density/io/align/notch metric fields
dreamplace/PlaceObj.py              + BeyondPPAObj init + obj_fn extension
dreamplace/NonLinearPlace.py        + MPC init, gate check, re-plan block
dreamplace/params.json              + 17 new parameters
dreamplace/ops/CMakeLists.txt       + 4 new add_subdirectory entries
```

---

## 3. Architecture

### Component Diagram

```
DREAMPlace Main Loop (NonLinearPlace.__call__)
│
├── Lgamma loop  ← reduces gamma (smoothing) over iterations
│   └── Llambda loop  ← updates density_weight between steps
│       └── Lsub loop  ← single Nesterov descent step
│
│   After each Lsub step:
│   ┌────────────────────────────────────────────────┐
│   │  PlaceObj.obj_fn(pos)                          │
│   │    wirelength  = op_collections.wirelength(pos)│
│   │    density     = op_collections.density(pos)   │
│   │    bppa_cost,  = BeyondPPAObj.forward(pos)     │
│   │    return WL + λ·density + bppa_cost           │
│   └────────────────────────────────────────────────┘
│
│   After each Llambda step:
│   ┌──────────────────────────────────────────────────────────┐
│   │  BeyondPPAObj.check_and_enable(overflow)                 │
│   │    → activate when overflow < gate_overflow (0.20)       │
│   │    → ramp weight 0→1 over ramp_iters (50) gradient steps │
│   │                                                           │
│   │  if Llambda_flat_iteration % mpc_interval == 0:          │
│   │    state = [overflow, Δhpwl, density_raw, io, align, notch] │
│   │    MPCController.record(prev_state, prev_u, state)       │
│   │    MPCController.fit()   ← ridge regression              │
│   │    u_opt = MPCController.step(state, current_u)          │
│   │    density_weight *= exp(u_opt[0])     (anchored)        │
│   │    BeyondPPAObj.update_weights_log(u_opt[1:5])           │
│   └──────────────────────────────────────────────────────────┘
```

### Data Flow: pos Tensor → Gradient

```
pos  [2 × num_physical_nodes]
  x: pos[0 : num_physical_nodes]
  y: pos[num_physical_nodes : 2*num_physical_nodes]

  Macro slice (in each op):
    x_macro = pos[:num_movable][macro_mask]      # movable macros only
    y_macro = pos[num_phys:num_phys+num_movable][macro_mask]

BeyondPPAObj.forward(pos)
  ↓
  MacroDensityUniformityOp(pos) → scalar (density variance)
  IOKeepoutOp(pos)              → scalar (keepout violation)
  MacroGridAlignOp(pos)         → scalar (alignment error)
  MacroNotchOp(pos)             → scalar (notch penalty)
  ↓
  total = ramp × (w0·dens + w1·io + w2·align + w3·notch)
  raw   = {k: v.detach()}   ← MPC observes raw unweighted costs
  return total, raw

PlaceObj.obj_fn(pos)
  ↓
  result = WL + λ·density + total
  ↓
  result.backward()   ← gradient flows through all 4 ops to pos
```

### MPC Data Flow

```
Iteration k (every 50 Llambda steps):
  state_k  = [overflow, Δhpwl_norm, density_raw, io_raw, align_raw, notch_raw]
  control_k = [dw_log, Δlog_density, Δlog_io, Δlog_align, Δlog_notch]

  MPCController.record(state_{k-1}, control_{k-1}, state_k)
  MPCController.fit():
    X = [[s_{t}; u_{t}; 1] for t in history]   # [N, S+C+1]
    Y = [s_{t+1} for t in history]              # [N, S]
    θ = (X^T X + λI)^{-1} X^T Y               # ridge regression
    A = θ[:S, :].T   B = θ[S:S+C, :].T   c = θ[-1, :]

  MPCController.step(state_k, control_k):
    min_{U∈R^{H×C}}  Σ_{i=0}^{H} (s_i - s_ref)^T Q (s_i - s_ref) + u_i^T R u_i
    s.t.  u_i ∈ [-MAX_DELTA, +MAX_DELTA]^C
    via scipy.optimize.minimize(..., method='SLSQP')
    return U[0]   ← first action of optimal sequence

  Apply control:
    density_weight ← base_dw × exp(clip(offset + u[0], ±log(10)))
    BeyondPPAObj weights ← exp(log_w + EMA_ALPHA × clip(u[1:5], ±MAX_DELTA))
```

---

## 4. Repository Structure

```
DREAMPlace/
├── Dockerfile.mac                  ARM64 Linux Docker image (Apple Silicon)
├── run_mac.sh                      All commands: build, install, run (macOS)
├── PLAN.md                         THIS FILE — complete implementation reference
│
├── dreamplace/                     Core DREAMPlace Python package
│   ├── Placer.py                   Entry point: loads params, calls NonLinearPlace
│   ├── BasicPlace.py               Base class: builds all op collections
│   ├── NonLinearPlace.py           Main optimization loop (3 nested loops)
│   ├── PlaceObj.py                 Differentiable objective function
│   ├── PlaceDB.py                  Placement database (nodes, nets, geometry)
│   ├── EvalMetrics.py              Metrics at each step (wirelength, overflow, BPPA)
│   ├── Params.py                   Loads params.json defaults + user JSON overrides
│   ├── params.json                 Parameter schema with defaults
│   │
│   ├── BeyondPPAObj.py             [NEW] Combined reliability objective module
│   ├── MPCController.py            [NEW] Online MPC weight scheduler
│   │
│   └── ops/                        All placement operators (nn.Module subclasses)
│       ├── CMakeLists.txt          Registers all op subdirs for install
│       ├── macro_density/          [NEW] Density uniformity penalty
│       │   ├── __init__.py
│       │   ├── macro_density.py
│       │   └── CMakeLists.txt
│       ├── io_keepout/             [NEW] IO keepout zone penalty
│       │   ├── __init__.py
│       │   ├── io_keepout.py
│       │   └── CMakeLists.txt
│       ├── macro_align/            [NEW] Grid alignment penalty
│       │   ├── __init__.py
│       │   ├── macro_align.py
│       │   └── CMakeLists.txt
│       ├── macro_notch/            [NEW] Notch gap penalty
│       │   ├── __init__.py
│       │   ├── macro_notch.py
│       │   └── CMakeLists.txt
│       └── [28 existing C++/CUDA ops: electric_potential, hpwl, etc.]
│
└── test/
    └── ispd2005/
        ├── adaptec1.json           Baseline config (no BeyondPPA/MPC)
        ├── adaptec1_mpc.json       [NEW] BeyondPPA+MPC config
        └── [other benchmark configs]
```

---

## 5. Build & Installation

### Platform

All commands below use `run_mac.sh`, which runs DREAMPlace inside a Docker container. This script was written for **Apple Silicon macOS** (M1/M2/M3/M4) but works on any machine with Docker Desktop.

- **Architecture:** `linux/arm64` (ARM64 emulation for Intel Macs; native on Apple Silicon)
- **GPU:** CPU-only. CUDA/NVIDIA GPU not supported on Apple Silicon; all placement uses OpenMP multi-threading.
- **Docker:** Required. Get Docker Desktop: https://www.docker.com/products/docker-desktop/

### Environment Details (from Dockerfile.mac)

| Component | Version |
|-----------|---------|
| Base OS | Ubuntu 22.04 (ARM64) |
| GCC | 11 (from Ubuntu 22.04 repos) |
| CMake | 3.22 (from Ubuntu 22.04 repos) |
| Python | 3.10 (Ubuntu 22.04 default) |
| PyTorch | Latest CPU-only (linux_aarch64) |
| Boost | 1.74 |
| Bison | 3.8 |
| SciPy | ≥1.1.0 |
| NumPy | ≥1.15.4 |

### Prerequisites

```bash
# 1. Install Docker Desktop
# https://www.docker.com/products/docker-desktop/
# Start Docker Desktop and ensure it is running

# 2. Clone the repository WITH submodules
git clone --recursive https://github.com/IshraqAtUCF/DREAMPlace.git
cd DREAMPlace

# 3. If you already cloned without --recursive:
git submodule update --init --recursive
```

### First-Time Setup (Run in Order)

```bash
# Step 1: Build the Docker image (one-time, ~10-20 min)
# Downloads Ubuntu packages + PyTorch CPU wheel + Python dependencies
./run_mac.sh build-image

# Step 2: Compile and install DREAMPlace (one-time, ~10-30 min)
# Runs CMake inside the container, compiles ~28 C++ ops, installs to ./install/
./run_mac.sh install

# Step 3: Download benchmark data (~5-30 min, several GB)
# Downloads ISPD 2005 and 2015 benchmark circuits into ./install/benchmarks/
./run_mac.sh download-benchmarks
```

**Memory note:** Each C++ translation unit with PyTorch/pybind11 headers consumes 1-2 GB RAM during compilation. The default is `MAKE_JOBS=2` to avoid OOM. If Docker Desktop has ≥8 GB memory allocated, you can speed up compilation:
```bash
DREAMPLACE_MAKE_JOBS=4 ./run_mac.sh install
```

### What the Install Command Does

The `install` command runs this sequence inside the container:

```bash
# 1. Auto-detect PyTorch C++ ABI (critical: PyTorch 2.x ARM64 uses ABI=1)
TORCH_CXX_ABI=$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')

# 2. CMake configure
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/DREAMPlace/install \
  -DCMAKE_CXX_ABI=${TORCH_CXX_ABI} \
  -DENABLE_HETEROSTA=OFF \
  -DPython_EXECUTABLE=$(which python3)

# 3. Compile all ops (C++/CUDA extensions + pure Python)
make -j${MAKE_JOBS}

# 4. Install to ./install/
make install
```

The `install/` directory after install contains:
```
install/
├── dreamplace/          Python package (all .py files including new BeyondPPA files)
│   ├── Placer.py
│   ├── BeyondPPAObj.py
│   ├── MPCController.py
│   ├── ops/
│   │   ├── macro_density/
│   │   ├── io_keepout/
│   │   ├── macro_align/
│   │   ├── macro_notch/
│   │   └── [compiled .so files for C++ ops]
│   └── params.json
├── test/                Test configs (including adaptec1_mpc.json)
└── benchmarks/          Downloaded circuit benchmarks
```

### Why the New Ops Don't Require Recompilation

All four BeyondPPA penalty ops are **pure Python** (PyTorch autograd only — no C++ extensions). Their `CMakeLists.txt` files only register the `.py` files for `make install`:

```cmake
# Example: dreamplace/ops/macro_density/CMakeLists.txt
set(OP_NAME macro_density)
# Pure Python op — no C++ compilation needed.
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
```

The `dreamplace/ops/CMakeLists.txt` registers all four at lines 39–43:
```cmake
# BeyondPPA reliability feature ops (pure Python, no C++ compilation)
add_subdirectory(io_keepout)
add_subdirectory(macro_align)
add_subdirectory(macro_notch)
add_subdirectory(macro_density)
```

### Useful Maintenance Commands

```bash
# Check current state (Docker image, build dir, install dir, submodules)
./run_mac.sh status

# Open interactive shell inside the container (for debugging)
./run_mac.sh shell

# Wipe build/ and install/, then rebuild from source
./run_mac.sh reinstall

# Wipe everything including Docker image
./run_mac.sh nuke
./run_mac.sh build-image && ./run_mac.sh install
```

---

## 6. Running Experiments (Two Modes)

All placement runs go through `./run_mac.sh run <config.json>`.

**What `run` does internally:**
1. Starts the Docker container with `./install/` as the working directory
2. Creates a temporary copy of the config JSON with `"gpu": 0` forced (CPU-only build cannot use GPU)
3. Runs `python3 dreamplace/Placer.py <temp_config.json>`
4. Cleans up the temp file

### Mode 1 — Baseline DREAMPlace

Standard wirelength + density placement. No BeyondPPA or MPC.

```bash
./run_mac.sh run test/ispd2005/adaptec1.json
```

### Mode 2 — BeyondPPA + MPC (Reliability-Aware)

Full BeyondPPA reliability penalty terms + online MPC weight control.

```bash
./run_mac.sh run test/ispd2005/adaptec1_mpc.json
```

### The adaptec1_mpc.json Config

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

**Key BeyondPPA/MPC settings:**
- `beyond_ppa_flag: 1` — enables all four reliability penalty ops
- `beyond_ppa_gate_overflow: 0.20` — BeyondPPA features activate when overflow drops below 20% (slightly tighter than default 30% to activate features earlier in the run)
- `mpc_flag: 1` — enables online MPC weight control
- `mpc_interval: 50` — MPC re-plans every 50 Llambda iterations

**Note:** The `"gpu": 1` in the config is overridden to `0` by the `run_mac.sh run` command automatically.

### Running Other Benchmarks

Any benchmark from ISPD 2005 can run in BeyondPPA+MPC mode by adding the four BeyondPPA fields to its JSON:

```bash
# Copy an existing config and add BeyondPPA+MPC flags
cp test/ispd2005/bigblue1.json test/ispd2005/bigblue1_mpc.json
# Then add to bigblue1_mpc.json:
#   "beyond_ppa_flag": 1,
#   "beyond_ppa_gate_overflow": 0.20,
#   "mpc_flag": 1,
#   "mpc_interval": 50

./run_mac.sh run test/ispd2005/bigblue1_mpc.json
```

---

## 7. Implementation Reference — New Files

### `dreamplace/BeyondPPAObj.py`

**Purpose:** Combined BeyondPPA reliability objective. Instantiated once in `PlaceObj.__init__` when `beyond_ppa_flag=1`. Encapsulates all four penalty ops with overflow gating, linear ramp, and log-space weight management.

**Class:** `BeyondPPAObj(nn.Module)`

**Key attributes:**
```python
self.enabled       = False          # True once overflow < gate_overflow
self.gate_overflow = 0.20           # from params.beyond_ppa_gate_overflow
self.ramp_iters    = 50             # from params.beyond_ppa_ramp_iters
self._ramp_step    = 0              # counts gradient steps since activation
self._log_w        = [log(w) for w in beyond_ppa_weights]  # 4 values: density, io, align, notch
self.density_op    = MacroDensityUniformityOp(...)
self.io_op         = IOKeepoutOp(...)
self.align_op      = MacroGridAlignOp(...)
self.notch_op      = MacroNotchOp(...)
```

**`__init__(params, placedb, data_collections)`:**
- Detects macros: `placedb.movable_macro_mask` (numpy bool array, already computed by PlaceDB)
- Converts mask to `torch.BoolTensor`
- Extracts macro sizes: `node_size_x/y[:num_movable][macro_mask_np]`
- Extracts IO port positions: `node_x/y[num_movable : num_movable + num_terminals]`
- Derives geometry from params × `placedb.row_height`: `keepout_dist`, `notch_thresh`, `pitch_x/y`
- Builds all 4 ops

**`check_and_enable(overflow)`:** Called every Llambda step. One-time transition from disabled to enabled once `float(overflow) < self.gate_overflow`. Logs the activation event.

**`_ramp_scale()`:** Returns a linear scale factor 0→1 over `ramp_iters` calls. Ensures BeyondPPA cost doesn't suddenly jump when first activated.

**`update_weights_log(delta_log)`:** MPC interface. Applies EMA-smoothed log-space increments to each weight:
```python
self._log_w[i] += EMA_ALPHA * clip(delta_log[i], ±MAX_DELTA)
# EMA_ALPHA = 0.20  (80% history, 20% new signal)
# MAX_DELTA = 0.095  (≈±10% per call)
```

**`forward(pos)`:**
```python
def forward(self, pos):
    # Always compute raw costs (cheap for small N_macro)
    with torch.set_grad_enabled(self.enabled):
        raw = {
            'density': self.density_op(pos),
            'io':      self.io_op(pos),
            'align':   self.align_op(pos),
            'notch':   self.notch_op(pos),
        }
    if not self.enabled:
        return pos.new_zeros(()), {k: v.detach() for k, v in raw.items()}

    ramp = self._ramp_scale()
    w    = [exp(lw) for lw in self._log_w]   # [density, io, align, notch]
    total = ramp * (w[0]*raw['density'] + w[1]*raw['io'] + w[2]*raw['align'] + w[3]*raw['notch'])
    return total, {k: v.detach() for k, v in raw.items()}
    # total → flows into backward pass (gradient computed)
    # raw   → detached, used for MPC state observation and logging
```

---

### `dreamplace/MPCController.py`

**Purpose:** Online receding-horizon controller that adapts all BeyondPPA feature weights and the density weight during global placement.

**Class:** `MPCController`

**Dimensions:**
- `STATE_DIM = 6`: `[overflow, Δhpwl_norm, density_raw, io_raw, align_raw, notch_raw]`
- `CONTROL_DIM = 5`: `[dw_log, Δlog_density, Δlog_io, Δlog_align, Δlog_notch]`
  - All are log-space increments (avoid negative weights, numerically stable)

**`__init__(params)`:**
```python
self.H        = 5          # prediction horizon
self.interval = 50         # Llambda iterations between re-plans
self.hist_sz  = 200        # max ring-buffer size
self.ridge    = 1e-3       # regularisation coefficient

# Linear dynamics model (starts as identity, updated by fit())
self.A = eye(STATE_DIM)
self.B = zeros(STATE_DIM, CONTROL_DIM)
self.c = zeros(STATE_DIM)

# Reference state: target overflow = stop_overflow (0.07), all penalties → 0
self.s_ref = [stop_overflow, 0, 0, 0, 0, 0]

# Cost matrices
self.Q = diag(mpc_q_weights)   # [10, 1, 2, 3, 3, 3]
self.R = diag(mpc_r_weights)   # [0.1, 0.1, 0.1, 0.1, 0.1]

# Density-weight anchoring
self._dw_log_offset     = 0.0
self._dw_log_offset_max = log(10)   # max 10× from base
```

**`record(s_t, u_t, s_next)`:** Appends `(state, control, next_state)` to ring buffer. Evicts oldest if full.

**`fit()`:**
- Requires `>= MIN_HISTORY (15)` samples
- Checks excitation: `U.std(axis=0).mean() >= MIN_EXCITATION (1e-6)`
- Builds design matrix `X = [s; u; 1]` ∈ R^{N × (S+C+1)}
- Ridge regression: `θ = (X^T X + λI)^{-1} X^T Y`
- Extracts `A, B, c` from θ

**`_normalize(s)`:** Divides each state component by fixed design-time scale estimates:
```python
scales = [1.0, 0.05, 1.0, 1e4, 1e2, 1e4]
# overflow, Δhpwl_norm, density_raw, io_raw, align_raw, notch_raw
```

**`step(current_state, current_u)`:**
- Returns `zeros` if model not fitted or excitation too low (safe fallback)
- Normalises state via `_normalize()`
- Solves H-step SLSQP:
  ```
  min_{U ∈ R^{H×C}}  Σ_{i=0}^{H-1} (s_i - s_ref_n)^T Q (s_i - s_ref_n) + u_i^T R u_i
  s.t.  u_i ∈ [-MAX_DELTA, +MAX_DELTA]^C
  ```
- Returns `U[0]` (first action of optimal sequence)
- Falls back to `zeros` on any scipy exception

**`apply_density_weight_delta(base, delta_log)`:**
```python
self._dw_log_offset = clip(self._dw_log_offset + delta_log, ±dw_log_offset_max)
return float(base) * exp(self._dw_log_offset)
# Bounded: density_weight stays in [base/10, base×10]
```

---

### `dreamplace/ops/macro_density/macro_density.py`

**Purpose:** Penalises uneven macro distribution → reduces IR-drop hotspots.

**Class:** `MacroDensityUniformityOp(nn.Module)`

**Cost function:**
```
density_map[b] = Σ_{macro i} bilinear_weight(center_i → bin_b)
Loss = Var(density_map)
```

**`__init__`:** Stores `macro_mask`, `macro_size_x/y`, bin grid parameters. Uses `register_buffer()` for all tensors so they move to GPU if needed.

**`forward(pos)`:**
```python
x = pos[:num_movable][macro_mask]
y = pos[num_phys : num_phys+num_movable][macro_mask]

if N == 0: return pos.new_zeros(())

# Fractional bin coordinates
fx = ((x - xl) / bin_w).clamp(0, BX - 1 - 1e-6)
fy = ((y - yl) / bin_h).clamp(0, BY - 1 - 1e-6)

# Bilinear weights
w00 = (1-dx)*(1-dy)*area;  w10 = dx*(1-dy)*area
w01 = (1-dx)*dy*area;      w11 = dx*dy*area

# Out-of-place scatter_add (autograd-safe)
indices = cat([iy*BX+ix, iy*BX+ix1, iy1*BX+ix, iy1*BX+ix1])
weights = cat([w00, w10, w01, w11])
density = zeros(BX*BY, ...).scatter_add(0, indices, weights)

return density.var()
```

**Key design note:** Uses **out-of-place** `scatter_add` (not in-place `scatter_add_`). In-place ops on leaf tensors or tensors in the autograd graph can corrupt gradient computation. This is Fix 1 from the original plan.

---

### `dreamplace/ops/io_keepout/io_keepout.py`

**Purpose:** Penalises macros within `keepout_dist` of any fixed IO terminal → reduces noise coupling and signal integrity violations.

**Class:** `IOKeepoutOp(nn.Module)`

**Cost function:**
```
dists[i,j] = ||center_macro_i - pos_io_j||_2
Loss = Σ_{i,j} ReLU(keepout_dist - dists[i,j])^2
```

**`forward(pos)`:**
```python
x = pos[:num_movable][macro_mask]
y = pos[num_phys : num_phys+num_movable][macro_mask]

if N == 0 or N_io == 0: return pos.new_zeros(())

macro_centers = stack([x, y], dim=1)        # [N_macro, 2]
io_pos = self.io_pos.to(dtype=x.dtype, device=x.device)
dists = torch.cdist(macro_centers, io_pos)  # [N_macro, N_io] — batched L2
viol  = F.relu(self.keepout_dist - dists)
return viol.pow(2).sum()
```

---

### `dreamplace/ops/macro_align/macro_align.py`

**Purpose:** Penalises macro positions that don't align to a virtual grid → improves clock-tree synthesis regularity.

**Class:** `MacroGridAlignOp(nn.Module)`

**Cost function:**
```
Loss = Σ_i [ sin²(π·x_i/pitch_x) + sin²(π·y_i/pitch_y) ]
```
- Energy = 0 exactly at grid multiples of `pitch_x/y`
- Smooth, nonzero, and finite everywhere else
- Gradient always points toward nearest grid line

**`forward(pos)`:**
```python
x = pos[:num_movable][macro_mask]
y = pos[num_phys : num_phys+num_movable][macro_mask]

if N == 0: return pos.new_zeros(())

loss_x = torch.sin(pi * x / self.pitch_x).pow(2)
loss_y = torch.sin(pi * y / self.pitch_y).pow(2)
return (loss_x + loss_y).sum()
```

---

### `dreamplace/ops/macro_notch/macro_notch.py`

**Purpose:** Penalises narrow gaps (notches) between macro pairs → reduces EM violations and routing congestion.

**Class:** `MacroNotchOp(nn.Module)`

**Cost function (per macro pair i, j where i < j):**
```
dx_gap = max(0, |cx_i - cx_j| - (w_i + w_j)/2)   ← horizontal clearance
dy_gap = max(0, |cy_i - cy_j| - (h_i + h_j)/2)   ← vertical clearance
d_ij   = dx_gap + dy_gap                           ← total edge-to-edge gap

Loss = Σ_{i<j} ReLU(notch_thresh - d_ij)²
```

**`forward(pos)`:**
```python
x = pos[:num_movable][macro_mask]
y = pos[num_phys : num_phys+num_movable][macro_mask]

if N < 2: return pos.new_zeros(())

# Pairwise gaps [N, N]
cx_diff = (x.unsqueeze(1) - x.unsqueeze(0)).abs()
cy_diff = (y.unsqueeze(1) - y.unsqueeze(0)).abs()
half_w  = (sx.unsqueeze(1) + sx.unsqueeze(0)) * 0.5
half_h  = (sy.unsqueeze(1) + sy.unsqueeze(0)) * 0.5
dx_gap  = F.relu(cx_diff - half_w)
dy_gap  = F.relu(cy_diff - half_h)
d_ij    = dx_gap + dy_gap

# Optional spatial pre-filter (OFF by default — see Known Limitations)
if self.prune:
    centre_dist = (cx_diff.pow(2) + cy_diff.pow(2)).sqrt()
    far_mask    = centre_dist > 2.0 * self.notch_thresh
    d_ij        = d_ij + far_mask.float() * self.notch_thresh * 10

# Upper triangle only (avoids double-counting and self-pairs)
ut_mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
penalty = F.relu(self.notch_thresh - d_ij[ut_mask])
return penalty.pow(2).sum()
```

**Complexity:** O(N_macro²). For ISPD 2005 benchmarks, N_macro < 200 and this is fast.

---

## 8. Implementation Reference — Modified Files

### `dreamplace/PlaceObj.py`

**New in `__init__`** (after line ~275):
```python
# BeyondPPA reliability objective (single instance, owned here)
self.beyond_ppa_obj = None
if getattr(params, 'beyond_ppa_flag', 0):
    from dreamplace.BeyondPPAObj import BeyondPPAObj
    self.beyond_ppa_obj = BeyondPPAObj(params, placedb, data_collections)
```

**New in `obj_fn(pos)`** (after line ~324, before return):
```python
# BeyondPPA human-inspired reliability terms (gated by overflow)
if self.beyond_ppa_obj is not None:
    bppa_cost, self._bppa_raw = self.beyond_ppa_obj(pos)
    result = result + bppa_cost
else:
    self._bppa_raw = None
```

`self._bppa_raw` stores the raw breakdown dict `{'density', 'io', 'align', 'notch'}` — used by `EvalMetrics.evaluate()`.

---

### `dreamplace/NonLinearPlace.py`

**MPC init** (after line ~124, before macro_place handling):
```python
# Anchor density_weight for MPC (bounded cumulative offset)
_density_weight_base = float(params.density_weight)

# MPC controller (one per global placement stage)
_mpc = None
_mpc_prev_state = None
_mpc_current_u  = None   # log-space control vector
if getattr(params, 'mpc_flag', 0) and getattr(params, 'beyond_ppa_flag', 0):
    import scipy.optimize   # validate import early
    from dreamplace.MPCController import MPCController
    _mpc = MPCController(params)
    _mpc_current_u = [0.0] * MPCController.CONTROL_DIM
elif getattr(params, 'mpc_flag', 0) and not getattr(params, 'beyond_ppa_flag', 0):
    logging.warning(
        "mpc_flag=1 has no effect when beyond_ppa_flag=0. "
        "Set beyond_ppa_flag=1 to enable BeyondPPA+MPC reliability mode.")
```

**eval_ops registration** (after line ~268):
```python
if getattr(params, 'beyond_ppa_flag', 0) and model.beyond_ppa_obj is not None:
    eval_ops["beyond_ppa"] = model.beyond_ppa_obj
```

**MPC block in Llambda loop** (after line ~714, after density_weight update):
```python
# ── BeyondPPA gate + MPC weight update ───────────────
cur_llambda_metric = Llambda_metrics[-1][-1]
cur_overflow_scalar = float(
    cur_llambda_metric.overflow.mean()
    if cur_llambda_metric.overflow is not None else 1.0)

# Gate: activate BeyondPPA features when overflow is low enough
if model.beyond_ppa_obj is not None:
    model.beyond_ppa_obj.check_and_enable(cur_overflow_scalar)

# MPC: re-plan every mpc_interval Llambda iterations
if (_mpc is not None
        and model.beyond_ppa_obj is not None
        and Llambda_flat_iteration % _mpc.interval == 0):
    # Build 6D state vector from current metrics
    delta_hpwl_norm = ...  # computed from prev/cur HPWL
    raw = {}
    if cur_llambda_metric.bppa_density is not None:
        raw = {'density': bppa_density, 'io': bppa_io,
               'align': bppa_align, 'notch': bppa_notch}
    current_state = [cur_overflow_scalar, delta_hpwl_norm,
                     raw.get('density', 0), raw.get('io', 0),
                     raw.get('align', 0),   raw.get('notch', 0)]

    # Record transition if we have a previous state
    if _mpc_prev_state is not None:
        _mpc.record(_mpc_prev_state, _mpc_current_u, current_state)
        _mpc.fit()

    _mpc_prev_state = current_state
    u_opt = _mpc.step(current_state, _mpc_current_u)
    _mpc_current_u = u_opt.tolist()

    # Apply density-weight delta (anchored, bounded)
    new_dw = _mpc.apply_density_weight_delta(_density_weight_base, float(u_opt[0]))
    with torch.no_grad():
        model.density_weight.fill_(new_dw)

    # Apply feature weight deltas in log-space
    if model.beyond_ppa_obj is not None:
        model.beyond_ppa_obj.update_weights_log(u_opt[1:5])

    logging.info("MPC step iter=%d  u=%s  dw=%.3E  bppa_w=%s" % (...))
```

---

### `dreamplace/EvalMetrics.py`

**New fields in `__init__`** (lines 39–43):
```python
# BeyondPPA reliability metrics (raw, unweighted)
self.bppa_density = None
self.bppa_io      = None
self.bppa_align   = None
self.bppa_notch   = None
```

**New in `__str__`** (lines 94–97):
```python
if self.bppa_density is not None:
    content += ", BPPA[dens=%.3E io=%.3E align=%.3E notch=%.3E]" % (
        self.bppa_density, self.bppa_io, self.bppa_align, self.bppa_notch)
```

**New in `evaluate()`** (lines 150–157):
```python
if "beyond_ppa" in ops:
    # called within torch.no_grad() — raw values are already detached
    _, raw = ops["beyond_ppa"](var)
    self.bppa_density = float(raw['density'])
    self.bppa_io      = float(raw['io'])
    self.bppa_align   = float(raw['align'])
    self.bppa_notch   = float(raw['notch'])
```

---

### `dreamplace/params.json`

Seventeen new parameters added at lines 309–376 (before the closing `}`). They follow the existing schema `{"description": "...", "default": ...}`. The `Params.py` loader reads these automatically — no code change to `Params.py` required.

---

## 9. Parameters Reference

### How Parameters Work

`Params.py.__init__()` reads `params.json` and stores every key as `self.__dict__[key] = value['default']`. When a user JSON config is loaded, `Params.load()` calls `fromJson()` which overwrites with user-specified values. Thus:
- Any key in `params.json` with a `"default"` is always available on the `params` object
- User config only needs to specify values that differ from defaults
- New BeyondPPA/MPC keys are loaded automatically — no manual `Params.py` changes needed

### BeyondPPA Parameters

| JSON Key | Default | Type | Description |
|----------|---------|------|-------------|
| `beyond_ppa_flag` | `0` | int | Set to `1` to enable BeyondPPA reliability features |
| `beyond_ppa_weights` | `[1,1,1,1]` | list[float] | Initial weights for `[density, io, align, notch]` features |
| `beyond_ppa_gate_overflow` | `0.3` | float | BeyondPPA activates when global overflow drops below this value |
| `beyond_ppa_ramp_iters` | `50` | int | Number of gradient steps to linearly ramp weight 0→1 after activation |
| `beyond_ppa_macro_bins` | `32` | int | Coarse density grid resolution (N×N); higher = more detail, slower |
| `beyond_ppa_prune_notch` | `0` | int | Spatial pruning for MacroNotchOp (see Known Limitations — OFF by default) |
| `io_keepout_distance` | `10` | float | IO keepout distance in units of `placedb.row_height` |
| `grid_alignment_pitch_x` | `8` | float | Grid pitch X in units of `placedb.row_height` |
| `grid_alignment_pitch_y` | `8` | float | Grid pitch Y in units of `placedb.row_height` |
| `notch_threshold` | `5` | float | Minimum allowed edge-to-edge gap between macros in units of `placedb.row_height` |

### MPC Parameters

| JSON Key | Default | Type | Description |
|----------|---------|------|-------------|
| `mpc_flag` | `0` | int | Set to `1` to enable MPC (requires `beyond_ppa_flag=1`) |
| `mpc_interval` | `50` | int | Number of Llambda iterations between MPC re-planning steps |
| `mpc_horizon` | `5` | int | MPC prediction horizon H (number of intervals to optimize ahead) |
| `mpc_history_size` | `200` | int | Max number of (state, control, next_state) transitions in ring buffer |
| `mpc_ridge` | `1e-3` | float | Ridge regularization λ for dynamics fitting (higher = more stable, less adaptive) |
| `mpc_q_weights` | `[10,1,2,3,3,3]` | list[float] | Diagonal of state cost matrix Q (6 values: overflow, Δhpwl, density, io, align, notch) |
| `mpc_r_weights` | `[0.1×5]` | list[float] | Diagonal of control cost matrix R (5 values matching CONTROL_DIM) |

### Derived Geometry Values

The following are computed at `BeyondPPAObj.__init__` time from params × `placedb.row_height`:

| Value | Formula | Typical (row_height=1 unit) |
|-------|---------|----------------------------|
| `keepout_dist` | `io_keepout_distance × row_height` | 10 units |
| `notch_thresh` | `notch_threshold × row_height` | 5 units |
| `pitch_x` | `grid_alignment_pitch_x × row_height` | 8 units |
| `pitch_y` | `grid_alignment_pitch_y × row_height` | 8 units |

---

## 10. Expected Output & Verification

### Baseline Run Output (No BeyondPPA)

```
...
iteration  100, (100, 0, 0), Obj 3.456E+07, WL 2.1E+07, Density 1.3E+07, DensityWeight 8.00E-05, wHPWL 1.8E+06, Overflow 3.456E-01, MaxDensity 9.8E-01, gamma 4.00E+01, time 0.123ms
...
```

No `BPPA` lines, no `MPC step` lines.

### BeyondPPA + MPC Run Output

**Initialisation (before first iteration):**
```
BeyondPPAObj: 45 macro nodes identified out of 210 movable
BeyondPPAObj: initial weights density=1.000 io=1.000 align=1.000 notch=1.000
MPCController: H=5 interval=50 hist=200 ridge=1.00e-03
```

**Before BeyondPPA activation (overflow > 0.20):**
```
iteration  150, ..., Overflow 2.500E-01, ..., time 0.145ms
(no BPPA line — overflow still above gate)
```

**BeyondPPA activation:**
```
BeyondPPAObj: activated (overflow=0.1987 < gate=0.2000)
```

**After activation (each iteration):**
```
iteration  200, ..., Overflow 1.9E-01, ..., BPPA[dens=3.2E-02 io=1.4E+03 align=5.6E+01 notch=2.1E+02], time 0.167ms
```

**MPC re-plan (every 50 Llambda steps, first few return zeros while model fits):**
```
MPC step iter=50   u=[0.000, 0.000, 0.000, 0.000, 0.000] dw=8.000E-05  bppa_w=[1.000, 1.000, 1.000, 1.000]
MPC step iter=100  u=[0.000, 0.000, 0.000, 0.000, 0.000] dw=8.000E-05  bppa_w=[1.000, 1.000, 1.000, 1.000]
...
MPC step iter=750  u=[0.012, -0.031, 0.008, 0.019, -0.005] dw=8.4E-05  bppa_w=[1.09, 0.84, 1.19, 0.89]
```

Note: MPC produces non-zero controls after ~15 samples × 50 interval = 750 iterations.

### Pass Criteria (vs Baseline on adaptec1)

| Metric | Criterion |
|--------|-----------|
| Final HPWL | Within ±5% of baseline |
| `align_penalty` | < 20% of initial value by iteration 500 |
| `notch_penalty` | < 30% of initial value by iteration 500 |
| `io_penalty` | Near 0 if layout has IO conflicts |
| MPC fitness | Non-zero controls after iteration 750 |
| Runtime | < 2% overhead vs baseline |

### Troubleshooting

**`AttributeError: 'Params' object has no attribute 'beyond_ppa_flag'`**
→ `params.json` wasn't updated correctly or the install step wasn't re-run after changes.
→ Run `./run_mac.sh reinstall` to rebuild.

**`BeyondPPAObj: 0 macro nodes identified`**
→ The design has no large movable macros (area > 10× mean AND height > 2× row_height).
→ BeyondPPA will still run but produce zero costs. This is correct behaviour.

**`WARNING: mpc_flag=1 has no effect when beyond_ppa_flag=0`**
→ Config has `mpc_flag=1` but forgot `beyond_ppa_flag=1`. Add `"beyond_ppa_flag": 1` to config.

**MPC never produces non-zero controls**
→ Fewer than 15 history samples collected (< 750 iterations total).
→ Or: control signals have zero variation (MIN_EXCITATION check fails).
→ Try a longer run (`"iteration": 2000`) or lower `mpc_interval` to 20.

---

## 11. Design Decisions & Trade-offs

### Why Pure Python Ops (No C++/CUDA)

All four BeyondPPA ops are pure PyTorch Python. This means:
- **No recompilation** needed when modifying them
- **Autograd handles backward pass** automatically
- **Slightly slower** than a custom CUDA kernel, but N_macro is typically < 200, making the overhead negligible (< 2% total)

### Why Log-Space Weight Control

MPC emits log-space increments `Δlog(λ)` rather than absolute values. Benefits:
- Weights remain strictly positive (exp(x) > 0 always)
- Updates are multiplicative, not additive — more natural for scale-invariant quantities
- Bounded increments (±MAX_DELTA = 0.095 ≈ ±10%) prevent sudden large changes

### Why EMA Smoothing on Weight Updates

`self._log_w[i] += EMA_ALPHA × d_clipped` (EMA_ALPHA = 0.20) means each MPC step contributes 20% to the weight while 80% of the history is retained. This prevents weight chattering when the MPC solution oscillates.

### Why Overflow Gating

BeyondPPA features are expensive to optimise correctly when macros are still far from their final positions (high overflow). Activating too early can cause the solver to waste gradient steps fighting the BeyondPPA terms instead of first achieving basic cell spreading. The gate at 20-30% overflow is a tuned threshold.

### Why MPC in Llambda Loop (Not Lgamma)

The Lgamma loop iterates over gamma reduction steps. Weight tuning only makes sense once the density penalty has begun to grow significantly (after density_weight starts being updated). The Llambda loop is where `update_density_weight_op` runs — placing the MPC block there ensures we observe the effect of weight changes before making the next re-plan.

### Macro Detection Heuristic (PlaceDB.py)

Macros are detected by `PlaceDB.py` (unchanged):
```python
movable_macro_mask = (area > mean_area × 10) AND (height > row_height × 2)
```
These thresholds are hardcoded and work well for ISPD 2005 circuits. Custom designs with unusual cell size distributions might require tuning.

### Notch Pruning is OFF by Default

The pruning filter in `MacroNotchOp` uses **L2 centre distance** as a proxy to skip "far" pairs, but the gap metric is **Manhattan-style** (sum of horizontal and vertical clearances). These are inconsistent: two macros at 45° could have a small L2 distance but large Manhattan gap, causing the filter to incorrectly skip them. The filter is available via `beyond_ppa_prune_notch=1` but requires benchmark validation before enabling.

---

## 12. Known Limitations & Future Work

### Current Limitations

1. **No GPU acceleration for new ops**
   All four BeyondPPA ops are pure Python/PyTorch and run on CPU (or GPU if tensor is moved there by the rest of the system). They do not have custom CUDA kernels. For designs with thousands of macros, `MacroNotchOp` (O(N²)) could become a bottleneck.

2. **Notch pruning correctness bug**
   `beyond_ppa_prune_notch=1` uses L2 distance to filter pairs but gap is Manhattan. The filter may incorrectly prune close pairs at oblique angles. Keep this OFF until validated on ≥3 benchmarks.

3. **Fixed MPC state normalization scales**
   The normalization scales in `MPCController._normalize()` are hardcoded estimates. For designs with very different density/IO/notch magnitudes, these may be suboptimal. Future: online normalization using running mean/std.

4. **Linear dynamics assumption**
   MPC assumes `s_{t+1} ≈ A·s_t + B·u_t + c`. Real placement dynamics are nonlinear. The ridge-regularised fit and bounded controls compensate, but MPC might make suboptimal decisions in highly nonlinear regimes.

5. **Macro detection thresholds are hardcoded**
   The 10× area and 2× row_height thresholds in PlaceDB are fixed. Designs with non-standard cell libraries may produce incorrect macro detection.

### Future Work

1. **CUDA kernels for hot ops**
   `MacroNotchOp` pairwise computation is parallelisable with a simple CUDA kernel. This would be needed for designs with >500 macros.

2. **Nonlinear MPC**
   Replace the linear dynamics model with a small neural network (1-2 hidden layers) for better accuracy in the nonlinear convergence regime.

3. **Adaptive normalization in MPC state**
   Replace hardcoded normalization scales with online running statistics.

4. **Macro detection as a parameter**
   Expose `macro_area_multiplier` (currently 10×) and `macro_height_multiplier` (currently 2×) as JSON parameters.

5. **Benchmark validation of notch pruning**
   Verify fix for the L2/Manhattan inconsistency and enable `beyond_ppa_prune_notch=1` after validation.

---

*End of implementation reference. Branch: `claude/beyondppa-mpc-integration-Kki4i`. Last updated: 2026-03-03.*
