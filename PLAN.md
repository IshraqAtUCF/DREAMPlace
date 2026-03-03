# BeyondPPA + MPC Integration into DREAMPlace

## Context

**Why this change exists:** The paper *BeyondPPA: Human-Inspired Reinforcement Learning for Post-Route Reliability-Aware Macro Placement* (2025 MLCAD) shows that DREAMPlace's current objective — wirelength + density — leaves critical reliability gaps: unaligned macros, IO port congestion, standard-cell notches, and uneven density clusters. These latent defects manifest post-routing as EM violations, IR drop, clock skew, and glitch noise.

**What we're building:** Five human-inspired structural cost terms (from BeyondPPA) are integrated directly into DREAMPlace's differentiable objective, removing the need for a separate RL training phase. On top of that, a Model Predictive Control (MPC) layer adaptively tunes all feature weights online (during a single placement run) to achieve the best reachable trade-off between PPA and reliability — staying fast.

**Better-than-paper approach:** The paper uses a DQN RL agent that requires offline training. Our approach skips that entirely by making all five features differentiable PyTorch penalty terms so Nesterov gradient descent optimizes them in one shot. MPC replaces static λ weights with a receding-horizon controller that predicts future metric trajectories and solves for optimal weight sequences every 50 iterations. This is faster, requires no pretraining, and adapts online to each design.

---

## Architecture Overview

```
DREAMPlace Main Loop (NonLinearPlace.__call__)
│
├── [NEW] BeyondPPAObj  ─ 5 differentiable penalty terms, macro-only
│   ├── MacroDensityUniformityOp   (density variance → IR drop)
│   ├── IOKeepoutOp                (IO proximity violation → noise)
│   ├── MacroGridAlignOp           (grid deviation → clock skew)
│   └── MacroNotchOp               (tight-gap penalty → EM/congestion)
│   (wirelength already exists in PlaceObj)
│
├── PlaceObj.obj_fn()  ─ extended objective
│   └── total = wirelength + λ_density*density + BeyondPPAObj(pos)
│
└── [NEW] MPCController  ─ every 50 iterations
    ├── State:   [overflow, Δhpwl, density_std, io_pen, align_pen, notch_pen]
    ├── Control: [density_weight_scale, λ_wire, λ_density_feat, λ_io, λ_align, λ_notch]
    ├── Model:   online-fitted linear dynamics  s_{t+1} = A·s_t + B·u_t + c
    └── Solver:  scipy SLSQP over H=5 step horizon
```

---

## Paper Feature Formulations (from Appendix A)

| # | Feature | Cost Term | Reliability Impact |
|---|---------|-----------|-------------------|
| 1 | Wirelength | Jwire = λ1 · Σ HPWL(Mi) | RC delay, EM, crosstalk |
| 2 | Density uniformity | Jdensity = λ2 · Σ (ρ(xi,yi) − ρ̄)² | IR drop, thermal |
| 3 | IO keepout | Jkeepout = λ3 · Σ Σ 1[Mi ∩ Kj ≠ ∅] | Signal integrity, noise |
| 4 | Grid alignment | Jalign = λ4 · Σ min‖(xi,yi) − (gx,gy)‖² | Clock skew, CTS |
| 5 | Notch avoidance | Jnotch = λ5 · Σ Σ 1[dij < δnotch] | EM, congestion |
| – | Total | J = Jwire + Jdensity + Jkeepout + Jalign + Jnotch | All of the above |

**Differentiable approximations used:**
- IO keepout: `ReLU(keepout_dist - ‖macro_center - io_pos‖)²`
- Grid alignment: `sin²(π·x/pitch_x) + sin²(π·y/pitch_y)` — periodic soft-snap
- Notch: `ReLU(δ_notch - edge_edge_distance(Mi, Mj))²` — soft hinge
- Density: bilinear-interpolated density variance over macro sub-grid

---

## New Files to Create

### `dreamplace/ops/macro_density/macro_density.py`
Purpose: Compute variance of macro density across the placement canvas.

```python
class MacroDensityUniformityOp(nn.Module):
    """
    Penalizes uneven macro distribution → mitigates IR drop hotspots.
    Uses bilinear-interpolated density into a coarse macro grid.
    """
    def __init__(self, macro_mask, num_bins, node_size_x, node_size_y,
                 xl, xh, yl, yh):
        # macro_mask: bool tensor [num_movable_nodes]
        # Returns: scalar variance of macro density map

    def forward(self, pos):
        # 1. Extract macro positions using macro_mask
        # 2. Bilinear-interpolate macro area into density_map [num_bins x num_bins]
        # 3. return density_map.var()
```

### `dreamplace/ops/io_keepout/io_keepout.py`
Purpose: Penalize macros that encroach on IO keepout zones.

```python
class IOKeepoutOp(nn.Module):
    """
    Soft penalty for macros within keepout_distance of any IO port.
    io_positions: tensor [num_ios, 2] of port centers from placedb.
    """
    def forward(self, pos):
        macro_centers = pos_to_macro_centers(pos)            # [N_macro, 2]
        dists = torch.cdist(macro_centers, self.io_pos)      # [N_macro, N_io]
        viol = F.relu(self.keepout_dist - dists)             # hinge loss
        return viol.pow(2).sum()
```

IO port extraction: `placedb.node_names` filtered by size (terminals) gives IO positions.

### `dreamplace/ops/macro_align/macro_align.py`
Purpose: Penalize macro deviation from virtual grid anchors.

```python
class MacroGridAlignOp(nn.Module):
    """
    Periodic alignment potential. sin² is 0 exactly on grid multiples,
    increasing smoothly between — naturally differentiable.
    """
    def forward(self, pos):
        macro_centers = ...
        loss_x = torch.sin(math.pi * macro_centers[:, 0] / self.pitch_x).pow(2)
        loss_y = torch.sin(math.pi * macro_centers[:, 1] / self.pitch_y).pow(2)
        return (loss_x + loss_y).sum()
```

Grid pitch defaults: `pitch_x = 8 * row_height`, `pitch_y = 8 * row_height`.

### `dreamplace/ops/macro_notch/macro_notch.py`
Purpose: Penalize narrow gaps (notches) between adjacent macros.

```python
class MacroNotchOp(nn.Module):
    """
    Edge-to-edge distance between macro pairs.
    d_ij = Chebyshev-style gap:
        dx_gap = max(0, |cx_i - cx_j| - (wx_i + wx_j)/2)
        dy_gap = max(0, |cy_i - cy_j| - (wy_i + wy_j)/2)
        d_ij = dx_gap + dy_gap
    Penalty = ReLU(δ_notch - d_ij)² for all pairs.
    Efficient: only evaluate pairs within 2*δ_notch neighborhood (spatial hash).
    """
    def forward(self, pos):
        ...
        gap = dx_gap + dy_gap                              # [N_macro, N_macro]
        return F.relu(self.notch_thresh - gap).pow(2).triu(1).sum()
```

### `dreamplace/BeyondPPAObj.py`
Purpose: Combines all BeyondPPA ops into a single module with adaptive weights.

```python
class BeyondPPAObj(nn.Module):
    """
    Computes J = λ_io*Jkeepout + λ_align*Jalign + λ_notch*Jnotch + λ_dens*Jdensity
    (wirelength is handled by PlaceObj)

    Weights are tensors so MPC can update them in-place.
    """
    def __init__(self, params, placedb, data_collections):
        self.macro_mask = torch.BoolTensor(placedb.movable_macro_mask)
        self.weights = {
            'io': params.beyond_ppa_weights[2],
            'align': params.beyond_ppa_weights[3],
            'notch': params.beyond_ppa_weights[4],
            'density': params.beyond_ppa_weights[1],
        }
        self.io_keepout_op = IOKeepoutOp(...)
        self.align_op = MacroGridAlignOp(...)
        self.notch_op = MacroNotchOp(...)
        self.density_op = MacroDensityUniformityOp(...)

    def forward(self, pos):
        costs = {
            'io':      self.weights['io']      * self.io_keepout_op(pos),
            'align':   self.weights['align']   * self.align_op(pos),
            'notch':   self.weights['notch']   * self.notch_op(pos),
            'density': self.weights['density'] * self.density_op(pos),
        }
        return sum(costs.values()), costs   # total + breakdown

    def update_weights(self, weight_dict):
        self.weights.update(weight_dict)
```

### `dreamplace/MPCController.py`
Purpose: Model Predictive Control for adaptive weight scheduling.

```python
class MPCController:
    """
    State:   s = [overflow, delta_hpwl, density_std, io_pen, align_pen, notch_pen]
    Control: u = [density_weight_scale, λ_wire, λ_density, λ_io, λ_align, λ_notch]

    Dynamics: s_{t+1} = A·s_t + B·u_t + c  (fitted online via least-squares)

    MPC objective (H-step receding horizon):
        min_{u0..uH} Σ (s_k - s_ref)ᵀ Q (s_k - s_ref) + Δu_kᵀ R Δu_k
        s.t.  u_min ≤ u_k ≤ u_max
              |u_{k+1} - u_k| ≤ Δu_max   (rate limits)

    Runs every `mpc_interval` iterations. O(H * control_dim²) per call.
    """

    STATE_DIM   = 6
    CONTROL_DIM = 6

    def __init__(self, params):
        self.H        = getattr(params, 'mpc_horizon', 5)
        self.interval = getattr(params, 'mpc_interval', 50)
        self.hist_sz  = getattr(params, 'mpc_history_size', 200)

        self.history  = []          # list of (state, control, next_state)
        self.A = np.eye(self.STATE_DIM)
        self.B = np.zeros((self.STATE_DIM, self.CONTROL_DIM))
        self.c = np.zeros(self.STATE_DIM)
        self.model_fitted = False

        # Target: converge overflow, minimize reliability penalties
        self.s_ref  = np.array([params.stop_overflow, 0, 0, 0, 0, 0])
        self.Q = np.diag(getattr(params, 'mpc_q_weights', [10, 1, 2, 3, 3, 3]))
        self.R = np.diag(getattr(params, 'mpc_r_weights', [0.1]*6))

        self.u_min = np.array([1e-5, 1e-2, 1e-3, 1e-3, 1e-3, 1e-3])
        self.u_max = np.array([0.1,  5.0,  5.0,  5.0,  5.0,  5.0])

    def record(self, state, control, next_state):
        self.history.append((state, control, next_state))
        if len(self.history) > self.hist_sz:
            self.history.pop(0)

    def fit(self):
        """Fit linear dynamics from history using numpy least squares."""
        if len(self.history) < 15:
            return
        SC = np.array([[*s, *u] for s, u, _ in self.history])
        SN = np.array([ns for _, _, ns in self.history])
        params_mat, _, _, _ = np.linalg.lstsq(
            np.hstack([SC, np.ones((len(SC), 1))]), SN, rcond=None)
        d = self.STATE_DIM
        self.A = params_mat[:d, :].T
        self.B = params_mat[d:d+self.CONTROL_DIM, :].T
        self.c = params_mat[-1, :]
        self.model_fitted = True

    def step(self, current_state, current_control):
        """Return optimal control for next interval via receding horizon opt."""
        if not self.model_fitted:
            return current_control  # fallback: no change

        s0 = np.array(current_state)
        u0 = np.array(current_control)

        def cost(u_flat):
            U = u_flat.reshape(self.H, self.CONTROL_DIM)
            s = s0.copy()
            total = 0.0
            for k in range(self.H):
                s = self.A @ s + self.B @ U[k] + self.c
                ds = s - self.s_ref
                du = U[k] - u0
                total += ds @ self.Q @ ds + du @ self.R @ du
            return total

        bounds = [(lo, hi) for lo, hi in
                  zip(np.tile(self.u_min, self.H),
                      np.tile(self.u_max, self.H))]
        res = scipy.optimize.minimize(
            cost, np.tile(u0, self.H), method='SLSQP',
            bounds=bounds, options={'maxiter': 50, 'ftol': 1e-6})

        return res.x[:self.CONTROL_DIM]   # first control action
```

---

## Files to Modify

### `dreamplace/EvalMetrics.py`
Add new metric fields (after existing fields):

```python
# Line ~79 — add to EvalMetrics.__init__:
self.io_penalty    = 0.0
self.align_penalty = 0.0
self.notch_penalty = 0.0
self.density_std   = 0.0
```

Add to `evaluate()` when `params.beyond_ppa_flag`:

```python
if 'beyond_ppa' in eval_ops:
    _, breakdown = eval_ops['beyond_ppa'](pos)
    self.io_penalty    = breakdown['io'].item()
    self.align_penalty = breakdown['align'].item()
    self.notch_penalty = breakdown['notch'].item()
    self.density_std   = breakdown['density'].item()
```

### `dreamplace/PlaceObj.py`
Extend `__init__` (after line ~196):

```python
self.beyond_ppa_obj = None
if params.beyond_ppa_flag:
    from dreamplace.BeyondPPAObj import BeyondPPAObj
    self.beyond_ppa_obj = BeyondPPAObj(params, placedb, data_collections)
```

Extend `obj_fn()` (line ~318, before `return result`):

```python
if self.beyond_ppa_obj is not None:
    bppa_cost, _ = self.beyond_ppa_obj(pos)
    result = result + bppa_cost
```

Expose weight update method:

```python
def update_beyond_ppa_weights(self, weight_dict):
    if self.beyond_ppa_obj is not None:
        self.beyond_ppa_obj.update_weights(weight_dict)
```

### `dreamplace/BasicPlace.py`
Add `beyond_ppa_op` to `PlaceOpCollection` (line ~244):

```python
self.beyond_ppa_op = None
```

Build it in `__init__` (after timing_op setup, ~line 405):

```python
if params.beyond_ppa_flag:
    from dreamplace.BeyondPPAObj import BeyondPPAObj
    self.op_collections.beyond_ppa_op = BeyondPPAObj(
        params, placedb, data_collections)
```

Add to `eval_ops` dict in `__call__`:

```python
if params.beyond_ppa_flag:
    eval_ops['beyond_ppa'] = self.op_collections.beyond_ppa_op
```

### `dreamplace/NonLinearPlace.py`
Import and initialize MPC (top of `__call__`, before main loop):

```python
mpc = None
if params.mpc_flag:
    import scipy.optimize
    from dreamplace.MPCController import MPCController
    mpc = MPCController(params)
    current_control = np.array([1.0, *params.beyond_ppa_weights[1:]])
    prev_state = None
```

Inject MPC step in the Llambda loop, after metrics evaluation (~line 702):

```python
# ---- MPC weight update ----
if mpc is not None and iteration % params.mpc_interval == 0:
    cur = cur_metric
    state = np.array([
        cur.overflow.mean().item() if torch.is_tensor(cur.overflow) else cur.overflow,
        (cur.hpwl - prev_metric.hpwl).item() / max(cur.hpwl.item(), 1),
        getattr(cur, 'density_std', 0.0),
        getattr(cur, 'io_penalty', 0.0),
        getattr(cur, 'align_penalty', 0.0),
        getattr(cur, 'notch_penalty', 0.0),
    ])
    if prev_state is not None:
        mpc.record(prev_state, current_control, state)
        mpc.fit()
    current_control = mpc.step(state, current_control)
    prev_state = state

    # Apply control to model
    dens_scale = float(current_control[0])
    with torch.no_grad():
        model.density_weight *= dens_scale
    if params.beyond_ppa_flag:
        model.update_beyond_ppa_weights({
            'density': float(current_control[2]),
            'io':      float(current_control[3]),
            'align':   float(current_control[4]),
            'notch':   float(current_control[5]),
        })
```

### `dreamplace/params.json`

```json
"beyond_ppa_flag" : {
    "description" : "enable BeyondPPA human-inspired reliability features for macro placement",
    "default" : 0
},
"beyond_ppa_weights" : {
    "description" : "initial weights [λ_wire, λ_density, λ_io, λ_align, λ_notch]",
    "default" : [1.0, 1.0, 1.0, 1.0, 1.0]
},
"io_keepout_distance" : {
    "description" : "minimum distance (in site rows) between macros and IO ports",
    "default" : 10
},
"grid_alignment_pitch_x" : {
    "description" : "macro alignment grid pitch in x (site rows)",
    "default" : 8
},
"grid_alignment_pitch_y" : {
    "description" : "macro alignment grid pitch in y (site rows)",
    "default" : 8
},
"notch_threshold" : {
    "description" : "minimum allowed edge-to-edge macro gap (site rows)",
    "default" : 5
},
"mpc_flag" : {
    "description" : "enable Model Predictive Control for adaptive feature weight tuning",
    "default" : 0
},
"mpc_interval" : {
    "description" : "iterations between MPC re-planning steps",
    "default" : 50
},
"mpc_horizon" : {
    "description" : "MPC prediction horizon H",
    "default" : 5
},
"mpc_history_size" : {
    "description" : "number of (state, control) pairs kept for online model fitting",
    "default" : 200
},
"mpc_q_weights" : {
    "description" : "MPC state cost diagonal Q weights [overflow, δhpwl, density_std, io, align, notch]",
    "default" : [10.0, 1.0, 2.0, 3.0, 3.0, 3.0]
},
"mpc_r_weights" : {
    "description" : "MPC control cost diagonal R weights",
    "default" : [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}
```

---

## Directory Structure (New Files)

```
dreamplace/
├── BeyondPPAObj.py                         NEW
├── MPCController.py                        NEW
├── ops/
│   ├── macro_density/
│   │   ├── __init__.py                     NEW
│   │   └── macro_density.py               NEW
│   ├── io_keepout/
│   │   ├── __init__.py                     NEW
│   │   └── io_keepout.py                  NEW
│   ├── macro_align/
│   │   ├── __init__.py                     NEW
│   │   └── macro_align.py                 NEW
│   └── macro_notch/
│       ├── __init__.py                     NEW
│       └── macro_notch.py                 NEW
├── PlaceObj.py                             MODIFY
├── NonLinearPlace.py                       MODIFY
├── BasicPlace.py                           MODIFY
├── EvalMetrics.py                          MODIFY
└── params.json                             MODIFY
```

---

## Macro Identification (Existing Code to Reuse)

`PlaceDB.py` lines 800–803 already compute `movable_macro_mask`:

```python
movable_node_area = node_size_x[:num_movable_nodes] * node_size_y[:num_movable_nodes]
movable_macro_mask = (movable_node_area > mean_movable_node_area * 10) & \
                     (node_size_y[:num_movable_nodes] > row_height * 2)
```

All new ops receive `data_collections.movable_macro_mask` (convert to `torch.BoolTensor` once at init).

---

## IO Port Positions (Existing Data to Reuse)

```python
io_start = placedb.num_movable_nodes
io_end   = placedb.num_movable_nodes + placedb.num_terminals
io_pos_x = data_collections.pos[0][io_start:io_end]
```

See `BasicPlace.py` `PlaceDataCollection` for exact layout of the pos tensor.

---

## MPC State & Control Reference

| Index | State Component | Units | Target |
|-------|----------------|-------|--------|
| 0 | overflow (mean) | fraction | stop_overflow (0.07) |
| 1 | Δhpwl / hpwl | fraction | 0 |
| 2 | density_std | normalized | 0 |
| 3 | io_penalty | normalized | 0 |
| 4 | align_penalty | normalized | 0 |
| 5 | notch_penalty | normalized | 0 |

| Index | Control Variable | Range | Default |
|-------|----------------|-------|---------|
| 0 | density_weight_scale | [1e-5, 0.1] | 1.0 |
| 1 | λ_wire | [0.01, 5.0] | 1.0 |
| 2 | λ_density | [0.001, 5.0] | 1.0 |
| 3 | λ_io | [0.001, 5.0] | 1.0 |
| 4 | λ_align | [0.001, 5.0] | 1.0 |
| 5 | λ_notch | [0.001, 5.0] | 1.0 |

---

## Implementation Order

| Step | Task | Goal |
|------|------|------|
| 1 | Create 4 op files in `ops/` | Unit test each; verify gradients flow |
| 2 | Build `BeyondPPAObj.py` | Combine ops; test `backward()` reaches macro positions |
| 3 | Update `EvalMetrics` + `params.json` | New fields visible in logs |
| 4 | Integrate into `PlaceObj` | Run with `beyond_ppa_flag=1, mpc_flag=0`; verify metrics appear |
| 5 | Build `MPCController.py` | Unit test SLSQP convergence on synthetic data |
| 6 | Inject MPC in `NonLinearPlace.py` | Enable `mpc_flag=1`; verify weight updates logged every 50 iters |
| 7 | Tune | Compare HPWL/overflow/alignment/notch vs baseline |

---

## Verification

```bash
# 1. Baseline
./run_mac.sh run test/ispd2005/adaptec1.json

# 2. BeyondPPA only (no MPC)
./run_mac.sh run test/ispd2005/adaptec1_bppa.json   # beyond_ppa_flag=1, mpc_flag=0

# 3. Full (BeyondPPA + MPC)
./run_mac.sh run test/ispd2005/adaptec1_mpc.json    # beyond_ppa_flag=1, mpc_flag=1
```

**Expected log additions:**
```
[INFO] BeyondPPA  io_penalty=0.142  align_penalty=0.083  notch_penalty=0.031  density_std=0.011
[INFO] MPC step   iteration=50   control=[1.00, 1.02, 0.80, 1.15, 0.92, 1.30]
[INFO] MPC step   iteration=100  control=[0.98, 1.05, 0.61, 1.42, 0.88, 1.55]
```

**Pass criteria vs baseline:**
- HPWL within ±5%
- `align_penalty` → < 20% of initial value by iteration 500
- `notch_penalty` → < 30% of initial value by iteration 500
- `io_penalty` → 0 or near-0
- MPC model fit R² > 0.7 after 200 history samples

---

## Key Dependencies (No New Packages Required)

- `torch`, `torch.nn`, `torch.nn.functional` — already in DREAMPlace
- `numpy` — already required
- `scipy.optimize.minimize` — already available; add `import scipy.optimize` in `MPCController.py`

---

## Critical Files Reference

| Role | File | Key Lines |
|------|------|-----------|
| Main objective | `dreamplace/PlaceObj.py` | 293–320 (obj_fn), 186–196 (density_weight init) |
| Main loop | `dreamplace/NonLinearPlace.py` | 624–700 (Lgamma/Llambda loops), 702–709 (density weight update) |
| Op collection | `dreamplace/BasicPlace.py` | 223–251 (PlaceOpCollection), 393–427 (op builds) |
| Macro mask | `dreamplace/PlaceDB.py` | 800–803 (movable_macro_mask) |
| Metrics | `dreamplace/EvalMetrics.py` | 12–100 |
| Parameters | `dreamplace/params.json` | All |
| Params loader | `dreamplace/Params.py` | 19 (__init__), 133 (load) |
