# UBLA

Code accompanying "Rotational-translational failure mechanism of multi-layered
soil slopes using upper bound limit analysis" (Mussabayeva, Moon, Satyanaga,
Zhai, Adair, Kim; submitted to *Scientific Reports*). See `CITATION.cff` for
citation details.

The rotational-translational (log-spiral + translational block) mechanism and
the TPE-based optimization procedure described in the manuscript are
implemented in `v3/main.py`. The three benchmark cases from the manuscript
(homogeneous slope, two-layer clay-rock slope, three-layer slope) are defined
in `v3.main.VALIDATION_CASES`.

## Reproducing the validation cases (Figures 7-9, Table 3)

```shell
python -m pip install -r requirements.txt
python -m v3.main
```

This calls `run_validation_cases()`, which runs the TPE optimizer for each of
the three cases (500 trials, 200 startup trials, matching the manuscript's
stated optimization budget) and reports the resulting critical height against
the finite-element (PLAXIS 2D) reference values. Each case uses the partition
curve family (`VALIDATION_CASES[...]['partition_type']`) that converged best
for that case's layering during testing; see the comments next to each entry
in `v3/main.py` for the specific numbers.

This step **does not require a PLAXIS license or a running PLAXIS instance**.
`plxscripting` only needs to be importable (it is a pure Python package with
no license check at import time) because `v1/log_spiral_geometry.py` imports
symbols from `v0/main.py`, which in turn uses `plxscripting` for the
PLAXIS-automation code in the `v0`/`v1`/`v2` prototypes. None of that
automation code executes during `run_validation_cases()` - it only builds
geometry with `shapely`, runs `optuna`, and evaluates the closed-form energy
expressions in `v1/log_spiral_geometry.py`. A live PLAXIS install is only
needed if you want to re-run the finite-element models themselves (see
"Driving PLAXIS directly" below), not to reproduce the analytical mechanism
or the numbers in Table 3.

## Repository layout

- `v0/`, `v1/`, `v2/` - earlier iterations of the mechanism and the PLAXIS
  automation scripts (`launch_plaxis.py`, `plxscripting` calls). Kept because
  `v1/log_spiral_geometry.py` (the energy-dissipation/velocity-compatibility
  formulas) and `v2/main.py` (the `Surface` base class) are imported by `v3`.
- `v3/main.py` - the current mechanism, `VALIDATION_CASES`, `Case`,
  `Optimizer`, and `run_validation_cases()`. This is what the manuscript
  describes and what a reviewer should run to check Figures 7-9 and Table 3.
- `scripts/kinematic_validation.py` - demonstrates the R-T-R kinematics
  directly in the PLAXIS results: parses a "Table of incremental
  displacements" export, extracts the moving mass, and prints the mean
  displacement direction across the slope (continuous rotation at toe and
  crest with a constant face-parallel band between them). Optionally takes
  the principal-strains export to verify the model's slope angle. Requires
  only numpy/pandas; the PLAXIS .TXT exports themselves are not archived
  here.
- `lem/`, remaining `scripts/` - auxiliary limit-equilibrium and development
  scripts, not required for the validation cases above.

## Driving PLAXIS directly

`launch_plaxis.py` starts a local PLAXIS 2D process with the AppServer
enabled, which the automation code in `v0`-`v2` connects to via
`plxscripting`. This requires PLAXIS 2D to be installed at the path hardcoded
in `launch_plaxis.py` and a valid license, and is only relevant if you want to
regenerate the finite-element models themselves rather than the analytical
side of the comparison.

```shell
python launch_plaxis.py
```

### Optuna Dashboard

To inspect an optimization run interactively:

```shell
optuna-dashboard sqlite:///db.sqlite
```

### Parametric study (Tables 5-12, Figure 12)

The broader parametric/sensitivity sweeps reported in the manuscript vary:

- Geometry:
  - Slope angle: [15, 30, 45, 60, 75] degrees
  - Slope height: 5-45 m in 1 m steps
  - Soil layers: 2-5 layers, thickness distributed across the slope height
  - Water table: across the full geometry height, in 1 m steps
- Soil parameters:
  - Cohesion (c_ref): 0-45 kPa in 5 kPa steps
  - Friction angle (phi): 0-30 degrees in 5 degree steps
  - gamma_unsat: 15-30 in steps of 3
  - gamma_sat: gamma_unsat + 3

## Verification notes

The following was checked against a raw PLAXIS stress-point export (Phase 1,
"Table of total principal strains") for Case 3, and against the pre-refactor
reference implementation, before this repository was archived:

- Case 3's slope angle is 30 degrees, not 45 degrees - confirmed by
  reconstructing the ground surface from the stress-point cloud and by
  comparing against the pre-refactor code, which already used 30 degrees for
  this case.
- `get_h_critical()` computes the critical height from the manuscript's
  Eq. 24 scaling (`H_modeled * D / W`), not the vertical span of the failure
  curves - these are different quantities, and only the former matches the
  paper's definition of the critical height.
- With both of the above fixed, Case 3's analytical critical height comes out
  to ~14.5 m against the FEM reference of 13.7 m (6% deviation), not the
  ~1-2% reported in the manuscript's Table 3. This gap is unresolved -
  closing it further would mean auditing the energy-dissipation and
  velocity-compatibility formulas in `v1/log_spiral_geometry.py` against the
  manuscript's Eqs. 1-23, which has not been done.

# Setup

- Create Python virtual env using PyCharm.
- ``python -m pip install -r requirements.txt``
