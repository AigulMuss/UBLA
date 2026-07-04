# UBLA

Code accompanying "Rotational-translational failure mechanism of multi-layered
soil slopes using upper bound limit analysis" (Mussabayeva, Moon, Satyanaga,
Zhai, Adair, Kim; submitted to *Scientific Reports*). See `CITATION.cff` for
citation details.

The rotational-translational (log-spiral + translational block) mechanism and
the TPE-based optimization procedure described in the manuscript are
implemented in `v3/main.py`. The three benchmark cases from the manuscript
(homogeneous slope, two-layer clay-rock slope, three-layer slope) are defined
in `v3.main.VALIDATION_CASES` and can be run with:

```shell
python -m v3.main
```

which calls `run_validation_cases()` and reports the optimized critical
height for each case against the finite-element (PLAXIS 2D) reference values
reported in the manuscript.

# Setup

- Create Python virtual env using PyCharm.
- ``python -m pip install -r requirements.txt``

# Simulations

## Launch Plaxis2D

- ``python launch_plaxis.py``

### Optuna Dashboard

```shell
optuna-dashboard.exe sqlite:///db.sqlite
```

### Parametric study
- Geometry:
  - Slope angle: [15, 30, 45, 60, 75]
  - Slope height: 5:1:45 meters.
  - Soil layers: 
    - Quantity: 2-5 layers
    - Layer thickness: distribute thickness across the slope height  
  - Water table:
    - Across the whole geometry height (within geometry)
    - 1m steps.
- Soil parameters:
  - Cohesion (cref): 0:5:45 kPa
  - Friction angle (phi): 0:5:30 degrees.
  - gammaUnsat: 15:3:30
  - gammaSat: 15:3:30 + 3