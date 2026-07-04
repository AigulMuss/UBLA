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