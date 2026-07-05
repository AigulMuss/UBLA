"""Kinematic validation of the rotational-translational mechanism against
PLAXIS 2D output tables.

Demonstrates the three-block (rotational-translational-rotational) failure
kinematics directly in the FEM results, complementing the energy-based
validation in v3.main.run_validation_cases():

1. Parses PLAXIS Output table exports (tab-separated, comma decimals,
   UTF-8 BOM): "Table of total principal strains" and "Table of
   incremental displacements".
2. Reconstructs the model's ground surface from the stress-point cloud
   (max Y per X bin) and fits the slope angle - this is how the 30 deg
   geometry of all three validation cases was confirmed.
3. Identifies the moving mass (|du| above a fraction of the maximum) and
   computes the mean incremental-displacement direction in bins across the
   slope. A direction that rotates continuously at the toe and crest but
   stays constant and face-parallel through the middle band is the
   kinematic signature of the R-T-R mechanism; a compact zone with 100% of
   moving points above a layer interface is the shallow regime flagged by
   the Eq. 29 screening rule.

Usage:
    python scripts/kinematic_validation.py <incremental_displacements.TXT> \
        [--strains <principal_strains.TXT>] [--threshold 0.10]
"""
import argparse

import numpy as np
import pandas as pd


def parse_plaxis_table(path: str, columns: list[str]) -> pd.DataFrame:
    """Parse a PLAXIS Output table export (tab-separated, comma decimals)."""
    with open(path, encoding='utf-8-sig') as f:
        lines = f.readlines()

    def to_float(s: str) -> float:
        s = s.strip()
        return np.nan if s == '' else float(s.replace(',', '.'))

    rows = []
    for line in lines[1:]:
        line = line.rstrip('\r\n')
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split('\t')]
        rows.append([parts[0]] + [to_float(p) for p in parts[1:]])
    df = pd.DataFrame(rows, columns=columns)
    return df.dropna(subset=[c for c in ('X', 'Y') if c in columns])


DISPLACEMENT_COLUMNS = ['soil_element', 'node', 'local', 'X', 'Y', 'dux', 'duy', 'dumag']
STRAIN_COLUMNS = ['soil_element', 'stress_point', 'local', 'X', 'Y',
                  'e1', 'e2', 'e3', 'avg13', 'half_diff13', 'angle', 'eV', 'gamma_s', 'e']


def reconstruct_surface(df: pd.DataFrame, bin_width: float = 1.0) -> np.ndarray:
    """Ground surface as max Y per X bin; returns (n, 2) array."""
    bins = np.arange(df.X.min(), df.X.max() + bin_width, bin_width)
    grouped = df.groupby(pd.cut(df.X, bins), observed=True)['Y'].max()
    return np.array([(iv.mid, y) for iv, y in grouped.items()])


def fit_slope_angle(surface: np.ndarray, y_lo: float = 1.0, y_hi: float = 13.0) -> float:
    """Slope angle (deg) fitted to the inclined part of the surface."""
    sub = surface[(surface[:, 1] > y_lo) & (surface[:, 1] < y_hi)]
    slope, _ = np.polyfit(sub[:, 0], sub[:, 1], 1)
    return float(np.degrees(np.arctan(slope)))


def direction_profile(moving: pd.DataFrame, bin_width: float = 2.5,
                      min_points: int = 20) -> pd.DataFrame:
    """Mean displacement direction (deg, wrapped below 0) in X bins."""
    bins = np.arange(moving.X.min(), moving.X.max() + bin_width, bin_width)
    out = []
    for iv, g in moving.groupby(pd.cut(moving.X, bins), observed=True):
        if len(g) < min_points:
            continue
        ang = np.degrees(np.arctan2(g.duy.mean(), g.dux.mean()))
        if ang > 0:
            ang -= 360.0
        dirs = np.degrees(np.arctan2(g.duy, g.dux))
        spread = np.sqrt(np.mean((np.mod(dirs - ang + 180, 360) - 180) ** 2))
        out.append(dict(x=iv.mid, direction_deg=ang, spread_deg=spread, n=len(g)))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('displacements', help='Table of incremental displacements (.TXT)')
    ap.add_argument('--strains', help='Table of total principal strains (.TXT)')
    ap.add_argument('--threshold', type=float, default=0.10,
                    help='moving-mass cutoff as fraction of max |du| (default 0.10)')
    args = ap.parse_args()

    du = parse_plaxis_table(args.displacements, DISPLACEMENT_COLUMNS)
    moving = du[du.dumag > args.threshold * du.dumag.max()].copy()
    print(f'{len(du)} nodes, {len(moving)} in moving mass '
          f'(>{args.threshold:.0%} of max |du|)')
    print(f'moving-mass extent: X {moving.X.min():.1f}-{moving.X.max():.1f}, '
          f'Y {moving.Y.min():.2f}-{moving.Y.max():.2f}')

    if args.strains:
        strains = parse_plaxis_table(args.strains, STRAIN_COLUMNS)
        surface = reconstruct_surface(strains)
        print(f'fitted slope angle: {fit_slope_angle(surface):.2f} deg, '
              f'crest elevation: {surface[:, 1].max():.2f} m')

    profile = direction_profile(moving)
    print('\nmean displacement direction across the sliding mass:')
    print(profile.to_string(index=False,
                            formatters={'x': '{:.1f}'.format,
                                        'direction_deg': '{:.1f}'.format,
                                        'spread_deg': '{:.1f}'.format}))
    print('\nInterpretation: continuous direction rotation at the toe/crest ends '
          'with a constant, face-parallel band between them is the kinematic '
          'signature of the rotational-translational-rotational mechanism.')


if __name__ == '__main__':
    main()
