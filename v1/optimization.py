import shapely.affinity as shaff
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shg
from shapely.plotting import plot_line

from v1.log_spiral_geometry import get_slope_surface


def main():
    surface = get_slope_surface(slope_angle=60, x_offset=10, elev_bottom=5, elev_top=15)
    interlayer = shg.LineString(np.array([x := np.linspace(-5, 20), y := 1.0 * x - 5]).T)

    plot_line(surface)
    plot_line(interlayer)
    plt.show()
    pass


if __name__ == '__main__':
    main()
