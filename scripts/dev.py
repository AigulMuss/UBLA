from v3.utils import _colors, get_child_polygons
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.plotting import plot_polygon, plot_line


def main():
    # Example polygon and cut lines
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    lines = [
        LineString([(0, 2), (5, 2)]),  # horizontal line (extends beyond right)
        LineString([(2, -1), (2, 5)])  # vertical line (extends beyond top/bottom)
    ]

    plot_polygon(poly)
    for line in lines:
        plot_line(line)

    polygons = get_child_polygons(poly, lines)
    for p in polygons:
        plot_polygon(p, color=next(_colors))

    plt.show()


if __name__ == '__main__':
    main()
