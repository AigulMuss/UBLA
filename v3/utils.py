import more_itertools as mit
import numpy as np
from optuna import Trial
from optuna.trial import FixedTrial
from shapely import geometry as shg, ops as shops, Polygon, LineString
from shapely.ops import split, unary_union, linemerge, polygonize

from v1.geom_utils import get_xy, get_bottom_line, log_spiral
from v2.main import colors, big_num, get_slope_tangent

_colors = iter(colors)


def _extract_polygons(shapely_geom) -> list[shg.Polygon]:
    match shapely_geom:
        case shg.Polygon():
            return [shapely_geom]
        case shg.MultiPolygon(geoms=geoms):
            return geoms
        case shg.GeometryCollection(geoms=geoms):
            return sum(list(map(_extract_polygons, geoms)), start=[])
        case shg.Point() | shg.LineString():
            return []
        case _:
            raise NotImplementedError(shapely_geom)


def get_slices_by_points(geom: shg.Polygon) -> list[shg.Polygon]:
    return list(mit.flatten([
        _extract_polygons(shg.box(x1, -big_num, x2, big_num).intersection(geom))
        for x1, x2 in mit.windowed(sorted(np.unique(get_xy(geom)[:, 0])), 2)
    ]))


def get_external_work(
        geom: shg.Polygon,
        phita: float,
        gamma: float,
        velocity: float,
) -> float:
    external_work = []
    # plot_polygon(geom, color=next(_colors))
    for slice_geom in get_slices_by_points(geom):
        # plot_polygon(slice_geom, color=next(_colors))
        slope_angle = np.arctan(get_slope_tangent(get_bottom_line(slice_geom)))
        external_work.append(
            gamma *
            velocity *
            slice_geom.area *
            abs(np.sin(slope_angle - phita))
        )

    return np.array(external_work).sum()


def get_normal(line: shg.LineString, point: shg.Point,
               x_offset: float) -> shg.LineString:
    normal_tangent = -1 / get_slope_tangent(line)
    xs = np.array([-x_offset, x_offset]) + point.x
    ys = get_line_points(normal_tangent, point, xs)
    return shg.LineString(np.stack([xs, ys], axis=-1))


def get_line_points(m, point: shg.Point, xs: np.ndarray | float):
    return m * (xs - point.x) + point.y


def spiral_xy(p0_x: float, p0_y: float, theta_x: float, rx: float):
    return np.stack([
        p0_x + rx * np.cos(theta_x),
        p0_y + rx * np.sin(theta_x),
    ], axis=-1)


def get_log_spiral_failure_curve_raw(
        origin: shg.Point,
        start: shg.Point,
        phita: float,
        num: int = 20,
) -> shg.LineString:
    toe_theta = np.arctan(get_slope_tangent(shg.LineString([start, origin])))
    toe_radius = start.distance(origin)
    R0 = toe_radius * np.exp(-(toe_theta - 0) * np.tan(phita))
    theta_x = np.linspace(toe_theta, np.pi / 2, num, True)
    Rx = log_spiral(R0, 0, theta_x, phita=phita)
    return shg.LineString(spiral_xy(origin.x, origin.y, theta_x, Rx))


def get_log_spiral_and_point(
        origin: shg.Point,
        start: shg.Point,
        end: shg.LineString,
        phita: float,
        num: int = 50,
) -> tuple[shg.LineString, shg.Point]:
    failure_curve = get_log_spiral_failure_curve_raw(origin, start, phita, num=num)
    endpoint: shg.Point = failure_curve.intersection(end)
    return split(failure_curve, end).geoms[0], endpoint


def get_linear_and_point(
        start: shg.Point,
        end: shg.LineString,
        angle: float,
) -> tuple[shg.LineString, shg.Point]:
    xs = np.array([start.x, big_num])
    ys = get_line_points(np.tan(angle), start, xs)
    failure_curve = shg.LineString(np.stack([xs, ys], axis=-1))
    endpoint: shg.Point = failure_curve.intersection(end)
    return split(failure_curve, end).geoms[0], endpoint


def get_split(geom, splitter, ret_closer_to: shg.Point = None):
    ret_closer_to = ret_closer_to or shg.Point(0, -1000000)
    g1, g2 = split(geom, splitter).geoms
    if g1.distance(ret_closer_to) < g2.distance(ret_closer_to):
        return g1
    else:
        return g2
    # if g1.centroid.y > g2.centroid.y:
    #     return g2 if ret_lower else g1
    # else:
    #     return g1 if ret_lower else g2


def convert_enclosed_area_to_polygon(lines: list[shg.LineString],
                                     buffer_size: float = 1e-3) -> shg.Polygon:
    """https://gis.stackexchange.com/questions/420755/convert-area-between-linestrings-to-polygon-in-shapely"""
    union_result = shops.unary_union(lines).buffer(buffer_size)
    geoms = [shg.Polygon(x) for x in union_result.interiors]
    biggest_geom = max(geoms, key=lambda g: g.area)
    return shg.Polygon(biggest_geom).buffer(buffer_size, join_style='mitre')


def suggest_origin(trial: Trial | FixedTrial, idx: int, x: list[float],
                   y: list[float]) -> shg.Point:
    return shg.Point(
        trial.suggest_float(f'origin_{idx}_x', *x),
        trial.suggest_float(f'origin_{idx}_y', *y),
    )


def get_child_polygons(
        poly: Polygon,
        lines: list[LineString],
        area_tol: float = 1e-6,
):
    # Combine polygon boundary and lines into one network
    network = unary_union([poly.boundary] + lines)  # snap line intersections
    merged = linemerge(network)  # merge collinear segments
    # Polygonize to get all pieces
    pieces = list(polygonize(merged))
    # Filter pieces inside original polygon
    return [p for p in pieces if p.intersects(poly) and p.area > area_tol]


def round_coordinates(geom, ndigits=12):
    def _round_coords(x, y, z=None):
        x = round(x, ndigits)
        y = round(y, ndigits)

        if z is not None:
            z = round(x, ndigits)
            return (x, y, z)
        else:
            return (x, y)

    return shops.transform(_round_coords, geom)
