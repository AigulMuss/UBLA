from typing import Iterable

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from shapely import geometry as shg, affinity as shaff, ops as shops

big_num: float = 1e5


def get_polygon_triangle(angle_betta: float, height: float) -> shg.Polygon:
    p1 = shg.Point(0, 0)
    line_bottom = shg.LineString([p1, shaff.translate(p1, big_num)])
    line_slope: shg.LineString = shaff.rotate(line_bottom, 90 - angle_betta, origin=p1)
    line_top = shaff.translate(line_bottom, yoff=height)
    # line_vertical = shg.LineString([p1, p3 := shaff.translate(p1, yoff=height)])
    polygon = shg.Polygon([
        p1,
        line_slope.intersection(line_top),
        shaff.translate(p1, yoff=height),
    ])
    return polygon


def log_spiral(r0: float, theta_0: float, theta: float, phita: float):
    return r0 * np.exp((theta - theta_0) * np.tan(phita))


def get_log_spiral_polygon(
        r0: float,
        theta_0: float,
        theta_h: float,
        phita: float,
        num_points_spiral: int = 100,
) -> shg.Polygon:
    def get_point_along_log_spiral(theta_x: float) -> shg.Point:
        rx = log_spiral(r0, theta_0, theta_x, phita)
        return shg.Point(
            p0.x + rx * np.cos(theta_x),
            p0.y - rx * np.sin(theta_x),
        )

    rh = log_spiral(r0, theta_0, theta_h, phita)

    # origin
    p0 = shg.Point(
        -rh * np.cos(theta_h),
        rh * np.sin(theta_h),
    )
    # logger.debug("p0:\n{}", p0)
    thetas = np.linspace(theta_0, theta_h, num_points_spiral)
    points = [get_point_along_log_spiral(theta) for theta in thetas]

    pb = get_point_along_log_spiral(theta_0)
    # logger.debug("pb:\n{}", pb)
    pa = shg.Point(0, pb.y)
    # logger.debug("pa:\n{}", pa)
    return shg.Polygon([pa] + points)


def get_xy(polygon: shg.Polygon):
    return np.array(list(polygon.boundary.coords))


def get_slices_by_points(geom: shg.Polygon) -> list[shg.Polygon]:
    xy = get_xy(geom)
    # x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    # logger.debug("x_min: {}", x_min)
    # logger.debug("x_max: {}", x_max)
    x_positions = sorted(np.unique(xy[:, 0]))
    return _slice_by_point_positions(geom, x_positions)


def _slice_by_point_positions(geom: shg.Polygon, x_positions: Iterable[float]) -> list[
    shg.Polygon]:
    geom_area = geom.area
    line_vertical = shg.LineString([shg.Point(0, -big_num), shg.Point(0, big_num)])
    slice_geoms = []
    for x_offset in x_positions[1:]:
        try:
            geom_slice, geom = shops.split(geom, shaff.translate(line_vertical,
                                                                 xoff=x_offset)).geoms
        except ValueError as exc:
            # logger.warning("exc: {}", exc)
            continue
        slice_geoms.append(geom_slice)
    slice_geoms.append(geom)
    sliced_area = sum([x.area for x in slice_geoms])
    assert abs(sliced_area - geom_area) < 1e-9, sliced_area - geom_area
    return [x for x in slice_geoms if x.area > 1e-9]


def get_slices(geom: shg.Polygon, step: float = 0.1) -> list[shg.Polygon]:
    xy = get_xy(geom)
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    x_positions = np.arange(x_min, x_max, step=step)
    return _slice_by_point_positions(geom, x_positions)


def plot_polygon(polygon: shg.Polygon):
    plt.plot(*polygon.exterior.xy)
    plt.gca().set_aspect('equal')
    plt.show()


def get_bottom_line(slice_geom: shg.Polygon) -> shg.LineString:
    xy = get_xy(slice_geom)
    # logger.debug("xy:\n{}", xy)
    xy_pairs = np.stack([xy[1:], xy[:-1]], 1)
    xy_pairs = np.stack([v for v in xy_pairs if shg.LineString(v).length > 1e-9])
    # logger.debug("xy_pairs:\n{}", xy_pairs)

    # logger.debug("xy_pairs.shape:\n{}", xy_pairs.shape)
    y_mean = xy_pairs[..., -1].mean(axis=1).round(9)
    # logger.debug("y_mean:\n{}", y_mean)
    x_delta = np.diff(xy_pairs[..., 0]).ravel().round(9)
    # logger.debug("x_delta:\n{}", x_delta)
    y_mean += 1e5 * (x_delta == 0)
    # logger.debug("y_mean:\n{}", y_mean)
    mask = (y_mean == y_mean.min()) * (x_delta != 0)
    # logger.debug("mask:\n{}", mask)
    # sys.exit()
    xy_bottom_line = xy_pairs[mask][0]
    # logger.debug("xy_bottom_line:\n{}", xy_bottom_line)
    xy_bottom_line = shg.LineString(xy_bottom_line)
    # logger.debug("xy_bottom_line:\n{}", xy_bottom_line)
    # energy_dissipation = cohesion * velocity * xy_bottom_line.length * np.cos(np.deg2rad(angle_friction))
    # logger.debug("energy_dissipation:\n{}", energy_dissipation)
    return xy_bottom_line


def get_line_inclination(line: shg.LineString, as_radians: bool = False) -> float:
    """Find angle w.r.t. vertical axis."""
    # logger.debug("line:\n{}", line)
    # logger.debug("line.coords:\n{}", list(line.coords))
    p1, p2 = line.coords
    # logger.debug("p1:\n{}", p1)
    # logger.debug("p2:\n{}", p2)
    line_height = abs(p1[1] - p2[1])
    line_angle_betta = np.arccos(line_height / line.length)
    if as_radians:
        return line_angle_betta
    else:
        return np.rad2deg(line_angle_betta)


def get_line_xy(line: shg.LineString | shg.MultiLineString) -> np.ndarray:
    match line:
        case shg.Point():
            return np.array([[line.x, line.y]])
        case shg.LineString():
            return np.array(list(line.coords))
        case shg.MultiLineString():
            return np.concatenate([get_line_xy(x) for x in line.geoms])
        case shg.GeometryCollection():
            return np.concatenate([get_line_xy(x) for x in line.geoms])
        case _:
            raise NotImplementedError(line)


def polar_to_cartesian(p0: shg.Point, theta_x: float, rx: float):
    return [
        p0.x + rx * np.cos(theta_x),
        p0.y - rx * np.sin(theta_x),
    ]


def get_elongated(line: shg.LineString, factor: float = 1.01):
    xy = get_line_xy(line)

    part1 = shaff.scale(shg.LineString(xy[-2:]), factor, factor,
                        origin=shg.Point(xy[-2]))
    part2 = shaff.scale(shg.LineString(xy[:2]), factor, factor, origin=shg.Point(xy[1]))
    # return union_all([part2, shg.LineString(xy[1:-1]), part1])
    return shg.LineString(np.concatenate([
        get_line_xy(part2)[:1],
        xy,
        get_line_xy(part1)[-1:],
    ]))


def split(geometry: shg.base.BaseGeometry, splitter: shg.LineString):
    return shops.split(geometry, get_elongated(splitter))


def get_fitted_points(xy: np.ndarray, degree: int, num_points: int = 100):
    # Fit a polynomial of given degree
    coeffs = np.polyfit(*xy.T, degree)

    # Generate x values using linspace
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    x_vals = np.linspace(x_min, x_max, num_points)

    # Generate the polynomial function from the coefficients
    poly_func = np.poly1d(coeffs)

    # Evaluate the polynomial at the x values to get y values
    y_vals = poly_func(x_vals)

    return np.stack([x_vals, y_vals], -1)
