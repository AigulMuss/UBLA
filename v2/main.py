import itertools
import sys
from typing import Optional, Any

import numpy as np
import shapely.affinity as shaff
import shapely.geometry as shg
from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict
from shapely.plotting import plot_line, plot_points, plot_polygon

from v1.geom_utils import log_spiral, get_line_xy, split, get_xy, get_bottom_line

big_num = 1e5


def spiral_xy(p0_x: float, p0_y: float, theta_x: float, rx: float):
    return np.stack([
        p0_x + rx * np.cos(theta_x),
        p0_y + rx * np.sin(theta_x),
    ], axis=-1)


def get_slope_surface(
        elev_bottom: float = 5,
        elev_top: float = 15,
        slope_angle: float = 15,
        x_offset: float = 30,
        toe_offset: float = 5,
):
    slope = np.deg2rad(slope_angle)
    slope_y = np.array([elev_bottom, elev_top])
    slope_x = slope_y / np.tan(slope)
    slope_xy = np.stack([slope_x, slope_y], axis=-1) - [slope_x.min(), 0]
    xy = [[-x_offset, elev_bottom]] + slope_xy.tolist() + [
        [slope_xy[:, 0].max() + x_offset, elev_top]]
    return shaff.translate(shg.LineString(xy), toe_offset)


class Surface(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elev_bottom: float = 5
    elev_top: float = 15
    slope_angle: float = np.deg2rad(30)
    x_offset: float = 30

    linestring: Optional[shg.LineString] = None
    slope_line: Optional[shg.LineString] = None
    toe: Optional[shg.Point] = None
    crown_point: Optional[shg.Point] = None

    def model_post_init(self, __context: Any) -> None:
        self.linestring = get_slope_surface(
            elev_bottom=self.elev_bottom, elev_top=self.elev_top,
            slope_angle=np.rad2deg(self.slope_angle), x_offset=self.x_offset,
        )
        line_xy = get_line_xy(self.linestring)
        self.toe = shg.Point(*line_xy[1])
        self.crown_point = shg.Point(*line_xy[2])
        self.slope_line = shg.LineString([self.toe, self.crown_point])

    def plot(self):
        plot_line(self.linestring)
        plot_points(self.toe, color='red')
        plot_points(self.crown_point, color='green')
        return self

    def get_failure_curve(
            self,
            origin: shg.Point,
            start_offset: tuple = (0, 0),
            **kwargs,
    ) -> shg.LineString:
        return get_failure_curve(
            origin,
            shaff.translate(self.toe, *start_offset),
            self.linestring, **kwargs
        )

    def get_failure_polygon(self, origin: shg.Point, **kwargs) -> shg.Polygon:
        failure_curve = self.get_failure_curve(origin, **kwargs)
        if failure_curve.length < 1e-3:
            raise ValueError('Failure curve too small.')
        # plot_line(failure_curve)
        # plt.show()
        # sys.exit()
        failure_xy = get_line_xy(failure_curve)
        # logger.debug("failure_xy:\n{}", failure_xy)
        failure_blob = shg.Polygon(failure_xy.tolist() + [
            [failure_xy[-1, 0], self.crown_point.y + 1],
            [failure_xy[0, 0], self.crown_point.y + 1],
        ])

        out = split(failure_blob, self.linestring).geoms[0]
        logger.debug("failure_blob.area:\n{}", failure_blob.area)
        logger.debug("out.area:\n{}", out.area)
        if abs(failure_blob.area - out.area) < 1e-3:
            raise ValueError("Undefined failure curve.")
        return out


def plot_grid_points():
    grid_xy = np.stack(np.meshgrid(np.arange(-20, 0 + 1, 4), np.arange(30, 60, 4)),
                       axis=-1).reshape(-1, 2)
    # logger.debug("grid_xy.shape:\n{}", grid_xy.shape)
    points = [shg.Point(*xy) for xy in grid_xy]
    plot_points(points)
    return points


def get_failure_curve(
        origin: shg.Point,
        start: shg.Point,
        end: shg.LineString,
        phita: float,
        step: float = 1e-1,
) -> shg.LineString:
    # plot_points([start, origin], color='k')
    R0 = start.distance(origin)
    logger.debug("R0: {}", R0)
    theta_0 = np.arctan(
        get_slope_tangent(shg.LineString([origin, start])))
    if theta_0 > 0:
        theta_0 -= np.pi
    logger.debug("theta_0:\n{}", theta_0)
    logger.debug("theta_0 (deg):\n{}", np.rad2deg(theta_0))
    theta_x = np.arange(theta_0, np.deg2rad(180), step=step)
    logger.debug("theta_x:\n{}", theta_x)
    Rx = log_spiral(R0, theta_0, theta_x, phita=phita)
    failure_curve = shg.LineString(
        spiral_xy(origin.x, origin.y, theta_x, Rx)
        # .dot([[-1, 0], [0, 1]])  # flip horizontally
    )
    # failure_curve = shaff.rotate(failure_curve, 90, origin=origin)
    return split(failure_curve, end).geoms[0]


def get_slices(geometry: shg.Polygon) -> list[shg.Polygon]:
    geom_xy = get_xy(geometry)
    x_coors = np.sort(np.unique(geom_xy[:, 0]))
    x_mins = x_coors[:-1]
    x_maxs = x_coors[1:]
    y_min = -big_num
    y_max = big_num
    return [
        geometry.intersection(shg.box(xmin, y_min, xmax, y_max))
        for xmin, xmax in zip(x_mins, x_maxs)
    ]


colors = itertools.cycle(['red', 'blue', 'green', 'black', 'orange', 'purple'])


class FailureZone(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    origin: shg.Point
    polygon: shg.Polygon

    def get_slices(self):
        return get_slices(self.polygon)

    def plot(self):
        for slice_geom in self.get_slices():
            plot_polygon(slice_geom, color=next(colors))
        return self


def get_slope_tangent(
        line: shg.LineString,
        start: int = None,
        stop: int = None,
) -> float:
    xy = get_line_xy(line)[slice(start, stop)]
    return np.divide(*np.subtract(*xy[::-1])[::-1])


class SliceInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    geom: shg.Polygon
    bottom_line: shg.LineString
    alpha: float
    beta: float
    W: float

    def __hash__(self):
        return hash(self.geom.wkt)

    @classmethod
    def from_geometry(cls, slice_geom: shg.Polygon, gamma: float, **kwargs):
        bottom_line = get_bottom_line(slice_geom)
        alpha = np.arctan(get_slope_tangent(bottom_line))
        beta = bottom_line.length  # m
        W = gamma * slice_geom.area  # kN/m3 * m2 => kN/m of depth
        return cls(
            geom=slice_geom,
            bottom_line=bottom_line,
            alpha=alpha,
            beta=beta,
            W=W,
            sigma_n=gamma * np.sin(alpha),
            **kwargs)

    def get_Sm(self, factor: float):
        return self.beta / factor * (
                self.c_eff +
                (self.sigma_n - self.u_air) * np.tan(self.phita_prime) +
                (self.u_air - self.u_water) * np.tan(self.phita_prime_b)
        )

    def get_m_alpha(self, factor: float):
        return np.cos(self.alpha) + factor / (
                np.sin(self.alpha) * np.tan(self.phita_prime))

    def get_N(self, dx, factor):
        return (
                self.W -
                dx -
                (self.c_eff * self.beta * np.sin(self.alpha)) / factor +
                self.u_water * self.beta * np.sin(self.alpha) / factor * np.tan(
            self.phita_prime_b)
            # phita_prime_b or phita_b?
        ) / self.get_m_alpha(factor)

    def get_delta_E(self, Sm, dx):
        # return N * np.cos(self.alpha) * np.tan(self.alpha) - sm * np.cos(self.alpha)
        return (self.W - dx) * np.tan(self.alpha) - Sm / np.cos(self.alpha)

    def get_Fm_numerator(self, N_updated, R):
        return (
                self.c_eff * self.beta +
                (
                        N_updated - self.u_water * self.beta - self.u_air * self.beta) * np.tan(
            self.phita_prime)
        ) * R

    def get_Ff_numerator(self, N):
        return (
                self.c_eff * self.beta * np.cos(self.alpha) +
                (N - self.u_water * self.beta * np.tan(self.phita_prime_b) / np.tan(
                    self.phita_prime))
                * np.tan(self.phita_prime) * np.cos(self.alpha)
        )

    def get_Ff_denominator(self, N):
        return N * np.sin(self.alpha)

    def plot(self):
        plot_line(self.bottom_line, color=next(colors))
        return self


def calculate_Fm(
        slice_infos: list[SliceInfo],
        origin: shg.Point,
        Fs: float,
        delta_X_values: dict,
        lambda_value: float,
):
    Fm_numerators = []
    Fm_denominators = []
    normal_forces = {}
    for slice_info in slice_infos:
        # plot_line(bottom_line, color=next(colors))

        Sm = slice_info.get_Sm(Fs)
        dx = delta_X_values[slice_info]
        N = slice_info.get_N(dx, Fs)
        normal_forces[slice_info] = N

        # delta_E = N * np.cos(alpha) * np.tan(alpha) - Sm * np.cos(alpha)
        # get_delta_E = lambda dx, sm: [W - dx] * np.tan(alpha) - sm / np.cos(alpha)
        delta_E = slice_info.get_delta_E(Sm, dx)
        slice_delta_X = lambda_value * delta_E
        delta_X_values[slice_info] = slice_delta_X

        N_updated = slice_info.get_N(slice_delta_X, Fs)
        moment_arm = slice_info.geom.centroid.x - origin.x

        R = origin.distance(slice_info.bottom_line.centroid)
        Fm_numerator = slice_info.get_Fm_numerator(N_updated, R)
        Fm_denominator = slice_info.W * moment_arm

        Fm_numerators.append(Fm_numerator)
        Fm_denominators.append(Fm_denominator)

    Fm = sum(Fm_numerators) / sum(Fm_denominators)
    # logger.debug("Fm: {}", Fm)
    return Fm, normal_forces


def calculate_Ff(
        slice_infos: list[SliceInfo],
        Fm: float,
        normal_forces: dict,
        lambda_value: float,
):
    Ff_numerators = []
    Ff_denominators = []
    # lambda_values = []
    delta_X_values = {}
    for slice_info in slice_infos:
        Sm = slice_info.get_Sm(Fm)
        N = slice_info.get_N(0, Fm)
        # N = normal_forces[slice_info]
        delta_E = slice_info.get_delta_E(N, Sm)
        delta_X = lambda_value * delta_E
        delta_X_values[slice_info] = delta_X
        # slice_lambda_value = delta_X / delta_E
        Ff_numerators.append(slice_info.get_Ff_numerator(N))
        Ff_denominators.append(slice_info.get_Ff_denominator(N))
        # lambda_values.append(slice_lambda_value)
        # plot_grid_points()
    Ff = sum(Ff_numerators) / sum(Ff_denominators)
    # lambda_value = np.mean(lambda_values)
    # logger.debug("lambda_value: {}", lambda_value)
    logger.debug("Ff: {}", Ff)
    return Ff, delta_X_values


def main():
    # data=[]
    # data.append(123)
    # data.append([1,2,3])
    # data.append('buba')
    # logger.debug("data:\n{}", data)
    #
    # data={}
    # data['a']=123
    # data['b']=[1,2,3]
    # data['haha']='buba'
    # logger.debug("data:\n{}", data)
    # return
    surface = Surface(elev_bottom=10, elev_top=20, slope_angle=30, x_offset=50).plot()
    origin = shg.Point(0, 50)
    plot_points(origin)

    phita = np.deg2rad(23)  # radians
    gamma = 15  # kN/m3
    c_eff = 5  # kPa
    phita_prime = np.deg2rad(5)  # radians
    phita_prime_b = np.deg2rad(2.5)  # radians
    u_air = 0  # kPa

    # water_table = 10
    # H_unsat = surface.crown_point.y - water_table
    # u_water = 9.81  # kPa
    u_water = 9.81 * 10

    failure_zone = FailureZone(
        origin=origin,
        polygon=surface.get_failure_polygon(origin, phita=phita, step=0.1),
    ).plot()

    tolerance = 1e-6

    Fs = 1
    # sigma_n = 10
    # theta_b = np.deg2rad(5)  # radians | What is theta_b?
    lambda_value = 0

    # get_delta_X = lambda de: lambda_value * de

    slice_infos = [
        SliceInfo.from_geometry(
            slice_geom, gamma,
            c_eff=c_eff,  # sigma_n=sigma_n,
            u_air=u_air, u_water=u_water,
            phita_prime=phita_prime, phita_prime_b=phita_prime_b,
        ).plot()
        for slice_geom in failure_zone.get_slices()
    ]
    delta_X_values = {slice_info: 0 for slice_info in slice_infos}
    while True:
        Fm, normal_forces = calculate_Fm(slice_infos, origin, Fs, delta_X_values,
                                         lambda_value)
        # logger.debug("Fm: {}", Fm)
        new_Fs, delta_X_values = calculate_Ff(slice_infos, Fm, normal_forces,
                                              lambda_value)
        if abs(new_Fs - Fs) < tolerance:
            break
        Fs = new_Fs

    plt.show()

    pass


if __name__ == '__main__':
    main()
