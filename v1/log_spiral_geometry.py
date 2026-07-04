import functools
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Callable, Optional, Any

import optuna
import pandas as pd
import shapely
import shapely.ops as shops
import shapely.affinity as shaff
import itertools

import more_itertools as mit
import matplotlib.pyplot as plt
import shapely.geometry as shg
import numpy as np
from box import Box
from loguru import logger
from optuna import Trial
from optuna.trial import FixedTrial
from plxscripting.easy import new_server
from shapely.plotting import plot_polygon, plot_line, plot_points

from launch_plaxis import port, password
from v0.main import PlaxisPolygon, SoilMaterial

from v1.geom_utils import log_spiral, big_num, get_xy, get_slices_by_points, get_bottom_line, get_line_inclination, \
    get_line_xy, polar_to_cartesian, get_elongated, split, get_fitted_points


def get_pairwise_fitted_points(xy: np.ndarray, degree: int | list[int], num_points: int = 100):
    points = []
    if isinstance(degree, int):
        degree = [degree]
    degrees = itertools.cycle(degree)
    for idx in range(xy.shape[0] - 1):
        segment_points = get_fitted_points(xy=xy[idx:idx + 2], degree=next(degrees), num_points=num_points)
        points.append(segment_points)
    return np.concatenate(points)


is_dev = True


def get_xy_coors(
        func: Callable,
        xs: np.ndarray,
        proc: Callable = lambda x, y: [x, y],
) -> np.ndarray:
    return np.array([proc(x, y) for x, y in zip(xs, func(xs))])


def get_log_spiral_curve(
        origin: shg.Point,
        theta_min: float,
        phita: float,
        start_surface: shg.LineString,
        end_surface: shg.LineString = None,
        num_points: int = 100,
        debug: bool = False,

) -> shg.LineString:
    if debug:
        plot_points(origin)
    ray_min: shg.LineString = shaff.rotate(
        geom=shg.LineString([origin, shaff.translate(origin, big_num)]),
        angle=-theta_min,
        origin=origin,
        use_radians=True,
    )

    point_min = ray_min.intersection(start_surface)
    segment_min = shg.LineString([origin, point_min])
    if debug:
        plot_line(segment_min)
    func = np.vectorize(partial(log_spiral, segment_min.length, theta_min, phita=phita))
    thetas = np.linspace(theta_min, np.pi * 2 / 2, num_points)
    proc_func = partial(polar_to_cartesian, origin)
    log_spiral_curve = shg.LineString(get_xy_coors(func=func, xs=thetas, proc=proc_func))
    if debug:
        plot_line(log_spiral_curve)
    if end_surface is None:
        point_max = log_spiral_curve.intersection(start_surface).geoms[0]
    else:
        result_intersection = log_spiral_curve.intersection(end_surface)
        match result_intersection:
            case shg.MultiPoint():
                point_max = result_intersection.geoms[-1]
            case shg.LineString():
                if is_dev:
                    plot_line(log_spiral_curve)
                    plot_line(end_surface)
                    plt.show()
                raise NotImplementedError()
            case shg.Point():
                point_max = result_intersection
            case _:
                raise NotImplementedError(result_intersection)
    ray_max = shaff.scale(shg.LineString([origin, point_max]), 2, 2, origin=origin)
    # ray_max = shg.LineString([origin, point_max])
    if debug:
        plot_line(ray_max)
    geoms = split(log_spiral_curve, ray_max).geoms
    # if debug:
    #     for geom in geoms:
    #         plot_line(geom)
    return geoms[0]


def elongation():
    line = shg.LineString([[0, 0], [5, 5]])
    plot_line(line)
    plot_line(shaff.translate(get_elongated(line), 0.5), color='red')
    plt.show()
    return


@dataclass
class GeometryComposer:
    objects: dict[str, shg.LineString | shg.Point | shg.Polygon] = field(default_factory=dict)

    def __getitem__(self, item):
        return self.objects[item]

    def __setitem__(self, key, value):
        self.objects[key] = value

    def split(self, key: str, by: str, index: int):
        self[key] = split(self[key], self[by]).geoms[index]
        return self

    def join(self, *keys):
        return shg.Polygon(sum([list(self[k].coords) for k in keys], start=[]))

    def compose(self):
        return self

    def xy(self, key: str):
        return get_line_xy(self[key])

    def segments(self, key: str):
        return get_segments(self[key])


def get_segments(line: shg.LineString) -> list[shg.LineString]:
    return [shg.LineString(pair) for pair in mit.windowed(get_line_xy(line), 2)]


@dataclass
class RotationalTranslationalComposer(GeometryComposer):
    def compose(self):
        self.split('curve_2', 'interlayer', -1)
        self.split('surface', 'curve_1', 0)
        self.split('surface', 'curve_2', -1)
        self.split('interlayer', 'curve_1', 0)
        self.split('interlayer', 'curve_2', -1)
        geometry = shapely.make_valid(self.join('surface', 'curve_1', 'interlayer', 'curve_2'))
        match geometry:
            case shg.Polygon():
                pass
            case shg.GeometryCollection():
                geometry = mit.only([x for x in geometry.geoms if isinstance(x, shg.Polygon)])
            case shg.MultiPolygon():
                # for p, color in zip(geometry.geoms, ['red', 'green', 'blue']):
                #     plot_polygon(p, color=color)
                #     logger.debug("p.area:\n{}", p.area)

                geometry = mit.only([x for x in geometry.geoms if x.area > 1e-9])
            case _:
                # for p, color in zip(geometry.geoms, ['red', 'green', 'blue']):
                #     plot_polygon(p, color=color)
                #     logger.debug("p.area:\n{}", p.area)
                # for k in ['surface', 'curve_1', 'interlayer', 'curve_2']:
                #     plot_line(self[k])
                res = functools.reduce(
                    shapely.line_merge,
                    [self[k] for k in
                     ['surface', 'curve_1', 'interlayer', 'curve_2']])
                # res=shapely.line_merge([get_elongated(self[k]) for k in ['surface', 'curve_1', 'interlayer', 'curve_2']])
                if is_dev:
                    logger.debug("res:\n{}", res)
                    plot_line(res)
                    plt.show()

                raise NotImplementedError(geometry)
        self['geometry'] = geometry
        return self

    @classmethod
    def create(
            cls,
            phita: float,
            surface: shg.LineString,
            interlayer: shg.LineString,
            origin1: shg.Point,
            origin2: shg.Point,
            theta_min_1: float,
            theta_min_2: float,
            debug: bool = False,
    ):
        composer = cls()
        composer['surface'] = surface
        composer['interlayer'] = interlayer

        if debug:
            plot_line(surface)
            plot_line(interlayer)
        # plt.show()
        composer['curve_1'] = get_log_spiral_curve(
            origin=origin1,
            theta_min=theta_min_1,
            phita=phita,
            start_surface=composer['surface'],
            end_surface=composer['interlayer'],
            debug=debug
        )
        if debug:
            plot_line(composer['curve_1'])

        # plot_line(curve_1, color='orange')
        # slope_surface = split(slope_surface, curve_1).geoms[0]
        # plot_line(slope_surface,color='red')
        #
        # plot_line(slope_surface)
        # return

        # plot_line(curve_1, color='orange')

        composer['curve_2'] = get_log_spiral_curve(
            origin=origin2,
            theta_min=theta_min_2,
            phita=phita,
            start_surface=composer['interlayer'],
            end_surface=composer['surface'],
            debug=debug,
        )
        if debug:
            plot_line(composer['curve_2'])
        return composer.compose()

    parts: list[shg.Polygon] = field(default_factory=list)
    boundaries: list[shg.LineString] = field(default_factory=list)

    def create_parts(self, angle_offset: float = 0, _rotate_factor: int = 1):
        xy_interlayer = self.xy('interlayer')
        geom = self['geometry']
        parts = []
        boundaries = []
        for idx, xy_point in enumerate([
            xy_interlayer[-1],
            xy_interlayer[0],
        ]):
            splitter = shaff.rotate(
                shaff.scale(self['interlayer'], 1e3, 1e3,
                            origin=self['interlayer'].centroid),
                90 - angle_offset, shg.Point(*xy_point))
            # plot_polygon(self['geometry'])
            # plot_line(splitter.intersection(self['geometry']), color='red')
            # plt.show()
            geom_part = list(filter(lambda g: g.area > 1e-9, shops.split(geom, splitter).geoms))[::_rotate_factor]
            if len(geom_part) == 2:
                geom, part = geom_part
            elif _rotate_factor == -1:
                raise NotImplementedError()
            else:
                return self.create_parts(angle_offset, _rotate_factor=-1)
            parts.append(part)
            boundary = splitter.intersection(self['geometry'])
            # for line, color in zip(boundary.geoms, ['red', 'green', 'blue']):
            #     plot_line(line, color=color)
            # plt.show()

            if isinstance(boundary, shg.MultiLineString):
                boundary = mit.only((x for x in boundary.geoms if x.length > 1e-9))
            assert isinstance(boundary, shg.LineString)
            boundaries.append(boundary)
        parts.append(geom)
        self.parts = parts
        self.boundaries = boundaries
        return self

    def plot(self, show: bool = True):
        # plot_polygon(self['geometry'], color='black', alpha=0.1)
        for part, color in zip(self.parts, ['purple', 'green', 'brown']):
            plot_polygon(part, color=color)

        for boundary, color in zip(self.boundaries, ['gray', 'orange']):
            plot_line(boundary, lw=5, color=color)
        if show:
            plt.show()

    def calculate_energy_dissipation(self, soil_1: dict, soil_2: dict, vs: dict):

        d_interlayer = get_energy_dissipation(
            self['interlayer'],
            cohesion=soil_2['cohesion'],
            phita=soil_2['phita'],
            velocity=vs['b'],
        )
        # logger.debug("d_interlayer:\n{}", d_interlayer)

        d_curve_1 = get_energy_dissipation(
            self['curve_1'],
            cohesion=soil_1['cohesion'],
            phita=soil_1['phita'],
            velocity=vs['a'],
        )
        # logger.debug("d_curve_1:\n{}", d_curve_1)

        d_curve_2 = get_energy_dissipation(
            self['curve_2'],
            cohesion=soil_1['cohesion'],
            phita=soil_1['phita'],
            velocity=vs['c'],
        )
        # logger.debug("d_curve_2:\n{}", d_curve_2)
        d_ab = get_energy_dissipation(
            self.boundaries[0],
            cohesion=soil_1['cohesion'],
            phita=soil_1['phita'],
            velocity=vs['ab'],
        )
        # logger.debug("d_ab:\n{}", d_ab)
        d_bc = get_energy_dissipation(
            self.boundaries[1],
            cohesion=soil_1['cohesion'],
            phita=soil_1['phita'],
            velocity=vs['bc'],
        )
        # logger.debug("d_bc:\n{}", d_bc)

        d_total = d_ab + d_interlayer + d_bc + d_curve_1 + d_curve_2
        # logger.debug("d_total:\n{}", d_total)
        return d_total

    def calculate_external_work(self, soil_1: dict, soil_2: dict, vs: dict):
        phitas = {
            'a': soil_1['phita'],
            'b': soil_2['phita'],
            'c': soil_1['phita'],
        }
        return np.array([
            get_external_work(
                part,
                phita=phitas[name],
                gamma=soil_1['gamma'],
                velocity=vs[name])
            for name, part in zip(['a', 'b', 'c'], self.parts)
        ]).sum()

    si: Any = None
    gi: Any = None

    def create_plaxis_project(self):
        # Start Plaxis server (for inputs)
        self.si, self.gi = new_server(
            'localhost', port,
            password=password, timeout=20
        )
        # Open new Plaxis project.
        self.si.new()
        self.gi.gotosoil()
        # gi.SoilContour.initializerectangular(0, 0, 100, 30)
        self.gi.borehole(0)
        # gi.soillayer(0)
        self.gi.setproperties("ModelType", "Axisymmetry")
        return self

    plaxis_polygons: list[PlaxisPolygon] = field(default_factory=list)

    def get_slope_geometries(self, debug: bool = False) -> tuple[shg.Polygon, list[shg.Polygon]]:
        surface = self['surface']
        interlayer = self['interlayer']

        bounds = shg.box(*shapely.total_bounds(shg.MultiLineString([surface, interlayer])))
        plaxis_bounds = bounds.buffer(distance=10, join_style='mitre')
        plaxis_surface = get_elongated(surface, 10)
        plaxis_interlayer = get_elongated(interlayer, 10)

        slope_geom, _ = shops.split(plaxis_bounds, plaxis_surface).geoms
        if not debug:
            xoff = -get_xy(slope_geom)[:, 0].min()
            slope_geom = shaff.translate(slope_geom, xoff=xoff)
            plaxis_interlayer = shaff.translate(plaxis_interlayer, xoff=xoff)
        geom_bottom, geom_top = shops.split(slope_geom, plaxis_interlayer).geoms

        if debug:
            plot_line(surface)
            plot_line(interlayer)
            plot_polygon(bounds)
            plot_polygon(geom_top, color='green')
            plot_polygon(geom_bottom, color='orange')
            plt.show()
        return slope_geom, [geom_bottom, geom_top]

    assignments: dict[str, tuple[SoilMaterial, PlaxisPolygon]] = field(default_factory=dict)

    def assign_soil_material(
            self,
            plaxis_poly: PlaxisPolygon,
            gammaUnsat: float,
            gammaSat: float,
            cref: float,
            phi: float,
            **kwargs,
    ) -> None:
        material_name = f'Mat_{len(self.assignments) + 1}'
        mat = self.gi.soilmat()
        mat_props = dict(
            MaterialName=material_name,
            SoilModel=2,
            Eref=210e6,
            nu=0.2,
            gammaUnsat=gammaUnsat,
            gammaSat=gammaSat,
            cref=cref,
            phi=phi,
        )
        mat.setproperties(*mat_props.items())
        plaxis_poly.soil.Material = mat
        soilmat = SoilMaterial(
            name=material_name,
            mat=mat,
            properties=mat_props,
            **kwargs
        )
        self.assignments[material_name] = soilmat, plaxis_poly

    def simulate(self, max_steps: int):
        gi = self.gi

        gi.gotostructures()
        gi.gotomesh()
        gi.mesh(0.02)
        gi.gotoflow()

        # Create phases.
        initial_phase = gi.phases[0]
        phase1 = gi.phase(initial_phase)
        phase2 = gi.phase(initial_phase)
        water_level_line = get_bottom_line(self.slope_geom)
        water_level = gi.waterlevel(*water_level_line.coords)
        gi.setglobalwaterlevel(water_level, initial_phase)
        # if self.water_level is not None:
        #     # Create water level.
        #     water_level_line = get_water_level_line(
        #         slope_geom=self.slope_geom_data.slope_geom,
        #         water_level=self.water_level)
        #     water_level = gi.waterlevel(*water_level_line.coords)
        #     gi.setglobalwaterlevel(water_level, initial_phase)

        gi.gotostages()

        # Activate all polygons in all phases.
        for phase in [initial_phase, phase1, phase2]:
            for plaxis_poly in self.plaxis_polygons:
                gi.activate(plaxis_poly.polygon, phase)

        initial_phase.setproperties(*dict(
            DeformCalcType="Gravity loading",
        ).items())
        phase1.setproperties(*dict(
            Identification="Deformations",
        ).items())

        phase2.setproperties(*dict(
            DeformCalcType="Safety",
            Identification="Calculate SF",
        ).items())
        phase2.Deform.UseDefaultIterationParams = False
        phase2.Deform.MaxSteps = max_steps

        # Calculate)
        gi.calculate()

        return phase2.Reached.SumMsf.value

    slope_geom: Optional[shg.Polygon] = None

    def run_plaxis_simulation(self, soil_1: dict, soil_2: dict):
        self.slope_geom, geometries, = self.get_slope_geometries()
        # plot_polygon(slope_geom)
        # plt.show()
        # sys.exit()

        # slope_geom_data = SlopeGeomData(
        #     elev_bottom=10,
        #     elev_top=20,
        #     x_offset=20,
        #     angle=45,
        #     layer_boundary_elevations=[15],
        #     x_edge_value=70,
        # )
        self.plaxis_polygons = [PlaxisPolygon(*self.gi.polygon(*geom.boundary.coords)) for geom in geometries]
        bottom_poly, top_poly = self.plaxis_polygons
        self.assign_soil_material(
            bottom_poly,
            gammaUnsat=soil_1['gamma'],
            gammaSat=soil_1['gamma'] + 3,
            cref=soil_1['cohesion'],
            phi=np.rad2deg(soil_1['phita']),
            cost=100,
        )
        self.assign_soil_material(
            top_poly,
            gammaUnsat=soil_2['gamma'],
            gammaSat=soil_2['gamma'] + 3,
            cref=soil_2['cohesion'],
            phi=np.rad2deg(soil_2['phita']),
            cost=100,
        )
        return self.simulate(max_steps=200)


def get_energy_dissipation(
        line: shg.LineString,
        cohesion: float,
        phita: float,
        velocity: float,
) -> float:
    energy_dissipation = [
        cohesion *
        velocity *
        segment_line.length *
        np.cos(phita)
        for segment_line in get_segments(line)
    ]
    return np.array(energy_dissipation).sum()


def get_external_work(
        geom: shg.Polygon,
        phita: float,
        gamma: float,
        velocity: float,
) -> float:
    external_work = []
    _colors = iter(colors)
    for slice_geom in get_slices_by_points(geom):
        plot_polygon(slice_geom, color=next(_colors))
        try:
            inclination = np.pi / 2 - get_line_inclination(get_bottom_line(slice_geom), as_radians=True)
        except IndexError:
            logger.debug("slice_geom:\n{}", slice_geom)
            logger.debug("slice_geom.area:\n{}", slice_geom.area)
            plot_polygon(slice_geom)
            plt.show()
            raise

        external_work.append(
            gamma *
            velocity *
            slice_geom.area *
            abs(np.sin(inclination - phita))
        )
    return np.array(external_work).sum()


colors = ['purple', 'green', 'brown', 'black', 'yellow', 'magenta'] * 1000


def get_velocities(soil_1: dict, soil_2: dict):
    vb = 1
    va = vb * np.cos(soil_1['phita'] + 2 * soil_2['phita']) / np.cos(soil_1['phita'] + 2 * soil_2['phita'])
    vc = vb * np.cos(soil_2['phita']) / np.cos(soil_1['phita'])
    vr_bc = vb * np.sin(soil_1['phita'] - soil_2['phita']) / np.cos(soil_1['phita'])
    vr_ab = vb * np.sin(2 * soil_2['phita']) / np.cos(soil_1['phita'])

    return dict(a=va, ab=vr_ab, b=vb, c=vc, bc=vr_bc)


def get_slope_surface(
        elev_bottom: float = 5,
        elev_top: float = 15,
        slope_angle: float = 15,
        x_offset: float = 30,
):
    slope = np.deg2rad(slope_angle)
    slope_y = np.array([elev_bottom, elev_top])
    slope_x = slope_y / np.tan(slope)
    slope_xy = np.stack([slope_x, slope_y], axis=-1) - [slope_x.min(), 0]
    xy = [[-x_offset, elev_bottom]] + slope_xy.tolist() + [[slope_xy[:, 1].max() + x_offset, elev_top]]
    return shg.LineString(xy)


def get_interlayer(
        depth: float,
        angle: float,
        ref_point: shg.Point,
        use_radians: bool = False,
        x_offset=big_num,
) -> shg.LineString:
    p1 = shaff.translate(ref_point, yoff=-depth)
    line = shg.LineString([shaff.translate(p1, xoff=-x_offset), shaff.translate(p1, xoff=x_offset)])
    return shaff.rotate(line, angle, origin=p1, use_radians=use_radians)


def get_inclination_angle(line: shg.LineString) -> float:
    return np.divide(*np.subtract(*get_line_xy(line)[::-1])[::-1])


def get_theta_min_range(
        surface: shg.LineString,
        interlayer: shg.LineString,
        origin: shg.Point,
        debug: bool = False,
):
    if debug:
        plot_line(surface)
        plot_line(interlayer)
    intersection_point = surface.intersection(interlayer)
    # logger.debug("intersection_point:\n{}", intersection_point)
    if intersection_point.is_empty:
        xy_max = get_line_xy(surface).max(0)
        start = get_inclination_angle(shg.LineString([origin, xy_max]))
    else:
        start = get_inclination_angle(shg.LineString([origin, intersection_point]))
        # logger.debug("start:\n{}", start)

    radius = shapely.distance(origin, surface)
    circle = origin.buffer(radius + 1e-2, quad_segs=32)
    result = surface.intersection(circle)
    assert not result.is_empty
    match result:
        case shg.Point() as end_point:
            pass
        case shg.MultiPoint():
            end_point = result.geoms[-1]
        case shg.LineString():
            end_point = shg.Point(get_line_xy(result)[-1])
        case _:
            if is_dev:
                plot_polygon(circle)
                plot_line(surface)
                plot_line(interlayer)
                plot_line(result)
                plt.show()
            raise NotImplementedError(result)
    end = get_inclination_angle(shg.LineString([origin, end_point]))
    if debug:
        plot_polygon(circle)
        plot_line(surface)
        plot_line(interlayer)
        plot_line(result)
        plot_points(end_point, color='red')
        plt.show()
    return -np.array([start, end])


def objective_function(trial: Trial | FixedTrial):
    surface = get_slope_surface(slope_angle=45, x_offset=100, elev_bottom=5, elev_top=15)
    interlayer = get_interlayer(4, 30, shg.Point(0, 5), x_offset=100)

    soil_1 = dict(
        gamma=18,
        cohesion=20,
        phita=np.deg2rad(30),
    )
    soil_2 = dict(
        gamma=20,
        cohesion=23,
        phita=np.deg2rad(20),
    )
    vs = get_velocities(soil_1, soil_2)
    origin1_x = trial.suggest_float('origin1_x', -10, 10)
    origin1_y = trial.suggest_float('origin1_y', 8, 30)
    origin1 = shg.Point(origin1_x, origin1_y)
    # origin1 = shg.Point(5, 20)

    origin2_x = trial.suggest_float('origin2_x', -5, 20)
    origin2_y = trial.suggest_float('origin2_y', 5, 30)
    origin2 = shg.Point(origin2_x, origin2_y)
    # origin2 = shg.Point(8, 12)
    theta_min_1_start, theta_min_1_end = get_theta_min_range(surface, interlayer, origin=origin1)
    theta_min_1 = trial.suggest_float('theta_min_1', theta_min_1_start, theta_min_1_end)
    theta_min_2 = trial.suggest_float('theta_min_2', np.deg2rad(30), np.deg2rad(70))
    # logger.debug("np.rad2deg(theta_min_1):\n{}", np.rad2deg(theta_min_1))
    try:
        composer = RotationalTranslationalComposer.create(
            phita=soil_1['phita'],
            surface=surface,
            interlayer=interlayer,
            origin1=origin1,
            origin2=origin2,
            theta_min_1=theta_min_1,
            theta_min_2=theta_min_2,
            debug=False,
        )
        # plot_polygon(composer['geometry'])
        # plt.show()
        angle_offset = trial.suggest_float('angle_offset', -20, 30)
        composer.create_parts(angle_offset=angle_offset)
        # composer.plot(show=True)
        # return
        d_total = composer.calculate_energy_dissipation(soil_1, soil_2, vs)

        external_work = composer.calculate_external_work(soil_1, soil_2, vs)
        if is_dev:
            logger.debug("d_total:\n{}", d_total)
            logger.debug("external_work:\n{}", external_work)

        # for part in composer.parts:

        # for slice_geom, color in zip(slice_geoms, colors):
        #     plot_polygon(slice_geom, color=color)
        # plt.show()
        # plt.close()
        # for segment, color in zip(segments,
        #                           ['purple', 'green', 'brown', 'black', 'yellow', 'magenta'] * 10):
        #     plot_line(segment, lw=5, color=color)
        # plt.show()
        # composer.plot()

        trial.set_user_attr('d_total', d_total)
        trial.set_user_attr('external_work', external_work)
        trial.set_user_attr('composer', composer)
        trial.set_user_attr('soil_1', soil_1)
        trial.set_user_attr('soil_2', soil_2)
        large_surface_penalty = np.abs(np.mean([d_total, external_work]) - 500) / 100
        equilibrium_penalty = abs(d_total - external_work) / min(d_total, external_work)
        # case 1: 500, 1000 -> 500
        return equilibrium_penalty + large_surface_penalty
        # return abs(d_total - external_work)
    except:
        return 10000


def simulate_in_plaxis(fixed_trial: Trial | FixedTrial):
    if isinstance(fixed_trial, FixedTrial):
        objective_function(fixed_trial)

    composer: RotationalTranslationalComposer = fixed_trial.user_attrs['composer']
    failure_surface_xy = np.concatenate([
        get_line_xy(composer[key]) for key in
        ['curve_1', 'interlayer', 'curve_2']
    ])
    logger.debug("failure_surface_xy:\n{}", failure_surface_xy)
    pd.DataFrame(failure_surface_xy, columns=['x', 'y']).to_csv(Path('failure_surface.csv'), index=False)
    # plt.plot(failure_surface_xy[:, 0], failure_surface_xy[:, 1], c='red')
    # plt.show()
    # return

    composer.plot()
    soil_1 = fixed_trial.user_attrs['soil_1']
    soil_2 = fixed_trial.user_attrs['soil_2']
    composer.create_plaxis_project()
    sf = composer.run_plaxis_simulation(soil_1, soil_2)
    print(f"sf: {sf}")


def main():
    simulate_in_plaxis(fixed_trial=FixedTrial(
        {
            "origin1_x": 1.8766483822016402,
            "origin1_y": 26.321246786008853,
            "origin2_x": 8.947390991060425,
            "origin2_y": 16.48670634125605,
            "theta_min_1": 0.5460808793050713,
            "theta_min_2": 1.0077924700413738,
            "angle_offset": -14.389482603034548
        }
    ))
    return
    case_study = optuna.create_study(
        # storage='sqlite:///db.sqlite',
        # study_name='reward_function_optimization2',
        # load_if_exists=True,
        # directions=['minimize'],
        sampler=optuna.samplers.TPESampler(n_startup_trials=200),
    )
    case_study.optimize(objective_function, n_trials=500, n_jobs=16,
                        catch=(NotImplementedError, TypeError, ValueError, AttributeError, AssertionError))
    logger.debug("case_study.best_trial:\n{}", case_study.best_trial.value)
    logger.debug("case_study.best_trial.params:\n{}", pformat(case_study.best_trial.params))
    logger.debug("case_study.best_trial.user_attrs:\n{}", case_study.best_trial.user_attrs)
    Box(case_study.best_trial.params).to_json('params.json')

    user_attrs = dict(case_study.best_trial.user_attrs)
    user_attrs.pop('composer')
    Box(user_attrs).to_json('user_attrs.json')

    simulate_in_plaxis(fixed_trial=case_study.best_trial)
    pass


if __name__ == '__main__':
    main()
