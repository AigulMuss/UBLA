import copy
import sys
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Callable

import more_itertools as mit
import numpy as np
import optuna
import shapely.affinity as shaff
import shapely.geometry as shg
import shapely.ops as shops
from optuna import Trial
from plxscripting.easy import new_server

from launch_plaxis import port, password

big_num: float = 1e5


def get_slope_geometries(
        elev_bottom: float,
        elev_top: float,
        x_offset: float,
        angle: float,
        layer_boundary_elevations: list[float] = (),
        x_edge_value: float = None,
) -> tuple[shg.Polygon, list[shg.Polygon]]:
    # Create bottom and top elevation lines.
    surf_elev_line_bottom = shg.LineString([[-big_num, elev_bottom], [big_num, elev_bottom]])
    surf_elev_line_top = shg.LineString([[-big_num, elev_top], [big_num, elev_top]])

    # Create inclined slope line.
    slope_line = shaff.rotate(surf_elev_line_bottom, angle=angle, origin=[0, elev_bottom])
    slope_line = shaff.translate(slope_line, xoff=x_offset)

    # Find start and end points of a slope.
    slope_start: shg.Point = slope_line.intersection(surf_elev_line_bottom)
    slope_end: shg.Point = slope_line.intersection(surf_elev_line_top)

    # Create rectangular lower part of slope geometry.
    base_part = shg.box(0, 0, x_edge_value or (slope_end.x + x_offset), elev_bottom)

    # Create the upper part of slope geometry point-by-point.
    slope_poly = shg.Polygon([
        slope_start, slope_end,
        p1 := (shaff.translate(slope_end, x_offset)
               if x_edge_value is None else
               shg.Point(x_edge_value, slope_end.y)
               ),
        shaff.translate(p1, 0, -(elev_top - elev_bottom)),
    ])

    # Combine lower and upper parts to form the whole slope geometry.
    slope_geom: shg.Polygon = base_part.union(slope_poly)

    # Split slope geometry by soil layers.
    geometries = []
    current_geom = copy.deepcopy(slope_geom)
    for layer_elev in layer_boundary_elevations:
        splitter = shg.LineString([
            shg.Point(-big_num, layer_elev),
            shg.Point(big_num, layer_elev),
        ])
        geom1, geom2 = list(shops.split(current_geom, splitter).geoms)
        geometries.append(geom1)
        current_geom = geom2
    geometries.append(current_geom)

    return slope_geom, geometries


def get_water_level_line(
        slope_geom: shg.Polygon,
        water_level: float,
) -> shg.LineString:
    water_level_linestring = shg.LineString([
        shg.Point(-big_num, water_level),
        shg.Point(big_num, water_level),
    ])
    split_parts = list(shops.split(slope_geom, water_level_linestring).geoms)
    if len(split_parts) == 1:
        raise ValueError('Water level does not intersect the slope geometry.')
    water_geom = split_parts[0]

    segments = []
    for segment in map(np.array, mit.windowed(water_geom.boundary.coords, 2)):
        if np.subtract(*segment[:, 0]) == 0 or np.all(segment[:, 1] == 0):
            continue
        segments.extend(segment)
    return shg.LineString(segments)


@dataclass
class SoilMaterial:
    name: str
    mat: Any
    cost: float
    properties: dict = field(default_factory=dict)


@dataclass
class PlaxisPolygon:
    polygon: Any
    soil: Any


@dataclass
class SlopeGeomData:
    elev_bottom: float
    elev_top: float
    x_offset: float
    angle: float
    layer_boundary_elevations: list[float] = field(default_factory=list)
    x_edge_value: float = None
    slope_geom: shg.Polygon = None
    geometries: list[shg.Polygon] = field(default_factory=list)

    def __post_init__(self):
        # Create slope geometries
        self.slope_geom, self.geometries = get_slope_geometries(
            elev_bottom=self.elev_bottom,
            elev_top=self.elev_top,
            x_offset=self.x_offset,
            angle=self.angle,
            layer_boundary_elevations=self.layer_boundary_elevations,
            x_edge_value=self.x_edge_value,
        )

    def create_plaxis_polygons(self, gi) -> list[PlaxisPolygon]:
        return [PlaxisPolygon(*gi.polygon(*geom.boundary.coords)) for geom in self.geometries]


@dataclass()
class Case:
    slope_geom_data: SlopeGeomData
    water_level: float = None
    plaxis_polygons: list[PlaxisPolygon] = field(default_factory=list)

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

    def __post_init__(self):
        self.create_plaxis_project()
        self.plaxis_polygons = self.slope_geom_data.create_plaxis_polygons(self.gi)

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

    max_steps: int = 200
    safety_factor: float = None

    def simulate(self):
        gi = self.gi

        gi.gotostructures()
        gi.gotomesh()
        gi.mesh(0.06)
        gi.gotoflow()

        # Create phases.
        initial_phase = gi.phases[0]
        phase1 = gi.phase(initial_phase)
        phase2 = gi.phase(initial_phase)
        # Set global water level.

        if self.water_level is not None:
            # Create water level.
            water_level_line = get_water_level_line(
                slope_geom=self.slope_geom_data.slope_geom,
                water_level=self.water_level)
            water_level = gi.waterlevel(*water_level_line.coords)
            gi.setglobalwaterlevel(water_level, initial_phase)

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
        phase2.Deform.MaxSteps = self.max_steps

        # Calculate)
        gi.calculate()

        self.safety_factor = phase2.Reached.SumMsf.value
        return self


def get_cost_of_changes(case1: Case, case2: Case) -> float:
    layer_costs = []
    for layer_geom_1, layer_geom_2, (soil_mat, _) in zip_longest(
            case1.slope_geom_data.geometries,
            case2.slope_geom_data.geometries,
            case1.assignments.values()
    ):
        soil_mat: SoilMaterial
        layer_geom_1: shg.Polygon
        layer_geom_2: shg.Polygon | None
        if layer_geom_2 is None:
            delta_area = layer_geom_1.area
        else:
            delta_area = abs(layer_geom_1.area - layer_geom_2.area)
        cost = delta_area * soil_mat.cost
        layer_costs.append(cost)
    print(f"{layer_costs=}")
    delta_water_level = case1.water_level - (case2.water_level or 0)
    total_cost = sum(layer_costs) + delta_water_level * 1000
    print(f"{total_cost=}")
    return total_cost


def create_pareto_curve(case_initial: Case):
    slope_geom_data = case_initial.slope_geom_data

    def objective(trial: Trial):
        # elev_bottom_offset = trial.suggest_float('elev_bottom_offset', 0, 3, step=0.1)
        # elev_top_offset = trial.suggest_float('elev_top_offset', 0, 3, step=0.1)
        angle_offset = trial.suggest_float('angle_offset', 0, slope_geom_data.angle - 15, step=1)

        water_level = None
        if case_initial.water_level:
            water_level_offset = trial.suggest_float('water_level_offset',
                                                     0, case_initial.water_level, step=0.1)
            if case_initial.water_level != water_level_offset:
                water_level = case_initial.water_level - water_level_offset

        # elev_bottom = slope_geom_data.elev_bottom + elev_bottom_offset
        # elev_top = slope_geom_data.elev_top - elev_top_offset
        # layer_boundary_elevations = [x for x in slope_geom_data.layer_boundary_elevations if x < elev_top]
        case = Case(
            slope_geom_data=SlopeGeomData(
                elev_bottom=slope_geom_data.elev_bottom,
                elev_top=slope_geom_data.elev_top,
                x_offset=slope_geom_data.x_offset,
                angle=slope_geom_data.angle - angle_offset,
                layer_boundary_elevations=slope_geom_data.layer_boundary_elevations,
                x_edge_value=slope_geom_data.x_edge_value,
            ),
            water_level=water_level,
        )
        assign_materials(case)

        cost = get_cost_of_changes(case_initial, case)

        case.simulate()

        return cost, case.safety_factor

    case_study = optuna.create_study(
        storage='sqlite:///db.sqlite',
        study_name='slope_optimization',
        load_if_exists=True,
        directions=['minimize', 'maximize'],
    )
    case_study.optimize(objective, n_trials=10)


safety_factor_target = 2.0


def penalty_function_1(cost: float, safety_factor: float) -> float:
    penalty = cost
    sf_delta = safety_factor - safety_factor_target
    if sf_delta < 0:
        penalty += 10000 * sf_delta
    else:
        penalty += 5000 * sf_delta
    return penalty


def penalty_function_sf(safety_factor: float) -> float:
    sf_delta = safety_factor - safety_factor_target
    return sf_delta


def minimize_penalty(case_initial: Case, penalty_func: Callable):
    slope_geom_data = case_initial.slope_geom_data
    assert slope_geom_data.angle > 15, "Minimum slope angle is 15 degrees."

    def objective(trial: Trial):
        angle_offset = trial.suggest_float('angle_offset', 0, slope_geom_data.angle - 15, step=1)

        water_level = None
        if case_initial.water_level:
            water_level_offset = trial.suggest_float('water_level_offset',
                                                     0, case_initial.water_level, step=0.1)
            if case_initial.water_level != water_level_offset:
                water_level = case_initial.water_level - water_level_offset

        # elev_bottom = slope_geom_data.elev_bottom + elev_bottom_offset
        # elev_top = slope_geom_data.elev_top - elev_top_offset
        # layer_boundary_elevations = [x for x in slope_geom_data.layer_boundary_elevations if x < elev_top]
        case = Case(
            slope_geom_data=SlopeGeomData(
                elev_bottom=slope_geom_data.elev_bottom,
                elev_top=slope_geom_data.elev_top,
                x_offset=slope_geom_data.x_offset,
                angle=slope_geom_data.angle - angle_offset,
                layer_boundary_elevations=slope_geom_data.layer_boundary_elevations,
                x_edge_value=slope_geom_data.x_edge_value,
            ),
            water_level=water_level,
        )
        assign_materials(case)

        # cost = get_cost_of_changes(case_initial, case)

        case.simulate()
        # trial.set_user_attr('cost', cost)
        trial.set_user_attr('safety_factor', case.safety_factor)
        # return penalty_func(cost, case.safety_factor)
        return penalty_function_sf(case.safety_factor)

    case_study = optuna.create_study(
        storage='sqlite:///db.sqlite',
        study_name='reward_function_optimization2',
        load_if_exists=True,
        directions=['minimize'],

    )
    case_study.optimize(objective, n_trials=10)


def assign_materials(case: Case):
    for plaxis_poly, mat_props in zip(
            case.plaxis_polygons,
            [
                dict(
                    gammaUnsat=18,
                    gammaSat=20,
                    cref=20,
                    phi=25,
                    cost=100,
                ),
                dict(
                    gammaUnsat=20,
                    gammaSat=23,
                    cref=25,
                    phi=30,
                    cost=100,
                )
            ]):
        case.assign_soil_material(plaxis_poly, **mat_props)


def main():
    slope_geom_data = SlopeGeomData(
        elev_bottom=10,
        elev_top=20,
        x_offset=20,
        angle=45,
        layer_boundary_elevations=[15],
        x_edge_value=70,
    )
    case_initial = Case(
        slope_geom_data=slope_geom_data,
        water_level=16,
    )

    assign_materials(case_initial)
    case_initial.simulate()
    print(case_initial.safety_factor)

    # create_pareto_curve(case_initial)
    minimize_penalty(case_initial, penalty_function_1)


# done: Remove error when starting a new simulation.
# postpone: Run without GUI:
# check: Generate Plaxis commands -> Save to log file -> Run Plaxis with this log file.
# done: Multi-layer model/sim
# done: Add water level
# done: Fix water level
# done: Parametric study
# done: Optimize reward function.


if __name__ == '__main__':
    main()
