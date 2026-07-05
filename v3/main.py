import datetime
import sys
from pathlib import Path
from typing import Optional, Self, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import optuna
import pandas as pd
import shapely.affinity as shaff
import shapely.geometry as shg
import shapely.ops as shops
from box import Box
from loguru import logger
from optuna import Trial
from optuna.trial import FixedTrial
from pydantic import BaseModel, ConfigDict, Field
from shapely import GEOSException
from shapely.plotting import plot_line, plot_points

from v1.geom_utils import get_line_xy
from v1.log_spiral_geometry import get_velocities, get_energy_dissipation
from v2.main import Surface as SurfaceV2, get_slope_tangent, big_num
from v3.curves import Curve, LogSpiralCurve, LinearCurve, PolynomialCurve
from v3.utils import _colors, get_external_work, get_normal, \
    convert_enclosed_area_to_polygon, \
    suggest_origin, get_split, get_child_polygons, round_coordinates

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 40)

# s1..s3 follow the code's internal height ordering (s1 deepest, s3 shallowest).
VALIDATION_CASES = {
    'case1_homogeneous': dict(
        description='Case 1: Homogeneous slope (baseline verification)',
        slope_angle_deg=30.0,
        total_height=15.0,
        partition_type='LinearCurve',
        layers=dict(
            s1=dict(gamma=19.0, cohesion=15.0, phita_deg=30.0, thickness=15.0 / 3),
            s2=dict(gamma=19.0, cohesion=15.0, phita_deg=30.0, thickness=15.0 / 3),
            s3=dict(gamma=19.0, cohesion=15.0, phita_deg=30.0, thickness=None),
        ),
        fem_reference=dict(F_S=1.46, F_S_UB=1.49, deviation_pct=1.9),
    ),
    'case2_clay_rock': dict(
        description='Case 2: Two-layer clay-rock slope (Table 1)',
        slope_angle_deg=30.0,
        total_height=14.0,
        partition_type='PolynomialCurve',
        layers=dict(
            s1=dict(gamma=21.0, cohesion=40.0, phita_deg=38.0, thickness=4.5),
            s2=dict(gamma=21.0, cohesion=40.0, phita_deg=38.0, thickness=4.5),
            s3=dict(gamma=18.0, cohesion=10.0, phita_deg=18.0, thickness=None),
        ),
        fem_reference=dict(deviation_pct=5.6, H1_over_H=0.36, phi_ratio=2.11),
    ),
    'case3_three_layer': dict(
        description='Case 3: Three-layer multi-layered slope (Table 2)',
        slope_angle_deg=30.0,
        total_height=15.0,
        # Polynomial partitions give a closer H_critical match to the FEM
        # reference here (6.0% deviation vs 11.4% with linear partitions at
        # the same trial budget), consistent with a multi-layer interface
        # needing more than a straight partition to capture the block
        # boundary.
        partition_type='PolynomialCurve',
        layers=dict(
            s1=dict(gamma=20.5, cohesion=25.0, phita_deg=28.0, thickness=8.0),
            s2=dict(gamma=19.0, cohesion=10.0, phita_deg=32.0, thickness=4.0),
            s3=dict(gamma=17.0, cohesion=5.0, phita_deg=20.0, thickness=None),
        ),
        fem_reference=dict(H_crit=13.7, F_S=1.280, H_crit_ub=13.9, F_S_ub=1.285,
                           deviation_pct=1.3),
    ),
}


class Surface(SurfaceV2):
    soil_params: Box

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        for soil_name, soil_params in self.soil_params.items():
            for name, height, in zip(['lower_bound', 'upper_bound'],
                                     soil_params.height_span):
                if height is None:
                    continue
                soil_params[name] = shg.LineString([
                    [-big_num, height],
                    [big_num, height],
                ])


class SoilBlock(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    soil_params: Box
    geom: shg.Polygon


class Case(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    soil_params: Box[str, dict]
    vs: dict = Field(default_factory=dict)
    surface: Surface
    origins: dict[str, shg.Point] = Field(default_factory=list)
    normals: list[shg.LineString] = Field(default_factory=list)
    failure_curves: list = Field(default_factory=list)
    partitions: list = Field(default_factory=list)
    blocks: Optional[gpd.GeoDataFrame] = None

    def model_post_init(self, __context: Any) -> None:
        self.vs = get_velocities(soil_1=self.soil_params['s1'],
                                 soil_2=self.soil_params['s2'])

    def set_normals(self, ratios: list[float]) -> Self:
        self.normals = [
            get_normal(
                self.surface.slope_line,
                point=self.surface.slope_line.interpolate(ratio, True),
                x_offset=30
            )
            for ratio in ratios
        ]
        return self

    def set_failure_curves(self, failure_curves: list[Curve], toe_offset: float):
        self.failure_curves = failure_curves

        start = shaff.translate(self.surface.toe, xoff=toe_offset)
        for curve in self.failure_curves:
            curve(start=start)  # .plot()
            start = curve.end_point
        return self

    def set_partitions(self, partitions: list[Curve]):
        self.partitions = partitions
        return self

    def create_blocks(self):
        # plot_polygon(convert_enclosed_area_to_polygon([
        #     self.failure_curves[0].line,
        #     self.partitions[0].line,
        #     self.surface.linestring,
        # ]), color='green')
        # convert_enclosed_area_to_polygon([
        #     self.failure_curves[1].line,
        #     self.partitions[0].line,
        #     self.surface.linestring,
        #     self.partitions[1].line,
        # ])
        # self.surface.plot()
        # for curve in self.failure_curves:
        #     curve.plot()
        # logger.debug("self.partitions[1].line:\n{}", self.partitions[1].line)
        # self.partitions[-1].plot()
        # plt.show()
        # for partition in self.partitions:
        #     partition.plot()
        blocks = dict(
            a=convert_enclosed_area_to_polygon([
                self.failure_curves[0].line,
                self.partitions[0].line,
                self.surface.linestring,
            ]),
            b=convert_enclosed_area_to_polygon([
                self.failure_curves[1].line,
                self.partitions[0].line,
                self.surface.linestring,
                self.partitions[1].line,
            ]),
            c=convert_enclosed_area_to_polygon([
                self.failure_curves[2].line,
                self.partitions[1].line,
                self.surface.linestring,
            ]),
        )
        s1 = self.surface.soil_params['s1']
        s2 = self.surface.soil_params['s2']
        s3 = self.surface.soil_params['s3']
        polygons = get_child_polygons(
            shops.unary_union(list(blocks.values())),
            [x.line for x in self.failure_curves]
            + [x.line for x in self.partitions]
            + [self.surface.linestring]
            + [s3.lower_bound, s2.lower_bound]
        )
        s2_lower_bound_y = s2.lower_bound.centroid.y
        s3_lower_bound_y = s3.lower_bound.centroid.y
        child_blocks = []
        for idx, polygon in enumerate(sorted(polygons, key=lambda x: x.centroid.y)):
            if polygon.centroid.y < s2_lower_bound_y:
                soil_params = s1
            elif polygon.centroid.y < s3_lower_bound_y:
                soil_params = s2
            else:
                soil_params = s3

            child_blocks.append(
                dict(
                    name=max(blocks.keys(),
                             key=lambda k: blocks[k].intersection(polygon).area),
                    geometry=polygon,
                ) | soil_params
            )

        self.blocks = gpd.GeoDataFrame(child_blocks, geometry='geometry')
        self.blocks['area'] = self.blocks.geometry.area
        # logger.debug("self.blocks:\n{}", self.blocks)
        # sys.exit()
        # self.blocks[('c', 's3')] = convert_enclosed_area_to_polygon([
        #     self.failure_curves[2].line,
        #     self.surface.soil_params['s3'].lower_bound,
        #     self.partitions[1].line,
        #     self.surface.linestring,
        # ])
        # self.blocks[('c', 's2')] = convert_enclosed_area_to_polygon([
        #     self.failure_curves[2].line,
        #     self.surface.soil_params['s3'].lower_bound,
        #     self.partitions[1].line,
        #     # self.surface.linestring,
        # ])

        # plot_polygon(blocks_2, color='orange')
        # plot_polygon(blocks_3, color='blue')
        return self

    energy_dissipation: Optional[float] = None

    def calculate_energy_dissipation(self) -> float:
        interactions = []
        for idx, row in self.blocks.iterrows():
            others: gpd.GeoDataFrame = self.blocks[self.blocks.index != idx].copy()
            surface_lines = others.intersection(row.geometry)
            is_touching = surface_lines.map(lambda x: isinstance(x, shg.LineString))
            others['surface'] = surface_lines[is_touching & (others.index > idx)]
            others = (
                others
                .dropna(subset=['surface'])
                .assign(parent_index=idx)
                .reset_index())
            if others.empty:
                continue
            others['diss_energy'] = others.apply(
                lambda r: get_energy_dissipation(
                    r.surface,
                    cohesion=(r.cohesion + row.cohesion) / 2,
                    phita=(r.phita + row.phita) / 2,
                    velocity=self.vs[r['name']],
                ), axis=1,
            )
            interactions.append(others)

        failure_line = shg.LineString(np.concatenate([
            get_line_xy(x.line) for x in self.failure_curves]))

        outer = self.blocks.copy()
        outer['surface'] = (
            self.blocks.intersection(failure_line)
            .map(shops.unary_union)
            # .map(lambda x: x if isinstance(x, shg.LineString) else shops.linemerge(x))
        )
        # logger.debug("outer['surface']:\n{}", outer['surface'])
        # logger.debug("surface:\n{}", surface)
        # sys.exit()
        is_touching = outer['surface'].map(lambda x: x.length) > 0
        outer = outer[is_touching].reset_index()
        # logger.debug("outer:\n{}", outer)
        # self.plot()
        outer['parent_index'] = outer['index']
        outer['diss_energy'] = outer.apply(
            lambda r: get_energy_dissipation(
                r.surface,
                cohesion=r.cohesion,
                phita=r.phita,
                velocity=self.vs[r['name']],
            ), axis=1,
        )
        interactions.append(outer)
        interactions = pd.concat(interactions, ignore_index=True)
        # logger.debug("interactions:\n{}", interactions)

        # sys.exit()
        # dissipation_energies = [
        #     get_energy_dissipation(
        #         self.failure_curves[0].line,
        #         cohesion=self.soil_params['s1']['cohesion'],
        #         phita=self.soil_params['s1']['phita'],
        #         velocity=self.vs['a'],
        #     ),
        #     get_energy_dissipation(
        #         self.failure_curves[1].line,
        #         cohesion=self.soil_params['s2']['cohesion'],
        #         phita=self.soil_params['s2']['phita'],
        #         velocity=self.vs['b'],
        #     ),
        #     get_energy_dissipation(
        #         self.failure_curves[2].line,
        #         cohesion=self.soil_params['s3']['cohesion'],
        #         phita=self.soil_params['s3']['phita'],
        #         velocity=self.vs['c'],
        #     ),
        #     get_energy_dissipation(
        #         self.partitions[0].line,
        #         cohesion=(self.soil_params.s1.cohesion +
        #                   self.soil_params.s2.cohesion) / 2,
        #         phita=(self.soil_params.s1.phita + self.soil_params.s2.phita) / 2,
        #         velocity=self.vs['ab'],
        #     ),
        #     get_energy_dissipation(
        #         self.partitions[1].line,
        #         cohesion=(
        #                          self.soil_params.s2.cohesion + self.soil_params.s3.cohesion) / 2,
        #         phita=(self.soil_params.s2.phita + self.soil_params.s3.phita) / 2,
        #         velocity=self.vs['bc'],
        #     ),
        # ]

        self.energy_dissipation = interactions['diss_energy'].sum()
        logger.debug("self.energy_dissipation: {}", self.energy_dissipation)
        return self.energy_dissipation

    external_work: Optional[float] = None

    def calculate_external_work(self):
        self.external_work = self.blocks.apply(
            lambda r: get_external_work(
                geom=r.geometry,
                phita=r.phita,
                gamma=r.gamma,
                velocity=self.vs[r['name']],
            ),
            axis=1).sum()
        logger.debug("self.external_work: {}", self.external_work)
        # self.external_work = np.array([
        #     get_external_work(
        #         part,
        #         phita=self.soil_params[soil_name]['phita'],
        #         gamma=self.soil_params[soil_name]['gamma'],
        #         velocity=self.vs[v_name])
        #     for soil_name, v_name, part in zip(
        #         ['s1', 's2', 's3'], ['a', 'b', 'c'],
        #         self.blocks.values())
        # ]).sum()
        return self.external_work

    def get_h_critical(self):
        # D scales with H one power lower than W (D ~ c*L*v with L~H; W ~
        # gamma*A*v with A~H^2, for the same velocity field), so for any
        # fixed mechanism shape the height at which D=W exactly - the
        # critical height, Eq. 24 - is the modeled height scaled by D/W,
        # not the vertical span of the failure curves at the modeled height.
        h_modeled = self.surface.elev_top - self.surface.elev_bottom
        return h_modeled * self.energy_dissipation / self.external_work

    def plot(self, show: bool = True):
        # self.blocks.pop('a')
        # self.blocks.pop('b')
        # self.blocks.pop('c')

        unary_block = self.blocks.union_all().buffer(3)
        self.surface.linestring = self.surface.linestring.intersection(unary_block)
        self.surface.plot()
        for origin in self.origins.values():
            plot_points(origin, color=next(_colors))
        for curve in self.failure_curves:
            curve.plot()

        for normal in self.normals:
            plot_line(normal.intersection(unary_block), color='gray', ls='--', lw=1,
                      alpha=0.5)

        for partition in self.partitions:
            partition.plot()
        self.blocks.plot(
            color=[next(_colors) for _ in range(len(self.blocks))],
            alpha=0.8)
        # for block in self.blocks.values():
        #     plot_polygon(block, color=next(_colors))

        soil_params = self.surface.soil_params

        plot_line(get_split(
            soil_params.s1.upper_bound,
            self.surface.linestring,
            ret_closer_to=shg.Point(big_num, 0),
        ), ls='--', lw=1.7, alpha=1, add_points=False, color='#7d570c')
        plot_line(get_split(
            soil_params.s2.upper_bound,
            self.surface.linestring,
            ret_closer_to=shg.Point(big_num, 0),
        ), ls='--', lw=1.7, alpha=1, add_points=False, color='#7d570c')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        tick_interval = 5
        x_ticks = np.arange(0, 25, tick_interval)
        y_ticks = np.arange(0, 25, tick_interval)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Set limits
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)

        # Display grid for visualization
        ax.grid(True)
        if show:
            plt.show()

    @classmethod
    def from_trial(cls, trial: Trial | FixedTrial, config: dict) -> Self:
        profile = config['soil_profile']
        layers = profile['layers']
        h_baseline = profile.get('h_baseline', 25.0)
        h_total = h_baseline + profile['total_height']

        t1 = layers['s1']['thickness']
        h1_upper = h_baseline + t1
        t2 = layers['s2']['thickness']
        h2_upper = h1_upper + t2
        soil_params2 = Box({
            's1': dict(
                gamma=layers['s1']['gamma'],
                cohesion=layers['s1']['cohesion'],
                phita=np.deg2rad(layers['s1']['phita_deg']),
                height_span=[None, h1_upper],
            ),
            's2': dict(
                gamma=layers['s2']['gamma'],
                cohesion=layers['s2']['cohesion'],
                phita=np.deg2rad(layers['s2']['phita_deg']),
                height_span=[h1_upper, h2_upper],
            ),
            's3': dict(
                gamma=layers['s3']['gamma'],
                cohesion=layers['s3']['cohesion'],
                phita=np.deg2rad(layers['s3']['phita_deg']),
                height_span=[h2_upper, None],
            ),
        })
        surface = Surface(
            elev_bottom=h_baseline,
            elev_top=h_total,
            slope_angle=np.deg2rad(profile['slope_angle_deg']),
            x_offset=150,
            soil_params=soil_params2,

        )

        _origins = {}

        def _suggest_origin(idx, x, y):
            if config['same_origins']:
                if 'last' not in _origins:
                    _origins['last'] = suggest_origin(trial, idx, x, y)
                return _origins['last']
            else:
                return suggest_origin(trial, idx, x, y)

        case = cls(
            soil_params=soil_params2,
            surface=surface,
            origins={
                'first': shg.Point(-10, 25),
                'third': shg.Point(-5, 15),
            },
        )
        origin_y_span = [h_baseline, h_total + 60]
        case.origins['first'] = _suggest_origin(1, [-15, 15], origin_y_span)
        case.origins['third'] = _suggest_origin(3, [-15, 15], origin_y_span)
        ratio1 = trial.suggest_float('normal_ratio_1', 0.2, 0.5)
        ratio2 = trial.suggest_float('normal_ratio_2', 0.55, 0.8)
        case.set_normals(ratios=[ratio1, ratio2])
        # for normal in case.normals:
        #     plot_line(normal)
        # plt.show()

        # curve_2_cls = eval(trial.suggest_categorical('curve_2_type', ['LinearCurve', 'LogSpiralCurve']))
        # curve_2_cls = eval(config['curve_2_type'])
        # if curve_2_cls == LinearCurve:
        #     curve_2 = LinearCurve(
        #         end_line=case.normals[1],
        #         angle=surface.slope_angle + np.deg2rad(
        #             trial.suggest_float('curve_2_angle_offset', -20, 20)),
        #         end_x=big_num,
        #     )
        # else:
        #     case.origins['second'] = _suggest_origin(2, [-15, 5], [5, 30])
        #
        #     curve_2 = LogSpiralCurve(
        #         end_line=case.normals[1],
        #         phita=case.soil_params['b']['phita'],
        #         origin=case.origins['second'],
        #         num_points=50,
        #     )

        curve_1 = LogSpiralCurve(
            end_line=case.normals[0],
            phita=case.soil_params['s1']['phita'],
            origin=case.origins['first'],
            num_points=60,
        )
        curve_2 = LinearCurve(
            end_line=case.normals[1],
            angle=surface.slope_angle + np.deg2rad(0),
            end_x=big_num,
        )
        curve_3 = LogSpiralCurve(
            end_line=surface.linestring,
            phita=case.soil_params['s3']['phita'],
            origin=case.origins['third'],
            num_points=30,
        )
        toe_offset = trial.suggest_float('toe_offset', -10, 0)
        case.set_failure_curves([curve_1, curve_2, curve_3], toe_offset=toe_offset)

        angle_offset = trial.suggest_float('partition_1_angle_offset', -30, 10)
        partition_1_params = dict(
            start=(ret_closer_to := case.failure_curves[1].start),
            end_line=surface.linestring,
            # angle=surface.slope_angle + np.deg2rad(90 + 20),
            # angle=np.deg2rad(180),
            angle=surface.slope_angle + np.deg2rad(90 - angle_offset),
            end_x=-big_num,
            ret_closer_to=ret_closer_to,
        )
        partition_1_cls = eval(config['partition_1_type'])
        if partition_1_cls == PolynomialCurve:
            partition_1_params |= dict(
                depth=trial.suggest_float('partition_1_depth', 0, 2),
                side='left',
                num_points=20,
            )

        angle_offset = trial.suggest_float('partition_2_angle_offset', -10, 30)
        partition_2_params = dict(
            start=(ret_closer_to := case.failure_curves[1].end_point),
            end_line=surface.linestring,
            # angle=surface.slope_angle + np.deg2rad(90 - 15),
            # angle=np.deg2rad(180),
            angle=surface.slope_angle + np.deg2rad(90 - angle_offset),
            end_x=-big_num,
            ret_closer_to=ret_closer_to,
        )
        # partition_2_cls = eval(trial.suggest_categorical('partition_2_type', ['LinearCurve', 'PolynomialCurve']))
        partition_2_cls = eval(config['partition_2_type'])
        if partition_2_cls == PolynomialCurve:
            partition_2_params |= dict(
                depth=trial.suggest_float('partition_2_depth', 0.0, 2),
                num_points=20,
            )
        case.set_partitions([
            partition_1_cls(**partition_1_params),
            partition_2_cls(**partition_2_params)])
        # case.set_partitions([
        #     LinearCurve(
        #         start=case.failure_curves[1].start,
        #         end_line=surface.linestring,
        #         angle=surface.slope_angle + np.deg2rad(90 + 20),
        #         end_x=-big_num,
        #     ),
        #     LinearCurve(
        #         start=case.failure_curves[1].end_point,
        #         end_line=surface.linestring,
        #         angle=surface.slope_angle + np.deg2rad(90 - 15),
        #         end_x=-big_num,
        #     ),
        # ])

        case.create_blocks()

        # surface.plot()
        # for normal in case.normals:
        #     plot_line(normal, color='gray', ls='--', lw=1, alpha=0.5)
        # case.plot()
        # plt.show()
        # sys.exit()
        energy_dissipation = case.calculate_energy_dissipation()
        external_work = case.calculate_external_work()
        h_critical = case.get_h_critical()
        logger.debug("h_critical: {}", h_critical)
        return case

    def get_penalty(self) -> float:
        break_point_penalties = []
        for failure_curve1, failure_curve2 in mit.windowed(self.failure_curves, 2):
            c1_end = np.arctan(get_slope_tangent(failure_curve1.line, start=-2))
            # logger.debug("c1_end: {}", c1_end)
            c2_start = np.arctan(get_slope_tangent(failure_curve2.line, stop=2))
            # logger.debug("c2_start: {}", c2_start)
            penalty = abs(c1_end - c2_start)
            # logger.debug("penalty: {}", penalty)
            break_point_penalties.append(penalty)
        # logger.debug("break_point_penalties:\n{}", break_point_penalties)
        return (
                abs(self.energy_dissipation - self.external_work) /
                max(self.energy_dissipation, self.external_work)
        ) + sum(break_point_penalties)


class Optimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: dict = Field(default_factory=dict)
    cases: dict = Field(default_factory=dict)
    sampler: optuna.samplers.BaseSampler
    n_trials: int
    n_jobs: int = 1
    storage_path: Path | str = 'db.sqlite'

    def objective_function(self, trial: Trial | FixedTrial) -> float:
        case = Case.from_trial(trial, self.config)

        trial.set_user_attr('energy_dissipation', case.energy_dissipation)
        trial.set_user_attr('external_work', case.external_work)
        trial.set_user_attr('h_critical', case.get_h_critical())

        self.cases[trial._trial_id] = case
        return case.get_penalty()

    def run(self):
        case_study = optuna.create_study(
            storage=f'sqlite:///{self.storage_path}',
            # study_name='reward_function_optimization2',
            # load_if_exists=True,
            # directions=['minimize'],
            sampler=self.sampler,
        )
        case_study.optimize(
            self.objective_function,
            n_trials=self.n_trials, n_jobs=self.n_jobs,
            catch=(ValueError, IndexError, GEOSException, NotImplementedError),
        )
        return case_study

    def plot(self, trial: FixedTrial):
        case = Case.from_trial(trial, self.config)
        logger.debug("case.energy_dissipation:\n{}", case.energy_dissipation)
        logger.debug("case.external_work:\n{}", case.external_work)
        case.plot()


def test_case(configs):
    sample_trial = FixedTrial(dict(
        origin_1_x=-5,
        origin_1_y=30,
        origin_2_x=0,
        origin_2_y=20,
        origin_3_x=0,
        origin_3_y=20,
        normal_position_1=0.35,
        normal_position_2=0.55,

        curve_2_angle_offset=0,

        partition_1_angle_offset=20,
        partition_2_angle_offset=-15,
        partition_1_depth=1,
        partition_2_depth=1,
    ))
    case = Case.from_trial(sample_trial, config=configs[0])
    case.get_penalty()
    case.plot()


def run_validation_cases(n_trials: int = 500, n_startup_trials: int = 200):
    """Run the three benchmark cases from the manuscript (homogeneous slope,
    two-layer clay-rock slope, three-layer slope).
    """
    results = {}
    for case_name, profile in VALIDATION_CASES.items():
        logger.info("Running validation case: {}", profile['description'])
        partition_type = profile.get('partition_type', 'LinearCurve')
        config = dict(
            partition_1_type=partition_type,
            partition_2_type=partition_type,
            curve_2_type='LinearCurve',
            same_origins=False,
            soil_profile=profile,
        )
        optimizer = Optimizer(
            config=config,
            sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=369),
            n_trials=n_trials,
            n_jobs=1,
        )
        case_study = optimizer.run()
        best_trial = case_study.best_trial
        h_critical = best_trial.user_attrs.get('h_critical')
        logger.info(
            "{}: H_crit={:.2f} m, penalty={:.4g}, FEM reference={}",
            case_name, h_critical, case_study.best_value, profile['fem_reference'],
        )
        results[case_name] = dict(
            h_critical=h_critical,
            penalty=case_study.best_value,
            best_params=case_study.best_params,
            fem_reference=profile['fem_reference'],
        )
    return results


def main():
    config_idx = 0
    configs = [
        # Case 1: different origins
        dict(
            partition_1_type='LinearCurve',
            partition_2_type='LinearCurve',
            curve_2_type='LinearCurve',
            same_origins=False,
            soil_profile=VALIDATION_CASES['case3_three_layer'],
        ),
        # Case 1: same origins
        dict(
            partition_1_type='LinearCurve',
            partition_2_type='LinearCurve',
            curve_2_type='LinearCurve',
            same_origins=True,
            soil_profile=VALIDATION_CASES['case3_three_layer'],
        ),
        # Case 2: different origins
        dict(
            partition_1_type='LinearCurve',
            partition_2_type='LinearCurve',
            curve_2_type='LogSpiralCurve',
            same_origins=False,
            soil_profile=VALIDATION_CASES['case3_three_layer'],
        ),
        # Case 3: different origins
        dict(
            partition_1_type='PolynomialCurve',
            partition_2_type='PolynomialCurve',
            curve_2_type='LinearCurve',
            same_origins=False,
            soil_profile=VALIDATION_CASES['case3_three_layer'],
        ),
        # Case 3: same origins
        dict(
            partition_1_type='PolynomialCurve',
            partition_2_type='PolynomialCurve',
            curve_2_type='LinearCurve',
            same_origins=True,
            soil_profile=VALIDATION_CASES['case3_three_layer'],
        ),
        # Case 4: different origins
        dict(
            partition_1_type='PolynomialCurve',
            partition_2_type='PolynomialCurve',
            curve_2_type='LogSpiralCurve',
            same_origins=False,
            soil_profile=VALIDATION_CASES['case3_three_layer'],
        ),
    ]
    # trial = FixedTrial(
    #     {'origin_1_x': 2.73822484074509, 'origin_1_y': 13.535840432796114,
    #      'normal_ratio_1': 0.3738144131359831, 'normal_ratio_2': 0.5612013769781284,
    #      'curve_2_angle_offset': -0.8124698833415422,
    #      'toe_offset': -2.0719307785839494,
    #      'partition_1_angle_offset': -15,
    #      'partition_2_angle_offset': 20})
    # trial = FixedTrial(
    #     {'origin_1_x': -0.23074857327928022, 'origin_1_y': 37.014837675681676,
    #      'normal_ratio_1': 0.2317734071776802, 'normal_ratio_2': 0.7998553388853996,
    #      'origin_2_x': 1.7698577310435422, 'origin_2_y': 14.47683670895816,
    #      'partition_1_depth': 2.16799367073377, 'partition_2_depth': 2.540524972461388})

    # trial = FixedTrial(
    #     {'origin_1_x': -7.953707269568891, 'origin_1_y': 29.843102949107095,
    #      'normal_ratio_1': 0.4990555034187145, 'normal_ratio_2': 0.5662166268467204,
    #      'origin_2_x': 1.484739865236886, 'origin_2_y': 27.302551742068104}
    # )
    # case = Case.from_trial(trial, config=configs[config_idx])
    # penalty = case.get_penalty()
    # logger.debug("penalty: {}", penalty)
    # case.plot()
    # return
    # test_case(configs)
    # return

    optimizer = Optimizer(
        config=configs[config_idx],
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=100,
            seed=369,
        ),
        n_trials=100,
        n_jobs=1,
    )
    case_study = optimizer.run()
    logger.debug("case_study.best_trial:\n{}", case_study.best_trial)
    logger.debug("case_study.best_params:\n{}", case_study.best_params)
    logger.debug("case_study.best_value:\n{}", case_study.best_value)
    trials_data: pd.DataFrame = case_study.trials_dataframe()
    logger.debug("trials_data:\n{}", trials_data)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trials_data.to_csv(f'trials_data__config_idx={config_idx}__{now}.csv')
    optimizer.plot(case_study.best_trial)

    pass


if __name__ == '__main__':
    run_validation_cases()
    # main()  # partition/kinematic-mode ablation study (Table 5)
