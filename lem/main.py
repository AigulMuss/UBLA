import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as shg
from box import Box

os.environ["LOGURU_LEVEL"] = "INFO"

from loguru import logger
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.trial import FixedTrial
from pydantic import BaseModel, ConfigDict
from shapely.plotting import plot_polygon, plot_points

from v1.geom_utils import get_bottom_line
from v2.main import Surface, get_slope_tangent
from v3.main import Optimizer as BaseOptimizer
from v3.utils import get_slices_by_points, suggest_origin


def get_fs(
        failure_geom: shg.Polygon,
        soil_info: Box,
        fs_assumed: float = None,
):
    slice_infos = []
    for geom in get_slices_by_points(failure_geom):
        if geom.area < 1e-3:
            continue
        # plot_polygon(geom, color=next(_colors))
        info = Box()
        base_line = get_bottom_line(geom)
        info.l = base_line.length
        info.alpha = np.arctan(get_slope_tangent(base_line))
        info.W = soil_info.gamma * geom.area  # kN/m
        info.N1 = info.W * np.sin(info.alpha)

        info.N2 = (
                (info.W / np.cos(info.alpha) - soil_info.pore_pressure * info.l) *
                np.tan(soil_info.phita) + soil_info.cohesion * info.l
        )
        if fs_assumed is not None:
            info.N2 /= (
                    1 + np.tan(info.alpha) * np.tan(soil_info.phita) / fs_assumed
            )
        slice_infos.append(info)

        # N2_denominator = (
        #         1 + np.tan(info.alpha) * np.tan(phita) /
        # )

        # logger.debug("info.alpha:\n{}", np.rad2deg(info.alpha))
    slice_infos = pd.DataFrame(slice_infos)
    logger.debug("slice_infos:\n{}", slice_infos)
    return slice_infos['N2'].sum() / abs(slice_infos['N1'].sum())


class Case(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    surface: Surface
    soil_info: Box
    origin: shg.Point
    debug: bool = False
    tolerance: float = 1e-4
    toe_offset: float = 0

    def get_failure_polygon(self):
        return self.surface.get_failure_polygon(
            self.origin,
            start_offset=(self.toe_offset, 0),
            phita=self.soil_info.phita)

    def get_fs(self) -> float:
        failure_geom = self.get_failure_polygon()
        if self.debug:
            plot_polygon(failure_geom)
        fs_assumed = get_fs(failure_geom, self.soil_info)
        # logger.debug("fs_assumed: {:.4f}", fs_assumed)
        while True:
            fs_calculated = get_fs(failure_geom, self.soil_info, fs_assumed=fs_assumed)
            # logger.debug("fs_calculated: {:.4f}", fs_calculated)
            if abs(fs_calculated - fs_assumed) < self.tolerance:
                break
            fs_assumed = fs_calculated
        return fs_calculated

    def plot(self):
        failure_geom = self.get_failure_polygon()
        self.surface.linestring = self.surface.linestring.intersection(
            failure_geom.buffer(5))
        self.surface.plot()
        plot_points([self.origin], color='k')
        plot_polygon(failure_geom)

    @classmethod
    def from_(cls, trial_params: dict, **kwargs):
        trial = FixedTrial(trial_params)
        origin = suggest_origin(trial, 0, [-20, 10], [20, 40])
        toe_offset = trial.suggest_int("toe_offset", -7, 0)
        return cls(
            origin=origin,
            toe_offset=toe_offset,
            **kwargs
        )

    def run_debug(self):
        fs = self.get_fs()
        logger.debug("fs: {}", fs)
        self.plot()
        plt.show()


class Optimizer(BaseOptimizer):
    surface: Surface
    soil_info: Box

    def objective_function(self, trial: Trial | FixedTrial) -> float:
        origin = suggest_origin(trial, 0, [-20, 10], [20, 40])
        toe_offset = trial.suggest_int("toe_offset", -7, 0)
        case = Case(
            origin=origin,
            soil_info=self.soil_info,
            surface=self.surface,
            toe_offset=toe_offset,
        )

        self.cases[trial._trial_id] = case
        return case.get_fs()

    def plot(self, trial: FixedTrial):
        case = self.cases[trial._trial_id]
        case.plot()
        plt.show()


def test():
    soil_info = Box(
        gamma=17.3,  # kN/m3
        phita=np.deg2rad(20),
        cohesion=15,  # kN/m2
        pore_pressure=0,
    )
    surface = Surface(
        elev_bottom=5,
        elev_top=20,
        slope_angle=np.deg2rad(50),
        x_offset=30,
    )
    surface.plot()
    origin = shg.Point(-7, 30)

    case = Case(
        surface=surface,
        soil_info=soil_info,
        origin=origin,
        debug=True,
    )
    fs = case.get_fs()
    logger.debug("fs: {}", fs)
    plt.show()


this_folder = Path(__file__).parent.absolute()


def main():
    soil_info = Box(
        gamma=17.3,  # kN/m3
        phita=np.deg2rad(20),
        cohesion=15,  # kN/m2
        pore_pressure=0,
    )
    surface = Surface(
        elev_bottom=5,
        elev_top=20,
        slope_angle=np.deg2rad(50),
        x_offset=100,
    )

    # Case.from_(
    #     {'origin_0_x': -15.90520485140059, 'origin_0_y': 20.61496636130893, 'toe_offset': -2},
    #     surface=surface,
    #     soil_info=soil_info,
    # ).run_debug()
    # return
    # test()

    optimizer = Optimizer(
        sampler=TPESampler(
            n_startup_trials=200,
            seed=369,
        ),
        n_trials=500,
        n_jobs=1,
        soil_info=soil_info,
        surface=surface,
        storage_path=this_folder / 'lem.sqlite',
    )
    case_study = optimizer.run()
    logger.info("case_study.best_trial:\n{}", case_study.best_trial)
    logger.info("case_study.best_params:\n{}", case_study.best_params)
    logger.info("case_study.best_value:\n{}", case_study.best_value)
    trials_data: pd.DataFrame = case_study.trials_dataframe()
    logger.info("trials_data:\n{}", trials_data)
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trials_data.to_csv(f'trials_data__{now}.csv')
    optimizer.plot(case_study.best_trial)


if __name__ == '__main__':
    main()
