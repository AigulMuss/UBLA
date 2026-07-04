from typing import Optional, Any, Self, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from shapely import geometry as shg, affinity as shaff
from shapely.plotting import plot_line

from v1.geom_utils import get_line_xy, get_fitted_points, get_elongated
from v2.main import get_slope_tangent

from v3.utils import get_split, _colors, get_log_spiral_failure_curve_raw, \
    get_line_points


class Curve(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    end_line: shg.LineString

    start: Optional[shg.Point] = None
    end_point: Optional[shg.Point] = None

    line: Optional[shg.LineString] = None
    ret_closer_to: shg.Point | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.start is not None:
            self(self.start)

    def __call__(self, start: shg.Point) -> Self:
        return self

    def setup(self, start: shg.Point, failure_curve: shg.LineString):
        self.start = start

        self.end_point = failure_curve.intersection(self.end_line)
        # plot_line(failure_curve, color='magenta')
        # plot_line(self.end_line, color='red')
        self.line = get_split(failure_curve, self.end_line, self.ret_closer_to)
        # plot_line(self.line, color='k')
        # plt.show()
        return self

    def plot(self):
        plot_line(self.line, color=next(_colors))
        return self


class LogSpiralCurve(Curve):
    origin: shg.Point
    phita: float
    num_points: int = 100

    def __call__(self, start: shg.Point) -> Self:
        failure_curve = get_log_spiral_failure_curve_raw(self.origin, start, self.phita,
                                                         num=self.num_points)
        return self.setup(start, failure_curve)


class LinearCurve(Curve):
    angle: float
    end_x: float

    def __call__(self, start: shg.Point) -> Self:
        xs = np.array([start.x, self.end_x])
        ys = get_line_points(np.tan(self.angle), start, xs)
        failure_curve = shg.LineString(np.stack([xs, ys], axis=-1))
        return self.setup(start, failure_curve)


class PolynomialCurve(LinearCurve):
    degree: int = 2
    depth: float
    num_points: int = 100
    side: Literal['left', 'right'] = 'right'

    def __call__(self, start: shg.Point) -> Self:
        super().__call__(start)
        angle = np.arctan(get_slope_tangent(self.line))
        # logger.debug("angle:\n{}", angle)
        # plot_line(self.line)
        line_h = shaff.rotate(self.line, -angle, origin=start, use_radians=True)
        # logger.debug("line_h:\n{}", line_h)
        # plot_line(line_h)
        midway = line_h.parallel_offset(self.depth, self.side).centroid
        # plot_points(midway, color='red')
        xy = get_line_xy(shg.LineString([start, midway, line_h.interpolate(1, True)]))
        curve_h = shg.LineString(get_fitted_points(xy, self.degree, self.num_points))
        # plot_line(curve_h)
        failure_curve = shaff.rotate(curve_h, angle, origin=start, use_radians=True)
        # logger.debug("failure_curve:\n{}", failure_curve)
        # plot_line(failure_curve)
        # plt.show()
        return self.setup(start, get_elongated(failure_curve))
