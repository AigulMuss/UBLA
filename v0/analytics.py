import numpy as np
import shapely.geometry as shg
from loguru import logger

from v1.geom_utils import get_log_spiral_polygon, get_slices_by_points, plot_polygon, \
    get_bottom_line, get_line_inclination


def get_external_work(
        slice_geoms: list[shg.Polygon],
        angle_friction: float,
        gamma: float,
) -> float:
    velocity = 1
    external_work = []
    for slice_geom in slice_geoms:
        dissipation_line = get_bottom_line(slice_geom)
        line_angle_betta = get_line_inclination(dissipation_line)
        line_velocity = velocity * np.cos((line_angle_betta + angle_friction) * np.pi / 180)
        external_work.append(gamma * slice_geom.area * line_velocity)
    return np.array(external_work).sum()


def get_energy_dissipation(
        slice_geoms: list[shg.Polygon],
        cohesion: float,
        angle_friction: float,
) -> float:
    velocity = 1
    energy_dissipation = []
    for slice_geom in slice_geoms:
        dissipation_line = get_bottom_line(slice_geom)
        slice_energy_dissipation = cohesion * velocity * dissipation_line.length * np.cos(
            np.deg2rad(angle_friction))
        energy_dissipation.append(slice_energy_dissipation)
    return np.array(energy_dissipation).sum()


def main():
    gamma = 18  # kN/m3
    cohesion = 20  # kPa
    angle_friction = 23  # degrees
    # angle_betta = 45
    # angle_betta = np.rad2deg(1 / 4 * np.pi - 1 / 2 * np.deg2rad(angle_friction))
    # logger.debug("angle_betta:\n{}", angle_betta)

    # height = 1
    # height = 4 * cohesion / gamma * np.tan(1 / 4 * np.pi + 1 / 2 * np.deg2rad(angle_friction))
    # height = (
    #         (2 * cohesion * np.cos(angle_friction * np.pi / 180)) /
    #         (gamma * np.sin(angle_betta * np.pi / 180) *
    #          np.cos((angle_friction + angle_betta) * np.pi / 180))
    # )
    # logger.debug("height:\n{}", height)

    # return
    polygon = get_log_spiral_polygon(
        r0=10,
        theta_0=40 * np.pi / 180,
        theta_h=70 * np.pi / 180,
        phita=angle_friction * np.pi / 180,
        num_points_spiral=100,
    )
    # logger.debug("polygon:\n{}", polygon)

    # polygon = get_polygon_triangle(angle_betta=angle_betta, height=height)
    # logger.debug("polygon:\n{}", polygon)
    # return
    # polygon = get_log_spiral_polygon()
    logger.debug("polygon.area:\n{}", polygon.area)
    plot_polygon(polygon)
    return
    # return
    # return
    # slice_geoms = get_slices(polygon, step=0.1)
    slice_geoms = get_slices_by_points(polygon)
    logger.debug("len(slice_geoms):\n{}", len(slice_geoms))
    # for slice_geom in slice_geoms:
    #     print(slice_geom)
    #     input()
    # return
    slice_areas = np.array([x.area for x in slice_geoms]).sum()
    logger.debug("slice_areas:\n{}", slice_areas)
    # manual_area = 1 / 2 * height ** 2 * np.tan(angle_betta * np.pi / 180)
    # logger.debug("manual_area:\n{}", manual_area)
    # return
    external_work = get_external_work(slice_geoms, angle_friction, gamma)
    logger.debug("external_work:\n{}", external_work)

    energy_dissipation = get_energy_dissipation(slice_geoms, cohesion, angle_friction)
    logger.debug("energy_dissipation:\n{}", energy_dissipation)

    pass


if __name__ == '__main__':
    main()
