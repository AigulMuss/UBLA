import numpy as np
from loguru import logger

from v1.geom_utils import get_log_spiral_polygon


def main():
    # Given points
    x1, y1 = 2, 3
    x2, y2 = 4, 5

    # Convert to polar coordinates
    r1 = np.sqrt(x1 ** 2 + y1 ** 2)
    theta1 = np.arctan2(y1, x1)
    r2 = np.sqrt(x2 ** 2 + y2 ** 2)
    theta2 = np.arctan2(y2, x2)

    # Calculate tan(phi)
    tan_phi = np.log(r2 / r1) / (theta2 - theta1)

    # Solve for r_0
    r_0 = r1 * np.exp(-theta1 * tan_phi)

    # Solve for theta_0
    theta_0 = theta1 - np.log(r1 / r_0) / tan_phi

    r_0, theta_0
    logger.debug("r_0:\n{}", r_0)
    logger.debug("theta_0:\n{}", theta_0)

    # get_log_spiral_polygon()
    pass


if __name__ == '__main__':
    main()
