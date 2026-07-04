import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

pd.set_option('display.max_columns', None)
# Create dataframe
# df = pd.DataFrame(columns=['Slope angle', 'Water table', 'Unit weight', 'Friction angle prime', 'Friction angle b',
# 			   'L','Ox', 'Oy','Factor of Safety'])

# Create a dictionary to store the data
# data = {'Slope angle': [], 'Water table': [], 'Unit weight': [], 'Cohesion': [], 'Friction angle prime': [],
#         'Friction angle b': [],
#         'L': [], 'Ox': [], 'Oy': [], 'Theta_init': [], 'Factor of Safety': []}
data = defaultdict(list)
# Create a dictionary to store the data
# df_final = pd.DataFrame(data, columns=['Slope angle', 'Water table', 'Unit weight', 'Cohesion', 'Friction angle prime',
#                                        'Friction angle b',
#                                        'L', 'Ox', 'Oy', 'Theta_init', 'Factor of Safety'])

# DEFINE CONSTANTS AND VARIABLES
Angle_list = [30, 45, 60, 75, 89]  # degrees
Slope_height = 20  # meter
Toe_dist = 20  # meter
Base_height = 30  # meter
Base_width = 200  # meter
Water_table_list = [32, 34, 36, 38, 40, 42, 44, 46, 48]  # np.arange(32, 48, 2).tolist()  #meter
Unit_weight_list = [15, 17, 19, 21, 23, 25]  # [kN/m3]
Cohesion_effective_list = [1, 10, 15, 20, 25, 30]  # [kPa]
Friction_angle_prime_list = [1, 5, 10, 15, 20, 25, 30]  # [degrees]
U_a = 0  # pore air pressure [kPa]
U_w = 9.81  # pore water pressure [kPa]
# suction matric = U_a - U_w

epsilon = 2.73
Number_of_slices = 10
Theta_initial_list = [20]  # ,25,30,35,40,45]

for slope_angle in Angle_list:

    for unit_weight in Unit_weight_list:
        for water_table in Water_table_list:
            for cohesion_eff in Cohesion_effective_list:

                for friction_angle_pr in Friction_angle_prime_list:

                    Grid_y_0 = 65
                    Grid_y_1 = 80
                    Grid_range_y = np.arange(Grid_y_0, Grid_y_1, 1).tolist()
                    Grid_x_0 = round(Toe_dist + Slope_height / (math.tan(slope_angle * math.pi / 180) * 4))
                    Grid_x_1 = round(Toe_dist + 3 * Slope_height / (math.tan(slope_angle * math.pi / 180) * 4))
                    Grid_range_x = np.arange(Grid_x_0, Grid_x_1, 1).tolist()

                    friction_angle_b = friction_angle_pr / 2
                    Crown_coor = Toe_dist + Slope_height / math.tan(
                        slope_angle * math.pi / 180)  # x coordinate of crown

                    # material properties for soil
                    # saturated and unsaturated
                    H_unsat = Base_height + Slope_height - water_table  # height of unsaturated soil
                    suction = H_unsat * 9.81
                    cohesion_unsat = cohesion_eff + suction * math.tan(friction_angle_b * math.pi / 180)
                    cohesion_sat = cohesion_eff

                    cohesion = cohesion_unsat * (
                            (Base_height + Slope_height - water_table) / Slope_height) + cohesion_sat * (
                                       (water_table - Base_height) / Slope_height)

                    Grid_range_x = Grid_range_x[:1]
                    Grid_range_y = Grid_range_y[:1]
                    for O_x in Grid_range_x:
                        for O_y in Grid_range_y:
                            # check the Theta_inital interval
                            # calculate Th
                            # O_y, O_x = 70, 37
                            dist_horizontal = O_x - Toe_dist
                            dist_vertical = abs(O_y - Base_height)
                            R_h = (dist_vertical ** 2 + dist_horizontal ** 2) ** 0.5
                            theta_h = math.atan((dist_vertical / dist_horizontal))
                            theta_h = 180 - math.degrees(theta_h)
                            # print(theta_h, dist_vertical, dist_horizontal)
                            # SELECT THETA INITIAL AND PERFORM ANALYSIS FOR EACH POSSIBLE CASE
                            for theta_init in Theta_initial_list:

                                R_0 = (O_y - (Base_height + Slope_height)) / math.sin(theta_init * math.pi / 180)
                                # calculate L - length of circular arc on the crown
                                L = R_0 * math.cos(theta_init * math.pi / 180) + O_x
                                # print("#################", L)
                                if L < 0:
                                    break

                                if L > 0:
                                    if L > Crown_coor:

                                        slice_x_coors = []
                                        slice_y_coors = []
                                        theta_slices = []
                                        alpha_slices = []
                                        R_slices = []
                                        R_mid_slices = []
                                        B_slices = []
                                        lambda_constant_slices = []

                                        # Spencer 1967 proposed
                                        # f_x = 1 #assume f(x)=1
                                        # lambda_constant = math.tan(theta_slice*math.pi/180) #theta is angle from horizontal to current slice center

                                        for slice_num in range(Number_of_slices):

                                            theta_slice = theta_init + slice_num * (
                                                    theta_h - theta_init) / Number_of_slices
                                            theta_slices.append(theta_slice)

                                            R = R_0 * epsilon ** (
                                                    ((theta_slice - theta_init) * math.pi / 180) * math.tan(
                                                friction_angle_pr * math.pi / 180))
                                            # print('R', R, 'R_0', R_0, 'theta_slice', theta_slice, 'L', L, 'toe dist', Toe_dist)
                                            V = O_y - R * math.sin(theta_slice * math.pi / 180)
                                            if theta_slice <= 90:
                                                H = O_x + R * math.cos(theta_slice * math.pi / 180)
                                            elif theta_slice > 90:
                                                H = O_x - R * math.cos(theta_slice * math.pi / 180)

                                            slice_x_coors.append(H)
                                            slice_y_coors.append(V)
                                            R_slices.append(R)
                                            lambda_constant_slices.append(0)  # math.tan(theta_slice*math.pi/180))

                                            # angle from the tangent to the slice and horizontal
                                            alpha = 90 - theta_slice
                                            alpha_slices.append(alpha)

                                        slice_x_coors.append(Toe_dist)
                                        slice_y_coors.append(Base_height)
                                        theta_slices.append(theta_h)
                                        R_slices.append(R_h)

                                        moment_arm_slices = []
                                        H_slices = []

                                        Area_slices = []
                                        for slice_num in range(Number_of_slices):
                                            B_slices.append(slice_x_coors[slice_num] - slice_x_coors[slice_num + 1])

                                            moment_arm_coor = O_x - (
                                                    slice_x_coors[slice_num] + slice_x_coors[slice_num + 1]) / 2
                                            moment_arm_slices.append(moment_arm_coor)

                                            V_top = Base_height + Slope_height

                                            if moment_arm_coor < Crown_coor:
                                                H_slices.append(Base_height + math.tan(slope_angle * math.pi / 180) * ((
                                                                                                                               (
                                                                                                                                       slice_x_coors[
                                                                                                                                           slice_num] +
                                                                                                                                       slice_x_coors[
                                                                                                                                           slice_num + 1]) / 2) - Toe_dist) -
                                                                slice_y_coors[slice_num])
                                            else:
                                                H_slices.append(V_top - slice_y_coors[slice_num])

                                            Area_slices.append(B_slices[slice_num] * H_slices[slice_num])
                                        # computing the slice height

                                        Weight_slices = [abs(unit_weight * Area) for Area in Area_slices]
                                        Normal_stress_slices = []
                                        Betta_slices = []
                                        for slice_num in range(Number_of_slices):
                                            # res=Weight_slices[slice_num] / Area_slices[slice_num]
                                            # logger.debug("res:\n{}", res)
                                            Normal_stress_slices.append(
                                                (Weight_slices[slice_num] / Area_slices[slice_num]) * math.cos(
                                                    (90 - alpha_slices[slice_num]) * math.pi / 180))
                                            Betta_slices.append(
                                                B_slices[slice_num] / math.cos(alpha_slices[slice_num]) * math.pi / 180)
                                        # logger.debug("Normal_stress_slices:\n{}", Normal_stress_slices)
                                        # sys.exit()
                                        for slice_num in range(Number_of_slices):
                                            R_mid_slices.append(R_0 * epsilon ** ((((theta_slices[slice_num] +
                                                                                     theta_slices[
                                                                                         slice_num + 1]) / 2 - theta_init) * math.pi / 180) * math.tan(
                                                friction_angle_pr * math.pi / 180)))

                                        # # INITIAL ASSUPMTION FOR SAFETY FACTOR, FORCE AND MOMENT EQUILIBRIUM

                                        Moment_equil = float('-inf')
                                        F_s = 1.0
                                        Force_equil_nums = 0
                                        Force_equil_denums = 0
                                        Moment_equil_nums = 0
                                        Moment_equil_denums = 0
                                        delta_X = 0
                                        F_mm_list = [2.0, 1.0]
                                        F_ff_list = [2.0, 1.0]
                                        while abs(F_ff_list[-1] - F_ff_list[-2]) >= 0.001:

                                            for slice_num in range(Number_of_slices):
                                                # compute normal force
                                                m_aplha = math.cos(alpha_slices[slice_num] * math.pi / 180) + F_s / (
                                                        math.sin(
                                                            alpha_slices[slice_num] * math.pi / 180) * math.tan(
                                                    friction_angle_pr * math.pi / 180))

                                                Normal_force = (Weight_slices[slice_num] - delta_X - cohesion *
                                                                Betta_slices[slice_num] * math.sin(
                                                            alpha * math.pi / 180) / F_s + U_w * Betta_slices[
                                                                    slice_num] * math.sin(
                                                            alpha_slices[slice_num] * math.pi / 180) * math.tan(
                                                            friction_angle_b * math.pi / 180) / F_s) / m_aplha

                                                # compute mobilized shear force

                                                Shear_force = (Betta_slices[slice_num] / F_s) * (cohesion + (
                                                        Normal_stress_slices[slice_num] - U_a) * math.tan(
                                                    friction_angle_pr * math.pi / 180) + suction * math.tan(
                                                    friction_angle_b * math.pi / 180))

                                                # compute the interslice normal force using dX = (E_l-E_r)*lambda_constant*f(x)
                                                delta_E = (Weight_slices[slice_num] - delta_X) * math.tan(
                                                    alpha_slices[slice_num] * math.pi / 180) - Shear_force / math.cos(
                                                    alpha_slices[slice_num] * math.pi / 180)
                                                # assume delta T = 0 from the book
                                                # (lambda_constant_slices[slice_num])*(Weight_slices[slice_num]*math.sin(alpha_slices[slice_num]*math.pi/180)-(Weight_slices[slice_num]*math.cos(alpha_slices[slice_num]*math.pi/180)*math.tan(friction_angle_pr*math.pi/180)-cohesion*B_slices[slice_num]/math.cos(alpha_slices[slice_num]*math.pi/180))/F_s)
                                                delta_X = lambda_constant_slices[slice_num] * delta_E
                                                # print(delta_X)
                                                # compute normal force
                                                Normal_force = (Weight_slices[slice_num] - delta_X - cohesion *
                                                                Betta_slices[slice_num] * math.sin(
                                                            alpha * math.pi / 180) / F_s + U_w * Betta_slices[
                                                                    slice_num] * math.sin(
                                                            alpha_slices[slice_num] * math.pi / 180) * math.tan(
                                                            friction_angle_b * math.pi / 180) / F_s) / m_aplha
                                                # interslice shear force = X_r-X_l -> denoted as Shear_inter_diff
                                                Moment_equil_num = (cohesion * Betta_slices[slice_num] + (
                                                        Normal_force - U_w * Betta_slices[slice_num]) * math.tan(
                                                    friction_angle_pr * math.pi / 180)) * R_mid_slices[slice_num]
                                                Moment_equil_denum = Weight_slices[slice_num] * abs(
                                                    moment_arm_slices[slice_num])

                                                Moment_equil_nums += Moment_equil_num
                                                Moment_equil_denums += Moment_equil_denum
                                            # print(Normal_force- U_w*Betta_slices[slice_num])
                                            F_s = Moment_equil_nums / Moment_equil_denums
                                            F_mm_list.append(F_s)

                                            for slice_num in range(Number_of_slices):
                                                # compute the interslice normal force using dX = (E_l-E_r)*lambda_constant*f(x)
                                                delta_E = (Weight_slices[slice_num] - delta_X) * math.tan(
                                                    alpha_slices[slice_num] * math.pi / 180) - Shear_force / math.cos(
                                                    alpha_slices[slice_num] * math.pi / 180)
                                                # assume delta T = 0 from the book
                                                # (lambda_constant_slices[slice_num])*(Weight_slices[slice_num]*math.sin(alpha_slices[slice_num]*math.pi/180)-(Weight_slices[slice_num]*math.cos(alpha_slices[slice_num]*math.pi/180)*math.tan(friction_angle_pr*math.pi/180)-cohesion*B_slices[slice_num]/math.cos(alpha_slices[slice_num]*math.pi/180))/F_s)
                                                delta_X = lambda_constant_slices[slice_num] * delta_E

                                                m_aplha = math.cos(alpha_slices[slice_num] * math.pi / 180) + F_s / (
                                                        math.sin(
                                                            alpha_slices[slice_num] * math.pi / 180) * math.tan(
                                                    friction_angle_pr * math.pi / 180))
                                                Normal_force = (Weight_slices[slice_num] - delta_X - cohesion *
                                                                Betta_slices[slice_num] * math.sin(
                                                            alpha * math.pi / 180) / F_s + U_w * Betta_slices[
                                                                    slice_num] * math.sin(
                                                            alpha_slices[slice_num] * math.pi / 180) * math.tan(
                                                            friction_angle_b * math.pi / 180) / F_s) / m_aplha

                                                # compute mobilized shear force

                                                Shear_force = (Betta_slices[slice_num] / F_s) * (cohesion + (
                                                        Normal_stress_slices[slice_num] - U_a) * math.tan(
                                                    friction_angle_pr * math.pi / 180) + suction * math.tan(
                                                    friction_angle_b * math.pi / 180))

                                                Force_equil_num = (cohesion * Betta_slices[slice_num] * math.cos(
                                                    alpha_slices[slice_num] * math.pi / 180) + (
                                                                           Normal_force - U_w * Betta_slices[
                                                                       slice_num] * math.tan(
                                                                       friction_angle_b * math.pi / 180) / math.tan(
                                                                       friction_angle_pr * math.pi / 180))) * math.tan(
                                                    friction_angle_pr * math.pi / 180) * math.cos(
                                                    alpha_slices[slice_num] * math.pi / 180)
                                                Force_equil_denum = Normal_force * math.sin(
                                                    alpha_slices[slice_num] * math.pi / 180)
                                                Force_equil_nums += Force_equil_num
                                                Force_equil_denums += Force_equil_denum
                                            # print(Force_equil_denum)
                                            # print(delta_X)
                                            F_s = Force_equil_nums / Force_equil_denums
                                            F_ff_list.append(F_s)
                                        # print(F_ff_list)
                    data['Slope angle'].append(slope_angle)
                    data['Water table'].append(water_table)
                    data['Unit weight'].append(unit_weight)
                    data['Cohesion'].append(cohesion)
                    data['Friction angle prime'].append(friction_angle_pr)
                    data['Friction angle b'].append(friction_angle_b)
                    data['L'].append(L)
                    data['Ox'].append(O_x)
                    data['Oy'].append(O_y)
                    data['Theta_init'].append(theta_init)
                    data['Factor of Safety'].append(F_s)

                    # print(data)
                    # df = pd.DataFrame(data,
                    #                   columns=['Slope angle', 'Water table', 'Unit weight', 'Friction angle prime',
                    #                            'Friction angle b',
                    #                            'L', 'Ox', 'Oy', 'Theta_init', 'Factor of Safety', 'Cohesion'])
                    # df = df.sort_values(by=['Factor of Safety'], ascending=False)
                    # final_data.append(data)
                    # df_final = df_final.append(df[:2])
                    # print(df[:2])

                    # Clean the disctionary
                    # data = {'Slope angle': [], 'Water table': [], 'Unit weight': [], 'Friction angle prime': [],
                    #         'Friction angle b': [],
                    #         'L': [], 'Ox': [], 'Oy': [], 'Theta_init': [], 'Factor of Safety': [], 'Cohesion': []}
# print(min(data['Factor of Safety']))
df_final = pd.DataFrame(data)
df_final.to_csv('data.csv')
