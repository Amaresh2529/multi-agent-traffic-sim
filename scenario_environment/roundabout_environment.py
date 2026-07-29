from sqlalchemy.sql.functions import random
import random
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
from params import *

def RGB_to_Hex(rgb):
    RGB = rgb.split(',')
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def scenario_outfit(ax, color=RGB_to_Hex('202,202,202')):
    radius = ROUNDABOUT_R - 5
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, c=color)

    radius = ROUNDABOUT_R
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, c=color)

    ax.plot([20, 100], [5, 5], c=color)
    ax.plot([20, 100], [-5, -5], c=color)
    ax.plot([-100, -20], [5, 5], c=color)
    ax.plot([-100, -20], [-5, -5], c=color)
    
    ax.plot([5, 5], [20, 100], c=color)
    ax.plot([-5, -5], [20, 100], c=color)
    ax.plot([5, 5], [-100, -20], c=color)
    ax.plot([-5, -5], [-100, -20], c=color)

    ax.plot([0, 0], [25, 100], c='black', linestyle='--', linewidth=0.5,alpha=0.5)  
    ax.plot([0, 0], [-100, -25], c='black', linestyle='--', linewidth=0.5,alpha=0.5)
    ax.plot([-100, -25], [0, 0], c='black', linestyle='--', linewidth=0.5,alpha=0.5)
    ax.plot([25, 100], [0, 0], c='black', linestyle='--', linewidth=0.5,alpha=0.5)

    # --- NEW: Pedestrian Crosswalk Visuals ---
    cw_style = ':'
    cw_color = 'gray'
    cw_width = 1.5
    # South crosswalk
    ax.plot([-7.5, 7.5], [-20, -20], c=cw_color, linestyle=cw_style, linewidth=cw_width)
    ax.plot([-7.5, 7.5], [-25, -25], c=cw_color, linestyle=cw_style, linewidth=cw_width)
    # North crosswalk
    ax.plot([-7.5, 7.5], [20, 20], c=cw_color, linestyle=cw_style, linewidth=cw_width)
    ax.plot([-7.5, 7.5], [25, 25], c=cw_color, linestyle=cw_style, linewidth=cw_width)
    # East crosswalk
    ax.plot([20, 20], [-7.5, 7.5], c=cw_color, linestyle=cw_style, linewidth=cw_width)
    ax.plot([25, 25], [-7.5, 7.5], c=cw_color, linestyle=cw_style, linewidth=cw_width)
    # West crosswalk
    ax.plot([-20, -20], [-7.5, 7.5], c=cw_color, linestyle=cw_style, linewidth=cw_width)
    ax.plot([-25, -25], [-7.5, 7.5], c=cw_color, linestyle=cw_style, linewidth=cw_width)


def smooth_ployline(cv_init, point_num=3000):
    cv = cv_init
    list_x = cv[:, 0]
    list_y = cv[:, 1]
    if type(cv) is not np.ndarray:
        cv = np.array(cv)
    delta_cv = cv[1:, ] - cv[:-1, ]
    s_cv = np.linalg.norm(delta_cv, axis=1)

    s_cv = np.array([0] + list(s_cv))
    s_cv = np.cumsum(s_cv)
    bspl_x = splrep(s_cv, list_x, s=0.1)
    bspl_y = splrep(s_cv, list_y, s=0.1)
    # values for the x axis
    s_smooth = np.linspace(0, max(s_cv), point_num)
    # get y values from interpolated curve
    x_smooth = splev(s_smooth, bspl_x)
    y_smooth = splev(s_smooth, bspl_y)
    new_cv = np.array([x_smooth, y_smooth]).T
    delta_new_cv = new_cv[1:, ] - new_cv[:-1, ]
    s_accumulated = np.cumsum(np.linalg.norm(delta_new_cv, axis=1))
    s_accumulated = np.concatenate(([0], s_accumulated), axis=0)
    return new_cv, s_accumulated

def if_going_straight(entrance, exit):
    if entrance == 'w':
        return True
    else:
        return False

def if_right_turning(entrance, exit):
    if entrance == 's':
        return True
    else:
        return False

def if_left_turning(entrance, exit):
    return False

def enter_roundabout_ref_line(entrance, exit):
    cv_init = None
    if entrance == 's':
        cv_init = np.array([[2.5, -25], [2.5, -24], [2.5, -23], [2.5, -22], [2.9, -20], [5, -18], [7, -16], [7.7, -15.75], [8, -15.6]])
    if entrance == 'n':
        cv_init = np.array([[-2.5, 25], [-2.5, 24], [-2.5, 23], [-2.5, 22], [-2.9, 20], [-5, 18], [-7, 16], [-7.7, 15.75], [-8, 15.6]])
    if entrance == 'e':
        cv_init = np.array([[25, 2.5], [24, 2.5], [23, 2.5], [22, 2.5], [20, 2.9], [18, 5], [16, 7], [15.75, 7.7], [15.6, 8]])
    if entrance == 'w':
        cv_init = np.array([[-25, -2.5], [-24, -2.5], [-23, -2.5], [-22, -2.5], [-20, -2.9], [-18, -5], [-16, -7], [-15.75, -7.7], [-15.6, -8]])

    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed  # , s_accumulated

def exit_roundabout_ref_line(entrance, exit):
    cv_init = None
    if exit == 's':
        cv_init = np.array([[-8, -15.6], [-7.7, -15.75], [-7, -16], [-5, -18], [-2.9, -20], [-2.5, -22], [-2.5, -23], [-2.5, -24], [-2.5, -25]])
    if exit == 'n':
        cv_init = np.array([[8, 15.6], [7.7, 15.75], [7, 16], [5, 18], [2.9, 20], [2.5, 22], [2.5, 23], [2.5, 24], [2.5, 25]])
    if exit == 'e':
        cv_init = np.array([[15.6, -8], [15.75, -7.7], [16, -7], [18, -5], [20, -2.9], [22, -2.5], [23, -2.5], [24, -2.5], [25, -2.5]])
    if exit == 'w':
        cv_init = np.array([[-15.6, 8], [-15.75, 7.7], [-16, 7], [-18, 5], [-20, 2.9], [-22, 2.5], [-23, 2.5], [-24, 2.5], [-25, 2.5]])

    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed  # , s_accumulated

def roundabout_ref_line(entrance, exit):
    theta = None
    radius = 17.5
    if entrance == 's':
        if exit == 'e':
            theta = np.linspace(1.65 * np.pi, 1.85 * np.pi, 2000)
        elif exit == 'n':
            theta = np.linspace(-0.35 * np.pi, 0.35 * np.pi, 4000)
        elif exit == 'w':
            theta = np.linspace(-0.35 * np.pi, 0.85 * np.pi, 6000)
        else:
            theta = np.linspace(-0.35 * np.pi, 1.35 * np.pi, 8000)

    if entrance == 'e':
        if exit == 'n':
            theta = np.linspace(0.15 * np.pi, 0.35 * np.pi, 2000)
        elif exit == 'w':
            theta = np.linspace(0.15 * np.pi, 0.85 * np.pi, 4000)
        elif exit == 's':
            theta = np.linspace(0.15 * np.pi, 1.35 * np.pi, 6000)
        else:
            theta = np.linspace(0.15 * np.pi, 1.85 * np.pi, 8000)

    if entrance == 'n':
        if exit == 'w':
            theta = np.linspace(0.65 * np.pi, 0.85 * np.pi, 2000)
        elif exit == 's':
            theta = np.linspace(0.65 * np.pi, 1.35 * np.pi, 4000)
        elif exit == 'e':
            theta = np.linspace(0.65 * np.pi, 1.85 * np.pi, 6000)
        else:
            theta = np.linspace(-1.35 * np.pi, 0.35 * np.pi, 8000)

    if entrance == 'w':
        if exit == 's':
            theta = np.linspace(1.15 * np.pi, 1.35 * np.pi, 2000)
        elif exit == 'e':
            theta = np.linspace(1.15 * np.pi, 1.85 * np.pi, 4000)
        elif exit == 'n':
            theta = np.linspace(-0.85 * np.pi, 0.35 * np.pi, 6000)
        else:
            theta = np.linspace(-0.85 * np.pi, 0.85 * np.pi, 8000)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ref_line = np.vstack((x, y))
    return ref_line.T

def record_ref_line_distance2exit(ref_line):
    cv = ref_line
    gap_list = np.zeros(len(cv))
    for point in range(len(ref_line) - 1):
        gap = np.sqrt((cv[point, 0] - cv[point + 1, 0]) ** 2 + (cv[point, 1] - cv[point + 1, 1]) ** 2)
        gap_list[point:] += gap
    ref_line_distance2exit = max(gap_list) - np.array(gap_list)
    return ref_line_distance2exit

def entrance_ref_line(entrance, exit):
    ref_line = None
    if entrance == 'w':
        x = np.linspace(-100, -25, 2000)
        y = -2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if entrance == 'e':
        x = np.linspace(100, 25, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if entrance == 'n':
        y = np.linspace(100, 25, 2000)
        x = -2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    if entrance == 's':
        y = np.linspace(-100, -25, 2000)
        x = 2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    return ref_line.T

def exit_ref_line(entrance, exit):
    ref_line = None
    if exit == 'w':
        x = np.linspace(-25.1, -100, 2000)
        y = 2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if exit == 'e':
        x = np.linspace(25.1, 100, 2000)
        y = -2.5 * np.ones_like(x)
        ref_line = np.vstack((x, y))
    if exit == 'n':
        y = np.linspace(25.1, 100, 2000)
        x = 2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    if exit == 's':
        y = np.linspace(-25.1, -100, 2000)
        x = -2.5 * np.ones_like(y)
        ref_line = np.vstack((x, y))
    return ref_line.T

def concatenate_ref_lane(entrance, exit):
    ref_lane1 = entrance_ref_line(entrance, exit)
    ref_lane2 = enter_roundabout_ref_line(entrance, exit)
    ref_lane3 = roundabout_ref_line(entrance, exit)
    ref_lane4 = exit_roundabout_ref_line(entrance, exit)
    ref_lane5 = exit_ref_line(entrance, exit)
    ref_lane = np.vstack((ref_lane1, ref_lane2))
    ref_lane = np.vstack((ref_lane, ref_lane3))
    ref_lane = np.vstack((ref_lane, ref_lane4))
    ref_lane = np.vstack((ref_lane, ref_lane5))
    return ref_lane

def default_exit_and_state(entrance, exit):
    state = None
    velocity = np.random.uniform(6, 9)  
    heading = None

    if entrance == 'w':
        state = [-100, -2.5]
        heading = 0 * math.pi
    if entrance == 'e':
        state = [100, 2.5]
        heading = math.pi
    if entrance == 'n':
        state = [-2.5, 100]
        heading = 1.5 * math.pi
    if entrance == 's':
        state = [2.5, -100]
        heading = 0.5 * math.pi

    state.append(velocity)
    state.append(heading)
    state.append(ALL_GAP_LIST[entrance][exit][0])
    state.append(entrance)
    state.append(exit)
    ori_dis2des = ALL_GAP_LIST[entrance][exit][0]
    if entrance == 'w':
        ori_dis2des -= 20
    return state[0], state[1], velocity, heading, ori_dis2des-20, SPEED_LIMIT

def record_all_possible_ref_line():
    possible_ref_line_list = []
    possible_ref_line_distance2exit_list = []
    total_length = []
    for entrance in POSSIBLE_ENTRANCE:
        entrance_possible_ref_line = []
        entrance_possible_ref_line_distance2exit = []
        entrance_possible_total_length = []
        for exit in ENTRANCE_EXIT_RELATION[entrance]:
            ref_line = concatenate_ref_lane(entrance, exit)
            entrance_possible_ref_line.append(ref_line)
            entrance_possible_ref_line_distance2exit.append(record_ref_line_distance2exit(ref_line))
            entrance_possible_total_length.append(max(record_ref_line_distance2exit(ref_line)))
        possible_ref_line_list.append(dict(zip(ENTRANCE_EXIT_RELATION[entrance], entrance_possible_ref_line)))
        possible_ref_line_distance2exit_list.append(dict(zip(ENTRANCE_EXIT_RELATION[entrance], entrance_possible_ref_line_distance2exit)))
        total_length.append(dict(zip(ENTRANCE_EXIT_RELATION[entrance], entrance_possible_total_length)))
    return dict(zip(POSSIBLE_ENTRANCE, possible_ref_line_list)), dict(zip(POSSIBLE_ENTRANCE, possible_ref_line_distance2exit_list)), dict(zip(POSSIBLE_ENTRANCE, total_length))

def find_dis2des(entrance, exit, x, y):
    ref_line = ALL_REF_LINE[entrance][exit]
    gap_list = ALL_GAP_LIST[entrance][exit]
    index = np.argmin(np.sqrt((ref_line[:,0] - x)**2 + (ref_line[:,1] - y)**2))
    dis2des = gap_list[index]
    return dis2des


# --- NEW: Pedestrian Splines and Caching ---

def pedestrian_ref_line(crosswalk_loc, direction):
    """
    Generates orthogonal straight lines across the lanes for pedestrians.
    crosswalk_loc: 'n', 's', 'e', 'w'
    direction: 1 (left to right / top to bottom), -1 (right to left / bottom to top)
    """
    if crosswalk_loc == 's':
        x = np.linspace(-7.5, 7.5, 1000) if direction == 1 else np.linspace(7.5, -7.5, 1000)
        y = -22.5 * np.ones_like(x)
    elif crosswalk_loc == 'n':
        x = np.linspace(-7.5, 7.5, 1000) if direction == 1 else np.linspace(7.5, -7.5, 1000)
        y = 22.5 * np.ones_like(x)
    elif crosswalk_loc == 'e':
        y = np.linspace(7.5, -7.5, 1000) if direction == 1 else np.linspace(-7.5, 7.5, 1000)
        x = 22.5 * np.ones_like(y)
    elif crosswalk_loc == 'w':
        y = np.linspace(-7.5, 7.5, 1000) if direction == 1 else np.linspace(7.5, -7.5, 1000)
        x = -22.5 * np.ones_like(y)
    else:
        return None
    return np.vstack((x, y)).T

def default_pedestrian_state(crosswalk_loc, direction):
    """ Initializes human kinematic states at the edge of the crosswalk """
    velocity = 1.5 
    max_speed = 3.0
    if crosswalk_loc == 's':
        state = [-7.5, -22.5] if direction == 1 else [7.5, -22.5]
        heading = 0.0 if direction == 1 else math.pi
    elif crosswalk_loc == 'n':
        state = [-7.5, 22.5] if direction == 1 else [7.5, 22.5]
        heading = 0.0 if direction == 1 else math.pi
    elif crosswalk_loc == 'e':
        state = [22.5, 7.5] if direction == 1 else [22.5, -7.5]
        heading = 1.5 * math.pi if direction == 1 else 0.5 * math.pi
    elif crosswalk_loc == 'w':
        state = [-22.5, -7.5] if direction == 1 else [-22.5, 7.5]
        heading = 0.5 * math.pi if direction == 1 else 1.5 * math.pi
    
    dis2des = 15.0 # Total crosswalk distance
    return state[0], state[1], velocity, heading, dis2des, max_speed

def record_all_pedestrian_ref_line():
    possible_locs = ['n', 's', 'e', 'w']
    directions = [1, -1]
    ped_ref_line_list = {}
    ped_gap_list = {}
    
    for loc in possible_locs:
        ped_ref_line_list[loc] = {}
        ped_gap_list[loc] = {}
        for d in directions:
            ref_line = pedestrian_ref_line(loc, d)
            ped_ref_line_list[loc][d] = ref_line
            ped_gap_list[loc][d] = record_ref_line_distance2exit(ref_line)
            
    return ped_ref_line_list, ped_gap_list


print('Initialize roundabout environment...')
ALL_REF_LINE, ALL_GAP_LIST, REF_LINE_TOTAL_LENGTH = record_all_possible_ref_line()

# Cache the pedestrian paths for real-time tracking
PED_REF_LINE, PED_GAP_LIST = record_all_pedestrian_ref_line()

CONFLICT_RELATION = {'s': {'e': {'we': 99.78564593363033, 'wn': 99.78564593363033}, 'w': {'we': 154.76351633116838, 'wn': 154.76351633116838}, 'n': {'we': 127.27458121290213, 'wn': 127.27458121290213}}, 'w': {'e': {'se': 99.66532772509943, 'sw': 99.66532772509943, 'sn': 99.66532772509943}, 'n': {'se': 127.15365197519074, 'sw': 127.15365197519074, 'sn': 127.15365197519074}}}
CONFLICT_RELATION_STATE = {'s': {'e': {'we': (6.362653632019546, -16.60060155498124), 'wn': (6.362653632019546, -16.60060155498124)},
                                 'w': {'we': (6.362653632019546, -16.60060155498124), 'wn': (6.362653632019546, -16.60060155498124)},
                                 'n': {'we': (6.362653632019546, -16.60060155498124), 'wn': (6.362653632019546, -16.60060155498124)}},
                           'w': {'e': {'se': (6.362653632019546, -16.60060155498124), 'sw': (6.362653632019546, -16.60060155498124), 'sn': (6.362653632019546, -16.60060155498124)},
                                 'n': {'se': (6.362653632019546, -16.60060155498124), 'sw': (6.362653632019546, -16.60060155498124), 'sn': (6.362653632019546, -16.60060155498124)}}}
print('Initialize done')