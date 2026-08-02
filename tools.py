import copy
import math
import random
import re
import numpy as np
from idm_controller import VEH_L
from vehicle import Vehicle
from params import *

'''tools for simulator and agent'''
if Scenario_name == 'intersection':
    from scenario_environment import intersection_environment as environment
elif Scenario_name == 'merge':
    from scenario_environment import merge_environment as environment
elif Scenario_name == 'roundabout':
    from scenario_environment import roundabout_environment as environment
else:
    raise ValueError('no such environment, check Scenario_name in params')

def initialize_vehicles(num=4):
    '''generate ego vehicle and inter vehicles safely based on available map entrances'''
    ego_entrance = random.choice(POSSIBLE_ENTRANCE)
    ego_info = Vehicle(ego_entrance, random.choice(ENTRANCE_EXIT_RELATION[ego_entrance]), 'cav', 0)
    
    available_entrances = [e for e in POSSIBLE_ENTRANCE if e != ego_entrance]
    other_infos = []
    vehicle_id = 1
    
    # --- NEW: Guaranteed Pedestrian Spawn for Gap 4 Visuals ---
    ped_entrance = random.choice(available_entrances)
    ped_exit = random.choice(ENTRANCE_EXIT_RELATION[ped_entrance])
    
    # Initialize as a standard entity, but strictly override its type and physics
    ped_info = Vehicle(ped_entrance, ped_exit, 'nor', vehicle_id)
    ped_info.type = 'pedestrian'
    ped_info.behavior = 'crossing'
    ped_info.speed = 1.2  # Human walking speed (m/s)
    ped_info.max_speed = 2.0
    ped_info.dis2des = 20  # Spawn them close enough to the crosswalk to be visible
    
    other_infos.append(ped_info)
    vehicle_id += 1
    available_entrances.remove(ped_entrance)
    
    # --- Keep spawning the remaining HDV cars ---
    while len(other_infos) < (num - 1) and len(available_entrances) > 0:
        other_entrance = random.choice(available_entrances)
        other_exit = random.choice(ENTRANCE_EXIT_RELATION[other_entrance])
        
        other_info = Vehicle(other_entrance, other_exit, random.choice(['nor', 'agg', 'con']), vehicle_id)  
        
        if not if_passed_conflict_point(ego_info, other_info):
            other_infos.append(other_info)
            vehicle_id += 1
            
        available_entrances.remove(other_entrance)
        
    # Fallback: Guarantee at least one car spawns alongside the pedestrian
    if len(other_infos) == 1:
        while True:
            backup_entrance = random.choice([e for e in POSSIBLE_ENTRANCE if e != ego_entrance])
            backup_exit = random.choice(ENTRANCE_EXIT_RELATION[backup_entrance])
            backup_info = Vehicle(backup_entrance, backup_exit, random.choice(['nor', 'agg', 'con']), vehicle_id)
            if not if_passed_conflict_point(ego_info, backup_info):
                other_infos.append(backup_info)
                break

    return ego_info, other_infos

def cal_ttcp(speed_limit, veh_dis2cp, veh_v, veh_acc):
    if veh_acc > 0:
        t_acc2max = (speed_limit - veh_v) / veh_acc
        dis_acc2max = veh_v * t_acc2max + 0.5 * veh_acc * t_acc2max ** 2
        if dis_acc2max < veh_dis2cp:
            t_left = (veh_dis2cp - dis_acc2max) / speed_limit
            ttcp = t_acc2max + t_left
        else:
            v = np.sqrt(veh_v ** 2 + 2 * veh_dis2cp * veh_acc)
            ttcp = (v - veh_v) / veh_acc
    elif veh_acc < 0:
        dis_acc2stop = veh_v ** 2 / (2 * abs(veh_acc))
        if dis_acc2stop < veh_dis2cp:
            ttcp = 10000
        else:
            v = np.sqrt(veh_v ** 2 + 2 * veh_dis2cp * veh_acc)
            ttcp = (v - veh_v) / veh_acc
    else:
        if veh_v == 0:
            ttcp = 10000
        else:
            ttcp = veh_dis2cp / veh_v

    if math.isnan(ttcp):
        ttcp = 10000
    return ttcp

def get_leading_rearing_vehicle_in_same_lane(ego_info, other_info):
    leading_vehicle_info, rearing_vehicle_info = None, None
    return leading_vehicle_info, rearing_vehicle_info

def get_delta_ttcp(ego_info, other_info):
    if not if_passed_conflict_point(ego_info, other_info):
        ttcp_ego = cal_ttcp(ego_info.max_speed, get_dis2cp(ego_info, other_info), ego_info.speed, ego_info.acc)
        ttcp_other = cal_ttcp(other_info.max_speed, get_dis2cp(other_info, ego_info), other_info.speed, other_info.acc)
        delta_ttcp = ttcp_other - ttcp_ego
        return delta_ttcp
    else:
        raise ValueError('passed conflict point, should not calculate delta ttcp')

def evaluate_safety_with_conflict_vehicles(ego_info, other_info):
    if not if_passed_conflict_point(ego_info, other_info):
        ttcp_ego = cal_ttcp(ego_info.max_speed, get_dis2cp(ego_info, other_info), ego_info.speed, ego_info.acc)
        ttcp_other = cal_ttcp(other_info.max_speed, get_dis2cp(other_info, ego_info), other_info.speed, other_info.acc)
        delta_ttcp = ttcp_other - ttcp_ego
        if other_info.speed == 0 and get_dis2cp(other_info, ego_info) > VEH_L:
            dangerous_level = -1
        else:
            if delta_ttcp > 5:
                dangerous_level = 0
            elif 5 >= delta_ttcp > 1:
                dangerous_level = 1
            elif 1 >= delta_ttcp > -5:
                dangerous_level = 3
            else:
                dangerous_level = 2
    else:
        dangerous_level = 0
    return dangerous_level

def get_euclidean_distance(obj1, obj2):
    return math.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2)

def find_opponent(ego_info, other_infos, p=False):
    most_danger_opponent = other_infos[0]
    most_danger_score = 10000  # Lower is more dangerous
    cp_occupied = None
    
    for other in other_infos:
        # --- NEW: Pedestrian Euclidean Override ---
        if getattr(other, 'type', 'vehicle') == 'pedestrian':
            dist = get_euclidean_distance(ego_info, other)
            if dist < 20:  # If within 20 meters, extreme danger
                score = dist - 100 # Deeply negative to mathematically override all cars
                if score < most_danger_score:
                    most_danger_score = score
                    most_danger_opponent = other
        else:
            if not if_passed_conflict_point(ego_info, other):
                delta_ttcp = abs(get_delta_ttcp(ego_info, other))
                # Only rank vehicles if a critical pedestrian hasn't triggered an override
                if delta_ttcp < most_danger_score and most_danger_score >= 0:
                    most_danger_score = delta_ttcp
                    most_danger_opponent = other
                if delta_ttcp < 4:
                    cp_occupied = other
                    
    if cp_occupied is not None and most_danger_score >= 0:
        most_danger_opponent = cp_occupied
    return most_danger_opponent

def plot_figs(ego_info, other_info, ax, llm_output, instruction_info, retrieved_instruction_info):
    ax.cla()
    ax.axis('scaled')
    environment.scenario_outfit(ax)
    ax.scatter(ego_info.x, ego_info.y, color='purple')
    ax.plot(environment.ALL_REF_LINE[ego_info.entrance][ego_info.exit][:, 0], environment.ALL_REF_LINE[ego_info.entrance][ego_info.exit][:, 1], color='grey')
    ax.text(ego_info.x + 2, ego_info.y + 2, round(ego_info.speed, 2))

    if isinstance(other_info, list):
        for other in other_info:
            # --- NEW: Plot Pedestrians as Orange Stars ---
            if getattr(other, 'type', 'vehicle') == 'pedestrian':
                ax.scatter(other.x, other.y, color='orange', marker='*', s=100)
                ax.text(other.x + 2, other.y + 2, f'Ped:{other.id}, v:{round(other.speed, 2)}')
            else:
                hv_color = 'blue'
                if other.aggressiveness == 'con':
                    hv_color = 'green'
                elif other.aggressiveness == 'agg':
                    hv_color = 'red'
                opponent = find_opponent(ego_info, other_info)
                if other == opponent:
                    hv_color = 'black'
                ax.scatter(other.x, other.y, color=hv_color)
                ax.plot(environment.ALL_REF_LINE[other.entrance][other.exit][:, 0],
                        environment.ALL_REF_LINE[other.entrance][other.exit][:, 1], color='grey')
                ax.text(other.x + 2, other.y + 2, f'id:{other.id}, v:{round(other.speed, 2)}')
    else:
        ax.scatter(other_info.x, other_info.y, color='black')
        if getattr(other_info, 'type', 'vehicle') != 'pedestrian':
            ax.plot(environment.ALL_REF_LINE[other_info.entrance][other_info.exit][:, 0], environment.ALL_REF_LINE[other_info.entrance][other_info.exit][:, 1], color='grey')
        ax.text(other_info.x + 2, other_info.y + 2, round(other_info.speed, 2))
        
    if Scenario_name == 'intersection' or Scenario_name == 'roundabout':
        if isinstance(other_info, list):
            for i, other in enumerate(other_info):
                t = getattr(other, 'type', 'vehicle')
                if t == 'pedestrian':
                    ax.text(20, 60 + 10*i, f'Ped_{other.id} behavior:{other.behavior}')
                else:
                    ax.text(20, 60 + 10*i, f'HDV_{other.id} true type:{other.aggressiveness}')
        else:
            if getattr(other_info, 'type', 'vehicle') == 'pedestrian':
                ax.text(20, 60, f'Ped behavior:{other_info.behavior}')
            else:
                ax.text(20, 60, f'HDV true type:{other_info.aggressiveness}')
        ax.text(-90, -50, llm_output[0])
        ax.text(-90, -60, llm_output[1])
        ax.text(-90, -70, llm_output[2])
        ax.text(-90, -80, llm_output[3])
        ax.text(-90, 50, f'instruct: {instruction_info}')
    elif Scenario_name == 'merge':
        ax.text(0, 30, f'instruction: {instruction_info}')
        if getattr(other_info, 'type', 'vehicle') == 'pedestrian':
             ax.text(0, 20, f'Ped behavior:{other_info.behavior}')
        else:
             ax.text(0, 20, f'HDV true type:{other_info.aggressiveness}')
        ax.text(0, -30, llm_output[0])
        ax.text(0, -40, llm_output[1])
        ax.text(0, -50, llm_output[2])
        ax.text(0, -60, llm_output[3])

def get_dis2cp(vehicle1_info, vehicle2_info):
    dis2cp = vehicle1_info.dis2des - environment.CONFLICT_RELATION[vehicle1_info.entrance][vehicle1_info.exit][str(vehicle2_info.entrance) + str(vehicle2_info.exit)]
    return dis2cp

def if_passed_conflict_point(ego_info, other_info):
    # --- Gap 4 Multi-Agent List Handler ---
    if isinstance(other_info, list):
        for other in other_info:
            if not if_passed_conflict_point(ego_info, other):
                return False
        return True

    # --- Pedestrian Spatial Bypass ---
    if getattr(other_info, 'type', 'vehicle') == 'pedestrian':
        dist = get_euclidean_distance(ego_info, other_info)
        if dist > 10.0: 
            return True
        return False

    # --- Standard Vehicle Conflict Logic ---
    if str(ego_info.entrance) + str(ego_info.exit) in environment.CONFLICT_RELATION[other_info.entrance][other_info.exit]:
        ego_dis2cp = get_dis2cp(ego_info, other_info)
        other_dis2cp = get_dis2cp(other_info, ego_info)
        if ego_dis2cp > -3 and other_dis2cp > -3:
            return False
        else:
            return True
    else:
        return True

def acc2action(speed, acc):
    if acc > 0.5:
        return 'ACCELERATE'
    elif acc < -1:
        return 'DECELERATE'
    else:
        if speed >= SPEED_LIMIT:
            return 'ACCELERATE'
        elif speed <= 0:
            return 'DECELERATE'
        else:
            return 'IDLE'

def generate_comments(ego_info, other_info, ego_info_old, other_info_old):
    return ''

def generate_scenario_description(ego_info, other_infos):
    """
    UPGRADED: Replaces TTC with spatial distance for pedestrians, warning 
    the LLM of absolute right-of-way.
    """
    if not isinstance(other_infos, list):
        other_infos = [other_infos]

    ego_direction = 'going straight' if environment.if_going_straight(ego_info.entrance, ego_info.exit) else 'turning'
    msg = f'You are Veh#{ego_info.id}, you are now {ego_direction}. Here is the information for the surrounding entities you must track and negotiate with: \n'

    danger_list = []
    for other in other_infos:
        if getattr(other, 'type', 'vehicle') == 'pedestrian':
            dist = get_euclidean_distance(ego_info, other)
            if dist < 20: 
                danger_list.append((dist - 100, other))
        else:
            if str(other.entrance) + str(other.exit) in environment.CONFLICT_RELATION[ego_info.entrance][ego_info.exit]:
                if not if_passed_conflict_point(ego_info, other):
                    delta_ttcp = abs(get_delta_ttcp(ego_info, other))
                    danger_list.append((delta_ttcp, other))

    danger_list.sort(key=lambda x: x[0])

    if len(danger_list) == 0:
        msg += 'You currently have no direct conflict with any surrounding entities.'
        return msg

    for rank, (score, other) in enumerate(danger_list[:3]):
        if getattr(other, 'type', 'vehicle') == 'pedestrian':
            dist = get_euclidean_distance(ego_info, other)
            msg += f'--[Target {rank+1}] Pedestrian#{other.id}: A human pedestrian is at the crosswalk. ' \
                   f'Their speed is {round(other.speed, 2)} m/s, Euclidean distance from you is {round(dist, 2)}m. ' \
                   f'PEDESTRIANS HAVE ABSOLUTE RIGHT-OF-WAY! YOU MUST YIELD! \n'
        else:
            other_direction = 'going straight' if environment.if_going_straight(other.entrance, other.exit) else 'turning'
            msg += f'--[Target {rank+1}] Veh#{other.id}: Veh#{other.id} is {other_direction}. ' \
                   f'The conflict point is at ({environment.CONFLICT_RELATION_STATE[ego_info.entrance][ego_info.exit][str(other.entrance) + str(other.exit)]}). ' \
                   f'Veh#{other.id} speed is {round(other.speed, 2)}, distance to conflict point is {round(get_dis2cp(other, ego_info), 2)}. ' \
                   f'Your distance to this conflict point is {round(get_dis2cp(ego_info, other), 2)}. \n'
    return msg

def extract_hmi(retrieved_instruction_info):
    hmi_pattern = r"HMI info: ([^;]+);"
    hmi_info = re.search(hmi_pattern, retrieved_instruction_info)
    hmi_info_value = hmi_info.group(1) if hmi_info else None
    return hmi_info_value

def extract_instruction(retrieved_instruction_info):
    human_instruction_pattern = r"Human instruction: ([^\.]+)\."
    human_instructions = re.search(human_instruction_pattern, retrieved_instruction_info)
    human_instruction_value = human_instructions.group(1) if human_instructions else None
    return human_instruction_value

def scenario_experience_generator(ego_info, other_infos, llm_output, human_instruction):
    if isinstance(other_infos, list):
        other_info = find_opponent(ego_info, other_infos)
    else:
        other_info = other_infos
        
    if getattr(other_info, 'type', 'vehicle') == 'pedestrian':
        delta_ttcp = 0.0 # TTC is irrelevant for pedestrians, padded to 0.0
        delta_speed = round(ego_info.speed - other_info.speed, 1)
        dist = get_euclidean_distance(ego_info, other_info)
        delta_dis2des = round(dist, 1)
        
        # Ensure instruction exists as a number to prevent IndexError
        if human_instruction is None:
            instruction = 0
        else:
            instruction = -1
            
        # FIXED: Mapped driving style to "normal" to prevent KeyError in memory.py
        sce_descrip = f'Conflict info: instruction is {instruction}, disdes is {ego_info.dis2des}, delta_ttcp is {delta_ttcp}, delta_disdes is {delta_dis2des}, delta_v is {delta_speed}; ' \
                      f'Interaction vehicle driving style: normal; Interaction vehicle intention: {llm_output[2]}; HMI info: {llm_output[1]}; Human instruction: {human_instruction} (PEDESTRIAN).'
        return sce_descrip

    ttcp_ego = cal_ttcp(ego_info.max_speed, get_dis2cp(ego_info, other_info), ego_info.speed, ego_info.acc)
    ttcp_other = cal_ttcp(other_info.max_speed, get_dis2cp(other_info, ego_info), other_info.speed, other_info.acc)
    delta_ttcp = ttcp_other - ttcp_ego
    
    if delta_ttcp > 20:
        delta_ttcp = 20
    elif delta_ttcp < -20:
        delta_ttcp = -20
    else:
        delta_ttcp = round(delta_ttcp, 1)

    delta_speed = round(ego_info.speed - other_info.speed, 1)
    delta_dis2des = round(ego_info.dis2des - other_info.dis2des, 1)
    driving_style = llm_output[3]
    
    if driving_style not in ['CONSERVATIVE', 'AGGRESSIVE']:
        driving_style = 'normal'
    else:
        if driving_style == 'CONSERVATIVE':
            driving_style = 'conservative'
        else:
            driving_style = 'aggressive'
            
    if human_instruction is None:
        instruction = 0
    else:
        if 'go first' in human_instruction or 'go ahead' in human_instruction or 'accelerate' in human_instruction or 'speed up' in human_instruction or 'faster' in human_instruction:
            instruction = 1
        else:
            instruction = -1
            
    sce_descrip = f'Conflict info: instruction is {instruction}, disdes is {ego_info.dis2des}, delta_ttcp is {delta_ttcp}, delta_disdes is {delta_dis2des}, delta_v is {delta_speed}; ' \
                  f'Interaction vehicle driving style: {driving_style}; Interaction vehicle intention: {llm_output[2]}; HMI info: {llm_output[1]}; Human instruction: {human_instruction}.'
    return sce_descrip

def update_pos_from_dis2des_to_Cartesian(entrance, exit, dis2des):
    ref_line = environment.ALL_REF_LINE[entrance][exit]
    gap_list = environment.ALL_GAP_LIST[entrance][exit]
    index = np.argmin(abs(gap_list - dis2des))
    x = ref_line[index, 0]
    y = ref_line[index, 1]
    return x, y

def calculate_heading(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    heading = math.atan2(dy, dx)
    return heading

def generate_simulation_hdv_instruction(ego_info, other_info):
    # --- Isolate primary threat from the N-Agent array ---
    if isinstance(other_info, list):
        other_info = find_opponent(ego_info, other_info)
        
    if getattr(other_info, 'type', 'vehicle') == 'pedestrian':
        return 'Pedestrian crossing'
        
    if not if_passed_conflict_point(ego_info, other_info):
        if abs(get_dis2cp(other_info, ego_info)) < 4 or (get_dis2cp(other_info, ego_info) < 15 and other_info.speed > 4.9):
            instruction_list = ['I will go first', 'I will go ahead', 'I will accelerate', 'I will speed up', 'I will faster']
            return random.choice(instruction_list)
        else:
            if other_info.speed <= 0.2:
                instruction_list = ['I will yield', 'I will stop', 'I will decelerate', 'I will slow down', 'I will slower']
                return random.choice(instruction_list)
            else:
                return None
    else:
        return None

def adjust_acc(vehicle_info, acc):
    dis2stop_line = STOP_LINE[vehicle_info.entrance] - environment.REF_LINE_TOTAL_LENGTH[vehicle_info.entrance][vehicle_info.exit] + vehicle_info.dis2des
    if acc < 0 and dis2stop_line > 0:
        if dis2stop_line > 25:
            acc = 0
        else:
            acc = - vehicle_info.speed ** 2 / (2 * dis2stop_line)
    return acc

def kinematic_model(vehicle_info_old, acc):
    # --- NEW: Handle Pedestrian Orthogonal Movements ---
    if getattr(vehicle_info_old, 'type', 'vehicle') == 'pedestrian':
        ped_info = copy.deepcopy(vehicle_info_old)
        ped_info.speed += acc * Dt
        ped_info.speed = np.clip(ped_info.speed, 0, ped_info.max_speed)
        ped_info.dis2des -= ped_info.speed * Dt
        
        loc = ped_info.entrance
        direction = ped_info.exit
        ref_line = environment.PED_REF_LINE[loc][direction]
        gap_list = environment.PED_GAP_LIST[loc][direction]
        
        index = np.argmin(abs(gap_list - ped_info.dis2des))
        x = ref_line[index, 0]
        y = ref_line[index, 1]
        ped_info.heading = calculate_heading(ped_info.x, ped_info.y, x, y)
        ped_info.x, ped_info.y = x, y
        ped_info.acc = acc
        return ped_info

    # Original Vehicle Kinematics
    acc = adjust_acc(vehicle_info_old, acc)
    vehicle_info = copy.deepcopy(vehicle_info_old)
    vehicle_info.speed += acc * Dt
    vehicle_info.speed = vehicle_info.speed if vehicle_info.speed < vehicle_info.max_speed else vehicle_info.max_speed
    vehicle_info.speed = vehicle_info.speed if vehicle_info.speed > 0 else 0
    vehicle_info.dis2des -= vehicle_info.speed * Dt
    x, y = update_pos_from_dis2des_to_Cartesian(vehicle_info.entrance, vehicle_info.exit, vehicle_info.dis2des)
    vehicle_info.heading = calculate_heading(vehicle_info.x, vehicle_info.y, x, y)
    vehicle_info.x, vehicle_info.y = x, y
    vehicle_info.acc = acc
    return vehicle_info