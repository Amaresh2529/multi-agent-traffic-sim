import tools
from params import *
if Scenario_name == 'intersection':
    from scenario_environment import intersection_environment as environment
elif Scenario_name == 'merge':
    from scenario_environment import merge_environment as environment
elif Scenario_name == 'roundabout':
    from scenario_environment import roundabout_environment as environment
else:
    raise ValueError('no such environment, check Scenario_name in params')

class PRE_DEF_PROMPT():
    """
    These rules can be modified to test if changing prompt leads to different behaviour pattern of our agent.
    """

    def __init__(self):
        self.SYSTEM_MESSAGE_PREFIX = """You are now acting as an autonomous vehicle motion planner generating safe decisions. 
    Except for generating decisions, to improve safety, you should also share your intention to express what you are going to do to surrounding vehicles. 
    Now you are driving in a complex traffic scenario.

    --- PSYCHOLOGICAL BEHAVIOR DICTIONARY ---
    Use this to predict the intentions of surrounding tracking targets:
    - NORMAL (nor): Rational drivers who follow speed limits and standard right-of-way.
    - AGGRESSIVE (agg): Impatient drivers who over-speed and accelerate into conflict zones.
    - CONSERVATIVE (con): Cautious drivers who yield early.
    - HESITANT (hes): Nervous drivers who reduce speed heavily near conflicts but act unpredictably. Be decisive.
    - DISTRACTED (dis): Erratic drivers whose velocities fluctuate randomly. Highly dangerous, give them a wide berth.
    - PEDESTRIAN (ped): Human beings walking orthogonally across lanes. THEY HAVE ABSOLUTE RIGHT-OF-WAY.
    -----------------------------------------
    """

        self.TRAFFIC_RULES = """
    1. Try to keep a safe distance to the car in front of you.
    2. DO NOT change lane frequently. If you want to change lane, double-check the safety of vehicles on target lane.
    3. PEDESTRIANS ALWAYS HAVE THE RIGHT OF WAY. If a pedestrian is in the conflict zone, you MUST yield.
    """

        self.DECISION_CAUTIONS = """
    1. You must output a decision when you finish this task. Your final output decision must be unique and not ambiguous. For example you cannot say "I can either keep lane or accelerate at current time".
    2. In every conflict between you and the above vehicle, only one vehicle can pass first at each conflict. Based on above information, share your intention to other vehicles
    3. Your decision and intention to surrounding vehicle should be consistent.
    """

    def get_traffic_rules(self):
        return self.TRAFFIC_RULES

    def get_decision_cautions(self):
        return self.DECISION_CAUTIONS


ACTIONS_DESCRIPTION = {
    'IDLE': 'remain in the current lane with current speed',
    'FASTER': 'accelerate the vehicle',
    'SLOWER': 'decelerate the vehicle'
}

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator

class getAvailableActions:
    @prompts(name='Get Available Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
    def _get_available_actions(self, ego_info):
        """
        Get the list of currently available actions.
        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.
        :return: the list of available actions
        """
        actions = ['IDLE']
        if ego_info.speed < ego_info.max_speed:
            actions.append('FASTER')
        if ego_info.speed > 0:
            actions.append('SLOWER')
        return actions

    def inference(self, ego_info) -> str:
        outputPrefix = 'You can ONLY use one of the following actions: IDLE, FASTER, SLOWER\n '
        return outputPrefix

class isAccelerationConflictWithCar:
    def __init__(self) -> None:
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0
        self.acceleration = 3.0

    @prompts(name='Is Acceleration Conflict With Car',
             description="""useful when you want to know whether acceleration is safe with a specific car, ONLY when your decision is accelerate. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, ego_info, leading_vehicle_info) -> str:
        if leading_vehicle_info is not None:
            relativeSpeed = ego_info.speed + self.acceleration - leading_vehicle_info.speed
            distance = ego_info.dis2des - leading_vehicle_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                return f"acceleration is safe with Veh#{leading_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                return f"acceleration may not safe with Veh#{leading_vehicle_info.id}, should be careful if you want to accelerate. \n"
            elif 10 >= ttc > 5:
                return f'acceleration will cause danger, you can not accelerate. \n'
            else:
                return f'acceleration will cause serious danger, must slower your speed. \n'
        else:
            return f"acceleration is safe."

class isKeepSpeedConflictWithCar:
    def __init__(self) -> None:
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is Keep Speed Conflict With Car',
             description="""useful when you want to know whether keep speed is safe with a specific car, ONLY when your decision is keep_speed. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, ego_info, leading_vehicle_info, rearing_vehicle_info) -> str:
        message = ""
        if leading_vehicle_info is not None:
            relativeSpeed = ego_info.speed - leading_vehicle_info.speed
            distance = ego_info.dis2des - leading_vehicle_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                message += f"keep lane with current speed is safe with Veh#{leading_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                message += f"keep lane with current speed may not safe with Veh#{leading_vehicle_info.id}, should consider decelerate. \n"
            elif 10 >= ttc > 5:
                message += f'keep lane with current speed will cause danger, you should consider decelerate. \n'
            else:
                message += f'keep lane with current speed will cause serious danger, must decelerate. \n'

        if rearing_vehicle_info is not None:
            relativeSpeed = rearing_vehicle_info.speed - ego_info.speed
            distance = rearing_vehicle_info.dis2des - ego_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                message += f"keep lane with current speed is safe with Veh#{rearing_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                message += f"keep lane with current speed may not safe with Veh#{rearing_vehicle_info.id}, should consider accelerate. \n"
            elif 10 >= ttc > 5:
                message += f'keep lane with current speed will cause danger, you should consider accelerate. \n'
            else:
                message += f'keep lane with current speed will cause serious danger, must accelerate. \n'
        return message

class isDecelerationSafe:
    def __init__(self) -> None:
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
        self.deceleration = 6.0

    @prompts(name='Is Deceleration Safe',
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate.The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, ego_info, rearing_vehicle_info) -> str:
        if rearing_vehicle_info is not None:
            relativeSpeed = rearing_vehicle_info.speed - (ego_info.speed - self.deceleration)
            distance = rearing_vehicle_info.dis2des - ego_info.dis2des - self.VEHICLE_LENGTH * 2
            ttc = distance / relativeSpeed
            if ttc > 20:
                return f"deceleration with current speed is safe with Veh#{rearing_vehicle_info.id}. \n"
            elif 20 >= ttc > 10:
                return f"deceleration with current speed may not safe with Veh#{rearing_vehicle_info.id}. \n"
            elif 10 >= ttc > 5:
                return f'deceleration with current speed will cause danger, if you have no other choice, try not to decelerate so fast as much as possible. \n'
            else:
                return f"deceleration with current speed may be conflict with Veh#{rearing_vehicle_info.id}, you should maintain speed or accelerate. \n"
        else:
            return f"acceleration is safe."

def available_action(toolModels, ego_info):
    available_action_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableActions)), None)
    available_action = {}
    available_lanes_analysis = available_action_tool.inference(ego_info)
    available_action[available_action_tool] = available_lanes_analysis
    return available_action

def interaction_vehicle(ego_info, other_info):
    ego_direction = 'going straight' if environment.if_going_straight(ego_info.entrance, ego_info.exit) else 'turning'
    other_info_direction = 'going straight' if environment.if_going_straight(other_info.entrance, other_info.exit) else 'turning'
    other_info_action = tools.acc2action(other_info.speed, other_info.acc)
    accelerate_safety_analysis = check_safety_with_conflict_vehicles(ego_info, other_info)
    msg = ''
    msg += f'Your are now {ego_direction}, these are vehicles information you should pay attention and share your intention to them when making decision: \n'
    if not tools.if_passed_conflict_point(ego_info, other_info):
        msg += f'Your surrounding vehicle is now {other_info_direction}, its ACTUAL last action is {other_info_action}, ' \
               f'the position of conflict point between you and him is ({environment.CONFLICT_RELATION_STATE[ego_info.entrance][ego_info.exit][str(other_info.entrance) + str(other_info.exit)]}). ' \
               f'his speed is {round(other_info.speed, 1)}, distance to conflict point is {round(tools.get_dis2cp(other_info, ego_info), 1)}. ' \
               f'Your speed is {ego_info.speed}, distance to conflict point is {round(tools.get_dis2cp(ego_info, other_info), 1)}. ' \
               f'Based on your states and his states, for you, {accelerate_safety_analysis}. \n'
    else:
        msg += f'You has no conflict with Veh#{other_info.id}'
    return msg

def check_safety_in_current_lane(toolModels, ego_info, other_info):
    safety_analysis = {
        'acceleration_conflict': None,
        'keep_speed_conflict': None,
        'deceleration_conflict': None
    }

    acceleration_tool = next((tool for tool in toolModels if isinstance(tool, isAccelerationConflictWithCar)), None)
    keep_speed_tool = next((tool for tool in toolModels if isinstance(tool, isKeepSpeedConflictWithCar)), None)
    deceleration_tool = next((tool for tool in toolModels if isinstance(tool, isDecelerationSafe)), None)
    leading_vehicle_info, rearing_vehicle_info = tools.get_leading_rearing_vehicle_in_same_lane(ego_info, other_info)

    if leading_vehicle_info is not None:  
        safety_analysis['acceleration_conflict'] = acceleration_tool.inference(ego_info, leading_vehicle_info)
    if leading_vehicle_info is not None or rearing_vehicle_info is not None:
        safety_analysis['keep_speed_conflict'] = keep_speed_tool.inference(ego_info, leading_vehicle_info, rearing_vehicle_info)
    if rearing_vehicle_info is not None:  
        safety_analysis['deceleration_conflict'] = deceleration_tool.inference(ego_info, rearing_vehicle_info)
    return safety_analysis

def check_safety_with_conflict_vehicles(ego_info, other_info):
    dangerous_level = tools.evaluate_safety_with_conflict_vehicles(ego_info, other_info)

    if dangerous_level == 0:
        return 'acceleration is safe, you should FASTER'
    elif dangerous_level == 1:
        return 'acceleration may not safe, should be careful if you want to accelerate'
    elif dangerous_level == 2:
        return 'acceleration will cause danger, you can not accelerate'
    elif dangerous_level == 3:
        return 'acceleration will cause serious danger, consider decelerate.'  
    elif dangerous_level == -1:
        return 'surround vehicle stopped, better accelerate for efficiency'
    else:
        raise ValueError(f'dangerous_level is {dangerous_level}, not in 0/1/2/3 check prompt.py-check_safety_with_conflict_vehicles & tools.py-evaluate_safety_with_conflict_vehicles')

def check_conflict_point_occupied(ego_info, other_info):
    if abs(tools.get_dis2cp(other_info, ego_info)) < 4 and other_info.speed == 0:
        return 'You can not accelerate! A vehicle has stopped on your planning trajectory, you should let him pass first'
    else:
        return ''

def format_decision_info(available_action_msg, interaction_vehicle_msg, current_lane_safety_msg, conflict_lane_safety_msg, conflict_point_occupied):
    formatted_message = ""

    formatted_message += "\nAvailable Actions:\n"
    for tool, action_info in available_action_msg.items():
        formatted_message += f"- {action_info}\n"

    formatted_message += "\nSurrounding Vehicle Information:\n"
    formatted_message += interaction_vehicle_msg

    formatted_message += conflict_point_occupied
    return formatted_message