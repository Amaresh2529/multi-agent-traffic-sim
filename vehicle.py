from sqlalchemy.sql.functions import random
import random
from params import *
import tools

if Scenario_name == 'intersection':
    from scenario_environment import intersection_environment as environment
elif Scenario_name == 'merge':
    from scenario_environment import merge_environment as environment
elif Scenario_name == 'roundabout':
    from scenario_environment import roundabout_environment as environment
else:
    raise ValueError('no such environment, check Scenario_name in params')


class Vehicle:
    def __init__(self, entrance, exit, aggressiveness, id):
        self.id = id
        self.entrance = entrance
        self.exit = exit
        self.aggressiveness = aggressiveness # 'nor', 'agg', 'con', 'hes', 'dis'
        self.initialize_info()

    def initialize_info(self):
        x, y, speed, heading, dis2des, max_speed = environment.default_exit_and_state(self.entrance, self.exit)
        self.x = x
        self.y = y
        self.heading = heading
        if self.entrance == 'm':
            random_init_dis2des = 0
        else:
            random_init_dis2des = random.randint(0, 30)
        self.dis2des = dis2des - random_init_dis2des
        self.x, self.y = tools.update_pos_from_dis2des_to_Cartesian(self.entrance, self.exit, self.dis2des)
        
        # --- NEW: Kinematic modifiers for extended human intents ---
        if self.aggressiveness == 'hes':  # Hesitant
            self.speed = speed * 0.6  # Hesitant drivers approach much slower
            self.max_speed = max_speed * 0.7
        elif self.aggressiveness == 'dis':  # Distracted
            self.speed = speed * random.uniform(0.8, 1.2)  # Erratic initial speed
            self.max_speed = max_speed
        elif self.aggressiveness == 'agg': # Aggressive
            self.speed = speed * 1.1
            self.max_speed = max_speed * 1.2
        elif self.aggressiveness == 'con': # Conservative
            self.speed = speed * 0.8
            self.max_speed = max_speed * 0.9
        else: # Normal ('nor')
            self.speed = speed
            self.max_speed = max_speed
            
        self.acc = 0


class Pedestrian:
    def __init__(self, start_node, end_node, behavior, ped_id):
        self.id = ped_id
        self.type = 'pedestrian'
        self.behavior = behavior  # 'cautious', 'normal', 'jaywalking'
        
        # Human kinematics
        self.speed = 1.5  # Standard walking speed in m/s
        self.max_speed = 3.0  # Running speed
        self.acc = 0.0
        
        # Spatial coordinates
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.dis2des = 0.0  # Distance to the end of the crosswalk
        
        self.entrance = start_node
        self.exit = end_node

    def initialize_info(self):
        # We will map this to the orthogonal crosswalk coordinates later
        pass