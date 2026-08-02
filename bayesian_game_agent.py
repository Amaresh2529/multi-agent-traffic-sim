import copy
import tools
import random
import numpy as np
from params import *

if Scenario_name == 'intersection':
    from scenario_environment import intersection_environment as environment
elif Scenario_name == 'merge':
    from scenario_environment import merge_environment as environment
elif Scenario_name == 'roundabout':
    from scenario_environment import roundabout_environment as environment
else:
    raise ValueError('no such environment, check Scenario_name in params')

class Bayesian_Agent:
    def __init__(self, hdv_info, cav_infos, action_type):
        # --- Gap 4: Multi-Agent List Wrapper ---
        self.is_multi_agent = isinstance(hdv_info, list)
        if self.is_multi_agent:
            self.agents = [Bayesian_Agent(hdv, cav_infos, action_type) for hdv in hdv_info]
            return
            
        self.ego_info = hdv_info
        self.action_type = action_type
        
        # --- Gap 2: Pedestrian Bypass for HDV Physics ---
        if getattr(self.ego_info, 'type', 'vehicle') == 'pedestrian':
            self.is_pedestrian = True
            return
        else:
            self.is_pedestrian = False
            
        self.ego_state = [hdv_info.x, hdv_info.y, hdv_info.speed, hdv_info.heading, hdv_info.dis2des, hdv_info.entrance, hdv_info.exit]
        self.ego_central_vertices = environment.ALL_REF_LINE[self.ego_state[5]][self.ego_state[6]]
        self.ego_aggressiveness = hdv_info.aggressiveness
        
        if not isinstance(cav_infos, list):
            cav_infos = [cav_infos]
            
        self.inter_infos = cav_infos
        self.inter_states = []
        self.inter_central_vertices = []
        
        for cav in self.inter_infos:
            state = [cav.x, cav.y, cav.speed, cav.heading, cav.dis2des, cav.entrance, cav.exit]
            self.inter_states.append(state)
            self.inter_central_vertices.append(environment.ALL_REF_LINE[state[5]][state[6]])
        
        self.aggressiveness_distribution = [0.10, 0.35, 0.35, 0.10, 0.10] 
        self.max_speed = self.get_max_speed()

    def get_max_speed(self):
        if self.ego_aggressiveness == 'agg':
            return Target_speed[0]
        elif self.ego_aggressiveness == 'nor':
            return Target_speed[1]
        elif self.ego_aggressiveness == 'con':
            return Target_speed[2]
        elif self.ego_aggressiveness == 'hes':
            return Target_speed[2] * 0.7  
        elif self.ego_aggressiveness == 'dis':
            return Target_speed[1]  
        else:
            return Target_speed[0]

    def kinematic_model(self, state, action, temp):
        state_now = copy.deepcopy(state)
        state_now[2] += Action_space[action, 0] * Dt  
        if state[0] == self.ego_state[0] and not temp:
            if state_now[2] > self.max_speed:
                state_now[2] = self.max_speed
            if state_now[2] < 0:
                state_now[2] = 0
        state_now[4] -= state_now[2] * Dt
        state_now[0], state_now[1] = tools.update_pos_from_dis2des_to_Cartesian(state_now[5], state_now[6], state_now[4])
        state_now[3] = tools.calculate_heading(state[0], state[1], state_now[0], state_now[1])
        return state_now

    def reward_weight(self, aggressiveness):
        if aggressiveness == 'agg':
            return Weight_hv[0]
        elif aggressiveness == 'nor':
            return Weight_hv[1]
        elif aggressiveness == 'con':
            return Weight_hv[2]
        elif aggressiveness == 'hes':
            return Weight_hv[2]  
        elif aggressiveness == 'dis':
            return Weight_hv[1]  
        else:
            return Weight_hv[1]

    def get_ttc_thr(self, aggressiveness):
        if aggressiveness == 'agg':
            return 2
        elif aggressiveness == 'nor':
            return 6
        elif aggressiveness == 'con':
            return 7
        elif aggressiveness == 'hes':
            return 9  
        elif aggressiveness == 'dis':
            return 3  
        else:
            return 6

    def update_state(self, output_action=False): 
        # --- Gap 4: Output Array Mapper ---
        if getattr(self, 'is_multi_agent', False):
            return [agent.update_state(output_action) for agent in self.agents]

        # --- Gap 2: Pedestrian Action Bypass ---
        if getattr(self, 'is_pedestrian', False):
            if not output_action:
                return tools.kinematic_model(self.ego_info, 0)
            else:
                return 0

        passed_all = True
        for inter_info in self.inter_infos:
            if not tools.if_passed_conflict_point(self.ego_info, inter_info):
                passed_all = False
                break

        if not passed_all:
            r_total = np.zeros(Action_length)
            
            for target_idx, inter_info in enumerate(self.inter_infos):
                if tools.if_passed_conflict_point(self.ego_info, inter_info):
                    continue 
                    
                r_target = np.zeros(Action_length)
                for inter, inter_aggressiveness in enumerate(['agg', 'nor', 'con', 'hes', 'dis']):
                    nash_equilibrium_solution = self.nash_equilibrium(target_idx, inter_aggressiveness)
                    if not nash_equilibrium_solution:
                        inter_pure_strategy = 0
                    else:
                        inter_pure_strategy = nash_equilibrium_solution[0][1]

                    for ego_action in range(Action_length):
                        r_target[ego_action] += self.aggressiveness_distribution[inter] * \
                                         self.reward(target_idx, ego_action, inter_pure_strategy, inter_vehicle_aggressiveness=inter_aggressiveness)[0] 
                r_total += r_target
                
            bayesian_pure_strategy = np.argmax(r_total)
            acc = Action_space[bayesian_pure_strategy, 0]
        else:
            acc = max(Acceleration_list)
            
        if not output_action:
            return tools.kinematic_model(self.ego_info, acc)
        else:
            return acc

    def nash_equilibrium(self, target_idx, inter_vehicle_aggressiveness):
        nash_matrix = np.zeros((Action_length, Action_length))
        ego_best_response, inter_best_response = self.get_best_response(target_idx, inter_vehicle_aggressiveness)
        for act in range(Action_length):
            nash_matrix[act, inter_best_response[act]] += 1
            nash_matrix[ego_best_response[act], act] += 1
        _ = [i.tolist() for i in np.where(nash_matrix == 2)]
        return list(zip(*_))

    def get_best_response(self, target_idx, inter_vehicle_aggressiveness):
        ego_reward_matrix = np.zeros((Action_length, Action_length))
        inter_reward_matrix = np.zeros((Action_length, Action_length))
        for act1 in range(Action_length):  
            for act2 in range(Action_length):  
                ego_reward, inter_reward = self.reward(target_idx, act1, act2, inter_vehicle_aggressiveness)
                ego_reward_matrix[act1, act2] = ego_reward
                inter_reward_matrix[act1, act2] = inter_reward

        inter_best_response = [np.argmax(inter_reward_matrix[act, :]) for act in range(Action_length)]
        ego_best_response = [np.argmax(ego_reward_matrix[:, act]) for act in range(Action_length)]
        return ego_best_response, inter_best_response

    def reward(self, target_idx, act1, act2, inter_vehicle_aggressiveness):
        ego_state = self.kinematic_model(state=self.ego_state, action=act1, temp=True)
        inter_state_base = self.inter_states[target_idx]
        inter_state = self.kinematic_model(state=inter_state_base, action=act2, temp=True)

        ego_dis2cv = np.amin(np.linalg.norm(self.ego_central_vertices - ego_state[0:2], axis=1))
        inter_dis2cv = np.amin(np.linalg.norm(self.inter_central_vertices[target_idx] - inter_state[0:2], axis=1))
        
        ego_reward1 = - max(0, ego_dis2cv) * 20 if environment.if_right_turning(self.ego_state[5], self.ego_state[6]) == 'rt' else - max(0.1, ego_dis2cv) * 10
        inter_reward1 = - max(0, inter_dis2cv) * 20 if environment.if_right_turning(inter_state[5], inter_state[6]) == 'rt' else - max(0.1, inter_dis2cv) * 10

        ego_reward2 = ego_state[2]
        inter_reward2 = inter_state[2]

        ego_destination = self.ego_central_vertices[-1]
        inter_destination = self.inter_central_vertices[target_idx][-1]
        
        ego_reward3 = - ((ego_state[0] - ego_destination[0])**2 + (ego_state[1] - ego_destination[1])**2)**0.5
        inter_reward3 = - ((inter_state[0] - inter_destination[0])**2 + (inter_state[1] - inter_destination[1])**2)**0.5

        dis = ((ego_state[0] - inter_state[0])**2 + (ego_state[1] - inter_state[1])**2)**0.5
        ego_ttc_thr = self.get_ttc_thr(self.ego_aggressiveness)
        inter_ttc_thr = self.get_ttc_thr(inter_vehicle_aggressiveness)
        
        ego_ttc = (dis / ego_state[2]) / ego_ttc_thr if ego_state[2] != 0 else float('inf')
        inter_ttc = (dis / inter_state[2]) / inter_ttc_thr if inter_state[2] != 0 else float('inf')
        
        ego_reward4 = (- 1/ego_ttc) if ego_ttc != 0 else 0
        inter_reward4 = (- 1/inter_ttc) if inter_ttc != 0 else 0
        
        ego_reward = np.array([ego_reward1, ego_reward2, ego_reward3, ego_reward4])
        inter_reward = np.array([inter_reward1, inter_reward2, inter_reward3, inter_reward4])
        return np.dot(ego_reward, self.reward_weight(self.ego_aggressiveness)), np.dot(inter_reward, self.reward_weight(inter_vehicle_aggressiveness))

    def state_without_inter_vehicle(self):
        reward_without_iv = []
        for act in range(Action_length):
            self_state = self.kinematic_model(state=self.ego_state, action=act, temp=True)
            dis2cv = np.amin(np.linalg.norm(self.ego_central_vertices - self_state[0:2], axis=1))
            
            reward1 = - max(0.1, dis2cv) * 20 if environment.if_right_turning(self.ego_state[5], self.ego_state[6]) == 'rt' else - max(0.1, dis2cv) * 10
            reward2 = self_state[2]
            destination = self.ego_central_vertices[-1]
            reward3 = - abs(self_state[0] - destination[0]) - abs(self_state[1] - destination[1]) * 2
            
            reward = np.array([reward1, reward2, reward3, 0])
            reward_without_iv.append(np.dot(reward, self.reward_weight(self.ego_aggressiveness)) + reward2)
            
        return self.kinematic_model(state=self.ego_state, action=np.argmax(reward_without_iv), temp=False)