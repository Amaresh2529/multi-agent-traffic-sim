import copy
import time
from prompt_llm import *
import subprocess
import os
import tools  

def run_llama3(prompt):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    process = subprocess.Popen(
        ["ollama", "run", "llama3"],  
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    output, _ = process.communicate(input=prompt)
    return output

class LLM_Agent():
    def __init__(self):
        self.pre_prompt = PRE_DEF_PROMPT()
        self.toolModels = [
            getAvailableActions(),
            isAccelerationConflictWithCar(),
            isKeepSpeedConflictWithCar(),
            isDecelerationSafe(),
        ]

    def llm_run(self, decision_info, instruction_info, ego_info, other_infos, memory_tools, if_train_mode=True):
        current_scenario = self.prompt_engineer(ego_info, other_infos)  
        llm_output = self.send_to_llm(decision_info, instruction_info, current_scenario, memory_tools, if_train_mode)
        if if_train_mode:
            self.add_memory2dataset(llm_output, ego_info, other_infos, instruction_info, memory_tools)
        return llm_output

    def relative_memory(self, memory, prompt_info):
        experience = ""
        extract_prompt = prompt_info.strip().split('\n')
        query_scenario = '\n'.join(extract_prompt)
        past_decisions = memory.retrieveMemory(query_scenario, top_k=2)[0]
        for past_decision in past_decisions:
            experience += f"- Last time {past_decision['negotiation_result']}, you choose to {past_decision['final_action']}, it is {past_decision['comments']}\n"
        experience += f"Above messages are some examples of how you make a decision in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario."
        return experience

    def send_to_llm(self, decision_info, instruction_info, current_scenario, memory_tools, if_train_mode):
        if if_train_mode:
            prompt = (f"{self.pre_prompt.SYSTEM_MESSAGE_PREFIX}"
                      f"You, the 'ego' car, are now driving. You have already driven for some seconds.\n"
                      "Here are your attention points:"
                      f"{self.pre_prompt.get_decision_cautions()}\n"
                      "Here is the current multi-agent scenario:\n"
                      f"{current_scenario}\n"
                      "Your decision and shared intention to other vehicles at the last moment are as follows, try not to change them frequently unless it will cause danger:"
                      f"Your last decision is {decision_info[0]}, your shared intention to surrounding vehicles is {decision_info[1]} \n"
                      "Here is a message from the driver of a surrounding vehicle. You should decide whether to accept his advice based on the current scenario, and express your intention to surrounding vehicles\n"
                      "They will not lie to you, and when they want to slow down or yield, you should better faster for efficiency\n"
                      f"Surrounding vehicle says: {instruction_info}\n"
                      "You need to consider the following questions step by step to reach your global conclusion:\n"
                      "1. What are the likely intentions of ALL active targets in the intersection?\n"
                      "2. Based on their past actions, what is the driving style of EACH target? (AGGRESSIVE, CONSERVATIVE, HESITANT, DISTRACTED, or PEDESTRIAN)\n"
                      "3. What single optimal action should you take to navigate ALL threats safely? STRICT RULE: IF ANY TARGET IS A PEDESTRIAN, YOU MUST CHOOSE SLOWER OR IDLE. (Options: IDLE, FASTER, SLOWER)\n"
                      "4. What multi-target eHMI signal will you broadcast to share your intention with the scene? (e.g., Yielding to Pedestrian, signaling Veh#2 to hold)\n"
                      "After reasoning, provide your final conclusion in the following format:\n"
                      "```\n"
                      "ONLY OUTPUT YOUR FINAL DECISION IN THE FOLLOWING FORMAT AND NOTHING ELSE:\n"
                      "```\n"
                      "Final Answer: \n"
                      "    \"thoughts\": {\"<your thoughts when consider the above questions step by step to reach your conclusion)>\"},\n"
                      "    \"surrounding vehicle intention\": {\"<Summary of intentions for all active targets)>\"},\n"
                      "    \"style\": {\"<Summary of driving styles for all active targets)>\"},\n"
                      "    \"decision\": {\"<ego car's global decision, ONE of: IDLE, FASTER, SLOWER)>\"},\n"
                      "    \"your intention to share\": {\"<Your eHMI broadcast to all targets (less than eight words, output with lowercase)>\"} \n"
                      "```\n")
        else:  
            prompt = (f"{self.pre_prompt.SYSTEM_MESSAGE_PREFIX}"
                      f"You, the 'ego' car, are now driving. You have already driven for some seconds.\n"
                      "Here are your attention points:"
                      f"{self.pre_prompt.get_decision_cautions()}\n"
                      "Here is the current multi-agent scenario:\n"
                      f"{current_scenario}\n"
                      "Your decision and shared intention to other vehicles at the last moment are as follows, try not to change them frequently unless it will cause danger:"
                      f"Your last decision is {decision_info[0]}, your shared intention to surrounding vehicles is {decision_info[1]} \n"
                      "Here is a message from the driver of a surrounding vehicle. You should decide whether to accept his advice based on the current scenario, and express your intention to surrounding vehicles\n"
                      f"Surrounding vehicle says: {instruction_info}\n"
                      "You need to consider the following questions step by step to reach your conclusion:\n"
                      "1. What are the likely intentions of ALL active targets in the intersection?\n"
                      "2. Based on their past actions, what is the driving style of EACH target? (AGGRESSIVE, CONSERVATIVE, HESITANT, DISTRACTED, or PEDESTRIAN)\n"
                      "3. What single optimal action should you take? STRICT RULE: IF ANY TARGET IS A PEDESTRIAN, YOU MUST YIELD AND CHOOSE SLOWER OR IDLE. (Options: IDLE, FASTER, SLOWER)\n"
                      "4. What distinct eHMI signal will you broadcast to the targets based on your action? (e.g., I will ...)\n"
                      "ONLY OUTPUT YOUR FINAL DECISION IN THE FOLLOWING FORMAT AND NOTHING ELSE (DO NOT OUTPUT YOUR THOUGHTS):\n"
                      "```\n"
                      "Final Answer: \n"
                      "    \"surrounding vehicle intention\": {\"<Summary of intentions for all active targets)>\"},\n"
                      "    \"style\": {\"<Summary of driving styles for all active targets)>\"},\n"
                      "    \"decision\": {\"<ego car's global decision, ONE of: IDLE, FASTER, SLOWER)>\"},\n"
                      "    \"your intention to share\": {\"<Your eHMI broadcast to all targets (LESS THAN EIGHT WORDS)>\"} \n"
                      "```\n")
        
        llm_response = run_llama3(prompt)
        llm_action, hmi_advice, inter_intention, inter_style = self.extract_output(llm_response)
        return [llm_action, hmi_advice, inter_intention, inter_style]

    def extract_output(self, response_content):
        try:
            start = response_content.find('"decision": {') + len('"decision": {')
            end = response_content.find('}', start)
            decision = response_content[start:end].strip().strip('"')

            start_advice = response_content.find('"your intention to share": {') + len('"your intention to share": {')
            end_advice = response_content.find('}', start_advice)
            advice = response_content[start_advice:end_advice].strip().strip('"')

            start_intention = response_content.find('"surrounding vehicle intention": {') + len('"surrounding vehicle intention": {')
            end_intention = response_content.find('}', start_intention)
            intention = response_content[start_intention:end_intention].strip().strip('"')

            start_style = response_content.find('"style": {') + len('"style": {')
            end_style = response_content.find('}', start_style)
            style = response_content[start_style:end_style].strip().strip('"')

            if "FASTER" in decision.upper():
                decision = "FASTER"
            elif "SLOWER" in decision.upper():
                decision = "SLOWER"
            elif "IDLE" in decision.upper():
                decision = "IDLE"

            llm_output = [decision, advice, intention, style]
            for i, output in enumerate(llm_output):
                if len(output) > 200: # Expanded threshold to allow for multi-target text blocks
                    if i == 0:
                        llm_output[i] = 'SLOWER'
                    elif i == 3:
                        llm_output[i] = 'GENERAL'
                    else:
                        llm_output[i] = None
            decision, advice, intention, style = llm_output
            return decision, advice, intention, style
        except Exception as e:
            print(f"Error when extract decision: {e}")
            return None, None, None, None

    def prompt_engineer(self, ego_info, other_infos):
        # --- NEW: Process N-Agents dynamically ---
        if not isinstance(other_infos, list):
            other_infos = [other_infos]

        msg0 = available_action(self.toolModels, ego_info)
        base_prompt_info = f"\nAvailable Actions:\n"
        for tool, action_info in msg0.items():
            base_prompt_info += f"- {action_info}\n"

        base_prompt_info += "\n--- MULTI-TARGET KINEMATIC ASSESSMENTS ---\n"
        
        for i, other in enumerate(other_infos):
            if getattr(other, 'type', 'vehicle') == 'pedestrian':
                base_prompt_info += f"Target {i+1} (Pedestrian): SAFETY WARNING. A pedestrian is in your trajectory. YOU MUST YIELD.\n\n"
            else:
                msg1 = interaction_vehicle(ego_info, other)
                safety_msg2 = check_safety_with_conflict_vehicles(ego_info, other)
                safety_msg3 = check_conflict_point_occupied(ego_info, other)
                base_prompt_info += f"Target {i+1} (Vehicle #{other.id}):\n{msg1}{safety_msg2}\n{safety_msg3}\n\n"
        
        multi_agent_msg = tools.generate_scenario_description(ego_info, other_infos)
        prompt_info = base_prompt_info + "\n--- MULTI-AGENT RADAR ---\n" + multi_agent_msg
        return prompt_info

    def add_memory2dataset(self, llm_output, ego_info, other_infos, instruction_info, memory_tools):
        # --- NEW: Global memory logging for N-Agents ---
        if not isinstance(other_infos, list):
            other_infos = [other_infos]

        # Prioritize pedestrian logging
        for other in other_infos:
            if getattr(other, 'type', 'vehicle') == 'pedestrian':
                action = llm_output[0]
                if action in ['SLOWER', 'IDLE']:  
                    sce_descrip = tools.scenario_experience_generator(ego_info, other_infos, llm_output, instruction_info)
                    memory_tools.addMemory(sce_descrip, action)
                return 

        action = llm_output[0]
        if action not in ['FASTER', 'SLOWER', 'IDLE']:
            return

        sce_descrip = tools.scenario_experience_generator(ego_info, other_infos, llm_output, instruction_info)
        
        # Evaluate global safety delta to decide if this multi-agent scenario is worth memorizing
        save_memory = False
        for other in other_infos:
            if not tools.if_passed_conflict_point(ego_info, other):
                current_delta_ttcp = tools.get_delta_ttcp(ego_info, other)
                ego_info_next = copy.deepcopy(ego_info)
                
                ego_acc = 2 if action == 'FASTER' else (-2 if action == 'SLOWER' else 0)
                ego_info_next.acc = ego_acc
                ego_info_next.speed += ego_acc * Dt
                next_delta_ttcp = tools.get_delta_ttcp(ego_info_next, other)
                
                # If the action successfully resolves a critical TTCP for ANY target, log it globally
                if abs(current_delta_ttcp) > 5 or abs(next_delta_ttcp) >= abs(current_delta_ttcp):
                    save_memory = True
                    break
                    
        if save_memory:
            memory_tools.addMemory(sce_descrip, action)