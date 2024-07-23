import csv
import math
import numpy as np
import time

from mrav.mcity_mr_av import (
    MRAVTemplateMcity,
)  # This Python class is a basic component for any developed AV decision-making module and the user should inherit from it.


class AVDecisionMakingModule(MRAVTemplateMcity):
    """This is an example AV decision making module that reads a logged trajectory from a file and follows it."""

    def initialize_av_algorithm(self):
        """This function will be used to initialize the developed AV ddecision-making module. In this example, we read the predefined trajectory from a file."""
        trajectory = []
        with open("/baseline_av_data/baseline_av_trajectory.csv", "r") as f:
            reader = csv.reader(f)
            trajectory = []
            for row in reader:
                orientation = float(row[3])
                if orientation > math.pi:
                    orientation -= 2 * math.pi
                trajectory.append(
                    {
                        "x": float(row[1]),
                        "y": float(row[2]),
                        "orientation": orientation,
                        "velocity": float(row[4]),
                    }
                )
        self.trajectory = {
            "x_vector": np.array([point["x"] for point in trajectory]),
            "y_vector": np.array([point["y"] for point in trajectory]),
            "orientation_vector": np.array(
                [point["orientation"] for point in trajectory]
            ),
            "velocity_vector": np.array([point["velocity"] for point in trajectory]),
        }
        self.trajectory_index = 0

    def derive_planning_result(self, step_info):
        """This function will be used to compute the planning results based on the observation from "step_info". In this example, we find the closest point in the predefined trajectory and return the next waypoint as the planning results."""
        # parse the step_info
        av_state = step_info["av_info"]
        tls_info = step_info["tls_info"]
        av_context_info = step_info["av_context_info"]
        # find the closest point in the predefined trajectory
        current_x = av_state["x"]
        current_y = av_state["y"]
        if self.trajectory_index > len(self.trajectory["x_vector"]) - 1:
            next_x = self.trajectory["x_vector"][-1]
            next_y = self.trajectory["y_vector"][-1]
        else:
            next_x = self.trajectory["x_vector"][self.trajectory_index]
            next_y = self.trajectory["y_vector"][self.trajectory_index]

        #Parse AV current info
        AV_longSpeed = av_state['speed_long']
        AV_latSpeed = av_state['speed_lat']
        AV_xPos = av_state['x']
        AV_yPos = av_state['y']
        AV_orient = av_state['orientation']

        AV_vel = AV_longSpeed + AV_latSpeed
        nextOrient = self.trajectory["orientation_vector"][self.trajectory_index]
        # Check if there is a leading vehicle   
        #for info in av_context_info.items():      
            

        # Check if there is a leading vehicle and a vehicle in adjacent lane
        for info in av_context_info.items():
            if info['leading_info']['is_leading_cav']:
                LV_xPos = info['x']
                LV_yPos = info['y']
                LV_longSpeed = info['speed_long']
                LV_latSpeed = info['speed_lat']
                LV_dist = math.sqrt((LV_xPos - AV_xPos)**2 + (LV_yPos - AV_yPos)**2)
                
                
                print("there is a LV")
                is_LV = True
                # Update with leading vehicle info acc. to LV distance
                if LV_dist >= 15:
                    AV_vel = LV_longSpeed + LV_longSpeed
                else:
                    AV_vel = LV_longSpeed + LV_longSpeed - 2

            else:
                is_LV = False
                AV_vel = AV_longSpeed + AV_latSpeed
            
            if info['lane_id']!=av_state['lane_id']:
                is_adj = True
                adj_xPos = info['x']
                adj_yPos = info['y']
                adj_longSpeed = info['speed_long']
                adj_latSpeed = info['speed_lat']
                adj_dist = d = math.sqrt((adj_xPos - AV_xPos)**2 + (adj_yPos - AV_yPos)**2)

                if adj_dist<=15:
                    AV_vel = adj_longSpeed + adj_longSpeed - 2
                elif is_LV == True and adj_dist>=15:
                    if LV_dist >= 15:
                        AV_vel = LV_longSpeed + LV_longSpeed
                    else:
                        AV_vel = LV_longSpeed + LV_longSpeed - 2
                elif is_LV == False:
                    AV_vel = AV_longSpeed + AV_latSpeed                

            else:
                is_adj = False
            break

        print("current AV position:", current_x, current_y)
        print("next AV position:", next_x, next_y)
        planning_result = {
            "timestamp": time.time(),
            "time_resolution": 0.1,
            "next_x": self.trajectory["x_vector"][self.trajectory_index],
            "next_y": self.trajectory["y_vector"][self.trajectory_index],
            "next_speed": AV_vel,
            "next_orientation": self.trajectory["orientation_vector"][self.trajectory_index],
        }
        self.trajectory_index += 1
        return planning_result


# Create an instance of the AV decision-making module and run it
av_decision_making_module = AVDecisionMakingModule()
av_decision_making_module.run()