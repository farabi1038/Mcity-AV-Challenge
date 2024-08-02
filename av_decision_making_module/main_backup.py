import csv
import math
import numpy as np
import time
import scipy.integrate as spi
from mrav.mcity_mr_av import MRAVTemplateMcity  # Ensure this module is available in your environment

class AVDecisionMakingModule(MRAVTemplateMcity):
    """This is an example AV decision-making module that reads a logged trajectory from a file and follows it."""

    def initialize_av_algorithm(self):
        """This function will be used to initialize the developed AV decision-making module. In this example, we read the predefined trajectory from a file."""
        trajectory = []
        with open("/baseline_av_data/baseline_av_trajectory.csv", "r") as f:
            reader = csv.reader(f)
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
        # IDM model parameters
        self.s0 = 2  # minimum distance between vehicles
        self.v0 = 30  # speed of vehicle in free traffic
        self.a = 0.73  # maximum acceleration
        self.b = 1.67  # comfortable deceleration
        self.T = 1.5  # safe time headway
        self.delta = 4  # acceleration exponent
        self.carlen = 5  # length of the vehicles
        self.a_max = 2  # maximum acceleration of AV
        self.d_max = 1  # maximum deceleration of AV
        self.conf_dist = 4.9  # conflict distance
        self.max_speed = 20  # maximum speed limit
        self.min_speed = 0  # minimum speed limit
        self.max_yaw_rate = math.radians(49)  # maximum yaw rate
        self.max_acceleration = 2.70  # maximum allowable acceleration
        self.max_deceleration = 7  # maximum allowable deceleration
        self.max_jerk = 9  # maximum allowable jerk
        self.prev_acceleration = 0  # previous acceleration for jerk calculation
        self.prev_orientation = 0  # previous orientation for yaw rate calculation

    def derive_planning_result(self, step_info):
        t = np.linspace(0, 0.2, 2)
        next_state = self.fsys(step_info, t)

        # Increment the trajectory index sequentially
        self.trajectory_index += 1

        # Check for end of trajectory and handle accordingly
        if self.trajectory_index >= len(self.trajectory["x_vector"]):
            self.trajectory_index = 0  # Reset to start or handle as needed
            # Implement any necessary actions when trajectory ends, e.g., slowing down
            print("End of trajectory reached. Taking necessary actions.")

        # Extract the next state values from the trajectory
        next_x = self.trajectory["x_vector"][self.trajectory_index]
        next_y = self.trajectory["y_vector"][self.trajectory_index]
        next_velocity = self.trajectory["velocity_vector"][self.trajectory_index]
        next_orientation = self.trajectory["orientation_vector"][self.trajectory_index]

        # Parse the step_info
        av_state = step_info["av_info"]
        tls_info = step_info["tls_info"]
        av_context_info = step_info["av_context_info"]
        current_x = av_state["x"]
        current_y = av_state["y"]
        current_velocity = av_state["speed_long"]
        current_orientation = av_state["orientation"]
        current_acceleration = av_state["accel_long"]

        # Compute constraints
        overall_acceleration = (next_velocity - current_velocity) / 0.1
        overall_jerk = (overall_acceleration - self.prev_acceleration) / 0.1
        desired_yaw_rate = (next_orientation - current_orientation) / 0.1

        # Ensure constraints are met
        next_velocity = self.enforce_velocity_constraints(next_velocity)
        overall_acceleration = self.enforce_acceleration_constraints(overall_acceleration)
        overall_jerk = self.enforce_jerk_constraints(overall_jerk)
        desired_yaw_rate = self.enforce_yaw_rate_constraints(desired_yaw_rate)

        # Update orientation based on adjusted yaw rate
        next_orientation = current_orientation + desired_yaw_rate * 0.1

        # Update position based on adjusted acceleration
        next_velocity = current_velocity + overall_acceleration * 0.1
        next_x = current_x + next_velocity * math.cos(next_orientation) * 0.1
        next_y = current_y + next_velocity * math.sin(next_orientation) * 0.1

        # Store the current acceleration and orientation for the next step
        self.prev_acceleration = overall_acceleration
        self.prev_orientation = next_orientation

        print("current AV position:", current_x, current_y)
        print("next AV position:", next_x, next_y)

        # Check and perform lane change if needed
        if self.decide_lane_change(av_state, av_context_info):
            print("Lane change initiated.")
            next_x, next_y, next_orientation = self.perform_lane_change(next_x, next_y, next_orientation, av_state, av_context_info)

        planning_result = {
            "timestamp": time.time(),
            "time_resolution": 0.1,
            "next_x": next_x,
            "next_y": next_y,
            "next_speed": next_velocity,
            "next_orientation": next_orientation,
        }
        return planning_result

    def enforce_velocity_constraints(self, velocity):
        if velocity > self.max_speed:
            velocity = self.max_speed
        elif velocity < self.min_speed:
            velocity = self.min_speed
        return velocity

    def enforce_acceleration_constraints(self, acceleration):
        if acceleration < -self.max_deceleration:
            acceleration = -self.max_deceleration
        elif acceleration > self.max_acceleration:
            acceleration = self.max_acceleration
        return acceleration

    def enforce_jerk_constraints(self, jerk):
        if jerk < -self.max_jerk:
            jerk = -self.max_jerk
        elif jerk > self.max_jerk:
            jerk = self.max_jerk
        return jerk

    def enforce_yaw_rate_constraints(self, yaw_rate):
        if abs(yaw_rate) > self.max_yaw_rate:
            yaw_rate = self.max_yaw_rate if yaw_rate > 0 else -self.max_yaw_rate
        return yaw_rate

    def fsys(self, step_info, t):
        # Extract AV information
        av_info = step_info['av_info']
        tls_info = step_info['tls_info']
        av_context_info = step_info['av_context_info']

        # Initialize simulation runs based on AV info
        x_follow = av_info['x']
        v_follow = av_info['speed_long']
        v_init = np.array([x_follow, x_follow + 100, v_follow])  # Assume no leading vehicle initially

        # Check if there is a leading vehicle
        for vehicle_id, info in av_context_info.items():
            if info['leading_info']['is_leading_cav']:
                x_lead = info['x']
                v_lead_init = info['speed_long']
                v_init = np.array([x_follow, x_lead, v_follow])  # Update with leading vehicle info
                break

        # Check traffic light status and adjust acceleration accordingly
        next_tls_state = tls_info['next_tls_state']
        distance_to_next_tls = tls_info['distance_to_next_tls']
        if next_tls_state in ['R', 'r', 'Y', 'y'] and distance_to_next_tls < self.conf_dist:
            acc_profile = np.full(t.shape, -self.d_max)  # Decelerate if traffic light is red or yellow and close
        else:
            acc_profile = np.full(t.shape, self.a)  # Accelerate otherwise

        # Simulate the ODE describing the IDM model for a very short time interval
        v = spi.odeint(self.f, v_init, t, args=(acc_profile,))

        return v[-1]  # Return only the last state (next immediate state)

    def f(self, v, t0, v_l_traj):
        # Differential equations for the IDM model
        idx = np.round(t0).astype('int')
        T = len(v_l_traj) - 1
        if idx > T:
            idx = T

        v_l = v_l_traj[idx]
        x_f_dot = v[2]
        x_l_dot = v_l
        v_f_dot = self.a * (1 - (v[2] / self.v0) ** self.delta - (self.s_star(v[2], (v[2] - v_l)) / (v[1] - v[0] - self.carlen)) ** 2)

        v_f_dot = self.enforce_acceleration_constraints(v_f_dot)

        return np.r_[x_f_dot, x_l_dot, v_f_dot]

    def s_star(self, v_f, v_f_del):
        return self.s0 + v_f * self.T + v_f * v_f_del / (2 * np.sqrt(self.a * self.b))

    def decide_lane_change(self, av_info, av_context_info):
        """Determine if a lane change is necessary based on traffic conditions."""
        lane_change_needed = False
        for vehicle_id, info in av_context_info.items():
            if info['leading_info']['is_leading_cav']:
                distance = info['leading_info']['distance']
                if distance is not None and distance < self.conf_dist:
                    lane_change_needed = True
                    break
        return lane_change_needed

    def perform_lane_change(self, next_x, next_y, next_orientation, av_state, av_context_info):
        """Perform a safe lane change maneuver."""
        lane_width = 3.5  # Assuming standard lane width in meters
        safety_margin = 1.5  # Safety margin in meters for lane change

        # Check if the left lane is safe to change
        left_lane_safe = self.check_adjacent_lane(av_state, av_context_info, lane_offset=lane_width)
        right_lane_safe = self.check_adjacent_lane(av_state, av_context_info, lane_offset=-lane_width)

        if left_lane_safe:
            next_y += lane_width
        elif right_lane_safe:
            next_y -= lane_width
        else:
            print("No safe lane change possible. Maintaining current lane.")
        
        return next_x, next_y, next_orientation

    def check_adjacent_lane(self, av_state, av_context_info, lane_offset):
        """Check if an adjacent lane is safe for a lane change."""
        for vehicle_id, info in av_context_info.items():
            if info['leading_info']['is_leading_cav']:
                vehicle_x = info['x']
                vehicle_y = info['y'] + lane_offset
                distance = np.sqrt((vehicle_x - av_state['x'])**2 + (vehicle_y - av_state['y'])**2)
                if distance < self.conf_dist + self.carlen:
                    return False
        return True

# Create an instance of the AV decision-making module and run it
av_decision_making_module = AVDecisionMakingModule()
av_decision_making_module.run()
