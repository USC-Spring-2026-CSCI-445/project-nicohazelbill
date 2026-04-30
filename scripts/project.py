#!/usr/bin/env python3
from random import uniform
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, sin, cos, pi, inf
import math
import json
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_starter import Map, ParticleFilter, angle_to_neg_pi_to_pi  # :contentReference[oaicite:2]{index=2}
from lab10_starter import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        self.linear_pid = WaypointPID(1.0, 0.0, 0.2, 10, -0.22, 0.22)
        self.angular_pid = WaypointPID(1.0, 0.0, 0.2, 10, -2.84, 2.84)

        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0

        self.rate = rospy.Rate(10)

        # Wait until we have initial odom + scan
        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            # propagate all particles
            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        twist = Twist()
        speed = 0.15  # m/s
        twist.linear.x = speed if distance >= 0 else -speed

        duration = abs(distance) / speed if speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        twist = Twist()
        angular_speed = 0.8  # rad/s
        twist.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """Use 4 beams (front, left, back, right) to update the PF."""
        if self.laserscan is None:
            return

        range_max = getattr(self.laserscan, "range_max", 10.0)
        beam_angles_deg = [0, 90, 180, 270]

        for deg in beam_angles_deg:
            target_angle = math.radians(deg)
            z = self._get_beam_range(target_angle, window_deg=5.0)
            if z is None:
                z = range_max
            self._pf.measure(z, target_angle)
            
    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def check_valid_measurements(self, r):
        # function that does what take_measurements does to check if a measurement is valid, and if not return range_max
        range_max = getattr(self.laserscan, "range_max", 10.0)
        if np.isnan(r) or np.isinf(r):
            return range_max
        if r <= self.laserscan.range_min + 1e-3:
            return range_max
        return r
    
    def _check_pf_consistency(self):
        """Compare PF predicted ranges vs actual laser. Returns (mean_err, max_err, std_err)."""
        if self.laserscan is None:
            return float('inf'), float('inf'), float('inf')

        x_est, y_est, theta_est = self._pf.get_estimate()
        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)
        range_max = getattr(self.laserscan, "range_max", 10.0)

        beam_angles_deg = [0, 90, 180, 270]
        errors = []
        for deg in beam_angles_deg:
            target_angle = math.radians(deg)
            idx = int(round((target_angle - angle_min) / angle_increment)) % num_ranges
            idx = max(0, min(num_ranges - 1, idx))

            z_actual = ranges[idx]
            if np.isinf(z_actual) or np.isnan(z_actual) or z_actual <= self.laserscan.range_min + 1e-3:
                z_actual = range_max

            beam_world_angle = theta_est + (angle_min + idx * angle_increment)
            z_predicted = self._pf.map_.closest_distance((x_est, y_est), beam_world_angle)
            if z_predicted is None:
                z_predicted = range_max

            errors.append(abs(min(z_actual, range_max) - min(z_predicted, range_max)))

        if not errors:
            return float('inf'), float('inf'), float('inf')
        return float(np.mean(errors)), float(max(errors)), float(np.std(errors))
    
    def _get_beam_range(self, target_angle_rad, window_deg=5.0):
        angle_min = self.laserscan.angle_min
        angle_inc = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        n = len(ranges)
        range_max = getattr(self.laserscan, "range_max", 10.0)
        range_min = self.laserscan.range_min

        w = max(1, int(round(math.radians(window_deg) / angle_inc)))
        center = int(round((target_angle_rad - angle_min) / angle_inc)) % n

        valid = []
        for off in range(-w, w + 1):
            r = ranges[(center + off) % n]
            if np.isnan(r) or np.isinf(r): continue
            if r < range_min + 1e-3: continue
            if r > range_max - 1e-3: continue
            valid.append(r)

        return float(np.median(valid)) if valid else range_max

    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
          - If front is free, go forward.
          - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """
        
        ######### Your code starts here #########

        # Note: might be better to just use autonmous exploration function in the pf file
        
        rate = rospy.Rate(1.0)
        rotation_attempts = 0
        move_distance = 0.2
        close_count = 0

        # Initial 360° sweep
        rospy.loginfo("Performing initial 360° sweep for localization.")
        self.take_measurements()
        self._pf.visualize_particles()
        self._pf.visualize_estimate()
        rospy.sleep(0.2)

        for sweep_i in range(4):
            self.rotate_in_place(math.pi / 2.0)
            rospy.sleep(0.3)
            self.take_measurements()
            self._pf.visualize_particles()
            self._pf.visualize_estimate()

            x_est, y_est, theta_est = self._pf.get_estimate()
            pts = np.array([[p.x, p.y] for p in self._pf._particles])
            if pts.shape[0] > 0:
                dists = np.linalg.norm(pts - np.array([x_est, y_est]), axis=1)
                std_dev = float(np.std(dists))
                mean_err, max_err, _ = self._check_pf_consistency()
                rospy.loginfo("[Sweep %d/4] spread=%.3f  mean_err=%.3f  max_err=%.3f" %
                              (sweep_i + 1, std_dev, mean_err, max_err))

                if std_dev < 0.20 and mean_err < 0.15 and max_err < 0.30:
                    rospy.loginfo("PF converged during initial sweep!")
                    self.cmd_pub.publish(Twist())
                    return

            rospy.sleep(0.2)

        rospy.loginfo("Initial sweep complete. Beginning exploration.")

        good_steps = 0
        GOOD_STEPS_NEEDED = 2

        for step in range(max_steps):
            if rospy.is_shutdown():
                break

            if rotation_attempts > 5:
                rospy.loginfo("Too many rotations; moving forward to escape.")
                self.move_forward(move_distance)
                rotation_attempts = 0

            front_range = None
            too_close = False

            if self.laserscan is not None:
                ranges = self.laserscan.ranges
                num_ranges = len(ranges)
                angle_inc = self.laserscan.angle_increment

                front_window_deg = 25.0
                window_size = int(round(math.radians(front_window_deg) / angle_inc))
                front_indices = (
                    list(range(0, window_size + 1)) +
                    list(range(num_ranges - window_size, num_ranges))
                )

                front_sector = []
                for i in front_indices:
                    if 0 <= i < num_ranges:
                        r = ranges[i]
                        if not np.isinf(r) and r != 0.0:
                            front_sector.append(r)

                front_range = ranges[0]
                if np.isinf(front_range) or front_range == 0.0:
                    front_range = None

                if len(front_sector) > 0 and min(front_sector) < 0.07:
                    close_count += 1
                else:
                    close_count = 0

                if close_count >= 2:
                    too_close = True

            # action selection
            if too_close:
                rospy.loginfo("Too close to obstacle, backing up & rotating.")
                self.move_forward(-1.5 * move_distance)
                self.rotate_in_place(uniform(math.pi / 5, math.pi / 3))
                rotation_attempts += 1
                rate.sleep()
                continue

            # When close to converged, gentle rotation only — don't add translation noise
            if good_steps >= 1:
                rospy.loginfo("Close to convergence — gentle rotation to settle.")
                self.rotate_in_place(uniform(-math.pi / 6, math.pi / 6))
            elif front_range is None or np.isinf(front_range) or front_range > 0.7:
                self.move_forward(move_distance)
                rotation_attempts = 0
            else:
                rospy.loginfo("Obstacle ahead, rotating to find new direction.")
                self.rotate_in_place(uniform(math.pi / 4, math.pi / 2))
                rotation_attempts += 1

            self.take_measurements()
            self._pf.visualize_particles()
            self._pf.visualize_estimate()

            # convergence check (all three: spread, mean_err, max_err)
            x_est, y_est, theta_est = self._pf.get_estimate()
            pts = np.array([[p.x, p.y] for p in self._pf._particles])
            if pts.shape[0] > 0:
                dists = np.linalg.norm(pts - np.array([x_est, y_est]), axis=1)
                std_dev = float(np.std(dists))
                mean_err, max_err, _ = self._check_pf_consistency()

                if std_dev < 0.20 and mean_err < 0.15 and max_err < 0.30:
                    good_steps += 1
                    rospy.loginfo("[Step %d] spread=%.3f mean_err=%.3f max_err=%.3f  GOOD (%d/%d)" %
                                  (step, std_dev, mean_err, max_err, good_steps, GOOD_STEPS_NEEDED))
                    if good_steps >= GOOD_STEPS_NEEDED:
                        rospy.loginfo("PF converged: %d consecutive good steps." % GOOD_STEPS_NEEDED)
                        break
                else:
                    good_steps = 0
                    rospy.loginfo("[Step %d] spread=%.3f mean_err=%.3f max_err=%.3f" %
                                  (step, std_dev, mean_err, max_err))

            rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("Localization phase complete.")

        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        x_est, y_est, _ = self._pf.get_estimate()
        start = {"x": x_est, "y": y_est}
        goal = self.goal_position
 
        rospy.loginfo("Planning from (%.2f, %.2f) to (%.2f, %.2f)" %
                      (start["x"], start["y"], goal["x"], goal["y"]))
 
        # generate_plan returns (plan, graph)
        plan, graph = self._planner.generate_plan(start, goal)
        self.plan = plan
        self.current_wp_idx = 0
 
        # Visualize for RViz
        self._planner.visualize_graph(graph)
        self._planner.visualize_plan(plan)
 
        rospy.loginfo("RRT produced %d waypoints." % len(plan))

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow the RRT waypoints using PID on (distance, heading) error.
        Keep updating PF along the way.
        """
        ######### Your code starts here #########

        # if self.plan is None or len(self.plan) == 0:
        #     rospy.logwarn("No plan to follow!")
        #     return
 
        # # Log the full plan once at startup
        # rospy.logwarn("=" * 50)
        # rospy.logwarn("PLAN HAS %d WAYPOINTS:" % len(self.plan))
        # for i, wp in enumerate(self.plan):
        #     rospy.logwarn("  WP[%d]: (%.2f, %.2f)" % (i, wp["x"], wp["y"]))
 
        # rospy.logwarn("STARTING STATE:")
        # rospy.logwarn("  Odom: x=%.2f y=%.2f theta=%.2f" % (
        #     self.current_position["x"],
        #     self.current_position["y"],
        #     self.current_position["theta"]))
        # pf_x, pf_y, pf_t = self._pf.get_estimate()
        # rospy.logwarn("  PF:   x=%.2f y=%.2f theta=%.2f" % (pf_x, pf_y, pf_t))
        # rospy.logwarn("=" * 50)
 
        # rate = rospy.Rate(20)
        # ctrl_msg = Twist()
        # current_wp_idx = 0
        # iteration = 0
 
        # MIN_DT = 1e-3
        # t0 = rospy.get_time()
        # self.linear_pid.t_prev = t0 - MIN_DT
        # self.angular_pid.t_prev = t0 - MIN_DT
        # last_pid_time = t0
 
        # while not rospy.is_shutdown():
        #     if current_wp_idx >= len(self.plan):
        #         ctrl_msg.linear.x = 0.0
        #         ctrl_msg.angular.z = 0.0
        #         self.cmd_pub.publish(ctrl_msg)
        #         rospy.loginfo("Reached all waypoints!")
        #         break
 
        #     if self.current_position is None:
        #         rate.sleep()
        #         continue
 
        #     goal = self.plan[current_wp_idx]
 
        #     # use ODOM for both position and heading (like lab10 does).
        #     rx = self.current_position["x"]
        #     ry = self.current_position["y"]
        #     rtheta = self.current_position["theta"]
 
        #     dx = goal["x"] - rx
        #     dy = goal["y"] - ry
        #     distance_error = sqrt(dx * dx + dy * dy)
        #     angle_to_goal = atan2(dy, dx)
        #     angle_error = atan2(
        #         sin(angle_to_goal - rtheta),
        #         cos(angle_to_goal - rtheta),
        #     )
 
        #     if distance_error < GOAL_THRESHOLD:
        #         current_wp_idx += 1
        #         rospy.logwarn("REACHED waypoint %d/%d" %
        #                       (current_wp_idx, len(self.plan)))
        #         rate.sleep()
        #         continue
 
        #     now = rospy.get_time()
        #     if now <= last_pid_time:
        #         now = last_pid_time + MIN_DT
        #     last_pid_time = now
 
        #     lin = self.linear_pid.control(distance_error, now)
        #     ang = self.angular_pid.control(angle_error, now)
 
        #     # Print state every 10 iterations (~0.5s at 20Hz)
        #     iteration += 1
        #     if iteration % 10 == 0:
        #         rospy.logwarn(
        #             "iter=%d wp=%d robot=(%.2f,%.2f,th=%.2f) "
        #             "goal=(%.2f,%.2f) dist=%.2f ang_to_goal=%.2f "
        #             "ang_err=%.2f -> lin=%.2f ang=%.2f" % (
        #                 iteration, current_wp_idx, rx, ry, rtheta,
        #                 goal["x"], goal["y"], distance_error, angle_to_goal,
        #                 angle_error, lin, ang
        #             ))
 
        #     ctrl_msg.linear.x = lin
        #     ctrl_msg.angular.z = ang
        #     self.cmd_pub.publish(ctrl_msg)
 
        #     rate.sleep()
        #     rospy.sleep(0.1)  # small extra sleep to prevent CPU overload

        if self.plan is None or len(self.plan) == 0:
            rospy.logwarn("No plan to follow!")
            return
 
        self.current_wp_idx = 0
        last_measure_time = rospy.get_time()
        MEASURE_PERIOD = 0.5
 
        MIN_DT = 1e-3
        t0 = rospy.get_time()
        self.linear_pid.t_prev = t0 - MIN_DT
        self.angular_pid.t_prev = t0 - MIN_DT
        last_pid_time = t0
 
        # Compute a constant offset between PF-estimated heading and odom heading
        # at the start of path following. We then use odom for heading throughout
        # (odom is smooth and locally accurate) and the PF for position.
        pf_x0, pf_y0, pf_theta0 = self._pf.get_estimate()
        odom_theta0 = self.current_position["theta"]
        theta_offset = angle_to_neg_pi_to_pi(pf_theta0 - odom_theta0)
        rospy.loginfo("Heading offset (PF - odom) = %.3f rad" % theta_offset)
 
        # Skip waypoints that are already behind us (within GOAL_THRESHOLD of
        # current PF position), so we don't try to drive backwards to wp[0].
        while self.current_wp_idx < len(self.plan):
            wp = self.plan[self.current_wp_idx]
            d = sqrt((wp["x"] - pf_x0) ** 2 + (wp["y"] - pf_y0) ** 2)
            if d < GOAL_THRESHOLD * 1.5:
                self.current_wp_idx += 1
            else:
                break
        rospy.loginfo("Starting at waypoint %d/%d" %
                      (self.current_wp_idx, len(self.plan)))
 
        while not rospy.is_shutdown() and self.current_wp_idx < len(self.plan):
            wp = self.plan[self.current_wp_idx]
            wx, wy = wp["x"], wp["y"]
 
            # Position from PF, heading from odom (with offset to align frames).
            pf_x, pf_y, _ = self._pf.get_estimate()
            rtheta = angle_to_neg_pi_to_pi(
                self.current_position["theta"] + theta_offset
            )
 
            dx = wx - pf_x
            dy = wy - pf_y
            dist = sqrt(dx * dx + dy * dy)
            desired_heading = atan2(dy, dx)
            heading_error = angle_to_neg_pi_to_pi(desired_heading - rtheta)
 
            if dist < GOAL_THRESHOLD:
                self.current_wp_idx += 1
                rospy.loginfo("Reached waypoint %d/%d" %
                              (self.current_wp_idx, len(self.plan)))
                continue
 
            now = rospy.get_time()
            if now <= last_pid_time:
                now = last_pid_time + MIN_DT
            last_pid_time = now
 
            linear_vel = self.linear_pid.control(dist, now)
            angular_vel = self.angular_pid.control(heading_error, now)
 
            # Two-phase control: rotate first, then drive.
            # If heading is way off, don't move forward at all — just turn.
            if abs(heading_error) > pi / 6:   # ~30 deg
                linear_vel = 0.0
            elif abs(heading_error) > pi / 12:  # ~15 deg
                linear_vel *= 0.5
 
            twist = Twist()
            twist.linear.x = linear_vel
            twist.angular.z = angular_vel
            self.cmd_pub.publish(twist)
 
            # Throttled PF measurement updates
            if now - last_measure_time > MEASURE_PERIOD:
                self.take_measurements()
                self._pf.visualize_estimate()
                last_measure_time = now
 
            self.rate.sleep()
 
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Plan complete – reached goal!")
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.35

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
