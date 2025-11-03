#!/usr/bin/env python3
# Feng Xu 

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, TransformStamped, Quaternion
from tf2_ros import Buffer, TransformListener
import argparse
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, RobotState, Constraints
from moveit_msgs.msg import PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
import math
import numpy as np
from transforms3d.euler import euler2quat, quat2euler


class RelativeEEFMover(Node):
    def __init__(self, dx, dy, dz, droll, dpitch, dyaw, simulation_mode=False, planning_time=5.0, velocity_scaling=0.1):
        super().__init__('relative_eef_mover')
        
        # Store movement parameters
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.droll = droll
        self.dpitch = dpitch
        self.dyaw = dyaw
        
        # Store planning parameters
        self.simulation_mode = simulation_mode
        self.planning_time = planning_time
        self.velocity_scaling = velocity_scaling
        
        # Create TF buffer and listener to get current end effector position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create the MoveGroup action client
        self.action_client = ActionClient(self, MoveGroup, '/move_action')
        
        # Wait for the action server
        self.get_logger().info('Waiting for MoveGroup action server...')
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Timed out waiting for action server')
            return
        
        self.get_logger().info('Connected to MoveGroup action server')
        
        # Flag to track if shutdown has been called
        self.shutdown_called = False
        
        # Create a timer to get the current position and then plan movement
        self.create_timer(1.0, self.get_current_pose_and_plan)
    
    def get_current_pose_and_plan(self):
        # Cancel the timer so this only runs once
        for timer in self.timers:
            timer.cancel()
        
        try:
            # Try to get the current transform of the end effector
            trans = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            
            # Get current position
            current_x = trans.transform.translation.x
            current_y = trans.transform.translation.y
            current_z = trans.transform.translation.z
            
            # Calculate target position (current + relative)
            target_x = current_x + self.dx
            target_y = current_y + self.dy
            target_z = current_z + self.dz
            
            # Get current rotation as quaternion
            current_quat = trans.transform.rotation
            current_quat_array = [current_quat.x, current_quat.y, current_quat.z, current_quat.w]
            
            # Convert to Euler angles (roll, pitch, yaw)
            current_euler = quat2euler(current_quat_array, 'sxyz')
            
            # Add the relative rotations
            target_roll = current_euler[0] + self.droll
            target_pitch = current_euler[1] + self.dpitch
            target_yaw = current_euler[2] + self.dyaw
            
            # Convert back to quaternion
            target_quat_array = euler2quat(target_roll, target_pitch, target_yaw, 'sxyz')
            target_quat = Quaternion()
            target_quat.x = target_quat_array[0]
            target_quat.y = target_quat_array[1]
            target_quat.z = target_quat_array[2]
            target_quat.w = target_quat_array[3]
            
            self.get_logger().info(f'Current position: x={current_x:.3f}, y={current_y:.3f}, z={current_z:.3f}')
            self.get_logger().info(f'Target position: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}')
            
            self.get_logger().info(f'Current orientation (roll, pitch, yaw): ({math.degrees(current_euler[0]):.2f}°, {math.degrees(current_euler[1]):.2f}°, {math.degrees(current_euler[2]):.2f}°)')
            self.get_logger().info(f'Target orientation (roll, pitch, yaw): ({math.degrees(target_roll):.2f}°, {math.degrees(target_pitch):.2f}°, {math.degrees(target_yaw):.2f}°)')
            
            # Send the goal with the calculated target position and orientation
            self.send_goal(target_x, target_y, target_z, target_quat)
            
        except Exception as e:
            self.get_logger().error(f'Error getting current pose: {e}')
            self.shutdown_safely()
    
    def send_goal(self, x, y, z, target_orientation):
        goal_msg = MoveGroup.Goal()
        
        # Fill in the request
        req = MotionPlanRequest()
        req.group_name = "ur_manipulator"
        req.planner_id = "RRTConnect"  # Using RRTConnect planner for more direct paths
        
        # Set workspace boundaries
        req.workspace_parameters.header.frame_id = "base_link"
        req.workspace_parameters.min_corner.x = -2.0
        req.workspace_parameters.min_corner.y = -2.0
        req.workspace_parameters.min_corner.z = -2.0
        req.workspace_parameters.max_corner.x = 2.0
        req.workspace_parameters.max_corner.y = 2.0
        req.workspace_parameters.max_corner.z = 2.0
        
        # Set current state as start state
        req.start_state.is_diff = True
        
        # Create goal constraints
        goal_constraints = Constraints()
        
        # Position constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "base_link"
        position_constraint.link_name = "tool0"
        
        # Bounding volume
        bounding_volume = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01]  # Small 1cm cube for more precise positioning
        bounding_volume.primitives.append(primitive)
        
        # Position for the bounding volume
        box_pose = Pose()
        box_pose.position.x = x
        box_pose.position.y = y
        box_pose.position.z = z
        box_pose.orientation = target_orientation  # Use the target orientation
        bounding_volume.primitive_poses.append(box_pose)
        
        position_constraint.constraint_region = bounding_volume
        goal_constraints.position_constraints.append(position_constraint)
        
        # Orientation constraint - use target orientation
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.link_name = "tool0"
        orientation_constraint.orientation = target_orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.01  # Very strict orientation constraint
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        orientation_constraint.weight = 1.0
        
        goal_constraints.orientation_constraints.append(orientation_constraint)
        
        # Add the constraints to the request
        req.goal_constraints.append(goal_constraints)
        
        # Planning parameters
        req.num_planning_attempts = 10
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.velocity_scaling
        
        # Set the request
        goal_msg.request = req
        
        # Planning options - Set plan_only to True in simulation mode
        goal_msg.planning_options.plan_only = self.simulation_mode
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 10
        goal_msg.planning_options.replan_delay = 0.1
        
        # Log execution mode
        mode_msg = "SIMULATION MODE (plan only)" if self.simulation_mode else "EXECUTION MODE (will move robot)"
        self.get_logger().warn(f"RUNNING IN {mode_msg}")
        
        # Send the goal
        self.get_logger().info(f'Sending goal to move EEF by: dx={self.dx}, dy={self.dy}, dz={self.dz}, droll={math.degrees(self.droll):.1f}°, dpitch={math.degrees(self.dpitch):.1f}°, dyaw={math.degrees(self.dyaw):.1f}°')
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            self.shutdown_safely()
            return
        
        self.get_logger().info('Goal accepted')
        
        # Get the result
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        result = future.result().result
        error_code = result.error_code.val
        
        # Common error codes
        error_map = {
            1: "SUCCESS",
            -1: "PLANNING_FAILED",
            -2: "INVALID_MOTION_PLAN",
            -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            -4: "CONTROL_FAILED",
            -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            -6: "TIMED_OUT",
            -7: "PREEMPTED",
            99999: "UNKNOWN_ERROR" 
        }
        
        if error_code == 1:  # SUCCESS
            if self.simulation_mode:
                self.get_logger().info('Motion plan succeeded in simulation! The path is displayed in RViz.')
                self.get_logger().info('To execute this movement on the real robot, run without --sim flag')
            else:
                self.get_logger().info('Motion plan succeeded and executed on the robot!')
        else:
            error_name = error_map.get(error_code, f"UNKNOWN CODE: {error_code}")
            self.get_logger().error(f'Motion plan failed with error: {error_name}')
        
        # Signal to main that we're done
        self.shutdown_safely()
    
    def shutdown_safely(self):
        # Only call shutdown if it hasn't been called already
        if not self.shutdown_called:
            self.shutdown_called = True
            # We don't call rclpy.shutdown() directly here
            # Just signal to the main thread that we're done
            # by setting a flag that the main spin loop can check
            self.done = True


def main():
    parser = argparse.ArgumentParser(description='Move UR5e end effector by a relative distance and rotation')
    parser.add_argument('--dx', type=float, default=0.0, help='Relative X movement (m)')
    parser.add_argument('--dy', type=float, default=0.0, help='Relative Y movement (m)')
    parser.add_argument('--dz', type=float, default=0.0, help='Relative Z movement (m)')
    parser.add_argument('--droll', type=float, default=0.0, help='Relative roll rotation around X axis (degrees)')
    parser.add_argument('--dpitch', type=float, default=0.0, help='Relative pitch rotation around Y axis (degrees)')
    parser.add_argument('--dyaw', type=float, default=0.0, help='Relative yaw rotation around Z axis (degrees)')
    parser.add_argument('--sim', action='store_true', help='Simulation mode - plan only, do not execute on robot')
    parser.add_argument('--planning-time', type=float, default=5.0, help='Planning time allowed (seconds)')
    parser.add_argument('--velocity', type=float, default=0.1, help='Velocity scaling factor (0.0-1.0)')
    args = parser.parse_args()
    
    # Convert rotation inputs from degrees to radians
    droll_rad = math.radians(args.droll)
    dpitch_rad = math.radians(args.dpitch)
    dyaw_rad = math.radians(args.dyaw)
    
    rclpy.init()
    node = RelativeEEFMover(
        args.dx, args.dy, args.dz, 
        droll_rad, dpitch_rad, dyaw_rad,
        simulation_mode=args.sim,
        planning_time=args.planning_time,
        velocity_scaling=args.velocity
    )
    
    # Set initial done state
    node.done = False
    
    # Spin until the action is complete or timeout
    try:
        timeout = 30.0  # 30 seconds timeout
        start_time = node.get_clock().now()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            # Check for timeout
            elapsed = (node.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout:
                node.get_logger().warn(f"Operation timed out after {timeout} seconds")
                break
    except KeyboardInterrupt:
        node.get_logger().info("Operation interrupted by user")
    finally:
        # Only call shutdown once at the end
        rclpy.shutdown()
    
    print("Script completed successfully")


if __name__ == '__main__':
    main()