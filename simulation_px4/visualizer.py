#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

from re import M
import numpy as np
import uuid
import math
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import TrajectorySetpoint
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField


from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from .fake_contact_detector import FakeContactDetector


def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    # msg.header.stamp = Clock().now().nanoseconds / 1000
    pose_msg.header.frame_id=frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = position[0]
    pose_msg.pose.position.y = position[1]
    pose_msg.pose.position.z = position[2]
    return pose_msg

class PX4Visualizer(Node):

    def __init__(self):
        super().__init__('px4_visualizer')

        # Box crop pointcloud
        self.x_crop = 0.44
        self.y_crop = 0.44

        # Fake obstacles
        self.ob1 = [
            np.array([0, 0.4, 2]),
            np.array([4, 0.4, 2]),
            np.array([4, 0.4, -1]), 
            np.array([0, 0.4, -1]), 
            ]
        self.ring_radius = 0.44

        self.rotation_from_world_to_drone = None
        
        self.contact_detector = FakeContactDetector(self.ob1, 0.44, self.get_logger)

        ## Configure subscritpions
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)
        self.setpoint_sub = self.create_subscription(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            self.trajectory_setpoint_callback,
            qos_profile)
        
        self.octomap_pc2_sub = self.create_subscription(
            PointCloud2, "/mid360_PointCloud2", self.octomap_pc2_callback, 10
        )

        self.vehicle_pose_pub = self.create_publisher(PoseStamped, '/px4_visualizer/vehicle_pose', 10)
        self.vehicle_vel_pub = self.create_publisher(Marker, '/px4_visualizer/vehicle_velocity', 10)
        self.vehicle_path_pub = self.create_publisher(Path, '/px4_visualizer/vehicle_path', 10)
        self.setpoint_path_pub = self.create_publisher(Path, '/px4_visualizer/setpoint_path', 10)
        self.odometry_publisher = self.create_publisher(Odometry, '/odom', 10)
        self.pc2_pub = self.create_publisher(PointCloud2, "mid360_filtered", 10)


        self.vehicle_attitude_px4 = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_path_msg = Path()
        self.setpoint_path_msg = Path()
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

    def octomap_pc2_callback(self, pointcloud: PointCloud2) -> None:

        self.octomap_occupied_pointcloud = []
        for p in point_cloud2.read_points(
            pointcloud, field_names=("x", "y", "z"), skip_nans=True
        ):
            # Get XYZ coordinates to calculate vertical angle and filter by vertical scans
            x = p[0]
            y = p[1]
            z = p[2]

            if z > 0.2 and (
                (x > self.x_crop or x < -self.x_crop)
                or (y > self.y_crop or y < -self.y_crop)
            ):
                self.octomap_occupied_pointcloud.append([x, y, z])
        
        pc2_cropped = PointCloud2()
        pc2_cropped.header = pointcloud.header
        pc2_cropped.height = 1
        pc2_cropped.width = len(self.octomap_occupied_pointcloud)
        pc2_cropped.fields.append(
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)
        )
        pc2_cropped.fields.append(
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1)
        )
        pc2_cropped.fields.append(
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        )
        pc2_cropped.is_bigendian = False
        pc2_cropped.point_step = 12  # 4 (x) + 4 (y) + 4 (z) bytes per point
        pc2_cropped.row_step = pc2_cropped.point_step * len(
            self.octomap_occupied_pointcloud
        )
        pc2_cropped.data = np.array(
            self.octomap_occupied_pointcloud, dtype=np.float32
        ).tobytes()
        self.pc2_pub.publish(pc2_cropped)
        print("Publishing pc2 cropped ...")
    
    def quaternion_multiply(self, q1, q2):
        """ Multiply two quaternions. """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]

    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation 
        self.vehicle_attitude_px4[0] = msg.q[0]
        self.vehicle_attitude_px4[1] = msg.q[1]
        self.vehicle_attitude_px4[2] = msg.q[2]
        self.vehicle_attitude_px4[3] = msg.q[3]

        # Rotation quaternion for -90 degrees around the z-axis
        rotation_quaternion_minus_90 = [np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2]
        # Rotate the original quaternion
        self.vehicle_attitude = self.quaternion_multiply(rotation_quaternion_minus_90,self.vehicle_attitude_px4)

        w, x, y, z = self.vehicle_attitude

        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(
            self.vehicle_attitude[0], 
            self.vehicle_attitude[1], 
            self.vehicle_attitude[2], 
            self.vehicle_attitude[3])
        
        # Get rotational matrix
        self.rotation_from_world_to_drone = R.from_quat([x, y, z, w]).as_matrix()
        """ self.get_logger().info(f"Rotation from quaternion: {r_from_quat.as_matrix()}")
        r_from_euler = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=True)
        self.get_logger().info(f"Rotation from euler: {r_from_euler.as_matrix()}") """
        
    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation 
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz
        self.vehicle_local_angular_velocity[0] = msg.ay
        self.vehicle_local_angular_velocity[1] = msg.ax
        self.vehicle_local_angular_velocity[2] = -msg.az

        position = np.array([
            self.vehicle_local_position[0],
            self.vehicle_local_position[1],
            self.vehicle_local_position[2],
        ])

        if position is not None and self.rotation_from_world_to_drone is not None:
            self.contact_detector.check_collision(position, self.rotation_from_world_to_drone)

    def trajectory_setpoint_callback(self, msg):
        self.setpoint_position[0] = msg.position[1]
        self.setpoint_position[1] = msg.position[0]
        self.setpoint_position[2] = -msg.position[2]

    def create_arrow_marker(self, id, tail, vector):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = 'odom'
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = 'arrow'
        msg.id = id
        msg.type = Marker.ARROW
        msg.scale.x = 0.1
        msg.scale.y = 0.2
        msg.scale.z = 0.0
        msg.color.r = 0.5
        msg.color.g = 0.5
        msg.color.b = 0.0
        msg.color.a = 1.0
        dt = 0.3
        tail_point = Point()
        tail_point.x = tail[0]
        tail_point.y = tail[1]
        tail_point.z = tail[2]
        head_point = Point()
        head_point.x = tail[0] + dt * vector[0]
        head_point.y = tail[1] + dt * vector[1]
        head_point.z = tail[2] + dt * vector[2]
        msg.points = [tail_point, head_point]
        return msg

    def create_odom_msg(self): 
        odom_msg = Odometry()

        #odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        
        # position
        odom_msg.pose.pose.position.x = self.vehicle_local_position[0]
        odom_msg.pose.pose.position.y = self.vehicle_local_position[1]
        odom_msg.pose.pose.position.z= self.vehicle_local_position[2]
        odom_msg.pose.pose.orientation.w = self.vehicle_attitude[0]
        odom_msg.pose.pose.orientation.x = self.vehicle_attitude[1]
        odom_msg.pose.pose.orientation.y = self.vehicle_attitude[2]
        odom_msg.pose.pose.orientation.z= self.vehicle_attitude[3]

        # velocity
        odom_msg.child_frame_id = 'base_link'
        odom_msg.twist.twist.linear.x = self.vehicle_local_velocity[0]
        odom_msg.twist.twist.linear.y = self.vehicle_local_velocity[1]
        odom_msg.twist.twist.linear.z = self.vehicle_local_velocity[2]
        odom_msg.twist.twist.angular.x = self.vehicle_local_angular_velocity[0]
        odom_msg.twist.twist.angular.y = self.vehicle_local_angular_velocity[1]
        odom_msg.twist.twist.angular.z = self.vehicle_local_angular_velocity[2]
        
        return odom_msg

    def create_odom_tf(self, position, attitude, parent_frame, child_frame): 
        # tf msg    
        t = TransformStamped()

        # Read message content and assign it to corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]

        t.transform.rotation.w = attitude[0]
        t.transform.rotation.x = attitude[1]
        t.transform.rotation.y = attitude[2]
        t.transform.rotation.z = attitude[3]

        return t
    
    def cmdloop_callback(self):
        vehicle_pose_msg = vector2PoseMsg('odom', self.vehicle_local_position, self.vehicle_attitude)
        self.vehicle_pose_pub.publish(vehicle_pose_msg)

        # Publish time history of the vehicle path
        self.vehicle_path_msg.header = vehicle_pose_msg.header
        self.vehicle_path_msg.poses.append(vehicle_pose_msg) 
        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        # Publish time history of the vehicle path
        setpoint_pose_msg = vector2PoseMsg('odom', self.setpoint_position, self.vehicle_attitude)
        self.setpoint_path_msg.header = setpoint_pose_msg.header
        self.setpoint_path_msg.poses.append(setpoint_pose_msg)
        self.setpoint_path_pub.publish(self.setpoint_path_msg)

        # Publish arrow markers for velocity
        velocity_msg = self.create_arrow_marker(1, self.vehicle_local_position, self.vehicle_local_velocity)
        self.vehicle_vel_pub.publish(velocity_msg)

        # Publish odometry
        odometry_msg = self.create_odom_msg()
        self.odometry_publisher.publish(odometry_msg)

        t = self.create_odom_tf(self.vehicle_local_position, self.vehicle_attitude, 'odom', 'base_link')
        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
    
    def euler_from_quaternion(self, w, x, y, z):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = math.atan2(t0, t1)
        
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
        
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = math.atan2(t3, t4)
        
            return np.rad2deg(roll_x), np.rad2deg(pitch_y), np.rad2deg(yaw_z) # in degrees


def main(args=None):
    rclpy.init(args=args)

    px4_visualizer = PX4Visualizer()

    rclpy.spin(px4_visualizer)

    px4_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
