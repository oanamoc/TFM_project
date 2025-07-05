import numpy as np
import cv2
import time
import math

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TransformStamped
import tf2_ros
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --- Camera Parameters ---
img_width = 256
img_height = 256
fov_deg = 80
fov_rad = np.deg2rad(fov_deg)
camera_height = 0.28869
camera_pitch_deg = 65
camera_pitch_rad = np.deg2rad(camera_pitch_deg)
angular_resolution = fov_rad / img_width  # Per column

# --- Robot velocity globals ---
current_linear = 0.0
current_angular = 0.0

# --- Odometry/TF globals ---
robot_x = 0.0
robot_y = 0.0
robot_yaw = 0.0
last_time = None

def cmd_vel_callback(msg):
    global current_linear, current_angular
    current_linear = msg.linear.x
    current_angular = msg.angular.z

def get_horizontal_distances(img, yellow_mask):
    horizontal_distances = np.zeros(img_width)
    for col in range(img_width):
        column_pixels = yellow_mask[:, col]
        yellow_rows = np.where(column_pixels > 0)[0]
        if len(yellow_rows) == 0:
            horizontal_distances[col] = 0.0
        else:
            closest_row = yellow_rows[-1]  # bottom-most pixel
            pixel_offset_from_center = (img_height / 2) - closest_row
            phi_rad = camera_pitch_rad + (pixel_offset_from_center * (fov_rad / img_height))
            if phi_rad <= 0 or phi_rad >= (np.pi / 2):
                horizontal_distances[col] = 0.0
            else:
                horizontal_distances[col] = camera_height * np.tan(phi_rad)
    return horizontal_distances

def main():
    global current_linear, current_angular
    global robot_x, robot_y, robot_yaw, last_time

    # ROS node setup
    rospy.init_node('camera_lidar_scan_publisher')
    pub = rospy.Publisher('/scan', LaserScan, queue_size=1)
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # CoppeliaSim API
    client = RemoteAPIClient()
    sim = client.require('sim')
    visionSensorHandle = sim.getObject('/visionSensor')
    left_motor = sim.getObject('/base_link/leftMotor')
    right_motor = sim.getObject('/base_link/rightMotor')
    sim.setStepping(True)
    sim.startSimulation()

    # LaserScan message setup
    scan_msg = LaserScan()
    scan_msg.header.frame_id = "base_link"
    scan_msg.angle_min = -fov_rad / 2
    scan_msg.angle_max = +fov_rad / 2
    scan_msg.angle_increment = fov_rad / img_width
    scan_msg.time_increment = 0
    scan_msg.range_min = 0.01
    scan_msg.range_max = 10.0

    # Robot speed constants (tune for your robot)
    K_lin = 3.0  # Linear speed scaling
    K_ang = 0.6  # Angular speed scaling (0.2 * 3.0)

    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            sim.step()
            img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle)
            img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            kernel = np.ones((3,3), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            ranges = get_horizontal_distances(img, yellow_mask)
            # Fill zeros with inf for LaserScan compatibility
            ranges[ranges == 0.0] = float('inf')
            scan_msg.header.stamp = rospy.Time.now()
            scan_msg.ranges = ranges.tolist()
            pub.publish(scan_msg)

            # --- Robot movement from /cmd_vel ---
            left_speed = current_linear * K_lin - current_angular * K_ang
            right_speed = current_linear * K_lin + current_angular * K_ang
            sim.setJointTargetVelocity(left_motor, left_speed)
            sim.setJointTargetVelocity(right_motor, right_speed)

            # --- Odometry/TF dead reckoning ---
            now = rospy.Time.now()
            if last_time is not None:
                dt = (now - last_time).to_sec()
                v = current_linear
                w = current_angular
                robot_x += v * math.cos(robot_yaw) * dt
                robot_y += v * math.sin(robot_yaw) * dt
                robot_yaw += w * dt
                print(f"robot_yaw: {robot_yaw:.3f} | current_angular: {current_angular:.3f} | dt: {dt:.3f}")
            else:
                print("First loop, initializing time.")
            last_time = now

            # Publish odom->base_link TF
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"
            t.transform.translation.x = robot_x
            t.transform.translation.y = robot_y
            t.transform.translation.z = 0.0
            q = quaternion_from_euler(0, 0, robot_yaw)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            tf_broadcaster.sendTransform(t)

            rate.sleep()
            sim.step()

    finally:
        print("Stopping simulation...")
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("Done.")

if __name__ == "__main__":
    main()
