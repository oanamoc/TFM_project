import numpy as np
import cv2
import time

import rospy
from sensor_msgs.msg import LaserScan

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --- Camera Parameters ---
img_width = 256
img_height = 256
fov_deg = 80
fov_rad = np.deg2rad(fov_deg)
camera_height = 0.28869
camera_pitch_deg = 25
camera_pitch_rad = np.deg2rad(camera_pitch_deg)
angular_resolution = fov_rad / img_width  # Per column (not per row!)

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
    # ROS node setup
    rospy.init_node('camera_lidar_scan_publisher')
    pub = rospy.Publisher('/scan', LaserScan, queue_size=1)

    # CoppeliaSim API
    client = RemoteAPIClient()
    sim = client.require('sim')
    visionSensorHandle = sim.getObject('/visionSensor')
    sim.setStepping(True)
    sim.startSimulation()

    # LaserScan message setup
    scan_msg = LaserScan()
    scan_msg.header.frame_id = "laser"  # Or base_link, or camera_link, as appropriate
    scan_msg.angle_min = -fov_rad/2
    scan_msg.angle_max = +fov_rad/2
    scan_msg.angle_increment = fov_rad / img_width
    scan_msg.time_increment = 0
    scan_msg.range_min = 0.01
    scan_msg.range_max = 10.0  # You can set this higher or lower

    rate = rospy.Rate(10)  # Try 10 Hz; increase if your code is fast enough

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
            # Fill zeros (invalid) with range_max for proper LaserScan (so SLAM won't think there's an object there)
            ranges[ranges == 0.0] = float('inf')
            scan_msg.header.stamp = rospy.Time.now()
            scan_msg.ranges = ranges.tolist()
            pub.publish(scan_msg)
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
