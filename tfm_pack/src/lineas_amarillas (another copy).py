import numpy as np
import cv2
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print('Starting horizontal distance estimation for yellow ground lines...')

# Camera Parameters
img_width = 256
img_height = 256
fov_deg = 80  # Square pyramid, both axes
camera_height = 0.28869  # meters
camera_pitch_deg = 65  # from vertical, pointing down-forward
camera_pitch_rad = np.deg2rad(camera_pitch_deg)

# Angular resolution per pixel
angular_resolution = np.deg2rad(fov_deg) / img_height  # radians per pixel

# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.require('sim')

visionSensorHandle = sim.getObject('/visionSensor')
sim.setStepping(True)
sim.startSimulation()

try:
    sim.step()

    # Get camera image
    img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    # Convert to HSV and detect yellow
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphology to clean mask
    kernel = np.ones((3,3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # Array to store horizontal distances (256 columns)
    horizontal_distances = np.zeros(img_width)

    # For each column, find closest yellow pixel
    for col in range(img_width):
        column_pixels = yellow_mask[:, col]
        yellow_rows = np.where(column_pixels > 0)[0]

        if len(yellow_rows) == 0:
            horizontal_distances[col] = 0.0
        else:
            closest_row = yellow_rows[-1]  # bottom-most pixel (closest)

            # Calculate vertical angle phi for this pixel
            pixel_offset_from_center = (img_height / 2) - closest_row
            phi_rad = camera_pitch_rad + (pixel_offset_from_center * angular_resolution)

            # Avoid angles that don't intersect ground realistically
            if phi_rad <= 0 or phi_rad >= (np.pi / 2):
                horizontal_distances[col] = 0.0
            else:
                # Compute horizontal distance using h * tan(phi)
                horizontal_distance = camera_height * np.tan(phi_rad)
                horizontal_distances[col] = horizontal_distance

    # Print results
    print("\nHorizontal distances (in meters):")
    print(horizontal_distances)

    # Visual check (optional)
    cv2.imshow("Original Image", img)
    cv2.imshow("Yellow Mask", yellow_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sim.step()

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Exiting safely.")

finally:
    print("Stopping simulation...")
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    print("Done.")
