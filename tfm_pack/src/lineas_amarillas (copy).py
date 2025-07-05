import numpy as np
import cv2
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print('Starting yellow object distance estimation...')

# --- CAMERA PARAMETERS FROM USER ---
resolution_y = 256  # Image height
fov_deg = 80        # Vertical field of view in degrees
fov_rad = np.deg2rad(fov_deg)
focal_length_px = resolution_y / (2 * np.tan(fov_rad / 2))  # ≈ 84.2 pixels
print(f"Focal length (calculated): {focal_length_px:.2f} pixels")

# --- USER-DEFINED REAL WORLD WIDTH OF OBJECT (IN METERS) ---
real_object_width_m = 0.05  # e.g. 5 cm wide yellow object

# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.require('sim')

visionSensorHandle = sim.getObject('/visionSensor')

# Start simulation in stepping mode (no robot motion)
sim.setStepping(True)
sim.startSimulation()

try:
    sim.step()  # One simulation step to capture current frame

    # Get RGB image
    img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    # Convert to HSV and threshold yellow
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # Detect yellow regions
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    distances = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue  # Skip small noise

        # Get bounding box to find object width in pixels
        x, y, w, h = cv2.boundingRect(cnt)
        pixel_width = w

        if pixel_width > 0:
            # Estimate distance using pinhole camera model
            z = (real_object_width_m * focal_length_px) / pixel_width
            distances.append(z)

            # Annotate image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, f"{z:.2f} m", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Print results
    print(f"\nDetected {len(distances)} yellow object(s).")
    if distances:
        print("Estimated distances (m):", ["{:.2f}".format(d) for d in distances])
        print(f"Closest: {min(distances):.2f} m | Farthest: {max(distances):.2f} m")

    # Show visual result
    cv2.imshow("Camera View", img)
    cv2.imshow("Yellow Mask", yellow_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

finally:
    print("Stopping simulation...")
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    print("Done.")
