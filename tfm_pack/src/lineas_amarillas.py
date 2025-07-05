
import time

import numpy as np
import cv2

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from opcua import Client
 
# Make sure to have the add-on "ZMQ remote API" running in
# CoppeliaSim. Do not launch simulation, but run this script


print('Program started')

client = RemoteAPIClient()
sim = client.require('sim')

# sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + '/messaging/synchronousImageTransmissionViaRemoteApi.ttt')

visionSensorHandle = sim.getObject('/visionSensor')
left_motor = sim.getObject('/base_link/leftMotor')
right_motor = sim.getObject('/base_link/rightMotor')

# Movimiento básico
forward_speed = 4.0
turn_speed = 1.5

# Run a simulation in stepping mode:
sim.setStepping(True)
sim.startSimulation()

#opcua = Client("opc.tcp://192.168.0.10:4840")  # IP y puerto de tu PLC
#opcua.connect()

#node_id = "ns=4;s=DeviceSet.PLC_PRG.fbControlRobot.CamDetected"
# Obtener nodos de CODESYS
#robo_ready = opcua.get_node("ns=4;s=DeviceSet.PLC_PRG.fbControlRobot.RoboReady")
#robo_busy = opcua.get_node("ns=4;s=DeviceSet.PLC_PRG.fbControlRobot.RoboBusy")
#cam_detected_node = opcua.get_node("ns=4;s=DeviceSet.PLC_PRG.fbControlRobot.CamDetected")

# Función para comprobar si una región está rodeada de suelo (blanco/gris claro)

try:
    while (t := sim.getSimulationTime()) < 10:
        img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        ## Define yellow range in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        height, width = img.shape[:2]
        detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            detected = True
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2

            cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
            img_width = img.shape[1]

            left_threshold = img_width * 0.4
            right_threshold = img_width * 0.6

            if area > 3000:
                # Close enough: stop
                sim.setJointTargetVelocity(left_motor, 0)
                sim.setJointTargetVelocity(right_motor, 0)
                cv2.putText(img, f"STOP", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif cx < left_threshold:
                # Turn left
                sim.setJointTargetVelocity(left_motor, 0)
                sim.setJointTargetVelocity(right_motor, turn_speed)
                cv2.putText(img, f"Izquierda", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            elif cx > right_threshold:
                # Turn right
                sim.setJointTargetVelocity(left_motor, turn_speed)
                sim.setJointTargetVelocity(right_motor, 0)
                cv2.putText(img, f"Derecha", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                # Go forward
                sim.setJointTargetVelocity(left_motor, forward_speed)
                sim.setJointTargetVelocity(right_motor, forward_speed)
                cv2.putText(img, f"Centrada", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if not detected:
            # If no red object found → stop or rotate slowly (your choice)
            sim.setJointTargetVelocity(left_motor, 0.5)
            sim.setJointTargetVelocity(right_motor, -0.5)
            cv2.putText(img, f"Buscando...", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imshow("Cámara Pioneer", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sim.step()  # ← ¡IMPORTANTE! Esto mantiene la simulación viva

finally:
    print("Cerrando...")
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print('Program ended')