import cv2
import numpy as np
from screeninfo import get_monitors
import leapuvc
import mediapipe as mp
import math
import time
import struct
import socket

# Function to get center coordinates of the screen
def get_screen_center():
    monitors = get_monitors()
    if len(monitors) > 0:
        monitor = monitors[0]  # Assuming the first monitor is used
        return monitor.width // 2, monitor.height // 2
    else:
        return 0, 0

# Define a function to apply zoom
def apply_zoom(image, zoom_factor):
    if zoom_factor == 1.0:
        return image
    height, width = image.shape[:2]
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    y1 = (height - new_height) // 2
    x1 = (width - new_width) // 2
    cropped_image = image[y1:y1+new_height, x1:x1+new_width]
    resized_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def add_pose_info_text(image, x, y, z):
    text_color = (255, 255, 255)  # White color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    cv2.putText(image, f"x: {np.round(x, 2)}", (0, 50), font, font_scale, text_color, thickness)
    cv2.putText(image, f"y: {np.round(y, 2)}", (0, 100), font, font_scale, text_color, thickness)
    cv2.putText(image, f"z: {np.round(z, 2)}", (0, 150), font, font_scale, text_color, thickness)

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start the Leap Capture Thread
leap = leapuvc.leapImageThread()
leap.start()

# Define Various Camera Control Settings
cv2.namedWindow('Settings')
cv2.resizeWindow('Settings', 410, 600)
cv2.createTrackbar('Rectify', 'Settings', 0, 1, lambda a: 0)  # Applies image rectification
cv2.createTrackbar('Exposure', 'Settings', 1000, 32222, leap.setExposure)  # Sets the exposure time in microseconds
cv2.createTrackbar('LEDs', 'Settings', 1, 1, lambda a: (leap.setLeftLED(a), leap.setCenterLED(a), leap.setRightLED(a)))  # Turns on the IR LEDs
cv2.createTrackbar('Gamma', 'Settings', 1, 1, leap.setGammaEnabled)  # Applies a sqrt(x) contrast-reducing curve in 10-bit space
cv2.createTrackbar('Anlg Gain', 'Settings', 0, 63, leap.setGain)  # Amplifies the signal in analog space, 16-63
cv2.createTrackbar('Dgtl Gain', 'Settings', 0, 16, leap.setDigitalGain)  # Digitally amplifies the signal in 10-bit space
cv2.createTrackbar('HDR', 'Settings', 0, 1, leap.setHDR)  # Selectively reduces the exposure of bright areas at the cost of fixed-pattern noise
cv2.createTrackbar('Rotate', 'Settings', 0, 1, leap.set180Rotation)  # Rotates each camera image in-place 180 degrees (need to unflip when using calibrations!)
cv2.createTrackbar('Zoom', 'Settings', 2, 10, lambda x: None)  # Zoom level

# Create Leap feed window
cv2.namedWindow('Leap Feed', cv2.WINDOW_NORMAL)

# Get screen center
screen_center_x, screen_center_y = get_screen_center()

# Calculate window position
window_x = screen_center_x - 320  # Half of the window width (640 // 2)
window_y = screen_center_y - 240  # Half of the window height (480 // 2)

# Move the Leap feed window to the center
cv2.moveWindow('Leap Feed', window_x, window_y)

# Initialize calibration variables
calibrated = False
calibration_offset = np.zeros(3)

# UDP settings
udp_address = ("127.0.0.1", 4242)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Capture images until 'q' is pressed
while (not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running:
    new_frame, left_right_image = leap.read()
    if new_frame:
        # Get the current zoom level from the trackbar
        zoom_level = cv2.getTrackbarPos('Zoom', 'Settings') + 1.0
        # Apply zoom to the left camera image
        zoomed_image = apply_zoom(left_right_image[0], zoom_level)
        
        # Convert grayscale image to BGR
        zoomed_image_bgr = cv2.cvtColor(zoomed_image, cv2.COLOR_GRAY2BGR)

        # Perform face detection
        results = face_mesh.process(zoomed_image_bgr)
        
        img_h, img_w, _ = zoomed_image_bgr.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:  # Specific landmarks for pose estimation
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Check for calibration reset key
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    calibrated = True
                    calibration_offset = rot_vec.flatten()
                    print("Calibration reset")

                # Apply calibration offset if calibrated
                if calibrated:
                    rot_vec = rot_vec.flatten() - calibration_offset
                    rot_vec = rot_vec.reshape((3, 1))

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Send orientation and positional data over UDP
                x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360  # Extracted angles
                data = [0, 0, 0, y, x, z]  # Zeros for X, Y, Z positional data
                buf = struct.pack('dddddd', *data)
                udp_socket.sendto(buf, udp_address)


                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(zoomed_image_bgr, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                add_pose_info_text(zoomed_image_bgr, x, y, z)
                
                # Draw landmarks on the face
                mp_drawing.draw_landmarks(
                    image=zoomed_image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        # Display the frame with face detection and pose estimation
        cv2.imshow('Leap Feed', zoomed_image_bgr)

cv2.destroyAllWindows()
