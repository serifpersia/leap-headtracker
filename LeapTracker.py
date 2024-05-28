import cv2
import numpy as np
import leapuvc
import mediapipe as mp
import math
import time
import struct
import socket

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
    thickness = 4
    
    cv2.putText(image, f"pitch: {np.round(x, 2)}", (15, 50), font, font_scale, text_color, thickness)
    cv2.putText(image, f"yaw: {np.round(y, 2)}", (15, 100), font, font_scale, text_color, thickness)
    cv2.putText(image, f"roll: {np.round(z, 2)}", (15, 150), font, font_scale, text_color, thickness)

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start the Leap Capture Thread
leap = leapuvc.leapImageThread()
leap.start()

# Function to check if the 'Settings' window already exists
def is_settings_window_open():
    return cv2.getWindowProperty('Settings', cv2.WND_PROP_VISIBLE) > 0

# Initialize settings
exposure_level = 1000
leds_on = 1
gamma_on = 0
analog_gain_level = 0
digital_gain_level = 0
hdr_on = 0
rotate_on = 1
zoom_level = 3


# Function to handle mouse click events
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and 100 <= x <= 200 and 350 <= y <= 400:
        if not is_settings_window_open():
            # Define Various Camera Control Settings
            cv2.namedWindow('Settings')
            cv2.resizeWindow('Settings', 410, 310)
            cv2.createTrackbar('Exposure', 'Settings', exposure_level, 32222, leap.setExposure)  # Sets the exposure time in microseconds
            cv2.createTrackbar('LEDs', 'Settings', leds_on, 1, lambda a: (leap.setLeftLED(a), leap.setCenterLED(a), leap.setRightLED(a)))  # Turns on the IR LEDs
            cv2.createTrackbar('Gamma', 'Settings', gamma_on, 1, leap.setGammaEnabled)  # Applies a sqrt(x) contrast-reducing curve in 10-bit space
            cv2.createTrackbar('Anlg Gain', 'Settings', analog_gain_level, 63, leap.setGain)  # Amplifies the signal in analog space, 16-63
            cv2.createTrackbar('Dgtl Gain', 'Settings', digital_gain_level, 16, leap.setDigitalGain)  # Digitally amplifies the signal in 10-bit space
            cv2.createTrackbar('HDR', 'Settings', hdr_on, 1, leap.setHDR)  # Selectively reduces the exposure of bright areas at the cost of fixed-pattern noise
            cv2.createTrackbar('Rotate', 'Settings', rotate_on, 1, leap.set180Rotation)  # Rotates each camera image in-place 180 degrees (need to unflip when using calibrations!)
            cv2.createTrackbar('Zoom', 'Settings', int(zoom_level - 1), 10, lambda x: None)  # Zoom level
          
# Create main window with dark gray background
window = np.full((450, 300, 3), 255, dtype=np.uint8)

# Add title and button to the window
button_color = (71, 178, 145)
cv2.putText(window, 'Leap', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(window, 'Tracker', (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (71, 178, 145), 2, cv2.LINE_AA)
cv2.rectangle(window, (100, 350), (200, 400), button_color, -1)
text, text_color = 'Settings', (255, 255, 255)
text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
text_x = int((200 + 100 - text_size[0]) / 2)
text_y = int((400 + 350 + text_size[1]) / 2)
cv2.putText(window, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

# Initialize calibration variables
calibrated = False
calibration_offset = np.zeros(3)

# UDP settings
udp_address = ("127.0.0.1", 4242)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Capture images until 'q' is pressed
while (not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running:
    newFrame, leftRightImage = leap.read()
    if newFrame:
        if is_settings_window_open():
            # Update settings values
            exposure_level = cv2.getTrackbarPos('Exposure', 'Settings')
            leds_on = cv2.getTrackbarPos('LEDs', 'Settings')
            gamma_on = cv2.getTrackbarPos('Gamma', 'Settings')
            analog_gain_level = cv2.getTrackbarPos('Anlg Gain', 'Settings')
            digital_gain_level = cv2.getTrackbarPos('Dgtl Gain', 'Settings')
            hdr_on = cv2.getTrackbarPos('HDR', 'Settings')
            rotate_on = cv2.getTrackbarPos('Rotate', 'Settings')
            zoom_level = cv2.getTrackbarPos('Zoom', 'Settings') + 1.0
            
        # Apply zoom to the left camera image
        zoomed_image = apply_zoom(leftRightImage[0], zoom_level)
        # Resize the Leap image to match the area in the window
        resized_image = cv2.resize(zoomed_image, (200, 200))
        # Convert the resized Leap image to 3-channel
        resized_image_3ch = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        # Display the resized frame in the window
        # Perform face detection

        zoomed_image_bgr = cv2.cvtColor(zoomed_image, cv2.COLOR_GRAY2BGR)
        
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
                
        # Resize the zoomed image with Mediapipe overlay
        resized_image_with_overlay = cv2.resize(zoomed_image_bgr, (200, 200))
        
        window[100:300, 50:250] = resized_image_with_overlay
        cv2.imshow('LeapTracker', window)
        cv2.setMouseCallback('LeapTracker', on_click)
        
# Close all windows
cv2.destroyAllWindows()
