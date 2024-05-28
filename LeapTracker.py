import cv2
import numpy as np
import leapuvc
import mediapipe as mp
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

  
# Function to add pose information text to the image
def add_pose_info_text(image, x, y, z, pitch, yaw, roll):
    text_color = (255, 255, 255)  # White color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 4

    cv2.putText(image, f"X: {np.round(x, 2)}", (15, 350), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Y: {np.round(y, 2)}", (15, 400), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Z: {np.round(z, 2)}", (15, 450), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Pitch: {np.round(pitch, 2)}", (400, 350), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Yaw: {np.round(yaw, 2)}", (400, 400), font, font_scale, text_color, thickness)
    cv2.putText(image, f"Roll: {np.round(roll, 2)}", (400, 450), font, font_scale, text_color, thickness)

# Function to calculate pitch, yaw, and roll angles
def calculate_pose(landmarks):
    # Define the reference points for the face
    nose_tip = landmarks[2]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Calculate the vectors representing the orientation of the face
    v1 = np.array([left_eye[0] - right_eye[0], left_eye[1] - right_eye[1], left_eye[2] - right_eye[2]])
    v2 = np.array([nose_tip[0] - (left_eye[0] + right_eye[0]) / 2, 
                   nose_tip[1] - (left_eye[1] + right_eye[1]) / 2, 
                   nose_tip[2] - (left_eye[2] + right_eye[2]) / 2])

    # Calculate yaw, pitch, and roll angles
    yaw = -np.arctan2(v1[2], v1[0]) * 180 / np.pi
    pitch = np.arctan2(v2[1], v2[2]) * 180 / np.pi
    roll = np.arctan2(v2[0], v2[2]) * 180 / np.pi

    return pitch, yaw, roll


# Function to extract position (x, y, z) of the face
def extract_position(landmarks):
    # Extract the coordinates of the nose tip
    nose_tip = landmarks[2]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    # Calculate the center point between the eyes (x, y)
    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2
    
    # Adjust x and y based on the center point between the eyes
    x = nose_tip[0] - center_x
    y = nose_tip[1] - center_y
    
    # Z coordinate remains the same
    z = nose_tip[2]

    return x, y, z

# Retrieve calibration data using DSHOW backend
capResolution = (640, 480)
cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, capResolution[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, capResolution[1])
calibration = leapuvc.retrieveLeapCalibration(cam, capResolution)
cam.release()

# Start the Leap Capture Thread with MSMF backend
leap = leapuvc.leapImageThread(resolution=(640, 480))
leap.start()
leap.calibration = calibration  # Use the retrieved calibration data

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
zoom_level = 4

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

# UDP settings
udp_address = ("127.0.0.1", 4242)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as face_mesh:
    
    # Capture images until 'q' is pressed
    while (cv2.waitKey(1) & 0xFF != ord('q')) and leap.running:
        new_frame, left_right_image = leap.read()
        if new_frame:
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
            zoomed_image = apply_zoom(left_right_image[0], zoom_level)

            # Rectify
            maps = leap.calibration['left']["undistortMaps"]
            zoomed_image = cv2.remap(zoomed_image, maps[0], maps[1], cv2.INTER_LINEAR)
            
            # Resize the Leap image to match the area in the window
            resized_image = cv2.resize(zoomed_image, (200, 200))

            # Perform face detection
            image = cv2.cvtColor(zoomed_image, cv2.COLOR_GRAY2RGB)

            results = face_mesh.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert landmarks into list of tuples
                    landmark_coords = [(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark]

                    # Calculate pose angles
                    pitch, yaw, roll = calculate_pose(landmark_coords)

                    # Calculate position coordinates
                    x, y, z = extract_position(landmark_coords)

                    data = [x, y, z, yaw, pitch, roll]
                    buf = struct.pack('dddddd', *data)
                    udp_socket.sendto(buf, udp_address)

                    # Draw landmarks on the face
                    mp_drawing.draw_landmarks(
                        image = image,
                        landmark_list = face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
                    
                    # Add pose info text to the image
                    add_pose_info_text(image,x, y, z, pitch, yaw, roll)

            # Resize the zoomed image with Mediapipe overlay
            resized_image_with_overlay = cv2.resize(image, (200, 200))

            window[100:300, 50:250] = resized_image_with_overlay
            cv2.imshow('LeapTracker', window)
            cv2.setMouseCallback('LeapTracker', on_click)

    # Close all windows
    cv2.destroyAllWindows()
