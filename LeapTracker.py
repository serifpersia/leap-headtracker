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
            cv2.namedWindow('Settings')
            cv2.resizeWindow('Settings', 400, 385)
            cv2.moveWindow('Settings', 0, 0)
            cv2.createTrackbar('Camera', 'Settings', cam_id, 1, lambda x: None)
            cv2.createTrackbar('Rectify', 'Settings', rectify_on, 1, lambda x: None)
            cv2.createTrackbar('Exposure', 'Settings', exposure_level, 32222, leap.setExposure)
            cv2.createTrackbar('LEDs', 'Settings', leds_on, 1, lambda a: (leap.setLeftLED(a), leap.setCenterLED(a), leap.setRightLED(a)))
            cv2.createTrackbar('Gamma', 'Settings', gamma_on, 1, leap.setGammaEnabled)
            cv2.createTrackbar('Anlg Gain', 'Settings', analog_gain_level, 63, leap.setGain)
            cv2.createTrackbar('Dgtl Gain', 'Settings', digital_gain_level, 16, leap.setDigitalGain)
            cv2.createTrackbar('HDR', 'Settings', hdr_on, 1, leap.setHDR)
            cv2.createTrackbar('Rotate', 'Settings', rotate_on, 1, leap.set180Rotation)
            cv2.createTrackbar('Zoom', 'Settings', int(zoom_level - 1), 10, lambda x: None)

# Function to add pose information text to the image
def add_pose_info_text(image, x, y, z, pitch, yaw, roll):
    text_color = (255, 255, 255)
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
    nose_tip = landmarks[2]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    v1 = np.array([left_eye[0] - right_eye[0], left_eye[1] - right_eye[1], left_eye[2] - right_eye[2]])
    v2 = np.array([nose_tip[0] - (left_eye[0] + right_eye[0]) / 2, 
                   nose_tip[1] - (left_eye[1] + right_eye[1]) / 2, 
                   nose_tip[2] - (left_eye[2] + right_eye[2]) / 2])
    yaw = -np.arctan2(v1[2], v1[0]) * 180 / np.pi
    pitch = np.arctan2(v2[1], v2[2]) * 180 / np.pi
    roll = np.arctan2(v2[0], v2[2]) * 180 / np.pi
    return pitch, yaw, roll

# Function to extract position (x, y, z) of the face
def extract_position(landmarks):
    nose_tip = landmarks[2]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2
    x = nose_tip[0] - center_x
    y = nose_tip[1] - center_y
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
leap.calibration = calibration

# Function to check if the 'Settings' window already exists
def is_settings_window_open():
    return cv2.getWindowProperty('Settings', cv2.WND_PROP_VISIBLE) > 0

# Initialize settings
cam_id = 0
rectify_on = 0
exposure_level = 1000
leds_on = 1
gamma_on = 1
analog_gain_level = 0
digital_gain_level = 0
hdr_on = 0
rotate_on = 1
zoom_level = 4

# Create main window with dark gray background
window = np.full((450, 300, 3), 255, dtype=np.uint8)
button_color = (71, 178, 145)
cv2.putText(window, 'Leap', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(window, 'Tracker', (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, button_color, 2, cv2.LINE_AA)
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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as face_mesh:
    
    while (cv2.waitKey(1) & 0xFF != ord('q')) and leap.running:
        new_frame, left_right_image = leap.read()
        if new_frame:
            if is_settings_window_open():
                cam_id = cv2.getTrackbarPos('Camera', 'Settings')
                rectify_on = cv2.getTrackbarPos('Rectify', 'Settings')
                exposure_level = cv2.getTrackbarPos('Exposure', 'Settings')
                leds_on = cv2.getTrackbarPos('LEDs', 'Settings')
                gamma_on = cv2.getTrackbarPos('Gamma', 'Settings')
                analog_gain_level = cv2.getTrackbarPos('Anlg Gain', 'Settings')
                digital_gain_level = cv2.getTrackbarPos('Dgtl Gain', 'Settings')
                hdr_on = cv2.getTrackbarPos('HDR', 'Settings')
                rotate_on = cv2.getTrackbarPos('Rotate', 'Settings')
                zoom_level = cv2.getTrackbarPos('Zoom', 'Settings') + 1.0
                
            CAM_INDEX = 0
            
            if cam_id:
                CAM_INDEX = 0
            else:
                CAM_INDEX = 1
                
            if rectify_on:
                if CAM_INDEX == 1:
                    calibration = 'right'
                else:
                    calibration = 'left'
                    
                maps = leap.calibration[calibration]["undistortMaps"]
                rectified_image = cv2.remap(left_right_image[CAM_INDEX], maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
                processed_image = rectified_image
            else:
                processed_image = left_right_image[CAM_INDEX]

            zoomed_image = apply_zoom(processed_image, zoom_level)
            resized_image = cv2.resize(zoomed_image, (200, 200))
            image = cv2.cvtColor(zoomed_image, cv2.COLOR_GRAY2RGB)
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmark_coords = [(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark]
                    pitch, yaw, roll = calculate_pose(landmark_coords)
                    x, y, z = extract_position(landmark_coords)
                    data = [x, y, z, yaw, pitch, roll]
                    buf = struct.pack('dddddd', *data)
                    udp_socket.sendto(buf, udp_address)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
                    add_pose_info_text(image, x, y, z, pitch, yaw, roll)

            resized_image_with_overlay = cv2.resize(image, (200, 200))
            window[100:300, 50:250] = resized_image_with_overlay
            cv2.imshow('LeapTracker', window)
            cv2.setMouseCallback('LeapTracker', on_click)
            
    cv2.destroyAllWindows()
