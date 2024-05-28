import cv2
import numpy as np
import mediapipe as mp
import struct
import socket

# Function to add pose information text to the image
def add_pose_info_text(image, x, y, z, pitch, yaw, roll):
    text_color = (255, 255, 255)  # White color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

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
    x = nose_tip[0] - center_x * 100
    y = nose_tip[1] - center_y * 100
    
    # Z coordinate remains the same
    z = nose_tip[2] * 1000

    return x, y, z

# UDP settings
udp_address = ("127.0.0.1", 4242)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Capture video from the default webcam
cap = cv2.VideoCapture(0)

# Create main window with dark gray background
window = np.full((350, 300, 3), 255, dtype=np.uint8)

# Add title to the window
cv2.putText(window, 'Webcam', (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(window, 'Tracker', (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 127, 0), 2, cv2.LINE_AA)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        # Convert the image color back to BGR for rendering with OpenCV.
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
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
                
                # Add pose info text to the image
                add_pose_info_text(image, x, y, z, pitch, yaw, roll)

        # Resize the image to fit the UI
        resized_image_with_overlay = cv2.resize(image, (200, 200))

        # Embed the camera feed into the window
        window[100:300, 50:250] = resized_image_with_overlay

        # Display the window
        cv2.imshow('Webcam Tracker', window)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # Press 'q' to exit
            break

cap.release()
cv2.destroyAllWindows()
