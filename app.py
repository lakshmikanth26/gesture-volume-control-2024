import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize mediapipe hand tracking and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to calculate the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Set up the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later mirror effect
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe hands
    results = hands.process(rgb_frame)

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks for the thumb and index fingers (for volume control)
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Get the pixel positions of the tips
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

            # Calculate the distance between the thumb and index fingers
            distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)

            # Use the distance to control the volume (can adjust thresholds and scale as needed)
            if distance < 50:  # Increase volume
                pyautogui.press('volumeup')
            elif distance > 150:  # Decrease volume
                pyautogui.press('volumedown')

    # Display the frame
    cv2.imshow("Gesture Volume Control", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
