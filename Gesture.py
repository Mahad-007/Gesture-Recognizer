
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam feed
cap = cv2.VideoCapture(0)  # 0 for default camera

# Define a function to recognize simple gestures
def detect_gesture(landmarks):
    """Detects simple gestures based on hand landmarks."""
    # Example: Detecting an open palm (distance between thumb and pinky is large)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    distance = ((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)**0.5

    if distance > 0.4:  # Arbitrary threshold for open palm
        return "Open Palm"
    else:
        return "Closed Fist or Unknown"

# Main loop
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB as MediaPipe requires
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Draw landmarks and detect gestures if hands are found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect gesture
                gesture = detect_gesture(hand_landmarks.landmark)

                # Display gesture on the frame
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the processed frame
        cv2.imshow('Gesture Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
