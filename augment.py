import os
import time

import cv2
import mediapipe as mp
import pandas as pd

train_dir = "./dataset/train"
output_csv_file = "dataset/csv/hand_landmark_dataset.csv"
data = []
num_samples_per_gesture = 1000
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
gesture_classes = [
    d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
]
print("Detected gesture classes:", gesture_classes)

for i, gesture_label in enumerate(gesture_classes):
    print(
        f"\nReady to collect data for gesture: {gesture_label} (Class {i+1}/{len(gesture_classes)})"
    )
    input("Press 's' then ENTER to start automatic sampling...")
    count = 0
    while count < num_samples_per_gesture:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:  # noqa
            for hand_landmarks in results.multi_hand_landmarks:  # noqa
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks.append(gesture_label)
                data.append(landmarks)
                count += 1
                print(f"Saved {count} samples for {gesture_label}")
                time.sleep(0.1)
        cv2.putText(
            frame,
            f"Gesture: {gesture_label} ({count}/{num_samples_per_gesture}) - Auto Sampling",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Hand Gesture Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
num_landmarks = 21
landmark_cols = [
    f"landmark_{i}_{coord}" for i in range(num_landmarks) for coord in ["x", "y", "z"]
]
df = pd.DataFrame(data, columns=landmark_cols + ["label"])
print("\nDataFrame created:")
print(df.head())
print(df.shape)
df.to_csv(output_csv_file, index=False)
print(f"Dataset saved to {output_csv_file}")
