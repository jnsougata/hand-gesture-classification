import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from training import GestureClassifier

if __name__ == "__main__":

    csv_path = "./dataset/csv/hand_landmark_dataset.csv"
    model_path = "./models/hand_gesture_model.pth"

    label_encoder_live = LabelEncoder()
    df_live = pd.read_csv(csv_path)
    label_encoder_live.fit(df_live["label"])
    gesture_classes_live = label_encoder_live.classes_
    mp_hands_live = mp.solutions.hands
    hands_live = mp_hands_live.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing_live = mp.solutions.drawing_utils

    input_dim_live = 21 * 3
    num_classes_live = len(gesture_classes_live)
    model_live = GestureClassifier(input_dim_live, num_classes_live)
    model_live.load_state_dict(torch.load(model_path))
    model_live.eval()
    cap_live = cv2.VideoCapture(0)

    while True:
        ret_live, frame_live = cap_live.read()
        if not ret_live:
            break
        frame_live = cv2.flip(frame_live, 1)
        rgb_frame_live = cv2.cvtColor(frame_live, cv2.COLOR_BGR2RGB)
        results_live = hands_live.process(rgb_frame_live)
        if results_live.multi_hand_landmarks:  # noqa
            for hand_landmarks_live in results_live.multi_hand_landmarks:  # noqa
                landmarks_live = []
                for landmark in hand_landmarks_live.landmark:
                    landmarks_live.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing_live.draw_landmarks(
                    frame_live, hand_landmarks_live, mp_hands_live.HAND_CONNECTIONS
                )
                if len(landmarks_live) == input_dim_live:
                    landmarks_np_live = (
                        np.array(landmarks_live).reshape(1, -1).astype(np.float32)
                    )
                    scaler_live = StandardScaler()
                    df_landmarks_live = df_live.drop("label", axis=1)
                    scaler_live.fit(df_landmarks_live)
                    landmarks_scaled_live = scaler_live.transform(landmarks_np_live)
                    landmarks_tensor_live = torch.tensor(
                        landmarks_scaled_live, dtype=torch.float32
                    )
                    with torch.no_grad():
                        output_live = model_live(landmarks_tensor_live)
                        _, predicted_index_live = torch.max(output_live.data, 1)
                        predicted_gesture_live = gesture_classes_live[
                            predicted_index_live.item()
                        ]
                    cv2.putText(
                        frame_live,
                        f"Gesture: {predicted_gesture_live}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame_live,
                        "Hand detected but not enough landmarks",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
        else:
            cv2.putText(
                frame_live,
                "No hand detected",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("Live Hand Gesture Detection", frame_live)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap_live.release()
    cv2.destroyAllWindows()
