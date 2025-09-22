import cv2
import mediapipe as mp
import numpy as np
import torch

from ModelTraining import HandPoseClassifier

label_names = ["hello", "thank_you", "i_love_you", "yes", "no", "please", "albania"]
label_names = ["hello", "i_love_you", "yes", "no", "please"]

hidden_dim = 100
num_classes = len(label_names)

input_dim = 210  # df.drop(columns=["pose"]).shape[1]

classifier = HandPoseClassifier(input_dim, hidden_dim, num_classes)
classifier.model.load_state_dict(torch.load("model.pth"))
classifier.model.eval()

# This is your trained model object for inference
model = classifier.model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip for selfie view
        image = cv2.flip(image, 1)

        # Mediapipe needs RGB
        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw + Predict
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            all_features = []

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Handedness one-hot per landmark
                label = results.multi_handedness[i].classification[0].label
                if label == "Right":
                    handedness = np.tile([1, 0], (21, 1))  # shape (21, 2)
                else:  # Left
                    handedness = np.tile([0, 1], (21, 1))

                # Landmark coordinates
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                )  # shape (21, 3)

                # Combine coords + handedness → shape (21, 5)
                features = np.hstack([coords, handedness])  # (21, 5)
                all_features.append(features)

            # If only one hand detected, pad with dummy hand (21x5 zeros)
            if len(all_features) == 1:
                all_features.append(np.zeros((21, 5)))

            # Flatten all landmarks into a single row → shape (42*5=210)
            full_input = np.concatenate(all_features).flatten()

            # Convert to tensor
            input_tensor = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                pred_class = torch.argmax(outputs, dim=1).item()
                pred_label = label_names[pred_class]

            # Draw label near wrist
            h, w, _ = image.shape
            x = int(results.multi_hand_landmarks[0].landmark[0].x * w)
            y = int(results.multi_hand_landmarks[0].landmark[0].y * h)
            cv2.putText(
                image,
                pred_label,
                (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Sign Language Translator", image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
