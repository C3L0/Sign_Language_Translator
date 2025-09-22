import os
import warnings
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp

# NEED TO BE READ : https://www.geeksforgeeks.org/python/how-to-disable-python-warnings/
warnings.filterwarnings("ignore")  # Suppress all warnings


class DataPreparator:
    def __init__(self, types: List[str], path: str = "./data/"):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.path = Path(path)
        self.types = types

    def delete_typed_file_from_path(self) -> None:
        for folder in self.path.iterdir():
            if folder.is_dir():
                for type in self.types:
                    for file in folder.glob(type):
                        file.unlink()

    def create_datapath_metapath_dict(self) -> Dict[str, str]:
        data_dictionary = {}
        for folder in os.listdir(
            self.path
        ):  # once I met: path/to/the/**/file - from pathlib?
            sub_path = os.path.join(self.path, folder)
            for bmp_file in os.listdir(sub_path):
                bmp_file_path = os.path.join(sub_path, bmp_file)
                file_name_no_ext = os.path.splitext(os.path.basename(bmp_file))[0]
                txt_file_name = f"{file_name_no_ext}.txt"
                txt_file_path = os.path.join(sub_path, txt_file_name)
                data_dictionary[bmp_file_path] = txt_file_path
                try:
                    # with open(txt_file_path, "w") as f:
                    #    pass
                    f = open(txt_file_path, "w")
                    f.close()
                except FileExistsError:
                    # I think I should clean all txt from the folder in case we do the same pose but with a different picture
                    pass
                except Exception as e:
                    print(f"Error occured while creating {txt_file_name} : {e}")
        return data_dictionary

    def get_hand_landmark(self):
        data_dictionary = self.create_datapath_metapath_dict()

        with self.mp_hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
        ) as hands:
            for data_file, metadata_file in data_dictionary.items():
                if not os.path.exists(data_file):
                    print(f"File: {data_file} do not exist\n")
                    continue

                image = cv2.imread(data_file)
                image = cv2.flip(image, 1)
                if image is None:
                    print("Image couldn't be loaded")
                    continue

                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                with open(metadata_file, "w") as f:
                    f.write(f"Handedness: {results.multi_handedness}\n")
                    if not results.multi_hand_landmarks:
                        continue
                    image_height, image_width, _ = image.shape
                    # annotated_image = image.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        # f.write(f"\nHand {idx + 1}:\n")
                        f.write(f"{hand_landmarks}\n")

                        index_tip = hand_landmarks.landmark[
                            self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ]
                        tip_x = index_tip.x * image_width
                        tip_y = index_tip.y * image_height

                        f.write("Index finger tip coordinates:\n")
                        f.write(f"X: {tip_x}\n")
                        f.write(f"Y: {tip_y}\n")

        print(
            "Data prepared, metadata loaded\nExiting..."
        )  # maybe put a progression bar


preparator = DataPreparator(types=["*.txt"])
preparator.delete_typed_file_from_path()
preparator.get_hand_landmark()
