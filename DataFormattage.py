import re
import pathlib
import glob
import pandas as pd
import numpy as np
import itertools
from typing import List


# class should be commented
class DataFormattage:
    def __init__(
        self, poses: List[str], path: str = "./data"
    ):  # maybe type path to Path type instead of str type
        self.path = path
        self.poses = poses

        self.encode_poses = {pose[1]: pose[0] for pose in enumerate(self.poses)}
        self.LANDMARK_NAMES = [
            "wrist",
            "thumb_cmc",
            "thumb_mcp",
            "thumb_ip",
            "thumb_tip",
            "index_finger_mcp",
            "index_finger_pip",
            "index_finger_dip",
            "index_finger_tip",
            "middle_finger_mcp",
            "middle_finger_pip",
            "middle_finger_dip",
            "middle_finger_tip",
            "ring_finger_mcp",
            "ring_finger_pip",
            "ring_finger_dip",
            "ring_finger_tip",
            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
            "pinky_tip",
        ]

    def parse_hand_landmarks(self, txt_path: str) -> pd.DataFrame:
        with open(txt_path, "r") as f:
            content = f.read()

        # Handedness
        handedness_match = re.search(r'label:\s*"(\w+)"', content)
        handedness = handedness_match.group(1) if handedness_match else None

        # Landmarks
        coords = re.findall(
            r"x:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
            r"y:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
            r"z:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            content,
            flags=re.DOTALL,
        )

        # DataFrame
        df = pd.DataFrame(coords, columns=["x", "y", "z"], dtype=float)

        if df.shape[0] == 21:
            dummy = pd.DataFrame([[-1.0, -1.0, -1.0]] * 21, columns=["x", "y", "z"])
            df = pd.concat([df, dummy], ignore_index=True)

        repeated_landmarks = list(
            itertools.islice(itertools.cycle(self.LANDMARK_NAMES), len(df))
        )
        df.insert(0, "landmark", repeated_landmarks)

        # handedness_list = [handedness] * 21 + (
        #     ["Left" if handedness == "Right" else "Right"] * 21 if len(df) == 42 else []
        # )

        # df["handedness"] = handedness_list

        one_hot_map = {"Right": [1, 0], "Left": [0, 1]}

        handedness_list = [one_hot_map[handedness]] * 21

        if len(df) == 42:
            opposite = "Left" if handedness == "Right" else "Right"
            handedness_list += [one_hot_map[opposite]] * 21

        # Assign to two new columns
        handed_array = np.array(handedness_list)
        df["handedness_right"] = handed_array[:, 0]
        df["handedness_left"] = handed_array[:, 1]

        return df

    def load_pose_from_files(self, file_name: str) -> int:
        pose = re.search(r"\./data/(\w+)/", file_name)
        if not pose:
            return -1
        pose_name = pose.group(1)

        return self.encode_poses.get(pose_name, -1)

    def all_metadata(self) -> pd.DataFrame:
        self.path += "/*/*.txt"
        all_files = glob.glob(self.path, recursive=True)

        keys = [self.load_pose_from_files(f) for f in all_files]

        df_all = pd.concat(
            [self.parse_hand_landmarks(f) for f in all_files],
            keys=keys,
            names=["pose", "row"],
        ).reset_index(level=0)

        return df_all


poses = ["hello", "thank_you", "i_love_you", "yes", "no", "please", "albania"]
formattage = DataFormattage(poses=poses)
df = formattage.all_metadata()
