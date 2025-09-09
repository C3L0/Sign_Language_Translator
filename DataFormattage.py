import re
import pathlib
import glob
import pandas as pd
import itertools
from typing import List

class  
LANDMARK_NAMES = [
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


def parse_hand_landmarks(txt_path: str):
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

    # if len(coords) != 21: # len(coords) % 21 == 0 # can be 21 for 1 hand or 42 for 2 hands
    #     print(f"Expected 21 landmarks, found {len(coords)}")
    # maybe return an empty df

    df = pd.DataFrame(coords, columns=["x", "y", "z"], dtype=float)
    repeated_landmarks = list(
        itertools.islice(itertools.cycle(LANDMARK_NAMES), len(df))
    )
    df.insert(0, "landmark", repeated_landmarks)
    df["handedness"] = (
        handedness  # might need to be change if  landmark >21 bc it means that it's the other hand
    )

    return df


def load_pose_from_files(file_name: List[str]):
    # ./data/{pose}/photo_{pose}_num.txt
    pose = re.findall(r"\./data/(\w+)/", file_name)

    return pose[0]


def all_metadata(pose: str):
    path = pose + "/*.txt"
    all_files = glob.glob(path, recursive=True)

    keys = [load_pose_from_files(f) for f in all_files]

    df_all = pd.concat(
        [parse_hand_landmarks(f) for f in all_files],
        keys=keys,
        names=["pose", "row"],
    ).reset_index(level=0)
    return df_all


df = all_metadata("./data/*")
print(df.head(15))
print(df.tail(15))
print(df.shape)
# def globalize_data(path: str = "/data/*"):
#     # get all pose name
#     path =
#     # collect the entire dataframes
#     # add the Y: pose in the dataframes
#     # return a complete dataset
#
# def get_dataset(path: str = "/data/*"):
#     return globalize_data(path)
