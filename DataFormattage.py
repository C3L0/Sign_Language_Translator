import re

# The ordered names of Mediapipe landmarks
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

# def parse_hand_landmarks_simple(txt_path: str):
#     with open(txt_path, "r") as f:
#         lines = f.readlines()
#
#     coords = []
#     x = y = z = None
#
#     for line in lines:
#         line = line.strip()
#         if line.startswith("x:"):
#             x = float(line.split(":")[1].strip())
#         elif line.startswith("y:"):
#             y = float(line.split(":")[1].strip())
#         elif line.startswith("z:"):
#             z = float(line.split(":")[1].strip())
#             # we complete one landmark here
#             coords.append((x, y, z))
#
#     if len(coords) != 21:
#         raise ValueError(f"Expected 21 landmarks, got {len(coords)}")
#
#     # Map to names
#     hand_positions = {}
#     for name, (x, y, z) in zip(LANDMARK_NAMES, coords):
#         hand_positions[name] = {"x": x, "y": y, "z": z}
#
#     return hand_positions


def parse_hand_landmarks(txt_path: str):
    with open(txt_path, "r") as f:
        content = f.read()

    # Find all x/y/z floats in the file
    coords = re.findall(
        r"""x:\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)\s*
            y:\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)\s*
            z:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)""",  # Imight need more research
        content,
    )

    if len(coords) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(coords)}")

    # Build dictionary
    hand_positions = {}
    for name, (x, y, z) in zip(LANDMARK_NAMES, coords):
        hand_positions[name] = {"x": float(x), "y": float(y), "z": float(z)}

    return hand_positions


hand_positions = parse_hand_landmarks("data/hello/photo_hello_5.txt")

print(hand_positions["index_finger_tip"])
