"""Upper-body keypoint subset of SAM 3D Body's MHR70 skeleton."""

# (student name, MHR70 index)
BODY = [
    ("nose", 0), ("left_eye", 1), ("right_eye", 2), ("left_ear", 3), ("right_ear", 4),
    ("neck", 69),
    ("left_shoulder", 5), ("right_shoulder", 6),
    ("left_elbow", 7), ("right_elbow", 8),
    ("left_wrist", 62), ("right_wrist", 41),
    ("left_hip", 9), ("right_hip", 10),
]

# 20 finger joints per hand (wrist is already in BODY). MHR70 21..40 right, 42..61 left.
HANDS = [(f"right_hand_{i}", 21 + i) for i in range(20)] + \
        [(f"left_hand_{i}", 42 + i) for i in range(20)]

ROOT = "neck"

BONES = [
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("neck", "nose"), ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]


# MHR skeleton joints whose global rotation the student distills (index into the 127 MHR joints).
# c_spine3 is the upper thoracic segment, where NOARK's sternum rigid body sits (trunk angle).
ROT_JOINTS = [
    ("trunk", 37),
    ("right_clavicle", 38), ("right_upper_arm", 39), ("right_forearm", 40), ("right_wrist", 42),
    ("left_clavicle", 74), ("left_upper_arm", 75), ("left_forearm", 76), ("left_wrist", 78),
]
# MHR global rotations are in the body model's y-up axes; conjugate by this to get OpenCV camera axes
MHR_TO_CAM = (1.0, -1.0, -1.0)


def joint_set(hands: bool = False):
    js = BODY + (HANDS if hands else [])
    names = [n for n, _ in js]
    mhr_idx = [i for _, i in js]
    flip = []
    for n in names:
        if n.startswith("left_"):
            m = "right_" + n[5:]
        elif n.startswith("right_"):
            m = "left_" + n[6:]
        else:
            m = n
        flip.append(names.index(m))
    bones = [(names.index(a), names.index(b)) for a, b in BONES]
    return dict(names=names, mhr_idx=mhr_idx, flip=flip, root=names.index(ROOT),
                bones=bones, num=len(names), rot_names=[n for n, _ in ROT_JOINTS],
                rot_idx=[i for _, i in ROT_JOINTS], num_rot=len(ROT_JOINTS))
