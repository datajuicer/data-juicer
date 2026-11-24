FOCAL_LENGTH = 5000.0

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
    "OP Nose",
    "OP Neck",
    "OP RShoulder",  # 0,1,2
    "OP RElbow",
    "OP RWrist",
    "OP LShoulder",  # 3,4,5
    "OP LElbow",
    "OP LWrist",
    "OP MidHip",  # 6, 7,8
    "OP RHip",
    "OP RKnee",
    "OP RAnkle",  # 9,10,11
    "OP LHip",
    "OP LKnee",
    "OP LAnkle",  # 12,13,14
    "OP REye",
    "OP LEye",
    "OP REar",  # 15,16,17
    "OP LEar",
    "OP LBigToe",
    "OP LSmallToe",  # 18,19,20
    "OP LHeel",
    "OP RBigToe",
    "OP RSmallToe",
    "OP RHeel",  # 21, 22, 23, 24  ##Total 25 joints  for openpose
    "Right Ankle",
    "Right Knee",
    "Right Hip",  # 0,1,2
    "Left Hip",
    "Left Knee",
    "Left Ankle",  # 3, 4, 5
    "Right Wrist",
    "Right Elbow",
    "Right Shoulder",  # 6
    "Left Shoulder",
    "Left Elbow",
    "Left Wrist",  # 9
    "Neck (LSP)",
    "Top of Head (LSP)",  # 12, 13
    "Pelvis (MPII)",
    "Thorax (MPII)",  # 14, 15
    "Spine (H36M)",
    "Jaw (H36M)",  # 16, 17
    "Head (H36M)",
    "Nose",
    "Left Eye",  # 18, 19, 20
    "Right Eye",
    "Left Ear",
    "Right Ear",  # 21,22,23 (Total 24 joints)
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    "OP Nose": 24,
    "OP Neck": 12,
    "OP RShoulder": 17,
    "OP RElbow": 19,
    "OP RWrist": 21,
    "OP LShoulder": 16,
    "OP LElbow": 18,
    "OP LWrist": 20,
    "OP MidHip": 0,
    "OP RHip": 2,
    "OP RKnee": 5,
    "OP RAnkle": 8,
    "OP LHip": 1,
    "OP LKnee": 4,
    "OP LAnkle": 7,
    "OP REye": 25,
    "OP LEye": 26,
    "OP REar": 27,
    "OP LEar": 28,
    "OP LBigToe": 29,
    "OP LSmallToe": 30,
    "OP LHeel": 31,
    "OP RBigToe": 32,
    "OP RSmallToe": 33,
    "OP RHeel": 34,
    "Right Ankle": 8,
    "Right Knee": 5,
    "Right Hip": 45,
    "Left Hip": 46,
    "Left Knee": 4,
    "Left Ankle": 7,
    "Right Wrist": 21,
    "Right Elbow": 19,
    "Right Shoulder": 17,
    "Left Shoulder": 16,
    "Left Elbow": 18,
    "Left Wrist": 20,
    "Neck (LSP)": 47,
    "Top of Head (LSP)": 48,
    "Pelvis (MPII)": 49,
    "Thorax (MPII)": 50,
    "Spine (H36M)": 51,
    "Jaw (H36M)": 52,
    "Head (H36M)": 53,
    "Nose": 24,
    "Left Eye": 26,
    "Right Eye": 25,
    "Left Ear": 28,
    "Right Ear": 27,
}

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21] + [
    25 + i for i in J24_FLIP_PERM
]
