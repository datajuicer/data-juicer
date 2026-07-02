# VLA Visualization Demo

This demo provides a complete pipeline for **egocentric video hand action recognition and LeRobot dataset export**. It extracts frames from ego-view videos, estimates camera intrinsics and poses, reconstructs 3D hands, computes hand actions, and exports the results as a [LeRobot v2.0](https://github.com/huggingface/lerobot) dataset.

## Pipeline Overview

```
Video Input
  │
  ▼
VideoExtractFramesMapper          # Extract video keyframes
  │
  ▼
VideoCameraCalibrationMogeMapper  # MoGe-2 camera calibration + depth estimation
  │
  ▼
VideoHandReconstructionHaworMapper # HaWoR 3D hand reconstruction
  │
  ▼
VideoCameraPoseMegaSaMMapper      # MegaSaM camera pose estimation (⚠️ requires separate conda env)
  │
  ▼
VideoHandActionComputeMapper      # Compute 7-DoF actions + 8-dim states
  │
  ▼
VideoActionCaptioningMapper       # action instruction captioning
  │
  ▼
ExportToLeRobotMapper            # Export to LeRobot v2.0 dataset
```

## Output Format

- **Action**: 7-dim `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- **State**: 8-dim `[x, y, z, roll, pitch, yaw, pad, gripper]`
- **Gripper**: 1.0 (open) to -1.0 (closed), estimated from finger joint angles

## Prerequisites

### 1. Base Environment

Create an image based on the Dockerfile.

The `VideoCameraPoseMegaSaMMapper` operator depends on MegaSaM (based on DROID-SLAM). Its CUDA compiled components (`droid_backends`, `lietorch`, `torch-scatter`) **conflict with the main environment** and must run in a separate conda environment.

> **Note**: This environment is automatically activated at runtime via Ray's `runtime_env={"conda": "mega-sam"}` mechanism. You do not need to manually switch environments. All other operators run in the default environment.


### 2. Ray Cluster

The pipeline runs on Ray. You need to start a Ray cluster.

### 3. MANO Hand Model

Download MANO v1.2 from the [MANO website](https://mano.is.tue.mpg.de/). Update the `mano_right_path` and `mano_left_path` in the config or script to point to your `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` files.


## Running the Demo

### Option 1: Python Script (Recommended)

```bash
cd demos/ego_hand_action_annotation
python vla_pipeline.py
```

### Option 2: YAML Config

```bash
python tools/process_data.py --config demos/ego_hand_action_annotation/configs/vla_pipeline.yaml
```

## Input Data Format

Each sample is a JSON object containing a video path list:

```json
{
    "videos": ["./data/1018.mp4"],
    "text": "",
    "__dj__meta__": {}
}
```

The demo includes two sample videos: `data/1018.mp4` and `data/1034.mp4`.

## Output Structure

```
output/
├── frames/                    # Extracted video frames
├── lerobot_dataset/           # LeRobot v2.0 dataset
│   ├── data/
│   │   └── chunk-000/
│   │       ├── episode_000000.parquet
│   │       └── ...
│   ├── videos/
│   │   └── chunk-000/
│   │       ├── observation.images.main/
│   │       │   ├── episode_000000.mp4
│   │       │   └── ...
│   ├── meta/
│   │   ├── info.json
│   │   ├── episodes.jsonl
│   │   ├── stats.json
│   │   └── tasks.jsonl
│   └── modality.json
└── *.parquet                  # Ray output results
```

## Visualization Tools

Two visualization scripts are provided for inspecting processing results:

### Action Annotation Verification (vis_hand_action_demo.py)

Verify hand action annotations with hand trajectory, state, and action value overlays:

```bash
python vis_hand_action_demo.py --data_path output/xxx.parquet
```

## Troubleshooting

### MegaSaM compilation fails
Ensure the `mega-sam` conda environment has a CUDA toolkit matching your PyTorch version. Verify with `nvcc --version`.

### MANO model loading fails
Check that `mano_right_path` and `mano_left_path` point to valid files. MANO models must be downloaded separately from the official website.

### Ray GPU resource exhaustion
Multiple operators require GPU. By default each uses 0.1 GPU (10 operators can share 1 GPU). Adjust `num_gpus` or add more GPUs if needed.
