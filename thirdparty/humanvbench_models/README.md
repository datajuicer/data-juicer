# HumanVBench Models Setup

This directory manages the external models and adapters required for HumanVBench. Please follow the instructions below to ensure all components are correctly initialized and patched.

## 1. Environment Initialization
These models are managed as Git submodules. To clone the contents, run the following command from the project root:

```bash
git submodule update --init --recursive

```

## 2. Applying Custom Patches

We maintain custom modifications for these models via `.diff` files. You **must** apply these patches after initializing the submodules to ensure the pipeline functions correctly:

```bash
# Navigate to this directory
cd thirdparty/humanvbench_models

# Apply patch to YOLOv8-human
cd YOLOv8_human && git apply ../YOLOv8_human_changes.diff && cd ..

# Apply patch to Light-ASD
cd Light-ASD && git apply ../Light-ASD_changes.diff && cd ..

# Apply patch to SenseVoice
cd SenseVoice && git apply ../SenseVoice_changes.diff && cd ..

```

## 3. External Weights Download

The following weight file must be downloaded manually and placed in the specific directory:

| Model | File | Target Path | Download Source |
| --- | --- | --- | --- |
| **Light-ASD** | `sfd_face.pth` | `./Light-ASD/model/faceDetector/s3fd/` | [HuggingFace - SyncNet](https://huggingface.co/lithiumice/syncnet/tree/main) |
