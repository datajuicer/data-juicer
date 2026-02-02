import os
import ray
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.mapper.video_calibration_mapper import VideoCalibrationMapper

def main():
    # 1. Initialize Ray
    ray.init(address='auto', ignore_reinit_error=True)

    # 2. Setup paths
    data_path = "./demos/video_calibration/data/demo.jsonl"
    output_dir = "./output/video_calibration"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    # 3. Load Dataset
    ds = RayDataset(ray.data.read_json(data_path))
    print(f"Dataset loaded with {ds.count()} samples.")

    # 4. Initialize Operator
    op = VideoCalibrationMapper(
        image_size=[384, 512],
        stride=2,
        max_frames=50,
        num_gpus=0.5,
    )

    # 5. Process Dataset
    ds = ds.process([op])

    # 6. Write Results
    os.makedirs(output_dir, exist_ok=True)
    ds.data.write_json(output_dir, force_ascii=False)
    print(f"Results written to {output_dir}")

    ray.shutdown()

if __name__ == '__main__':
    main()
