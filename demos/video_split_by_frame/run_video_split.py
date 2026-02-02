import os
import sys

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_split_by_frame_mapper import VideoSplitByFrameMapper

def main():
    # 1. Setup paths
    data_path = 'demos/video_split_by_frame/data/demo.jsonl'
    output_dir = 'demos/video_split_by_frame/output'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")
    
    # 2. Load Dataset
    ds = Dataset.from_json(data_path)
    print(f"Dataset loaded with {len(ds)} samples.")

    # 3. Initialize Operator
    op = VideoSplitByFrameMapper(
        split_len=50,
        overlap_len=10,
        unit='frame',
        keep_original_sample=False,
        save_dir=output_dir
    )

    # 4. Process Dataset
    # Since VideoSplitByFrameMapper is a batched operator, we must set batched=True
    ds = ds.map(op.process, batched=True)

    # 5. Write Results
    os.makedirs(output_dir, exist_ok=True)
    ds.to_json(os.path.join(output_dir, 'result.jsonl'), force_ascii=False)
    print(f"Results written to {output_dir}")

if __name__ == '__main__':
    main()
