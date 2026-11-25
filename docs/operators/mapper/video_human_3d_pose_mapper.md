# video_human_3d_pose_mapper

Human3R SMPL-X pose extraction 三维人体姿态提取

## Input and frame order 输入与帧顺序

Provide a list of videos in `videos` (or the configured `video_key`), or provide
`video_frames` containing one ordered list of image paths or a list of frame lists,
one per video. Existing frame paths take precedence over video extraction.
Paths are processed in the supplied order. Directory contents are not used to
expand or reorder a frame list. An unreadable frame raises an error with its path.
Empty input returns empty result lists without loading a model.

输入可为视频路径列表，或 `video_frames` 中已按顺序排列的帧路径。
多个视频使用嵌套帧列表；不同视频分别处理，不将目录中的其他图片混入输入。

`frame_num` defaults to 3. With `duration: 0`, frames are sampled from the whole
video; a positive duration samples each segment separately. `frame_dir` is the
extraction directory. A null directory uses the Data-Juicer assets cache.

## Human3R setup 模型准备

Prepare a [Human3R](https://github.com/fanegg/Human3R) checkout and install its
upstream dependencies. Obtain the SMPL/SMPL-X assets under their respective
licenses following the upstream instructions, including `scripts/fetch_smplx.sh`.
Set `human3r_repo_path` to the checkout. If omitted, the loader uses
`Human3R` under the Data-Juicer assets cache and clones it when absent.

The checkout must contain these files under `src/models/`:

- `smplx/SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`, and `smplx2smpl.pkl`.
- `smpl/SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl`, and `J_regressor_h36m.npy`.

Missing assets produce an error listing the missing files and setup command.
`model_path` accepts a local checkpoint; otherwise the loader downloads
`human3r_896L.pth` from `faneggg/human3r`. Checkout and checkpoint preparation are
protected by a file lock. The model is loaded onto the worker's selected device.

```yaml
- video_human_3d_pose_mapper:
    human3r_repo_path: /path/to/Human3R
    model_path: /path/to/human3r_896L.pth
    frame_num: 3
    frame_dir: /path/to/frames
```

## Output 输出

Results are stored in `__dj__meta__.video_human_3d_pose_tags`. The lists
`camera_pose`, `conf`, `smpl_shape`, `smpl_transl`, `smpl_rotmat`,
`smpl_expression`, `smpl_id`, and `smpl_loc` contain the model predictions
for frames with an SMPL prediction, with the batch dimension removed.
All tensors are converted to JSON-serializable lists.

- `valid_frame_list`: zero-based indices in the flattened input frame lists,
  including positions of frames without an SMPL prediction when counting offsets.
- `frame_paths`: paths of the frames that have exported predictions.
- `video_indices`: corresponding source video indices.

Inference state is reset for each video. The extra overlap views used internally
at 100-frame reset boundaries do not create duplicate exported frames.

## Related links 相关链接

- [Source](../../../data_juicer/ops/mapper/video_human_3d_pose_mapper.py)
- [Regression tests](../../../tests/ops/mapper/test_human3r_regression.py)
- [Operators](../../Operators.md)
