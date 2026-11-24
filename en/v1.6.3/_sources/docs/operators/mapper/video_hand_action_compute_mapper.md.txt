# video_hand_action_compute_mapper

Compute 7-DoF actions and 8-dim states from hand reconstruction
and camera pose results.

Reads hand MANO parameters (from VideoHandReconstructionHaworMapper)
and camera-to-world transforms (from VideoCameraPoseMegaSaMMapper),
then produces per-frame state [x,y,z,roll,pitch,yaw,pad,gripper]
and per-frame action [dx,dy,dz,droll,dpitch,dyaw,gripper] compatible
with LIBERO / StarVLA LeRobot format.

根据手部重建和相机位姿结果计算 7 自由度动作和 8 维状态。

读取手部 MANO 参数（来自 VideoHandReconstructionHaworMapper）和相机到世界的变换（来自 VideoCameraPoseMegaSaMMapper），然后生成与 LIBERO / StarVLA LeRobot 格式兼容的逐帧状态 [x,y,z,roll,pitch,yaw,pad,gripper] 和逐帧动作 [dx,dy,dz,droll,dpitch,dyaw,gripper]。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hand_reconstruction_field` | <class 'str'> | `'hand_reconstruction_hawor_tags'` | Meta field storing HaWoR hand reconstruction results. |
| `camera_pose_field` | <class 'str'> | `'video_camera_pose_tags'` | Meta field storing camera pose (cam_c2w) results. |
| `tag_field_name` | <class 'str'> | `'hand_action_tags'` | Output field name in Fields.meta. |
| `hand_type` | <class 'str'> | `'both'` | Which hand to compute actions for. 'right', 'left', or 'both'. Default is 'both'. |
| `args` |  | `` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_hand_action_compute_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_hand_action_compute_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)