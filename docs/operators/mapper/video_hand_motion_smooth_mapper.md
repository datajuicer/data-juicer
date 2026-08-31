# video_hand_motion_smooth_mapper

Apply smoothing to world-space hand motions and remove outliers.

Reads hand action results (states, actions, joints_world) produced by
``VideoHandActionComputeMapper`` and applies:

1. **Extreme outlier replacement** — frames whose instantaneous wrist
   speed exceeds ``median + outlier_velocity_threshold * MAD`` are
   replaced by linear interpolation from neighbors (not deleted).
2. **Savitzky-Golay smoothing** — positions are smoothed with a
   Savitzky-Golay filter that preserves motion peaks while removing
   high-frequency jitter.
3. **Quaternion smoothing** — orientations are smoothed in quaternion
   space to avoid gimbal lock and discontinuities.
4. **Action recomputation** — 7-DoF actions are re-derived from the
   smoothed states so they stay consistent.

Reference (paper §3.1):
    "we apply spline smoothing to the world-space hand motions and remove
    outliers"

对世界坐标系下的手部运动应用平滑处理并剔除异常值。

读取由 ``VideoHandActionComputeMapper`` 生成的手部动作结果（states、actions、joints_world）并应用以下处理：

1. **极端异常值替换** — 瞬时手腕速度超过 ``median + outlier_velocity_threshold * MAD`` 的帧，将通过相邻帧的线性插值进行替换（而非删除）。
2. **Savitzky-Golay 平滑** — 使用 Savitzky-Golay 滤波器对位置进行平滑处理，在保留运动峰值的同时消除高频抖动。
3. **四元数平滑** — 在四元数空间中对方向进行平滑处理，以避免万向节死锁和不连续性。
4. **动作重新计算** — 从平滑后的状态重新推导 7 自由度动作，以保持一致性。

参考文献（论文 §3.1）：
    “我们对世界坐标系下的手部运动应用样条平滑并剔除异常值”

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hand_action_field` | <class 'str'> | `'hand_action_tags'` | Meta field storing hand action results (output of VideoHandActionComputeMapper). |
| `savgol_window` | <class 'int'> | `11` | Window length for Savitzky-Golay filter. Must be odd.  Larger = smoother but may lose fast motions. |
| `savgol_polyorder` | <class 'int'> | `3` | Polynomial order for Savitzky-Golay filter. Must be less than savgol_window. |
| `outlier_velocity_threshold` | <class 'float'> | `5.0` | Frames whose wrist speed exceeds ``median + threshold * MAD`` are replaced by interpolation. Higher = more conservative (fewer replacements). |
| `min_frames_for_smoothing` | <class 'int'> | `5` | Minimum number of valid frames required to apply smoothing. |
| `smooth_joints` | <class 'bool'> | `True` | Whether to also smooth ``joints_world`` (21-joint MANO skeleton in world space). |
| `args` |  | `` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_hand_motion_smooth_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_hand_motion_smooth_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)