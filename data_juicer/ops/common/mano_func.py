from typing import Optional

from data_juicer.utils.lazy_loader import LazyLoader

smplx = LazyLoader("smplx")
torch = LazyLoader("torch")


class MANO(smplx.MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(MANO, self).__init__(*args, **kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        # 2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            import pickle

            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"), dtype=torch.float32),
            )
        self.register_buffer(
            "extra_joints_idxs",
            smplx.utils.to_tensor(list(smplx.vertex_ids.vertex_ids["mano"].values()), dtype=torch.long),
        )
        self.register_buffer("joint_map", torch.tensor(mano_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs) -> smplx.utils.MANOOutput:
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANO, self).forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, "joint_regressor_extra"):
            extra_joints = smplx.lbs.vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output

    def query(self, hmr_output):
        batch_size = hmr_output["pred_rotmat"].shape[0]
        pred_rotmat = hmr_output["pred_rotmat"].reshape(batch_size, -1, 3, 3)
        pred_shape = hmr_output["pred_shape"].reshape(batch_size, 10)

        mano_output = self(
            global_orient=pred_rotmat[:, [0]], hand_pose=pred_rotmat[:, 1:], betas=pred_shape, pose2rot=False
        )

        return mano_output
