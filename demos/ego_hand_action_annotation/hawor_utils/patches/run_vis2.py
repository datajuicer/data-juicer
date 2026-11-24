"""
Modified from https://github.com/ThunderVVV/HaWoR/blob/main/lib/vis/run_vis2.py.
"""
from ..common_utils import prepare_hawor_and_add_to_path
prepare_hawor_and_add_to_path()

import os
import cv2
import numpy as np

import lib.vis.viewer as viewer_utils


def run_vis2_on_video_cam(res_dict, res_dict2, output_pth, focal_length, image_names, R_w2c=None, t_w2c=None, interactive=True):

    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            # "color": "pace-green",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1

    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            # "color": "pace-blue",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_w2c[:num_frames]
    Rt[:, :3, 3] = t_w2c[:num_frames]

    cols, rows = (width, height)
    K = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ]
    )
    vis_h = height
    vis_w = width

    data = viewer_utils.ViewerData(Rt, K, cols, rows, imgnames=image_names)
    batch = (meshes, data)

    if interactive:
        viewer = viewer_utils.ARCTICViewer(interactive=True, size=(vis_w, vis_h))
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    else:
        viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'])
        if os.path.exists(os.path.join(output_pth, 'aitviewer', "video_0.mp4")):
            os.remove(os.path.join(output_pth, 'aitviewer', "video_0.mp4"))
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
        return os.path.join(output_pth, 'aitviewer', "video_0.mp4")
