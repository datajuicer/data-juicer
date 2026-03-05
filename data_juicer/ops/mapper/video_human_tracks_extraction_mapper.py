from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.ASD_mapper_utils import scene_detect, \
    get_video_array_cv2, inference_video, track_shot, get_face_and_human_tracks,\
    post_merge

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.utils.model_utils import get_model, prepare_model
import gc,os

OP_NAME = 'video_human_tracks_extraction_mapper'

import torch
import pickle
import os
import tqdm


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoHumanTracksExtractionMapper(Mapper):
    """
    Extract face and human bounding box tracks from videos.

    This operator performs multi-stage processing including scene detection, 
    face detection (S3FD), face tracking, and human detection (YOLOv8). 
    It eventually generates synchronized face and human tracks and saves 
    the bbox sequences into pickle files.
    
    Source: This operator is a part of HumanVBench (CVPR 2026).
    """
    _accelerator = 'cuda'
    _batched_op = True
    _default_kwargs = {'upsample_num_times': 0}

    def __init__(self,
                 face_track_bbox_path: str = './HumanVBenchRecipe/dj_human_track',
                 YOLOv8_human_model_path: str = './thirdparty/humanvbench_models/YOLOv8_human/weights/best.pt',
                 tag_field_name_human_track_path: str = MetaKeys.human_track_data_path,
                 tag_field_name_people_num: str = MetaKeys.number_people_in_video,
                 *args,
                 **kwargs):
        """
        Initialization method.

        """
        kwargs.setdefault('mem_required', '10GB')
        super().__init__(*args, **kwargs)
        self._accelerator = 'cuda'
        self._init_parameters = self.remove_extra_parameters(locals())

        self.face_track_bbox_path = face_track_bbox_path

        self.human_detection_model_key = prepare_model(model_type='YOLOv8_human',  # 240MB
                                       pretrained_model_name_or_path=YOLOv8_human_model_path)
        
        self.face_detect_S3FD_model_key = prepare_model(model_type='face_detect_S3FD',
                                                        pretrained_model_name_or_path=None)
        self.tag_field_name_human_track_path = tag_field_name_human_track_path
        self.tag_field_name_people_num = tag_field_name_people_num

    def process_single(self, sample, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        loaded_video_keys = sample[self.video_key]

        Total_result = []
        min_people_in_video = []

        face_detect_S3FD = get_model(self.face_detect_S3FD_model_key, rank, self.use_cuda())
        human_detection_model = get_model(self.human_detection_model_key, rank, self.use_cuda())
        
        for id_out,video_key in enumerate(loaded_video_keys):
            # Scene detection for the video frames
            scene = scene_detect(video_key)

            video_array = get_video_array_cv2(video_key)

            # Face detection for the video frames
            faces = inference_video(video_array, face_detect_S3FD)

            # Face tracking
            allTracks, vidTracks = [], []
            minTrack = 10
            for shot in scene:
                if shot[1].frame_num - shot[0].frame_num >= minTrack: # Discard the shot frames less than minTrack frames
                    allTracks.extend(track_shot(faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
            
            # Get face and human tracks
            for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
                result = get_face_and_human_tracks(video_array, track, human_detection_model)
                if result:
                    vidTracks.append(result)
            # merge
            people_num_atleast, update_track = post_merge(vidTracks,video_array)   
            
            for i in range(len(update_track)):
                save_bbox_name = os.path.join(self.face_track_bbox_path, video_key.split("/")[-1][:-4] +'_'+str(i)+'.pkl')
                xy_bbox = update_track[i]['track']['bbox']
                xys_bbox = update_track[i]['proc_track']
                xy_human_bbox = update_track[i]['human_bbox']
                frames = update_track[i]['track']['frame']
                bbox_dict = {'frame':frames, 'xy_bbox':xy_bbox, 'xys_bbox':xys_bbox, 'xy_human_bbox':xy_human_bbox}
                f_save = open(save_bbox_name, 'wb')
                pickle.dump(bbox_dict, f_save)
                f_save.close()
                del update_track[i]['human_bbox']
                del update_track[i]['proc_track']
                del update_track[i]['track']
                update_track[i]['bbox_path'] = save_bbox_name


            Total_result.append(update_track)
            min_people_in_video.append(people_num_atleast)
            torch.cuda.empty_cache()
 
        sample[Fields.meta][self.tag_field_name_human_track_path] = Total_result
        sample[Fields.meta][self.tag_field_name_people_num] = min_people_in_video

        gc.collect()
        torch.cuda.empty_cache()

        return sample
