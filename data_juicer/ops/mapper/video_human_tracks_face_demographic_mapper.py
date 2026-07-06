import gc
import os
import pickle

import numpy as np

from data_juicer.utils.ASD_mapper_utils import get_video_array_cv2
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

torch = LazyLoader("torch")
deepface = LazyLoader("deepface")

OP_NAME = "video_human_tracks_face_demographic_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoHumantrackFaceDemographicMapper(Mapper):
    """
    Mapper to generate samples whose captions are generated based on
    a video-to-text model and sampled video frame.

    Source: This operator is a part of HumanVBench (CVPR 2026).
    """

    _batched_op = True

    def __init__(
        self,
        original_data_save_path="",
        detect_interval: int = 5,
        tag_field_name: str = MetaKeys.video_facetrack_attribute_demographic,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_video_blip: video-blip model name on huggingface
            to generate caption
        """
        super().__init__(*args, **kwargs)
        # avoid hanging when calling clip in multiprocessing
        torch.set_num_threads(1)

        self.interval = detect_interval
        self.original_data_save_path = original_data_save_path
        self.tag_field_name = tag_field_name

    def process_single(self, sample, rank=None):
        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        if MetaKeys.human_track_data_path not in sample[Fields.meta]:
            raise ValueError(
                "video_human_tracks_face_demographic_mapper must be operated after video_human_tracks_extraction_mapper."
            )

        Total_information = []
        video_samples = sample[Fields.meta][MetaKeys.human_track_data_path]
        loaded_video_keys = sample[self.video_key]

        for vedio_id, ASD_attribute_all_tracks_for_one_video in enumerate(video_samples):
            if len(ASD_attribute_all_tracks_for_one_video) == 0:
                Total_information.append([])
                continue
            description_for_each_track = []
            video_array = get_video_array_cv2(loaded_video_keys[vedio_id])
            for track_id, tracks_now in enumerate(ASD_attribute_all_tracks_for_one_video):
                face_attribute_dict_with_framestamp = {}

                bbox_path = tracks_now["bbox_path"]
                with open(bbox_path, "rb") as f:
                    bbox_data = pickle.load(f)
                    xys_bbox = bbox_data["xys_bbox"]
                    track_frame = bbox_data["frame"]

                total_len = len(track_frame)
                if total_len > 75:
                    interval = int(total_len / 15)
                else:
                    interval = self.interval

                start_frame_id_in = 0
                cs = 0.5
                while start_frame_id_in < len(track_frame):
                    start_frame_id_out = track_frame[start_frame_id_in]  # tag
                    frame_before_crop = video_array[start_frame_id_out]
                    bs = xys_bbox["s"][start_frame_id_in]  #
                    bsi = int(bs * (1 + 2 * cs))  #
                    frame = np.pad(
                        frame_before_crop, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110)
                    )
                    my = xys_bbox["y"][start_frame_id_in] + bsi  # BBox center Y
                    mx = xys_bbox["x"][start_frame_id_in] + bsi  # BBox center X
                    face = frame[
                        int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))
                    ]

                    face_message = deepface.DeepFace.analyze(
                        img_path=face,
                        actions=["age", "gender", "race"],
                        enforce_detection=False,
                        detector_backend="skip",
                    )
                    face_attribute_dict_with_framestamp[start_frame_id_out] = face_message

                    start_frame_id_in = start_frame_id_in + interval

                mini_dict = {}
                mini_dict["data"] = face_attribute_dict_with_framestamp

                if not os.path.exists(self.original_data_save_path):
                    os.makedirs(self.original_data_save_path, exist_ok=True)
                    print(f"Created directory: {self.original_data_save_path}")

                save_path = os.path.join(
                    self.original_data_save_path,
                    loaded_video_keys[vedio_id].split("/")[-1][:-4] + "_" + str(track_id) + ".pkl",
                )
                with open(save_path, "wb") as f:
                    pickle.dump(face_attribute_dict_with_framestamp, f)

                mini_dict["face_attribute_path"] = save_path
                description_for_each_track.append(mini_dict)

            save_track_infor = []
            for track_data in description_for_each_track:
                new_track_data = {}
                new_track_data["face_attribute_path"] = track_data["face_attribute_path"]
                track_attri_data = track_data["data"]
                age_list = [track_attri_data[key][0]["age"] for key in track_attri_data if track_attri_data[key] != []]
                dominant_gender_list = [
                    track_attri_data[key][0]["dominant_gender"]
                    for key in track_attri_data
                    if track_attri_data[key] != []
                ]
                dominant_race_list = [
                    track_attri_data[key][0]["dominant_race"] for key in track_attri_data if track_attri_data[key] != []
                ]
                # emotion_list = [track_attri_data[key][0]['emotion'] for key in track_attri_data if track_attri_data[key] != []]

                new_track_data["age"] = self.find_median(age_list) if age_list else None
                new_track_data["gender"] = (
                    self.most_frequent_element_ratio(dominant_gender_list) if dominant_gender_list else (None, "0")
                )
                new_track_data["race"] = (
                    self.most_frequent_element_ratio(dominant_race_list) if dominant_race_list else (None, "0")
                )

                save_track_infor.append(new_track_data)

            Total_information.append(save_track_infor)

        sample[Fields.meta][self.tag_field_name] = Total_information

        gc.collect()
        return sample

    def find_median(self, int_list):
        """
        Finds the median of a list of integers.

        :param int_list: List of integers.
        :return: The median of the list.
        """
        sorted_list = sorted(int_list)
        n = len(sorted_list)

        if n == 0:
            raise ValueError("The list is empty.")

        mid = n // 2

        if n % 2 == 0:
            median = (sorted_list[mid - 1] + sorted_list[mid]) / 2
        else:
            median = sorted_list[mid]

        return median

    def most_frequent_element_ratio(self, str_list):
        """
        Finds the most frequent element in a list of strings and its proportion.

        :param str_list: List of strings.
        :return: A tuple (most_frequent_element, proportion).
        """
        if not str_list:
            raise ValueError("The list is empty.")

        from collections import Counter

        element_counts = Counter(str_list)
        most_common_element, most_common_count = element_counts.most_common(1)[0]
        proportion = most_common_count / len(str_list)

        return most_common_element, str(proportion)

    def top_3_emotions(self, emotion_list):
        """
        Calculates the average score for each emotion and returns the top 3 emotions with the highest average scores.

        :param emotion_list: List of dictionaries, each containing seven emotions and their corresponding scores.
        :return: List of tuples containing the top 3 emotions and their average scores.
        """
        if not emotion_list:
            raise ValueError("The list is empty.")

        emotion_totals = {}
        emotion_counts = {}

        for emotion_dict in emotion_list:
            for emotion, score in emotion_dict.items():
                if emotion in emotion_totals:
                    emotion_totals[emotion] += score
                    emotion_counts[emotion] += 1
                else:
                    emotion_totals[emotion] = score
                    emotion_counts[emotion] = 1

        emotion_averages = {emotion: emotion_totals[emotion] / emotion_counts[emotion] for emotion in emotion_totals}

        top_3 = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)[:3]

        return top_3
