import gc
import glob
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import tqdm
from loguru import logger

from data_juicer.utils.ASD_mapper_utils import (
    crop_video_with_facetrack,
    evaluate_network,
    get_video_array_cv2,
    longest_continuous_actives,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

torch = LazyLoader("torch")

sys.path.append("./thirdparty/humanvbench_models/Light-ASD")

OP_NAME = "video_active_speaker_detect_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoActiveSpeakerDetectMapper(Mapper):
    """
    Detect active speakers in a video by analyzing visual face tracks and
    audio signals, including consistency checks for gender and age.

    Source: This operator is a part of HumanVBench (CVPR 2026).
    """

    _accelerator = "cuda"
    _batched_op = True

    _default_kwargs = {"upsample_num_times": 0}

    def __init__(
        self,
        temp_save_path: str = "./temp_path",
        Light_ASD_model_path: str = "./thirdparty/humanvbench_models/Light-ASD/weight/finetuning_TalkSet.model",
        active_threshold: int = 15,
        active_speaker_flag: str = MetaKeys.active_speaker_flag,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param blur_type:
        """
        kwargs.setdefault("mem_required", "10GB")
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self.active_threshold = active_threshold

        self.temp_save_path = temp_save_path
        os.makedirs(self.temp_save_path, exist_ok=True)

        # Initialize ASD model
        self.ASD_model_key = prepare_model(model_type="Light_ASD", pretrained_model_name_or_path=Light_ASD_model_path)

        self.active_speaker_flag = active_speaker_flag

    def active_speaker_detection_revise(self, active_score, is_child_descrip, speech_audio, face_gender):
        speech_child = speech_audio["child"][0]
        speech_male = speech_audio["male"][0]
        speech_female = speech_audio["female"][0]
        if speech_male > speech_female:
            speech_gender = "Man"
            speech_gender_confidence = speech_male
        else:
            speech_gender = "Woman"
            speech_gender_confidence = speech_female

        if "No" in is_child_descrip or "no" in is_child_descrip:
            is_child_apperance = False
        else:
            is_child_apperance = True

        if speech_child < 0.1:
            is_child_voice = False
        elif speech_audio["Age"][0] <= 12:
            is_child_voice = True
        else:
            is_child_voice = "Not Sure"

        # Consistency detection: only perform false positive detection on positive samples
        if active_score > self.active_threshold:
            speak_active = True
            # age consistency test:
            if not is_child_voice == "Not Sure":
                if is_child_apperance == is_child_voice:
                    # gender consistency test
                    if speech_gender_confidence > 0.85 and float(face_gender[1]) > 0.85:
                        if not speech_gender == face_gender[0]:
                            speak_active = False
                else:
                    speak_active = False
            return speak_active
        else:
            return False

    def process_single(self, sample, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            if Fields.meta not in sample:
                sample[Fields.meta] = {}
            sample[Fields.meta][self.active_speaker_flag] = []
            return sample

        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        meta = sample.get(Fields.meta, {})

        # Core dependencies: Both human tracks and audio tags are required for ASD
        has_tracks = MetaKeys.human_track_data_path in meta
        has_audio_tags = MetaKeys.video_audio_tags in meta

        if not (has_tracks and has_audio_tags):
            missing = []
            if not has_tracks:
                missing.append(MetaKeys.human_track_data_path)
            if not has_audio_tags:
                missing.append(MetaKeys.video_audio_tags)

            logger.warning(
                f"[{OP_NAME}] Skip sample: Missing mandatory keys {missing}. "
                f"video_active_speaker_detect_mapper must be operated after video_tagging_from_audio_mapper. "
                f"video_active_speaker_detect_mapper must be operated after video_human_tracks_extraction_mapper. "
                f"Please ensure prior Mappers are executed correctly."
            )
            return sample

        # Optional dependencies for 'revise' function
        has_audio_attr = MetaKeys.audio_speech_attribute in meta
        has_face_attr = MetaKeys.video_facetrack_attribute_demographic in meta
        has_child_attr = MetaKeys.video_track_is_child in meta

        revise_available = has_audio_attr and has_face_attr and has_child_attr
        if not revise_available:
            logger.info(
                f"[{OP_NAME}] Some metadata missing. Running in 'Basic Mode' without consistency detection. "
                f"To enable full consistency detection, ensure these OPs are executed: video_audio_detect_age_gender_mapper, video_humantrack_face_demographic_mapper and video_captioning_from_human_tracks_mapper."
            )

        loaded_video_keys = sample[self.video_key]

        if revise_available:
            audio_speech_attribute = sample[Fields.meta][MetaKeys.audio_speech_attribute]
            face_demographic = sample[Fields.meta][MetaKeys.video_facetrack_attribute_demographic][0]
            child_flag = sample[Fields.meta][MetaKeys.video_track_is_child][0]

        Total_result = []

        temp_dir = tempfile.mkdtemp(dir=self.temp_save_path)
        pyaviPath = os.path.join(temp_dir, "pyavi")
        pyframesPath = os.path.join(temp_dir, "pyframes")
        pyworkPath = os.path.join(temp_dir, "pywork")
        pycropPath = os.path.join(temp_dir, "pycrop")

        audio_tag = sample[Fields.meta][MetaKeys.video_audio_tags]
        asd_detection_model = get_model(self.ASD_model_key, rank=rank)

        for id_out, video_key in enumerate(loaded_video_keys):
            os.makedirs(pyaviPath, exist_ok=True)  # The path for the input video, input audio, output video
            os.makedirs(pyframesPath, exist_ok=True)  # Save all the video frames
            os.makedirs(pyworkPath, exist_ok=True)  # Save the results in this process by the pckl method
            os.makedirs(pycropPath, exist_ok=True)  # Save the detected face clips (audio+video) in this process

            # Extract audio
            audio_is_empty = False
            audioFilePath = os.path.join(pyaviPath, "audio.wav")
            command = "ffmpeg -y -i '%s' -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % (
                video_key,
                10,
                audioFilePath,
            )
            if audio_tag[id_out] == "EMPTY":
                audio_is_empty = True
            else:
                subprocess.call(command, shell=True, stdout=None)

            video_array = get_video_array_cv2(video_key)

            def load_pkl(file_path):
                with open(file_path, "rb") as file:
                    return pickle.load(file)

            # get allTracks
            allTracks = [
                load_pkl(item["bbox_path"]) for item in sample[Fields.meta][MetaKeys.human_track_data_path][id_out]
            ]

            # Face clips cropping
            for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
                result = crop_video_with_facetrack(
                    video_array, track, os.path.join(pycropPath, "%05d" % ii), audioFilePath, audio_is_empty
                )
                if not result:
                    raise ValueError("something wrong with crop_video_with_facetrack.")

            # Active Speaker Detection
            if audio_tag[id_out] == "Speech":
                files = glob.glob("%s/*.avi" % pycropPath)
                files.sort()
                try:
                    scores = evaluate_network(files, asd_detection_model, pycropPath)
                except Exception:
                    scores = [[-10000]] * len(allTracks)

            else:
                scores = [[-10000]] * len(allTracks)

            for id in range(len(scores)):
                allTracks[id]["active_scores"] = scores[id]

            update_track = allTracks
            # for validation
            # visualization(vidTracks, scores, video_array, pyaviPath)

            # Clean up subdirectories for this iteration
            for sub_dir in [pyaviPath, pyframesPath, pyworkPath, pycropPath]:
                if os.path.exists(sub_dir):
                    shutil.rmtree(sub_dir)

            speak_flag_for_tracks_in_a_video = []
            for track_idx, track_i in enumerate(update_track):
                active_count = longest_continuous_actives(track_i["active_scores"])

                if revise_available:
                    try:
                        audio_attri = audio_speech_attribute[id_out][0]
                        is_child_descrip = child_flag[id_out][track_idx][0]
                        face_gender = face_demographic[id_out][track_idx]["gender"]
                        flag = self.active_speaker_detection_revise(
                            active_count, is_child_descrip, audio_attri, face_gender
                        )
                    except Exception:
                        if active_count > self.active_threshold:
                            flag = True
                        else:
                            flag = False
                else:
                    if active_count > self.active_threshold:
                        flag = True
                    else:
                        flag = False
                speak_flag_for_tracks_in_a_video.append(flag)

            Total_result.append(speak_flag_for_tracks_in_a_video)
            torch.cuda.empty_cache()

        # Clean up the entire temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        sample[Fields.meta][self.active_speaker_flag] = Total_result

        gc.collect()
        torch.cuda.empty_cache()

        return sample
