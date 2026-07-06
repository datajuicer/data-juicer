import copy
import math
import os
import pickle
import subprocess

import numpy as np
import tqdm
from scipy.interpolate import interp1d
from scipy.io import wavfile

from data_juicer.utils.lazy_loader import LazyLoader

cv2 = LazyLoader("cv2", "opencv-contrib-python")
mp = LazyLoader("moviepy")
python_speech_features = LazyLoader("python_speech_features")
torch = LazyLoader("torch")
DeepFace = None  # lazy loaded below
VideoManager = None  # lazy loaded below


def _ensure_deepface():
    global DeepFace
    if DeepFace is None:
        from deepface import DeepFace as _DeepFace

        DeepFace = _DeepFace
    return DeepFace


def _ensure_scenedetect():
    global VideoManager
    if VideoManager is None:
        from scenedetect.video_manager import VideoManager as _VM

        VideoManager = _VM
    return VideoManager


def scene_detect(videoFilePath):
    # CPU: Scene detection, output is the list of each shot's time duration
    _VideoManager = _ensure_scenedetect()
    videoManager = _VideoManager([videoFilePath])
    sceneList = [(videoManager.base_timecode, videoManager.duration)]
    return sceneList


def inference_video(video_array, DET):
    # from model.faceDetector.s3fd import S3FD
    # GPU: Face detection, output is the list contains the face location and score in this frame
    # DET = S3FD(device='cuda')
    dets = []
    total_frame = video_array.shape[0]
    for fidx in range(total_frame):
        image = video_array[fidx]
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[0.25])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
            )  # dets has the frames info, bbox info, conf info
        # sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))

    return dets


def get_video_array_cv2(videoFilePath):
    cap = cv2.VideoCapture(videoFilePath)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {videoFilePath}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    frames_array = np.array(frames)
    return frames_array


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol is True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(sceneFaces, numFailedDet=8, minTrack=10):
    # CPU: Face tracking
    iouThres = 0.55  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            best_match = None
            max_iou = 0
            frameFaces_ori = copy.deepcopy(frameFaces)
            for face in frameFaces_ori:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= numFailedDet and not face["frame"] == track[-1]["frame"]:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])

                    if iou > iouThres and iou > max_iou:
                        best_match = face
                        max_iou = iou
                else:
                    break

            if best_match is not None:
                track.append(best_match)
                frameFaces.remove(best_match)

        if track == []:
            break
        elif len(track) > minTrack:
            frameNum = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])
            frameI = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:, 2] - bboxesI[:, 0]), np.mean(bboxesI[:, 3] - bboxesI[:, 1])) > 1:
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks


def find_human_bounding_box(face_bbox, human_bboxes):
    head_x1, head_y1, head_x2, head_y2 = face_bbox
    head_center_x = (head_x1 + head_x2) / 2

    candidate_bboxes = []

    for human_bbox in human_bboxes:
        human_x1, human_y1, human_x2, human_y2 = human_bbox

        if (human_x1 <= head_x1 and head_x2 <= human_x2) and (human_y1 <= head_y1 and head_y2 <= human_y2):
            candidate_bboxes.append(human_bbox)

    if not candidate_bboxes:
        return ()

    # Select the human body bounding box with the smallest distance between (x1 + x2) / 2 and (x1 + x2) / 2 of face_bbox
    closest_bbox = min(
        candidate_bboxes,
        key=lambda bbox: (((bbox[0] + bbox[2]) / 2) - head_center_x) ** 2 + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
    )

    return closest_bbox


def update_negative_ones(values):
    n = len(values)
    i = 0

    while i < n:
        if values[i] == -1:
            # Find the nearest number on the left
            left_index = i - 1
            while left_index >= 0 and values[left_index] == -1:
                left_index -= 1

            # Find the nearest number on the right
            right_index = i + 1
            while right_index < n and values[right_index] == -1:
                right_index += 1

            # Update the value of -1
            if left_index >= 0 and right_index < n:
                left_value = values[left_index]
                right_value = values[right_index]
                values[i] = (left_value + right_value) / 2
            elif left_index >= 0:
                values[i] = values[left_index]
            elif right_index < n:
                values[i] = values[right_index]
            else:
                raise ValueError("Unable to find valid values ​​on both the left and right to update -1 at index {i}")
        i += 1

    return values


def detect_and_mark_anomalies(data, window_size=7, std_multiplier=2):
    data = np.array(data)
    result = data.copy()

    for i in range(len(data)):
        if data[i] > 0:
            start = max(0, i - window_size)
            end = min(len(data), i + window_size + 1)
            neighbors = data[start:end]

            neighbors = np.delete(neighbors, np.where(neighbors == data[i]))

            positive_neighbors = neighbors[neighbors > 0]

            if len(positive_neighbors) < 2:
                continue

            mean = np.mean(positive_neighbors)
            std = np.std(positive_neighbors)

            if abs(data[i] - mean) > std * std_multiplier:
                result[i] = -1

    return result


def crop_video_with_facetrack(video_array, track, cropFile, audioFilePath, is_empty=False):
    if is_empty:
        return True

    dets = track["xys_bbox"]
    # CPU: crop the face clips
    vOut = cv2.VideoWriter(cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224))  # Write video

    for fidx, frame in enumerate(track["frame"]):
        cs = 0.4
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = video_array[frame]
        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (audioFilePath, 10, audioStart, audioEnd, audioTmp)
    )
    subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % (
        cropFile,
        audioTmp,
        10,
        cropFile,
    )  # Combine audio and video file
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")
    return True


def evaluate_network(files, s, pycropPath):
    # GPU: active speaker detection by pretrained model
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}  # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split("/")[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(pycropPath, fileName + ".wav"))
        if len(audio) == 0:
            scores = np.array([-5])
            allScores.append(scores)
            continue

        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        video = cv2.VideoCapture(os.path.join(pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)) : int(112 + (112 / 2)), int(112 - (112 / 2)) : int(112 + (112 / 2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = np.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]
        allScore = []  # Evaluation use model
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = (
                        torch.FloatTensor(audioFeature[i * duration * 100 : (i + 1) * duration * 100, :])
                        .unsqueeze(0)
                        .to(next(s.parameters()).device)
                    )
                    inputV = (
                        torch.FloatTensor(videoFeature[i * duration * 25 : (i + 1) * duration * 25, :, :])
                        .unsqueeze(0)
                        .to(next(s.parameters()).device)
                    )
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
                    del inputA
                    del inputV
                    del embedA
                    del embedV
            allScore.append(scores)
        allScore = np.round((np.mean(np.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores, video_array, pyaviPath):
    # CPU: visulize the result for video format

    faces = [[] for i in range(video_array.shape[0])]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]  # average smoothing
            s = np.mean(s)
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    firstImage = video_array[0]
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(pyaviPath, "video_only.avi"), cv2.VideoWriter_fourcc(*"XVID"), 25, (fw, fh))
    colorDict = {0: 0, 1: 255}
    for fidx in tqdm.tqdm(range(video_array.shape[0])):
        image = video_array[fidx]
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        vOut.write(image)
    vOut.release()
    command = "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % (
        os.path.join(pyaviPath, "video_only.avi"),
        os.path.join(pyaviPath, "audio.wav"),
        10,
        os.path.join(pyaviPath, "video_out.avi"),
    )
    subprocess.call(command, shell=True, stdout=None)


def calculate_good_matches(matches, ratio=0.75):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return len(good_matches)


def find_max_intersection_and_remaining_dicts(dicts):
    if not dicts:
        return [], []

    track_frames = [d["track"]["frame"] for d in dicts]

    all_elements = set()
    for frame in track_frames:
        all_elements.update(frame)

    max_combination_indices = []
    _ = set()

    for elem in all_elements:
        current_combination_indices = []
        current_intersection = set([elem])

        for i, frame in enumerate(track_frames):
            if elem in frame:
                current_combination_indices.append(i)
                current_intersection.intersection_update(frame)

        if len(current_combination_indices) > len(max_combination_indices):
            max_combination_indices = current_combination_indices
            _ = current_intersection

    max_combination = [dicts[i] for i in max_combination_indices]
    remaining_dicts = [d for i, d in enumerate(dicts) if i not in max_combination_indices]

    return max_combination, remaining_dicts


def get_faces_array(frame, s, x, y):
    cs = 0.4
    bs = s  # Detection box size
    bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
    image = frame
    frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
    my = y + bsi  # BBox center Y
    mx = x + bsi  # BBox center X
    face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
    return face


def order_track_distance(track1, track2, video_array):
    # Get the last face frame of track1 and the first face frame of track2
    track1_end_frame = video_array[track1["track"]["frame"][-1]]
    track1_s = track1["proc_track"]["s"][-1]
    track1_x = track1["proc_track"]["x"][-1]
    track1_y = track1["proc_track"]["y"][-1]
    track1_end_face_array = get_faces_array(track1_end_frame, track1_s, track1_x, track1_y)

    track2_start_frame = video_array[track2["track"]["frame"][0]]
    track2_s = track2["proc_track"]["s"][0]
    track2_x = track2["proc_track"]["x"][0]
    track2_y = track2["proc_track"]["y"][0]
    track2_strat_face_array = get_faces_array(track2_start_frame, track2_s, track2_x, track2_y)

    # Calculate the area overlap ratio
    track1_bbox = track1["track"]["bbox"][-1]
    track2_bbox = track2["track"]["bbox"][0]
    iou = bb_intersection_over_union(track1_bbox, track2_bbox)
    if iou <= 0.2:
        distance_iou = 10000
    else:
        distance_iou = math.exp(-5 * iou)

    normalized_distance = 0

    # face_id distance (with facenet)
    result = _ensure_deepface().verify(
        track1_end_face_array, track2_strat_face_array, model_name="Facenet", detector_backend="skip"
    )
    facenet_distance = result["distance"]
    if facenet_distance > 0.85:
        facenet_distance = facenet_distance + 10000

    distance = 2 * distance_iou + normalized_distance + facenet_distance

    return distance


def update_remain(remaining_dicts, pop_item):
    updated_dicts = [
        item
        for item in remaining_dicts
        if item["track"]["bbox"].shape != pop_item["track"]["bbox"].shape
        or (item["track"]["bbox"] != pop_item["track"]["bbox"]).any()
    ]
    return updated_dicts


def order_merge_tracks(track1, track2):
    new_track = {}
    new_track["proc_track"] = {}
    new_track["proc_track"]["x"] = track1["proc_track"]["x"] + track2["proc_track"]["x"]
    new_track["proc_track"]["y"] = track1["proc_track"]["y"] + track2["proc_track"]["y"]
    new_track["proc_track"]["s"] = track1["proc_track"]["s"] + track2["proc_track"]["s"]
    new_track["human_bbox"] = {}
    new_track["human_bbox"]["x1"] = track1["human_bbox"]["x1"] + track2["human_bbox"]["x1"]
    new_track["human_bbox"]["y1"] = track1["human_bbox"]["y1"] + track2["human_bbox"]["y1"]
    new_track["human_bbox"]["x2"] = track1["human_bbox"]["x2"] + track2["human_bbox"]["x2"]
    new_track["human_bbox"]["y2"] = track1["human_bbox"]["y2"] + track2["human_bbox"]["y2"]

    new_track["track"] = {}
    for key in list(track1["track"].keys()):
        object1 = track1["track"][key]
        object2 = track2["track"][key]
        if isinstance(object1, np.ndarray):
            new_track["track"][key] = np.concatenate((object1, object2))
        elif isinstance(object1, list):
            new_track["track"][key] = object1 + object2
        else:
            raise ("new data type")

    return new_track


def post_merge(vidTracks, video_array):
    # Find the maximum overlapping tracks as the initial anchor
    anchor_combination, remaining_dicts = find_max_intersection_and_remaining_dicts(vidTracks)
    end_frame = video_array.shape[0]
    continue_flag = np.ones((len(anchor_combination), 2))
    max_iteration = 10
    iteration_count = 0
    while iteration_count < max_iteration and continue_flag.sum() > 0:
        for track_ind in range(len(anchor_combination)):
            track = anchor_combination[track_ind]
            # Try to extend forward
            if continue_flag[track_ind][0]:
                if track["track"]["frame"][0] == 0:
                    continue_flag[track_ind][0] = 0
                else:
                    # Find the candidate that is connected to it and is in the front row
                    possible_prior_tracks = []
                    for checktrack in remaining_dicts:
                        if (
                            checktrack["track"]["frame"][-1] + 1 == track["track"]["frame"][0]
                            or checktrack["track"]["frame"][-1] + 2 == track["track"]["frame"][0]
                        ):
                            possible_prior_tracks.append(checktrack)
                    # If it is not zero, then check the calculated distance
                    if len(possible_prior_tracks) > 0:
                        distance_score_list = []
                        for possible_prior_track in possible_prior_tracks:
                            distance_score_list.append(order_track_distance(possible_prior_track, track, video_array))
                        distance_score_array = np.array(distance_score_list)
                        if min(distance_score_array) < 10000:
                            min_index = np.argmin(distance_score_array)
                            new_anchor = order_merge_tracks(possible_prior_tracks[min_index], track)
                            # update_anchor()
                            anchor_combination[track_ind] = new_anchor
                            track = new_anchor
                            remaining_dicts = update_remain(remaining_dicts, possible_prior_tracks[min_index])
                        else:
                            continue_flag[track_ind][0] = 0
                    else:
                        continue_flag[track_ind][0] = 0
            # Try to extend backwards
            if continue_flag[track_ind][1]:
                if track["track"]["frame"][-1] == end_frame:
                    continue_flag[track_ind][0] = 0
                else:
                    # Find the candidate that is connected to it and in front of it
                    possible_after_tracks = []
                    for checktrack in remaining_dicts:
                        if (
                            checktrack["track"]["frame"][0] - 1 == track["track"]["frame"][-1]
                            or checktrack["track"]["frame"][0] - 2 == track["track"]["frame"][-1]
                        ):
                            possible_after_tracks.append(checktrack)
                    # If it is not zero, then check the calculated distance
                    if len(possible_after_tracks) > 0:
                        distance_score_list = []
                        for possible_after_track in possible_after_tracks:
                            distance_score_list.append(order_track_distance(track, possible_after_track, video_array))
                        distance_score_array = np.array(distance_score_list)
                        if min(distance_score_array) < 10000:
                            min_index = np.argmin(distance_score_array)
                            new_anchor = order_merge_tracks(track, possible_after_tracks[min_index])
                            # update_anchor()
                            anchor_combination[track_ind] = new_anchor
                            remaining_dicts = update_remain(remaining_dicts, possible_after_tracks[min_index])
                        else:
                            continue_flag[track_ind][1] = 0
                    else:
                        continue_flag[track_ind][1] = 0

    final_tracks = anchor_combination + remaining_dicts
    if len(final_tracks) > 5:
        sorted_tracks = sorted(final_tracks, key=lambda x: len(x["track"]["frame"]), reverse=True)
        top_tracks = sorted_tracks[:5]
    else:
        top_tracks = final_tracks
        # return len(anchor_combination), top_5_tracks
    returntracks = []
    for item in top_tracks:
        if len(item["track"]["frame"]) > 15:
            returntracks.append(item)
    return len(anchor_combination), returntracks


def longest_continuous_actives(arr):
    max_length = 0
    current_length = 0

    for num in arr:
        if num > 0:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
        else:
            current_length = 0

    return max_length


def annotate_video_with_bounding_boxes_with_audio(video_path, q_human_video_track_bbox, output_path):
    bbox_path = q_human_video_track_bbox["bbox_path"]
    frame_indices = q_human_video_track_bbox["track"]["frame"]
    video_array = get_video_array_cv2(video_path)

    with open(bbox_path, "rb") as f:
        bbox_data = pickle.load(f)
        xy_bbox = bbox_data["xy_bbox"]

    # Get video dimensions and frame rate
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original video frame rate
    num_frames, height, width, channels = video_array.shape
    assert channels == 3, "Input video must have 3 channels (BGR)."

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
    temp_video_path = output_path.split(".")[0] + "temp.mp4"
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))  # Use original FPS

    # Annotate video frames with bounding boxes
    for i in range(num_frames):
        frame = video_array[i]
        if i in frame_indices:
            idx = frame_indices.index(i)
            x1, y1, x2, y2 = xy_bbox[idx]
            # Draw bounding box
            thickness = max(int((x2 - x1) / 40), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness)

        # Write frame to temporary video
        out.write(frame)

    out.release()
    cap.release()  # Release the video capture object

    # Load original video and audio
    original_video = mp.VideoFileClip(video_path)
    annotated_video = mp.VideoFileClip(temp_video_path)

    # Combine annotated video with original audio, ensuring alignment
    final_video = annotated_video.set_audio(original_video.audio)

    # Write the final output video with audio
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)

    # Clean up temporary video file
    annotated_video.close()
    original_video.close()

    # Optionally, remove the temporary video file
    import os

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    return output_path


def annotate_video_with_bounding_boxes_withText_with_audio(video_path, q_human_video_track_bbox, output_path, numbers):
    bbox_path = q_human_video_track_bbox["bbox_path"]
    frame_indices = q_human_video_track_bbox["track"]["frame"]
    video_array = get_video_array_cv2(video_path)

    with open(bbox_path, "rb") as f:
        bbox_data = pickle.load(f)
        xy_bbox = bbox_data["xy_bbox"]

    # Get video dimensions and frame rate
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original video frame rate
    num_frames, height, width, channels = video_array.shape
    assert channels == 3, "Input video must have 3 channels (BGR)."

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
    temp_video_path = output_path.split(".")[0] + "temp.mp4"
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))  # Use original FPS

    # Annotate video frames with bounding boxes
    for i in range(num_frames):
        frame = video_array[i]
        if i in frame_indices:
            idx = frame_indices.index(i)
            x1, y1, x2, y2 = xy_bbox[idx]
            # Draw bounding box
            thickness = max(int((x2 - x1) / 40), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness)
            # Put the number in the top-left corner of the bounding box
            cv2.putText(frame, numbers, (int(x1) + 10, int(y1) + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        # Write frame to temporary video
        out.write(frame)

    out.release()
    cap.release()  # Release the video capture object

    # Load original video and audio
    original_video = mp.VideoFileClip(video_path)
    annotated_video = mp.VideoFileClip(temp_video_path)

    # Combine annotated video with original audio, ensuring alignment
    final_video = annotated_video.set_audio(original_video.audio)

    # Write the final output video with audio
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)

    # Clean up temporary video file
    annotated_video.close()
    original_video.close()

    # Optionally, remove the temporary video file
    import os

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    return output_path


def annotate_video_with_bounding_boxes(video_array, frame_indices, bounding_boxes, output_path):
    """
    Annotates specified frames in the video with bounding boxes and saves the result to a new video file.

    :param video_array: Input video as a numpy array with shape (num_frames, height, width, channels).
    :param frame_indices: List of frame indices to annotate.
    :param bounding_boxes: Array of bounding box coordinates with shape (num_frames_to_annotate, 4), where each bounding box is (x, y, w, h).
    :param output_path: Path to save the output video.
    """
    # Get video dimensions
    num_frames, height, width, channels = video_array.shape
    assert channels == 3, "Input video must have 3 channels (BGR)."

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # option 1: keep all video
    for i in range(num_frames):
        frame = video_array[i]
        if i in frame_indices:
            idx = frame_indices.index(i)
            x1, y1, x2, y2 = bounding_boxes[idx]
            # Draw bounding box
            thinkness = max(int((x2 - x1) / 40), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thinkness)

        # Write frame to output video
        out.write(frame)

    # option 2:crap
    # for in_id, out_id in enumerate(frame_indices):
    #     frame = video_array[out_id]
    #     x1, y1, x2, y2 = bounding_boxes[in_id]
    #     # Draw bounding box
    #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    #     # Write frame to output video
    #     out.write(frame)

    out.release()
    return output_path


def crop_from_array(frame_before_crop, coords):
    x1, y1, x2, y2 = coords
    cropped_frame = frame_before_crop[y1:y2, x1:x2]
    return cropped_frame
