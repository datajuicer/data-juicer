import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, MetaKeys

from ..base_op import OPERATORS, Mapper

OP_NAME = "video_atomic_action_segment_mapper"


@OPERATORS.register_module(OP_NAME)
class VideoAtomicActionSegmentMapper(Mapper):
    """Segment a unified hand trajectory into atomic action clips.

    Implements the algorithm from paper https://arxiv.org/pdf/2510.21571:

        "we detect speed minima of the 3D hand wrists in the world space
        and use them as cutting points.  We smooth the hand trajectory and
        select points that are local speed minima within a fixed window
        centered on each point."

    The operator reads the merged hand_action_tags (output of
    ``VideoClipReassemblyMapper``) and produces a list of segments.
    Each segment contains the start and end frame indices, plus sliced
    states / actions / joints for that segment.

    Segmentation is applied **independently** for left and right hands.
    A frame is a cutting point if it is a speed local minimum within a
    window of ``min_window`` frames on each side.

    Output field (``segment_field``) structure::

        [
            {
                "hand_type": "right",
                "segment_id": 0,
                "start_frame": 10,
                "end_frame": 45,
                "states": [...],
                "actions": [...],
                "valid_frame_ids": [...],
                "joints_world": [...],
            },
            ...
        ]
    """

    def __init__(
        self,
        hand_action_field: str = MetaKeys.hand_action_tags,
        segment_field: str = "atomic_action_segments",
        speed_smooth_window: int = 5,
        min_window: int = 15,
        min_segment_frames: int = 8,
        max_segment_frames: int = 300,
        hand_type: str = "both",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hand_action_field: Meta field storing merged hand action
            results (output of VideoClipReassemblyMapper).
        :param segment_field: Output meta field for atomic segments.
        :param speed_smooth_window: Window size for Savitzky-Golay
            smoothing of the speed signal before minima detection.
            Must be odd.
        :param min_window: Half-window size for local minima detection.
            A frame is a local minimum only if it is the minimum
            within ``[t - min_window, t + min_window]``.
            Larger values → fewer, longer segments.
        :param min_segment_frames: Minimum frames per segment.
            Segments shorter than this are merged with neighbors.
        :param max_segment_frames: Maximum frames per segment.
            Segments longer than this are forcibly split at
            the deepest speed minimum.
        :param hand_type: Which hand(s) to segment: 'left', 'right',
            or 'both'.
        """
        super().__init__(*args, **kwargs)
        self.hand_action_field = hand_action_field
        self.segment_field = segment_field
        self.speed_smooth_window = speed_smooth_window
        self.min_window = min_window
        self.min_segment_frames = min_segment_frames
        self.max_segment_frames = max_segment_frames
        self.hand_type = hand_type

    # ------------------------------------------------------------------
    # Speed computation & smoothing
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_speed(positions: np.ndarray) -> np.ndarray:
        """Compute per-frame wrist speed from world-space positions.

        Returns an array of length N where speed[0] = 0.
        """
        if len(positions) < 2:
            return np.zeros(len(positions))
        vel = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return np.concatenate([[0.0], vel])

    @staticmethod
    def _smooth_speed(
        speed: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Smooth speed signal with Savitzky-Golay filter."""
        n = len(speed)
        if n < 5:
            return speed.copy()

        try:
            from scipy.signal import savgol_filter

            win = min(window, n)
            if win % 2 == 0:
                win -= 1
            if win < 3:
                return speed.copy()
            return savgol_filter(speed, win, polyorder=2)
        except Exception:
            return speed.copy()

    # ------------------------------------------------------------------
    # Local minima detection
    # ------------------------------------------------------------------
    @staticmethod
    def _find_local_minima(
        speed: np.ndarray,
        half_window: int,
    ) -> list[int]:
        """Find indices that are local speed minima within a window.

        A frame t is a local minimum if speed[t] <= speed[k] for all k
        in [t - half_window, t + half_window].
        """
        n = len(speed)
        minima = []
        for t in range(1, n - 1):
            lo = max(0, t - half_window)
            hi = min(n, t + half_window + 1)
            if speed[t] <= np.min(speed[lo:hi]):
                minima.append(t)
        return minima

    # ------------------------------------------------------------------
    # Segment merging (too-short) and splitting (too-long)
    # ------------------------------------------------------------------
    def _merge_short_segments(
        self,
        cut_points: list[int],
        n_frames: int,
    ) -> list[int]:
        """Remove cut points that would produce segments shorter than
        ``min_segment_frames``."""
        if not cut_points:
            return cut_points

        filtered = [cut_points[0]]
        for cp in cut_points[1:]:
            if cp - filtered[-1] >= self.min_segment_frames:
                filtered.append(cp)
        # Check last segment
        if n_frames - filtered[-1] < self.min_segment_frames and len(filtered) > 1:
            filtered.pop()
        return filtered

    def _split_long_segments(
        self,
        cut_points: list[int],
        speed: np.ndarray,
        n_frames: int,
    ) -> list[int]:
        """Split segments exceeding ``max_segment_frames`` at the
        deepest speed minimum within the segment."""
        boundaries = [0] + cut_points + [n_frames]
        new_cuts = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if i > 0:
                new_cuts.append(start)

            seg_len = end - start
            if seg_len <= self.max_segment_frames:
                continue

            # Find the deepest minimum in this segment to split
            mid = start + np.argmin(speed[start:end])
            if mid > start + self.min_segment_frames and end - mid > self.min_segment_frames:
                new_cuts.append(mid)

        return sorted(set(new_cuts))

    # ------------------------------------------------------------------
    # Segment one hand
    # ------------------------------------------------------------------
    def _segment_hand(
        self,
        hand_data: dict,
        hand_type: str,
    ) -> list[dict]:
        """Segment a single hand's trajectory into atomic actions."""
        states = hand_data.get("states")
        if not states or len(states) < self.min_segment_frames:
            return []

        states_arr = np.asarray(states, dtype=np.float64)
        positions = states_arr[:, 0:3]
        n_frames = len(states_arr)

        # 1. Compute and smooth speed
        speed = self._compute_speed(positions)
        smooth_speed = self._smooth_speed(speed, self.speed_smooth_window)

        # 2. Detect local minima
        minima = self._find_local_minima(smooth_speed, self.min_window)

        # 3. Merge short segments, split long ones
        cut_points = self._merge_short_segments(minima, n_frames)
        cut_points = self._split_long_segments(
            cut_points,
            smooth_speed,
            n_frames,
        )

        # 4. Build segment boundaries
        boundaries = [0] + cut_points + [n_frames]

        valid_fids = hand_data.get("valid_frame_ids", list(range(n_frames)))
        actions = hand_data.get("actions", [])
        joints_world = hand_data.get("joints_world", [])
        joints_cam = hand_data.get("joints_cam", [])

        segments = []
        for seg_idx in range(len(boundaries) - 1):
            s = boundaries[seg_idx]
            e = boundaries[seg_idx + 1]
            if e - s < 2:
                continue

            seg = {
                "hand_type": hand_type,
                "segment_id": seg_idx,
                "start_frame": valid_fids[s] if s < len(valid_fids) else s,
                "end_frame": (valid_fids[e - 1] if e - 1 < len(valid_fids) else e - 1),
                "states": states[s:e],
                "actions": actions[s:e] if actions else [],
                "valid_frame_ids": valid_fids[s:e],
            }
            if joints_world:
                seg["joints_world"] = joints_world[s:e]
            if joints_cam:
                seg["joints_cam"] = joints_cam[s:e]

            segments.append(seg)

        logger.debug(
            f"Segmented {hand_type} hand: {len(segments)} atomic actions "
            f"from {n_frames} frames, cut_points={cut_points}",
        )
        return segments

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def process_single(self, sample=None, rank=None):
        if Fields.meta not in sample:
            return sample

        meta = sample[Fields.meta]
        hand_action_list = meta.get(self.hand_action_field)
        if not hand_action_list:
            return sample

        # After reassembly, hand_action_list is [merged_result]
        # merged_result is a dict: {"right": {...}, "left": {...}}
        hand_types = ["right", "left"] if self.hand_type == "both" else [self.hand_type]

        all_segments = []
        for clip_result in hand_action_list:
            if not clip_result or not isinstance(clip_result, dict):
                continue
            for ht in hand_types:
                hand_data = clip_result.get(ht)
                if not hand_data or not hand_data.get("states"):
                    continue
                segs = self._segment_hand(hand_data, ht)
                all_segments.extend(segs)

        # Sort segments by start_frame for consistent ordering
        all_segments.sort(key=lambda s: (s["start_frame"], s["hand_type"]))

        meta[self.segment_field] = all_segments
        logger.info(
            f"Atomic action segmentation: {len(all_segments)} segments",
        )
        return sample
