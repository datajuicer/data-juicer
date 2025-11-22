import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

# Lazy loading for OpenCV
cv2 = LazyLoader("cv2", globals(), "opencv-python")


@OPERATORS.register_module("image_subplot_filter")
@LOADED_IMAGES.register_module("image_subplot_filter")
class ImageSubplotFilter(Filter):
    """Filter to detect and remove samples with images containing subplots.

    This filter uses Hough Line Transform to detect straight lines in images,
    which is particularly effective for detecting grid-like subplot layouts
    with perfectly straight edges.

    The algorithm works by:
    1. Converting images to grayscale and applying edge detection
    2. Using Hough Line Transform to detect straight lines
    3. Classifying lines as horizontal or vertical based on angle
    4. Counting lines that meet length and angle requirements
    5. Calculating confidence based on line counts and distribution
    """

    _batched_op = True

    def __init__(
        self,
        min_horizontal_lines: int = 3,
        min_vertical_lines: int = 3,
        min_confidence: float = 0.5,
        any_or_all: str = "any",
        canny_threshold1: int = 70,
        canny_threshold2: int = 190,
        hough_threshold: int = 110,
        min_line_length: int = 110,
        max_line_gap: int = 18,
        angle_tolerance: float = 4.0,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_horizontal_lines: Minimum number of horizontal lines
            to consider an image as containing subplots.
        :param min_vertical_lines: Minimum number of vertical lines
            to consider an image as containing subplots.
        :param min_confidence: Minimum confidence score for filtering.
            Images with subplot confidence above this threshold will be
            considered as containing subplots.
        :param any_or_all: Strategy for multi-image samples. 'any' keeps
            the sample if any image meets the condition. 'all' keeps the
            sample only if all images meet the condition.
        :param canny_threshold1: First threshold for Canny edge detector.
        :param canny_threshold2: Second threshold for Canny edge detector.
        :param hough_threshold: Accumulator threshold for Hough transform.
        :param min_line_length: Minimum line length to be detected.
        :param max_line_gap: Maximum gap between line segments to be treated as a single line.
        :param angle_tolerance: Tolerance in degrees for classifying lines as horizontal/vertical.
        :param args: Extra args.
        :param kwargs: Extra args.
        """
        super().__init__(*args, **kwargs)

        # Validate inputs
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f"Can only be one of ['any', 'all'].")

        self.min_horizontal_lines = min_horizontal_lines
        self.min_vertical_lines = min_vertical_lines
        self.min_confidence = min_confidence
        self.any = any_or_all == "any"
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_tolerance = angle_tolerance

    def compute_stats_single(self, sample, context=False):
        """Detect subplots in images and compute statistics.

        :param sample: Single sample.
        :param context: Whether to store context information.
        :return: Sample with computed statistics.
        """
        # Check if already computed
        if StatsKeys.image_subplot_confidence in sample[Fields.stats]:
            return sample

        # Get image paths
        image_list = sample[self.image_key]
        if not image_list:
            # No images in this sample
            sample[Fields.stats][StatsKeys.image_subplot_confidence] = 0.0
            sample[Fields.stats][StatsKeys.horizontal_peak_count] = 0
            sample[Fields.stats][StatsKeys.vertical_peak_count] = 0
            sample[Fields.stats][StatsKeys.subplot_detected] = False
            return sample

        # Load images
        sample, images = load_data_with_context(
            sample, context, image_list, load_image, mm_bytes_key=self.image_bytes_key
        )

        # Compute subplot statistics for each image
        subplot_confidences = []
        horizontal_line_counts = []
        vertical_line_counts = []

        for image_path in image_list:
            if image_path not in images:
                # Image loading failed
                subplot_confidences.append(0.0)
                horizontal_line_counts.append(0)
                vertical_line_counts.append(0)
                continue

            try:
                confidence, h_count, v_count = self._detect_subplot(images[image_path])
                subplot_confidences.append(confidence)
                horizontal_line_counts.append(h_count)
                vertical_line_counts.append(v_count)
            except Exception:
                # Handle any processing errors gracefully
                subplot_confidences.append(0.0)
                horizontal_line_counts.append(0)
                vertical_line_counts.append(0)

        # Store statistics
        sample[Fields.stats][StatsKeys.image_subplot_confidence] = subplot_confidences
        sample[Fields.stats][StatsKeys.horizontal_peak_count] = horizontal_line_counts
        sample[Fields.stats][StatsKeys.vertical_peak_count] = vertical_line_counts

        # Determine if any image contains subplots
        max_confidence = max(subplot_confidences) if subplot_confidences else 0.0
        sample[Fields.stats][StatsKeys.subplot_detected] = max_confidence >= self.min_confidence

        return sample

    def process_single(self, sample):
        """Process single sample to determine if it should be filtered.

        :param sample: Single sample.
        :return: True for keeping, False for filtering.
        """
        subplot_confidences = sample[Fields.stats][StatsKeys.image_subplot_confidence]

        if not subplot_confidences:
            # No images, keep the sample
            return True

        # Apply threshold to determine which images contain subplots
        subplot_flags = [confidence >= self.min_confidence for confidence in subplot_confidences]

        # Check horizontal and vertical line requirements
        horizontal_lines = sample[Fields.stats][StatsKeys.horizontal_peak_count]
        vertical_lines = sample[Fields.stats][StatsKeys.vertical_peak_count]

        valid_subplot_flags = []
        for i, has_subplot in enumerate(subplot_flags):
            if not has_subplot:
                valid_subplot_flags.append(False)
                continue

            # Check if minimum line requirements are met
            h_lines_ok = horizontal_lines[i] >= self.min_horizontal_lines
            v_lines_ok = vertical_lines[i] >= self.min_vertical_lines

            valid_subplot_flags.append(h_lines_ok and v_lines_ok)

        if self.any:
            # Filter if any image contains valid subplots
            return not any(valid_subplot_flags)
        else:
            # Filter only if all images contain valid subplots
            return not all(valid_subplot_flags)

    def _detect_subplot(self, image):
        """Detect subplots in a single image using Hough Line Transform.

        :param image: PIL Image object.
        :return: Tuple of (confidence_score, horizontal_line_count, vertical_line_count).
        """
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return 0.0, 0, 0

        # Classify lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line angle
            if x2 - x1 == 0:  # Vertical line
                angle = 90.0
            else:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Classify based on angle
            if abs(angle) <= self.angle_tolerance or abs(angle - 180) <= self.angle_tolerance:
                # Horizontal line (0° or 180°)
                horizontal_lines.append(line[0])
            elif abs(angle - 90) <= self.angle_tolerance or abs(angle + 90) <= self.angle_tolerance:
                # Vertical line (90° or -90°)
                vertical_lines.append(line[0])

        # Calculate confidence score with enhanced algorithm
        confidence = self._calculate_confidence_enhanced(horizontal_lines, vertical_lines, width, height)

        return confidence, len(horizontal_lines), len(vertical_lines)

    def _calculate_confidence_enhanced(self, horizontal_lines, vertical_lines, width, height):
        """Enhanced confidence calculation considering line regularity and grid structure.

        :param horizontal_lines: List of horizontal line coordinates.
        :param vertical_lines: List of vertical line coordinates.
        :param width: Image width.
        :param height: Image height.
        :return: Confidence score between 0 and 1.
        """
        if not horizontal_lines or not vertical_lines:
            return 0.0

        # 1. Base score from number of lines
        h_score = min(len(horizontal_lines) / self.min_horizontal_lines, 1.0)
        v_score = min(len(vertical_lines) / self.min_vertical_lines, 1.0)

        # 2. Regularity score (line spacing consistency)
        h_regularity = self._calculate_line_regularity(horizontal_lines, axis=0, dimension=height)
        v_regularity = self._calculate_line_regularity(vertical_lines, axis=1, dimension=width)

        # 3. Grid structure score (intersection analysis)
        grid_score = self._calculate_grid_score(horizontal_lines, vertical_lines, width, height)

        # 4. Line length consistency score
        h_length_score = self._calculate_length_consistency(horizontal_lines, axis=0, dimension=width)
        v_length_score = self._calculate_length_consistency(vertical_lines, axis=1, dimension=height)

        # 5. Combined confidence with weighted scores
        confidence = (
            h_score * 0.2
            + v_score * 0.2
            + h_regularity * 0.15
            + v_regularity * 0.15
            + grid_score * 0.2
            + h_length_score * 0.05
            + v_length_score * 0.05
        )

        return min(confidence, 1.0)

    def _calculate_line_regularity(self, lines, axis, dimension):
        """Calculate regularity score based on line spacing consistency.

        :param lines: List of line coordinates.
        :param axis: 0 for horizontal lines (y-coordinate), 1 for vertical lines (x-coordinate).
        :param dimension: Image dimension (height for horizontal, width for vertical).
        :return: Regularity score between 0 and 1.
        """
        if len(lines) < 2:
            return 0.0

        # Extract line positions
        positions = []
        for line in lines:
            if axis == 0:  # Horizontal lines - use y-coordinate
                positions.append((line[1] + line[3]) / 2)  # Average y-coordinate
            else:  # Vertical lines - use x-coordinate
                positions.append((line[0] + line[2]) / 2)  # Average x-coordinate

        positions = sorted(positions)

        # Calculate spacings between consecutive lines
        spacings = np.diff(positions)

        if len(spacings) < 1:
            return 0.0

        # Calculate coefficient of variation (lower = more regular)
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)

        if mean_spacing == 0:
            return 0.0

        cv = std_spacing / mean_spacing

        # Convert CV to regularity score (lower CV = higher regularity)
        regularity = max(0, 1 - cv)

        # Penalize if spacings are too small (likely noise)
        min_spacing = dimension * 0.05  # At least 5% of dimension
        if mean_spacing < min_spacing:
            regularity *= 0.5

        return regularity

    def _calculate_grid_score(self, horizontal_lines, vertical_lines, width, height):
        """Calculate grid structure score based on line intersections.

        :param horizontal_lines: List of horizontal line coordinates.
        :param vertical_lines: List of vertical line coordinates.
        :param width: Image width.
        :param height: Image height.
        :return: Grid score between 0 and 1.
        """
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return 0.0

        # Extract line positions
        h_positions = sorted([(line[1] + line[3]) / 2 for line in horizontal_lines])
        v_positions = sorted([(line[0] + line[2]) / 2 for line in vertical_lines])

        # For subplots, we expect lines to form a grid
        # Simple heuristic: more lines should form more intersections
        intersection_density = (len(h_positions) * len(v_positions)) / (width * height) * 10000

        # Normalize to 0-1 range
        grid_score = min(intersection_density / 10.0, 1.0)

        return grid_score

    def _calculate_length_consistency(self, lines, axis, dimension):
        """Calculate line length consistency score.

        :param lines: List of line coordinates.
        :param axis: 0 for horizontal lines, 1 for vertical lines.
        :param dimension: Image dimension for normalization.
        :return: Length consistency score between 0 and 1.
        """
        if len(lines) < 2:
            return 0.0

        # Calculate line lengths
        lengths = []
        for line in lines:
            if axis == 0:  # Horizontal lines
                length = abs(line[2] - line[0])  # x2 - x1
            else:  # Vertical lines
                length = abs(line[3] - line[1])  # y2 - y1
            lengths.append(length)

        # Calculate coefficient of variation for lengths
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        if mean_length == 0:
            return 0.0

        cv = std_length / mean_length

        # Convert CV to consistency score (lower CV = higher consistency)
        consistency = max(0, 1 - cv)

        # Penalize if lines are too short
        min_length = dimension * 0.3  # At least 30% of dimension
        if mean_length < min_length:
            consistency *= 0.5

        return consistency
