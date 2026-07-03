# yapf: disable
import json
import re
from typing import Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import image_path_to_base64, image_byte_to_base64
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper

vllm = LazyLoader("vllm")

OP_NAME = 'video_action_captioning_mapper'

DEFAULT_SYSTEM_PROMPT = (
    'You are a multimodal expert specializing in video captioning '
    'for egocentric human-object interaction (HOI) clips.'
)

DEFAULT_USER_PROMPT_TEMPLATE = """I will send you a set of video frames. Your goal is to describe the specific {hand_type}-hand action shown in the provided video frames below. These frames are sampled from an egocentric video and contain a single atomic hand-object interaction. A projected 2D hand trajectory is overlaid \u2013 this path represents the 3D palm center over time, with color gradually transitioning from blue to green to red to indicate temporal progression. The {hand_type}-hand palm position is marked with a blue dot. Do not confuse it with the {opposite_hand_type} hand. Respect the temporal order of frames. Each one is labeled by number (e.g., "Frame 1", "Frame 2", etc.), indicating its place in the time sequence. Please analyze the action step by step. Consider the hand status in each frame, whether there is an interacted object in each frame, and the temporal order of the frames.

Generate a one-sentence description of the {hand_type}-hand action shown in the entire sequence. When describing the {hand_type}-hand action, please follow these rules:

Only describe {hand_type}-hand actions. Ignore the {opposite_hand_type} hand completely.
Write in imperative form (e.g., "Insert the key," not "The hand is inserting..."). Do not use personal pronouns.
Use specific, descriptive verbs. If the action clearly involves picking up or placing an object, prefer verbs like "pick" and "place" to highlight the action intent. Avoid vague or generic terms like "clean", "spray", or "fix".
Describe the interacted object only if:
(1) the {hand_type} hand clearly interacts with it,
(2) or, if not, the hand is purposefully moving toward it with clear intent.
(3) If neither applies, return "N/A".
Be careful not to misidentify objects or their colors due to the trajectory overlay.
Do not hallucinate: if no clear or meaningful hand action, or object is present, return: "N/A".
Do not guess the action based on context. For example, do not assume someone is brushing something just because there's a sink.
Return your answer in JSON format with two fields:
(1) "think": a brief, step-by-step reasoning process (no longer than 3-4 sentences) explaining how the {hand_type}-hand action was determined from the hand motion trajectory and visual content.
(2) "action": the final one-sentence description of the {hand_type}-hand action, following all the rules above.
Please prepare to receive the frames.
"""  # noqa: E501


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoActionCaptioningMapper(Mapper):
    """Generate per-segment hand action captions using a VLM.

    This operator iterates over atomic action segments produced by
    ``VideoAtomicActionSegmentMapper`` and, for each segment, sends its
    ``overlay_frames`` (trajectory-overlaid images from
    ``VideoTrajectoryOverlayMapper``) to a VLM to obtain a structured
    JSON caption (``{"think": "...", "action": "..."}``).

    Pipeline position: must run **after** both
    ``VideoAtomicActionSegmentMapper`` (stage 7) and
    ``VideoTrajectoryOverlayMapper`` (stage 8).

    Supports filtering by ``hand_type`` ('left', 'right', or 'both').
    The per-segment caption is stored inside each segment dict, and all
    non-N/A actions are joined into the sample's ``text`` field.
    """

    _accelerator = 'cuda'

    def __init__(
        self,
        api_or_hf_model: str = 'Qwen/Qwen2.5-VL-7B-Instruct',
        is_api_model: bool = False,
        *,
        hand_type: str = 'right',
        segment_field: str = 'atomic_action_segments',
        frame_field: str = MetaKeys.video_frames,
        tag_field_name: str = 'hand_action_caption',
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        model_params: Dict = {},
        sampling_params: Dict = {},
        try_num: PositiveInt = 3,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_or_hf_model: API model name or HuggingFace model name.
        :param is_api_model: Whether the model is an API model.
            If true, use OpenAI-compatible API; otherwise use vLLM.
        :param hand_type: Which hand to caption: 'left', 'right', or
            'both'.  Only segments matching the specified hand_type(s)
            are sent to the VLM.
        :param segment_field: Meta field storing atomic action segments
            (output of VideoAtomicActionSegmentMapper).  Each segment
            must contain an ``overlay_frames`` list (output of
            VideoTrajectoryOverlayMapper).
        :param frame_field: Fallback field for raw frame paths.  Only
            used when a segment has no ``overlay_frames``.
        :param tag_field_name: The meta field name to store the generated
            caption result (JSON with 'think' and 'action').
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the VLM. If None, uses the
            default egocentric HOI system prompt.
        :param user_prompt_template: User prompt template string. Supports
            {hand_type} and {opposite_hand_type} placeholders.
            If None, uses the default template.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Extra parameters passed to the model.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)
        self.is_api_model = is_api_model

        if hand_type not in ('left', 'right', 'both'):
            raise ValueError(
                f"hand_type must be 'left', 'right', or 'both', "
                f"got '{hand_type}'")
        self.hand_type = hand_type

        self.segment_field = segment_field
        self.frame_field = frame_field
        self.tag_field_name = tag_field_name
        self.try_num = try_num

        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = (
            user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
        )

        sampling_params = update_sampling_params(
            sampling_params, api_or_hf_model, not self.is_api_model)

        if self.is_api_model:
            self.sampling_params = sampling_params
            self.model_key = prepare_model(
                model_type='api',
                model=api_or_hf_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )
        else:
            self.num_proc = 1
            self.model_key = prepare_model(
                model_type='vllm',
                pretrained_model_name_or_path=api_or_hf_model,
                **model_params,
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)

    def _build_messages(self, frames, hand_type, opposite_hand_type):
        """Build the chat messages with frames embedded as images.

        Matches the paper prompt format:
            <prompt text>
            Frame 1: [image] Frame 2: [image] ...
            Please now analyze and generate the results.
        """
        user_text = self.user_prompt_template.format(
            hand_type=hand_type,
            opposite_hand_type=opposite_hand_type,
        )

        # Build multimodal content: prompt text + Frame N: [image] ...
        user_content = [{'type': 'text', 'text': user_text}]
        for i, frame in enumerate(frames):
            image_data = image_byte_to_base64(frame) if isinstance(frame, bytes) else image_path_to_base64(frame)
            user_content.append({
                'type': 'text',
                'text': f'Frame {i + 1}:',
            })
            user_content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,'
                           f'{image_data}',
                },
            })
        user_content.append({
            'type': 'text',
            'text': '\nPlease now analyze and generate the results.',
        })

        messages = []
        if self.system_prompt:
            messages.append({
                'role': 'system',
                'content': self.system_prompt,
            })
        messages.append({
            'role': 'user',
            'content': user_content,
        })
        return messages

    def _call_model(self, messages, rank=None):
        """Call the model and return raw text output."""
        if self.is_api_model:
            output = ''
            for attempt in range(self.try_num):
                try:
                    client = get_model(self.model_key, rank=rank)
                    output = client(messages, **self.sampling_params)
                    break
                except Exception as e:
                    logger.warning(
                        f'API call failed (attempt {attempt + 1}'
                        f'/{self.try_num}): {e}')
        else:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
            response = model.chat(messages, self.sampling_params)
            output = response[0].outputs[0].text
        return output

    @staticmethod
    def _parse_output(raw_output):
        """Parse the JSON output from the model.

        Handles cases where the model wraps JSON in ```json...``` fences
        and/or appends extra commentary after the JSON block.
        """
        text = raw_output.strip()

        # Try to extract JSON from markdown code fences first
        fence_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            # Try to extract the first {...} block
            brace_match = re.search(r'\{.*\}', text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0)

        try:
            result = json.loads(text, strict=False)
        except json.JSONDecodeError:
            try:
                result = json.loads(text.replace("'", '"'), strict=False)
            except Exception:
                logger.warning(
                    f'Failed to parse model output as JSON: {raw_output}')
                return {'think': '', 'action': ''}

        if not isinstance(result, dict):
            return {'think': '', 'action': str(result)}

        return {
            'think': result.get('think', ''),
            'action': result.get('action', ''),
        }

    def _caption_single_hand(self, frames, hand_type, rank=None):
        """Run captioning for a single hand and return parsed result."""
        opposite = 'left' if hand_type == 'right' else 'right'
        messages = self._build_messages(frames, hand_type, opposite)
        output = self._call_model(messages, rank=rank)
        return self._parse_output(output)

    def process_single(self, sample, rank=None, context=False):
        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        meta = sample[Fields.meta]
        segments = meta.get(self.segment_field)
        if not segments:
            return sample

        # Determine which hand types to caption
        target_hands = (
            {'left', 'right'}
            if self.hand_type == 'both'
            else {self.hand_type}
        )

        all_actions = []  # collect non-N/A actions for the text field

        for seg in segments:
            hand_type = seg.get('hand_type', 'right')
            if hand_type not in target_hands:
                continue

            # Skip if already captioned
            if 'caption' in seg:
                action = seg['caption'].get('action', '')
                if action and action != 'N/A':
                    all_actions.append(
                        f"{hand_type} hand seg{seg.get('segment_id', '?')}: "
                        f"{action}")
                continue

            # Use overlay_frames (trajectory-overlaid images) if available,
            # otherwise fall back to raw frames via sampled_frame_indices.
            frames = seg.get('overlay_frames', [])
            if not frames:
                frames = self._fallback_frames(sample, seg)

            if not frames:
                seg['caption'] = {'think': '', 'action': 'N/A'}
                continue

            result = self._caption_single_hand(frames, hand_type, rank=rank)
            seg['caption'] = result

            action = result.get('action', '')
            if action and action != 'N/A':
                all_actions.append(
                    f"{hand_type} hand seg{seg.get('segment_id', '?')}: "
                    f"{action}")

        # Store the updated segments back
        meta[self.segment_field] = segments

        # Join all non-N/A actions into the text field
        if all_actions:
            sample[self.text_key] = '; '.join(all_actions)

        return sample

    def _fallback_frames(self, sample, seg):
        """Extract raw frames for a segment when overlay_frames is empty."""
        frame_data = sample.get(self.frame_field, [])
        if not frame_data:
            return []

        # Unwrap nested list from reassembly: [[frames]] → [frames]
        if (isinstance(frame_data, list) and frame_data
                and isinstance(frame_data[0], list)):
            all_frames = frame_data[0]
        else:
            all_frames = frame_data

        # Use sampled_frame_indices if available, else evenly sample
        indices = seg.get('sampled_frame_indices', [])
        if not indices:
            start = seg.get('start_frame', 0)
            end = seg.get('end_frame', len(all_frames) - 1)
            n = min(8, end - start + 1)
            if n <= 0:
                return []
            import numpy as np
            indices = np.linspace(start, end, n, dtype=int).tolist()

        frames = []
        for idx in indices:
            if 0 <= idx < len(all_frames) and all_frames[idx]:
                frames.append(all_frames[idx])
        return frames
