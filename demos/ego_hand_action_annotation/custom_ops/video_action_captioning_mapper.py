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

DEFAULT_USER_PROMPT_TEMPLATE = """Below are video frames sampled from an egocentric video containing a single atomic hand-object interaction. Describe the specific {hand_type}-hand action shown in these frames.

The {hand_type}-hand palm position is marked with a blue dot. Do not confuse it with the {opposite_hand_type} hand. Respect the temporal order of frames (Frame 1 is earliest, last frame is latest). Consider the hand status in each frame, whether there is an interacted object, and the temporal progression.

Rules for describing the {hand_type}-hand action:
- Only describe {hand_type}-hand actions. Ignore the {opposite_hand_type} hand completely.
- Write in imperative form (e.g., "Insert the key," not "The hand is inserting..."). Do not use personal pronouns.
- Use specific, descriptive verbs. Prefer verbs like "pick" and "place" when applicable. Avoid vague terms like "clean", "spray", or "fix".
- Describe the interacted object only if: (1) the {hand_type} hand clearly interacts with it, or (2) the hand is purposefully moving toward it with clear intent. Otherwise, return "N/A" as the action.
- Do not hallucinate: if no clear hand action or object is present, return "N/A" as the action.
- Do not guess the action based on context.

Return your answer in JSON format:
{{"think": "<brief 3-4 sentence reasoning>", "action": "<one-sentence action description or N/A>"}}

Here are the frames:
"""  # noqa: E501


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoActionCaptioningMapper(Mapper):
    """Generates hand action captions from pre-extracted video frames
    using a VLM model (via API or vLLM).

    This operator reads frames from a specified field (e.g., video_frames),
    sends them along with a configurable prompt to a VLM, and stores the
    structured JSON response (think + action) in a meta field.
    The action description is also written to the text field.

    Supports annotating 'left', 'right', or 'both' hands. When hand_type
    is 'both', the operator runs two separate VLM calls (one per hand)
    and joins the action descriptions with '; ' in the text field.
    """

    _accelerator = 'cuda'

    def __init__(
        self,
        api_or_hf_model: str = 'Qwen/Qwen2.5-VL-7B-Instruct',
        is_api_model: bool = False,
        *,
        hand_type: str = 'right',
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
        :param hand_type: Which hand to describe: 'left', 'right', or
            'both'. When 'both', two separate calls are made and actions
            are joined with '; ' in the text field.
        :param frame_field: The field name where pre-extracted frames
            are stored. Each element is a list of frame paths (one list
            per video).
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
        """Build the chat messages with frames embedded as images."""
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
            'text': '\nAnalyze the frames above and return the JSON result.',
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
        # check if it's generated already
        if self.tag_field_name in sample.get(Fields.meta, {}):
            return sample

        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        # get frames from the frame_field
        frame_data = sample.get(self.frame_field, [])
        if not frame_data:
            sample[Fields.meta][self.tag_field_name] = {
                'think': '', 'action': 'N/A'}
            return sample

        # frame_data is a list of lists (one per video), flatten if needed
        if isinstance(frame_data[0], list):
            frames = frame_data[0]
        else:
            frames = frame_data

        if not frames:
            sample[Fields.meta][self.tag_field_name] = {
                'think': '', 'action': 'N/A'}
            return sample

        if self.hand_type == 'both':
            right_result = self._caption_single_hand(
                frames, 'right', rank=rank)
            left_result = self._caption_single_hand(
                frames, 'left', rank=rank)

            sample[Fields.meta][self.tag_field_name] = {
                'right': right_result,
                'left': left_result,
            }

            # join non-N/A actions into text
            actions = []
            for side, result in [('right', right_result),
                                 ('left', left_result)]:
                action = result.get('action', '')
                if action and action != 'N/A':
                    actions.append(f'{side} hand: {action}')
            if actions:
                sample[self.text_key] = '; '.join(actions)
        else:
            result = self._caption_single_hand(
                frames, self.hand_type, rank=rank)
            sample[Fields.meta][self.tag_field_name] = result
            action = result.get('action', '')
            if action and action != 'N/A':
                sample[self.text_key] = action

        return sample
