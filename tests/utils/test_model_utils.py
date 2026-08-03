import unittest
from functools import partial
from unittest.mock import patch, MagicMock
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np

from data_juicer.utils.model_utils import (
    check_model,
    get_backup_model_link,
    prepare_simple_aesthetics_model,
    prepare_api_model,
    prepare_huggingface_model,
    prepare_vllm_model,
    prepare_embedding_model,
    prepare_diffusion_model,
    prepare_fasttext_model,
    prepare_kenlm_model,
    prepare_nltk_model,
    prepare_sentencepiece_model,
    prepare_video_blip_model,
    prepare_fastsam_model,
    prepare_sdxl_prompt2prompt,
    prepare_deepcalib_model,
    prepare_opencv_classifier,
    prepare_model,
    get_model,
    free_models,
    prepare_recognizeAnything_model,
    update_sampling_params,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork


# ---------------------------------------------------------------------------
# Mock Server for API client error-path testing.
# Kept intentionally: real APIs cannot reliably reproduce broken JSON, 404,
# or connection errors.  This server is NOT for happy-path testing — happy
# paths should be covered by real API tests with @skip_if_from_fork.
# ---------------------------------------------------------------------------

class LocalAPIHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, format, *args):
        return

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.endswith("/models"):
            self._send_json({"data": [{"id": "server-model"}]})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        body_len = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(body_len) or b"{}")
        self.__class__.requests.append((path, body))

        if path.endswith("/broken-json"):
            payload = b"{"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path.endswith("/chat/completions"):
            user_text = body["messages"][0]["content"]
            self._send_json(
                {
                    "choices": [{"message": {"content": f"chat:{user_text}"}}],
                    "usage": {"total_tokens": 3},
                }
            )
            return

        if path.endswith("/embeddings"):
            input_value = body["input"]
            width = len(input_value) if isinstance(input_value, list) else len(str(input_value))
            self._send_json({"data": [{"embedding": [1.0, 2.0, float(width)]}]})
            return

        if path.endswith("/responses"):
            self._send_json({"output": [{"content": [{"text": f"response:{body['input']}"}]}]})
            return

        self._send_json({"error": "not found"}, status=404)


# other funcs are called by ops already
#
# ===================================================================
# Pure logic / mock-based tests — no network, no server, always runnable.
# ===================================================================
class ModelUtilsTest(DataJuicerTestCaseBase):

    def _start_local_api_server(self):
        LocalAPIHandler.requests = []
        server = HTTPServer(("127.0.0.1", 0), LocalAPIHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        self.addCleanup(self._stop_local_api_server, server, thread)
        return f"http://127.0.0.1:{server.server_port}"

    @staticmethod
    def _stop_local_api_server(server, thread):
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    @staticmethod
    def _local_client_params(base_url):
        return {"base_url": base_url, "api_key": "local-token"}  # noqa: S106

    def test_get_backup_model_link(self):
        test_data = [
            ('lid.176.bin', 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/'),  # exact match
            ('zh.sp.model', 'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/'),  # pattern match
            ('invalid_model_name', None),  # invalid model name
        ]
        for model_name, expected_link in test_data:
            self.assertEqual(get_backup_model_link(model_name), expected_link)

    @patch('data_juicer.utils.model_utils.aes_pred')
    @patch('data_juicer.utils.model_utils.transformers')
    def test_prepare_simple_aesthetics_model(self, mock_transformers, mock_aes_pred):
        # Test V1 model
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_aes_pred.AestheticsPredictorV1.from_pretrained.return_value = mock_model
        mock_transformers.CLIPProcessor.from_pretrained.return_value = mock_processor

        model = prepare_simple_aesthetics_model('v1_model')
        self.assertEqual(model[0], mock_model)
        self.assertEqual(model[1], mock_processor)
        mock_aes_pred.AestheticsPredictorV1.from_pretrained.assert_called_once()
        mock_transformers.CLIPProcessor.from_pretrained.assert_called_once()

        # Test V2 Linear model
        mock_aes_pred.reset_mock()
        mock_transformers.reset_mock()
        mock_aes_pred.AestheticsPredictorV2Linear.from_pretrained.return_value = mock_model
        model = prepare_simple_aesthetics_model('v2_linear_model')
        self.assertEqual(model[0], mock_model)
        self.assertEqual(model[1], mock_processor)
        mock_aes_pred.AestheticsPredictorV2Linear.from_pretrained.assert_called_once()
        mock_transformers.CLIPProcessor.from_pretrained.assert_called_once()

        # Test V2 ReLU model
        mock_aes_pred.reset_mock()
        mock_transformers.reset_mock()
        mock_aes_pred.AestheticsPredictorV2ReLU.from_pretrained.return_value = mock_model
        model = prepare_simple_aesthetics_model('v2_relu_model')
        self.assertEqual(model[0], mock_model)
        self.assertEqual(model[1], mock_processor)
        mock_aes_pred.AestheticsPredictorV2ReLU.from_pretrained.assert_called_once()
        mock_transformers.CLIPProcessor.from_pretrained.assert_called_once()

        # Test invalid model
        with self.assertRaises(ValueError):
            prepare_simple_aesthetics_model('invalid_model')

    @patch('data_juicer.utils.model_utils.openai')
    @patch('data_juicer.utils.model_utils.tiktoken')
    def test_prepare_api_model(self, mock_tiktoken, mock_openai):
        # Test basic API model with default endpoint
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_processor = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_processor

        # Pass explicit base_url to avoid environment variable (e.g. OPENAI_BASE_URL)
        # from CI being merged and triggering DashScope model remapping.
        model = prepare_api_model('test_model', base_url='https://api.openai.com/v1')
        self.assertEqual(model._client, mock_client)
        self.assertEqual(model.model, 'test_model')
        self.assertEqual(model.endpoint, '/chat/completions')

        # Test with processor for chat model
        mock_openai.OpenAI.reset_mock()
        mock_tiktoken.encoding_for_model.reset_mock()
        model, processor = prepare_api_model('test_model', base_url='https://api.openai.com/v1', return_processor=True)
        self.assertEqual(model._client, mock_client)
        self.assertEqual(processor, mock_processor)
        mock_tiktoken.encoding_for_model.assert_called_once()

        # Test explicit chat endpoint with different casing
        mock_openai.OpenAI.reset_mock()
        chat_model = prepare_api_model('test_model', endpoint='/v1/CHAT/completions')
        self.assertEqual(chat_model.endpoint, '/v1/CHAT/completions')
        self.assertEqual(chat_model.response_path, 'choices.0.message.content')
        
        # Test embedding endpoint with default response path
        embed_model = prepare_api_model('test_model', endpoint='/embeddings')
        self.assertEqual(embed_model.endpoint, '/embeddings')
        self.assertEqual(embed_model.response_path, 'data.0.embedding')
        
        # Test with processor for embedding model
        mock_tiktoken.encoding_for_model.reset_mock()
        embed_model, processor = prepare_api_model(
            'text_embedding_model',
            endpoint='/embeddings',
            return_processor=True
        )
        self.assertEqual(processor, mock_processor)
        mock_tiktoken.encoding_for_model.assert_called_with('text_embedding_model')

        # Test responses endpoint with default response path
        mock_openai.OpenAI.reset_mock()
        responses_model = prepare_api_model('test_model', endpoint='/responses')
        self.assertEqual(responses_model.endpoint, '/responses')
        self.assertEqual(responses_model.response_path, 'output.0.content.0.text')

        # Test responses endpoint with different casing
        mock_openai.OpenAI.reset_mock()
        responses_model = prepare_api_model('test_model', endpoint='/api/RESPONSES/v1')
        self.assertEqual(responses_model.endpoint, '/api/RESPONSES/v1')
        self.assertEqual(responses_model.response_path, 'output.0.content.0.text')

        # Test unsupported endpoint
        with self.assertRaises(ValueError) as context:
            prepare_api_model('test_model', endpoint='/unsupported/endpoint')
        self.assertIn('Unsupported endpoint', str(context.exception))

    @patch('data_juicer.utils.model_utils.transformers')
    def test_prepare_huggingface_model(self, mock_transformers):
        # Test model with processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model, processor = prepare_huggingface_model('test_model', return_model=True)
        self.assertEqual(model, mock_model)
        self.assertEqual(processor, mock_processor)

        # Test processor only
        processor = prepare_huggingface_model('test_model', return_model=False)
        self.assertEqual(processor, mock_processor)

    @patch('data_juicer.utils.model_utils.check_model_home', return_value='test_model')
    @patch('data_juicer.utils.model_utils.is_ray_mode', return_value=False)
    @patch('data_juicer.utils.model_utils.torch.cuda.device_count', return_value=0)
    def test_prepare_vllm_model(self, mock_cuda, mock_ray, mock_check):
        # Create a mock vllm module
        mock_vllm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_vllm.LLM.return_value = mock_model
        mock_model.get_tokenizer.return_value = mock_tokenizer

        # Replace the vllm module in model_utils and restore afterwards
        import data_juicer.utils.model_utils as model_utils
        original_vllm = model_utils.vllm
        model_utils.vllm = mock_vllm
        try:
            # Test basic functionality
            model, tokenizer = prepare_vllm_model('test_model')
            self.assertEqual(model, mock_model)
            self.assertEqual(tokenizer, mock_tokenizer)
            mock_vllm.LLM.assert_called_once_with(model='test_model', generation_config='auto')
            mock_model.get_tokenizer.assert_called_once()

            # Test environment setup
            self.assertEqual(os.environ['VLLM_WORKER_MULTIPROC_METHOD'], 'spawn')

            # Test device handling
            mock_vllm.LLM.reset_mock()
            model, _ = prepare_vllm_model('test_model', device='cuda:0')
            mock_vllm.LLM.assert_called_once_with(model='test_model', generation_config='auto')

            # Test model parameters
            mock_vllm.LLM.reset_mock()
            model_params = {'tensor_parallel_size': 2, 'max_model_len': 2048}
            model, _ = prepare_vllm_model('test_model', **model_params)
            mock_vllm.LLM.assert_called_once_with(model='test_model', generation_config='auto', **model_params)
        finally:
            model_utils.vllm = original_vllm
            os.environ.pop('VLLM_WORKER_MULTIPROC_METHOD', None)

    @patch('data_juicer.utils.model_utils.torch')
    @patch('data_juicer.utils.model_utils.transformers')
    def test_prepare_embedding_model(self, mock_transformers, mock_torch):
        # Test embedding model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model = prepare_embedding_model('test_model', device='cuda:0')
        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            'test_model', trust_remote_code=True)
        mock_transformers.AutoModel.from_pretrained.assert_called_once_with(
            'test_model', trust_remote_code=True)

        # Test the model is moved to the target device
        mock_model.to.assert_called_once_with('cuda:0')

    @patch('data_juicer.utils.model_utils.diffusers')
    def test_prepare_diffusion_model(self, mock_diffusers):
        mock_model = MagicMock()
        mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value = mock_model

        model = prepare_diffusion_model('test_model', 'text2image')
        self.assertEqual(model, mock_model)

        # Test invalid diffusion type
        with self.assertRaises(ValueError):
            prepare_diffusion_model('test_model', 'invalid_type')

    @patch('data_juicer.utils.model_utils.fasttext')
    def test_prepare_fasttext_model_mock(self, mock_fasttext):
        mock_model = MagicMock()
        mock_fasttext.load_model.return_value = mock_model

        model = prepare_fasttext_model('test_model')
        self.assertEqual(model, mock_model)

    # ===================================================================
    # Real model download tests — require network access.
    # Skipped in fork CI; run in main repo CI with model cache available.
    # ===================================================================
    @skip_if_from_fork("Skipping real model download test because running from a fork repo")
    def test_prepare_fasttext_model_real(self):
        """Test FastText model loading and prediction functionality with real model."""
        # Test with default language identification model
        model = prepare_fasttext_model()
        
        # Test basic prediction functionality
        test_texts = [
            "Hello, this is an English text.",
            "Bonjour, ceci est un texte français.",
            "你好，这是一段中文文本。"
        ]
        
        for text in test_texts:
            predictions = model.predict(text)
            # FastText predict returns a tuple of (labels, scores)
            self.assertIsInstance(predictions, tuple, "Predictions should be a tuple")
            self.assertEqual(len(predictions), 2, "Predictions should contain labels and scores")
            
            labels, scores = predictions
            self.assertIsInstance(labels, tuple, "Labels should be a tuple")
            self.assertIsInstance(scores, np.ndarray, "Scores should be a numpy array")
            self.assertEqual(len(labels), len(scores), "Number of labels should match number of scores")
            
            # Check first prediction
            self.assertTrue(labels[0].startswith('__label__'), 
                          "Label should start with __label__")
            self.assertIsInstance(scores[0], (float, np.floating), "Score should be a float")

    def test_prepare_fasttext_model_invalid(self):
        """Test FastText model with invalid model file."""
        with self.assertRaises(Exception):
            prepare_fasttext_model("invalid_model.bin")

    @skip_if_from_fork("Skipping real model download test because running from a fork repo")
    def test_prepare_fasttext_model_force_download(self):
        """Test FastText model with force download."""
        # First remove the model file if it exists
        from data_juicer.utils.cache_utils import DATA_JUICER_MODELS_CACHE
        from data_juicer.utils.model_utils import prepare_fasttext_model
        
        # Get the default model name from the function's default parameter
        default_model_name = prepare_fasttext_model.__defaults__[0]
        model_path = os.path.join(DATA_JUICER_MODELS_CACHE, default_model_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Test loading with force download
        model = prepare_fasttext_model(force=True)
        self.assertIsNotNone(model, "Model should be loaded after force download")
        
        # Test prediction after force download
        predictions = model.predict("This is a test.")
        self.assertGreater(len(predictions), 0, "Model should return predictions after force download")

    @patch('data_juicer.utils.model_utils.kenlm')
    def test_prepare_kenlm_model(self, mock_kenlm):
        mock_model = MagicMock()
        mock_kenlm.Model.return_value = mock_model

        model = prepare_kenlm_model('en')
        self.assertEqual(model, mock_model)

    @patch('data_juicer.utils.model_utils.nltk')
    def test_prepare_nltk_model(self, mock_nltk):
        mock_model = MagicMock()
        mock_nltk.data.load.return_value = mock_model

        model = prepare_nltk_model('en')
        self.assertEqual(model, mock_model)

        # Test invalid language
        with self.assertRaises(AssertionError):
            prepare_nltk_model('invalid_lang')

    @patch('data_juicer.utils.model_utils.sentencepiece')
    def test_prepare_sentencepiece_model(self, mock_sentencepiece):
        mock_model = MagicMock()
        mock_sentencepiece.SentencePieceProcessor.return_value = mock_model

        model = prepare_sentencepiece_model('test_model')
        self.assertEqual(model, mock_model)

    @patch('data_juicer.utils.model_utils.transformers')
    def test_prepare_video_blip_model(self, mock_transformers):
        # Set up mock classes and methods
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Set up the mock transformers module
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.Blip2ForConditionalGeneration = MagicMock()
        mock_transformers.Blip2ForConditionalGeneration.from_pretrained.return_value = mock_model
        mock_transformers.Blip2VisionModel = MagicMock()
        
        # Mock the custom VideoBlipForConditionalGeneration class
        class MockVideoBlipForConditionalGeneration:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        mock_transformers.Blip2ForConditionalGeneration = MockVideoBlipForConditionalGeneration

        model, processor = prepare_video_blip_model('test_model')
        self.assertEqual(model, mock_model)
        self.assertEqual(processor, mock_processor)

    @patch('data_juicer.utils.model_utils.ultralytics')
    def test_prepare_fastsam_model(self, mock_ultralytics):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Make .to() return the same mock
        mock_ultralytics.FastSAM.return_value = mock_model

        model = prepare_fastsam_model('test_model')
        self.assertEqual(model, mock_model)
        mock_model.to.assert_called_once()  # Verify .to() was called

    @patch('data_juicer.utils.model_utils.diffusers')
    def test_prepare_sdxl_prompt2prompt(self, mock_diffusers):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Make .to() return the same mock
        mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value = mock_model

        model = prepare_sdxl_prompt2prompt('test_model', mock_diffusers.AutoPipelineForText2Image)
        self.assertEqual(model, mock_model)
        mock_model.to.assert_called_once()  # Verify .to() was called

    def test_prepare_model(self):
        # Test valid model type returns a keyed partial bound to the right factory
        model_func = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        self.assertIsInstance(model_func, partial)
        self.assertIs(model_func.func, prepare_huggingface_model)
        self.assertEqual(model_func.keywords, {'pretrained_model_name_or_path': 'test_model'})

        model_func = prepare_model('embedding', model_path='test_embedding_model', device='cuda:0')
        self.assertIsInstance(model_func, partial)
        self.assertIs(model_func.func, prepare_embedding_model)
        self.assertEqual(model_func.keywords, {'model_path': 'test_embedding_model', 'device': 'cuda:0'})

        # Test invalid model type
        with self.assertRaises(AssertionError):
            prepare_model('invalid_type')

    @patch('data_juicer.utils.model_utils.transformers')
    def test_get_model(self, mock_transformers):
        # Test getting a model
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model_key = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        model = get_model(model_key)
        self.assertEqual(model, (mock_model, mock_processor))

        # Test getting a model with CUDA returns the same cached instance
        model = get_model(model_key, use_cuda=True)
        self.assertEqual(model, (mock_model, mock_processor))

    @patch('data_juicer.utils.model_utils.transformers')
    def test_free_models(self, mock_transformers):
        # Test freeing models
        from data_juicer.utils.model_utils import MODEL_ZOO
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model_key = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        get_model(model_key)
        self.assertIn(model_key, MODEL_ZOO)
        free_models()
        # Model zoo should be cleared after freeing
        self.assertEqual(len(MODEL_ZOO), 0)

    def test_filter_arguments_keeps_only_supported_parameters(self):
        from data_juicer.utils.model_utils import filter_arguments

        def limited(alpha, beta=1):
            return alpha + beta

        def accepts_kwargs(alpha, **kwargs):
            return alpha, kwargs

        source = {"alpha": 1, "beta": 2, "gamma": 3}
        self.assertEqual(filter_arguments(limited, source), {"alpha": 1, "beta": 2})
        self.assertEqual(filter_arguments(accepts_kwargs, source), source)

    def test_check_model_returns_existing_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "local-model.bin")
            with open(model_path, "wb") as f:
                f.write(b"model")

            self.assertEqual(check_model(model_path), model_path)

    def test_prepare_deepcalib_rejects_cpu_before_external_setup(self):
        with self.assertRaisesRegex(ValueError, "CUDA device must be specified"):
            prepare_deepcalib_model("weights.h5", device="cpu")

    def test_prepare_opencv_classifier_uses_local_cascade_path(self):
        from data_juicer.utils.model_utils import cv2

        cascade_path = os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml",
        )
        classifier = prepare_opencv_classifier(cascade_path)

        self.assertFalse(classifier.empty())

    def test_update_sampling_params_uses_defaults_and_preserves_existing(self):
        params = update_sampling_params(
            {},
            "plain-model-name",
            enable_vllm=False,
            fetch_generation_config_from_hf=False,
        )
        self.assertEqual(params["max_new_tokens"], 512)

        existing = update_sampling_params(
            {"max_new_tokens": 32},
            "plain-model-name",
            fetch_generation_config_from_hf=False,
        )
        self.assertEqual(existing["max_new_tokens"], 32)

        vllm_params = update_sampling_params(
            {},
            "plain-model-name",
            enable_vllm=True,
            fetch_generation_config_from_hf=False,
        )
        self.assertEqual(vllm_params["max_tokens"], 512)

    def test_update_sampling_params_reads_local_generation_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "generation_config.json"), "w") as f:
                f.write('{"max_new_tokens": 77}')

            params = update_sampling_params(
                {},
                tmp,
                enable_vllm=False,
                fetch_generation_config_from_hf=True,
            )

        self.assertEqual(params["max_new_tokens"], 77)

    def test_update_sampling_params_falls_back_when_local_config_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "generation_config.json"), "w") as f:
                f.write("{")

            params = update_sampling_params(
                {},
                tmp,
                enable_vllm=False,
                fetch_generation_config_from_hf=True,
            )

        self.assertEqual(params["max_new_tokens"], 512)

    # ===================================================================
    # Real API happy-path tests — require OPENAI_BASE_URL / OPENAI_API_KEY.
    # ===================================================================
    @skip_if_from_fork("Skipping API-based test because running from a fork repo")
    def test_prepare_api_model_chat_with_real_api(self):
        from data_juicer.utils.constant import DEFAULT_API_MODEL

        client = prepare_api_model(DEFAULT_API_MODEL)
        result = client(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # The model is asked to reply with exactly "OK"; verify the semantic
        # content rather than only that *some* string came back.
        self.assertIn("OK", result.upper())
        self.assertIsNotNone(client.last_response)

    @skip_if_from_fork("Skipping API-based test because running from a fork repo")
    def test_prepare_api_model_embedding_with_real_api(self):
        embedding_model = "text-embedding-v4"

        embeddings = prepare_api_model(
            embedding_model,
            endpoint="/embeddings",
        )
        result = embeddings(["hello world"])
        # The default response_path is "data.0.embedding", so a single
        # embedding vector (a flat list of floats) is returned.
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(v, (int, float)) for v in result))

    # ===================================================================
    # Mock Server error-path tests.
    # Kept intentionally: real APIs cannot reliably reproduce broken JSON,
    # 404, or connection errors.
    # ===================================================================

    def test_api_models_return_defaults_for_malformed_local_response(self):
        base_url = self._start_local_api_server()
        chat = prepare_api_model(
            "local-chat",
            endpoint="/chat/broken-json",
            **self._local_client_params(base_url),
        )
        self.assertEqual(chat([{"role": "user", "content": "hello"}]), "")
        self.assertIsNone(chat.last_response)

        embeddings = prepare_api_model(
            "local-embedding",
            endpoint="/embeddings/broken-json",
            **self._local_client_params(base_url),
        )
        self.assertEqual(embeddings("hello"), [])

        responses = prepare_api_model(
            "local-response",
            endpoint="/responses/broken-json",
            **self._local_client_params(base_url),
        )
        self.assertEqual(responses("hello"), "")

    def test_prepare_model_get_model_caching_with_mock_server(self):
        """Model caching test uses Mock Server because it verifies identity
        (same object returned), not API correctness."""
        base_url = self._start_local_api_server()
        free_models()
        try:
            model_key = prepare_model(
                "api",
                model="local-chat",
                endpoint="/chat/completions",
                **self._local_client_params(base_url),
            )

            first = get_model(model_key)
            second = get_model(model_key)

            self.assertIs(first, second)
            self.assertIsNone(get_model(None))
        finally:
            free_models()


# ===================================================================
# DashScope / OpenAI compatibility tests — pure logic, no network.
# ===================================================================
class DashScopeOpenAICompatTest(DataJuicerTestCaseBase):
    """Env merge + model remap for DashScope OpenAI-compatible REST."""

    def test_merge_env_from_openai_and_dashscope_aliases(self):
        from data_juicer.utils.model_utils import _merge_openai_compatible_env_into_model_params

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "DASHSCOPE_API_KEY": "ds-key",
                "OPENAI_API_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
            },
            clear=False,
        ):
            m = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(m.get("api_key"), "ds-key")
        self.assertTrue(m["base_url"].endswith("/v1"))

    def test_remap_gpt4o_on_dashscope_chat_only(self):
        from data_juicer.utils.model_utils import _maybe_remap_model_for_dashscope

        base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        overrides = {"DASHSCOPE_DEFAULT_MODEL": "", "OPENAI_DEFAULT_MODEL": ""}
        with patch.dict(os.environ, overrides, clear=False):
            self.assertEqual(
                _maybe_remap_model_for_dashscope("gpt-4o", base, "/chat/completions"),
                "qwen-plus",
            )
            self.assertEqual(
                _maybe_remap_model_for_dashscope("gpt-4o", base, "/embeddings"),
                "gpt-4o",
            )
            self.assertEqual(
                _maybe_remap_model_for_dashscope(
                    "qwen-turbo", base, "/chat/completions"
                ),
                "qwen-turbo",
            )


if __name__ == '__main__':
    unittest.main()
