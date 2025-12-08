import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np

from data_juicer.utils.model_utils import (
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
    prepare_model,
    get_model,
    free_models,
    prepare_recognizeAnything_model,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

# other funcs are called by ops already
class ModelUtilsTest(DataJuicerTestCaseBase):

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

        model = prepare_api_model('test_model')
        self.assertEqual(model._client, mock_client)
        self.assertEqual(model.model, 'test_model')
        self.assertEqual(model.endpoint, '/chat/completions')

        # Test with processor for chat model
        mock_openai.OpenAI.reset_mock()
        mock_tiktoken.encoding_for_model.reset_mock()
        model, processor = prepare_api_model('test_model', return_processor=True)
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

    @patch('data_juicer.utils.model_utils.os')
    def test_prepare_vllm_model(self, mock_os):
        mock_os.path.join.return_value = 'test_model'

        # Create a mock vllm module
        mock_vllm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_vllm.LLM.return_value = mock_model
        mock_model.get_tokenizer.return_value = mock_tokenizer

        # Replace the vllm module in model_utils
        import data_juicer.utils.model_utils as model_utils
        model_utils.vllm = mock_vllm

        # Test basic functionality
        model, tokenizer = prepare_vllm_model('test_model')
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)
        mock_vllm.LLM.assert_called_once()
        mock_model.get_tokenizer.assert_called_once()

        # Test environment setup
        mock_os.environ.__setitem__.assert_called_with('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

        # Test device handling
        mock_vllm.LLM.reset_mock()
        model, _ = prepare_vllm_model('test_model', device='cuda:0')
        mock_vllm.LLM.assert_called_once_with(model='test_model', generation_config='auto')

        # Test model parameters
        mock_vllm.LLM.reset_mock()
        model_params = {'tensor_parallel_size': 2, 'max_model_len': 2048}
        model, _ = prepare_vllm_model('test_model', **model_params)
        mock_vllm.LLM.assert_called_once_with(model='test_model', generation_config='auto', **model_params)

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

        # Test the return function
        self.assertTrue(hasattr(model, 'encode'))
        self.assertTrue(callable(model.encode))

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
        # Test valid model type
        model_func = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        self.assertIsNotNone(model_func)

        model_func = prepare_model('embedding', model_path='test_embedding_model', device='cuda:0')
        self.assertIsNotNone(model_func)

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
        self.assertIsNotNone(model)

        # Test getting a model with CUDA
        model = get_model(model_key, use_cuda=True)
        self.assertIsNotNone(model)

    @patch('data_juicer.utils.model_utils.transformers')
    def test_free_models(self, mock_transformers):
        # Test freeing models
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model_key = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        get_model(model_key)
        free_models()
        # No assertion needed, just checking it doesn't raise an exception


if __name__ == '__main__':
    unittest.main()
