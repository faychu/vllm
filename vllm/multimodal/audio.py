from functools import lru_cache

import torch
import numpy as np

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer

from .base import MultiModalInputs, MultiModalPlugin

logger = init_logger(__name__)

def get_processor(
    processor_name: str,
    *args,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Gets a processor for the given model name via HuggingFace.

    Derived from `vllm.transformers_utils.image_processor.get_image_processor`.
    """
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return processor

cached_get_processor = lru_cache(get_processor)
cached_get_tokenizer = lru_cache(get_tokenizer)


class AudioPlugin(MultiModalPlugin):
    """Plugin for audio data."""

    def get_data_key(self) -> str:
        return "audio"

    def _get_hf_audio_feature_extractor(self, model_config: ModelConfig):
        processor = cached_get_processor(model_config)
        return processor.feature_extractor

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
        model_config = ctx.model_config
        if isinstance(data, np.ndarray):
            audio_feature_extractor = self._get_hf_audio_feature_extractor(model_config)
            if audio_feature_extractor is None:
                raise RuntimeError("No HuggingFace audio_feature_extractor is available "
                                   "to process the audio object")
            try:
                batch_data = audio_feature_extractor(data, return_tensors="pt").data
            except Exception:
                logger.error("Failed to process audio (%s)", data)
                raise

            return MultiModalInputs(batch_data)
        elif isinstance(data, torch.Tensor):
            raise NotImplementedError("Embeddings input is not supported yet")

        raise TypeError(f"Invalid image type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000
