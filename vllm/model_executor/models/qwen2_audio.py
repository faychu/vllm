from typing import Tuple, Optional, List, Iterable, Union
from functools import lru_cache
import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen2AudioEncoder, Qwen2AudioConfig, Qwen2AudioEncoderConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalInputs
from vllm.sequence import SequenceData, SamplerOutput, IntermediateTensors
from vllm.logger import init_logger
from .interfaces import SupportsVision

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


class Qwen2AudioMultiModalProjector(nn.Module):
    def __init__(self, audio_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(audio_hidden_size, text_hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states

def dummy_data_for_qwen2_audio(ctx: InputContext, seq_len: int):
    max_llm_audio_tokens = get_max_qwen2_audio_audio_tokens(ctx)

    audio_token_index = ctx.model_config.hf_config.audio_token_index

    token_ids = [audio_token_index] * max_llm_audio_tokens
    token_ids += [0] * (seq_len - max_llm_audio_tokens)
    dummy_seqdata = SequenceData(token_ids)

    dummy_audio = np.full((max_llm_audio_tokens*2*2*160,), 0.)

    return dummy_seqdata, {"audio": [dummy_audio]}

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
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
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


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths

def get_max_qwen2_audio_audio_tokens(ctx: InputContext) -> int:
    max_source_position = ctx.model_config.hf_config.audio_config.max_source_positions
    output_lengths = (max_source_position - 2) // 2 + 1
    return output_lengths

def input_processor_for_qwen2_audio(ctx: InputContext, llm_inputs: LLMInputs) -> LLMInputs:
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        return llm_inputs

    audios = multi_modal_data['audio']
    processor = cached_get_processor(ctx.model_config.model)

    audio_inputs = processor.feature_extractor(
        audios, sampling_rate=16000, return_attention_mask=True, padding="max_length")
    if not audios:
        return llm_inputs

    feature_attention_mask = audio_inputs.pop("attention_mask")
    input_features = audio_inputs['input_features']

    audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
        feature_attention_mask.sum(-1)
    )

    audio_token_index = ctx.model_config.hf_config.audio_token_index

    input_ids = llm_inputs['prompt_token_ids']

    new_input_ids = []
    audio_num = input_ids.count(audio_token_index)
    assert len(input_features) == audio_num, \
        f'The text input contains {audio_num} audio tokens, but {len(input_features)} audios provided'
    start = 0
    for audio_idx in range(audio_num):
        end = input_ids.index(audio_token_index, start)
        new_input_ids.extend(input_ids[start:end])  # text part

        new_input_ids.extend([audio_token_index] * audio_output_lengths[audio_idx])
        start = end + 1
    new_input_ids.extend(input_ids[start:])

    return LLMInputs(
        prompt_token_ids=new_input_ids,
        prompt=llm_inputs['prompt'],
        multi_modal_data=multi_modal_data,
    )

def input_mapper_for_qwen2_audio(
    ctx: InputContext,
    multi_modal_data: Union[np.ndarray, List[np.ndarray]],
) -> MultiModalInputs:
    """Input mapper for Qwen2-Audio.

    Notes:
        This input mapper support multiple audios (passed by a List[np.ndarray]).
    """
    processor = cached_get_processor(ctx.model_config.model)
    audio_feature_extractor = processor.feature_extractor
    if audio_feature_extractor is None:
        raise RuntimeError("No HuggingFace audio_feature_extractor is available "
                           "to process the audio object")
    try:
        batch_data = audio_feature_extractor(multi_modal_data, sampling_rate=16000, return_attention_mask=True, padding="max_length",return_tensors="pt").data
        batch_data["feature_attention_mask"] = batch_data.pop(
            "attention_mask"
        )
    except Exception:
        logger.error("Failed to process audio (%s)", multi_modal_data)
        raise

    return MultiModalInputs(batch_data)

@MULTIMODAL_REGISTRY.register_audio_input_mapper(input_mapper_for_qwen2_audio)
@MULTIMODAL_REGISTRY.register_max_audio_tokens(get_max_qwen2_audio_audio_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_qwen2_audio)
@INPUT_REGISTRY.register_input_processor(input_processor_for_qwen2_audio)
class Qwen2AudioForConditionalGeneration(nn.Module, SupportsVision):
    def __init__(self,
                 config: Qwen2AudioConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        self.audio_tower = Qwen2AudioEncoder(config.audio_config)
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config.audio_config.d_model, config.text_config.hidden_size)

        self.quant_config = quant_config

        self.language_model = Qwen2Model(
            config.text_config, cache_config, quant_config
        )
        self.unpadded_vocab_size = config.text_config.vocab_size
        if config.text_config.tie_word_embeddings:
            self.lm_head = self.language_model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.text_config.vocab_size,
                                          config.text_config.hidden_size,
                                          quant_config=quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.text_config.vocab_size,
                                                logit_scale)
        self.sampler = Sampler()

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            **kwargs: object,
    ) -> SamplerOutput:
        input_features: torch.Tensor = kwargs.get('input_features', None)
        feature_attention_mask: torch.Tensor = kwargs.get('feature_attention_mask', None)

        seq_len = input_ids.size(-1)

        if input_features is not None and seq_len != 1 and input_features.size(0) > 0:
            # compute audio embeddings
            audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.multi_modal_projector(selected_audio_feature)
            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
                audio_output_lengths.device
            ) < audio_output_lengths.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)


            # compute llm embeddings
            inputs_embeds = self.language_model.embed_tokens(input_ids)

            # merge llm embeddings and audio features
            mask = (input_ids == self.config.audio_token_index)
            inputs_embeds[mask, :] = masked_audio_features

            input_ids = None
        else:
            inputs_embeds = None

        result = self.language_model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return result

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name or 'audio' in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)