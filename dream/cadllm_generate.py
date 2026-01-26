# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def add_gumbel_noise(logits, temperature):
    '''
    Apply Gumbel-max trick noise. Use float64 to avoid precision issues.
    '''
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def detect_repetition(generated_tokens: List[int], window_size: int = 8, min_repeat_length: int = 2) -> float:
    if len(generated_tokens) < window_size:
        return 0.0
    recent_tokens = generated_tokens[-window_size:]
    if len(set(recent_tokens[-4:])) == 1:
        return 1.0
    max_repetition = 0.0
    for repeat_len in range(min_repeat_length, window_size // 2 + 1):
        if len(recent_tokens) >= repeat_len * 2:
            pattern = recent_tokens[-repeat_len:]
            prev_pattern = recent_tokens[-repeat_len*2:-repeat_len]
            if pattern == prev_pattern:
                max_repetition = max(max_repetition, repeat_len / window_size)
    return max_repetition


def calculate_adaptive_vocab_size(
    generation_progress: float,
    confidence_history: List[float],
    generated_tokens: List[int],
    initial_vocab_size: int = 100,
    min_vocab_size: int = 35,
    max_vocab_size: int = 1000,
    adaptive_vocab_size: bool = True,
    repetition_detection: bool = True,
):
    if not adaptive_vocab_size:
        return int(max(min(initial_vocab_size, max_vocab_size), min_vocab_size))

    if generation_progress < 0.2:
        phase_vocab_size = max_vocab_size * 0.8
    elif generation_progress < 0.7:
        phase_vocab_size = initial_vocab_size * 0.7
    else:
        phase_vocab_size = initial_vocab_size * 0.9

    if confidence_history:
        import numpy as _np
        recent_confidence = _np.mean(confidence_history[-5:])
        confidence_trend = 0
        if len(confidence_history) >= 3:
            confidence_trend = confidence_history[-1] - confidence_history[-3]
        if recent_confidence < 0.3:
            confidence_factor = 1.5
        elif recent_confidence < 0.6:
            confidence_factor = 1.2
        elif recent_confidence > 0.8:
            confidence_factor = 0.8
        else:
            confidence_factor = 1.0
        if confidence_trend < -0.1:
            confidence_factor *= 1.3
        phase_vocab_size *= confidence_factor

    repetition_score = detect_repetition(generated_tokens) if repetition_detection else 0.0
    if repetition_score > 0.5:
        repetition_factor = 1.0 + (repetition_score * 2.0)
        phase_vocab_size *= repetition_factor

    import numpy as _np2
    return int(_np2.clip(phase_vocab_size, min_vocab_size, max_vocab_size))


def softmax_confidence_adaptive(logits, x0, vocab_size: int = 15):
    vocab_size_vals, vocab_size_idx = torch.topk(logits, k=vocab_size, dim=-1)
    p_vocab_size = F.softmax(vocab_size_vals.to(torch.float64), dim=-1)
    x0_expanded = x0.unsqueeze(-1)
    match = (vocab_size_idx == x0_expanded).float()
    x0_p = (p_vocab_size * match).sum(dim=-1)
    return x0_p


def entropy_confidence_adaptive(logits, temperature: float = 1.0, vocab_size: int = 15):
    scaled_logits = logits / temperature
    vocab_size_vals, _ = torch.topk(scaled_logits, k=vocab_size, dim=-1)
    probs = F.softmax(vocab_size_vals, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)
    confidence = 1.0 / (1.0 + entropy)
    return confidence


def calculate_adaptive_threshold(step: int, total_steps: int, initial_thresh: float = 0.85, min_thresh: float = 0.4):
    progress = step / max(total_steps, 1)
    threshold = initial_thresh * (1 - progress) + min_thresh * progress
    return max(threshold, min_thresh)


def compute_confidence_gap(logits):
    probs = F.softmax(logits, dim=-1)
    top2_probs, _ = torch.topk(probs, k=2, dim=-1)
    confidence_gap = top2_probs[..., 0] - top2_probs[..., 1]
    return confidence_gap.mean().item()


def prophet_early_exit(confidence_gap_history, confidence_history, min_gap=0.3, min_confidence=0.8, patience=2):
    if len(confidence_gap_history) < patience or len(confidence_history) < patience:
        return False
    recent_gaps = confidence_gap_history[-patience:]
    recent_confidences = confidence_history[-patience:]
    gap_ok = all(gap > min_gap for gap in recent_gaps)
    confidence_ok = all(conf > min_confidence for conf in recent_confidences)
    return gap_ok and confidence_ok

def get_transfer_index_threshold_based_adaptive(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
    confidence_method='softmax',
    logits_history=None,
    generation_progress: float = 0.0,
    confidence_history: Optional[List[float]] = None,
    generated_tokens: Optional[List[int]] = None,
    adaptive_vocab_size: bool = True,
    repetition_detection: bool = True,
):
    confidence_history = confidence_history or []
    generated_tokens = generated_tokens or []
    adaptive_vocab_size = calculate_adaptive_vocab_size(
        generation_progress,
        confidence_history,
        generated_tokens,
        initial_vocab_size=100,
        min_vocab_size=35,
        max_vocab_size=1000,
        adaptive_vocab_size=adaptive_vocab_size,
        repetition_detection=repetition_detection,
    )

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        if confidence_method == 'softmax':
            x0_p = softmax_confidence_adaptive(logits, x0, vocab_size=adaptive_vocab_size)
        elif confidence_method == 'entropy':
            x0_p = entropy_confidence_adaptive(logits, temperature=1.0, vocab_size=adaptive_vocab_size)
        else:
            raise ValueError(f"Unsupported confidence method: {confidence_method}")
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -float('inf'))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    # Null-guard: when num_transfer_tokens is None, fall back to transferring all currently masked tokens
    if num_transfer_tokens is None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        # Ensure k is a valid integer within bounds
        k_j = int(num_transfer_tokens[j])
        k_j = max(0, min(k_j, confidence.shape[1]))
        if k_j == 0:
            continue
        _, select_index = torch.topk(confidence[j], k=k_j)
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, k_j):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False

    return x0, transfer_index, adaptive_vocab_size

def get_transfer_index_factor_based_adaptive(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    factor=1,
    generation_progress: float = 0.0,
    confidence_history: Optional[List[float]] = None,
    generated_tokens: Optional[List[int]] = None,
    confidence_method: str = 'softmax',
    adaptive_vocab_size: bool = True,
    repetition_detection: bool = True,
):
    confidence_history = confidence_history or []
    generated_tokens = generated_tokens or []
    adaptive_vocab_size = calculate_adaptive_vocab_size(
        generation_progress,
        confidence_history,
        generated_tokens,
        initial_vocab_size=15,
        min_vocab_size=5,
        max_vocab_size=35,
        adaptive_vocab_size=adaptive_vocab_size,
        repetition_detection=repetition_detection,
    )

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        if confidence_method == 'softmax':
            x0_p = softmax_confidence_adaptive(logits, x0, vocab_size=adaptive_vocab_size)
        elif confidence_method == 'entropy':
            x0_p = entropy_confidence_adaptive(logits, temperature=1.0, vocab_size=adaptive_vocab_size)
        else:
            raise ValueError(f"Unsupported confidence method: {confidence_method}")
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -float('inf'))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, int(num_transfer_tokens[j]) + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]
        threshs[0] = -1
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        assert len(sorted_confidence) == len(threshs)
        top_i = 0
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break
        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1
        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index, adaptive_vocab_size

def calculate_adaptive_block_size(confidence_history, adaptive_blocks, initial_block_length, max_block, min_block, remaining_length):
    if not adaptive_blocks:
        return min(initial_block_length, remaining_length)
    if confidence_history:
        recent_conf_tensor = torch.tensor(confidence_history[-2:]).mean() if len(confidence_history) >= 2 else torch.tensor(confidence_history).mean()
        block_length = min_block + int((max_block - min_block) * recent_conf_tensor.item())
    else:
        block_length = initial_block_length
    block_length = max(min_block, min(block_length, max_block))
    return min(block_length, remaining_length)

def calculate_adaptive_step(confidence_history, adaptive_steps, initial_steps, max_steps, block_length, initial_block_length):
    if confidence_history and adaptive_steps:
        recent_conf_tensor = torch.tensor(confidence_history[-2:]).mean()
        avg_confidence = float(recent_conf_tensor.item())
        avg_confidence = max(0.0, min(1.0, avg_confidence))
        confidence_factor = 1.0 - avg_confidence
        steps_for_conf = initial_steps + int((max_steps - initial_steps) * confidence_factor)
        steps_for_conf = max(initial_steps, min(steps_for_conf, max_steps))
        block_steps = max(1, int(steps_for_conf * block_length / initial_block_length))
    else:
        block_steps = initial_steps
    return block_steps

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", True)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        threshold = kwargs.get("threshold", None)
        block_length = kwargs.get("block_length", 32)
        dual_cache = kwargs.get("dual_cache", False)
        # CAD-LLM extras
        initial_steps = kwargs.get("initial_steps", 128)
        max_steps = kwargs.get("max_steps", 256)
        initial_block_length = kwargs.get("initial_block_length", 24)
        adaptive_blocks = kwargs.get("adaptive_blocks", True)
        max_block = kwargs.get("max_block", 256)
        min_block = kwargs.get("min_block", 8)
        confidence_method = kwargs.get("confidence_method", "softmax")
        adaptive_vocab_size = kwargs.get("adaptive_vocab_size", True)
        repetition_detection = kwargs.get("repetition_detection", True)
        adaptive_steps = kwargs.get("adaptive_steps", True)
        adaptive_threshold = kwargs.get("adaptive_threshold", True)
        prophet_enabled = kwargs.get("prophet_enabled", False)
        min_confidence_gap = kwargs.get("min_confidence_gap", 0.3)
        prophet_patience = kwargs.get("prophet_patience", 2)
        min_confidence_commit = kwargs.get("min_confidence_commit", 0.6)
        factor = kwargs.get("factor", None)
        remasking = kwargs.get("remasking", 'low_confidence')

        # # CadLLM trace: confirm we're using the CadLLM diffusion path and its params
        # try:
        #     print(
        #         "[CADLLM] diffusion_generate (cadllm path): "
        #         f"dual_cache={dual_cache}, adaptive_blocks={adaptive_blocks}, "
        #         f"initial_block_length={initial_block_length}, initial_steps={initial_steps}, max_steps={max_steps}, "
        #         f"min_block={min_block}, max_block={max_block}, confidence_method={confidence_method}, remasking={remasking}"
        #     )
        # except Exception:
        #     pass

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            block_length=block_length,
            dual_cache=dual_cache,
            # CAD-LLM timing/shape params
            initial_steps=initial_steps,
            max_steps=max_steps,
            initial_block_length=initial_block_length,
            adaptive_blocks=adaptive_blocks,
            max_block=max_block,
            min_block=min_block,
            confidence_method=confidence_method,
            adaptive_vocab_size=adaptive_vocab_size,
            repetition_detection=repetition_detection,
            adaptive_steps=adaptive_steps,
            adaptive_threshold=adaptive_threshold,
            prophet_enabled=prophet_enabled,
            min_confidence_gap=min_confidence_gap,
            prophet_patience=prophet_patience,
            min_confidence_commit=min_confidence_commit,
            factor=factor,
            remasking=remasking,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = None,
        block_length: Optional[int] = 32,
        dual_cache: bool = True,
        # cadllm parameters
        initial_steps: int = 128,
        max_steps: int = 256,
        initial_block_length: int = 24,
        adaptive_blocks: bool = True,
        max_block: int = 256,
        min_block: int = 8,
        confidence_method: str = 'softmax',
        adaptive_vocab_size: bool = True,
        repetition_detection: bool = True,
        adaptive_steps: bool = True,
        adaptive_threshold: bool = True,
        prophet_enabled: bool = False,
        min_confidence_gap: float = 0.3,
        prophet_patience: int = 2,
        min_confidence_commit: float = 0.6,
        factor: Optional[float] = None,
        remasking: str = 'low_confidence',
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # Initialize generation config values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None
        nfe = 0

        # Prepare working sequence and masks
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        past_key_values = None

        remaining_length = gen_length
        current_pos = input_ids.shape[1]

        confidence_history: List[float] = []
        logits_history: List[torch.Tensor] = []
        generated_tokens: List[int] = []
        vocab_size_history: List[int] = []
        confidence_gap_history: List[float] = []
        # Debug histories
        # selected_count_history: List[int] = []
        # avg_conf_history: List[float] = []
        # remaining_masked_history: List[int] = []
        # threshold_history: List[float] = []

        # # CadLLM trace: sampling entry
        # try:
        #     print(
        #         "[CADLLM] _sample start: "
        #         f"gen_length={gen_length}, current_pos={current_pos}, "
        #         f"initial_block_length={initial_block_length}, adaptive_blocks={adaptive_blocks}, dual_cache={dual_cache}"
        #     )
        # except Exception:
        #     pass

        while remaining_length > 0:
            generation_progress = (gen_length - remaining_length) / gen_length

            # Adaptive block and steps
            block_length = calculate_adaptive_block_size(
                confidence_history,
                adaptive_blocks,
                initial_block_length,
                max_block,
                min_block,
                remaining_length,
            )

            block_steps = calculate_adaptive_step(
                confidence_history,
                adaptive_steps,
                initial_steps,
                max_steps,
                block_length,
                initial_block_length,
            )

            # # CadLLM trace: log first block configuration only
            # if remaining_length == gen_length:
            #     try:
            #         print(
            #             "[CADLLM] first block: "
            #             f"block_length={block_length}, block_steps={block_steps}"
            #         )
            #     except Exception:
            #         pass

            current_block_start = current_pos
            current_block_end = current_pos + block_length

            # Precompute transfer tokens per step for this block
            block_mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
            num_transfer_tokens_block = get_num_transfer_tokens(block_mask_index, block_steps)

            # Update cache and decode first token of the block
            model_output = self(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x[:, current_block_start] = x0[:, current_block_start]
            nfe += 1

            logits_history.append(model_output.logits.clone())
            if len(logits_history) > 5:
                logits_history.pop(0)

            # dual cache
            if dual_cache:
                # print(f"[CadLLM] using dual cache")
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1
            else:
                # print("[CadLLM] using prefix cache")
                # prefix cache: recompute prefix-only KV to avoid any misalignment
                prefix_output = self(
                    x[:, :current_block_start],
                    attention_mask,
                    tok_idx[:, :current_block_start] if tok_idx is not None else None,
                    use_cache=True,
                )
                past_key_values = prefix_output.past_key_values

            i = 1
            step_confidence_values: List[float] = []
            step_confidence_gap_values: List[float] = []
            while True:
                # Stop when block is filled
                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    break

                # For both dual and prefix cache, let tokens attend to the full (prompt + prefix + block)
                # so we don't accidentally hide left context in prefix mode.
                current_attention_mask = attention_mask

                x_block = x[:, current_block_start:current_block_end]
                mask_index = (x_block == mask_token_id)

                if dual_cache:
                    model_output = self(
                        x[:, current_block_start:current_block_end],
                        current_attention_mask,
                        tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                        dual_cache=True,
                        replace_position=replace_position,
                    )
                else:
                    model_output = self(
                        x[:, current_block_start:current_block_end],
                        current_attention_mask,
                        tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                nfe += 1

                # Adaptive threshold
                current_threshold = calculate_adaptive_threshold(i, block_steps) if (adaptive_threshold and threshold is None) else threshold

                # Step progress
                step_progress = generation_progress + (i / max(1, block_steps)) * (block_length / max(1, gen_length))

                # Select transfers
                if factor is None:
                    x0_adapt, transfer_index, used_vocab_size = get_transfer_index_threshold_based_adaptive(
                        logits,
                        temperature,
                        remasking,
                        mask_index,
                        x_block,
                        num_transfer_tokens_block[:, i] if (i < block_steps and threshold is None) else None,
                        current_threshold,
                        confidence_method,
                        logits_history,
                        step_progress,
                        confidence_history,
                        generated_tokens,
                        adaptive_vocab_size=adaptive_vocab_size,
                        repetition_detection=repetition_detection,
                    )
                else:
                    x0_adapt, transfer_index, used_vocab_size = get_transfer_index_factor_based_adaptive(
                        logits,
                        temperature,
                        remasking,
                        mask_index,
                        x_block,
                        None,
                        factor,
                        step_progress,
                        confidence_history,
                        generated_tokens,
                        confidence_method,
                        adaptive_vocab_size=adaptive_vocab_size,
                        repetition_detection=repetition_detection,
                    )

                vocab_size_history.append(used_vocab_size)

                # Confidence tracking
                if remasking == 'low_confidence':
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0_for_conf = torch.argmax(logits_with_noise, dim=-1)
                    if confidence_method == 'softmax':
                        step_confidence = softmax_confidence_adaptive(logits, x0_for_conf, vocab_size=used_vocab_size)
                    elif confidence_method == 'entropy':
                        step_confidence = entropy_confidence_adaptive(logits, temperature=1.0, vocab_size=used_vocab_size)
                    if mask_index.sum() > 0:
                        avg_step_conf = step_confidence[mask_index].mean().item()
                        step_confidence_values.append(avg_step_conf)
                        confidence_history.append(avg_step_conf)
                    # Debug: record selected count and avg confidence of this step
                    # try:
                    #     num_selected = int(transfer_index.sum().item())
                    #     num_masked_remaining = int(mask_index.sum().item())
                    #     avg_conf = float(step_confidence[mask_index].mean().item()) if num_masked_remaining > 0 else float("nan")
                    #     thr = float(current_threshold) if current_threshold is not None else float("nan")
                    #     selected_count_history.append(num_selected)
                    #     avg_conf_history.append(avg_conf)
                    #     remaining_masked_history.append(num_masked_remaining)
                    #     threshold_history.append(thr)
                    #     print(f"[CADLLM][block_step {i}/{block_steps}] selected={num_selected}, masked_remaining={num_masked_remaining}, avg_conf_masked={avg_conf:.4f}, threshold={thr:.4f}")
                    # except Exception:
                    #     pass

                # Prophet early-exit
                if prophet_enabled and i > 0.2 * block_steps:
                    current_step_gap = compute_confidence_gap(logits)
                    step_confidence_gap_values.append(current_step_gap)
                    should_commit = prophet_early_exit(
                        step_confidence_gap_values,
                        step_confidence_values,
                        min_gap=min_confidence_gap,
                        min_confidence=min_confidence_commit,
                        patience=prophet_patience,
                    )
                    if should_commit:
                        mask_index_current = (x[:, current_block_start:current_block_end] == mask_token_id)
                        logits_current = self(
                            x[:, current_block_start:current_block_end],
                            current_attention_mask,
                            tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None,
                            past_key_values=past_key_values,
                            use_cache=True,
                            dual_cache=dual_cache,
                            replace_position=replace_position if dual_cache else None,
                        ).logits
                        x0_current = torch.argmax(logits_current, dim=-1)
                        nfe += 1
                        x[:, current_block_start:current_block_end][mask_index_current] = x0_current[mask_index_current]
                        break

                # Track tokens for repetition
                if repetition_detection:
                    new_tokens = x0_adapt[transfer_index].detach().cpu().tolist()
                    generated_tokens.extend(new_tokens)
                    if len(generated_tokens) > 50:
                        generated_tokens = generated_tokens[-50:]

                # Apply transfer
                x[:, current_block_start:current_block_end][transfer_index] = x0_adapt[transfer_index]
                i += 1

            current_pos = current_block_end
            remaining_length -= block_length

        if return_dict_in_generate:
            metrics = {"nfe": nfe}
            # Attach debug histories if available
            # if len(selected_count_history) > 0:
            #     metrics["num_selected_per_step"] = selected_count_history
            #     metrics["avg_conf_per_step"] = avg_conf_history
            #     metrics["avg_conf_masked_per_step"] = avg_conf_history
            #     metrics["num_masked_remaining_per_step"] = remaining_masked_history
            #     metrics["threshold_per_step"] = threshold_history
            return DreamModelOutput(
                sequences=x,
                history=metrics,
            )
        else:
            return x