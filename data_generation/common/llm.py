from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLMInference:
    """Thin vLLM wrapper for chat-style generation."""

    def __init__(
        self,
        model_id: str,
        token: str = "",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.95,
    ):
        self.model_id = model_id
        self.token = token
        self.max_model_len = max_model_len

        self.model = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        num_return_sequences: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> str:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            seed=seed,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        output = self.model.generate([prompt], sampling_params)

        return output[0].outputs[0].text
