from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_from_string(name: str | None) -> torch.dtype | str:
    if name is None or str(name).lower() == "auto":
        return "auto"
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def can_reuse_peft_base_as_reference(
    *,
    use_lora: bool,
    policy_name: str,
    reference_name: str | None,
    model: torch.nn.Module,
) -> bool:
    resolved_reference = reference_name or policy_name
    return bool(use_lora) and resolved_reference == policy_name and hasattr(model, "disable_adapter")


def load_causal_lm_for_lora(
    model_name: str,
    *,
    use_lora: bool = True,
    load_in_4bit: bool = True,
    torch_dtype: str | None = "bfloat16",
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = True,
    lora_config: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Load a CausalLM and optionally attach LoRA adapters.

    For PEFT/LoRA, the base model is the reference policy. RMO-DPO can compute
    reference log-probs by disabling the adapter instead of loading a second copy.
    """
    dtype = dtype_from_string(torch_dtype)
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == "auto" else dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False

    if use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=gradient_checkpointing
            )
        elif gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        lora_config = lora_config or {}
        target_modules = lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        peft_config = LoraConfig(
            r=int(lora_config.get("r", 32)),
            lora_alpha=int(lora_config.get("alpha", 64)),
            lora_dropout=float(lora_config.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        if gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.print_trainable_parameters()
    elif gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model


def load_frozen_reference_model(
    model_name: str,
    *,
    load_in_4bit: bool = True,
    torch_dtype: str | None = "bfloat16",
    trust_remote_code: bool = True,
) -> torch.nn.Module:
    dtype = dtype_from_string(torch_dtype)
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == "auto" else dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def save_policy(model: torch.nn.Module, tokenizer: Any, output_dir: str) -> None:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_policy_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    adapter_name: str = "default",
) -> None:
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    checkpoint_dir = Path(checkpoint_dir)
    peft_state = load_peft_weights(str(checkpoint_dir), device=None)
    set_peft_model_state_dict(model, peft_state, adapter_name=adapter_name)
