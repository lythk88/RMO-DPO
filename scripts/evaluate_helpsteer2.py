#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from rmo_dpo.config import as_list, load_config
from rmo_dpo.data import DPODataCollator, build_objective_datasets, resolve_pair_dir
from rmo_dpo.losses import dpo_loss_from_logps, pair_log_probs
from rmo_dpo.models import dtype_from_string, load_tokenizer
from rmo_dpo.utils import first_parameter_device, move_batch_to_device, setup_logger


def _looks_like_hf_network_error(exc: Exception) -> bool:
    text = str(exc)
    markers = [
        "Temporary failure in name resolution",
        "Cannot send a request, as the client has been closed.",
        "custom_generate/generate.py",
    ]
    return any(marker in text for marker in markers)


def _load_eval_tokenizer(model_name: str, trust_remote_code: bool) -> Any:
    try:
        return load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    except Exception as exc:
        if not _looks_like_hf_network_error(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer


def _load_eval_base_model(model_name: str, dtype: torch.dtype | str, trust_remote_code: bool) -> Any:
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": trust_remote_code,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        if not _looks_like_hf_network_error(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=False,
            local_files_only=True,
        )


def load_policy_for_eval(config: dict[str, Any], checkpoint: str) -> tuple[Any, Any]:
    model_cfg = config["model"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = _load_eval_tokenizer(model_cfg["policy_name"], trust_remote_code=trust_remote_code)
    dtype = dtype_from_string(model_cfg.get("torch_dtype", "bfloat16"))
    base = _load_eval_base_model(model_cfg["policy_name"], dtype=dtype, trust_remote_code=trust_remote_code)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RMO-DPO adapter on HelpSteer2 pairs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Adapter checkpoint directory, e.g. outputs/.../final")
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on evaluated examples per objective. Applied before aggregation.",
    )
    parser.add_argument(
        "--per_objective_batch_size",
        type=int,
        default=None,
        help="Optional batch-size override for evaluation only.",
    )
    parser.add_argument("--output_json", default=None, help="Optional path to write metrics as JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("evaluate_helpsteer2")
    model, tokenizer = load_policy_for_eval(cfg, args.checkpoint)
    device = first_parameter_device(model)
    data_cfg = cfg["data"]
    rmo_cfg = cfg["rmo_dpo"]
    objectives = list(data_cfg["objectives"])
    beta = [float(x) for x in as_list(rmo_cfg.get("beta", 0.1), len(objectives), "beta")]
    split = args.split or data_cfg.get("eval_split", "validation")

    eval_pair_dir = resolve_pair_dir(data_cfg, eval_mode=True)
    datasets = build_objective_datasets(eval_pair_dir, split, objectives)
    collator = DPODataCollator(
        tokenizer,
        max_length=int(data_cfg.get("max_length", 2048)),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 1024)),
        max_response_length=int(data_cfg.get("max_response_length", 1024)),
        system_message=data_cfg.get("system_message"),
    )
    eval_batch_size = int(args.per_objective_batch_size or data_cfg.get("per_objective_batch_size", 1))
    metrics: dict[str, float] = {}
    for obj_idx, obj in enumerate(objectives):
        loader = DataLoader(
            datasets[obj],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=int(data_cfg.get("dataloader_num_workers", 0)),
        )
        losses = []
        accs = []
        margins = []
        seen_examples = 0
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            if args.max_examples is not None and seen_examples >= args.max_examples:
                break
            batch = move_batch_to_device(batch, device)
            policy_logps = pair_log_probs(model, batch)
            with model.disable_adapter():
                ref_logps = pair_log_probs(model, batch)
            dpo = dpo_loss_from_logps(
                policy_logps.a,
                policy_logps.b,
                ref_logps.a,
                ref_logps.b,
                labels=batch["preference_label"],
                beta=beta[obj_idx],
            )
            batch_losses = dpo.losses.detach().float().cpu()
            batch_accs = dpo.accuracy.detach().float().cpu()
            batch_margins = dpo.margins.detach().float().cpu()
            if args.max_examples is not None:
                remaining = args.max_examples - seen_examples
                batch_losses = batch_losses[:remaining]
                batch_accs = batch_accs[:remaining]
                batch_margins = batch_margins[:remaining]
            losses.append(batch_losses)
            accs.append(batch_accs)
            margins.append(batch_margins)
            seen_examples += int(batch_losses.numel())
        if losses:
            metrics[f"{obj}/loss"] = float(torch.cat(losses).mean())
            metrics[f"{obj}/accuracy"] = float(torch.cat(accs).mean())
            metrics[f"{obj}/margin"] = float(torch.cat(margins).mean())
    loss_values = [v for k, v in metrics.items() if k.endswith("/loss")]
    acc_values = [v for k, v in metrics.items() if k.endswith("/accuracy")]
    metrics["mean_loss"] = float(sum(loss_values) / len(loss_values)) if loss_values else float("nan")
    metrics["worst_loss"] = float(max(loss_values)) if loss_values else float("nan")
    metrics["mean_accuracy"] = float(sum(acc_values) / len(acc_values)) if acc_values else float("nan")
    metrics["worst_accuracy"] = float(min(acc_values)) if acc_values else float("nan")
    for key, value in metrics.items():
        logger.info("%s = %.6f", key, value)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "split": split,
            "eval_pair_dir": str(eval_pair_dir),
            "metrics": metrics,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info("Wrote metrics JSON to %s", output_path)


if __name__ == "__main__":
    main()
