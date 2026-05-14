import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

try:
    from datatrove.utils.dataset import DatatroveFolderDataset
except ImportError:
    DatatroveFolderDataset = None
from openrlhf.datasets import SFTDataset, DatatroveSFTDataset, PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import OriginalActor
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer


def identity_collate(batch):
    return batch


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = OriginalActor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    def load_sft_source(dataset_path, dataset_probs, dataset_split, max_samples, split_label):
        if os.path.isdir(dataset_path):
            from datasets import DatasetDict, load_from_disk

            hf_local_data = None
            hf_load_error = None
            try:
                hf_local_data = load_from_disk(dataset_path)
            except Exception as e:
                hf_load_error = e

            if hf_local_data is not None:
                if isinstance(hf_local_data, DatasetDict):
                    if dataset_split in hf_local_data:
                        data = hf_local_data[dataset_split]
                    elif "train" in hf_local_data:
                        strategy.print(
                            f"[Warning] {split_label} split '{dataset_split}' not found in local DatasetDict at "
                            f"{dataset_path}; falling back to 'train'."
                        )
                        data = hf_local_data["train"]
                    else:
                        raise ValueError(
                            f"Local DatasetDict at {dataset_path} does not contain split '{dataset_split}' "
                            f"or 'train'. Available splits: {list(hf_local_data.keys())}"
                        )
                else:
                    data = hf_local_data

                data = data.select(range(min(max_samples, len(data))))
                return "hf", data

            if DatatroveFolderDataset is None:
                raise RuntimeError(
                    f"{split_label} directory '{dataset_path}' is not a HuggingFace dataset loadable via "
                    f"load_from_disk, and datatrove is not installed. Original load_from_disk error: {hf_load_error}"
                )

            data = DatatroveFolderDataset(
                folder_path=dataset_path,
                seq_len=args.max_len,
                token_size=(2 if tokenizer.vocab_size < 65535 else 4),
                shuffle=True,
                seed=args.seed,
                return_positions=False,
            )
            return "datatrove", data

        data = blending_datasets(
            dataset_path,
            dataset_probs,
            strategy,
            args.seed,
            max_count=max_samples,
            dataset_split=dataset_split,
        )
        if max_samples and max_samples > 0:
            data = data.select(range(min(max_samples, len(data))))
        return "hf", data

    # prepare for data and dataset
    train_data_kind, train_data = load_sft_source(
        args.dataset,
        args.dataset_probs,
        args.dataset_split,
        args.max_samples,
        "dataset",
    )
    if train_data_kind == "datatrove":
        train_dataset = DatatroveSFTDataset(
            train_data,
            tokenizer,
            args.max_len,
            args.max_samples,
            strategy,
            pretrain_mode=args.pretrain_mode,
        )
    else:
        train_dataset = SFTDataset(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            pretrain_mode=args.pretrain_mode,
            input_template=args.input_template,
            multiturn=args.multiturn,
        )
    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )


    eval_dataloader = None
    eval_perplexity_dataloader = None
    humaneval_dataloader = None
    mbpp_dataloader = None
    multipl_dataloader = None
    _eval_max = getattr(args, "eval_max_samples", 1)
    if getattr(args, "eval_dataset", None) and _eval_max != 0:
        eval_data_kind, eval_data = load_sft_source(
            args.eval_dataset,
            None,
            args.eval_split,
            args.eval_max_samples,
            "eval_dataset",
        )
        if eval_data_kind == "datatrove":
            eval_perplexity_dataset = DatatroveSFTDataset(
                eval_data,
                tokenizer,
                args.max_len,
                args.eval_max_samples,
                strategy,
                pretrain_mode=args.pretrain_mode,
            )
        else:
            eval_perplexity_dataset = SFTDataset(
                eval_data,
                tokenizer,
                args.max_len,
                strategy,
                pretrain_mode=args.pretrain_mode,
                input_template=args.input_template,
                multiturn=args.multiturn,
            )
            eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, args.eval_batch_size, True, False)#1, True, False)

        eval_perplexity_dataloader = strategy.setup_dataloader(
            eval_perplexity_dataset,
            args.micro_train_batch_size,
            True,
            False,
            eval_perplexity_dataset.collate_fn,
        )

    def load_benchmark_source(dataset_path, dataset_split, max_samples, split_label, config_name=None):
        from datasets import DatasetDict, load_dataset, load_from_disk

        ext = os.path.splitext(dataset_path)[-1].lower()
        data = None

        if os.path.isdir(dataset_path):
            try:
                data = load_from_disk(dataset_path)
            except Exception:
                load_kwargs = {}
                if config_name:
                    load_kwargs["name"] = config_name
                data = load_dataset(dataset_path, **load_kwargs)
        elif ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
            file_type = ext.strip(".")
            if file_type == "jsonl":
                file_type = "json"
            data = load_dataset(file_type, data_files=dataset_path)
        else:
            load_kwargs = {}
            if config_name:
                load_kwargs["name"] = config_name
            try:
                data = load_dataset(dataset_path, split=dataset_split, **load_kwargs)
            except Exception:
                data = load_dataset(dataset_path, **load_kwargs)

        if isinstance(data, DatasetDict):
            if dataset_split in data:
                data = data[dataset_split]
            elif "train" in data:
                strategy.print(
                    f"[Warning] {split_label} split '{dataset_split}' not found; falling back to 'train'."
                )
                data = data["train"]
            else:
                data = data[next(iter(data.keys()))]

        if max_samples and max_samples > 0:
            data = data.select(range(min(max_samples, len(data))))
        return data

    _eval_down_max = getattr(args, "eval_down_max_samples", 1)
    if getattr(args, "humaneval_dataset", None) and _eval_down_max != 0:
        humaneval_data = load_benchmark_source(
            args.humaneval_dataset,
            args.humaneval_split,
            args.eval_down_max_samples,
            "humaneval_dataset",
            args.humaneval_config,
        )
        humaneval_data = humaneval_data.map(
            lambda row: {
                "prompt": row["prompt"],
                "label": row.get("canonical_solution", ""),
                "unit_test": row["test"],
                "entry_point": row.get("entry_point"),
            }
        )
        humaneval_dataloader = strategy.setup_dataloader(
            humaneval_data,
            args.eval_down_batch_size,
            False,
            False,
            collate_fn=identity_collate,
        )

    if getattr(args, "mbpp_dataset", None) and _eval_down_max != 0:
        mbpp_data = load_benchmark_source(
            args.mbpp_dataset,
            args.mbpp_split,
            args.eval_down_max_samples,
            "mbpp_dataset",
            args.mbpp_config,
        )

        def _normalize_mbpp_row(row):
            prompt = row.get("prompt") or row.get("text") or ""
            tests = row.get("test_list") or row.get("unit_tests") or []
            test_imports = row.get("test_imports") or []
            test_setup_code = row.get("test_setup_code") or ""
            helper_parts = []
            if isinstance(test_imports, list):
                helper_parts.extend(str(item) for item in test_imports if item)
            elif test_imports:
                helper_parts.append(str(test_imports))
            if test_setup_code:
                helper_parts.append(str(test_setup_code))
            helper_code = "\n".join(part.strip("\n") for part in helper_parts if part).strip()

            function_name = None
            function_signature = None
            for line in str(row.get("code", "")).splitlines():
                stripped = line.strip()
                if stripped.startswith("def "):
                    function_signature = stripped
                    function_name = stripped[4:].split("(", 1)[0].strip()
                    break

            return {
                "prompt": prompt,
                "unit_test": tests,
                "code_context": {
                    "helper_code": helper_code,
                    "function_name": function_name,
                    "function_signature": function_signature,
                },
            }

        mbpp_data = mbpp_data.map(_normalize_mbpp_row)
        mbpp_dataloader = strategy.setup_dataloader(
            mbpp_data,
            args.eval_down_batch_size,
            False,
            False,
            collate_fn=identity_collate,
        )

    if getattr(args, "multipl_dataset", None) and _eval_down_max != 0:
        multipl_data = load_benchmark_source(
            args.multipl_dataset,
            args.multipl_split,
            args.eval_down_max_samples,
            "multipl_dataset",
            args.multipl_config,
        )

        def _normalize_multipl_row(row):
            entry_point = None
            for line in str(row.get("prompt", "")).splitlines():
                stripped = line.strip()
                if stripped.startswith("def "):
                    entry_point = stripped[4:].split("(", 1)[0].strip()
                    break

            return {
                "prompt": row.get("prompt", ""),
                "label": "",
                "unit_test": row.get("tests", ""),
                "entry_point": entry_point,
            }

        multipl_data = multipl_data.map(_normalize_multipl_row)
        multipl_dataloader = strategy.setup_dataloader(
            multipl_data,
            args.eval_down_batch_size,
            False,
            False,
            collate_fn=identity_collate,
        )



    # scheduler
    
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    # print(f"LEN: {len(train_dataset)}; BS: {args.train_batch_size}; NUM UPDATE: {num_update_steps_per_epoch}",flush=True)
    # wqqw
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_perplexity_dataloader=eval_perplexity_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        humaneval_dataloader=humaneval_dataloader,
        mbpp_dataloader=mbpp_dataloader,
        multipl_dataloader=multipl_dataloader,
    )

    try:
        trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

        # save model checkpoint after fitting on only rank0
        strategy.save_model(model, tokenizer, args.save_path)
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--eval_down_batch_size", type=int, default=128, help="Global downstream eval batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # ring-attention
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # vLLM configuration
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size for vLLM engine")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7,
                       help="GPU memory utilization for vLLM engine")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate during evaluation")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for sampling during evaluation")
    parser.add_argument("--eval_n_samples_per_prompt", type=int, default=1,
                       help="Number of samples to generate per prompt for pass@k calculation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling parameter for generation")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum number of tokens to generate")
    
    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--eval_max_samples", type=int, default=1e8, help="Max number of eval samples")
    parser.add_argument("--eval_down_max_samples", type=int, default=1e8, help="Max prompts for downstream eval")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--multiturn", action="store_true", default=False, help="Use compacted multiturn dataset")
    parser.add_argument("--humaneval_dataset", type=str, default=None, help="HumanEval dataset name or path")
    parser.add_argument("--humaneval_config", type=str, default=None, help="Optional HumanEval config name")
    parser.add_argument("--humaneval_split", type=str, default="test", help="HumanEval split")
    parser.add_argument("--mbpp_dataset", type=str, default=None, help="MBPP dataset name or path")
    parser.add_argument("--mbpp_config", type=str, default="sanitized", help="Optional MBPP config name")
    parser.add_argument("--mbpp_split", type=str, default="test", help="MBPP split")
    parser.add_argument("--multipl_dataset", type=str, default=None, help="MultiPL-E dataset name or path")
    parser.add_argument("--multipl_config", type=str, default="humaneval-py", help="MultiPL-E config name")
    parser.add_argument("--multipl_split", type=str, default="test", help="MultiPL-E split")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_debug")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.multiturn:
        assert args.apply_chat_template, "apply_chat_template must be enabled when using multiturn format"

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)