import torch
from typing import Union
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import (
    GRPOConfig,
    GRPOTrainer,
    TrlParser,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
)
from rewards import (
    atype_reward,
    correctness_reward,
    format_reward,
)
from functools import partial

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ImportError:
    print("liger not installed, falling back to transformers kernels")


@dataclass
class CustomArgs:
    use_liger: bool = False
    use_gradient_checkpointing: str = "unsloth"
    use_sft: bool = False
    add_reasoning_tokens: bool = False
    r1_format: bool = False


@dataclass
class DatasetArgs:
    dataset_name: str
    split: str = "train"
    system_prompt: str | None = (
        "You are a reasoning agent. Format your response with xml:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
    )
    task_prompt: str | None = "Answer with a single word or number."
    subset: str = "default"
    subsample_train: float = 1.0


def get_dataset(dataset_args: DatasetArgs, use_sft: bool) -> Dataset:
    data = load_dataset(dataset_args.dataset_name, dataset_args.subset)[
        dataset_args.split
    ]
    if dataset_args.subsample_train < 1.0:
        data = data.class_encode_column("qtype")
        data = data.train_test_split(
            train_size=dataset_args.subsample_train, seed=42, stratify_by_column="qtype"
        )["train"]
    if use_sft:
        data = data.map(
            lambda x: {
                "messages": [
                    {"role": "system", "content": dataset_args.system_prompt},
                    {
                        "role": "user",
                        "content": dataset_args.task_prompt + "\n" + x["question"],
                    },
                    {"role": "assistant", "content": x["answer"]},
                ],
            }
        )
    else:
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": dataset_args.system_prompt},
                    {
                        "role": "user",
                        "content": dataset_args.task_prompt + "\n" + x["question"],
                    },
                ],
                "answer": x["answer"],
            }
        )
    return data


def main(
    model_args: ModelConfig,
    training_args: Union[SFTConfig, GRPOConfig],
    dataset_args: DatasetArgs,
    custom_args: CustomArgs,
):
    if training_args.hub_token:
        login(token=training_args.hub_token)
    if custom_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_implementation,
        )

    processing_class = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    processing_class.pad_token = processing_class.eos_token
    if custom_args.add_reasoning_tokens:
        special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
        tokens_to_add = [
            token
            for token in special_tokens
            if token not in processing_class.get_vocab()
        ]
        if tokens_to_add:
            processing_class.add_tokens(tokens_to_add)
            print("Added special tokens:", tokens_to_add)
            processing_class.save_pretrained(training_args.output_dir)
        else:
            print("Special tokens are already present in the tokenizer.")

    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
            lora_alpha=model_args.lora_alpha,
            # init_lora_weights="pissa_niter_16",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model.save_pretrained(training_args.output_dir)
    training_args.use_liger = custom_args.use_liger or True
    training_args.use_liger_kernel = custom_args.use_liger or True

    train_dataset = get_dataset(dataset_args, custom_args.use_sft)
    if custom_args.use_sft:
        eval_dataset = {}
        if training_args.do_eval:
            dataset_args.split = "val"
            val_dataset = get_dataset(dataset_args, custom_args.use_sft)
            eval_dataset = {"val": val_dataset}
            dataset_args.split = "test"
            for i in [32, 64]:
                dataset_args.subset = f"seq_len_{i}"
                eval_dataset |= {
                    f"test_{dataset_args.subset}": get_dataset(
                        dataset_args, custom_args.use_sft
                    )
                }
        trainer = SFTTrainer(
            model=model,
            processing_class=processing_class,
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template="<|im_start|>assistant", tokenizer=processing_class
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            args=training_args,
        )
    else:
        reward_funcs = [
            partial(atype_reward, r1_format=custom_args.r1_format),
            partial(format_reward, r1_format=custom_args.r1_format),
            partial(correctness_reward, r1_format=custom_args.r1_format),
        ]
        for reward_func in reward_funcs:
            if not hasattr(reward_func, "__name__"):
                reward_func.__name__ = reward_func.func.__name__
        trainer = GRPOTrainer(
            model=model,
            processing_class=processing_class,
            train_dataset=train_dataset,
            reward_funcs=reward_funcs,
            args=training_args,
        )
    trainer.train()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.evaluate()


if __name__ == "__main__":
    # Parse arguments dynamically based on use_sft flag
    parser = TrlParser((ModelConfig, DatasetArgs, CustomArgs))
    model_args, dataset_args, custom_args = parser.parse_args_and_config()
    if custom_args.use_sft:
        parser = TrlParser((SFTConfig,))
        training_args = parser.parse_args_and_config()[0]
    else:
        parser = TrlParser((GRPOConfig,))
        training_args = parser.parse_args_and_config()[0]
    main(model_args, training_args, dataset_args, custom_args)
