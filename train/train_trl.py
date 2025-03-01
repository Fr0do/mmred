import torch
from typing import Union
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig, SFTConfig, SFTTrainer
from rewards import (
    atype_reward,
    correctness_reward,
    soft_format_reward,
    strict_format_reward,
    len_reward,
    xmlcount_reward,
    cosine_length_correctness_reward,
)

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ImportError:
    print("liger not installed, falling back to transformers kernels")


@dataclass
class CustomArgs:
    use_liger: bool = False
    use_gradient_checkpointing: str = "unsloth"
    use_sft: bool = False


@dataclass
class DatasetArgs:
    dataset_name: str
    split: str = "train"
    system_prompt: str | None = (
        "You are a reasoning agent. Format your response with xml:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
    )
    task_prompt: str | None = "Answer with a single word or number."


def get_dataset(dataset_args: DatasetArgs, use_sft: bool) -> Dataset:
    data = load_dataset(dataset_args.dataset_name)[dataset_args.split]
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
            # fused_linear_cross_entropy=custom_args.use_sft,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_implementation,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    if not custom_args.use_sft:
        special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
        tokens_to_add = [
            token for token in special_tokens if token not in tokenizer.get_vocab()
        ]
        if tokens_to_add:
            tokenizer.add_tokens(tokens_to_add)
            print("Added special tokens:", tokens_to_add)
            tokenizer.save_pretrained(training_args.output_dir)
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

    train_dataset = get_dataset(dataset_args, custom_args.use_sft)
    if custom_args.use_sft:
        dataset_args.split = "test"
        training_args.use_liger = custom_args.use_liger
        eval_dataset = get_dataset(dataset_args, custom_args.use_sft)
        print("Starting SFT Training...")
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
    else:
        print("Starting GRPO Training...")
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=[
                atype_reward,
                strict_format_reward,
                xmlcount_reward,
                correctness_reward,
            ],
            args=training_args,
        )

    trainer.train()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)


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
