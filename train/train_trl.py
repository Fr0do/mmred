import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# try:
#     from unsloth import PatchFastRL
#     PatchFastRL("GRPO", None)
#     from unsloth import vLLMSamplingParams as SamplingParams
# vllm_sampling_params = SamplingParams(
#     min_p=0.05,
#     seed=1337,
#    )
# except ImportError:
#     print("unsloth not in_stalled, falling back to trl GRPO")
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig


@dataclass
class CustomArgs:
    use_liger: bool = False
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class DatasetArgs:
    dataset_name: str
    split: str = "train"
    system_prompt: str | None = (
        "You are a reasoning agent. Format your response with xml: \n\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
    )
    task_prompt: str | None = "Answer with a single word or number."


def get_dataset(dataset_args: DatasetArgs) -> Dataset:
    data = load_dataset(dataset_args.dataset_name)[dataset_args.split]
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
    training_args: GRPOConfig,
    dataset_args: DatasetArgs,
    custom_args: CustomArgs,
):
    if training_args.hub_token:
        login(token=training_args.hub_token)

    dataset = get_dataset(dataset_args)

    if custom_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            # These args will get passed to the appropriate apply_liger_kernel_to_* function
            # to override the default settings
            # cross_entropy=True,
            # fused_linear_cross_entropy=False,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
    tokens_to_add = [
        token for token in special_tokens if token not in tokenizer.get_vocab()
    ]
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
        print("Added thinking tokens:", tokens_to_add)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        print("Thinking tokens are already present in the tokenizer.")

    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
            lora_alpha=model_args.lora_alpha,
            init_lora_weights="pissa_niter_16",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
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
    parser = TrlParser((ModelConfig, GRPOConfig, DatasetArgs, CustomArgs))
    model_args, training_args, dataset_args, custom_args = (
        parser.parse_args_and_config()
    )
    main(model_args, training_args, dataset_args, custom_args)
