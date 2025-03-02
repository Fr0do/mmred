from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig
from huggingface_hub import login
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
from unsloth import vLLMSamplingParams
from rewards import (
    atype_reward,
    xmlcount_reward,
    correctness_reward,
    soft_format_reward,
    strict_format_reward,
)


@dataclass
class DatasetArgs:
    dataset_name: str
    split: str = "train"
    system_prompt: str | None = (
        "You are a reasoning agent. Format your response with xml: \n\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
    )


@dataclass
class BaseModelConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_len: int = 2048
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.5


@dataclass
class PeftModelConfig:
    target_modules: list[str]
    lora_rank: int = 16
    lora_alpha: float = 16
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42
    use_dora: bool = False


def get_dataset(dataset_args: DatasetArgs) -> Dataset:
    data = load_dataset(dataset_args.dataset_name)[dataset_args.split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": dataset_args.system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }
    )  # type: ignore
    return data  # type: ignore


vllm_sampling_params = vLLMSamplingParams(
    min_p=0.05,
    seed=1337,
)


def main(
    base_model_args: BaseModelConfig,
    peft_model_args: PeftModelConfig,
    training_args: GRPOConfig,
    dataset_args: DatasetArgs,
):
    PatchFastRL("GRPO", FastLanguageModel)

    if training_args.hub_token:
        login(token=training_args.hub_token)

    dataset = get_dataset(dataset_args)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_args.model_name,
        max_seq_length=base_model_args.max_seq_len,
        load_in_4bit=base_model_args.load_in_4bit,
        fast_inference=base_model_args.fast_inference,
        max_lora_rank=peft_model_args.lora_rank,
        gpu_memory_utilization=base_model_args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_model_args.lora_rank,
        target_modules=peft_model_args.target_modules,
        lora_alpha=peft_model_args.lora_alpha,
        use_gradient_checkpointing=peft_model_args.use_gradient_checkpointing,
        random_state=peft_model_args.random_state,
        use_dora=peft_model_args.use_dora,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[
            atype_reward,
            xmlcount_reward,
            correctness_reward,
            soft_format_reward,
            strict_format_reward,
        ],
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((BaseModelConfig, PeftModelConfig, GRPOConfig, DatasetArgs))
    base_model_args, peft_model_args, training_args, dataset_args = (
        parser.parse_args_and_config()
    )

    main(base_model_args, peft_model_args, training_args, dataset_args)
