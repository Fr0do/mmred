import torch
from typing import Union
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
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

from PIL import Image
import io

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
    subset: str = "default"


from PIL import Image
import io
def make_conversation_image(x):
    images_content = [{"type": "image", "image": Image.open(io.BytesIO(elem['bytes']))} for elem in x['images']]
    messages = [{"role": "system", "content": dataset_args.system_prompt},
                {
    "role": "user", "content": images_content + [{"type": "text", "text": dataset_args.task_prompt + "\n" + x["question"]}]
                },
    {"role": "assistant", "content": x['answer']}
               ]
    return messages

def get_dataset1(dataset_args: DatasetArgs, processor, use_sft: bool) -> Dataset:
    print (dataset_args.dataset_name, dataset_args.subset,dataset_args.split)
    data = load_dataset(dataset_args.dataset_name, "default")[dataset_args.split]
    data = [make_conversation_image(sample) for sample in data]
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
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_implementation,
        )

    processing_class = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    processing_class.pad_token = processing_class.eos_token

    print ("model, processor and tokenizer loaded")

    def collate_fn(examples):
        #print ("len(examples)", len(examples))
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ] 

        #print ("\n\n\n")
        #print ("texts", texts)
        #print ("\n\n\n")
        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract 
        #print ("len(image_inputs[0])", len(image_inputs[0]))
        batch = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=processor.tokenizer.model_max_length
        ) 
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
    
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID
    
        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels
    
        batch["labels"] = labels  # Add labels to the batch

        #print("Input IDs:", batch["input_ids"].shape)
        #print("Labels:", labels.shape)
        #print("Max label:", labels.max().item())
        #print("Image inputs:", [img.shape if hasattr(img, 'shape') else type(img) for img in image_inputs])
    
        return batch  # Return the prepared batch
    
    if not custom_args.use_sft:
        special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
        tokens_to_add = [
            token for token in special_tokens if token not in processing_class.get_vocab()
        ]
        if tokens_to_add:
            processing_class.add_tokens(tokens_to_add)
            print("Added special tokens:", tokens_to_add)
            processing_class.save_pretrained(training_args.output_dir)
        else:
            print("Special tokens are already present in the tokenizer.")

    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=32,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_alpha=32,
            # init_lora_weights="pissa_niter_16",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset = get_dataset1(dataset_args, processor, custom_args.use_sft)
    print ("use sft", custom_args.use_sft)
    if custom_args.use_sft:
        training_args.dataset_kwargs = {
        "skip_prepare_dataset": True}
        training_args.remove_unused_columns = False
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=collate_fn,
        )

    else:
        training_args.use_liger = custom_args.use_liger
        training_args.use_liger_kernel = custom_args.use_liger
        trainer = GRPOTrainer(
            model=model,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=[
                atype_reward,
                strict_format_reward,
                xmlcount_reward,
                correctness_reward,
            ],
            args=training_args,
        )
    print ("trainer created")
    trainer.train()
    trainer.evaluate()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    trainer.test()


if __name__ == "__main__":
    # Parse arguments dynamically based on use_sft flag
    parser = TrlParser((ModelConfig, DatasetArgs, CustomArgs))
    model_args, dataset_args, custom_args = parser.parse_args_and_config()
    print ("dataset args", dataset_args)
    print ("custom_args", custom_args)
    print ("model args", model_args)
    custom_args.use_sft = True
    if custom_args.use_sft:
        parser = TrlParser((SFTConfig,))
        training_args = parser.parse_args_and_config()[0]
    else:
        parser = TrlParser((GRPOConfig,))
        training_args = parser.parse_args_and_config()[0]

    training_args = SFTConfig(
    output_dir="checkpoints/mv1_sft_qwen_3b",  # Directory to save the model
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to=["none"],  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
    )
    training_args.report_to=None
    main(model_args, training_args, dataset_args, custom_args)
