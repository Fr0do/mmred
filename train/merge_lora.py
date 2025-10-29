from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftModelForCausalLM
import os
import argparse
import torch
import modeling_rmt

def save_model_and_tokenizer(model, tokenizer, output_dir):
    """Save the model and tokenizer to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir, dtype=model.dtype)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model and tokenizer saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA layers with the original model using merge_and_unload."
    )
    parser.add_argument(
        "--base_path", type=str, required=True, help="Path to the base layers."
    )
    parser.add_argument(
        "--config_path", type=str, default=None, required=False, help="Path to the config of base model."
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to the LoRA layers."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model.",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    if args.config_path:
        config = AutoConfig.from_pretrained(args.config_path)
        model = AutoModelForCausalLM.from_pretrained(args.base_path, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.lora_path, device_map="cpu", dtype=torch.bfloat16
        )
    peft_model = PeftModelForCausalLM.from_pretrained(
        model, args.lora_path, device_map="cpu", dtype=torch.bfloat16
    )
    model = peft_model.merge_and_unload(progressbar=True)
    model = model.to(dtype=torch.bfloat16)
    save_model_and_tokenizer(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
