"""Debug script to trace RMT forward pass on multi-segment inputs."""
import sys
sys.path.insert(0, "/workspace-SR004.nfs2/kurkin/long-vqa/train")

import torch
from transformers import AutoTokenizer
from modeling_rmt import RMTQwen3Config, RMTQwen3ForCausalLM

def debug_rmt_forward():
    model_path = "checkpoints/rmt_qwen_4b_lora_frozen_mem_fix_attn_mask"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = RMTQwen3Config.from_pretrained(model_path)
    
    print(f"Config: segment_size={config.segment_size}, num_mem_tokens={config.num_mem_tokens}")
    
    model = RMTQwen3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    
    # Create a test input that spans 2 segments
    segment_size = config.segment_size  # 256
    # Create input just over 1 segment to trigger multi-segment
    test_text = "Hello world " * 100  # Should be > 256 tokens
    
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda:0")
    attention_mask = inputs["attention_mask"].to("cuda:0")
    
    # Pad to segment_size multiple
    seq_len = input_ids.shape[1]
    pad_len = (segment_size - seq_len % segment_size) % segment_size
    if pad_len > 0:
        input_ids = torch.cat([input_ids, torch.full((1, pad_len), tokenizer.pad_token_id, device="cuda:0")], dim=1)
        attention_mask = torch.cat([attention_mask, torch.zeros((1, pad_len), dtype=torch.long, device="cuda:0")], dim=1)
    
    print(f"Input shape: {input_ids.shape} ({input_ids.shape[1] // segment_size} segments)")
    print(f"Attention mask: {attention_mask.sum().item()} real tokens, {(attention_mask == 0).sum().item()} padding")
    
    # Run segmentation to see what we get
    segmented = model.segment(input_ids=input_ids, attention_mask=attention_mask)
    print(f"\nSegmented into {len(segmented)} segments:")
    for i, seg in enumerate(segmented):
        seg_ids = seg.get("input_ids")
        seg_mask = seg.get("attention_mask")
        print(f"  Segment {i}: input_ids shape={seg_ids.shape}, mask sum={seg_mask.sum().item()}")
    
    # Run forward with debug prints
    print("\n--- Forward pass ---")
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    
    logits = outputs.logits
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: (1, {seq_len}, vocab_size)")
    
    # Check if shapes match
    if logits.shape[1] != seq_len:
        print(f"!!! SHAPE MISMATCH: got {logits.shape[1]}, expected {seq_len}")
        print(f"    With padding: padded_len={input_ids.shape[1]}")
    
    # Try a single-segment input for comparison
    print("\n--- Single segment comparison ---")
    short_text = "Hello world"
    short_inputs = tokenizer(short_text, return_tensors="pt")
    short_ids = short_inputs["input_ids"].to("cuda:0")
    short_mask = short_inputs["attention_mask"].to("cuda:0")
    
    print(f"Short input: {short_ids.shape[1]} tokens")
    with torch.inference_mode():
        short_outputs = model(input_ids=short_ids, attention_mask=short_mask, use_cache=False)
    print(f"Short output logits shape: {short_outputs.logits.shape}")
    
    # Decode predictions
    pred_tokens = torch.argmax(short_outputs.logits[0], dim=-1)
    pred_text = tokenizer.decode(pred_tokens)
    print(f"Short prediction (next tokens): {pred_text[:100]}...")

if __name__ == "__main__":
    debug_rmt_forward()
