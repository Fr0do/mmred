import sys
import os
import torch
from transformers import AutoConfig, AutoTokenizer

# Add train directory to path
sys.path.append("/workspace-SR004.nfs2/kurkin/long-vqa/train")

from modeling_rmt.language_modeling_qwen import RMTQwen3Config, RMTQwen3ForCausalLM

def test_rmt_position_ids():
    print("Testing RMT Position IDs...")
    
    # Mock Config
    segment_size = 10
    num_mem = 2
    vocab_size = 1000
    
    config = RMTQwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        segment_size=segment_size,
        num_mem_tokens=num_mem,
        max_position_embeddings=512,
        use_cache=False
    )
    
    model = RMTQwen3ForCausalLM(config)
    model.eval()
    
    # Verify Position IDs in segments
    # We can hook into the underlying Qwen model to see what position_ids it gets
    
    captured_pos_ids = []
    
    def forward_hook(module, args, kwargs):
        # args[0] might be input_ids, args[1] attention_mask, etc.
        # But we check kwargs first for position_ids
        pos_ids = kwargs.get('position_ids')
        if pos_ids is None and len(args) > 2:
             pos_ids = args[2] # standard signature often has position_ids at pos 2
        
        if pos_ids is not None:
            captured_pos_ids.append(pos_ids.detach().clone())
    
    # Hook the base Qwen model (which is called by RMT per segment)
    # The structure is model -> (RMT logic) -> super().forward() which is Qwen3ForCausalLM.forward
    # But Qwen3ForCausalLM.forward calls model.model.forward (QwenModel). 
    # Let's hook the first layer of the model to be sure
    
    model.model.layers[0].register_forward_pre_hook(forward_hook, with_kwargs=True)

    # Input longer than 1 segment (10)
    input_ids = torch.randint(0, vocab_size, (1, 25)) # 3 segments: 10, 10, 5
    
    print(f"Input length: {input_ids.shape[1]}")
    model(input_ids=input_ids)
    
    print(f"Captured {len(captured_pos_ids)} forward calls (one per segment expected).")
    
    for i, pos in enumerate(captured_pos_ids):
        print(f"Segment {i} Position IDs: {pos[0].tolist()}")
        if i > 0:
            # Check if it reset
            if pos[0, num_mem].item() == 0: 
                 # Index num_mem because 0..num_mem-1 might be memory tokens
                 # Wait, if position_ids reset, the first token of segment (after memory) will be small.
                 # Memory tokens are prepended. 
                 print(f"  -> WARNING: Position IDs seem to have reset! (First text token pos: {pos[0, num_mem].item()})")
            else:
                 print(f"  -> Position IDs look continuous (First text token pos: {pos[0, num_mem].item()})")

if __name__ == "__main__":
    test_rmt_position_ids()
