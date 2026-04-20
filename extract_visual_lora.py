import sys
from safetensors.torch import load_file, save_file

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_visual_lora.py <input_lora.safetensors> <output_lora.safetensors>")
        print("Example: python extract_visual_lora.py output/my_lora/checkpoint-final/lora_for_inference.safetensors output/my_lora/visual_only_lora.safetensors")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Loading {input_path}...")
    try:
        tensors = load_file(input_path)
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        sys.exit(1)
        
    visual_tensors = {}
    removed_count = 0
    kept_count = 0
    
    for key, tensor in tensors.items():
        # 只要层名字里不包含 'audio'，就保留（即纯视觉层 self_attn, cross_attn, ffn）
        if "audio" not in key.lower():
            visual_tensors[key] = tensor
            kept_count += 1
        else:
            removed_count += 1
            
    print(f"Original keys : {len(tensors)}")
    print(f"Removed keys (Audio) : {removed_count}")
    print(f"Kept keys (Visual)   : {kept_count}")
    
    if kept_count == 0:
        print("Warning: No visual layers found! Double check the input file.")
    else:
        print(f"Saving to {output_path}...")
        save_file(visual_tensors, output_path)
        print("Done! 🎉 把这个 Visual Only 的模型拿去推理即可！")

if __name__ == "__main__":
    main()
