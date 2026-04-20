import sys
from safetensors.torch import load_file, save_file

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_audio_lora.py <input_lora.safetensors> <output_lora.safetensors>")
        print("Example: python extract_audio_lora.py output/my_lora/checkpoint-final/lora_for_inference.safetensors output/my_lora/audio_only_lora.safetensors")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Loading {input_path}...")
    try:
        tensors = load_file(input_path)
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        sys.exit(1)
        
    audio_tensors = {}
    removed_count = 0
    kept_count = 0
    
    for key, tensor in tensors.items():
        # 只要层名字里包含 'audio' (比如 audio_cross_attn, audio_proj)，就保留
        # 否则统统过滤掉（比如 self_attn, cross_attn, ffn）
        # 注意: audio_cross_attn 包含了 cross_attn，所以查 'audio' 是最安全的。
        if "audio" in key.lower():
            audio_tensors[key] = tensor
            kept_count += 1
        else:
            removed_count += 1
            
    print(f"Original keys : {len(tensors)}")
    print(f"Removed keys (Visual): {removed_count}")
    print(f"Kept keys (Audio)   : {kept_count}")
    
    if kept_count == 0:
        print("Warning: No audio layers found! Double check the input file.")
    else:
        print(f"Saving to {output_path}...")
        save_file(audio_tensors, output_path)
        print("Done! 🎉")

if __name__ == "__main__":
    main()
