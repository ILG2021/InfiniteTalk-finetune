import os
import glob
import argparse
from safetensors.torch import load_file, save_file

def convert_dir(base_dir, alpha):
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    converted_count = 0
    for ckpt in checkpoints:
        if not os.path.isdir(ckpt):
            continue
        adapter_path = os.path.join(ckpt, "adapter_model.safetensors")
        inference_path = os.path.join(ckpt, "lora_for_inference.safetensors")
        state_path = os.path.join(ckpt, "trainer_state.json")
        
        if os.path.exists(adapter_path) and not os.path.exists(inference_path):
            print(f"Processing {ckpt} ...")
            
            # Auto-detect alpha from trainer_state.json if available
            current_alpha = alpha
            if os.path.exists(state_path):
                try:
                    import json
                    with open(state_path, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                    if "args" in state_data and "lora_alpha" in state_data["args"]:
                        current_alpha = float(state_data["args"]["lora_alpha"])
                        print(f"  -> Auto-detected lora_alpha = {current_alpha} from trainer_state.json")
                except Exception as e:
                    print(f"  -> Could not parse trainer_state.json, falling back to alpha = {current_alpha} ({e})")
            
            sd = load_file(adapter_path)
            new_sd = {}
            for k, v in sd.items():
                new_k = f"diffusion_model.{k}"
                if k.endswith("lora_up.weight"):
                    # Find corresponding lora_down to get rank
                    down_key = k.replace("lora_up.weight", "lora_down.weight")
                    if down_key in sd:
                        rank = sd[down_key].shape[0]
                        scaling = current_alpha / rank
                        new_sd[new_k] = v * scaling
                    else:
                        new_sd[new_k] = v # fallback
                else:
                    new_sd[new_k] = v
                    
            save_file(new_sd, inference_path)
            print(f"  -> Saved {inference_path}")
            converted_count += 1
            
    if converted_count == 0:
        print("No new checkpoints needed conversion.")
    else:
        print(f"Successfully converted {converted_count} checkpoints.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="output/my_lora", help="Directory containing checkpoint-* folders")
    parser.add_argument("--alpha", type=float, default=64.0, help="LoRA alpha used during training")
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist.")
    else:
        convert_dir(args.dir, args.alpha)
