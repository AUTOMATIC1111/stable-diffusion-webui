import argparse
import torch
from safetensors.torch import load_file, save_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--dst", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Cast to FP16.")
    args = parser.parse_args()

    assert args.src is not None, "Must provide a model path!"
    assert args.dst is not None, "Must provide a checkpoint path!"

    if args.src.endswith(".safetensors"):
        state_dict = load_file(args.src)
    else:
        state_dict = torch.load(args.src)

    if any([k.startswith("control_model.") for k, v in state_dict.items()]):
        dtype = torch.float16 if args.half else torch.float32
        state_dict = {k.replace("control_model.", ""): v.to(dtype) for k, v in state_dict.items() if k.startswith("control_model.")}
    
    if args.dst.endswith(".safetensors"):
        save_file(state_dict, args.dst)
    else:
        torch.save({"state_dict": state_dict}, args.dst)
