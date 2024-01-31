import argparse
import torch
from safetensors.torch import load_file, save_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd15", default=None, type=str, required=True, help="Path to the original sd15.")
    parser.add_argument("--control", default=None, type=str, required=True, help="Path to the sd15 with control.")
    parser.add_argument("--dst", default=None, type=str, required=True, help="Path to the output difference model.")
    parser.add_argument("--fp16", action="store_true", help="Save as fp16.")
    parser.add_argument("--bf16", action="store_true", help="Save as bf16.")
    args = parser.parse_args()

    assert args.sd15 is not None, "Must provide a original sd15 model path!"
    assert args.control is not None, "Must provide a sd15 with control model path!"
    assert args.dst is not None, "Must provide a output path!"

    # make differences: copy from https://github.com/lllyasviel/ControlNet/blob/main/tool_transfer_control.py

    def get_node_name(name, parent_name):
        if len(name) <= len(parent_name):
            return False, ''
        p = name[:len(parent_name)]
        if p != parent_name:
            return False, ''
        return True, name[len(parent_name):]

    # remove first/cond stage from sd to reduce memory usage
    def remove_first_and_cond(sd):
        keys = list(sd.keys())
        for key in keys:
            is_first_stage, _ = get_node_name(key, 'first_stage_model')
            is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
            if is_first_stage or is_cond_stage:
                sd.pop(key, None)
        return sd
    
    print(f"loading: {args.sd15}")
    if args.sd15.endswith(".safetensors"):
        sd15_state_dict = load_file(args.sd15)
    else:
        sd15_state_dict = torch.load(args.sd15)
        sd15_state_dict = sd15_state_dict.pop("state_dict", sd15_state_dict)
    sd15_state_dict = remove_first_and_cond(sd15_state_dict)

    print(f"loading: {args.control}")
    if args.control.endswith(".safetensors"):
        control_state_dict = load_file(args.control)
    else:
        control_state_dict = torch.load(args.control)
    control_state_dict = remove_first_and_cond(control_state_dict)

    # make diff of original and control
    print(f"create difference")
    keys = list(control_state_dict.keys())
    final_state_dict = {"difference": torch.tensor(1.0)}                  # indicates difference
    for key in keys:
        p = control_state_dict.pop(key)

        is_control, node_name = get_node_name(key, 'control_')
        if not is_control:
            continue

        sd15_key_name = 'model.diffusion_' + node_name
        if sd15_key_name in sd15_state_dict:                              # part of U-Net
            # print("in sd15", key, sd15_key_name)
            p_new = p - sd15_state_dict.pop(sd15_key_name)
            if torch.max(torch.abs(p_new)) < 1e-6:                        # no difference?
                print("no diff", key, sd15_key_name)
                continue
        else:
            # print("not in sd15", key, sd15_key_name)
            p_new = p                                                     # hint or zero_conv

        final_state_dict[key] = p_new

    save_dtype = None
    if args.fp16:
        save_dtype = torch.float16
    elif args.bf16:
        save_dtype = torch.bfloat16
    if save_dtype is not None:
        for key in final_state_dict.keys():
            final_state_dict[key] = final_state_dict[key].to(save_dtype)

    print("saving difference.")
    if args.dst.endswith(".safetensors"):
        save_file(final_state_dict, args.dst)
    else:
        torch.save({"state_dict": final_state_dict}, args.dst)
    print("done!")
