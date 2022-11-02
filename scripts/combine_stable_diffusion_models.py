# The purpose of this is to take 2 Stabble Diffusion finetuned model weights and add or subtract the difference from eachother into a brand new model
# Might be a bit CPU ram hungry as it is now
import torch, sys, copy, os, argparse
from tkinter import filedialog, Tk


####### Instructions ####
# you may need to do "pip install torch numpy pytorch-lightning" first

### Merging ###
    # Ez-mode for merging just run `python combine_stable_diffusion_models.py` and it will ask you to select 2 to 3 models to merge together via GUI.  No parameters needed!

    # manual modes below (no GUI)
    # Run python script like this to average two models together (optionally add --modelNth to take the average of all 3)
    # python combine_stable_diffusion_models.py --model1 "<path>" --model2 "<path>" --alpha 0.5

### Transfering Weights ###
    # Ez-mode for weight transfer just run `python combine_stable_diffusion_models.py --type "transfer_diff"` and it will ask you to select 3 models (model1 is the one the weights are transfered to, model2 is the base model that 3 is trained from).  No additional parameters needed!

    # manual modes below (no GUI)
    # Run like this to transfer the difference of two models onto a third
    #   modelTarget + (modelNth - modelBase) * alpha
    #   OR model1 + (model3(+) - model2) * alpha
    # You can also diff multuple models together at once with an average, min, or max of the differences
    #   modelTarget + Math.Average(modelNth - modelBase, modelNth - modelBase) * alpha
    # no gui:  python combine_stable_diffusion_models.py --modelTarget "<path>" --modelBase "<path>" --modelNth "<path>" --type "transfer_diff"


# Note:
# Your existing models will be un-touched, this wil create and save a new model
# If your new model is outputting black images, try merging it in reverse order model2->model1.  I saw this happen when using a custom pruned model, so I reversed the model order and it worked.
#########################


def main():    
    weights_key = "state_dict"
    combined_model = None
    weights_action = "merge" # 'merge' | 'transfer_diff'
    alpha = 0.5 # 0 == model1 weights, 0.5 == average of model1 and model2
    fp16 = True
    layer_counter = 0
    hash = True
    max_diff = False


    if len(sys.argv) < 2:
        # When no args are included, prompt user to select models first
        model_path1, model_path2, model_pathN = user_select_models()

    else:
        #when executed via command line, parse users args
        ap = argparse.ArgumentParser()
        ap.add_argument("-m1", "--model1", required=False, default="", help="the file path of the first model")
        ap.add_argument("-m2", "--model2", required=False, default="", help="the file path of the second model")
        ap.add_argument("-mt", "--modelTarget", required=False, default="", help="transfer_diff only: the file path of the model we want to transfer knowledge to")
        ap.add_argument("-mb", "--modelBase", required=False, default="", help="transfer_diff only: the file path of the base model used to train the Nth model")
        ap.add_argument("-mn", "--modelNth", required=False, default="", help="the file path of the Nth model (if needed)")        
        ap.add_argument("-t", "--type", required=False, default="merge", help="the type of model to output, either 'merge' for a weight merged model, or 'transfer_diff' for transfering the weight differences of two models onto a third")
        ap.add_argument("-a", "--alpha", required=False, type=float, help="the alpha of the merge. 0 == model1 weights, 0.5 == average of model1 and model2.  For merge, default is 0.5, for transfer_diff default is 1")
        ap.add_argument("--no_fp16", action='store_true', help="Will save the full fp32 precision model (larger)")
        ap.add_argument("--no_hash", action='store_true', help="Dont bother computing model hashes")
        ap.add_argument("--max", action='store_true', help="transfer_diff only: take the maximum of the difference instead of the minimum by default")
        args = vars(ap.parse_args())   

        model_path1 = args["model1"] or args["modelTarget"]
        model_path2 = args["model2"] or args["modelBase"]
        model_pathN = [args["modelNth"]]
        weights_action = args["type"]
        alpha = args["alpha"]
        fp16 = not args["no_fp16"]
        hash = not args["no_hash"]
        max_diff = args["max"]

        # When no model args are included as args, prompt user to select models
        if not model_path1 and not model_path2:            
            model_path1, model_path2, model_pathN = user_select_models()

        # Set default alpha if it was not included
        if not alpha and weights_action == "transfer_diff":
            # when diff merging with torch.max on many models, we need a lower alpha because it blows out the results earlier 
            if len(model_pathN) > 1 and max_diff:
                alpha = 0.5
            else:
                alpha = 1
        elif not alpha:
            alpha = 0.5

        # check parameters are valid
        validations(weights_action, alpha, model_pathN)

    print(model_path1)
    print(model_path2)
    if len(model_pathN):
        for model in model_pathN:
            print(model)    
    print()  
    
    if hash:
        print("Computing model hashes...")
        hash1 = model_hash(model_path1)
        hash2 = model_hash(model_path2)
        if len(model_pathN):
            for model in model_pathN:
                hash3 = model_hash(model)

    # output will be same directory as first selected weight file
    out_dir = os.path.dirname(model_path1)    

    model1 = torch.load(model_path1, map_location=torch.device('cpu'))
    model2 = torch.load(model_path2, map_location=torch.device('cpu')) 
    modelN = []
    if len(model_pathN):
        for model in model_pathN:
            modelN.append(torch.load(model, map_location=torch.device('cpu')))

    # Make sure this is a SD model
    if not weights_key in model1:
        print(f"Could not find weights property {weights_key} in the model.  Did you load the correct SD weight file ending in .ckpt?")
        exit()

    # create new weights that we will be altering
    combined_model = model1

    print("Computing diffs...")
    # For each layer of the NN
    for layer in model1[weights_key].keys():

        # merge both layer weights based on the alpha value
        if weights_action == "merge" and not len(model_pathN) and has_layer(combined_model[weights_key], layer) and has_layer(model2[weights_key], layer):            
            combined_model[weights_key][layer] = merge_weight(model1[weights_key], model2[weights_key], layer, alpha)
            layer_counter+=1

        # merge 3+ model weights together
        elif weights_action == "merge" and len(model_pathN) and has_layer(combined_model[weights_key], layer):
            #merge the weights based on the alpha value
            combined_model[weights_key][layer] = merge_weight_nth(model1[weights_key], model2[weights_key], modelN, weights_key, layer)
            layer_counter+=1
        
        # add the difference of model1 and model2 layers to model3
        elif weights_action == "transfer_diff" and has_layer(combined_model[weights_key], layer) and has_layer(model2[weights_key], layer):            
            combined_model[weights_key][layer] = add_diff_weight(model1[weights_key], model2[weights_key], modelN, weights_key, layer, alpha, max_diff) 
            layer_counter+=1
        
        if fp16 and has_layer(combined_model[weights_key], layer):
            try:
                combined_model[weights_key][layer] = combined_model[weights_key][layer].half()            
            except:
                print(f"Warning, cant convert this layer to fp16 {layer}")


    print(f"{str(layer_counter)} layers merged!")
    print("")

    # save the new model
    print(f"Saving new model to {out_dir}")
    new_model_name = get_new_model_name(model_path1, model_path2, model_pathN, weights_action, alpha, max_diff)

    print("name", new_model_name)
    torch.save(combined_model, f'{out_dir}/{new_model_name}.ckpt')


    if hash:
        # Print model hashes so you can tell if the operation was successful (hash changed)
        hashc = model_hash(f'{out_dir}/{new_model_name}.ckpt')    
        print()
        print(f"model hash 1 {hash1}")
        print(f"model hash 2 {hash2}")
        if len(model_pathN):
            print(f"model hash 3 {hash3}")
        print(f"model hash combined {hashc}   <- this should not match any above")

    print() 
    print("All done")







# interpolate between checkpoints with mixing coefficient alpha
def merge_weight(model1_weights, model2_weights, layer, alpha):
    #ignore mismatched layer shapes  
    if str(model1_weights[layer].shape) != str(model2_weights[layer].shape):
        return model1_weights[layer]

    return (1-alpha) * model1_weights[layer] + alpha * model2_weights[layer]


# average the weights of 3+ models together
def merge_weight_nth(model1_weights, model2_weights, modelN, weights_key, layer):    
    #ignore mismatched layer shapes  
    if has_layer(model2_weights,layer) and str(model1_weights[layer].shape) != str(model2_weights[layer].shape):
        return model1_weights[layer]

    # if model_x is missing layer, then use model_1's layer
    model1_tensor = model1_weights[layer]
    model2_tensor = model2_weights[layer] if has_layer(model2_weights,layer) else model1_weights[layer]
    
    # init model sum variable
    modeln_tensor = torch.zeros_like(model1_weights[layer])

    # for each nth model add up its tensors
    for model_n in modelN:
        # make sure the layer exists
        model_n = model_n[weights_key][layer] if has_layer(model_n[weights_key], layer) else model1_weights[layer]
        # prevent adding floats to longs occasionally?
        try:
            modeln_tensor += model_n
        except:
            modeln_tensor += model1_weights[layer]

    # compute the average of all the models
    return (model1_tensor + model2_tensor + modeln_tensor)/(2 + len(modelN))


# add the difference of modelNth and model2 to model1 like  modelTarget + (modelNth - modelBase) * alpha 
def add_diff_weight(model1_weights, model2_weights, modelN, weights_key, layer, alpha, max_diff):  
    #ignore mismatched layer shapes  
    if str(model1_weights[layer].shape) != str(model2_weights[layer].shape):
        return model1_weights[layer]

    # initilize the diff weights to 0
    weight_diffs = torch.zeros_like(model1_weights[layer])
    missing_layers = 0 # when a model is missing the layer dont includ it in the average
    
    # if model_x or _y is missing layer, then use model_3's layer
    model1_tensor = model1_weights[layer]    

    for model_n in modelN:
        # make sure the layer exists in both base and diff model
        if not has_layer(model_n[weights_key], layer) or not has_layer(model2_weights, layer):
            missing_layers += 1            
            # print(layer)
            continue

        # subbtract the two layers
        current_diff = (model_n[weights_key][layer] - model2_weights[layer])

        if max_diff:
            # take the diff that is the furthest from 0 (like Math.max)
            weight_diffs = torch.where(torch.abs(weight_diffs) > torch.abs(current_diff), weight_diffs, current_diff)
        else:
            weight_diffs = torch.where((torch.abs(weight_diffs) < torch.abs(current_diff)) & (torch.abs(weight_diffs) > torch.zeros_like(model1_weights[layer])), weight_diffs, current_diff)

    # add the difference to the third model (scale the difference by alpha, default 1)
    return model1_tensor + (weight_diffs * alpha)


def has_layer(model_weights, layer):
    # detect when layer does not exist, or is not a tensor (ignore dicts)
    return layer in model_weights and type(model_weights[layer]) is not dict


# select a pytorch model to open
def get_file():
	root = Tk()
	root.withdraw()
	root.wm_attributes('-topmost', 1)

	path = filedialog.askopenfilename()
	path = path.rstrip()

	# if not path:
	# 	print('Unrecognised path:', path)
	
	return path

# prompt the user to select the models manually
def user_select_models():
    model_pathN = []

    print("Select first model")
    model_path1 = get_file()        
    if not model_path1:
        print("no model select, ending script")
        exit()

    print("Select second model")
    model_path2 = get_file()

    while True:
        print("Select Nth model (if needed). Press cancel to continue")
        nth_path = get_file()  
        if not nth_path:
            break
        model_pathN.append(nth_path)

    return (model_path1, model_path2, model_pathN)


def model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'

# A path like 'directory/file.txt' returns 'file'
def getFileBaseName(path):    
    baseName = os.path.basename(path)
    filename = os.path.splitext(baseName)[0]
    return filename


def get_new_model_name(model_path1, model_path2, model_pathN, weights_action, alpha, max_diff):
    model1_name = getFileBaseName(model_path1)
    model2_name = getFileBaseName(model_path2)
    # A1111's repo looks for the string inpainting to load those models correct, so append it here too when needed
    is_inpainting_model = "inpainting" in model_path1
    optional_inpaint_str = "-inpainting" if is_inpainting_model else ""

    # if 3+ models
    if len(model_pathN):
        if weights_action == "transfer_diff":            
            diff_names = ""
            max_or_min = "max" if max_diff else "min"
            for model in model_pathN:
                diff_names += getFileBaseName(model) + ("_~_" if len(model_pathN) > 1 else "")
            new_model_name= f"{model1_name}_+_({diff_names})x{alpha}_{max_or_min}-diff{optional_inpaint_str}"
        else:
            new_model_name= f"{model1_name}_{model2_name}_{getFileBaseName(model_pathN[0])}_a{alpha}{optional_inpaint_str}"

    else:
        new_model_name = f"{model1_name}_{model2_name}_a{alpha}{optional_inpaint_str}"

    return new_model_name


# stop script and warn user when params are bad
def validations(weights_action, alpha, model_pathN):
    if weights_action != "transfer_diff" and weights_action != "merge":
        print("--action must be one of 'merge' or 'transfer_diff'")
        exit()

    if weights_action == "transfer_diff" and not len(model_pathN):
        print("a third model must be selected when using --type 'transfer_diff'")
        exit()

    if len(model_pathN) and alpha != 0.5 and weights_action != "transfer_diff":
        print("Alpha will be set to 0.5 when averaging 3 models at a time, too lazy to implement it properly")

main()