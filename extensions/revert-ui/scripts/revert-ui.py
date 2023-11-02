# This program can be enabled as an extension
# for the StableDiffusion WebUI in order to
# revert changes made by the UI modification based's extensions


with open('modules/ui.py', "r", encoding="utf8") as fichier :
    lignes = fichier.readlines()


model_index = 0
train_index = 0
for i in range(len(lignes)):

    # Searching for commented Checkpoint Merger tab
    if lignes[i].__contains__('#(modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger")') :
        print('\033[34m[REVERT UI EXTENSION]\033[0m Commented Checkpoint Merger tab found, reverting...')
        model_index = i

    # Searching for commented Train tab
    if lignes[i].__contains__('#(train_interface, "Train", "train")') :
        print('\033[34m[REVERT UI EXTENSION]\033[0m Train tab found, reverting...')
        train_index = i

#Doing modifications
if model_index != 0 :
    # if line after modelmerger is commented
    if lignes[model_index+1].__contains__("#"):
        # Putting the line back to normal 
        lignes[model_index-1] = lignes[model_index-1][:-1] + ","
        lignes[model_index] = '\n\t     (modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger")\n'
    # if the line after modelmerger is not commented
    else:
        lignes[model_index] = '\t     (modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger"),\n'

if train_index != 0 :
    # Putting the line back to normal 
    lignes[train_index-1] = lignes[train_index-1][:-1] + ","
    lignes[train_index] = '\n\t     (train_interface, "Train", "train")\n'

with open('modules/ui.py', 'w', encoding='utf-8') as file:
    # Writing modifications in the UI
    file.writelines(lignes)


# The revert-ui extension will then disable itself to prevent useless compute time at startup
with open('config.json', "r", encoding="utf8") as fichier :
    lignes = fichier.readlines()


# If all extensions are disabled, we prevent the rest of the program
# of executing
i = 0
no_extensions = False
while i < (len(lignes)-1) and not no_extensions :
    if lignes[i].__contains__('"disabled_extensions": [],'):
        no_extensions = True
    i+= 1

# Putting our extension in the disabled extensions tab directly
if no_extensions:
    lignes[i-1] = '\t"disabled_extensions": ["revert-ui"],\n'


# if there are disabled extensions, we search for the correct line
else:
    found = False

    i = 0 
    while i < (len(lignes)-1) and not found :

        if lignes[i].__contains__("revert-ui"):
            found = True
        i += 1

    j = 0
    #If revert-ui isn't found (means it isn't disabled)
    #Searching for the disabled extensions line (index)
    disabled_found = False
    if not found:
        while j < (len(lignes)-1) and not disabled_found :
            if lignes[j].__contains__("disabled_extensions") :
                disabled_found = True
            j+=1

    # At this point, the disabled_extensions line is j 
    # We add the reverted-ui extenson at the beggining of the list
    if disabled_found:
        lignes[j] = lignes[j]+'\n\t\t "revert-ui", \n'
    else:
        print("\033[34m[REVERT UI EXTENSION]\033[0m Disabled Extensions tab wasn't found, can't add the extension to it.")

with open('config.json', 'w', encoding='utf-8') as file:
    # Writing modifications in the config file
    file.writelines(lignes)


print("\033[34m[REVERT UI EXTENSION]\033[0m UI reverted to normal.")