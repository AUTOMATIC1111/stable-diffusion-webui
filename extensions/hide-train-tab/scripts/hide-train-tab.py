# This program will search for the train_interface variable in the modules/ui.py file
# Then, it will comment the associated line in order to prevent the tab from displaying

with open('modules/ui.py', "r", encoding="utf8") as fichier :
    lignes = fichier.readlines()

for i in range(len(lignes)):
    # If already commented, do nothing :
    if lignes[i].__contains__('#(train_interface, "Train", "train")') :
        print('\033[34m[HIDE TRAIN TAB EXTENSION] \033[0m Already hidden train tab found, skipping...')

    # If not, commenting it
    elif lignes[i].__contains__('(train_interface, "Train", "train")') :
        print('\033[34m[HIDE TRAIN TAB EXTENSION] \033[0m Train tab found, desactivating...')

        # Special Case 1 : 
        # the previous ligne is commented, we aren't changing it
        # However we need to get rid of it's own comma
        if (lignes[i-1].__contains__("#")):
            lignes[i] = '\t     #(train_interface, "Train", "train")\n'
            lignes[i-2] = lignes[i-2][:-2] + "\n"

        # Default case : the line is the last of the list
        # We get rid of the previous comma and comment the line
        else:
            lignes[i-1] = lignes[i-1][:-2] + "\n"
            lignes[i] = '\t     #(train_interface, "Train", "train")\n'


        print("\033[34m[HIDE TRAIN TAB EXTENSION] \033[0m Train tab removed.")


with open('modules/ui.py', 'w', encoding='utf-8') as file:
    # Writing modifications in the UI
    file.writelines(lignes)



