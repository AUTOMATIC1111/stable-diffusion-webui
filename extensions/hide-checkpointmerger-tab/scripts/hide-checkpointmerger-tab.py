# This program will search for the modelmerger variable in the modules/ui.py file
# Then, it will comment the associated line in order to prevent the tab from displaying

with open('modules/ui.py', "r", encoding="utf8") as fichier :
    lignes = fichier.readlines()


for i in range(len(lignes)):

    # If already commented, do nothing :
    if lignes[i].__contains__('#(modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger")') :
        print('\033[34m[HIDE CHECKPOINT MERGER TAB EXTENSION]\033[0m Already hidden Checkpoint Merger tab found, skipping...')



    # If not, commenting it
    elif lignes[i].__contains__('(modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger")') :
        print('\033[34m[HIDE CHECKPOINT MERGER TAB EXTENSION]\033[0m Checkpoint Merger tab found, desactivating...')

        # If the line afterwards is not commented, we need to keep the previous comma
        if lignes[i+1].__contains__('#'):
            lignes[i-1] = lignes[i-1][:-2] + "\n"
            lignes[i] = '\t     #(modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger")\n'
        else :
            lignes[i] = '\t     #(modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger")\n'
        
        print("\033[34m[HIDE CHECKPOINT MERGER TAB EXTENSION]\033[0m Checkpoint Merger tab removed.")


with open('modules/ui.py', 'w', encoding='utf-8') as file:
    # Writing modifications in the UI
    file.writelines(lignes)



