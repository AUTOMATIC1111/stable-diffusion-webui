import os

# Get the current directory
current_directory = os.getcwd()

# Find all ".safetensor" files in the current directory
safetensor_files = [file for file in os.listdir(current_directory) if file.endswith('.safetensors')]

print("This script will generate .txt files to ALL undescribed LoRas (.safetensor) located in the working directory. Inside them you can put description of the model and prompts to be used when you click on the model in webUI.")
stop = False
while stop == False:
    check = input("Do you wish to procceed? [Y/N]: ")
    if check.lower() == 'y':
        stop = True

        # Loop over the safetensor files
        for safetensor_file in safetensor_files:
            # Generate the corresponding txt filename
            txt_file = os.path.splitext(safetensor_file)[0] + '.txt'

            # Check if the txt file already exists
            if not os.path.exists(txt_file):
                # Create an empty txt file
                with open(txt_file, 'w') as file:
                    pass
                print(f"Created {txt_file}")

        # Print a message after completing the task
        print("Finished creating missing txt files.")
    elif check.lower() == 'n':
        stop = True
    else:
        print("Invalid input!")