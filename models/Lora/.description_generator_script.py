import os

# Get the current directory
current_directory = os.getcwd()

# Find all ".safetensor" files in the current directory
safetensor_files = [file for file in os.listdir(current_directory) if file.endswith('.safetensors')]

print("This script will generate empty .txt files with the same name as ALL LoRAs (.safetensor) located in the current directory. It does not overwrite existing ones. Inside the file with the same name as the LoRA, just ending with .txt you can put description of it and preset prompt that will be used when you active the LoRA from webUI. Put description on the first line and preset prompt on the next line.")
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