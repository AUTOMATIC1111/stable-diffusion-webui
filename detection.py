# This script detects which GPU is currently used in Windows and Linux
import os
import sys

def check_gpu():
    # First, check if the `lspci` command is available
    if not os.system("which lspci > /dev/null") == 0:
        # If the `lspci` command is not available, try the `dxdiag` command on Windows
        if os.name == "nt":
            # On Windows, run the `dxdiag` command and check the output for the "Card name" field
            # Create the dxdiag.txt file
            os.system("dxdiag /t dxdiag.txt")

            # Read the dxdiag.txt file
            with open("dxdiag.txt", "r") as f:
                output = f.read()

            if "Card name" in output:
                card_name_start = output.index("Card name: ") + len("Card name: ")
                card_name_end = output.index("\n", card_name_start)
                card_name = output[card_name_start:card_name_end]
            else:
                card_name = "Unknown"
            print(f"Card name: {card_name}")
            os.remove("dxdiag.txt")
            if "AMD" in card_name:
                return "AMD"
            elif "Intel" in card_name:
                return "Intel"
            elif "NVIDIA" in card_name:
                return "NVIDIA"
            else:
                return "Unknown"
    else:
        # If the `lspci` command is available, use it to get the GPU vendor and model information
        output = os.popen("lspci | grep -i vga").read()
        if "AMD" in output:
            return "AMD"
        elif "Intel" in output:
            return "Intel"
        elif "NVIDIA" in output:
            return "NVIDIA"
        else:
            return "Unknown"

