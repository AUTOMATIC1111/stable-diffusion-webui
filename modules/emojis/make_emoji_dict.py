import os

current_directory = os.getcwd()
emoji_dict = {}

# loop through the files in the directory
files = [f for f in os.listdir(current_directory)]
for filename in files:
    if filename.endswith(".svg"):
        # get the emoji hex code from the filename
        emoji_hex = filename.split(".")[0]
        emoji_key = "".join(["\\u{" + x + "}" for x in emoji_hex.split("-")])
        # add the contents to the dictionary with the hex code as the key
        emoji_dict[emoji_key] = filename

#print the dictionary 
with open(os.path.join(current_directory, "emoji_dict.txt"), "w") as f:
    f.write(repr(emoji_dict))
