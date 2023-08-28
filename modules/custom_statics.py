import os
import gc
import re

import modules.paths as paths
            
class CustomStatics:

    @staticmethod
    # loads a file with strings structured as below, on each line with a : between the search and replace strings, into a list
    # search0:replace0
    # search string:replace string
    #
    # Then replaces all occurrences of the list's search strings with the list's replace strings in one go
    def mass_replace_strings(input_string):
        with open(os.path.join(paths.data_path, "custom_statics/Replacements.txt"), "r", encoding="utf8") as file:
            replacements = file.readlines()
    
            replacement_dict = {}
            for line in replacements:
                search, replace = line.strip().split(":")
                replacement_dict[search] = replace
    
            def replace(match_text):
                return replacement_dict[match_text.group(0)]
    
            return re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in replacement_dict.keys()), replace, str(input_string))
    
        return str(geninfo)