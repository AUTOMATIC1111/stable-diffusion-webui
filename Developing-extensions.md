[Adding to Index](#official-extension-index)

An extension is just a subdirectory in the `extensions` directory.

Web ui interacts with installed extensions in the following way:

- extension's `install.py` script, if it exists, is executed.
- extension's scripts in the `scripts` directory are executed as if they were just usual user scripts, except:
  - `sys.path` is extended to include the extension directory, so you can import anything in it without worrying. **HOWEVER, please either use a unique file name, or put your files in a uniquely-named folder, as the module will be cached to the global Python module tree by the name you used to import, and create surprises for other components that happen to use the same name.**
  - you can use `scripts.basedir()` to get the current extension's directory (since user can name it anything he wants)
  - - note: `scripts.basedir()` must be used druing extension import stage,<br>if during other time it will return `webui root`.<details><summary><code>click to see scripts.basedir() usage example</code></summary><p>
      > stable-diffusion-webui\extensions\example_extension_dir\scripts\example.py
      ```py
      from modules import scripts
      current_extension_directory = scripts.basedir()  # 'stable-diffusion-webui\extensions\example_extension_dir'
      # save it here for later use
      class ExampleScript(scripts.Script):
          def title(self):
              return 'Example script'
          def show(self, is_img2img):
              scripts.basedir()  # 'B:\GitHub\stable-diffusion-webui'
              return scripts.AlwaysVisible
      ```
      </p>
      </details> 

- extension's javascript files in the `javascript` directory are added to the page
- extension's localization files in the `localizations` directory are added to settings; if there are two localizations with same name, they are not merged, one replaces another.
- extension's `style.css` file is added to the page
- if extension has `preload.py` file in its root directory, it is loaded before parsing commandline args
- if extension's `preload.py` has a `preload` function, it is called, and commandline args parser is passed to it as an argument. Here's an example of how to use it to add a command line argument:
```python
def preload(parser):
    parser.add_argument("--wildcards-dir", type=str, help="directory with wildcards", default=None)
```

For how to develop custom scripts, which usually will do most of extension's work, see [Developing custom scripts](Developing-custom-scripts).

- Example on how to write infotext (aka PNG info) to images [sd-webui-infotext-example](https://github.com/w-e-w/sd-webui-infotext-example) 

## Localization extensions
The preferred way to do localizations for the project is via making an extension. The basic file structure for the extension should be:

```

 📁 webui root directory
 ┗━━ 📁 extensions
     ┗━━ 📁 webui-localization-la_LA        <----- name of extension
         ┗━━ 📁 localizations                <----- the single directory inside the extension
             ┗━━ 📄 la_LA.json              <----- actual file with translations
```

Create a github repository with this file structure and ask any of people listed in collaborators section to add your extension to wiki.

If your language needs javascript/css or even python support, you can add that to the extension too.

## install.py
`install.py` is the script that is launched by the `launch.py`, the launcher, in a separate process before webui starts, and it's meant to install dependencies of the extension. It must be located in the root directory of the extension, not in the scripts directory. The script is launched with `PYTHONPATH` environment variable set to webui's path, so you can just `import launch` and use its functionality:

```python
import launch

if not launch.is_installed("aitextgen"):
    launch.run_pip("install aitextgen==0.6.0", "requirements for MagicPrompt")
```

## metadata.ini
`metadata.ini` contains metadata about the extension. It is optional, but if it exists, it must be located in the root directory of the extension. It is a [configparser](https://docs.python.org/3.10/library/configparser.html) ini file with the following contents:

```ini
# This section contains information about the extension itself.
# This section is optional.
[Extension]

# A canonical name of the extension. 
# Only lowercase letters, numbers, dashes and underscores are allowed. 
# This is a unique identifier of the extension, and the loader will refuse to 
# load two extensions with the same name. If the name is not supplied, the 
# name of the extension directory is used. Other extensions can use this 
# name to refer to this extension in the file.
Name = demo-extension

# A comma-or-space-separated list of extensions that this extension requires 
# to be installed and enabled.
# The loader will generate a warning if any of the extensions in this list is
# not installed or disabled.
# Vertical pipe can be used to list a requirement that's satisfied by
# either one of multiple extensions. No spaces between pipes and extension names.
Requires = sd-webui-controlnet|sd_forge_controlnet, sd-webui-buttons, this|that|other

# Declaring relationships of folders
# 
# This section declares relations of all files in `scripts` directory.
# By changing the section name, it can also be used on other directories 
# walked by `load_scripts` function (for example `javascript` and `localization`).
# This section is optional.
[scripts]

# A comma-or-space-separated list of extensions that files in this folder requires
# to be present.
# It is only allowed to specify an extension here.
# The loader will generate a warning if any of the extensions in this list is
# not installed or disabled.
Requires = another-extension, yet-another-extension

# A comma-or-space-separated list of extensions that files in this folder wants
# to be loaded before. 
# It is only allowed to specify an extension here.
# The loading order of all files in the specified folder will be moved so that 
# the files in the current extension are loaded before the files in the same 
# folder in the listed extension.
Before = another-extension, yet-another-extension

# A comma-or-space-separated list of extensions that files in this folder wants
# to be loaded after.
# Other details are the same as `Before` key.
After = another-extension, yet-another-extension

# Declaring relationships of a specific file
# 
# This section declares relations of a specific file to files in the same 
# folder of other extensions.
# By changing the section name, it can also be used on other directories
# walked by `load_scripts` function (for example `javascript` and `localization`).
# This section is optional.
[scripts/another-script.py]

# A comma-or-space-separated list of extensions/files that this file requires
# to be present.
# The `Requires` key in the folder section will be prepended to this list.
# The loader will generate a warning if any of the extensions/files in this list is
# not installed or disabled.
# It is allowed to specify either an extension or a specific file.
# When referencing a file, the folder name must be omitted.
# 
# For example, the `yet-another-extension/another-script.py` item refers to 
# `scripts/another-script.py` in `yet-another-extension`.
Requires = another-extension, yet-another-extension/another-script.py, xyz_grid.py

# A comma-or-space-separated list of extensions that this file wants
# to be loaded before.
# The `Before` key in the folder section will be prepended to this list.
# The loading order of this file will be moved so that this file is 
# loaded before the referenced file in the list.
Before = another-extension, yet-another-extension/another-script.py, xyz_grid.py

# A comma-or-space-separated list of extensions that this file wants
# to be loaded after.
# Other details are the same as `Before` key.
After = another-extension, yet-another-extension/another-script.py, xyz_grid.py

# A section starting with "callbacks/" allows you to change the position of
# the mentioned callback relative to others.
#
# Each callback is identified by its extension's canonical name, its filename,
# its category (here, it's ui_settings for all of them), and, optionally, by a user-specified name.
#
# You can see identifiers for existing callbacks in settings (see screenshot below).
[callbacks/swinir/swinir_model.py/ui_settings]

# This makes our swinir_model.py/ui_settings callback happen before ldsr_model.py/ui_settings
Before = ldsr/ldsr_model.py/ui_settings

# This makes our swinir_model.py/ui_settings callback happen after hypertile_script.py/ui_settings
After = hypertile/hypertile_script.py/ui_settings

```

When authoring a metadata file, please note that while the section names are case-insensitive, the keys are not.

## Minor tips
### Adding extra textual inversion dirs
This code goes into extension's script:
```python
path = os.path.join(modules.scripts.basedir(), "embeddings")
modules.sd_hijack.model_hijack.embedding_db.add_embedding_dir(path)
```
## User Examples
https://github.com/udon-universe/stable-diffusion-webui-extension-templates \
https://github.com/AliceQAQ/sd-webui-gradio-demo \
https://github.com/wcdnail/sd-web-ui-wexperimental \
https://github.com/EnsignMK/ExampleSendText

## Official Extension Index
- Add extensions here - https://github.com/AUTOMATIC1111/stable-diffusion-webui-extensions

(additionally, you could add working commit versions of your extensions+webui here: )

- https://github.com/camenduru/sd-webui-extension-records

## Internals Diagram by [@hananbeer](https://github.com/hananbeer)
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8601

https://miro.com/app/board/uXjVMdgY-TY=/?share_link_id=547908852229

![image](https://user-images.githubusercontent.com/98228077/229259967-15556a72-774c-44ba-bab5-687f854a0fc7.png)

## Licensing
I have no objections if you want to make and share an extension under a different open source license from what this project uses.
