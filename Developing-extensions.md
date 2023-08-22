[Adding to Index](#official-extension-index)

An extension is just a subdirectory in the `extensions` directory.

Web ui interacts with installed extensions in the following way:

- extension's `install.py` script, if it exists, is executed.
- extension's scripts in the `scripts` directory are executed as if they were just usual user scripts, except:
  - `sys.path` is extended to include the extension directory, so you can import anything in it without worrying
  - you can use `scripts.basedir()` to get the current extension's directory (since user can name it anything he wants)
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
