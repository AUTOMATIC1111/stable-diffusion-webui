An extension is just a subdirectory in the `extensions` directory.

Web ui interacts with installed extensions in the following way:

- extension's `install.py` script, if it exists, is executed.
- extension's scripts in the `scripts` directory are executed as if they were just usual user scripts, except:
  - `sys.path` is extended to include the extension directory, so you can import anything in it without worrying
  - you can use `scripts.basedir()` to get the current extension's directory (since user can name it anything he wants)
- extension's javascript files in the `javascript` directory are added to the page
- extension's `style.css` file is added to the page

For how to develop custom scripts, which usually will do most of extension's work, see [Developing custom scripts](Developing-custom-scripts).

## install.py
`install.py` is the script that is launched by the `launch.py`, the launcher, in a separate process before webui starts, and it's meant to install dependencies of the extension. The script is launched with `PYTHONPATH` environment variable set to webui's path, so you can just `import launch` and use its functionality:

```python
import launch

if not launch.is_installed("aitextgen"):
    launch.run_pip("install aitextgen==0.6.0", "requirements for MagicPrompt")
```
