An extension is just a subdirectory in the `extensions` directory.

Web ui interacts with installed extensions in the following way:

- extension's scripts in the `scripts` directory are executed as if they were just usual user scripts, except:
  - `sys.path` is extended to include the extension directory, so you can import anything in it without worrying
  - you can use `scripts.basedir()` to get the current extension's directory (since user can name it anything he wants)
- extension's javascript files in the `javascript` directory are added to the page
- extension's `style.css` file is added to the page

