# Pull requests
To contribute, clone the repository, make your changes, commit and push to your clone, and submit a pull request.

> **Note**
If you're not a contributor to this repository, you need to fork and clone the repository before pushing your changes. For more information, check out [Contributing to Projects](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) in the GitHub documentation.

* If you are adding a lot of code, **consider making it an [extension](Extensions) instead**.
  * this way, you will be able to make changes in the future without needing my approval
  * I also won't have to study your code
  * if your would-be extension needs some changes to the repo (like adding an API or callback), this change is welcome to be added, and it will also profit other developers
* Do not add multiple unrelated things in same PR.
* PRs should target the `dev` branch.
* Make sure that your changes do not break anything by running [tests](Tests).
* Do not submit PRs where you just take existing lines and reformat them without changing what they do.
* If you are submitting a bug fix, there must be a way for me to reproduce the bug.
* Do not use your clone's `master` or `main` branch to make a PR - create a branch and PR that.

<details><summary>There is a discord channel for development of the webui (click to expand). Join if you want to talk about a PR in real time. Don't join if you're not involved in development.</summary><blockquote>
<details><summary>This is a discord for development only, NOT for tech support.
</summary><blockquote>

[Dev discord](https://discord.gg/WG2nzq3YEH)  
</details></blockquote></details>

If you are making changes to used libraries or the installation script, you must verify them to work on default Windows installation from scratch. If you cannot test if it works (due to your OS or anything else), do not make those changes (with possible exception of changes that explicitly are guarded from being executed on Windows by `if`s or something else).

# Code style
We use linters to enforce style for python and javascript. If you make a PR that fails the check, I will ask you to fix the code until the linter does not complain anymore.

Here's how to use linters locally:
#### python
Install: `pip install ruff`

Run: `ruff .` (or `python -mruff .`)

#### javascript
Install: install npm on your system.

Run: `npx eslint .`

# Quirks
* `webui.user.bat` is never to be edited
* `requirements_versions.txt` is for python 3.10.6
* `requirements.txt` is for people running on colabs and whatnot using python 3.7

# Gradio
Gradio at some point wanted to add this section to shill their project in the contributing section, which I didn't have at the time, so here it is now.

For [Gradio](https://github.com/gradio-app/gradio) check out the [docs](https://gradio.app/docs/) to contribute:
Have an issue or feature request with Gradio? open a issue/feature request on github for support: https://github.com/gradio-app/gradio/issues

