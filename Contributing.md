# Pull requests
To contribute, clone the repository, make your changes, commit and push to your clone, and submit a pull request.

* if you are adding a lot of code, **consider making it an [extension](Extensions) instead**.
* do not add multiple unrelated things in same PR.
* make sure that your changes do not break anything by running [tests](Tests).
* do not submit PRs where you just take existing lines and reformat them without changing what they do.
* if you are submitting a bug fix, there must be a way for me to reproduce the bug.

There is a discord channel for development of the webui: [link](https://discord.gg/WG2nzq3YEH). Join if you want to talk about a PR in real time. Don't join if you're not involved in development.

If you are making changes to used libraries or the installation script, you must verify them to work on default Windows installation from scratch. If you cannot test if it works (due to your OS or anything else), do not make those changes (with possible exception of changes that explicitly are guarded from being executed on Windows by `if`s or something else).

# Code style
I mostly follow code style suggested by PyCharm, with the exception of disabled line length limit.

# Quirks
* `webui.user.bat` is never to be edited
* `requirements_versions.txt` is for python 3.10.6
* `requirements.txt` is for people running on colabs and whatnot using python 3.7

# Gradio
Gradio at some point wanted to add this section to shill their project in the contributing section, which I didn't have at the time, so here it is now.

For [Gradio](https://github.com/gradio-app/gradio) check out the [docs](https://gradio.app/docs/) to contribute:
Have an issue or feature request with Gradio? open a issue/feature request on github for support: https://github.com/gradio-app/gradio/issues

