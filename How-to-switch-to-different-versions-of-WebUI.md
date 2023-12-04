# Release candidate
Release candidate is a version that will soon be released as a new stable version. For example, before 1.7.0 is released, there is 1.7.0-RC version, which is a release candidate - it has all new features and is available for testing.

### How to switch existing installation to release candidate
Run those commands in webui directory:

```
git switch release_candidate
git pull
```

### How to switch back to stable version in master branch:
Run those commands in webui directory:
```
git switch master
```

### How to get release candidate in a new webui installation

Run those commands (this will create a directory called `webuirc` - you can use a different name, and you can also rename the directory afterwards):

```
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git webuirc
cd webuirc
git switch release_candidate
```

