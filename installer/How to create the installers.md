**Important:** These instructions are for the developers of this project, not for users! The users should use the pre-created zip files for installation.

This guide explains how to create the zip files that users will use for installing.

The installer zip contains two files: the script, and the micromamba binary.

Micromamba is a single ~8mb binary file, that acts like a package manager (drop-in replacement for conda).

# Download micromamba from:
* Windows x64: `curl -Ls https://micro.mamba.pm/api/micromamba/win-64/latest | tar -xvj bin/micromamba_win_x64.exe`

* Linux x64: `curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba_linux_x64`
* Linux arm64: `curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba_linux_arm64`

* Mac x64: `curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj bin/micromamba_mac_x64`
* Mac arm64 (M1/Apple Silicon): `curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba_mac_arm64`

The download link provides tar.bz2 files.

(source https://mamba.readthedocs.io/en/latest/installation.html)

# Create the installer
Create the following folder structure, and zip it up.

For Linux/Mac: Make sure the `chmod u+x` permission is granted to `install.sh` and the corresponding `micromamba` binary.

### Windows x64:
```
.\install.bat
.\installer_files\micromamba_win_x64.exe
```

### Linux x64:
```
.\install.sh
.\installer_files\micromamba_linux_x64
```

### Linux arm64:
```
.\install.sh
.\installer_files\micromamba_linux_arm64
```

### Mac x64:
```
.\install.sh
.\installer_files\micromamba_mac_x64
```

### Mac arm64 (M1/Apple Silicon):
```
.\install.sh
.\installer_files\micromamba_mac_arm64
```
