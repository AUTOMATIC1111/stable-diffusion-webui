{ pkgs ? import <nixpkgs> {}}:

with pkgs;
let
  pythonPackages = python310Packages;
  nixGLSrc = builtins.fetchGit {
    url = "https://github.com/guibou/nixGL";
    rev = "7165ffbccbd2cf4379b6cd6d2edd1620a427e5ae";
  };

  nixGLDefault = (pkgs.callPackage nixGLSrc {}).auto.nixGLDefault;
  nvidiaPackages = pkgs:
      with pkgs; [
        cudaPackages_10_2.cudatoolkit
        cudaPackages.cudnn
        nixGLDefault
      ];
  requiredPackags = [
    python310
    git
    stdenv
    glib
    pythonPackages.venvShellHook
    pythonPackages.fastapi
    pythonPackages.pytorch-bin
    pythonPackages.torchvision
    pythonPackages.tqdm
    pythonPackages.pip
    pythonPackages.setuptools
    pythonPackages.yapf
  ];
in pkgs.mkShell rec {
  name = "Imagine.nvim-StableDiffusion";
  venvDir = ".venv";

  #nrizq2w7q86fgpbmcx178vv5s4hxdlfa-hello.dr Required for building C extensions
  CUDA_PATH = "${cudaPackages_10_2.cudatoolkit}";
  LD_LIBRARY_PATH =
    "${cudaPackages_10_2.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:${cudaPackages_10_2.cudatoolkit.lib}/lib:${zlib}/lib:${stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:/run/opengl-driver-32/lib:/usr/lib:/usr/lib32:/run/opengl-driver/lib:/run/opengl-driver-32/lib:/usr/lib:/usr/lib32:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH";

  buildInputs = requiredPackags ++ (nvidiaPackages pkgs);

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    export LD_LIBRARY_PATH=$(nixGL printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH:${R}/lib/R/lib:${readline}/lib
    unset SOURCE_DATE_EPOCH
    SOURCE_DATE_EPOCH=$(date +%s)
    pip install -r requirements.txt --no-cache-dir --prefer-binary
    pip install -e .
    pip install -r repositories/CodeFormer/requirements.txt --prefer-binary
    pip install -r repositories/CodeFormer/requirements.txt --prefer-binary
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${R}/lib/R/lib:${readline}/lib
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
    SOURCE_DATE_EPOCH=$(date +%s)
  '';
}
