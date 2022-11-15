let
  nixpkgs = import <nixpkgs> {};
in
  with nixpkgs;
  mkShell rec {
    name = "stable-diffusion-webui";
    buildInputs = [
      git # The program instantly crashes if git is not present, even if everything is already downloaded
      python310
      stdenv.cc.cc.lib
    ];
    shellHook = ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath buildInputs}";
    alias webui.sh="./webui.sh"
    alias webui="webui.sh"
    echo "Type 'webui' to start the server."
    echo "Note: You may need to switch to compute mode using a tool such as system76-power."
    '';
  }
