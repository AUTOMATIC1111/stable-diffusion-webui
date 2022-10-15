{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    git
    libGL
    glib
    python39
    python39Packages.pip
    python39Packages.virtualenv
    cudatoolkit
  ]);
  runScript = "bash";
}).env
