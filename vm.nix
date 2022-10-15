{ pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaSupport = true;
  };
} , ... }: {
  boot.kernelPackages = pkgs.linuxPackages_latest;
  users.users.ai = {
  isNormalUser = true;
  home = "/home/ai-vm";
  description = "AI User";
  extraGroups = [ "wheel" ];
  initialPassword = "toor";
  };
  
 environment.systemPackages = with pkgs; [
   git gitRepo gnupg autoconf curl
   procps gnumake util-linux m4 gperf unzip
   cudatoolkit linuxPackages.nvidia_x11
   libGLU libGL
   xorg.libXi xorg.libXmu freeglut
   xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib
   ncurses5 binutils
   glib
   conda
 ];

  nix.nixPath = [ "nixpkgs=${pkgs.path}" ];

  virtualisation.memorySize = 16 * 1024;
  virtualisation.cores = 8;
  virtualisation.diskSize = 500 * 1024;
}
