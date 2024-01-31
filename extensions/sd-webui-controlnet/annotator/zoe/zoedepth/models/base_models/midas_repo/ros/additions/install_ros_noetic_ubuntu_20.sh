#@title  { display-mode: "code" }

#from http://wiki.ros.org/indigo/Installation/Ubuntu

#1.2 Setup sources.list
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 1.3 Setup keys
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | sudo apt-key add -

# 1.4 Installation
sudo apt-get update
sudo apt-get upgrade

# Desktop-Full Install:
sudo apt-get install ros-noetic-desktop-full

printf "\nsource /opt/ros/noetic/setup.bash\n" >> ~/.bashrc

# 1.5 Initialize rosdep
sudo rosdep init
rosdep update 


# 1.7 Getting rosinstall (python)
sudo apt-get install python3-rosinstall
sudo apt-get install python3-catkin-tools
sudo apt-get install python3-rospy
sudo apt-get install python3-rosdep
sudo apt-get install python3-roscd
sudo apt-get install python3-pip