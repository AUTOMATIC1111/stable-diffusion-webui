cd ~/catkin_ws/src
catkin_create_pkg midas_cpp std_msgs roscpp cv_bridge sensor_msgs image_transport
cd ~/catkin_ws
catkin_make

chmod +x ~/catkin_ws/devel/setup.bash
printf "\nsource ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/catkin_ws/devel/setup.bash


sudo rosdep init
rosdep update
#rospack depends1 midas_cpp 
roscd midas_cpp
#cat package.xml
#rospack depends midas_cpp