# place any test.mp4 file near with this file

# roscore
# rosnode kill -a

source ~/catkin_ws/devel/setup.bash

roscore &
P1=$!
rosrun midas_cpp talker.py &
P2=$!
rosrun midas_cpp listener_original.py &
P3=$!
rosrun midas_cpp listener.py &
P4=$!
wait $P1 $P2 $P3 $P4