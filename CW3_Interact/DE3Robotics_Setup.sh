#!/bin/bash

cd Desktop/DE3Robotics
catkin_make

cd ~/Desktop/DE3Robotics/src/coursework_3/src/

cp left_end_effector.urdf.xacro ~/Desktop/DE3Robotics/src/coursework_2/deniro_sim_ws/src/baxter_common/baxter_description/urdf/

cp right_end_effector.urdf.xacro ~/Desktop/DE3Robotics/src/coursework_2/deniro_sim_ws/src/baxter_common/baxter_description/urdf/

# New terminal
gnome-terminal --tab -e "bash -c 'roscore'" 
echo "Launched Roscore"

sleep 5s

gnome-terminal --tab -e "bash -c 'cd ~/Desktop/DE3Robotics; source devel/setup.bash; roslaunch baxter_gazebo baxter_world.launch'"

echo "Launched Gazebo"

# Wait for Gazebo to start up before running the next commands
echo "Waiting for Gazebo to load..."
while [[ -z $(rostopic echo -n 1 /gazebo/model_states) ]]; do
    sleep 1s
done
echo "Gazebo loaded!"

cd ~/Desktop/DE3Robotics
source devel/setup.bash

cd src/coursework_3/src
python position_controller.py initPose

