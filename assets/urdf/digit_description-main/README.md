# digit_model
This package provides URDF and Meshes for the Agility Robotics Digit Robot, as well as a convenience function for creating `RigidBodyDynamics.Mechanism`.

Package inspired heavily by [Twan Koolen's AtlasRobot.jl](https://github.com/tkoolen/AtlasRobot.jl)

# Running The RVIZ Visualizer
Refer to https://github.gatech.edu/GeorgiaTechLIDARGroup/digit_main on preparing necessary packages and setting up the environment running digit.

1. Clone this repo under ~/catkin_ws/src
```
cd ~/catkin_ws/src
git clone https://github.gatech.edu/GeorgiaTechLIDARGroup/digit_description.git
```
2. Navigate to catkin_ws directory and build the project
```
cd ~/catkin_ws
catkin build
```
3. Once built successfully, source the following setup file. Otherwise a wierd error will occur.
```
source devel/setup.bash
```

4. Once all done, Run RVIZ
```
roslaunch digit_description display.launch
```




