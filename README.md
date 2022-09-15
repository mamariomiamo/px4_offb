# px4_offb
Offb developed for px4 autopilot testing

The following assumes PX4-Autopilot is cloned to 
``` ~/autopilot/official/PX4-Autopilot``` and is checked out at v1.12.3

Clone this repository and its submodules:

```
mkdir -p ~/px4_offb_ws/src
cd ~/px4_offb_ws/src
git clone --recursive git@github.com:mamariomiamo/px4_offb.git
```
Build the repository:
```
cd ~/px4_offb_ws
catkin build
```
Source the workspace:
```
cd ~/px4_offb_ws
source devel/setup.bash
```
Source PX4 and Gazebo environment
```
mv ~/px4_offb_ws/src/px4_offb/px4_source.sh ~/
source ~/px4_source.sh
```
Launch the node:
```
roslaunch px4_offb sitl_circle_trajectory.launch
```

In another terminal, send takeoff command:
```
rostopic pub /user_cmd std_msgs/Byte "data: 1"
```
Ctrl-C and send follow waypoint command:
```
rostopic pub /user_cmd std_msgs/Byte "data: 2"
```

