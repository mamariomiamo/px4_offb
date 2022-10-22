#include <ros/ros.h>
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/CommandTOL.h>
#include <sensor_msgs/NavSatFix.h>
#include <mavros_msgs/HomePosition.h>
#include <std_msgs/Byte.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <tf/transform_listener.h>
#include "quadrotor_msgs/TrajectoryPoint.h"
#include "offb_ctrl.h"
#include <memory>
// #define ENGINE0 0
// #define TAKEOFF 1
#define TAKEOFF 1
#define WAYPOINT 2
#define MISSION 3
#define LAND 5

enum UavTaskState
{
    kIdle,    // 0
    kReady,   // 1
    kTakeOff, // 2
    kHover,   // 3
    kMission, // 4
    kWaypoint,
    kSwarm, // 5
    kLand,  // 6
};
// #define MISSION 2
// #define HOVER 3
// #define LAND 4
// #define POS_CONTROL 0
// #define VEL_CONTROL 1

// enum VehicleTask
// {
//     kIdle,
//     kReady,
//     kTakeOff,
//     kHover,
//     kMission,
//     kLand,
// };

class OffbNode
{
public:
    OffbNode(ros::NodeHandle &nodeHandle);
    ~OffbNode() = default;

    bool loadTrajectory();
    void clearTrajectory();
    void pubTrajectory();
    // void state_cb(const mavros_msgs::State::ConstPtr &msg);
    void cmd_cb(const std_msgs::Byte::ConstPtr &msg);

    void navGoal_cb(const geometry_msgs::PoseStamped::ConstPtr &msg);

    void uavStateCallBack(const mavros_msgs::State::ConstPtr &msg);

    void uavPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);
    // void missionTimerCallBack(const ros::TimerEvent &);

    bool set_offboard();

    void land();

    void missionTimer(const ros::TimerEvent &);

    void gpsCurrentCallback(const sensor_msgs::NavSatFix::ConstPtr &msg);

    void gpsHomeCallback(const mavros_msgs::HomePosition::ConstPtr &msg);

    void refPoseCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg);

    // void refTrajPtCallBack(const quadrotor_msgs::TrajectoryPoint &msg);

    void refTrajPtCallBack(const quadrotor_msgs::TrajectoryPoint::ConstPtr &msg);

    void trajSpENUCallBack(const quadrotor_msgs::TrajectoryPoint::ConstPtr &msg);

    void vel_callback(const geometry_msgs::TwistStamped::ConstPtr &msg);
    // void calculateYaw(const mavros_msgs::PositionTarget::ConstPtr &msg);

private:
    ros::NodeHandle _nh;
    // VehicleTask _vehicle_task;
    ros::Subscriber uav_state_sub;
    ros::Subscriber cmd_sub;
    ros::Subscriber navGoal_sub;
    ros::Subscriber ref_pose_sub;
    ros::Subscriber trajectory_point_nwu_sub;
    ros::Publisher local_pos_pub;
    ros::ServiceClient arming_client;
    ros::ServiceClient land_client;
    ros::ServiceClient set_mode_client;
    ros::ServiceClient takeoff_client;
    ros::Publisher Position_Setpoint_Pub;
    ros::Subscriber uav_pose_sub;
    ros::Subscriber uav_vel_sub;
    ros::Subscriber uav_gps_cur_sub;
    ros::Subscriber uav_gps_home_sub;
    ros::Subscriber traj_sp_enu_sub;
    ros::Publisher _att_rate_pub;

    geometry_msgs::Point pos_init;
    geometry_msgs::PoseStamped navGoal_sp;

    UavTaskState uavTask;

    std::vector<mavros_msgs::PositionTarget> _traj_list;
    std::string trajectory_location;
    mavros_msgs::State uav_current_state;
    mavros_msgs::CommandBool arm_cmd_;

    geometry_msgs::PoseStamped waypoint_sp;
    geometry_msgs::PoseStamped uav_pose;

    quadrotor_msgs::TrajectoryPoint traj_pt_nwu;
    quadrotor_msgs::TrajectoryPoint traj_sp_enu;

    // geometry_msgs::PoseStamped traj_pt_nwu;

    int _points_id;
    double _missionPeriod;
    double takeoff_height;
    double takeoff_x;
    double takeoff_y;
    ros::Timer mission_timer;

    sensor_msgs::NavSatFix uav_gps_cur;
    mavros_msgs::HomePosition uav_gps_home;

    bool takeoff_flag; // this flag is set if the drone has took off
    bool takeoff_announced;
    bool navGoal_init;
    bool arm_safety_check;
    bool user_give_goal_;
    bool use_px4Ctrl;

    Eigen::Vector3d uav_local_vel_enu; // uav local velocity in ENU
    Eigen::Vector3d uav_local_pos_enu; // uav local position in ENU

    double battery_volt;
    // Control gains (position, velocity, drag)
    Eigen::Vector3d Kpos_, Kvel_, D_;
    Eigen::Vector3d gravity_;
    double Kpos_x_, Kpos_y_, Kpos_z_, Kvel_x_, Kvel_y_, Kvel_z_;
    std::vector<double> thrust_coeff_, volt_coeff_, thrust_original_;

    bool voltage_compensation_;
    bool using_yawTgt;
    double yaw_ref;
    double max_fb_acc_;
    double attctrl_tau_;
    double norm_thrust_offset_;
    double norm_thrust_const_;
    double mass_; // mass of platform

    double m_a_, m_b_, m_c_, volt_k_, volt_b_;
    double throttle_offset_, throttle_limit_;

    Eigen::Vector4d q_des, uav_attitude_q;
    std::string uav_id_;

    std::unique_ptr<offboard_controller::OffbCtrl> pos_ctrl;
};
