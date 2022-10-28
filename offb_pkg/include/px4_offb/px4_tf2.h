// ros node to broadcast uav fixed frame w.r.t local origin using tf2
// and also publish position in global nwu frame

#pragma once
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <string>
#include "quadrotor_msgs/TrajectoryPoint.h"
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <iostream>
#include <Eigen/Dense>

namespace px4_tf2
{
    class px4_tf2
    {
    public:
        px4_tf2(ros::NodeHandle &nh);
        ~px4_tf2() = default;

        ros::NodeHandle nh_;

        ros::Subscriber uav_pose_sub_, navGoal_sub, ref_pose_sub, traj_nwu_sub;

        ros::Publisher global_nwu_pose_pub_, global_nwu_odom_pub_, navGoal_enu_pub_, traj_enu_pub;

        ros::Timer listener_timer_;

        bool m_timer_started_, init_map_to_enu_homo_; // homo as in homogeneous

        std::string m_uav_id_;

        geometry_msgs::PoseStamped navGoal_sp;
        quadrotor_msgs::TrajectoryPoint traj_sp_nwu, traj_sp_enu;

        tf2_ros::Buffer tfBuffer;
        tf2_ros::TransformListener tfListener;

        void poseCallback(const geometry_msgs::PoseStampedConstPtr & msg);

        void navGoal_cb(const geometry_msgs::PoseStamped::ConstPtr &msg);

        void refPoseCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg);

        void trajSetpointCallBack(const quadrotor_msgs::TrajectoryPoint::ConstPtr &msg);

        void listenerTimerCb(const ros::TimerEvent &);

        Eigen::Matrix3d map_to_enu_rot;
        Eigen::Vector3d map_to_enu_trans;
        Eigen::Affine3d map_to_enu_homo;

    };

}; // namespace px4_tf2