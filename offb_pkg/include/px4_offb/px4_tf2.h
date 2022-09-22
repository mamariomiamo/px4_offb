// ros node to broadcast uav fixed frame w.r.t local origin using tf2
// and also publish position in global nwu frame

#pragma once
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <string>

#include <geometry_msgs/PoseStamped.h>
#include <iostream>

namespace px4_tf2
{
    class px4_tf2
    {
    public:
        px4_tf2(ros::NodeHandle &nh);
        ~px4_tf2() = default;

        ros::NodeHandle nh_;

        ros::Subscriber uav_pose_sub_;

        ros::Publisher global_nwu_pose_pub_;

        ros::Timer listener_timer_;

        bool m_timer_started_;

        std::string m_uav_id_;

        tf2_ros::Buffer tfBuffer;
        tf2_ros::TransformListener tfListener;

        void poseCallback(const geometry_msgs::PoseStampedConstPtr & msg);

        void listenerTimerCb(const ros::TimerEvent &);

    };

}; // namespace px4_tf2