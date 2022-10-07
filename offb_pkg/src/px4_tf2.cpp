#include "px4_tf2.h"

// #include <ros/ros.h>
// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2_ros/transform_broadcaster.h>
// #include <tf2_ros/transform_listener.h>
// #include <geometry_msgs/PoseStamped.h>
namespace px4_tf2
{
    px4_tf2::px4_tf2(ros::NodeHandle &nh) : nh_(nh), tfListener(tfBuffer)
    {
        nh_.param<std::string>("uav_id", m_uav_id_, "");

        uav_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>( 
            "/" + m_uav_id_ + "/mavros/local_position/pose", 1, &px4_tf2::poseCallback, this);

        global_nwu_pose_pub_ =
            nh_.advertise<geometry_msgs::PoseStamped>("/" + m_uav_id_ + "/" + "global_nwu", 10);

        listener_timer_ = nh_.createTimer(ros::Duration(0.05), &px4_tf2::listenerTimerCb, this, false, false);

        m_timer_started_ = false;
    }

    // we publish the transformation between /droneX/local_enu i.e. drone local origin (parent) and /drone0 (child) in poseCallback

    void px4_tf2::poseCallback(const geometry_msgs::PoseStampedConstPtr &msg)
    {
        static tf2_ros::TransformBroadcaster br;
        geometry_msgs::TransformStamped transformStamped;

        transformStamped.header.stamp = ros::Time::now();
        transformStamped.header.frame_id = m_uav_id_+ "_" + "local_enu_origin"; // parent frame
        transformStamped.child_frame_id = m_uav_id_ + "_body"; // child frame
        transformStamped.transform.translation.x = msg->pose.position.x;
        transformStamped.transform.translation.y = msg->pose.position.y;
        transformStamped.transform.translation.z = msg->pose.position.z;
        tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);

        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();

        br.sendTransform(transformStamped);

        if (!m_timer_started_)
        {
            listener_timer_.start();
            m_timer_started_ = true;
        }
    }

    void px4_tf2::listenerTimerCb(const ros::TimerEvent &)
    {
        geometry_msgs::TransformStamped transformStamped;
        try
        {
            transformStamped = tfBuffer.lookupTransform("map", m_uav_id_ + "_body",
                                                        ros::Time(0));
                                                        // first argument is target frame
                                                        // second argument is source frame
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
            ros::Duration(1.0).sleep();
            return;
        }

        geometry_msgs::PoseStamped global_nwu_pose;

        global_nwu_pose.header.frame_id = "map";
        global_nwu_pose.header.stamp = transformStamped.header.stamp;
        global_nwu_pose.pose.position.x = transformStamped.transform.translation.x;
        global_nwu_pose.pose.position.y = transformStamped.transform.translation.y;
        global_nwu_pose.pose.position.z = transformStamped.transform.translation.z;
        global_nwu_pose.pose.orientation.w = transformStamped.transform.rotation.w;
        global_nwu_pose.pose.orientation.x = transformStamped.transform.rotation.x;
        global_nwu_pose.pose.orientation.y = transformStamped.transform.rotation.y;
        global_nwu_pose.pose.orientation.z = transformStamped.transform.rotation.z;
        global_nwu_pose_pub_.publish(global_nwu_pose);
    }
}; // namespace px4_tf2
