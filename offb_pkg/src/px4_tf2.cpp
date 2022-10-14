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

        navGoal_sub = nh.subscribe<geometry_msgs::PoseStamped>(
            "/goal", 1, &px4_tf2::navGoal_cb, this);

        ref_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(
            "/uav/ref_pose/nwu", 1, &px4_tf2::refPoseCallBack, this);

        global_nwu_pose_pub_ =
            nh_.advertise<geometry_msgs::PoseStamped>("/" + m_uav_id_ + "/" + "global_nwu", 10);

        navGoal_enu_pub_ =
            nh_.advertise<geometry_msgs::PoseStamped>("/" + m_uav_id_ + "/" + "navGoal_enu", 10);

        listener_timer_ = nh_.createTimer(ros::Duration(0.05), &px4_tf2::listenerTimerCb, this, false, false);

        m_timer_started_ = false;
        init_map_to_enu_homo_ = false;
        map_to_enu_homo = Eigen::Affine3d::Identity();
    }

    // we publish the transformation between /droneX/local_enu i.e. drone local origin (parent) and /drone0 (child) in poseCallback

    void px4_tf2::poseCallback(const geometry_msgs::PoseStampedConstPtr &msg)
    {
        static tf2_ros::TransformBroadcaster br;
        geometry_msgs::TransformStamped transformStamped;

        transformStamped.header.stamp = ros::Time::now();
        transformStamped.header.frame_id = m_uav_id_ + "_" + "local_enu_origin"; // parent frame
        transformStamped.child_frame_id = m_uav_id_ + "_body";                   // child frame
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

    void px4_tf2::navGoal_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (init_map_to_enu_homo_)
        {
            navGoal_sp = *msg;

            Eigen::Affine3d navGoal_nwu_homo = Eigen::Affine3d::Identity();
            navGoal_nwu_homo.translation() = Eigen::Vector3d(navGoal_sp.pose.position.x,
                                                             navGoal_sp.pose.position.y,
                                                             navGoal_sp.pose.position.z);
            navGoal_nwu_homo.linear() = Eigen::Quaterniond(navGoal_sp.pose.orientation.w,
                                                           navGoal_sp.pose.orientation.x,
                                                           navGoal_sp.pose.orientation.y,
                                                           navGoal_sp.pose.orientation.z)
                                            .toRotationMatrix();
            
            Eigen::Affine3d navGoal_enu_homo;
            navGoal_enu_homo.matrix() = map_to_enu_homo.matrix() * navGoal_nwu_homo.matrix();

            Eigen::Vector3d navGoal_pos_enu(navGoal_enu_homo.translation());

            Eigen::Quaterniond att_sp_enu(navGoal_enu_homo.linear());

            geometry_msgs::PoseStamped navGoal_enu;
            navGoal_enu.header.frame_id = m_uav_id_ + "_" + "local_enu_origin";
            navGoal_enu.header.stamp = ros::Time::now();
            navGoal_enu.pose.position.x = navGoal_pos_enu.x();
            navGoal_enu.pose.position.y = navGoal_pos_enu.y();
            navGoal_enu.pose.position.z = navGoal_pos_enu.z();
            navGoal_enu.pose.orientation.w = att_sp_enu.w();
            navGoal_enu.pose.orientation.x = att_sp_enu.x();
            navGoal_enu.pose.orientation.y = att_sp_enu.y();
            navGoal_enu.pose.orientation.z = att_sp_enu.z();
            navGoal_enu_pub_.publish(navGoal_enu);
        }

        else
        {
            std::cout << "navGoal_cb: transformation matrix not initialized" << std::endl;
        }
    }

    void px4_tf2::refPoseCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        navGoal_sp = *msg;
        std::cout << "refPose received, require converion to from map to local_enu" << std::endl;
        // navGoal_init = true;
    }

    void px4_tf2::listenerTimerCb(const ros::TimerEvent &)
    {
        if (!init_map_to_enu_rot_)
        {
            geometry_msgs::TransformStamped transformStamped;
            try
            {
                transformStamped = tfBuffer.lookupTransform(m_uav_id_ + "_" + "local_enu_origin", "map",
                                                            ros::Time(0));
                // first argument is target frame
                // second argument is source frame
            }
            catch (tf2::TransformException &ex)
            {
                ROS_WARN("%s", ex.what());
                ros::Duration(1.0).sleep();
                goto check_next;
            }

            map_to_enu_rot = Eigen::Quaterniond(transformStamped.transform.rotation.w,
                                                transformStamped.transform.rotation.x,
                                                transformStamped.transform.rotation.y,
                                                transformStamped.transform.rotation.z);

            map_to_enu_trans << transformStamped.transform.translation.x,
                transformStamped.transform.translation.y,
                transformStamped.transform.translation.z;

            map_to_enu_homo.linear() = map_to_enu_rot;
            map_to_enu_homo.translation() = map_to_enu_trans;

            init_map_to_enu_homo_ = true;
        }

    check_next:

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
