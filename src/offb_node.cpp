#include "px4_offb/offb_node.h"

OffbNode::OffbNode(ros::NodeHandle &nodeHandle) : _nh(nodeHandle)
{
    _points_id = 0;

    uavTask = kIdle;

    takeoff_flag = false;

    // publish file timer
    _nh.param("missionPeriod", _missionPeriod, 0.2);
    std::cout << "missionPeriod: " << _missionPeriod << std::endl;

    cmd_sub = _nh.subscribe<std_msgs::Byte>("/user_cmd", 1, &OffbNode::cmd_cb, this);

    uav_state_sub = _nh.subscribe<mavros_msgs::State>("/mavros/state", 10, &OffbNode::uavStateCallBack, this);

    local_pos_pub = _nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    // arming_client =
    // _nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");

    arming_client = _nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");

    takeoff_client = _nh.serviceClient<mavros_msgs::CommandTOL>("/mavros/cmd/takeoff");

    land_client = _nh.serviceClient<mavros_msgs::CommandTOL>("/mavros/cmd/land");

    set_mode_client = _nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");

    // set_mode_client =
    // _nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");

    Position_Setpoint_Pub = _nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 10);

    //reference_timer = _nh.createTimer(ros::Duration(0.05), &OffbNode::referenceCallbackTimer, this, false, false);
    reference_timer = _nh.createTimer(ros::Duration(0.05), &OffbNode::referenceCallbackTimer, this, false, false);

    takeoff_timer = _nh.createTimer(ros::Duration(0.05), &OffbNode::takeoffCallbackTimer, this, false, false);
    // get current uav pose
    uav_pose_sub = _nh.subscribe<geometry_msgs::PoseStamped>(
        "/mavros/local_position/pose", 1, &OffbNode::uavPoseCallback, this);

    uav_gps_cur_sub = _nh.subscribe<sensor_msgs::NavSatFix>(
        "/mavros/global_position/global", 1, &OffbNode::gpsCurrentCallback, this);

    uav_gps_home_sub = _nh.subscribe<mavros_msgs::HomePosition>(
        "/mavros/home_position/home", 1, &OffbNode::gpsHomeCallback, this);

    uav_ref_prep_pub = _nh.advertise<mavros_msgs::PositionTarget>("/uav_ref_prep", 1);

    uav_ref_prep_sub = _nh.subscribe<mavros_msgs::PositionTarget>("/uav_ref_prep", 1, &OffbNode::referencePrepCallback, this);

    uav_ref_pub = _nh.advertise<mavros_msgs::PositionTarget>("/uav_ref", 1);

    uav_ref_sub = _nh.subscribe<mavros_msgs::PositionTarget>("/uav_ref", 1, &OffbNode::referenceCallback, this); //this is like current state call back

    _nh.param<std::string>("WP_Location", trajectory_location, "/home/zhengtian/px4_offb/src/px4_offb/param/waypoints");
    std::cout << "WP_Location: " << trajectory_location << std::endl;

    ROS_INFO("Offboard node is ready!");
}

OffbNode::~OffbNode()
{
    // ROS_INFO("Bye task manager~");
}

void OffbNode::gpsHomeCallback(const mavros_msgs::HomePosition::ConstPtr &msg)
{
    uav_gps_home = *msg;
}

void OffbNode::gpsCurrentCallback(const sensor_msgs::NavSatFix::ConstPtr &msg)
{
    uav_gps_cur = *msg;
}

void OffbNode::referencePrepCallback(const mavros_msgs::PositionTarget::ConstPtr &msg)
{
    tgt = *msg;
    ROS_INFO("referencePrepcallback running");
}

void OffbNode::referenceCallback(const mavros_msgs::PositionTarget::ConstPtr &msg)
{
    mavros_msgs::PositionTarget pos_sp;
    pos_sp.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED; // To be converted to NED in setpoint_raw plugin
    pos_sp.position.x = -msg->position.y;
    pos_sp.position.y = msg->position.x;
    pos_sp.position.z = msg->position.z;
    pos_sp.velocity.x = -msg->velocity.y;
    pos_sp.velocity.y = msg->velocity.x;
    pos_sp.velocity.z = msg->velocity.z;
    pos_sp.acceleration_or_force.x = -msg->acceleration_or_force.y;
    pos_sp.acceleration_or_force.y = msg->acceleration_or_force.x;
    pos_sp.acceleration_or_force.z = msg->acceleration_or_force.z;
    Position_Setpoint_Pub.publish(pos_sp);
}

void OffbNode::takeoffCallbackTimer(const ros::TimerEvent &)
{
    // mavros_msgs::PositionTarget takeoff_sp;
    // takeoff_sp.position.x = takeoff_x;
    // takeoff_sp.position.y = takeoff_y;
    // takeoff_sp.position.z = takeoff_height;
    // takeoff_sp.velocity.x = 0;
    // takeoff_sp.velocity.y = 0;
    // takeoff_sp.velocity.z = 0;
    // takeoff_sp.acceleration_or_force.x = 0;
    // takeoff_sp.acceleration_or_force.x = 0;
    // takeoff_sp.acceleration_or_force.x = 0;
    // takeoff_sp.yaw = 0;
    // takeoff_sp.type_mask = 32768;

    // printf("running takeoff\n");
    // Position_Setpoint_Pub.publish(takeoff_sp);

    switch (uavTask)
    {
    case kTakeOff:
    {
        ROS_INFO("Takeoff timer Doing takeoff!");
        // geometry_msgs::PoseStamped pose;
        // pose.pose.position.x = takeoff_x;
        // pose.pose.position.y = takeoff_y;
        // pose.pose.position.z = takeoff_height;

        // local_pos_pub.publish(pose);

        mavros_msgs::PositionTarget pos_sp;
        pos_sp.position.x = takeoff_x;
        pos_sp.position.y = takeoff_y;
        pos_sp.position.z = takeoff_height;

        pos_sp.type_mask = 3576;
        Position_Setpoint_Pub.publish(pos_sp);
        break;
    }

    case kMission:
    {
        mavros_msgs::PositionTarget pos_sp;
        ROS_INFO("Takeoff timer Doing mission!");
        if (_points_id <= _traj_list.size() - 1)
        {
            std::cout << "Go to next ref: " << _points_id << std::endl;
            std::cout << "Position: " << _traj_list[_points_id].pos.x << " " << _traj_list[_points_id].pos.y << " " << _traj_list[_points_id].pos.z << std::endl;
            std::cout << "Velocity: " << _traj_list[_points_id].vel.x << " " << _traj_list[_points_id].vel.y << " " << _traj_list[_points_id].vel.z << std::endl;
            std::cout << "Aceleration: " << _traj_list[_points_id].acc.x << " " << _traj_list[_points_id].acc.y << " " << _traj_list[_points_id].acc.z << std::endl;
            pos_sp.position.x = _traj_list[_points_id].pos.x;
            pos_sp.position.y = _traj_list[_points_id].pos.y;
            pos_sp.position.z = _traj_list[_points_id].pos.z;
            pos_sp.velocity.x = _traj_list[_points_id].vel.x;
            pos_sp.velocity.y = _traj_list[_points_id].vel.y;
            pos_sp.velocity.z = _traj_list[_points_id].vel.z;
            pos_sp.acceleration_or_force.x = _traj_list[_points_id].acc.x;
            pos_sp.acceleration_or_force.y = _traj_list[_points_id].acc.y;
            pos_sp.acceleration_or_force.z = _traj_list[_points_id].acc.z;

            _points_id++;
        }
        else if (_points_id == _traj_list.size())
        {
            _points_id--;
            std::cout << "Hold last ref: " << _points_id << std::endl;
            std::cout << "Position: " << _traj_list[_points_id].pos.x << " " << _traj_list[_points_id].pos.y << " " << _traj_list[_points_id].pos.z << std::endl;
            std::cout << "Velocity: " << _traj_list[_points_id].vel.x << " " << _traj_list[_points_id].vel.y << " " << _traj_list[_points_id].vel.z << std::endl;
            std::cout << "Aceleration: " << _traj_list[_points_id].acc.x << " " << _traj_list[_points_id].acc.y << " " << _traj_list[_points_id].acc.z << std::endl;
            pos_sp.position.x = _traj_list[_points_id].pos.x;
            pos_sp.position.y = _traj_list[_points_id].pos.y;
            pos_sp.position.z = _traj_list[_points_id].pos.z;
            pos_sp.velocity.x = _traj_list[_points_id].vel.x;
            pos_sp.velocity.y = _traj_list[_points_id].vel.y;
            pos_sp.velocity.z = _traj_list[_points_id].vel.z;
            pos_sp.acceleration_or_force.x = _traj_list[_points_id].acc.x;
            pos_sp.acceleration_or_force.y = _traj_list[_points_id].acc.y;
            pos_sp.acceleration_or_force.z = _traj_list[_points_id].acc.z;
            _points_id++;
        }

        pos_sp.yaw = 0;
        pos_sp.type_mask = 32768;
        Position_Setpoint_Pub.publish(pos_sp);
    }

    default:
        break;
    }
}

void OffbNode::referenceCallbackTimer(const ros::TimerEvent &)
{
    // mavros_msgs::PositionTarget pos_sp;
    // pos_sp.position.x = -uav_pose.pose.position.y;
    // pos_sp.position.y = uav_pose.pose.position.x;
    // pos_sp.position.z = uav_pose.pose.position.z;
    // pos_sp.velocity.x = 0;
    // pos_sp.velocity.y = 0;
    // pos_sp.velocity.z = 0;
    // pos_sp.acceleration_or_force.x = 0;
    // pos_sp.acceleration_or_force.y = 0;
    // pos_sp.acceleration_or_force.z = 0;
    // pos_sp.yaw = 0;
    //uav_ref_pub.publish(tgt);
    //ROS_INFO("ref cb timer running");
    mavros_msgs::PositionTarget pos_sp;
    if (uavTask == kMission)
    {
        ROS_INFO("Doing mission!");
        if (_points_id <= _traj_list.size() - 1)
        {
            std::cout << "Go to next ref: " << _points_id << std::endl;
            std::cout << "Position: " << _traj_list[_points_id].pos.x << " " << _traj_list[_points_id].pos.y << " " << _traj_list[_points_id].pos.z << std::endl;
            std::cout << "Velocity: " << _traj_list[_points_id].vel.x << " " << _traj_list[_points_id].vel.y << " " << _traj_list[_points_id].vel.z << std::endl;
            std::cout << "Aceleration: " << _traj_list[_points_id].acc.x << " " << _traj_list[_points_id].acc.y << " " << _traj_list[_points_id].acc.z << std::endl;
            pos_sp.position.x = _traj_list[_points_id].pos.x;
            pos_sp.position.y = _traj_list[_points_id].pos.y;
            pos_sp.position.z = _traj_list[_points_id].pos.z;
            pos_sp.velocity.x = _traj_list[_points_id].vel.x;
            pos_sp.velocity.y = _traj_list[_points_id].vel.y;
            pos_sp.velocity.z = _traj_list[_points_id].vel.z;
            pos_sp.acceleration_or_force.x = _traj_list[_points_id].acc.x;
            pos_sp.acceleration_or_force.y = _traj_list[_points_id].acc.y;
            pos_sp.acceleration_or_force.z = _traj_list[_points_id].acc.z;

            _points_id++;
        }
        else if (_points_id == _traj_list.size())
        {
            _points_id--;
            std::cout << "Hold last ref: " << _points_id << std::endl;
            std::cout << "Position: " << _traj_list[_points_id].pos.x << " " << _traj_list[_points_id].pos.y << " " << _traj_list[_points_id].pos.z << std::endl;
            std::cout << "Velocity: " << _traj_list[_points_id].vel.x << " " << _traj_list[_points_id].vel.y << " " << _traj_list[_points_id].vel.z << std::endl;
            std::cout << "Aceleration: " << _traj_list[_points_id].acc.x << " " << _traj_list[_points_id].acc.y << " " << _traj_list[_points_id].acc.z << std::endl;
            pos_sp.position.x = _traj_list[_points_id].pos.x;
            pos_sp.position.y = _traj_list[_points_id].pos.y;
            pos_sp.position.z = _traj_list[_points_id].pos.z;
            pos_sp.velocity.x = _traj_list[_points_id].vel.x;
            pos_sp.velocity.y = _traj_list[_points_id].vel.y;
            pos_sp.velocity.z = _traj_list[_points_id].vel.z;
            pos_sp.acceleration_or_force.x = _traj_list[_points_id].acc.x;
            pos_sp.acceleration_or_force.y = _traj_list[_points_id].acc.y;
            pos_sp.acceleration_or_force.z = _traj_list[_points_id].acc.z;
            _points_id++;
        }
    }

    pos_sp.yaw = 0;
    pos_sp.type_mask = 32768;
    Position_Setpoint_Pub.publish(pos_sp);
}

// bool OffbNode::takeoff()
// {
//     ros::Rate rate(20.0);

//     if (!uav_current_state.armed)
//     {
//         ROS_WARN("Takeoff rejected. Arm first.");
//         return false;
//     }

//     if (uav_pose.pose.position.z > 1.0)
//     {
//         ROS_WARN("Takeoff rejected. Already took off.");
//         return false;
//     }

//     double home_alt = uav_gps_home.geo.altitude;
//     double home_lon = uav_gps_home.geo.longitude;
//     double home_lat = uav_gps_home.geo.latitude;

//     mavros_msgs::CommandTOL takeoff_cmd;

//     takeoff_cmd.request.altitude = home_alt - 41; // this is hardcoded ToDo: figure out why the altitude cannot be set accurately
//     takeoff_cmd.request.longitude = home_lon;
//     takeoff_cmd.request.latitude = home_lat;

//     ros::Time takeoff_last_request = ros::Time::now();

//     printf("take off height is why %f", takeoff_cmd.request.altitude);

//     //ROS_DEBUG("set lat: %f, lon: %f, alt: %f", home_lat, home_lon, home_alt);
//     printf("set lat: %f, lon: %f, alt: %f", home_lat, home_lon, home_alt);

//     while (!(takeoff_client.call(takeoff_cmd)) && takeoff_cmd.response.success)
//     {
//         ros::spinOnce();
//         rate.sleep();
//         if (ros::Time::now() - takeoff_last_request > ros::Duration(3.0))
//         {
//             ROS_WARN("[TIMEOUT] Takeoff service call failed");
//             return false;
//         }
//     }

//     //check if height reached, else report takeoff failed

//     ROS_INFO("Vehicle takeoff.");
//     takeoff_flag = true;
//     return true;
// }

void OffbNode::land()
{
    ros::Rate rate(20.0);
    double curr_alt = uav_gps_cur.altitude;
    double curr_lon = uav_gps_cur.longitude;
    double curr_lat = uav_gps_cur.latitude;
    double home_alt = uav_gps_home.geo.altitude;

    mavros_msgs::CommandTOL landing_cmd;
    //landing_cmd.request.altitude = curr_alt - (curr_alt - home_alt) + 0.5;
    landing_cmd.request.altitude = home_alt;
    landing_cmd.request.longitude = curr_lon;
    landing_cmd.request.latitude = curr_lat;
    while (!(land_client.call(landing_cmd)) &&
           landing_cmd.response.success)
    {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Vehicle landing...");
    // while (lpos.pose.position.z > 0.1)
    // {
    //     ros::spinOnce();
    //     rate.sleep();
    // }
    ROS_INFO("Vehicle landed.");
    return;
}

void OffbNode::uavStateCallBack(const mavros_msgs::State::ConstPtr &msg)
{
    uav_current_state = *msg;
}

void OffbNode::uavPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    uav_pose = *msg;
}

bool OffbNode::takeoff()
{
    ros::Rate rate(20.0);
    // mavros_msgs::PositionTarget takeoff_sp;
    // takeoff_sp.position.x = uav_pose.pose.position.x;
    // takeoff_sp.position.y = uav_pose.pose.position.y;
    // takeoff_sp.position.z = uav_pose.pose.position.z + 1.0;
    // takeoff_sp.velocity.x = 0;
    // takeoff_sp.velocity.y = 0;
    // takeoff_sp.velocity.z = 0;
    // takeoff_sp.acceleration_or_force.x = 0;
    // takeoff_sp.acceleration_or_force.x = 0;
    // takeoff_sp.acceleration_or_force.x = 0;
    // takeoff_sp.yaw = 0;

    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = 1;

    for (int i = 100; ros::ok() && i > 0; --i)
    {
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    printf("running takeoff\n");
    takeoff_flag = true;

    while (ros::ok() && uavTask == kTakeOff)
    {
        printf("pubbing?\n");
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
        //Position_Setpoint_Pub.publish(takeoff_sp);
    }
}

bool OffbNode::set_offboard()
{
    ros::Rate rate(20.0);
    ros::Time last_request = ros::Time::now();
    mavros_msgs::PositionTarget init_sp;
    init_sp.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED; // To be converted to NED in setpoint_raw plugin
    //FIXME check
    init_sp.position.x = uav_pose.pose.position.x;
    init_sp.position.y = uav_pose.pose.position.y;
    init_sp.position.z = uav_pose.pose.position.z;
    init_sp.velocity.x = 0;
    init_sp.velocity.y = 0;
    init_sp.velocity.z = 0;
    init_sp.acceleration_or_force.x = 0;
    init_sp.acceleration_or_force.y = 0;
    init_sp.acceleration_or_force.z = 0;
    init_sp.yaw = 0;

    //send a few setpoints before starting
    for (int i = 100; ros::ok() && i > 0; --i)
    {
        Position_Setpoint_Pub.publish(init_sp);
        ros::spinOnce();
        rate.sleep();
    }

    // geometry_msgs::PoseStamped pose;
    // pose.pose.position.x = 0;
    // pose.pose.position.y = 0;
    // pose.pose.position.z = 2;

    // //send a few setpoints before starting
    // for (int i = 100; ros::ok() && i > 0; --i)
    // {
    //     local_pos_pub.publish(pose);
    //     ros::spinOnce();
    //     rate.sleep();
    // }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    bool is_mode_ready = false;
    last_request = ros::Time::now();

    while (!is_mode_ready)
    {
        if (uav_current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(1.0)))
        {
            ROS_INFO("Try set offboard");
            if (set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent)
            {
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        }
        else
        {
            is_mode_ready = (uav_current_state.mode == "OFFBOARD");
        }
        Position_Setpoint_Pub.publish(init_sp);
        ros::spinOnce();
        rate.sleep();
    }

    if (is_mode_ready)
    {
        ROS_INFO("Offboard mode activated!");
    }

    // while (ros::ok())
    // {
    //     if (uav_current_state.mode != "OFFBOARD" &&
    //         (ros::Time::now() - last_request > ros::Duration(1.0)))
    //     {
    //         ROS_INFO("ros ok, try to set offboard");
    //         if (set_mode_client.call(offb_set_mode))
    //         {
    //             ROS_INFO("set mode client offb send");
    //         }

    //         if (offb_set_mode.response.mode_sent)
    //         {
    //             ROS_INFO("set mode got response");
    //         }
    //         if (set_mode_client.call(offb_set_mode) &&
    //             offb_set_mode.response.mode_sent)
    //         {
    //             ROS_INFO("Offboard enabled");
    //         }

    //         last_request = ros::Time::now();
    //     }
    //     // else
    //     // {
    //     //     if (!uav_current_state.armed &&
    //     //         (ros::Time::now() - last_request > ros::Duration(2.0)))
    //     //     {
    //     //         ROS_INFO("ros ok, try to arm");
    //     //         if (arming_client.call(arm_cmd) &&
    //     //             arm_cmd.response.success)
    //     //         {
    //     //             ROS_INFO("Vehicle armed");
    //     //         }
    //     //         last_request = ros::Time::now();
    //     //     }
    //     // }

    //     //Position_Setpoint_Pub.publish(init_sp);
    //     local_pos_pub.publish(pose);

    //     ros::spinOnce();
    //     rate.sleep();
    // }

    return is_mode_ready;
}

void OffbNode::cmd_cb(const std_msgs::Byte::ConstPtr &msg)
{
    int cmd = msg->data;
    ROS_INFO("haha, cmd received");
    switch (cmd)
    {
    case TAKEOFF:
    {
        ROS_INFO("TAKEOFF command received!");
        uavTask = kTakeOff;
        takeoff_height = uav_pose.pose.position.z + 1.0;
        takeoff_x = uav_pose.pose.position.x;
        takeoff_y = uav_pose.pose.position.y;
        if (set_offboard())
        {
            ROS_INFO("offboard is setr going to run takeoff");
            //takeoff();
            takeoff_timer.start();
            ROS_INFO("takeoff timer started!");
            takeoff_flag = true;
        }
        // if (takeoff())
        // {
        //     ROS_INFO("UAV is taking off!");
        // }
        break;
    }

    case MISSION:
    {
        //uavTask = kIdle;
        if (!takeoff_flag)
        {
            ROS_ERROR("Vehilce has not taken off, please issue takeoff command first.");
            break;
        }
        ROS_INFO("command received!");
        ROS_INFO("Loading Trajectory...");
        if (loadTrajectory())
        {

            ROS_INFO("trajectory is loaded.");
            uavTask = kMission;

            //reference_timer.start();
            //takeoff_timer.stop();
            // if (set_offboard())
            // {
            //     reference_timer.start();
            // }

            //pubTrajectory();
        }
        else
        {
            break;
        }

        break;
    }

    case LAND:
    {
        ROS_INFO("command received!");
        uavTask = kLand;
        land();
        ROS_INFO("UAV is landing");
        takeoff_flag = false;
        break;
    }

    default:
        break;
    }
    //ROS_INFO("command received!");
    // if (loadTrajectory())
    // {
    //     pubTrajectory();
    // }
}

bool OffbNode::loadTrajectory()
{
    std::ifstream infile(trajectory_location);
    std::string line;
    //clearTrajectory();
    bool succeed = false;
    ROS_INFO("Loading from trajectory reference file");

    std::ifstream f(trajectory_location.c_str());
    if (!f.good())
    {
        ROS_ERROR("Wrong file name!");
        return false;
    }

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        double _1, _2, _3, _4, _5, _6, _7, _8, _9;
        if (!(iss >> _1 >> _2 >> _3 >> _4 >> _5 >> _6 >> _7 >> _8 >>
              _9))
        {
            ROS_ERROR("The data size is not correct!");
            return false;
        } // error
        common_msgs::state p;
        p.pos.x = _1;
        p.pos.y = _2;
        p.pos.z = _3;
        p.vel.x = _4;
        p.vel.y = _5;
        p.vel.z = _6;
        p.acc.x = _7;
        p.acc.y = _8;
        p.acc.z = _9;
        _traj_list.push_back(p);
        // printf("size of the list is %lu\n", _traj_list.size());
        std::cout << "Position : " << _traj_list[_points_id].pos.x << " " << _traj_list[_points_id].pos.y << " " << _traj_list[_points_id].pos.z << " "
                  << "Velocity : " << _traj_list[_points_id].vel.x << " " << _traj_list[_points_id].vel.y << " " << _traj_list[_points_id].vel.z << " "
                  << "Accel : " << _traj_list[_points_id].acc.x << " " << _traj_list[_points_id].acc.y << " " << _traj_list[_points_id].acc.z << " "
                  << std::endl;

        _points_id++;
    }
    printf("size of the list is %lu\n", _traj_list.size());
    ROS_INFO("trajctory reference has been loaded.");
    succeed = true;
    _points_id = 0;
    return true;
}

void OffbNode::clearTrajectory()
{
    _traj_list.clear();
    ROS_INFO("Clear Mission");
}

void OffbNode::pubTrajectory()
{
    mavros_msgs::PositionTarget pos_sp;
    while (ros::ok && _points_id <= _traj_list.size() - 1)
    {
        std::cout << "Goto next ref: " << _points_id << std::endl;
        std::cout << "Position : " << _traj_list[_points_id].pos.x << " " << _traj_list[_points_id].pos.y << " " << _traj_list[_points_id].pos.z << " "
                  << "Velocity : " << _traj_list[_points_id].vel.x << " " << _traj_list[_points_id].vel.y << " " << _traj_list[_points_id].vel.z << " "
                  << "Accel : " << _traj_list[_points_id].acc.x << " " << _traj_list[_points_id].acc.y << " " << _traj_list[_points_id].acc.z << " "
                  << std::endl;
        // NWU frame
        pos_sp.position.x = -_traj_list[_points_id].pos.y;
        pos_sp.position.y = _traj_list[_points_id].pos.x;
        pos_sp.position.z = _traj_list[_points_id].pos.z;
        pos_sp.velocity.x = -_traj_list[_points_id].vel.y;
        pos_sp.velocity.y = _traj_list[_points_id].vel.x;
        pos_sp.velocity.z = _traj_list[_points_id].vel.z;
        pos_sp.acceleration_or_force.x = -_traj_list[_points_id].acc.y;
        pos_sp.acceleration_or_force.y = _traj_list[_points_id].acc.x;
        pos_sp.acceleration_or_force.z = _traj_list[_points_id].acc.z;

        _points_id++;
        pos_sp.yaw = 0;
        // typemask 32768 will request rpt controller in px4
        pos_sp.type_mask = 32768;
        Position_Setpoint_Pub.publish(pos_sp);
    }
    while (ros::ok && _points_id == _traj_list.size())
    {
        _points_id--;
        std::cout << "hold last ref: " << _points_id - 1 << std::endl;
        std::cout << "Position : " << _traj_list[_points_id].pos.x << " " << _traj_list[_points_id].pos.y << " " << _traj_list[_points_id].pos.z << " "
                  << "Velocity : " << _traj_list[_points_id].vel.x << " " << _traj_list[_points_id].vel.y << " " << _traj_list[_points_id].vel.z << " "
                  << "Accel : " << _traj_list[_points_id].acc.x << " " << _traj_list[_points_id].acc.y << " " << _traj_list[_points_id].acc.z << " "
                  << std::endl;
        // NWU frame
        pos_sp.position.x = -_traj_list[_points_id].pos.y;
        pos_sp.position.y = _traj_list[_points_id].pos.x;
        pos_sp.position.z = _traj_list[_points_id].pos.z;
        pos_sp.velocity.x = -_traj_list[_points_id].vel.y;
        pos_sp.velocity.y = _traj_list[_points_id].vel.x;
        pos_sp.velocity.z = _traj_list[_points_id].vel.z;
        pos_sp.acceleration_or_force.x = -_traj_list[_points_id].acc.y;
        pos_sp.acceleration_or_force.y = _traj_list[_points_id].acc.x;
        pos_sp.acceleration_or_force.z = _traj_list[_points_id].acc.z;
        _points_id++;
        pos_sp.yaw = 0;
        // typemask 32768 will request rpt controller in px4
        pos_sp.type_mask = 32768;
        Position_Setpoint_Pub.publish(pos_sp);
    }
}