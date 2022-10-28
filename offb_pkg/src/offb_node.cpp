#include "px4_offb/offb_node.h"

OffbNode::OffbNode(ros::NodeHandle &nodeHandle) : _nh(nodeHandle)
{
  _points_id = 0;

  uavTask = kIdle;

  takeoff_flag = false;

  takeoff_announced = false;

  navGoal_init = false;

  // publish file timer
  _nh.param("missionPeriod", _missionPeriod, 0.2);
  _nh.param("user_give_goal", user_give_goal_, true);
  _nh.param("use_px4Ctrl", use_px4Ctrl, true);
  _nh.param<std::string>("uav_id", uav_id_, "");

  /** @brief Control gains**/
  _nh.param<double>("gains/p_x", Kpos_x_, 8.0);
  _nh.param<double>("gains/p_y", Kpos_y_, 8.0);
  _nh.param<double>("gains/p_z", Kpos_z_, 10.0);
  _nh.param<double>("gains/v_x", Kvel_x_, 1.5);
  _nh.param<double>("gains/v_y", Kvel_y_, 1.5);
  _nh.param<double>("gains/v_z", Kvel_z_, 3.3);

  _nh.param<bool>("use_yawTarget", using_yawTgt, false);
  _nh.param<double>("max_acc", max_fb_acc_, 9.0);

  _nh.param<double>("attctrl_constant", attctrl_tau_, 0.1);
  _nh.param<double>("normalizedthrust_offset", norm_thrust_offset_, 0.1);
  _nh.param<double>("normalizedthrust_constant", norm_thrust_const_, 0.05);
  _nh.param<bool>("voltage_compensation", voltage_compensation_, false);

  _nh.param<double>("m_a", m_a_, 202.33);
  _nh.param<double>("m_b", m_b_, 145.56);
  _nh.param<double>("m_c", m_c_, -8.0219);

  _nh.param<double>("volt_k", volt_k_, -0.1088);
  _nh.param<double>("volt_b", volt_b_, 2.1964);

  _nh.param<double>("throttle_offset", throttle_offset_, 0.06);
  _nh.param<double>("throttle_limit", throttle_limit_, 1);

  _nh.param<double>("mass", mass_, 195.5);

  _nh.param<std::string>("WP_Location", trajectory_location, "/home/catkin_ws/src/px4_offb/param/waypoints.txt");
  std::cout << "WP_Location: " << trajectory_location << std::endl;

  _nh.param<bool>("arm_safety_check", arm_safety_check, true);

  Kpos_ << -Kpos_x_, -Kpos_y_, -Kpos_z_;
  Kvel_ << -Kvel_x_, -Kvel_y_, -Kvel_z_;

  thrust_coeff_ = {m_a_, m_b_, m_c_};
  volt_coeff_ = {volt_k_, volt_b_};
  thrust_original_ = {norm_thrust_const_, norm_thrust_offset_};

  gravity_ << 0.0, 0.0, -9.8;

  std::cout << "missionPeriod: " << _missionPeriod << std::endl;

  cmd_sub = _nh.subscribe<std_msgs::Byte>("/user_cmd", 1, &OffbNode::cmd_cb, this);

  uav_state_sub = _nh.subscribe<mavros_msgs::State>("/" + uav_id_ + "/mavros/state", 10, &OffbNode::uavStateCallBack, this);

  ref_pose_sub = _nh.subscribe<geometry_msgs::PoseStamped>("/uav/ref_pose/nwu", 1, &OffbNode::refPoseCallBack, this);

  trajectory_point_nwu_sub = _nh.subscribe<quadrotor_msgs::TrajectoryPoint>("/uav/trajectory_point/nwu", 20, &OffbNode::refTrajPtCallBack, this);

  traj_sp_enu_sub = _nh.subscribe<quadrotor_msgs::TrajectoryPoint>("/" + uav_id_ + "/" + "traj_sp_enu", 20, &OffbNode::trajSpENUCallBack, this);

  uav_vel_sub = _nh.subscribe<geometry_msgs::TwistStamped>(
      "/" + uav_id_ + "/mavros/local_position/velocity_local", 1, &OffbNode::vel_callback, this);

  if (user_give_goal_)
  {
    // navGoal_sub = _nh.subscribe<geometry_msgs::PoseStamped>("/goal", 1, &OffbNode::navGoal_cb, this);
    navGoal_sub = _nh.subscribe<geometry_msgs::PoseStamped>("/" + uav_id_ + "/" + "navGoal_enu",
                                                            1, &OffbNode::navGoal_cb, this);
  }
  else
  {
    ref_pose_sub = _nh.subscribe<geometry_msgs::PoseStamped>("/uav/ref_pose/nwu", 1, &OffbNode::refPoseCallBack, this);
  }

  local_pos_pub = _nh.advertise<geometry_msgs::PoseStamped>("/" + uav_id_ + "/mavros/setpoint_position/local", 10);

  _att_rate_pub = _nh.advertise<mavros_msgs::AttitudeTarget>("/" + uav_id_ + "/mavros/setpoint_raw/attitude", 10);

  arming_client = _nh.serviceClient<mavros_msgs::CommandBool>("/" + uav_id_ + "/mavros/cmd/arming");

  takeoff_client = _nh.serviceClient<mavros_msgs::CommandTOL>("/" + uav_id_ + "/mavros/cmd/takeoff");

  land_client = _nh.serviceClient<mavros_msgs::CommandTOL>("/" + uav_id_ + "/mavros/cmd/land");

  set_mode_client = _nh.serviceClient<mavros_msgs::SetMode>("/" + uav_id_ + "/mavros/set_mode");

  Position_Setpoint_Pub = _nh.advertise<mavros_msgs::PositionTarget>("/" + uav_id_ + "/mavros/setpoint_raw/local", 10); // Publisher that publishes control setpoints to mavros

  mission_timer = _nh.createTimer(ros::Duration(0.05), &OffbNode::missionTimer, this, false, false);
  // get current uav pose
  uav_pose_sub = _nh.subscribe<geometry_msgs::PoseStamped>(
      "/" + uav_id_ + "/mavros/local_position/pose", 1, &OffbNode::uavPoseCallback, this);

  uav_gps_cur_sub = _nh.subscribe<sensor_msgs::NavSatFix>(
      "/" + uav_id_ + "/mavros/global_position/global", 1, &OffbNode::gpsCurrentCallback, this);

  uav_gps_home_sub = _nh.subscribe<mavros_msgs::HomePosition>(
      "/" + uav_id_ + "/mavros/home_position/home", 1, &OffbNode::gpsHomeCallback, this);

  if (!use_px4Ctrl)
  {
    // std::unique_ptr<offboard_controller::OffbCtrl> pos_ctrl(

    pos_ctrl = std::make_unique<offboard_controller::OffbCtrl>(
        mass_,
        Kpos_,
        Kvel_,
        thrust_coeff_,
        volt_coeff_,
        max_fb_acc_,
        throttle_offset_,
        throttle_limit_,
        thrust_original_);
  }
  ROS_INFO("Offboard node is ready!");
  std::cout << "for uav " << uav_id_ << std::endl;
}

// OffbNode::~OffbNode()
// {
// }

void OffbNode::refPoseCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
  navGoal_sp = *msg;
  navGoal_init = true;
}

void OffbNode::navGoal_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
  navGoal_sp = *msg;
  navGoal_init = true;
}

void OffbNode::gpsHomeCallback(const mavros_msgs::HomePosition::ConstPtr &msg)
{
  uav_gps_home = *msg;
}

void OffbNode::gpsCurrentCallback(const sensor_msgs::NavSatFix::ConstPtr &msg)
{
  uav_gps_cur = *msg;
}

void OffbNode::missionTimer(const ros::TimerEvent &)
{

  switch (uavTask)
  {
  case kTakeOff:
  {
    if (!takeoff_announced)
    {
      ROS_INFO("Mission timer Doing takeoff!");

      std::cout << "Current Altitude is: " << uav_pose.pose.position.z << std::endl;

      if (abs(uav_pose.pose.position.z - takeoff_height) < 0.01)
      {
        ROS_INFO("Takeoff Complete");
        takeoff_announced = true;
      }
    }

    mavros_msgs::PositionTarget pos_sp;
    pos_sp.position.x = takeoff_x;
    pos_sp.position.y = takeoff_y;
    pos_sp.position.z = takeoff_height;
    pos_sp.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;

    pos_sp.type_mask = 3576;
    Position_Setpoint_Pub.publish(pos_sp);
    break;
  }

  case kWaypoint:
  {
    // if (use_px4Ctrl)
    // {
    if (navGoal_init)
    {

      if (user_give_goal_)
      {
        local_pos_pub.publish(navGoal_sp);
        break;
      }

      else
      {
        if (use_px4Ctrl)
        {
          waypoint_sp.pose.position.x = traj_sp_enu.position.x; // take this as position setpoint in local enu frame
          waypoint_sp.pose.position.y = traj_sp_enu.position.y;
          waypoint_sp.pose.position.z = traj_sp_enu.position.z;
          tf2::Quaternion traj_quat;
          traj_quat.setRPY(0, 0, traj_sp_enu.heading);
          waypoint_sp.pose.orientation.w = traj_quat.getW();
          waypoint_sp.pose.orientation.x = traj_quat.getX();
          waypoint_sp.pose.orientation.y = traj_quat.getY();
          waypoint_sp.pose.orientation.z = traj_quat.getZ();
          local_pos_pub.publish(waypoint_sp);
          break;
        }

        else
        {
          Eigen::Vector3d p_ref_enu, v_ref_enu, a_ref_enu;
          Eigen::Vector4d cmdBodyRate_; //{wx, wy, wz, Thrust}

          tf::vectorMsgToEigen(traj_sp_enu.position, p_ref_enu);
          tf::vectorMsgToEigen(traj_sp_enu.velocity, v_ref_enu);
          tf::vectorMsgToEigen(traj_sp_enu.acceleration, a_ref_enu);

          double cmd_yaw = traj_sp_enu.heading;
          // yaw_ref = (float)constrain_between_180(cmd_yaw); // fixme
          yaw_ref = cmd_yaw;
          // reference attitude in quaternion
          Eigen::Vector4d q_ref = helper::acc2quaternion(a_ref_enu - gravity_, yaw_ref);
          Eigen::Matrix3d R_ref = helper::quat2RotMatrix(q_ref);

          Eigen::Vector3d pos_error = uav_local_pos_enu - p_ref_enu;
          Eigen::Vector3d vel_error = uav_local_vel_enu - v_ref_enu;

          Eigen::Vector3d a_des = pos_ctrl->calDesiredAcceleration(pos_error, vel_error, a_ref_enu);
          q_des = pos_ctrl->calDesiredAttitude(a_des, yaw_ref);
          cmdBodyRate_(3) = pos_ctrl->calDesiredThrottle(a_des, uav_attitude_q, battery_volt, voltage_compensation_);

          mavros_msgs::AttitudeTarget msg;
          msg.header.stamp = ros::Time::now();
          msg.header.frame_id = uav_id_ + "_body";
          msg.body_rate.x = cmdBodyRate_(0);
          msg.body_rate.y = cmdBodyRate_(1);
          msg.body_rate.z = cmdBodyRate_(2);
          msg.type_mask = 7; // Ignore orientation messages (128); Ignore body rate messages (7)
          msg.orientation.w = q_des(0);
          msg.orientation.x = q_des(1);
          msg.orientation.y = q_des(2);
          msg.orientation.z = q_des(3);
          msg.thrust = cmdBodyRate_(3);
          _att_rate_pub.publish(msg);
          // std::cout << msg.thrust << std::endl;
          break;
        }
      }
    }
    else // no user input nav goal, just use takeoff pose
    {
      waypoint_sp.pose.position.x = takeoff_x;
      waypoint_sp.pose.position.y = takeoff_y;
      waypoint_sp.pose.position.z = takeoff_height;
      local_pos_pub.publish(waypoint_sp);
      break;
    }
    // }

    // else
    // {
    //     // use offoard position mode here
    //     break;
    // }
  }

  case kMission:
  {
    mavros_msgs::PositionTarget pos_sp;
    ROS_INFO("Mission timer Doing mission!");
    if (_points_id <= _traj_list.size() - 1)
    {
      std::cout << "Go to next ref: " << _points_id << std::endl;
      std::cout << "Position: " << _traj_list[_points_id].position.x << " " << _traj_list[_points_id].position.y << " " << _traj_list[_points_id].position.z << std::endl;
      std::cout << "Velocity: " << _traj_list[_points_id].velocity.x << " " << _traj_list[_points_id].velocity.y << " " << _traj_list[_points_id].velocity.z << std::endl;
      std::cout << "Aceleration: " << _traj_list[_points_id].acceleration_or_force.x << " " << _traj_list[_points_id].acceleration_or_force.y << " " << _traj_list[_points_id].acceleration_or_force.z << std::endl;
      pos_sp.position.x = _traj_list[_points_id].position.x;
      pos_sp.position.y = _traj_list[_points_id].position.y;
      pos_sp.position.z = _traj_list[_points_id].position.z;
      pos_sp.velocity.x = _traj_list[_points_id].velocity.x;
      pos_sp.velocity.y = _traj_list[_points_id].velocity.y;
      pos_sp.velocity.z = _traj_list[_points_id].velocity.z;
      pos_sp.acceleration_or_force.x = _traj_list[_points_id].acceleration_or_force.x;
      pos_sp.acceleration_or_force.y = _traj_list[_points_id].acceleration_or_force.y;
      pos_sp.acceleration_or_force.z = _traj_list[_points_id].acceleration_or_force.z;
      // pos_sp.yaw = atan2(pos_sp.velocity.y, pos_sp.velocity.x); // yaw control
      pos_sp.yaw = 0; // fixed yaw
      pos_sp.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;

      _points_id++;
    }
    else if (_points_id == _traj_list.size())
    {
      _points_id--;
      std::cout << "Hold last ref: " << _points_id << std::endl;
      std::cout << "Position: " << _traj_list[_points_id].position.x << " " << _traj_list[_points_id].position.y << " " << _traj_list[_points_id].position.z << std::endl;
      std::cout << "Velocity: " << _traj_list[_points_id].velocity.x << " " << _traj_list[_points_id].velocity.y << " " << _traj_list[_points_id].velocity.z << std::endl;
      std::cout << "Aceleration: " << _traj_list[_points_id].acceleration_or_force.x << " " << _traj_list[_points_id].acceleration_or_force.y << " " << _traj_list[_points_id].acceleration_or_force.z << std::endl;
      pos_sp.position.x = _traj_list[_points_id].position.x;
      pos_sp.position.y = _traj_list[_points_id].position.y;
      pos_sp.position.z = _traj_list[_points_id].position.z;
      pos_sp.velocity.x = _traj_list[_points_id].velocity.x;
      pos_sp.velocity.y = _traj_list[_points_id].velocity.y;
      pos_sp.velocity.z = _traj_list[_points_id].velocity.z;
      pos_sp.acceleration_or_force.x = _traj_list[_points_id].acceleration_or_force.x;
      pos_sp.acceleration_or_force.y = _traj_list[_points_id].acceleration_or_force.y;
      pos_sp.acceleration_or_force.z = _traj_list[_points_id].acceleration_or_force.z;
      pos_sp.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
      _points_id++;
    }

    pos_sp.type_mask = 32768;
    std::cout << "Yaw: " << pos_sp.yaw << std::endl;
    Position_Setpoint_Pub.publish(pos_sp);
  }

  default:
    break;
  }
}

void OffbNode::land()
{
  ros::Rate rate(20.0);
  double curr_alt = uav_gps_cur.altitude;
  double curr_lon = uav_gps_cur.longitude;
  double curr_lat = uav_gps_cur.latitude;
  double home_alt = uav_gps_home.geo.altitude;

  mavros_msgs::CommandTOL landing_cmd;
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
  return;
}

void OffbNode::uavStateCallBack(const mavros_msgs::State::ConstPtr &msg)
{
  uav_current_state = *msg;
}

void OffbNode::uavPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
  uav_pose = *msg;
  tf::pointMsgToEigen(uav_pose.pose.position, uav_local_pos_enu);
  uav_attitude_q << uav_pose.pose.orientation.w, uav_pose.pose.orientation.x, uav_pose.pose.orientation.y, uav_pose.pose.orientation.z;
}

bool OffbNode::set_offboard()
{
  ros::Rate rate(20.0);
  ros::Time last_request = ros::Time::now();
  mavros_msgs::PositionTarget init_sp;
  init_sp.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED; // To be converted to NED in setpoint_raw plugin
  // FIXME check
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
  init_sp.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;

  // send a few setpoints before starting
  for (int i = 100; ros::ok() && i > 0; --i)
  {
    Position_Setpoint_Pub.publish(init_sp);
    ros::spinOnce();
    rate.sleep();
  }

  mavros_msgs::SetMode offb_set_mode;
  offb_set_mode.request.custom_mode = "OFFBOARD";

  bool is_mode_ready = false;
  last_request = ros::Time::now();
  arm_cmd_.request.value = true;

  while (!is_mode_ready)
  {
    if (uav_current_state.mode != "OFFBOARD" &&
        (ros::Time::now() - last_request > ros::Duration(5.0)))
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
      if (!uav_current_state.armed && (ros::Time::now() - last_request > ros::Duration(5.0)))
      {
        ROS_INFO("Try Arming");
        if (arming_client.call(arm_cmd_) && arm_cmd_.response.success)
        {
          ROS_INFO("Vehicle armed");
        }
        last_request = ros::Time::now();
      }
      // is_mode_ready = (uav_current_state.mode == "OFFBOARD");
    }
    Position_Setpoint_Pub.publish(init_sp);
    is_mode_ready = (uav_current_state.mode == "OFFBOARD" && uav_current_state.armed);
    ros::spinOnce();
    rate.sleep();
  }

  if (is_mode_ready)
  {
    ROS_INFO("Offboard mode activated!");
  }

  return is_mode_ready;
}

void OffbNode::cmd_cb(const std_msgs::Byte::ConstPtr &msg)
{
  int cmd = msg->data;
  ROS_INFO("user cmd received");
  switch (cmd)
  {
  case TAKEOFF: // 1
  {
    if (arm_safety_check)
    {
      if (!uav_current_state.armed)
      {
        ROS_ERROR("Vehicle is not armed, please ask safety pilot to arm the vehicle.");
        break;
      }
    }

    ROS_INFO("TAKEOFF command received!");
    uavTask = kTakeOff;
    takeoff_height = uav_pose.pose.position.z + 1.0;
    takeoff_x = uav_pose.pose.position.x;
    takeoff_y = uav_pose.pose.position.y;
    if (set_offboard())
    {
      ROS_INFO("offboard mode activated going to run takeoff");
      mission_timer.start();
      ROS_INFO("Mission timer started!");
      takeoff_flag = true;
    }
    break;
  }

  case WAYPOINT: // 2
  {
    if (!takeoff_flag)
    {
      ROS_ERROR("Vehicle has not taken off, please issue takeoff command first.");
      break;
    }
    ROS_INFO("Waypoint command received!");
    uavTask = kWaypoint;
    if (!navGoal_init)
    {
      navGoal_sp = uav_pose;
    }
    break;
  }

  case MISSION: // 3
  {
    if (!takeoff_flag)
    {
      ROS_ERROR("Vehicle has not taken off, please issue takeoff command first.");
      break;
    }
    ROS_INFO("Mission command received!");
    ROS_INFO("Loading Trajectory...");
    if (loadTrajectory())
    {

      ROS_INFO("trajectory is loaded.");
      uavTask = kMission;
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
}

bool OffbNode::loadTrajectory()
{
  std::ifstream infile(trajectory_location.c_str());
  std::string line;
  clearTrajectory();
  bool succeed = false;
  ROS_INFO("Loading from trajectory reference file");

  std::ifstream f(trajectory_location.c_str());
  if (!f.good())
  {
    ROS_ERROR("Wrong file name!");
    printf("%s\n", trajectory_location.c_str());
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
    mavros_msgs::PositionTarget p;
    p.position.x = _1;
    p.position.y = _2;
    p.position.z = _3;
    p.velocity.x = _4;
    p.velocity.y = _5;
    p.velocity.z = _6;
    p.acceleration_or_force.x = _7;
    p.acceleration_or_force.y = _8;
    p.acceleration_or_force.z = _9;
    _traj_list.push_back(p);
    std::cout << "Position : " << _traj_list[_points_id].position.x << " " << _traj_list[_points_id].position.y << " " << _traj_list[_points_id].position.z << " "
              << "Velocity : " << _traj_list[_points_id].velocity.x << " " << _traj_list[_points_id].velocity.y << " " << _traj_list[_points_id].velocity.z << " "
              << "Accel : " << _traj_list[_points_id].acceleration_or_force.x << " " << _traj_list[_points_id].acceleration_or_force.y << " " << _traj_list[_points_id].acceleration_or_force.z << " "
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
  _points_id = 0;
  ROS_INFO("Clear Mission");
}

void OffbNode::refTrajPtCallBack(const quadrotor_msgs::TrajectoryPoint::ConstPtr &msg)
{
  traj_pt_nwu = *msg;
}

void OffbNode::trajSpENUCallBack(const quadrotor_msgs::TrajectoryPoint::ConstPtr &msg)
{
  navGoal_init = true;
  traj_sp_enu = *msg;
}

void OffbNode::vel_callback(const geometry_msgs::TwistStamped::ConstPtr &msg)
{
  // uav_local_vel_enu.x() = *msg->twist.linear.x;
  // uav_local_vel_enu.y() = *msg->twist.linear.y;
  // uav_local_vel_enu.z() = *msg->twist.linear.z;

  tf::vectorMsgToEigen(msg->twist.linear, uav_local_vel_enu);
}