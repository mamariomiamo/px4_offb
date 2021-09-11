#!/usr/bin/env python
import rospy
from enum import Enum
from std_msgs.msg import Int64, Header, Byte
from std_srvs.srv import SetBool
import math
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Quaternion
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, \
                            WaypointList, PositionTarget, AttitudeTarget, Thrust
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, \
                            WaypointPush
from pymavlink import mavutil
from sensor_msgs.msg import NavSatFix, Imu
from six.moves import xrange
from threading import Thread
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np


class uavTaskType(Enum):
    Idle = 0
    TakeOff = 1
    Mission = 2
    Land = 3


class NumberCounter:
    def __init__(self):
        self.counter = 0
        self.pub = rospy.Publisher("/number_count", Int64, queue_size=10)
        self.number_subscriber = rospy.Subscriber(
            "/number", Int64, self.callback_number)
        self.reset_service = rospy.Service(
            "/reset_counter", SetBool, self.callback_reset_counter)

    def callback_number(self, msg):
        self.counter += msg.data
        new_msg = Int64()
        new_msg.data = self.counter
        self.pub.publish(new_msg)

    def callback_reset_counter(self, req):
        if req.data:
            self.counter = 0
            return True, "Counter has been successfully reset"
        return False, "Counter has not been reset"


class TaskManager:
    def __init__(self):
        self.altitude = Altitude()
        self.extened_state = ExtendedState()
        self.global_position = NavSatFix()
        self.imu_data = Imu()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.attitude_sp = PoseStamped()
        self.pen_pose = PoseStamped()
        self.state = State()
        self.local_velocity = TwistStamped()  # local_velocity initialize
        self.attitude_rate = AttitudeTarget()  # use for attitude setpoints pub
        self.thrust = Thrust()

        self.pos = PoseStamped()
        self.position = PositionTarget()  # thrust control commands

        self.task_state = uavTaskType.Idle
        self.euler = Vector3()  # Euler angles
        self.pos_sp = Vector3() #position setpoint

        # ROS publisher
        self.pos_control_pub = rospy.Publisher(
            'mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.position_pub = rospy.Publisher(
            'mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.attitude_sp_pub = rospy.Publisher(
            'mavros/setpoint_attitude/attitude', PoseStamped, queue_size=1)
        self.attitude_rate_sp_pub = rospy.Publisher(
            'mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        self.attitude_thrust_pub = rospy.Publisher(
            'mavros/setpoint_attitude/thrust', Thrust, queue_size = 1)
        # ROS subscribers

        self.local_pos_sub = rospy.Subscriber(
            'mavros/local_position/pose', PoseStamped, self.local_position_callback)
        self.state_sub = rospy.Subscriber(
            'mavros/state', State, self.state_callback)
        self.cmd_sub = rospy.Subscriber('user/cmd', Byte, self.cmd_callback)
        self.vel_sub = rospy.Subscriber('mavros/local_position/velocity_local',
                                        TwistStamped, self.local_velocity_callback)  # local_velocity susbcriber
        #self.vel_global_sub = rospy.Subscriber('mavros/local_position/velocity_local', TwistStamped, self.global_velocity_callback)
        # send setpoints in seperate thread to better prevent failsafe
        self.pen_pose_sub = rospy.Subscriber('/vrpn_client_ros/pen/pose', PoseStamped, self.pen_position_callback)
        self.pos_thread = Thread(target=self.send_pos_ctrl, args=())
        self.pos_thread.daemon = True
        self.pos_thread.start()

        # ROS services
        service_timeout = 30
        rospy.loginfo("Waiting for ROS services")
        try:
            rospy.wait_for_service('mavros/param/get', service_timeout)
            rospy.wait_for_service('mavros/cmd/arming', service_timeout)
            rospy.wait_for_service('mavros/mission/push', service_timeout)
            rospy.wait_for_service('mavros/mission/clear', service_timeout)
            rospy.wait_for_service('mavros/set_mode', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            rospy.logerr("failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy('mavros/param/get', ParamGet)
        self.set_arming_srv = rospy.ServiceProxy(
            'mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)

    def local_velocity_callback(self, data):  # local_velocity callback
        self.local_velocity = data

    def send_pos(self):
        rate = rospy.Rate(10)
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.position_pub.publish(self.pos)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def send_pos_ctrl(self):
        rate = rospy.Rate(10)
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.pos_control_pub.publish(self.position)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def cmd_callback(self, data):
        # self.task_state = data
        cmd = data.data
        rospy.loginfo("Command received: {0}".format(self.task_state))
        rospy.loginfo("Command received: {0}".format(data))
        if cmd == 1:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.TakeOff
        elif cmd == 2:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.Mission
        elif cmd == 3:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.Land

    def local_position_callback(self, data):
        self.local_position = data
        q = [data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w]
        self.euler = euler_from_quaternion(q)

    def state_callback(self, data):
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(
                self.state.armed, data.armed))

        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(
                self.state.connected, data.connected))

        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(
                self.state.mode, data.mode))

        if self.state.system_status != data.system_status:
            rospy.loginfo("system_status changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_STATE'][
                    self.state.system_status].name, mavutil.mavlink.enums[
                        'MAV_STATE'][data.system_status].name))

        self.state = data
    def pen_position_callback(self, data):
        self.pen_pose = data
    #
    # Helper methods
    #

    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr("fail to arm")

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr("fail to set mode")


if __name__ == '__main__':
    rospy.init_node('number_counter')
    print("hahaha")
    NumberCounter()
    uavTask = TaskManager()

    uavTask.pos.pose.position.x = 0
    uavTask.pos.pose.position.y = 0
    uavTask.pos.pose.position.z = 0

    uavTask.set_mode("OFFBOARD", 5)
    uavTask.set_arm(True, 5)

    while not rospy.is_shutdown():
        rate = rospy.Rate(200)
        print(uavTask.task_state)
        # uavTask.position_pub.publish(uavTask.pos)
        if uavTask.task_state == uavTaskType.TakeOff:
            rospy.loginfo("Doing LQR takeoff")
            uavTask.pos_sp = [0, 0, 0.6]
            # Get position feedback from PX4
            x = uavTask.local_position.pose.position.x
            y = uavTask.local_position.pose.position.y
            z = uavTask.local_position.pose.position.z  # ENU used in ROS
            vx_enu = uavTask.local_velocity.twist.linear.x  # NWU body frame
            vy_enu = uavTask.local_velocity.twist.linear.y
            vz_enu = uavTask.local_velocity.twist.linear.z
            # LQR-based controller, x-gamma, y-beta, z-alpha
            # gamma = uavTask.euler[0]
            # beta = uavTask.euler[1]

            yaw = 0/57.3 # attitude_rate setpoint body_z
            # yaw = 0 #simulation face east
            state_x = np.array([[x, vx_enu]]).T
            # K_x = np.array([[0.1,0.1724]]) heading East!!!
            K_x = np.array([[0.1,0.1744]]) #less aggressive
            beta = -np.matmul(K_x, state_x) # attitude setpoint body_y
            state_y = np.array([[y, vy_enu]]).T
            # K_y = np.array([[-0.1, -0.1724])
            K_y = np.array([[-0.1, -0.1744]])
            gamma = -np.matmul(K_y, state_y) # attitude setpoint body_x
            state_z = np.array([[z-uavTask.pos_sp[2], vz_enu]]).T
            # K_z = np.array([[0.7071, 1.2305]])
            K_z = np.array([[0.7071,1.3836]]) #less aggresive
            a = -np.matmul(K_z, state_z)/(3*9.8)+0.355 #throttle sp
            #a = float(a)

            uavTask.attitude_rate.body_rate = Vector3()
            uavTask.attitude_rate.header = Header()
            uavTask.attitude_rate.header.frame_id = "base_footprint"
            #uavTask.attitude_rate.orientation = 
            quat = quaternion_from_euler(gamma, beta, yaw)
            #quat = quaternion_from_euler(0, 0, 0)
            # uavTask.attitude_rate.body_rate.y = 0
            # uavTask.attitude_rate.body_rate.z = 0
            #eu = np.array([[gamma, beta, yaw]]).T
            #quat = quaternion_from_euler(gamma, beta, yaw) # X,Y,Z,W
            #uavTask.attitude_rate.orientation = quat
            uavTask.attitude_rate.orientation.x = quat[0]
            uavTask.attitude_rate.orientation.y = quat[1]
            uavTask.attitude_rate.orientation.z = quat[2]
            uavTask.attitude_rate.orientation.w = quat[3]
            #uavTask.attitude_sp.pose.position.x = 0
            #uavTask.attitude_sp.pose.position.y = 0
            #uavTask.attitude_sp.pose.position.z = 0
            #uavTask.attitude_sp.pose.orientation.x = quat[0]
            #uavTask.attitude_sp.pose.orientation.y = quat[1]
            #uavTask.attitude_sp.pose.orientation.z = quat[2]
            #uavTask.attitude_sp.pose.orientation.w = quat[3]
            #uavTask.thrust.thrust = a
            uavTask.attitude_rate.thrust = a
            uavTask.attitude_rate.type_mask = 7
            uavTask.attitude_rate_sp_pub.publish(uavTask.attitude_rate)
            #uavTask.attitude_thrust_pub.publish(uavTask.thrust)
            ## Controller will be used here ###

            #uavTask.pos_control_pub.publish(uavTask.position)

        elif uavTask.task_state == uavTaskType.Mission:
            rospy.loginfo("Flying pendulum")
            uavTask.pos_sp = [0, 0, 0.6]
            # Get position feedback from PX4
            # uavTask.pen_pose.pose.position.x
            x = uavTask.local_position.pose.position.x
            y = uavTask.local_position.pose.position.y
            z = uavTask.local_position.pose.position.z  # ENU used in ROS
            dx = uavTask.pen_pose.pose.position.x - x 
            dy = uavTask.pen_pose.pose.position.y - y
            vx_enu = uavTask.local_velocity.twist.linear.x  # NWU body frame
            vy_enu = uavTask.local_velocity.twist.linear.y
            vz_enu = uavTask.local_velocity.twist.linear.z
            # LQR-based controller, x-gamma, y-beta, z-alpha
            # gamma = uavTask.euler[0]
            # beta = uavTask.euler[1]

            yaw = 0/57.3 # attitude_rate setpoint body_z
            # yaw = 0 #simulation face east
            state_x = np.array([[dx, 0, x, vx_enu]]).T
            # K_x = np.array([[0.1,0.1724]]) heading East!!!
            K_x = np.array([[-2.801, -1.0236, -0.1, -0.2745]]) #less aggressive
            beta = -np.matmul(K_x, state_x) # attitude setpoint body_y
            state_y = np.array([[dy, 0, y, vy_enu]]).T
            # K_y = np.array([[-0.1, -0.1724])
            K_y = np.array([[2.801, 1.0236, 0.1, 0.2745]])
            gamma = -np.matmul(K_y, state_y) # attitude setpoint body_x
            state_z = np.array([[z-uavTask.pos_sp[2], vz_enu]]).T
            # K_z = np.array([[0.7071, 1.2305]])
            K_z = np.array([[0.7071,1.7071]]) #less aggresive
            a = -np.matmul(K_z, state_z)/(3*9.8)+0.355 #throttle sp
            #a = float(a)

            uavTask.attitude_rate.body_rate = Vector3()
            uavTask.attitude_rate.header = Header()
            uavTask.attitude_rate.header.frame_id = "base_footprint"
            #uavTask.attitude_rate.orientation = 
            quat = quaternion_from_euler(gamma, beta, yaw)
            #quat = quaternion_from_euler(0, 0, 0)
            # uavTask.attitude_rate.body_rate.y = 0
            # uavTask.attitude_rate.body_rate.z = 0
            #eu = np.array([[gamma, beta, yaw]]).T
            #quat = quaternion_from_euler(gamma, beta, yaw) # X,Y,Z,W
            #uavTask.attitude_rate.orientation = quat
            uavTask.attitude_rate.orientation.x = quat[0]
            uavTask.attitude_rate.orientation.y = quat[1]
            uavTask.attitude_rate.orientation.z = quat[2]
            uavTask.attitude_rate.orientation.w = quat[3]
            #uavTask.attitude_sp.pose.position.x = 0
            #uavTask.attitude_sp.pose.position.y = 0
            #uavTask.attitude_sp.pose.position.z = 0
            #uavTask.attitude_sp.pose.orientation.x = quat[0]
            #uavTask.attitude_sp.pose.orientation.y = quat[1]
            #uavTask.attitude_sp.pose.orientation.z = quat[2]
            #uavTask.attitude_sp.pose.orientation.w = quat[3]
            #uavTask.thrust.thrust = a
            uavTask.attitude_rate.thrust = a
            uavTask.attitude_rate.type_mask = 7
            uavTask.attitude_rate_sp_pub.publish(uavTask.attitude_rate)
            #uavTask.attitude_thrust_pub.publish(uavTask.thrust)
            ## Controller will be used here ###

            #uavTask.pos_control_pub.publish(uavTask.position)

        elif uavTask.task_state == uavTaskType.Land:
            rospy.loginfo("Doing Land")
            uavTask.pos.pose.position.x = 0
            uavTask.pos.pose.position.y = 0
            uavTask.pos.pose.position.z = 0
            uavTask.position_pub.publish(uavTask.pos)


        rate.sleep()
    rospy.spin()
