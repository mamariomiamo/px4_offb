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
        self.quad_vel = TwistStamped() #quadrotor velocity from filtered vicon
        self.quad_pose = PoseStamped()
        self.relative_speed = TwistStamped()
        self.pen_vel = TwistStamped() #pendulum velocity from filtered vicon
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
        self.relative_speed_pub = rospy.Publisher('relative_speed', TwistStamped, queue_size = 1)
        # ROS subscribers

        self.local_pos_sub = rospy.Subscriber(
            'mavros/local_position/pose', PoseStamped, self.local_position_callback)
        self.quad_pos_sub = rospy.Subscriber('mavros/vision_pose/pose',PoseStamped, self.quad_pos_callback)
        self.state_sub = rospy.Subscriber(
            'mavros/state', State, self.state_callback)
        self.cmd_sub = rospy.Subscriber('user/cmd', Byte, self.cmd_callback)
        self.vel_sub = rospy.Subscriber('mavros/local_position/velocity_local',
                                        TwistStamped, self.local_velocity_callback)  # local_velocity susbcriber
        self.quad_vel_sub = rospy.Subscriber('quadrotor_vel_filtered', TwistStamped, self.quad_vel_callback)
        self.pen_vel_sub = rospy.Subscriber('pendulum_vel_filtered',TwistStamped, self.pen_vel_callback)
        #self.vel_global_sub = rospy.Subscriber('mavros/local_position/velocity_local', TwistStamped, self.global_velocity_callback)
        # send setpoints in seperate thread to better prevent failsafe
        self.pen_pose_sub = rospy.Subscriber('pendulum_pos_filtered', PoseStamped, self.pen_position_callback)
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

    def quad_pos_callback(self, data):
        self.quad_pos = data

    def quad_vel_callback(self, data):
        self.quad_vel = data
        print("subbed filtered quad_vel")

    def pen_vel_callback(self, data):
        self.pen_vel = data

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
            #rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.TakeOff
        elif cmd == 2:
            #rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.Mission
        elif cmd == 3:
            #rospy.loginfo("Taks state changed to {0}".format(self.task_state))
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
        #self.pen_vel = data
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
    dX0 = 0
    dY0 = 0
    dx0 = 0
    dy0 = 0

    uavTask.set_mode("OFFBOARD", 5)
    uavTask.set_arm(True, 5)
    #xr_hat = np.array([[0,0]]).T
    #xs_hat = np.array([[0,0]]).T
    dt = 0.01

    #gamma = 0
    #beta = 0
    dvx_filter = 0
    dvy_filter = 0
    dvz_filter = 0
    cutoff_freq = 10

    while not rospy.is_shutdown():
        rate = rospy.Rate(100)
        #print(uavTask.task_state)
        # uavTask.position_pub.publish(uavTask.pos)
        if uavTask.task_state == uavTaskType.Idle:
            quad_vx = uavTask.quad_vel.twist.linear.x
            pen_vx = uavTask.pen_vel.twist.linear.x
            dvx = pen_vx - quad_vx
            #px = uavTask.pen_pose.pose.position.x
            #dx = px - x
            #print("pendulum x is",px)
            #print("quadrotor x is", x)
            print("relative vx is", dvx)
            uavTask.pos.pose.position.x = 0
            uavTask.pos.pose.position.y = 0
            uavTask.pos.pose.position.z = 0
            uavTask.position_pub.publish(uavTask.pos)
        if uavTask.task_state == uavTaskType.TakeOff:
            #rospy.loginfo("Doing LQR takeoff")
            uavTask.pos_sp = [0, 0, 0.75]
            # Get position feedback from PX4
            x = uavTask.local_position.pose.position.x
            #x = uavTask.quad_pose.pose.position.x
            #print("current p_x is {0}".format(x))
            y = uavTask.local_position.pose.position.y
            #y = uavTask.quad_pose.pose.position.y
            #print("curent p_y is {0}".format(y))
            z = uavTask.local_position.pose.position.z  # ENU used in ROS
            #z = uavTask.quad_pose.pose.position.z
            quad_vx = uavTask.quad_vel.twist.linear.x
            quad_vy = uavTask.quad_vel.twist.linear.y
            quad_vz = uavTask.quad_vel.twist.linear.z
            #filter_a = dt/(dt + 1/(2*3.14*cutoff_freq))
            #dvx_filter = (1-filter_a)*dvx_filter + filter_a * quad_vx
            #dvy_filter = (1-filter_a)*dvy_filter + filter_a * quad_vy
            #dvz_filter = (1-filter_a)*dvz_filter + filter_a * quad_vz

            vx_enu = uavTask.local_velocity.twist.linear.x  # NWU body frame
            vy_enu = uavTask.local_velocity.twist.linear.y
            vz_enu = uavTask.local_velocity.twist.linear.z
            # LQR-based controller, x-gamma, y-beta, z-alpha
            # gamma = uavTask.euler[0]
            # beta = uavTask.euler[1]
            #dx = uavTask.pen_pose.pose.position.x - x
            #dvx = (dx - dx0)/0.01
            #dx0 = dx
            #print("relative vx is",dvx)
            #print("previous dx0 is",dx0)
            #dy = uavTask.pen_pose.pose.position.y - y
            #dvy = (dy - dy0)/0.01
            #dy0 = dy
            #print("relative vy is",dvy)
            #uavTask.relative_speed.twist.linear.x = dvx
            #uavTask.relative_speed.twist.linear.y = dvy
            #if (dvx != 0 and dvy !=0):
            #   uavTask.relative_speed_pub.publish(uavTask.relative_speed)
            yaw = 0/57.3 # attitude_rate setpoint body_z
            # yaw = 0 #simulation face east
            state_x = np.array([[x, vx_enu]]).T
            # K_x = np.array([[0.1,0.1724]]) heading East!!!
            K_x = np.array([[0.7071, 1.0697]]) #less aggressive
            beta = -np.matmul(K_x, state_x) # attitude setpoint body_y
            state_y = np.array([[y, vy_enu]]).T
            # K_y = np.array([[-0.1, -0.1724])
            K_y = np.array([[-0.7071, -1.0697]])
            gamma = -np.matmul(K_y, state_y) # attitude setpoint body_x
            state_z = np.array([[z-uavTask.pos_sp[2], vz_enu]]).T
            # K_z = np.array([[0.7071, 1.2305]])
            K_z = np.array([[3.1623,2.8852]]) #less aggresive
            a = -np.matmul(K_z, state_z)/(3*9.8)+0.34 #throttle sp
            #a = float(a)

            uavTask.attitude_rate.body_rate = Vector3()
            uavTask.attitude_rate.header = Header()
            uavTask.attitude_rate.header.frame_id = "base_footprint"
            #uavTask.attitude_rate.orientation =
            quat = quaternion_from_euler(gamma, beta, yaw)
            #print("current gamma is {0}".format(gamma))
            #print("current beta is {0}".format(beta))
            #quat = quaternion_from_euler(0, 0, 0)
            dX = uavTask.pen_pose.pose.position.x - x
            dvx = uavTask.pen_vel.twist.linear.x - vx_enu
            uavTask.attitude_rate.body_rate.y = dX
            uavTask.attitude_rate.body_rate.z = dvx
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
            #rospy.loginfo("Flying pendulum")
            #Ar = np.array([[0, 1],[9.81, 0]])
            #As = Ar
            #Br = np.array([[0, -9.81]]).T
            #Bs = np.array([[0, 9.81]]).T
            #L = np.array([[30, 9.81+200]]).T #estimator gain
            uavTask.pos_sp = [0, 0, 0.7]
            # Get position feedback from PX4
            # uavTask.pen_pose.pose.position.x
            x = uavTask.local_position.pose.position.x
            y = uavTask.local_position.pose.position.y
            z = uavTask.local_position.pose.position.z  # ENU used in ROS
            dX = uavTask.pen_pose.pose.position.x - x
            quad_vx = uavTask.local_velocity.twist.linear.x
            pen_vx = uavTask.pen_vel.twist.linear.x
            dvx = pen_vx - quad_vx
            #dxr_hat = np.matmul(Ar,xr_hat) + Br*beta + L * (dX-xr_hat[0,0])
            #dvx = (dX - dX0)/0.01
            #dX0 = dX
            #threshold = 0.00
            #if abs(dX) >= threshold:
            #   dx = dX
            #else:
            #   dx = 0
            dY = uavTask.pen_pose.pose.position.y - y
            quad_vy = uavTask.local_velocity.twist.linear.y
            pen_vy = uavTask.pen_vel.twist.linear.y
            dvy = pen_vy - quad_vy
            #dxs_hat = np.matmul(As,xs_hat) + Bs*gamma + L*(dY-xs_hat[0,0])
            #dvy = (dY - dY0) / 0.01
            #dY0 = dY
            #if abs(dY) >= threshold:
            #   dy = dY
            #else:
            #   dy = 0
            #dx = 0
            #dy = 0
            vx_enu = uavTask.local_velocity.twist.linear.x  # NWU body frame
            vy_enu = uavTask.local_velocity.twist.linear.y
            vz_enu = uavTask.local_velocity.twist.linear.z
            # LQR-based controller, x-gamma, y-beta, z-alpha
            # gamma = uavTask.euler[0]
            # beta = uavTask.euler[1]

            yaw = 0/57.3 # attitude_rate setpoint body_z
            # yaw = 0 #simulation face east
            state_x = np.array([[dX, dvx, x, vx_enu]]).T
            #xr_hat = xr_hat + dt * dxr_hat
            # K_x = np.array([[0.1,0.1724]]) heading East!!!
            #LQR gains
            #R = 200, Q_pen= 400
            #K_x = np.array([[-4.3285, -1.9256, -0.0707, -0.2303]])
            #R = 100
            #K_x = np.array([[-3.3041,-1.0597,-0.1,-0.2388]])
            #R = 1000
            #K_x = np.array([[-2.6048, -0.8326, -0.0316, -0.1066]])
            #R = 200
            #K_x = np.array([[-3.02, -0.967, -0.0707, -0.1848]]) #less aggressive
            # R = 200, Q_pen = 10
            #K_x = np.array([[-3.0713, -1.0037, -0.0707,-0.1868]])
            # R = 200, Q_pen = 100 better performance
            #K_x = np.array([[-3.4785, -1.2976, -0.0707, -0.2019]])
            # R=200, Q_pen= 200 best
            K_x = np.array([[-3.8147, -1.5437, -0.0707, -0.2136]])
            beta = -np.matmul(K_x, state_x) # attitude setpoint body_y
            state_y = np.array([[dY, dvy, y, vy_enu]]).T
            #xs_hat = xs_hat + dt * dxs_hat
            # K_y = np.array([[-0.1, -0.1724])
            # R=200 Q_pen = 400 oscillate a lot
            #K_y = np.array([[4.3285, 1.9256, 0.0707, 0.2303]])
            # R=100
            #K_y = np.array([[3.3041, 1.0597, 0.1, 0.2388]])
            # R = 1000
            #K_y = np.array([[2.6048, 0.8326, 0.0316, 0.1066]])
            # R = 200
            #K_y = np.array([[3.02, 0.967, 0.0707, 0.1848]])
            # R = 200, Q_pen = 10
            #K_y = np.array([[3.0713, 1.0037, 0.0707, 0.1868]])
            # R = 200, Q_pen = 100
            #K_y = np.array([[3.4785, 1.2976, 0.0707, 0.2019]])
            #R= 200, Q_pen = 200 best (oscillate when battery low?)
            K_y = np.array([[3.8147,1.5437,0.0707,0.2136]])
            gamma = -np.matmul(K_y, state_y) # attitude setpoint body_x
            state_z = np.array([[z-uavTask.pos_sp[2], vz_enu]]).T
            # K_z = np.array([[0.7071, 1.2305]])
            K_z = np.array([[3.1623,2.8852]]) #less aggresive
            a = -np.matmul(K_z, state_z)/(3*9.8)+0.36 #throttle sp
            #a = float(a)

            uavTask.attitude_rate.body_rate = Vector3()
            uavTask.attitude_rate.header = Header()
            uavTask.attitude_rate.header.frame_id = "base_footprint"
            #uavTask.attitude_rate.orientation = 
            quat = quaternion_from_euler(gamma, beta, yaw)
            #quat = quaternion_from_euler(0, 0, 0)
            uavTask.attitude_rate.body_rate.y = dX
            uavTask.attitude_rate.body_rate.z = dvx
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
