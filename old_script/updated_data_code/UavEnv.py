"""
This file defines the simulation environment for a quad-rotor
-------------------------------------------------------------
Wang Bingheng, 15 Dec. 2020 at Advanced Control Lab, ECE Dept. NUS
"""

from casadi import *
import numpy as np
import math
from numpy import linalg as LA

class quadrotor:
    def __init__(self, para, IGE_para, dt_sample):
        # Position in inertial frame
        rx, ry, rz    = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I      = vertcat(rx, ry, rz)
        # Position of payload in inertial frame
        px, py, pz    = SX.sym('px'), SX.sym('py'), SX.sym('pz')
        self.p_I      = vertcat(px, py, pz)
        # Velocity in inertial frame
        vx, vy, vz    = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I      = vertcat(vx, vy, vz)
        # velocity of payload in inertial frame
        vpx, vpy, vpz = SX.sym('vpx'), SX.sym('vpy'), SX.sym('vpz')
        self.vp_I     = vertcat(vpx, vpy, vpz)
        # Rotation matrix from body frame to inertial frame
        self.r11, self.r12, self.r13 = SX.sym('r11'), SX.sym('r12'), SX.sym('r13')
        self.r21, self.r22, self.r23 = SX.sym('r21'), SX.sym('r22'), SX.sym('r23')
        self.r31, self.r32, self.r33 = SX.sym('r31'), SX.sym('r32'), SX.sym('r33')
        self.R_B      = vertcat(
            horzcat(self.r11, self.r12, self.r13),
            horzcat(self.r21, self.r22, self.r23),
            horzcat(self.r31, self.r32, self.r33)
        )
        # Angular velocity in body frame
        self.wx, self.wy, self.wz    = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w_B      = vertcat(self.wx, self.wy, self.wz)
        # Square of motor speed
        r1, r2, r3, r4 = SX.sym('r1'), SX.sym('r2'), SX.sym('r3'), SX.sym('r4')
        self.ctrl     = vertcat(r1, r2, r3, r4)
        # Disturbance forces in inertial frame
        dfx, dfy, dfz = SX.sym('dfx'), SX.sym('dfy'), SX.sym('dfz')
        self.df_I     = vertcat(dfx, dfy, dfz)
        # Disturbance torques in body frame
        dtx, dty, dtz = SX.sym('dtx'), SX.sym('dty'), SX.sym('dtz')
        self.dt_B     = vertcat(dtx, dty, dtz)
        # Inertial parameters
        self.mass     = para[0]
        jx, jy, jz    = para[1], para[2], para[3]
        self.J_B      = diag(vertcat(jx, jy, jz))
        # Parameters of the control mapping (interface)
        self.l, self.b, self.kt      = para[4], para[5], para[6]
        # Control mapping
        self.fm       = horzcat(self.b, self.b, self.b, self.b)
        self.tm       = vertcat(
            horzcat(0, self.b*self.l, 0, -self.b*self.l),
            horzcat(self.b*self.l, 0, -self.b*self.l, 0),
            horzcat(self.kt, -self.kt, self.kt, -self.kt)
        )
        # Disturbance force noise vector
        wfx, wfy, wfz = SX.sym('wfx'), SX.sym('wfy'), SX.sym('wfz')
        self.wf       = vertcat(wfx, wfy, wfz)
        # Disturbance torque noise vector
        wtx, wty, wtz = SX.sym('wtx'), SX.sym('wty'), SX.sym('wtz')
        self.wt       = vertcat(wtx, wty, wtz)
        # Low-order model for In-Ground-Effect
        self.Ca, self.Cb, self.radius = IGE_para[0], IGE_para[1], IGE_para[2]
        # Tether tension
        self.T = SX.sym('T', 3, 1)
        # Sampling time
        self.DT = dt_sample

    def position_dyn_mh(self, state_p_mh, f_d, noise, R_b):
        g = 9.81
        # Z direction vector free of coordinate
        z = vertcat(0, 0, 1)
        v_I = vertcat(state_p_mh[3], state_p_mh[4], state_p_mh[5])
        df_I = vertcat(state_p_mh[6], state_p_mh[7], state_p_mh[8])
        dp_I = v_I
        dv_I = 1 / self.mass * (self.mass * g * z - mtimes(R_b, mtimes(transpose(f_d), mtimes(R_b, z)) * z) + df_I)
        ddf_I = noise
        Xmh_p_dot = vertcat(dp_I, dv_I, ddf_I)
        return Xmh_p_dot

    def position_dyn(self, state_p, f_d, R_b, df_I):
        g = 9.81
        # Z direction vector free of coordinate
        z = vertcat(0, 0, 1)
        v_I = vertcat(state_p[3], state_p[4], state_p[5])
        dr_I = v_I
        dv_I = 1 / self.mass * (self.mass * g * z - mtimes(R_b, mtimes(transpose(f_d), mtimes(R_b, z)) * z) + df_I)
        X_p_dot = vertcat(dr_I, dv_I)
        return X_p_dot

    def dir_cosine(self, Euler):
        # Euler angles for roll, pitch and yaw
        phi, theta, psi = Euler[0], Euler[1], Euler[2]
        # Initial rotation matrix from body frame to inertial frame
        r11_0 = math.cos(theta)*math.cos(psi)
        r12_0 = math.sin(phi)*math.sin(theta)*math.cos(psi)-math.cos(phi)*math.sin(psi)
        r13_0 = math.cos(phi)*math.sin(theta)*math.cos(psi)+math.sin(phi)*math.sin(psi)
        r21_0 = math.cos(theta)*math.sin(psi)
        r22_0 = math.sin(phi)*math.sin(theta)*math.sin(psi)+math.cos(phi)*math.cos(psi)
        r23_0 = math.cos(phi)*math.sin(theta)*math.sin(psi)-math.sin(phi)*math.cos(psi)
        r31_0, r32_0, r33_0 = -math.sin(theta), math.sin(phi)*math.cos(theta), math.cos(phi)*math.cos(theta)
        R_Bv0 = np.array([[r11_0, r12_0, r13_0, r21_0, r22_0, r23_0, r31_0, r32_0, r33_0]]).T
        R_B0  = vertcat(
             horzcat(r11_0, r12_0, r13_0),
             horzcat(r21_0, r22_0, r23_0),
             horzcat(r31_0, r32_0, r33_0)
         )
        return R_Bv0, R_B0

    def Model(self):
        # Global parameter
        g = 9.81

        # Z direction vector free of coordinate
        z = vertcat(0, 0, 1)
        # Position kinematics in inertial frame
        dr_I = self.v_I
        # Attitude kinematics in body frame
        dr11, dr12, dr13 = self.r12 * self.wz - self.r13 * self.wy, self.r13 * self.wx - self.r11 * self.wz, self.r11 * self.wy - self.r12 * self.wx
        dr21, dr22, dr23 = self.r22 * self.wz - self.r23 * self.wy, self.r23 * self.wx - self.r21 * self.wz, self.r21 * self.wy - self.r22 * self.wx
        dr31, dr32, dr33 = self.r32 * self.wz - self.r33 * self.wy, self.r33 * self.wx - self.r31 * self.wz, self.r31 * self.wy - self.r32 * self.wx
        # Dynamics of disturbance force and torque
        ddf_I = self.wf
        ddt_B = self.wt

        # Velocity dynamics in inertial frame
        dv_I = 1 / self.mass * (self.mass * g * z - mtimes(self.R_B, mtimes(self.fm, self.ctrl) * z) + self.df_I)
        self.f_d = SX.sym('f_d', 3, 1)
        dv_I_mh = 1 / self.mass * (self.mass * g * z - mtimes(self.R_B, mtimes(transpose(self.f_d), mtimes(self.R_B, z)) * z) + self.df_I)
        # Angular velocity dynamics in body frame
        dw_B = mtimes(inv(self.J_B), mtimes(self.tm, self.ctrl) + self.dt_B - mtimes(self.skew(self.w_B), mtimes(self.J_B, self.w_B)))
        # States
        self.X = vertcat(self.r_I, self.v_I, self.r11, self.r12, self.r13, self.r21,
                         self.r22, self.r23, self.r31, self.r32, self.r33, self.w_B)
        # Disturbances
        self.dis  = vertcat(self.df_I, self.dt_B)
        # Time-derivative
        self.Xdot = vertcat(dr_I, dv_I, dr11, dr12, dr13, dr21,
                            dr22, dr23, dr31, dr32, dr33, dw_B)
        # Define dynamics function
        self.Dyn  = Function('Dyn', [self.X, self.ctrl, self.dis], [self.Xdot], ['X0', 'Ctrl0', 'Dis0'], ['Xdotf'])

        # Extended states
        self.Xmhe = vertcat(self.r_I, self.v_I, self.df_I, self.r11, self.r12, self.r13, self.r21,
                            self.r22, self.r23, self.r31, self.r32, self.r33, self.w_B, self.dt_B)
        # Output
        self.output = self.X
        # Output for position
        self.output_p = vertcat(self.r_I, self.v_I)
        # Noise input
        self.noise   = vertcat(self.wf, self.wt)
        # Time-derivative of extended states
        self.Xmhedot = vertcat(dr_I, dv_I, ddf_I, dr11, dr12, dr13, dr21,
                               dr22, dr23, dr31, dr32, dr33, dw_B, ddt_B)
        # Position dynamics for DMHE
        # self.pdyn_mhe = vertcat(dr_I, dv_I_mh, ddf_I)

        # Extended position states
        self.Xmhe_p = vertcat(self.r_I, self.v_I, self.df_I)

        # 4-order Runge-Kutta discretization of dynamics model used in MHE
        kp1 = self.position_dyn_mh(self.Xmhe_p, self.f_d, self.wf, self.R_B)
        kp2 = self.position_dyn_mh(self.Xmhe_p + self.DT / 2 * kp1, self.f_d, self.wf, self.R_B)
        kp3 = self.position_dyn_mh(self.Xmhe_p + self.DT / 2 * kp2, self.f_d, self.wf, self.R_B)
        kp4 = self.position_dyn_mh(self.Xmhe_p + self.DT * kp3, self.f_d, self.wf, self.R_B)
        self.pdyn_mhe = (kp1 + 2 * kp2 + 2 * kp3 + kp4)/6
        # 4-order Runge-Kutta discretization of dynamics model used in DMHE
        k1  = self.position_dyn(self.output_p, self.f_d, self.R_B, self.df_I)
        k2  = self.position_dyn(self.output_p + self.DT / 2 * k1, self.f_d, self.R_B, self.df_I)
        k3  = self.position_dyn(self.output_p + self.DT / 2 * k2, self.f_d, self.R_B, self.df_I)
        k4  = self.position_dyn(self.output_p + self.DT * k3, self.f_d, self.R_B, self.df_I)
        self.pdyn = (k1 + 2 * k2 + 2 * k3 + k4)/6

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def Aerodynamics(self, state, a_I_new, ctrl, constant_dis, a_noise, pmass, aero_para):
        if constant_dis:
            dis = aero_para
        else:
            # IGE disturbance
            P_com = np.array([[state[0, 0], state[1, 0], state[2, 0]]]).T

            R_b   = np.array([[state[6, 0], state[7, 0], state[8, 0]],
                              [state[9, 0], state[10, 0], state[11, 0]],
                              [state[12, 0], state[13, 0], state[14, 0]]])
            r1_b  = np.array([[self.l, 0, 0]]).T
            r2_b  = np.array([[0, -self.l, 0]]).T
            r3_b  = np.array([[-self.l, 0, 0]]).T
            r4_b  = np.array([[0, self.l, 0]]).T
            Z1_i  = P_com + np.matmul(R_b, r1_b)
            Z2_i  = P_com + np.matmul(R_b, r2_b)
            Z3_i  = P_com + np.matmul(R_b, r3_b)
            Z4_i  = P_com + np.matmul(R_b, r4_b)
            z1_i  = np.clip(Z1_i[2], -2.5, -0.1)
            z2_i  = np.clip(Z2_i[2], -2.5, -0.1)
            z3_i  = np.clip(Z3_i[2], -2.5, -0.1)
            z4_i  = np.clip(Z4_i[2], -2.5, -0.1)
            IGE_1 = self.Ca * np.exp(self.Cb * z1_i / self.radius)
            IGE_2 = self.Ca * np.exp(self.Cb * z2_i / self.radius)
            IGE_3 = self.Ca * np.exp(self.Cb * z3_i / self.radius)
            IGE_4 = self.Ca * np.exp(self.Cb * z4_i / self.radius)
            delta_w1 = IGE_1 * ctrl[0, 0]
            delta_w2 = IGE_2 * ctrl[1, 0]
            delta_w3 = IGE_3 * ctrl[2, 0]
            delta_w4 = IGE_4 * ctrl[3, 0]
            disf_b   = self.b*delta_w1 + self.b*delta_w2 + self.b*delta_w3 + self.b*delta_w4
            Disf_b   = np.array([[0, 0, -disf_b[0]]]).T
            Disf_i   = np.matmul(R_b, Disf_b)
            distx_b  = [0] #self.b*self.l*(delta_w2 - delta_w4)
            disty_b  = [0] #self.b*self.l*(delta_w1 - delta_w3)
            distz_b  = [0] #self.kt*(delta_w1 - delta_w2 + delta_w3 - delta_w4)

            # Disturbance force due to the payload
            g = 9.81
            z = np.array([[0, 0, 1]]).T
            ap_noise = np.array([[a_noise[0], a_noise[1], a_noise[2]]]).T
            a_p = a_I_new + ap_noise
            if pmass<0:
                pmass = 0
            T   = pmass*(a_p-g*z)
            # tension acting on the drone
            T_drone = -T
            Disf_total = T_drone+ Disf_i
            dis = np.array([[Disf_total[0, 0], Disf_total[1, 0], Disf_total[2, 0], distx_b[0], disty_b[0], distz_b[0]]]).T

        return dis

    """
    The step function takes control (square of motor speed) as input and 
    returns the new states in the next step
    """
    def step(self, state, control, dis, dt):
        self.Model()
        # define discrete-time dynamics using 4-th order Runge-Kutta
        k1 = self.Dyn(X0=state, Ctrl0=control, Dis0=dis)['Xdotf'].full()
        k2 = self.Dyn(X0=state+dt/2*k1, Ctrl0=control, Dis0=dis)['Xdotf'].full()
        k3 = self.Dyn(X0=state+dt/2*k2, Ctrl0=control, Dis0=dis)['Xdotf'].full()
        k4 = self.Dyn(X0=state+dt*k3, Ctrl0=control, Dis0=dis)['Xdotf'].full()
        dstate = (k1 + 2*k2 + 2*k3 + k4)/6
        state_new = state + dt * dstate

        r_I_new = vertcat(state_new[0], state_new[1], state_new[2])
        a_I_new = vertcat(dstate[3], dstate[4], dstate[5])
        v_I_new = vertcat(state_new[3], state_new[4], state_new[5])
        R_B_new = vertcat(
            horzcat(state_new[6], state_new[7], state_new[8]),
            horzcat(state_new[9], state_new[10], state_new[11]),
            horzcat(state_new[12], state_new[13], state_new[14])
        )
        w_B_new = vertcat(state_new[15], state_new[16], state_new[17])
        # 1-2-3 rotation from {b} to {I}
        phi     = np.arctan(R_B_new[2, 1]/R_B_new[2, 2])
        theta   = np.arcsin(-R_B_new[2, 0])
        psi     = np.arctan(R_B_new[1, 0]/R_B_new[0, 0])
        Euler_new = vertcat(phi, theta, psi)

        output = {"state_new": state_new,
                  "r_I_new": r_I_new,
                  "v_I_new": v_I_new,
                  "Euler_new": Euler_new,
                  "w_B_new": w_B_new,
                  "a_I_new": a_I_new
                  }
        return output



