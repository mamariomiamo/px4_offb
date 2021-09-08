"""
This is the main function that trains the DMHE and evaluates the performance
----------------------------------------------------------------------------
Wang, Bingheng, 24 Dec. 2020, at Advanced Control Lab, ECE Dept. NUS
"""
import UavEnv
import Robust_Flight
from casadi import *
import time
import numpy as np
import matplotlib.pyplot as plt
import uavNN
import torch
from numpy import linalg as LA


"""---------------------------------Learn or Evaluate?-------------------------------------"""
train = True

"""---------------------------------Type of disturbance?-----------------------------------"""
constant_dis = False

"""---------------------------------Load environment---------------------------------------"""
# Sys_para = np.array([2.5, 0.0291, 0.0291, 0.0467, 0.225, 1.29e-5, 1.74e-7])
# Initial parameters are used in the paper 'Backstepping Sliding-mode and Cascade Active
# Disturbance Rejection Control for a Quadrotor UAV' IEEE/ASME Transactions on Mechatronics
Sys_para = np.array([1.74, 0.5769, 0.573, 0.2475, 0.21, 9e-8, 1.74e-7])
# The above inertial parameters are provided by Zhentian Ma
u_max = 12000 # 12000 pm, the maximum rotation speed per minute
u_min = 0
U_upper = np.array([[u_max**2, u_max**2, u_max**2, u_max**2]]).T
U_lower = np.array([[u_min, u_min, u_min, u_min]]).T
IGE_para = np.array([0.44, 1.8, 0.1])

# Simulation time-step
T_step = 1e-2
# Sampling time-step
dt_sample = 5e-2
uav = UavEnv.quadrotor(Sys_para, IGE_para, dt_sample)
uav.Model()
# Initial states
r_I0 = np.array([[0, 0, -1]]).T # column vector
p_I0 = np.array([[0, 0, -0.05]]).T
vp_I0= np.array([[0, 0, 0]]).T
v_I0 = np.array([[0, 0, 0]]).T
df_I0 = np.array([[0, 0, 0]]).T
Euler_0 = np.array([[0, 0, 0]]).T
R_Bv0, R_Bd0 = uav.dir_cosine(Euler_0)
dR_Bd0 = np.zeros((3, 3))
w_B0 = np.array([[0, 0, 0]]).T
dt_B0 = np.array([[0, 0, 0]]).T  # disturbance torque in body frame
state0 = np.vstack((r_I0, v_I0, R_Bv0, w_B0))
Xmhe0 = np.vstack((r_I0, v_I0, df_I0, R_Bv0, w_B0, dt_B0)) # CANNOT define the type as vertcat!!!!


horizon = 10 # previous 30
# Time interval of taking-off
t_end = 5
# Time interval of an episode
t_ep  = 10
t_epv = 10
# Iteration number
N = int(t_ep / T_step)
N_ev = int(t_epv/T_step)
# Learning rate
lr_nn = 1e-4
lr_r  = 1e-4
# Total training times
N_train = 20
"""---------------------------------Define parameterization model-----------------------------"""
# Define neural network for process noise
D_in, D_h, D_out_p = 6, 50, 20
model_QR = uavNN.Net(D_in, D_h, D_out_p)
P_min = 0
P_max = 1e-1
gammar_min = 0
gammar_max = 5e-1
gammaq_min = 5e-1
gammaq_max = 1
R_min = 0
R_max = 1e2
Q_min = 0
Q_max = 1e2
para_bound_p  = (P_max-P_min)*np.ones((1, 9))
para_bound_gr = (gammar_max-gammar_min)*np.ones((1, 1))
para_bound_r  = (R_max-R_min)*np.ones((1, 6))
para_bound_gq = (gammaq_max-gammaq_min)*np.ones((1, 1))
para_bound_q  = (Q_max-Q_min)*np.ones((1, 3))
para_bound = np.hstack((para_bound_p, para_bound_gr, para_bound_r, para_bound_gq, para_bound_q))
para_jaco = np.diag(para_bound[0])


def para(para_bar):
    P_min = 0
    P_max = 1e-1
    gammar_min = 0
    gammar_max = 5e-1
    gammaq_min = 5e-1
    gammaq_max = 1
    R_min = 0
    R_max = 1e2
    Q_min = 0
    Q_max = 1e2
    tunable = np.zeros((1, 20))
    for i in range(20):
        tunable[0, i] = para_bar[i, 0]  # convert tensor to array
    tunable_para = np.zeros((1, 20))
    for i in range(9):
        tunable_para[0, i] = P_min + (P_max - P_min) * tunable[0, i]
    for i in range(6):
        tunable_para[0, i + 10] = R_min + (R_max - R_min) * tunable[0, i + 10]
    for i in range(3):
        tunable_para[0, i + 17] = Q_min + (Q_max - Q_min) * tunable[0, i + 17]
    tunable_para[0, 9] = gammar_min + (gammar_max - gammar_min) * tunable[0, 10]
    tunable_para[0, 16] = gammaq_min + (gammaq_max - gammaq_min) * tunable[0, 16]
    return tunable_para

# Parameterize control gain
D_in, D_h, D_out = 6, 20, 6
model_gain = uavNN.NetCtrl(D_in, D_h, D_out)
gp_min = 10  # lower bound of position control gain
gp_max = 25  # upper bound of position control gain
gv_min = 1e-1  # lower bound of attitude control gain, previous value was 1e-3
gv_max = 5   # upper bound of attitude control gain
ctrl_gain_jaco = np.diag([(gp_max-gp_min), (gp_max-gp_min), (gp_max-gp_min),
                          (gp_max-gp_min), (gp_max-gp_min), (gp_max-gp_min)])
def control_gain(nn_gain):
    ctrl_gain = np.zeros((1, 6))
    for i in range(6):
        if i<=2:
            ctrl_gain[0, i] = gp_min + (gp_max-gp_min)*nn_gain[i, 0]
        else:
            ctrl_gain[0, i] = gv_min + (gv_max-gv_min)*nn_gain[i, 0]
    return ctrl_gain


"""---------------------------------Define reference trajectory-----------------------------"""
# Target point for take-off
x_t, y_t, z_t = 0, 0, -1.5
dx_t, dy_t, dz_t = 0, 0, 0
ddx_t, ddy_t, ddz_t = 0, 0, 0
target = np.hstack((x_t, dx_t, ddx_t, y_t, dy_t, ddy_t, z_t, dz_t, ddz_t))

def polynomial_ref(t_end, target, initial_state):
    x_0, dx_0, ddx_0 = initial_state[0], 0, 0
    y_0, dy_0, ddy_0 = initial_state[1], 0, 0
    z_0, dz_0, ddz_0 = -0.1, 0, 0
    x_t, dx_t, ddx_t = target[0], target[1], target[2]
    y_t, dy_t, ddy_t = target[3], target[4], target[5]
    z_t, dz_t, ddz_t = target[6], target[7], target[8]
    x_con = np.vstack((x_0, dx_0, ddx_0, x_t, dx_t, ddx_t))
    y_con = np.vstack((y_0, dy_0, ddy_0, y_t, dy_t, ddy_t))
    z_con = np.vstack((z_0, dz_0, ddz_0, z_t, dz_t, ddz_t))
    A_p   = np.array([[0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 2, 0, 0],
                      [t_end**5, t_end**4, t_end**3, t_end**2, t_end, 1],
                      [5*t_end**4, 4*t_end**3, 3*t_end**2, 2*t_end, 1, 0],
                      [20*t_end**3, 12*t_end**2, 6*t_end, 2, 0, 0]])
    Coeff_x = np.matmul(LA.inv(A_p), x_con)
    Coeff_y = np.matmul(LA.inv(A_p), y_con)
    Coeff_z = np.matmul(LA.inv(A_p), z_con)
    return Coeff_x, Coeff_y, Coeff_z

def Reference(Coeff_x, Coeff_y, Coeff_z, time, t_end, target):
    # Position reference as a polynomial of time
    p_t  = np.array([[time**5, time**4, time**3, time**2, time, 1]])
    v_t  = np.array([[5*time**4, 4*time**3, 3*time**2, 2*time, 1, 0]])
    a_t  = np.array([[20*time**3, 12*time**2, 6*time, 2, 0, 0]])
    if time <= t_end:
        ref_p = np.array([[np.matmul(p_t, Coeff_x)[0, 0], np.matmul(p_t, Coeff_y)[0, 0], np.matmul(p_t, Coeff_z)[0, 0]]]).T
        ref_v = np.array([[np.matmul(v_t, Coeff_x)[0, 0], np.matmul(v_t, Coeff_y)[0, 0], np.matmul(v_t, Coeff_z)[0, 0]]]).T
        ref_a = np.array([[np.matmul(a_t, Coeff_x)[0, 0], np.matmul(a_t, Coeff_y)[0, 0], np.matmul(a_t, Coeff_z)[0, 0]]]).T
    else:
        ref_p = np.array([[target[0], target[3], target[6]]]).T
        ref_v = np.array([[target[1], target[4], target[7]]]).T
        ref_a = np.array([[target[2], target[5], target[8]]]).T

    b1_d = np.array([[1, 0, 0]]).T
    reference = {"ref_p": ref_p,
                 "ref_v": ref_v,
                 "ref_a": ref_a,
                 "b1_d": b1_d}
    return reference

"""---------------------------------Define controller---------------------------------------"""
# Controller
nn_gain0 = model_gain(np.zeros((6, 1)))
ctrl_gain0 = control_gain(nn_gain0)
GeoCtrl = Robust_Flight.Controller(Sys_para, uav.X, uav.Xmhe)
# Random initial x and y postions
P_xy0 = np.random.normal(0, 0, 2)
Coeff_x, Coeff_y, Coeff_z = polynomial_ref(t_end, target, P_xy0)
time0 = 0
ref0  = Reference(Coeff_x, Coeff_y, Coeff_z, time0, t_end, target)
ref_p0, ref_v0, ref_a0, b1_d0 = ref0['ref_p'], ref0['ref_v'], ref0['ref_a'], ref0['b1_d']
R_Bdin, Tfin, e_x, e_v, fd = GeoCtrl.position_ctrl(ctrl_gain0, state0, ref_p0, ref_v0, ref_a0, b1_d0, df_I0)
dR_Bdin, ddR_Bdin = np.zeros((3, 3)), np.zeros((3, 3))
tau_cin, w_Bdin, e_R, e_w = GeoCtrl.attitude_ctrl(dR_Bdin, ddR_Bdin, dt_B0)
ctrl0 = GeoCtrl.ctrl_mapping(Tfin, tau_cin)

"""---------------------------------Define MHE----------------------------------------------"""

uavMHE = Robust_Flight.MHE(horizon, dt_sample)
uavMHE.SetStateVariable(uav.Xmhe_p)
uavMHE.SetOutputVariable(uav.output_p)
uavMHE.SetControlVariable(uav.f_d)
uavMHE.SetRotationVariable(uav.R_B)
uavMHE.SetNoiseVariable(uav.wf)
uavMHE.SetModelDyn(uav.pdyn_mhe)
uavMHE.SetCostDyn()

"""---------------------------------Define DMHE----------------------------------------------"""
uavDMHE = Robust_Flight.Auxiliary_MHE(uav.Xmhe_p.numel(), uav.X.numel(), dt_sample, uav.pdyn, GeoCtrl.ctrl_gain_v,
                                      GeoCtrl.ctrl_gain_v.numel(), GeoCtrl.ref_v, GeoCtrl.ex_v, GeoCtrl.ev_v,
                                      uav.output_p, uav.f_d, GeoCtrl.fd_v, uav.Xmhe_p, GeoCtrl.mass, GeoCtrl.J_B, uav.R_B)

"""---------------------------------Training process-----------------------------"""
# Estimated disturbance
ratio = T_step/dt_sample
ratio_inv = int(1/ratio)
Df_Imh = np.zeros((3, int(N_ev/ratio_inv)+1))
Dt_Bmh = np.zeros((3, int(N_ev/ratio_inv)+1))
# Ground truth
Dis_t = np.zeros((6, int(N_ev/ratio_inv)+1))
D_G   = np.zeros((6, N_ev+1))
# Simulation record
r_Is = np.zeros((3, N_ev+1))
p_Is = np.zeros((3, N_ev+1))
v_Is = np.zeros((3, N_ev+1))
Euler = np.zeros((3, N_ev+1))
w_Bs = np.zeros((3, N_ev+1))
# Tunable parameters
Tunable_para = np.zeros(((int(N_ev/ratio_inv)+1), D_out_p))
# Control gain
Control_gain = np.zeros(((int(N_ev/ratio_inv)), 6))
# Loss for training and episode
LOSS = np.zeros((N_train, (int(N/ratio_inv)-horizon)))
J_loss = np.zeros((int(N/ratio_inv)-horizon))
def Train():
    # Loss function
    Loss = []
    # Training times
    Time_t = []
    # Load trajectory data
    filename = 'waypoints'
    traj_waypoints = np.loadtxt(filename)
    for i in range(N_train):
        # Initial time
        time = 0
        # Initial weighting matrix of the arrival cost
        P0 = 100 * np.identity(uav.Xmhe_p.numel())
        # Initial states
        P_xy0 = np.random.normal(0, 0, 2)
        print('P_xy0=', P_xy0)
        r_I0  = np.array([[P_xy0[0], P_xy0[1], -1]]).T
        state = np.vstack((r_I0, v_I0, R_Bv0, w_B0))

        x_hatmh = np.vstack((r_I0, v_I0, df_I0))
        xmhe_traj = x_hatmh
        R_Bd  = R_Bd0
        dR_Bd = dR_Bd0
        # Control force and torque list
        ctrl_f = []
        # Control list
        ctrl = []
        # Control gain list
        Ctrl_Gain =[]
        # Measurement list
        Y = []
        # Tracking error list
        track_e = []
        track_e += [np.zeros((6, 1))]
        # add measurement noise
        noise = np.random.normal(0, 1e-6, uav.X.numel())
        Noise = np.zeros((1, 18))
        for iN in range(18):
            Noise[0, iN] = noise[iN]
        state_m = state.T + Noise
        state_m = state_m.T
        # sample the measurement and ground truth
        y_p = state_m[0:6, 0]
        y_p = np.reshape(y_p, (6, 1))
        Y += [state_m]
        Y_p = []
        Y_p += [y_p]
        # Initialize the tunable parameters
        t_para = model_QR(track_e[-1])
        tunable_para0= para(t_para)
        print('learned', i, 'tunable_para0=', tunable_para0)
        # Reference list
        Ref = []
        # Rotation list
        R_B = []
        # Index for sampling
        k = 0
        # reset disturbance randomly
        dfxy = np.random.normal(0, 0.05, 2)
        dfz = np.random.normal(2, 0.2, 1)
        dt = np.random.normal(0, 0, 3)
        Afxy = np.random.normal(0, 0.1, 2)
        Afz  = np.random.normal(3, 0.5, 1)
        wf   = np.random.normal(10, 2, 1)
        At   = np.random.normal(0, 0, 1)
        tsquare = np.random.normal(0, 0.1, 5)
        amplify = np.random.normal(2.5, 0.5, 2)
        # Flag for breaking the episode
        flag = 0
        # Sum of loss
        sum_loss = 0.0
        # index for LOSS in each episode
        j_loss = 0
        payload_mass = np.random.normal(0.5, 0.1, 1)
        time_pay = np.random.normal(4.5, 0.1, 1)
        a_I_new = np.zeros((3, 1))
        kr = 0
        for j in range(N):
            if j % 5 == 0:
                ref_p = np.array([[traj_waypoints[kr, 0], traj_waypoints[kr, 1], -traj_waypoints[kr, 2]]]).T
                ref_v = np.array([[traj_waypoints[kr, 3], traj_waypoints[kr, 4], -traj_waypoints[kr, 5]]]).T
                ref_dv = np.array([[traj_waypoints[kr, 6], traj_waypoints[kr, 7], -traj_waypoints[kr, 8]]]).T
                b1_d = np.array([[1, 0, 0]]).T
                kr += 1
            if (j % ratio_inv) == 0:
                # Solve MHE to obtain the estimated state trajectory within a horizon
                opt_sol = uavMHE.MHEsolver(Y_p, track_e, x_hatmh, xmhe_traj, ctrl_f, model_QR, k, R_B)
                xmhe_traj = opt_sol['state_traj_opt']
                costate_traj = opt_sol['costate_traj_opt']
                # Establish the auxiliary MHE system
                auxSys = uavMHE.GetAuxSys(xmhe_traj, costate_traj, model_QR, Y_p, track_e)
                matA, matB, matH = auxSys['matA'], auxSys['matB'], auxSys['matH']
                matD, matE, matF = auxSys['matD'], auxSys['matE'], auxSys['matF']
                # Solve the auxiliary MHE system to obtain the gradient
                gra_opt = uavDMHE.AuxMHESolver(matA, matB, matD, matE, matF, matH, model_QR, track_e)
                X_opt = gra_opt['state_gra_traj']
                if time>(horizon*dt_sample):
                    # Update x_hatmh based on xmhe_traj
                    for ix in range(len(x_hatmh)):
                        x_hatmh[ix] = xmhe_traj[1, ix]
                else:
                    for ix in range(len(x_hatmh)):
                        x_hatmh[ix] = xmhe_traj[0, ix]

                # Position control
                df_Imh = np.transpose(xmhe_traj[-1, 6:9])
                df_Imh = np.reshape(df_Imh, (3, 1))
                dt_Bmh = np.zeros((3, 1))


                # Coeff_x, Coeff_y, Coeff_z = polynomial_ref(t_end, target, P_xy0)
                # ref = Reference(Coeff_x, Coeff_y, Coeff_z, time, t_end, target)
                # ref_p, ref_v, ref_dv, b1_d = ref['ref_p'], ref['ref_v'], ref['ref_a'], ref['b1_d']
                input_gain = np.vstack((track_e[-1]))
                nn_gain = model_gain(input_gain)
                ctrl_gain = control_gain(nn_gain)
                # ctrl_gain = np.array([[12, 12, 12, 2.5, 2.5, 2.5]])
                Ctrl_Gain += [ctrl_gain]
                print('sample=', k, 'control_gain=', ctrl_gain, 'reference=', ref_p.T)
                R_Bd_next, Tf, e_x, e_v, fd = GeoCtrl.position_ctrl(Ctrl_Gain[-1], Y[-1], ref_p, ref_v, ref_dv, b1_d, df_Imh)
                dR_Bd_next = (R_Bd_next - R_Bd) / dt_sample
                R_Bd = R_Bd_next  # update R_Bd0 for computing dR_Bd in the next iteration
                ddR_Bd_next = (dR_Bd_next - dR_Bd) / dt_sample
                dR_Bd = dR_Bd_next  # update dR_Bd0 for computing ddR_Bd in the next iteration
                # Attitude control
                tau_c, w_Bd, e_R, e_w = GeoCtrl.attitude_ctrl(dR_Bd_next, ddR_Bd_next, dt_Bmh)
                # Store tracking error
                error_track = np.vstack((e_x, e_v))
                track_e += [error_track]
                # Map the control force and torque to the square of the rotor speed
                u = GeoCtrl.ctrl_mapping(Tf, tau_c)
                # Limit the rotor speed
                u = np.clip(u, U_lower, U_upper)
                # U_lower = 0.9*U_lower
                ctrl_f += [fd]
                ctrl += [u]
                # Store the rotation
                R_bc = np.array([
                    [Y[-1][6, 0], Y[-1][7, 0], Y[-1][8, 0]],
                    [Y[-1][9, 0], Y[-1][10, 0], Y[-1][11, 0]],
                    [Y[-1][12, 0], Y[-1][13, 0], Y[-1][14, 0]]]
                )
                R_B  += [R_bc]
                # Store the reference
                # R_Bd_h = np.array([[R_Bd[0, 0], R_Bd[0, 1], R_Bd[0, 2],
                #                     R_Bd[1, 0], R_Bd[1, 1], R_Bd[1, 2],
                #                     R_Bd[2, 0], R_Bd[2, 1], R_Bd[2, 2]]])
                ref = np.hstack((np.transpose(ref_p), np.transpose(ref_v)))
                Ref  += [ref]

            # Take a next step from the environment
            a_noise = np.random.normal(0, 1e-3, 3)
            if time > time_pay[0]:
                p_mass = 1/2*payload_mass[0]
            else:
                p_mass = payload_mass[0]
            dis, T = uav.Aerodynamics(state, a_I_new, ctrl[-1], constant_dis, a_noise, p_mass, None)

            print('sample=', k, 'Tension=', T)
            print('sample=', k, 'Dis_x=', dis[0, 0], 'dt_Imh_x=', df_Imh[0, 0], 'Dis_z=', dis[2, 0], 'df_Imh_z=', df_Imh[2, 0])
            zeros_rotor = np.where(ctrl[-1]==0)[0]
            if np.size(zeros_rotor)>=2:
                flag += 1
            else:
                flag = 0
            if flag >= 20:
                break
            output = uav.step(state, ctrl[-1], dis, T_step)
            # Update state
            state_next = output['state_new']
            a_I_next = output['a_I_new']
            if (j % ratio_inv) == 0:
                k += 1
                # Take measurement
                noise = np.random.normal(0, 1e-6, uav.X.numel())
                Noise = np.zeros((1, 18))
                for iN in range(18):
                    Noise[0, iN] = noise[iN]
                state_m = state.T + Noise
                state_m = state_m.T
                # sample the measurement and ground truth
                y_p = state_m[0:6, 0]
                y_p = np.reshape(y_p, (6, 1))
                Y += [state_m]
                Y_p += [y_p]
                # Compute the gradient of loss
                dp, dp_ctrl, loss_track = uavDMHE.ChainRule(Ctrl_Gain, Y_p, Ref, xmhe_traj, X_opt, R_B)
                dldp = np.matmul(dp, para_jaco)
                dldg = np.matmul(dp_ctrl, ctrl_gain_jaco)
                # Train the neural network
                t_para = model_QR(track_e[-1])
                loss_nn_p = model_QR.myloss(t_para, dldp)
                loss_nn_g = model_gain.myloss(nn_gain, dldg)
                optimizer_p = torch.optim.Adam(model_QR.parameters(), lr=lr_nn)
                optimizer_g = torch.optim.Adam(model_gain.parameters(), lr=lr_nn)
                model_QR.zero_grad()
                model_gain.zero_grad()
                loss_nn_p.backward()
                loss_nn_g.backward()
                # Update tunable parameter and control gain
                optimizer_p.step()
                optimizer_g.step()

                # Sum the loss
                loss_track = np.reshape(loss_track, (1))
                sum_loss += loss_track
                if time>=horizon*dt_sample:
                    J_loss[j_loss]  = j_loss
                    LOSS[i, j_loss] = loss_track
                    print('sample=', k, 'loss=', loss_track)
                    j_loss += 1

            r_I = output['r_I_new']
            v_I = output['v_I_new']
            euler = output['Euler_new']
            # Update time and state
            time += T_step
            state = state_next
            a_I_new = a_I_next
            print('learning=', i+1, 'time=', time, 'position', r_I.T, 'control=', ctrl[-1].T, 'attitude', euler)
            if j == (N-1):
                # Save the loss and training time
                Time_t += [i+1]
                mean_loss = sum_loss/k
                Loss += [mean_loss]
                print('learned', i + 1, 'mean_loss=', mean_loss)
                np.save('Loss', Loss)
                np.save('Time_t', Time_t)

        # Save the trained NN weights
        PATH1 = "Trained_model_QR.pt"
        torch.save(model_QR, PATH1)
        PATH2 = "Trained_model_gain.pt"
        torch.save(model_gain, PATH2)
        np.save('LOSS', LOSS)
        np.save('J_loss', J_loss)


"""---------------------------------Evaluation process-----------------------------"""
def Evaluate():
    # Initial time
    time = 0
    Time = np.zeros(N_ev + 1)
    Time[0] = 0
    # Load trajectory data
    filename = 'waypoints'
    traj_waypoints = np.loadtxt(filename)
    # Initial weighting matrix of the arrival cost
    P0 = 100 * np.identity(uav.Xmhe_p.numel())
    # Initial states
    P_xy0 = np.random.normal(0, 0, 2)
    np.save('P_xy0', P_xy0)
    r_I0 = np.array([[P_xy0[0], P_xy0[1], -1]]).T
    state = np.vstack((r_I0, v_I0, R_Bv0, w_B0))
    x_hatmh = np.vstack((r_I0, v_I0, df_I0))
    xmhe_traj = x_hatmh
    R_Bd = R_Bd0
    dR_Bd = dR_Bd0
    # Load the neural network model
    PATH1 = "Trained_model_QR.pt"
    model_QR = torch.load(PATH1)
    PATH2 = "Trained_model_gain.pt"
    model_gain = torch.load(PATH2)
    # Control force and torque list
    ctrl_f = []
    # Control list
    ctrl = []
    # Rotation list
    R_B = []
    # Measurement list
    Y = []
    # add measurement noise
    noise = np.random.normal(0, 1e-6, uav.X.numel())
    Noise = np.zeros((1, 18))
    for iN in range(18):
        Noise[0, iN] = noise[iN]
    state_m = state.T + Noise
    state_m = state_m.T
    # sample the measurement and ground truth
    y_p = state_m[0:6, 0]
    y_p = np.reshape(y_p, (6, 1))
    Y += [state_m]
    Y_p = []
    Y_p += [y_p]
    # Tracking error list
    track_e = []
    track_e += [np.zeros((6, 1))]
    # Initialize the tunable parameters
    t_para = model_QR(track_e[-1])
    tunable_para0 = para(t_para)
    Tunable_para[0, :] = tunable_para0
    print('learned', 0, 'tunable_para0=', tunable_para0)

    # Set initial disturbance
    ctrl0 = np.zeros((4, 1))
    payload_mass = 0.5
    a_I_new = np.zeros((3, 1))
    a_noise = np.random.normal(0, 1e-3, 3)
    dis0, T = uav.Aerodynamics(state, a_I_new, ctrl0, constant_dis, a_noise, payload_mass, None)
    # Store initial conditions
    r_Is[:, 0:1] = r_I0
    p_Is[:, 0:1] = p_I0
    v_Is[:, 0:1] = v_I0
    Euler[:, 0:1] = Euler_0
    w_Bs[:, 0:1] = w_B0
    Dis_t[:, 0:1] = dis0
    Df_Imh[:, 0:1] = df_I0
    Dt_Bmh[:, 0:1] = dt_B0
    D_G[:, 0:1] = dis0
    # Flag for breaking the episode
    flag = 0
    # Index for sampling
    k = 0
    time_s = 0
    Time_s = np.zeros((int(N_ev /ratio_inv) + 1))
    Time_s[0] = time_s
    # Store altitude tracking error
    eZ = np.zeros((int(N_ev/ratio_inv) + 1))
    eZ[0] = 0
    # Store position
    ref_x = np.zeros((int(N_ev/ratio_inv) + 1))
    ref_y = np.zeros((int(N_ev/ratio_inv) + 1))
    ref_z = np.zeros((int(N_ev / ratio_inv) + 1))
    act_x = np.zeros((int(N_ev/ratio_inv) + 1))
    act_y = np.zeros((int(N_ev/ratio_inv) + 1))
    act_z = np.zeros((int(N_ev / ratio_inv) + 1))
    ref_x[0] = 0
    ref_y[0] = 0
    ref_z[0] = -1
    act_x[0] = 0
    act_y[0] = 0
    act_z[0] = -1
    # open a file
    a_file = open("flight_test.txt", "w")
    kr = 0
    for j in range(N_ev):
        if j % 5 == 0:
            ref_p = np.array([[traj_waypoints[kr, 0], traj_waypoints[kr, 1], -traj_waypoints[kr, 2]]]).T
            ref_v = np.array([[traj_waypoints[kr, 3], traj_waypoints[kr, 4], -traj_waypoints[kr, 5]]]).T
            ref_dv = np.array([[traj_waypoints[kr, 6], traj_waypoints[kr, 7], -traj_waypoints[kr, 8]]]).T
            b1_d = np.array([[1, 0, 0]]).T
            kr +=1
        if (j % ratio_inv) == 0:
            # Solve MHE to obtain the estimated state trajectory within a horizon
            opt_sol = uavMHE.MHEsolver(Y_p, track_e, x_hatmh, xmhe_traj, ctrl_f, model_QR, k, R_B)
            xmhe_traj = opt_sol['state_traj_opt']
            costate_traj = opt_sol['costate_traj_opt']
            # Establish the auxiliary MHE system
            auxSys = uavMHE.GetAuxSys(xmhe_traj, costate_traj, model_QR, Y_p, track_e)
            matA, matB, matH = auxSys['matA'], auxSys['matB'], auxSys['matH']
            matD, matE, matF = auxSys['matD'], auxSys['matE'], auxSys['matF']
            if time > (horizon * dt_sample):
                # Update x_hatmh based on xmhe_traj
                for ix in range(len(x_hatmh)):
                    x_hatmh[ix] = xmhe_traj[1, ix]
            else:
                for ix in range(len(x_hatmh)):
                    x_hatmh[ix] = xmhe_traj[0, ix]

            # Position control
            df_Imh = np.transpose(xmhe_traj[-1, 6:9])
            df_Imh = np.reshape(df_Imh, (3, 1))
            dt_Bmh = np.zeros((3, 1))
            # Coeff_x, Coeff_y, Coeff_z = polynomial_ref(t_end, target, P_xy0)
            # ref = Reference(Coeff_x, Coeff_y, Coeff_z, time, t_end, target)
            # ref_p, ref_v, ref_dv, b1_d = ref['ref_p'], ref['ref_v'], ref['ref_a'], ref['b1_d']
            input_gain = np.vstack((track_e[-1]))
            nn_gain = model_gain(input_gain)
            ctrl_gain = control_gain(nn_gain)
            # ctrl_gain = np.array([[12, 12, 12, 2.5, 2.5, 2.5]]) # 12, 12, 12, 2.5, 2.5, 2.5 for T_step = 1e-2, dt_sample = 5e-2
            print('sample=', k, 'control_gain=', ctrl_gain)
            Control_gain[k, :] = ctrl_gain
            R_Bd_next, Tf, e_x, e_v, fd = GeoCtrl.position_ctrl(ctrl_gain, Y[-1], ref_p, ref_v, ref_dv, b1_d, df_Imh)
            dR_Bd_next = (R_Bd_next - R_Bd) / dt_sample
            R_Bd = R_Bd_next  # update R_Bd0 for computing dR_Bd in the next iteration
            ddR_Bd_next = (dR_Bd_next - dR_Bd) / dt_sample
            dR_Bd = dR_Bd_next  # update dR_Bd0 for computing ddR_Bd in the next iteration
            # Attitude control
            tau_c, w_Bd, e_R, e_w = GeoCtrl.attitude_ctrl(dR_Bd_next, ddR_Bd_next, dt_Bmh)
            # Store tracking error
            error_track = np.vstack((e_x, e_v))
            track_e += [error_track]
            # Map the control force and torque to the square of the rotor speed
            u = GeoCtrl.ctrl_mapping(Tf, tau_c)
            # Limit the rotor speed
            u = np.clip(u, U_lower, U_upper)
            ctrl += [u]
            ctrl_f += [fd]
            # save fd and b1_d into txt. file
            save_data = np.hstack((fd.T, b1_d.T))
            np.savetxt(a_file, save_data)
            # Store the rotation
            R_bc = np.array([
                [Y[-1][6, 0], Y[-1][7, 0], Y[-1][8, 0]],
                [Y[-1][9, 0], Y[-1][10, 0], Y[-1][11, 0]],
                [Y[-1][12, 0], Y[-1][13, 0], Y[-1][14, 0]]]
            )
            R_B += [R_bc]

        # Take a next step from the environment
        a_noise = np.random.normal(0, 1e-3, 3)
        if time>6:
            p_mass = 0*payload_mass
        else:
            p_mass = payload_mass
        dis, T = uav.Aerodynamics(state, a_I_new, ctrl[-1], constant_dis, a_noise, p_mass, None)

        # Store disturbance ground truth
        D_G[:, (j+1):(j+2)] = dis
        print('sample=', k, 'Dis=', dis[2, 0], 'df_Imh=', df_Imh[2, 0], 'Dis_t=', dis[3, 0], 'dt_Bmh=', dt_Bmh[0, 0])
        zeros_rotor = np.where(ctrl[-1] == 0)[0]
        if np.size(zeros_rotor) >= 2:
            flag += 1
        else:
            flag = 0
        if flag >= 20:
            break
        output = uav.step(state, ctrl[-1], dis, T_step)
        # Update state
        state_next = output['state_new']
        a_I_next = output['a_I_new']
        if (j % ratio_inv) == 0:
            # Take measurement
            noise = np.random.normal(0, 1e-6, uav.X.numel())
            Noise = np.zeros((1, 18))
            for iN in range(18):
                Noise[0, iN] = noise[iN]
            state_m = state.T + Noise
            state_m = state_m.T
            # sample the measurement and ground truth
            y_p = state_m[0:6, 0]
            y_p = np.reshape(y_p, (6, 1))
            Y += [state_m]
            Y_p += [y_p]
            # Update sampling index, time and tunable_para
            k += 1
            time_s += dt_sample
            Time_s[k] = time_s
            t_para = model_QR(Y_p[-1])
            tunable_para = para(t_para)
            Tunable_para[k, :] = tunable_para
            # Store loss function and estimated disturbance
            Df_Imh[:, k:(k + 1)] = df_Imh
            Dt_Bmh[:, k:(k + 1)] = dt_Bmh
            Dis_t[:, k:(k + 1)] = dis
            eZ[k] = ref_p[2, 0] - state[2, 0]
            ref_x[k] = ref_p[0, 0]
            ref_y[k] = ref_p[1, 0]
            ref_z[k] = ref_p[2, 0]
            act_x[k] = state[0, 0]
            act_y[k] = state[1, 0]
            act_z[k] = state[2, 0]

        r_I = output['r_I_new']
        v_I = output['v_I_new']
        euler = output['Euler_new']
        w_B = output['w_B_new']
        r_Is[:, (j + 1):(j + 2)] = r_I #(j + 1):(j + 2)
        v_Is[:, (j + 1):(j + 2)] = v_I
        Euler[:, (j + 1):(j + 2)] = euler
        w_Bs[:, (j + 1):(j + 2)] = w_B
        # Update time and state
        time += T_step
        state = state_next
        a_I_new = a_I_next
        Time[j + 1] = time
        print('time=', time, 'sample=', k, 'position=', r_I.T, 'control=', ctrl[-1].T, 'attitude', euler)
    np.save('Dis_t', Dis_t)
    np.save('D_G', D_G)
    np.save('r_Is', r_Is)
    np.save('Df_Imh', Df_Imh)
    np.save('Dt_Bmh', Dt_Bmh)
    np.save('Time_s', Time_s)
    np.save('eZ', eZ)
    np.save('Tunable_para', Tunable_para)
    np.save('control_gain', Control_gain)
    np.save('ref_x', ref_x)
    np.save('ref_y', ref_y)
    np.save('act_x', act_x)
    np.save('act_y', act_y)
    a_file.close()


    """
    Plot figures
    """
    # Loss function
    Time_t = np.load('Time_t.npy')
    Loss   = np.load('Loss.npy')
    plt.figure(1)
    plt.plot(Time_t, Loss, linewidth=1.5)
    plt.xlabel('Number of training')
    plt.ylabel('Mean loss')
    plt.grid()
    plt.savefig('./mean_loss_train.png')
    plt.show()
    # Loss for each episode
    jloss = np.load('J_loss.npy')
    loss_ep = np.load('LOSS.npy')
    plt.figure(2)
    plt.plot(jloss, loss_ep[-1, :], linewidth=1.5)
    plt.xlabel('Times')
    plt.ylabel('Loss for the last training')
    plt.grid()
    plt.savefig('./loss_train.png')
    plt.show()
    # Disturbance
    plt.figure(3)
    plt.plot(Time_s, Dis_t[2, :], linewidth=1, linestyle='--')
    plt.plot(Time_s, Df_Imh[2, :], linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Disturbance force in z axis')
    plt.legend(['Ground truth', 'MHE estimation'])
    plt.grid()
    plt.savefig('./disturbance_train.png')
    plt.show()
    # Position
    plt.figure(4)
    plt.plot(Time, r_Is[2, :], linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Altitude in z axis [m]')
    plt.grid()
    plt.savefig('./altitude_train.png')
    plt.show()
    # Tunable parameters
    plt.figure(5)
    plt.plot(Time_s, Tunable_para[:, 9], linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Gamma_r')
    plt.grid()
    plt.savefig('./tunable_para_train.png')
    plt.show()
    # Control gain
    plt.figure(6)
    plt.plot(Time_s[0:int(N_ev/ratio_inv)], Control_gain[:, -6], linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Position control gain')
    plt.grid()
    plt.savefig('./control_gain_train.png')
    plt.show()
    # Altitude tracking error
    plt.figure(7)
    ax = plt.axes(projection="3d")
    ax.plot3D(act_x, act_y, -act_z, linewidth=1.5)
    ax.plot3D(ref_x, ref_y, -ref_z, linewidth=1, linestyle='--')
    plt.legend(['Actual', 'Desired'])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.savefig('./tracking_3D.png')
    plt.show()
    plt.figure(8)
    plt.plot(act_x, act_y,  linewidth=1.5)
    plt.plot(ref_x, ref_y, linewidth=1, linestyle='--')
    plt.legend(['Actual', 'Desired'])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.savefig('./tracking_2D.png')
    plt.show()


"""---------------------------------Main function-----------------------------"""
if train:
    Train()
    Evaluate()
else:
    Evaluate()