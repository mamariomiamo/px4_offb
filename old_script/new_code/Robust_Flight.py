"""
This file includes 3 classes that define the controller, MHE and DMHE respectively
--------------------------------------------------------------------------------------
Wang Bingheng, 19 Dec. 2020 at Advanced Control Lab, ECE Dept. NUS
"""

from casadi import *
from numpy import linalg as LA
import numpy as np
import math

class Controller:
    """
    Geometric flight controller based on SE(3)
    """
    def __init__(self, para, state_var, xmhe_var):
        # Control gain variables
        self.ctrl_gain_v = horzcat(SX.sym('kpx'), SX.sym('kpy'), SX.sym('kpz'),
                                   SX.sym('kvx'), SX.sym('kvy'), SX.sym('kvz'))
        # Inertial parameters
        self.mass = para[0]
        jx, jy, jz = para[1], para[2], para[3]
        self.J_B = np.diag([jx, jy, jz])
        # Parameters of the control mapping (interface)
        l, b, kt = para[4], para[5], para[6]
        # Control mapping
        self.B = vertcat(
            horzcat(b, b, b, b),
            horzcat(0, b * l, 0, -b * l),
            horzcat(b * l, 0, -b * l, 0),
            horzcat(kt, -kt, kt, -kt)
        )
        # State variable (18-dimension)
        self.state_v = state_var
        # Disturbance force variable in initial frame
        self.df_I_v  = vertcat(xmhe_var[6], xmhe_var[7], xmhe_var[8])
        # Disturbance torque variable in body frame
        self.dt_B_v  = vertcat(xmhe_var[21], xmhe_var[22], xmhe_var[23])

    def skew(self, v):
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross

    def vee_map(self, v):
        vect = np.array([[v[2, 1], v[0, 2], v[1, 0]]]).T
        return vect

    def trace(self, v):
        v_trace = v[0, 0] + v[1, 1] + v[2, 2]
        return v_trace

    def position_ctrl(self, ctrl_gain, state, ref_p, ref_v, ref_dv, b1_d, df_Imh):
        self.kpx, self.kpy, self.kpz = ctrl_gain[0, 0], ctrl_gain[0, 1], ctrl_gain[0, 2]
        self.kvx, self.kvy, self.kvz = ctrl_gain[0, 3], ctrl_gain[0, 4], ctrl_gain[0, 5]
        self.kax, self.kay, self.kaz = 5, 5, 5 # 3.5, 3.5, 3.5 for T_step = 1e-2, dt_sample = 5e-2
        self.kwx, self.kwy, self.kwz = 3.5, 3.5, 3.5 # 2.5, 2.5, 2.5
        self.kp = np.diag([self.kpx, self.kpy, self.kpz])
        self.kv = np.diag([self.kvx, self.kvy, self.kvz])
        self.ka = np.diag([self.kax, self.kay, self.kaz])
        self.kw = np.diag([self.kwx, self.kwy, self.kwz])
        self.kp_vh = horzcat(self.ctrl_gain_v[0, 0], self.ctrl_gain_v[0, 1], self.ctrl_gain_v[0, 2])
        self.kp_v = diag(self.kp_vh)
        self.kv_vh = horzcat(self.ctrl_gain_v[0, 3], self.ctrl_gain_v[0, 4], self.ctrl_gain_v[0, 5])
        self.kv_v = diag(self.kv_vh)
        # self.ka_vh = horzcat(self.ctrl_gain_v[0, 6], self.ctrl_gain_v[0, 7], self.ctrl_gain_v[0, 8])
        # self.ka_v = diag(self.ka_vh)
        # self.kw_vh = horzcat(self.ctrl_gain_v[0, 9], self.ctrl_gain_v[0, 10], self.ctrl_gain_v[0, 11])
        # self.kw_v = diag(self.kw_vh)
        # States
        self.r_I = np.array([[state[0, 0], state[1, 0], state[2, 0]]]).T
        self.r_I_v = vertcat(self.state_v[0], self.state_v[1], self.state_v[2])
        self.v_I = np.array([[state[3, 0], state[4, 0], state[5, 0]]]).T
        self.v_I_v = vertcat(self.state_v[3], self.state_v[4], self.state_v[5])
        self.R_B = np.array([
            [state[6, 0], state[7, 0], state[8, 0]],
            [state[9, 0], state[10, 0], state[11, 0]],
            [state[12, 0], state[13, 0], state[14, 0]]]
        )
        self.R_B_v = vertcat(
            horzcat(self.state_v[6], self.state_v[7], self.state_v[8]),
            horzcat(self.state_v[9], self.state_v[10], self.state_v[11]),
            horzcat(self.state_v[12], self.state_v[13], self.state_v[14])
        )
        self.w_B = np.array([[state[15, 0], state[16, 0], state[17, 0]]]).T
        self.w_B_v = vertcat(self.state_v[15], self.state_v[16], self.state_v[17])

        # Position tracking error
        e_x = self.r_I - ref_p
        self.ref_p_v = SX.sym('ref_p', 3, 1)
        self.ex_v = self.r_I_v - self.ref_p_v
        e_v = self.v_I - ref_v
        self.ref_v_v = SX.sym('ref_v', 3, 1)
        self.ev_v = self.v_I_v - self.ref_v_v
        # Global parameter
        g = 9.81
        # Z direction vector free of coordinate
        z = np.array([[0, 0, 1]]).T
        # Desired control force
        fd = np.matmul(self.kp, e_x) + np.matmul(self.kv, e_v) + self.mass*g*z - self.mass*ref_dv + df_Imh
        self.fd_v = mtimes(self.kp_v, self.ex_v) + mtimes(self.kv_v, self.ev_v) + self.mass*g*z - self.mass*ref_dv + self.df_I_v
        # Magnitude of the desired force (2-norm)
        fdm = LA.norm(fd)
        # Total thrust (project of the desired thrust to body z axis expressed in inertial frame)
        Tf = dot(fd, mtimes(self.R_B, z))
        self.Tf_v = dot(self.fd_v, mtimes(self.R_B_v, z))
        # Desired direction of body z axis in inertial frame
        self.b3_d = fd/fdm
        # Desired rotation matrix R_Bd
        b2_d = np.matmul(self.skew(self.b3_d), b1_d)
        self.R_Bd = np.hstack((b1_d, b2_d, self.b3_d))
        self.R_Bd_v = vertcat(
            horzcat(SX.sym('r11_d'), SX.sym('r12_d'), SX.sym('r13_d')),
            horzcat(SX.sym('r21_d'), SX.sym('r22_d'), SX.sym('r23_d')),
            horzcat(SX.sym('r31_d'), SX.sym('r32_d'), SX.sym('r33_d'))
        )
        return self.R_Bd, Tf, e_x, e_v, fd

    def attitude_ctrl(self, dR_Bd, ddR_Bd, dt_Bmh):
        # dR_Bd: time-derivative of desired rotation matrix R_Bd
        # ddR_Bd: time-derivative of dR_Bd
        # dt_Bmh: disturbance torques in body frame estimated by MHE

        # Desired angular velocity
        w_Bd = self.vee_map(np.matmul(np.transpose(self.R_Bd), dR_Bd))
        self.w_Bd_v = SX.sym('w_Bd', 3, 1)
        # Desired angular acceleration
        dw_Bd = self.vee_map(mtimes(np.transpose(self.R_Bd), ddR_Bd) - mtimes(self.skew(w_Bd), self.skew(w_Bd)))

        # Attitude tracking error
        e_R = 1/2*self.vee_map(mtimes(np.transpose(self.R_Bd), self.R_B) - mtimes(np.transpose(self.R_B), self.R_Bd))
        self.eR_v = 1/2*self.vee_map(mtimes(transpose(self.R_Bd_v), self.R_B_v) - mtimes(transpose(self.R_B_v), self.R_Bd_v))
        e_w = self.w_B - np.matmul(np.matmul(np.transpose(self.R_B), self.R_Bd), w_Bd)
        self.error_func_v = 1/2*self.trace(np.identity(3)-mtimes(transpose(self.R_Bd_v), self.R_B_v))
        self.ew_v = self.w_B_v - mtimes(mtimes(transpose(self.R_B_v), self.R_Bd_v), self.w_Bd_v)
        # Desired control torque
        tau_c = - np.matmul(self.ka, e_R) - np.matmul(self.kw, e_w) + np.matmul(self.skew(self.w_B), np.matmul(self.J_B, self.w_B)) - dt_Bmh - \
                np.matmul(self.J_B, np.matmul(np.matmul(self.skew(self.w_B), np.transpose(self.R_B)), np.matmul(self.R_Bd, w_Bd)) - \
                       np.matmul(np.transpose(self.R_B), np.matmul(self.R_Bd, dw_Bd)))
        # self.tau_c_v = -mtimes(self.ka_v, self.eR_v) - mtimes(self.kw_v, self.ew_v) + mtimes(self.skew(self.w_B_v), mtimes(self.J_B, self.w_B_v)) - self.dt_B_v - \
        #         mtimes(self.J_B, mtimes(mtimes(self.skew(self.w_B_v), transpose(self.R_B_v)), mtimes(self.R_Bd_v, self.w_Bd_v)) - \
        #                mtimes(transpose(self.R_B_v), mtimes(self.R_Bd_v, dw_Bd)))
        self.ref_v = vertcat(self.ref_p_v, self.ref_v_v)
        return tau_c, w_Bd, e_R, e_w

    def ctrl_mapping(self, Tf, tau_c):
        assert hasattr(self, 'Tf_v'), "Define the position control first!"
        ctrl_wrench = vertcat(Tf, tau_c)
        # self.ctrl_wrench_v = vertcat(self.Tf_v, self.tau_c_v)
        # Square of motor speed as control
        u = mtimes(inv(self.B), ctrl_wrench)
        # self.u_v = mtimes(inv(self.B), self.ctrl_wrench_v)
        return u

class MHE:
    def __init__(self, horizon, dt_sample):
        self.N = horizon
        self.DT = dt_sample

    def SetStateVariable(self, state_mhe):
        self.state = state_mhe
        self.n_state = state_mhe.numel()

    def SetOutputVariable(self, output):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.output = output
        self.n_output = output.numel()
        self.H = jacobian(self.output, self.state)

    def SetControlVariable(self, ctrlv):
        self.ctrl = ctrlv
        self.n_ctrl = ctrlv.numel()

    def SetRotationVariable(self, rotation):
        self.R_B_v = rotation

    def SetNoiseVariable(self, noise):
        self.noise = noise
        self.n_noise = noise.numel()

    def SetModelDyn(self, ode_mh):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'ctrl'), "Define the control variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # discrete-time dynamic model based on Euler-method
        self.ModelDyn = self.state + self.DT*ode_mh
        self.MDyn_fn  = Function('MDyn', [self.state, self.ctrl, self.noise, self.R_B_v], [self.ModelDyn],
                                 ['s', 'c', 'n', 'R_B'], ['MDynf'])

    def SetArrivalCost(self, x_hatmh):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.P0_v = diag(self.tunable_para[0, 0:9])
        error_a = self.state - x_hatmh
        self.cost_a = 1/2 * mtimes(mtimes(transpose(error_a), self.P0_v), error_a)
        self.cost_a_fn = Function('cost_a', [self.state, self.tunable_para], [self.cost_a], ['s', 'tp'], ['cost_af'])

    def SetCostDyn(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # Tunable parameters
        self.tunable_para = SX.sym('t_para', 1, 20)
        self.horizon1 = SX.sym('h1') # horizon - 1
        self.horizon2 = self.horizon1 - 1 # horizon - 2
        self.index = SX.sym('ki')
        self.gamma_r = self.tunable_para[0, 9]
        self.gamma_q = self.tunable_para[0, 16]
        self.R = diag(self.tunable_para[0, 10:16])*self.gamma_r**(self.horizon1-self.index)
        self.Q = diag(self.tunable_para[0, 17:20])*self.gamma_q**(self.horizon2-self.index)
        # Measurement variable
        self.measurement = SX.sym('y', self.n_output, 1)

        # Discrete dynamics of the running cost (time-derivative of the running cost) based on Euler-method
        error_running = self.measurement - mtimes(self.H, self.state)
        self.dJ_running = 1/2*(mtimes(mtimes(error_running.T, self.R), error_running) +
                               mtimes(mtimes(self.noise.T, self.Q), self.noise))
        self.dJ_fn = Function('dJ_running', [self.state, self.measurement, self.noise, self.tunable_para, self.horizon1, self.index],
                              [self.dJ_running], ['s', 'm', 'n', 'tp', 'h1', 'ind'], ['dJrunf'])
        self.dJ_T  = 1/2*mtimes(mtimes(error_running.T, self.R), error_running)
        self.dJ_T_fn = Function('dJ_T', [self.state, self.measurement, self.tunable_para, self.horizon1, self.index], [self.dJ_T],
                                ['s', 'm', 'tp', 'h1', 'ind'], ['dJ_Tf'])

    def para(self, para_bar):
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
            tunable[0, i] = para_bar[i, 0] # convert tensor to array
        tunable_para = np.zeros((1, 20))
        for i in range(9):
            tunable_para[0, i] = P_min + (P_max - P_min)*tunable[0, i]
        for i in range(6):
            tunable_para[0, i+10] = R_min + (R_max - R_min) * tunable[0, i+10]
        for i in range(3):
            tunable_para[0, i+17] = Q_min + (Q_max - Q_min) * tunable[0, i+17]
        tunable_para[0, 9] = gammar_min + (gammar_max - gammar_min)*tunable[0, 10]
        tunable_para[0, 16]= gammaq_min + (gammaq_max - gammaq_min)*tunable[0, 16]
        return  tunable_para


    def MHEsolver(self, Y, track_e, x_hatmh, xmhe_traj, ctrl, model_QR, time, R_b, print_level=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # arrival cost setting
        self.SetArrivalCost(x_hatmh)
        """
        Formulate MHE as a nonlinear programming problem solved by CasADi nlpsol() function
        """
        # Start with an empty NLP
        w   = [] # optimal trajectory list
        w0  = [] # initial guess of optimal trajectory
        lbw = [] # lower boundary of optimal variables
        ubw = [] # upper boundary of optimal variables
        g   = [] # equality or inequality constraints
        lbg = [] # lower boundary of constraints
        ubg = [] # upper boundary of constraints

        # Initial state for the arrival cost
        Xk  = SX.sym('X0', self.n_state, 1)
        w  += [Xk]
        X_hatmh = []
        for i in range(len(x_hatmh)):
            X_hatmh += [x_hatmh[i]]
        w0 += X_hatmh
        lbw+= self.n_state*[-1e20]
        ubw+= self.n_state*[1e20]
        # Formulate the NLP
        # time_mhe = self.N*self.DT
        if time < self.N:
            # Full-information estimator
            self.horizon = time + 1
        else:
            # Moving horizon estimation
            self.horizon = self.N

        t_para = model_QR(track_e[-1])
        tunable_para = self.para(t_para)
        J = self.cost_a_fn(s=Xk, tp=tunable_para)['cost_af']

        for k in range(self.horizon-1):
            # New NLP variables for the process noise
            Nk   = SX.sym('N_' + str(k), self.n_noise, 1)
            w   += [Nk]
            lbw += self.n_noise*[-1e20]
            ubw += self.n_noise*[1e20]
            w0  += self.n_noise*[0] # because of zero-mean noise

            # Integrate the cost function till the end of horizon
            J    += self.dJ_fn(s=Xk, m=Y[len(Y)-self.horizon+k], n=Nk, tp=tunable_para, h1=self.horizon-1, ind=k)['dJrunf']
            Xnext = self.MDyn_fn(s=Xk, c=ctrl[len(ctrl)-self.horizon+1+k], n=Nk, R_B=R_b[len(R_b)-self.horizon+1+k])['MDynf']
            # Next state based on the discrete model dynamics and current state
            Xk    = SX.sym('X_' + str(k + 1), self.n_state, 1)
            w    += [Xk]
            lbw  += self.n_state*[-1e20]
            ubw  += self.n_state*[1e20]
            if k == 0:
                w0 += X_hatmh
            else:
                X_guess = []
                for ix in range(self.n_state):
                    X_guess += [xmhe_traj[k, ix]]
                w0 += X_guess

            # Add equality constraint
            g    += [Xnext - Xk]
            lbg  += self.n_state*[0]
            ubg  += self.n_state*[0]

        # Add the final cost
        t_para = model_QR(track_e[-1])
        tunable_para = self.para(t_para)
        J += self.dJ_T_fn(s=Xk, m=Y[-1], tp=tunable_para, h1=self.horizon-1, ind=self.horizon-1)['dJ_Tf']

        # Create an NLP solver
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # Take the optimal noise and state
        sol_traj1 = np.concatenate((w_opt, self.n_noise * [0]))
        sol_traj = np.reshape(sol_traj1, (-1, self.n_state + self.n_noise))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        noise_traj_opt = np.delete(sol_traj[:, self.n_state:], -1, 0)

        # compute the co-states based on the noise_traj_opt
        B_bar = self.DT*np.identity(self.n_state)
        costate_traj_opt = np.zeros((self.horizon, self.n_state))
        for kc in range(self.horizon-1):
            # last co-state is zero, so the iteration ends at horizon-1
            w_opt_bar = vertcat(0, 0, 0, 0, 0, 0, noise_traj_opt[kc, 0], noise_traj_opt[kc, 1],
                                noise_traj_opt[kc, 2])
            gamma_q = tunable_para[0, 16]
            Q1, Q2, Q3   = tunable_para[0, 17]*gamma_q**(self.horizon-2-kc), tunable_para[0, 18]*gamma_q**(self.horizon-2-kc), tunable_para[0, 19]*gamma_q**(self.horizon-2-kc)
            Q_bar = np.diag(np.array([1, 1, 1, 1, 1, 1, Q1, Q2, Q3]))
            costate_traj_opt[kc:(kc+1), :] = np.transpose(mtimes(mtimes(LA.inv(np.transpose(B_bar)), Q_bar), w_opt_bar))

        # Output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "noise_traj_opt": noise_traj_opt,
                   "costate_traj_opt": costate_traj_opt}
        return opt_sol

    def lossMHE(self, Ground_truth, state_traj_opt):
        assert hasattr(self, 'state'), "Define the state variable first!"
        # Compute loss function
        df_Ix, df_Iy, df_Iz = SX.sym('df_Ix'), SX.sym('df_Iy'), SX.sym('df_Iz')
        dt_Bx, dt_By, dt_Bz = SX.sym('dt_Bx'), SX.sym('dt_By'), SX.sym('dt_Bz')
        wrench_true = vertcat(df_Ix, df_Iy, df_Iz, dt_Bx, dt_By, dt_Bz)
        wrench_mhe = vertcat(self.state[6], self.state[7], self.state[8], self.state[-3], self.state[-2],
                             self.state[-1])
        # Define the time-derivative of the loss function
        dloss = mtimes(transpose(wrench_true - wrench_mhe), wrench_true - wrench_mhe)
        dloss_fn = Function('dloss', [wrench_true, self.state], [dloss], ['gt', 'Xmhe'], ['dlossf'])
        loss = 0
        for k in range(self.horizon):
            loss += dloss_fn(gt=Ground_truth[len(Ground_truth)-self.horizon+k], Xmhe=state_traj_opt[k, :])['dlossf']
        return loss

    def diffKKT(self, P):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        self.P0 = P
        # Define co-state variables
        self.costate = SX.sym('lambda', self.n_state, 1)

        # Differentiate the dynamics to get the system Jacobian
        self.A    = jacobian(self.ModelDyn, self.state)
        self.A_fn = Function('A', [self.state], [self.A], ['s'], ['Af'])
        self.B    = jacobian(self.ModelDyn, self.noise) # B matrix is constant
        self.B_fn = Function('B', [self.state], [self.B], ['s'], ['Bf'])
        self.fu   = mtimes(mtimes(self.B, inv(self.Q)), mtimes(transpose(self.B), self.costate))
        self.D    = jacobian(self.fu, self.tunable_para)
        self.D_fn = Function('D', [self.costate, self.tunable_para, self.horizon1, self.index], [self.D], ['cs', 'tp', 'h1', 'ind'], ['Df'])
        self.H_fn = Function('H', [self.state], [self.H], ['s'], ['Hf'])

        # Second-order derivative of Lagrangian
        self.dJrun_x = mtimes(mtimes(self.H.T, self.R), self.measurement - mtimes(self.H, self.state))
        self.E    = jacobian(self.dJrun_x, self.tunable_para)
        self.E_fn = Function('E', [self.state, self.measurement, self.tunable_para, self.horizon1, self.index], [self.E], ['s', 'm', 'tp', 'h1', 'ind'], ['Ef'])
        self.dJ_x0= (mtimes(mtimes(mtimes(inv(self.P0), self.H.T), self.R), self.measurement - mtimes(self.H, self.state)) + \
                    mtimes(mtimes(inv(self.P0), transpose(self.A)), self.costate))
        self.F0   = jacobian(self.dJ_x0, self.tunable_para)
        self.F0_fn= Function('F0', [self.state, self.measurement, self.costate, self.tunable_para, self.horizon1, self.index],
                             [self.F0], ['s', 'm', 'cs', 'tp', 'h1', 'ind'], ['F0f'])

    def GetAuxSys(self, state_traj_opt, costate_traj_opt, model_QR, Y, track_e):
        # statement = [hasattr(self, 'A_fn'), hasattr(self, 'D_fn'), hasattr(self, 'E_fn'), hasattr(self, 'F0_fn')]
        horizon = np.size(state_traj_opt, 0)
        # if not all(statement):
        t_para = model_QR(track_e[-1])
        tunable_para = self.para(t_para)
        P = np.diag(tunable_para[0, 0:9])
        self.diffKKT(P)

        # Initialize the coefficient matrices of the auxiliary MHE system:
        matA, matD, matE = [], [], []
        matB, matH, matF = [], [], []
        # Solve the above coefficient matrices

        if horizon == 1:
            curr_s = state_traj_opt[0, :]
            curr_cs = costate_traj_opt[0, :]
            curr_m = Y[len(Y) - horizon]
            matF = self.F0_fn(s=curr_s, m=curr_m, cs=curr_cs, tp=tunable_para, h1=horizon-1, ind=horizon-1)['F0f'].full()
            matH = self.H_fn(s=curr_s)['Hf'].full()
        else:
            for t in range(horizon - 1):
                curr_s = state_traj_opt[t, :]
                curr_cs = costate_traj_opt[t, :]
                curr_m = Y[len(Y) - horizon + t]
                next_s = state_traj_opt[t + 1, :]
                next_m = Y[len(Y) - horizon + t + 1]
                next_e = track_e[len(track_e) - horizon + t + 1]
                if t == 0:
                    matF = self.F0_fn(s=curr_s, m=curr_m, cs=curr_cs, tp=tunable_para, h1=horizon-1, ind=t)['F0f'].full()
                matD += [self.D_fn(cs=curr_cs, tp=tunable_para, h1=horizon-1, ind=t)['Df'].full()]
                t_para = model_QR(next_e)
                tunable_para = self.para(t_para)
                matE += [self.E_fn(s=next_s, m=next_m, tp=tunable_para, h1=horizon-1, ind=t)['Ef'].full()]
                matA += [self.A_fn(s=curr_s)['Af'].full()]
                matB = self.B_fn(s=curr_s)['Bf'].full()
                matH = self.H_fn(s=curr_s)['Hf'].full()

        auxSys = {"matA": matA,
                  "matB": matB,
                  "matH": matH,
                  "matD": matD,
                  "matE": matE,
                  "matF": matF}
        return auxSys

    def Riccati(self, matA, matB, matH, tunable_para):
        assert hasattr(self, 'H'), "Define the output variable first!"
        assert hasattr(self, 'P0'), "Define the arrival cost first!"
        R1, R2, R3, R4, R5, R6 = tunable_para[0, 0], tunable_para[0, 1], tunable_para[0, 2], tunable_para[0, 3], tunable_para[0, 4], tunable_para[0, 5]
        # R7, R8, R9, R10, R11, R12 = tunable_para[0, 6], tunable_para[0, 7], tunable_para[0, 8], tunable_para[0, 9], tunable_para[0, 10], tunable_para[0, 11]
        # R13, R14, R15, R16, R17, R18 = tunable_para[0, 12], tunable_para[0, 13], tunable_para[0, 14], tunable_para[0, 15], tunable_para[0, 16], tunable_para[0, 17]
        R = np.diag(np.array([R1, R2, R3, R4, R5, R6]))
        Q1, Q2, Q3 = tunable_para[0, 6], tunable_para[0, 7], tunable_para[0, 8]
        # Q4, Q5, Q6 = tunable_para[0, 21], tunable_para[0, 22], tunable_para[0, 23]
        Q = np.diag(np.array([Q1, Q2, Q3]))
        P1_next  = np.matmul(np.matmul(matB, Q), np.transpose(matB))
        P2_next  = np.matmul(np.matmul(matA[1], self.P0), np.transpose(matA[1]))
        P3_next  = np.matmul(np.matmul(np.matmul(np.matmul(matA[1], self.P0), np.transpose(matH)), LA.inv(R + np.matmul(np.matmul(matH, self.P0), np.transpose(matH)))), np.matmul(np.matmul(matH, self.P0), np.transpose(matA[1])))
        P_a_next = P1_next + \
                   P2_next - \
                   P3_next
        return P_a_next

"""
The Auxiliary_MHE class solves for the explicit solutions of the gradients of optimal trajectories
w.r.t the tunable parameters 
"""
class Auxiliary_MHE:
    def __init__(self, dim_xmhe, dim_state, dt_sample, xdot, ctrl_gain_v, n_ctrl_gain, ref_v, ex_v, ev_v, state_v, ctrl_v, u_v, xmhe_v, mass, J_B, R_B_v):
        # ctrl_v is the symbolic variable of control force
        # u_v is the symbolic expression of control law
        # state_v = X defined in UavEnv
        self.n_xmhe = dim_xmhe
        self.n_state = dim_state
        self.DT    = dt_sample
        self.dyn   = xdot
        self.ctrl_gain_v = ctrl_gain_v
        self.n_ctrl_gain = n_ctrl_gain
        self.ref_v = ref_v
        self.ex_v = ex_v
        self.ev_v = ev_v
        self.state_v = state_v
        self.ctrl_v = ctrl_v
        self.u_v = u_v
        self.xmhe_v = xmhe_v
        self.pointer = 0
        self.mass = mass
        self.J_B = J_B
        self.R_B_v = R_B_v
        # self.loss = dot(traj_e, traj_e)
        self.loss = 1 / 2 * dot(self.ex_v, self.ex_v) + 1 / 2 * self.mass * dot(self.ev_v, self.ev_v)
        # Define the discrete dynamics using Euler method
        self.Dyn  = self.state_v + self.DT*self.dyn

    def SetDyn_Cost(self, matA, matB, matD, matE, matF, matH):
        self.matA, self.matB, self.matD, self.matE, self.matF, self.H = matA, matB, matD, matE, matF, matH
        self.horizon = len(matA)+1

    def SetPara(self, tunable_para, horizon1, index):
        self.P0 = np.diag(tunable_para[0, 0:9])
        gamma_r = tunable_para[0, 9]
        gamma_q = tunable_para[0, 16]
        self.R = np.diag(tunable_para[0, 10:16])*gamma_r**(horizon1-index)
        self.Q = np.diag(tunable_para[0, 17:20])*gamma_q**(horizon1-1-index)
        self.n_para = np.size(tunable_para)

    def para(self, para_bar):
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
            tunable[0, i] = para_bar[i, 0] # convert tensor to array
        tunable_para = np.zeros((1, 20))
        for i in range(9):
            tunable_para[0, i] = P_min + (P_max - P_min)*tunable[0, i]
        for i in range(6):
            tunable_para[0, i+10] = R_min + (R_max - R_min) * tunable[0, i+10]
        for i in range(3):
            tunable_para[0, i+17] = Q_min + (Q_max - Q_min) * tunable[0, i+17]
        tunable_para[0, 9] = gammar_min + (gammar_max - gammar_min)*tunable[0, 10]
        tunable_para[0, 16]= gammaq_min + (gammaq_max - gammaq_min)*tunable[0, 16]
        return  tunable_para

    def AuxMHESolver(self, matA, matB, matD, matE, matF, matH, model_QR, track_e):
        # statement = [hasattr(self, 'matA'), hasattr(self, 'matB'), hasattr(self, 'matD'), hasattr(self, 'matE'),
        #             hasattr(self, 'matF'), hasattr(self, 'P0'), hasattr(self, 'R'), hasattr(self, 'Q')]
        # if not all(statement):
        self.SetDyn_Cost(matA, matB, matD, matE, matF, matH)
        t_para = model_QR(track_e[-1])
        tunable_para = self.para(t_para)
        self.SetPara(tunable_para, self.horizon-1, 0)

        """-------------------------Forward Kalman filter-----------------------------"""
        # Initialize the state and covariance matrix
        X_kf = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        C_k = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        X_0 = self.matF-np.matmul(LA.inv(np.identity(self.n_xmhe)+np.matmul(np.matmul(LA.inv(self.P0), np.transpose(self.H)), np.matmul(self.R, self.H))),
                                np.matmul(np.matmul(LA.inv(self.P0), np.transpose(self.H)), np.matmul(np.matmul(self.R, self.H), self.matF)))
        C_0  = np.matmul(LA.inv(np.identity(self.n_xmhe)+np.matmul(np.matmul(LA.inv(self.P0), np.transpose(self.H)), np.matmul(self.R, self.H))), self.P0)
        X_kf[0] = X_0
        C_k[0]  = C_0
        for k in range(self.horizon-1):
            self.SetPara(tunable_para, self.horizon-1, k)
            X_kfp    = np.matmul(self.matA[k], X_kf[k]) + self.matD[k]
            P_next   = np.matmul(np.matmul(self.matA[k], C_k[k]), np.transpose(self.matA[k])) + np.matmul(np.matmul(self.matB, LA.inv(self.Q)), np.transpose(self.matB))
            C_k[k+1]  = np.matmul(LA.inv(np.identity(self.n_xmhe)+np.matmul(np.matmul(P_next, np.transpose(self.H)), mtimes(self.R, self.H))), P_next)
            X_kf[k+1] = X_kfp - np.matmul(np.matmul(np.matmul(C_k[k+1], np.transpose(self.H)), np.matmul(self.R, self.H)), X_kfp) + np.matmul(C_k[k+1], self.matE[k])

        """-------------------------Backward costate gradient--------------------------"""
        LAMBDA = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        Lambda_last = np.zeros((self.n_xmhe, self.n_para))
        LAMBDA[-1] = Lambda_last

        for i in range((self.horizon-1), 0, -1):
            if i == self.horizon-1:
                index = self.horizon-1 - i
                self.SetPara(tunable_para, self.horizon-1, index)
                Lambda_pre = -np.matmul(np.matmul(np.transpose(self.H), self.R), np.matmul(self.H, X_kf[i])) + self.matE[i-1]
            else:
                index = self.horizon - 1 - i
                self.SetPara(tunable_para, self.horizon - 1, index)
                Lambda_pre = np.matmul(np.identity(self.n_xmhe) - np.matmul(np.matmul(np.transpose(self.H), self.R), np.matmul(self.H, C_k[i])), np.matmul(np.transpose(self.matA[i]), LAMBDA[i])) - \
                             np.matmul(np.matmul(np.transpose(self.H), self.R), np.matmul(self.H, X_kf[i])) + self.matE[i - 1]
            LAMBDA[i-1] = Lambda_pre

        """-------------------------Forward state gradient-----------------------------"""
        X_opt = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        for j in range(self.horizon-1):
            A_Tj   = np.transpose(self.matA[j])
            X_j    = np.matmul(np.matmul(C_k[j], A_Tj), LAMBDA[j])
            X_optj = X_kf[j] + X_j
            X_opt[j] = X_optj
        X_opt[-1] = X_kf[-1]

        gra_opt = {"state_gra_traj": X_opt,
                   "costate_gra_traj": LAMBDA}
        return gra_opt

    def loss_tracking(self, ref, state):
        loss_fn = Function('loss', [self.state_v, self.ref_v], [self.loss], ['s', 'ref'], ['lossf'])
        loss_track = loss_fn(s=state, ref=ref)['lossf']
        return loss_track

    def ChainRule(self, Ctrl_Gain, Y, ref, xmhe_traj, X_opt, R_b):
        # Define the gradient of loss w.r.t state
        Ddlds = jacobian(self.loss, self.state_v)
        Ddlds_fn = Function('Ddlds', [self.state_v, self.ref_v], [Ddlds], ['s', 'ref'], ['dldsf'])
        # Define the gradient of dynamics w.r.t control
        Ddyndu = jacobian(self.Dyn, self.ctrl_v)
        Ddyndu_fn = Function('Ddyndu', [self.state_v, self.R_B_v], [Ddyndu], ['s', 'R_b'], ['Ddynduf'])
        # Define the gradient of control law w.r.t estimated state
        Dudx = jacobian(self.u_v, self.xmhe_v)
        Dudx_fn = Function('Dudx', [self.ref_v, self.xmhe_v, self.ctrl_gain_v], [Dudx], ['ref', 'x', 'gain'], ['Dudxf'])
        # Define the gradient of control law w.r.t control gain
        Dudg = jacobian(self.u_v, self.ctrl_gain_v)
        Dudg_fn = Function('Dudg', [self.ref_v, self.xmhe_v, self.ctrl_gain_v], [Dudg], ['ref', 'x', 'gain'], ['Dudgf'])
        # Initialize the parameter gradient
        dp = np.zeros((1, self.n_para))
        dp_ctrl = np.zeros((1, self.n_ctrl_gain))
        # Initialize the loss
        loss_track = 0
        for t in range(self.horizon-2):
            # Replace states in xmhe_traj with Y except estimated disturbance
            x_mhe = xmhe_traj[t, :]
            x_mhe = np.reshape(x_mhe, (self.n_xmhe, 1))
            x_mhe[0:6, 0] = Y[len(Y)-self.horizon+t][0:6, 0]
            # Compute the loss
            dloss_track = self.loss_tracking(ref[len(ref)-self.horizon+2+t], Y[len(Y)-self.horizon+1+t])
            loss_track += dloss_track
            # Compute the gradient of loss w.r.t control
            dlds = Ddlds_fn(s=Y[len(Y)-self.horizon+1+t], ref=ref[len(ref)-self.horizon+2+t])['dldsf'].full()
            dsdu = Ddyndu_fn(s=Y[len(Y)-self.horizon+t], R_b=R_b[len(R_b)-self.horizon+1+t])['Ddynduf'].full()
            dldu = np.matmul(dlds, dsdu)
            # Compute the gradient of loss w.r.t MHE weight
            dudx = Dudx_fn(ref=ref[len(ref)-self.horizon+1+t], x=x_mhe, gain=Ctrl_Gain[len(Ctrl_Gain)-self.horizon+1+t])['Dudxf'].full()
            dxdp = X_opt[t]
            dudp = np.matmul(dudx, dxdp)
            dp  += np.matmul(dldu, dudp)
            # Compute the gradient of loss w.r.t controller gain
            dudg = Dudg_fn(ref=ref[len(ref)-self.horizon+1+t], x=x_mhe, gain=Ctrl_Gain[len(Ctrl_Gain)-self.horizon+1+t])['Dudgf'].full()
            dp_ctrl += np.matmul(dldu, dudg)
        return dp, dp_ctrl, loss_track


































