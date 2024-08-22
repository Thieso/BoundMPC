import time
import copy
from importlib.resources import files
import casadi
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

from ..ReferencePath import ReferencePath
from ..RobotModel import RobotModel
from .mpc_utils_casadi import compute_bound_params
from .bound_mpc_functions import reference_function, error_function
from .casadi_ocp_formulation import setup_optimization_problem
from ..utils import integrate_rotation_reference, compute_initial_rot_errors
from ..utils.lie_functions import jac_SO3_inv_right, jac_SO3_inv_left
from .jerk_trajectory_casadi import calcAngle, calcVelocity, calcAcceleration


class BoundMPC:
    def __init__(self,
                 pos_points,
                 rot_points,
                 pos_lim,
                 rot_lim,
                 bp1,
                 br1,
                 s,
                 e_p_min,
                 e_r_min,
                 e_p_max,
                 e_r_max,
                 p0=np.zeros(6),
                 params=None):
        # Prediction horizon
        self.N = params.n

        self.robot_model = RobotModel()

        self.updated = False
        self.updated_once = False


        # Flag whether to compile the OCP problem
        self.build = params.build

        # Flag wether to create logging data
        self.log = not params.real_time

        # Initial position
        self.p0 = p0

        # Number of consecutive infeasible solutions
        self.error_count = 0

        # Time steps
        self.dt = params.dt

        # Time horizon
        self.T = self.dt * self.N

        # Create reference trajectory object
        self.nr_segs = params.nr_segs
        self.ref_path = ReferencePath(pos_points, rot_points, pos_lim, rot_lim,
                                      bp1, br1, s, e_p_min, e_r_min, e_p_max,
                                      e_r_max, self.nr_segs)

        # Initial error
        self.dtau_init = np.empty((3, self.nr_segs))
        self.dtau_init_par = np.empty((3, self.nr_segs))
        self.dtau_init_orth1 = np.empty((3, self.nr_segs))
        self.dtau_init_orth2 = np.empty((3, self.nr_segs))

        # Max path parameter
        self.phi_max = np.array([self.ref_path.phi_max - 0.0001])

        # Objective function weights
        self.weights = np.array(params.weights)
        # self.weights[4] /= self.phi_max[0]
        self.dphi_max = np.array([self.weights[4]])

        # Reference integration variables
        self.pr_ref = p0[3:]
        self.iw_ref = np.zeros(3)

        # Bounds
        limits = self.robot_model.get_robot_limits()
        self.q_lim_upper = limits[0]
        self.q_lim_lower = limits[1]
        self.dq_lim_upper = limits[2]
        self.dq_lim_lower = limits[3]
        self.tau_lim_upper = limits[4]
        self.tau_lim_lower = limits[5]
        self.u_max = limits[6]
        self.u_min = limits[7]
        self.ut_max = self.u_max
        self.ut_min = self.u_min

        # Path parameter init
        self.phi_current = np.array([0.0])
        self.phi_prev = np.array([0.0])
        self.dphi_current = np.array([0.0])
        self.ddphi_current = np.array([0.0])
        self.dddphi_current = np.array([0.0])

        # Dimension variables
        self.nr_joints = 7
        self.nr_u = self.nr_joints + 1
        self.nr_x = 44

        # Solution of previous run
        self.prev_solution = None
        self.prev_infeasible_solution = None
        self.lam_g0 = 0
        self.lam_x0 = 0
        self.lam = None
        self.pi = None

        # Setup the optimization problem
        path = files("bound_mpc")
        ipopt_options = {
            'tol': 10e-6,
            'max_iter': 500,
            'limited_memory_max_history': 6,
            'limited_memory_initialization': 'scalar1',
            'limited_memory_max_skipping': 2,
            # 'linear_solver': 'ma27',
            # 'linear_system_scaling': 'mc19',
            # 'ma57_automatic_scaling': 'no',
            # 'ma57_pre_alloc': 100,
            'mu_strategy': 'adaptive',
            'adaptive_mu_globalization': 'kkt-error',
            'print_info_string': 'no',
            'fast_step_computation': 'yes',
            'warm_start_init_point': 'yes',
            'mu_oracle': 'loqo',
            # 'max_wall_time': self.dt - 0.01,
            'fixed_mu_oracle': 'quality-function',
            'line_search_method': 'filter',
            'expect_infeasible_problem': 'no',
            'print_level': 0
        }

        self.solver_opts = {
            'verbose': False,
            'verbose_init': False,
            'print_time': False,
            'ipopt': ipopt_options
        }
        # Setup the optimization problem in casadi syntax
        self.solver, self.lbu, self.ubu, self.lbg, self.ubg, self.g_names = setup_optimization_problem(
            self.N, self.nr_joints, self.nr_segs, self.dt, self.u_min,
            self.u_max, self.ut_min, self.ut_max, self.q_lim_lower,
            self.q_lim_upper, self.dq_lim_lower, self.dq_lim_upper,
            self.solver_opts)
        if self.build:
            codegenopt = {'cpp': True}
            self.solver.generate_dependencies('gen_traj_opt_nlp_deps.cpp', codegenopt);
        else:
            solver_file = f'{path}/code_generation/mpc{self.N}_segs{self.nr_segs}.so'
            self.solver = casadi.nlpsol('solver', 'ipopt', solver_file,
                                        self.solver_opts)

    def update(self,
               pos_points,
               rot_points,
               pos_lim,
               rot_lim,
               bp1,
               br1,
               s,
               e_p_min,
               e_r_min,
               e_p_max,
               e_r_max,
               p,
               v,
               a,
               jerk,
               p0=np.zeros(6),
               params=None):
        self.updated = True
        self.updated_once = True
        self.p0 = p0

        # Create reference path object
        self.ref_path = ReferencePath(pos_points, rot_points, pos_lim, rot_lim,
                                      bp1, br1, s, e_p_min, e_r_min, e_p_max,
                                      e_r_max, self.nr_segs)

        # Max path parameter
        self.phi_max = np.array([self.ref_path.phi_max - 0.0001])

        # Objective function weights
        self.weights = np.array(params.weights)

        # Path parameter init
        # self.phi_current = np.array([0.0])
        dp0 = self.ref_path.dp[0]
        dp0 /= np.linalg.norm(dp0)
        self.phi_current = np.array([(p0[:3] - pos_points[0]).T @ dp0])
        # self.phi_current = np.array([np.max([0, self.phi_current])])
        # self.phi_current = np.array([0.0])
        self.phi_prev = self.phi_current
        dp_new = self.ref_path.dpd[:3, 0]
        v_proj = v[:3].T @ dp_new
        a_proj = a[:3].T @ dp_new
        j_proj = jerk[:3].T @ dp_new
        self.dphi_current = np.array([v_proj])
        self.ddphi_current = np.array([a_proj])
        self.dddphi_current = np.array([j_proj])

        # Reference integration variables
        self.pr_ref = integrate_rotation_reference(
            R.from_matrix(rot_points[0]).as_rotvec(), self.ref_path.dr[0],
            0.0, self.phi_current)
        self.iw_ref = self.ref_path.pd[
            3:, 0] + self.phi_current * self.ref_path.dpd[3:, 0]

    def compute_error_bounds(self, asymm_upper, asymm_lower, phi_switch, s,
                             e_p_min, e_r_min, e_p_max, e_r_max):
        asymm_bounds = np.concatenate(
            (asymm_upper[:2, :], -asymm_lower[:2, :], asymm_upper[2:, :],
             -asymm_lower[2:, :]))
        # TODO do not only use the first one
        peu = e_p_min[0] * np.ones((self.nr_segs, 4))
        reu = e_r_min[0] * np.ones((self.nr_segs, 4))
        pel = -e_p_min[0] * np.ones((self.nr_segs, 4))
        rel = -e_r_min[0] * np.ones((self.nr_segs, 4))
        e_p_max = np.array([e_p_max[0]])
        e_r_max = np.array([e_r_max[0]])
        s = np.array([s[0]])
        re_par0 = [e_r_min[0]]

        # Compute error bound parameters
        a4 = np.empty((self.nr_segs + 1, 9))
        a3 = np.empty((self.nr_segs + 1, 9))
        a2 = np.empty((self.nr_segs + 1, 9))
        a1 = np.empty((self.nr_segs + 1, 9))
        a0 = np.empty((self.nr_segs + 1, 9))
        for i in range(a4.shape[0]-1):
            phi0 = 0
            phi1 = phi_switch[i+1] - phi_switch[i]
            e0 = np.concatenate(
                (peu[i, :2], pel[i, :2], reu[i, :2], rel[i, :2], re_par0))
            e1 = np.concatenate(
                (peu[i, 2:], pel[i, 2:], reu[i, 2:], rel[i, 2:], re_par0))
            max_error = np.concatenate(
                (e_p_max, e_p_max, -e_p_max,
                 -e_p_max, e_r_max, e_r_max,
                 -e_r_max, -e_r_max, e_r_max))
            sv = np.concatenate((s, s, -s, -s, s, s, -s, -s, s))

            # Apply asymmetric boundings
            max_error[:-1] *= asymm_bounds[:, i]
            sv[:-1] *= asymm_bounds[:, i]

            # TODO better way to asymm bound the tangential path error
            max_error[-1] *= asymm_bounds[-1, i]
            sv[-1] *= asymm_bounds[-1, i]

            # Compute bounding function parameters
            a4[i, :], a3[i, :], a2[i, :], a1[i, :], a0[i, :] = compute_bound_params(
                    phi0, phi1, e0, e1, sv, max_error)

        return a0, a1, a2, a3, a4

    def compute_orientation_projection_vectors(self, br1, br2, dp_normed_ref):
        # Compute necessary values for the projection vectors
        dp_ref_proj = np.empty_like(dp_normed_ref)
        br1_proj = np.empty_like(br1)
        br2_proj = np.empty_like(br2)
        for i in range(dp_normed_ref.shape[1]):
            dtau_rest1 = R.from_rotvec(self.dtau_init[:, 0]).as_matrix() @ R.from_rotvec(self.dtau_init_orth1[:, i]).as_matrix().T
            dtau_rest2 = dtau_rest1 @ R.from_rotvec(self.dtau_init_par[:, i]).as_matrix().T
            jac_dtau_r = jac_SO3_inv_right(self.dtau_init[:, 0])
            jac_dtau_l = jac_SO3_inv_left(self.dtau_init[:, 0])
            jac_r1_r = jac_SO3_inv_right(R.from_matrix(dtau_rest1).as_rotvec())
            jac_r2_r = jac_SO3_inv_right(R.from_matrix(dtau_rest2).as_rotvec())

            dp_ref_proj[:, i] = jac_r1_r @ dp_normed_ref[:, i]
            br1_proj[:, i] = jac_dtau_r @ br1[:, i]
            br2_proj[:, i] = jac_r2_r @ br2[:, i]

        # Projection vectors for the orientation errors
        v_1 = np.empty_like(br1)
        v_2 = np.empty_like(br1)
        v_3 = np.empty_like(br1)
        for j in range(dp_normed_ref.shape[1]):
            v1 = br1_proj[:, j]
            v2 = dp_ref_proj[:, j]
            v3 = br2_proj[:, j]
            a = np.dot(v1, v1)
            b = np.dot(v1, v2)
            c = np.dot(v1, v3)
            d = np.dot(v3, v3)
            e = np.dot(v2, v2)
            f = np.dot(v2, v3)
            g = v1
            h = v2
            i = v3
            v_1[:, j] = (-b*d*h + b*f*i - c*e*i + c*f*h + d*e*g - f**2*g)/(a*d*e - a*f**2 - b**2*d + 2*b*c*f - c**2*e)
            v_2[:, j] = (a*d*h - a*f*i + b*c*i - b*d*g - c**2*h + c*f*g)/(a*d*e - a*f**2 - b**2*d + 2*b*c*f - c**2*e)
            v_3[:, j] = (a*e*i - a*f*h - b**2*i + b*c*h + b*f*g - c*e*g)/(a*d*e - a*f**2 - b**2*d + 2*b*c*f - c**2*e)
        return v_1, v_2, v_3, jac_dtau_l, jac_dtau_r

    def step(self, q0, dq0, ddq0, p0, v0, x_phi_d, jerk_current, x_des=None):
        """ One optimization step.
        """
        # Update the reference trajectory
        p_ref, dp_normed_ref, dp_ref, ddp_ref, phi_switch = self.ref_path.get_parameters(self.phi_current)
        asymm_lower, asymm_upper, bp1, bp2, br1, br2 = self.ref_path.get_limits()
        e_p_min, e_r_min, e_p_max, e_r_max, s = self.ref_path.get_bound_params()

        # Set the initial guess based on wheter a previous solution was
        # acquired.
        if self.prev_solution is None:
            w0 = np.zeros((self.N, self.nr_x))
            for i in range(self.N):
                w0[i, 8:15] = q0
                w0[i, 29:35] = p0
            w0 = w0.flatten().tolist()
        else:
            w0 = self.prev_solution.tolist()

            # Swap integrated omega if necessary
            w0 = np.array(w0).reshape((self.N, -1))
            i_omega = w0[:, 29:35]
            if np.linalg.norm(p0[3:] - i_omega[0, 3:]) > 1.5:
                print("[INFO] Reversing integrated omega")
                prev_p1 = i_omega[0, 3:]
                i_omega[:-1, 3:] = p0[3:] + (i_omega[1:, 3:] - prev_p1)
                i_omega[-1, 3:] = i_omega[-2, 3:]
            w0[:, 29:35] = i_omega

            if self.updated:
                # self.updated = False
                idx_cur = 0
                dp_new = dp_ref[:3, idx_cur]
                p_ref_new = p_ref[:3, idx_cur]
                for i in range(self.N):
                    pk = self.prev_traj[:3, i]
                    vk = self.prev_vel[:3, i]
                    ak = self.prev_acc[:3, i]
                    jk = self.prev_jerk[:3, i]
                    phik = phi_switch[idx_cur] + (pk[:3] - p_ref_new).T @ dp_new
                    dphik = vk[:3].T @ dp_new
                    ddphik = ak[:3].T @ dp_new
                    dddphik = jk[:3].T @ dp_new
                    # Make sure that the initial phi value never exceeds the
                    # segment
                    if phik > phi_switch[idx_cur + 1] - 0.01:
                        w0[i, 41] = phi_switch[idx_cur + 1] - 0.01
                        w0[i, 42] = 0.0  # Small bias
                        w0[i, 43] = 0.0
                        # w0[i, 7] = 0.0
                    elif phik < 0:
                        print("[WARNING] PHI TOO LOW")
                        w0[i, 8:15] = q0
                        w0[i, 41] = 0.0
                        w0[i, 42] = 0.0  # Small bias
                        w0[i, 43] = 0.0
                        # w0[i, 7] = 0.0
                        w0[i, 29:35] = p0
                        w0[i, 35:39] = 0.0
                    else:
                        w0[i, 41] = phik
                        w0[i, 42] = dphik
                        w0[i, 43] = ddphik
                        w0[i, 7] = dddphik

            w0 = w0.flatten().tolist()
            # Shift solution
            if not self.updated:
                s_N = int(len(w0)/self.N)
                w0[:(self.N-1)*s_N] = w0[s_N:]

        # Compute initial orientation errors at each via point for iterative
        # computation within the MPC
        for i in range(dp_ref.shape[1]):
            dtau_inits = compute_initial_rot_errors(p0[3:], self.pr_ref, dp_ref[3:, i],
                                                    br1[:, i], br2[:, i])
            self.dtau_init[:, i] = dtau_inits[0]
            self.dtau_init_par[:, i] = dtau_inits[1]
            self.dtau_init_orth1[:, i] = dtau_inits[2]
            self.dtau_init_orth2[:, i] = dtau_inits[3]

        # Compute orientation error projection vectors
        v_1, v_2, v_3, jac_dtau_l, jac_dtau_r = self.compute_orientation_projection_vectors(
            br1, br2, dp_normed_ref)

        # Compute polynomial parameters of error bounds
        a0, a1, a2, a3, a4 = self.compute_error_bounds(asymm_upper,
                                                       asymm_lower, phi_switch,
                                                       s, e_p_min, e_r_min,
                                                       e_p_max, e_r_max)

        # Limit desired path parameter to achieve limited speed
        x_phi_d_current = np.copy(x_phi_d)
        weights_current = np.copy(self.weights)
        if x_phi_d[0] < 1:
            scaling_factor = 1/(self.phi_max[0]**2)
            scaling_factor = np.min((scaling_factor, 2.0))
            weights_current[6] *= scaling_factor

        # For very long trajectories, this is necessary to avoid numerical
        # issues, performance does not change at all
        phi_max = np.array([np.min((self.phi_current + 5.0, self.phi_max))])
        x_phi_d_current[0] = np.array([np.min((self.phi_current[0] + 5.0, x_phi_d_current[0]))])

        # TODO Desired joint config
        qd = np.zeros(7)
        if phi_max - self.phi_current[0] < 0.05:
            qd = q0

        # Create parameter array
        params = np.concatenate(
                (self.iw_ref,
                 self.dtau_init[:, 0],
                 self.dtau_init_par.T.flatten(),
                 self.dtau_init_orth1.T.flatten(),
                 self.dtau_init_orth2.T.flatten(),
                 x_phi_d_current,
                 jerk_current,
                 self.dddphi_current,
                 phi_switch,
                 jac_dtau_r.T.flatten(),
                 jac_dtau_l.T.flatten(),
                 p_ref.flatten(),
                 dp_ref.flatten(),
                 dp_normed_ref.flatten(),
                 bp1.flatten(), bp2.flatten(), br1.flatten(), br2.flatten(),
                 a4.T.flatten(), a3.T.flatten(),
                 a2.T.flatten(), a1.T.flatten(),
                 a0.T.flatten(),
                 weights_current, phi_max,
                 self.dphi_max,
                 v_1.flatten(), v_2.flatten(),
                 v_3.flatten(),
                 qd))

        params = np.concatenate(
            (q0, dq0, ddq0, self.phi_current, self.dphi_current,
             self.ddphi_current, p0, v0, params))

        time_start = time.perf_counter()
        sol = self.solver(x0=w0,
                          lbx=self.lbu,
                          ubx=self.ubu,
                          lbg=self.lbg,
                          ubg=self.ubg,
                          # lam_g0=self.lam_g0,
                          # lam_x0=self.lam_x0,
                          p=params)
        w_curr = np.array(sol['x']).flatten()
        time_elapsed = time.perf_counter() - time_start
        stats = self.solver.stats()
        iters = stats['iter_count']
        # cost = float(sol['f']) / self.N

        # Check for constraint violations
        g = sol['g']
        g_viol = -np.sum(g[np.where(g < np.array(self.lbg) - 1e-6)[0]])
        g_viol += np.sum(g[np.where(g > np.array(self.ubg) + 1e-6)[0]])

        success = stats['success'] or g_viol < 1e-4

        using_previous = False
        if not success:
            self.error_count += 1
            print(
                f"[ERROR] Could not find feasible solution. Using previous solution. Error count: {self.error_count}"
            )
            print(f"Constraint Violation Sum: {g_viol}")
            print(f"Casadi status: {stats['return_status']}")
            if self.prev_solution is not None:
                self.prev_infeasible_solution = w_curr
                w_opt = np.copy(self.prev_solution)
                using_previous = True
            else:
                print(
                    "[WARNING] Previous solution not found, using infeasible solution."
                )
                self.error_count = 0
                w_opt = w_curr
                using_previous = True
                # self.prev_solution = w_opt
                self.lam_g0 = sol['lam_g']
                self.lam_x0 = sol['lam_x']
                self.prev_infeasible_solution = self.prev_solution
        else:
            self.error_count = 0
            w_opt = w_curr
            self.prev_solution = copy.deepcopy(w_opt)
            self.lam_g0 = sol['lam_g']
            self.lam_x0 = sol['lam_x']
            self.prev_infeasible_solution = w_opt

        if self.error_count < self.N:
            traj_data, ref_data, err_data = self.compute_return_data(
                q0, dq0, ddq0, jerk_current, p0, w_opt, using_previous, a4, a3,
                a2, a1, a0, jac_dtau_l, jac_dtau_r, x_des, p_ref,
                dp_normed_ref, dp_ref, phi_switch, asymm_lower, asymm_upper,
                bp1, bp2, br1, br2, v_1, v_2, v_3, x_phi_d_current, phi_max)
            return traj_data, ref_data, err_data, time_elapsed, iters
        else:
            return None, None, None, None, None

    def compute_return_data(self, q0, dq0, ddq0, jerk_current, p0, w_opt,
                            using_previous, a4, a3, a2, a1, a0, jac_dtau_l,
                            jac_dtau_r, x_des, p_ref, dp_normed_ref, dp_ref,
                            phi_switch, asymm_lower, asymm_upper, bp1, bp2,
                            br1, br2, v1, v2, v3, x_phi_d, phi_max):
        w_opt = np.array(w_opt.reshape((self.N, -1))).T
        optimal_jerk = w_opt[:self.nr_joints, self.error_count:]
        optimal_jerk_phi = w_opt[self.nr_joints, self.error_count:]

        # Convert trajectory to lie space by using the jacobian
        optimal_q = w_opt[8:15, self.error_count:]
        optimal_dq = w_opt[15:22, self.error_count:]
        optimal_ddq = w_opt[22:29, self.error_count:]
        optimal_traj = w_opt[29:35, self.error_count:]
        optimal_phi = w_opt[41, self.error_count:]
        optimal_dphi = w_opt[42, self.error_count:]
        optimal_ddphi = w_opt[43, self.error_count:]

        # Integrate the system. This is necessary because the solver
        # might not find the optimal solution which makes the dynamic system
        # constraints go to zero.
        jerk_mat = np.concatenate(
            (np.expand_dims(jerk_current, axis=1), optimal_jerk), axis=1)
        jerk_mat_phi = np.concatenate(
            (np.expand_dims(self.dddphi_current, axis=1),
             np.expand_dims(optimal_jerk_phi, axis=1).T),
            axis=1)
        self.phi_prev = np.copy(self.phi_current)
        for i in range(optimal_jerk.shape[1]):
            t = self.dt * (i + 1)
            optimal_phi[i] = calcAngle(jerk_mat_phi,
                                       t,
                                       self.phi_current,
                                       self.dphi_current,
                                       self.ddphi_current,
                                       self.dt)
            optimal_dphi[i] = calcVelocity(jerk_mat_phi,
                                           t,
                                           self.dphi_current,
                                           self.ddphi_current,
                                           self.dt)
            optimal_ddphi[i] = calcAcceleration(jerk_mat_phi,
                                                t,
                                                self.ddphi_current,
                                                self.dt)
            optimal_q[:, i] = calcAngle(jerk_mat, t, q0, dq0, ddq0, self.dt)
            optimal_dq[:, i] = calcVelocity(jerk_mat, t, dq0, ddq0, self.dt)
            optimal_ddq[:, i] = calcAcceleration(jerk_mat, t, ddq0, self.dt)

        # Save previous solution
        self.prev_traj = w_opt[29:35, :]
        self.prev_vel = w_opt[35:41, :]
        self.prev_acc = np.empty_like(self.prev_vel)
        self.prev_jerk = np.empty_like(self.prev_vel)
        for i in range(self.N):
            p_c, jac_fk, djac_fk = self.robot_model.forward_kinematics(w_opt[8:15, i], w_opt[15:22, i])
            ddjac_fk = self.robot_model.ddjacobian_fk(w_opt[8:15, i], w_opt[15:22, i], w_opt[22:29, i])
            self.prev_acc[:, i] = jac_fk @ w_opt[22:29, i] + djac_fk @ w_opt[15:22, i]
            self.prev_jerk[:, i] = jac_fk @ w_opt[:self.nr_joints, i] + djac_fk @ w_opt[22:29, i] + ddjac_fk @ w_opt[15:22, i]

        # Compute cartesian trajectories
        optimal_i_omega = np.empty((6, optimal_q.shape[1]))
        optimal_acc = np.empty((6, optimal_q.shape[1]))
        optimal_vel = np.empty((6, optimal_q.shape[1]))
        jac_fk = self.robot_model.jacobian_fk(q0)
        omega_prev = (jac_fk @ dq0)[3:]
        for i in range(optimal_q.shape[1]):
            p_c, jac_fk, djac_fk = self.robot_model.forward_kinematics(
                optimal_q[:, i], optimal_dq[:, i])
            optimal_traj[:, i] = np.copy(p_c)
            optimal_vel[:, i] = jac_fk @ optimal_dq[:, i]
            optimal_acc[:, i] = jac_fk @ optimal_ddq[:, i] + djac_fk @ optimal_dq[:, i]
            k1 = omega_prev
            k2 = optimal_vel[3:, i]
            omega_prev = np.copy(k2)
            if i > 0:
                optimal_i_omega[3:, i] = optimal_i_omega[3:, i-1] + 1/2 * self.dt * (k1 + k2)
            else:
                optimal_i_omega[3:, i] = p0[3:] + 1/2 * self.dt * (k1 + k2)
        optimal_i_omega[:3, :] = optimal_traj[:3, :]
        if self.error_count > 0:
            if np.linalg.norm(p0[3:] - optimal_i_omega[3:, 0]) > 3.1:
                optimal_i_omega[3:, :] *= -1

        # Integrate the rotation reference
        iw_ref_copy = np.copy(self.iw_ref)
        if optimal_phi[0] > phi_switch[1]:
            self.pr_ref = R.from_matrix(self.ref_path.r[self.ref_path.sector+1]).as_rotvec()
            self.pr_ref = integrate_rotation_reference(
                self.pr_ref, dp_ref[3:, 1], phi_switch[1],
                optimal_phi[0])
            self.iw_ref = p_ref[3:, 1] + (optimal_phi[0] - phi_switch[1]) * dp_ref[3:, 1]
        else:
            self.pr_ref = integrate_rotation_reference(
                self.pr_ref, dp_ref[3:, 0],
                self.phi_current, optimal_phi[0])
            self.iw_ref = p_ref[3:, 0] + (optimal_phi[0] - phi_switch[0]) * dp_ref[3:, 0]

        # Set new path parameter state
        phi_prev = np.copy(self.phi_current)
        self.phi_current = np.array([optimal_phi[0]])
        self.dphi_current = np.array([optimal_dphi[0]])
        self.ddphi_current = np.array([optimal_ddphi[0]])
        self.dddphi_current = np.array([optimal_jerk_phi[0]])

        # Create reference trajectory
        if self.log:
            ref_data = defaultdict(list)
            err_data = defaultdict(list)
            i_omega_0 = p0[3:]
            for i in range(optimal_phi.shape[0]):
                phi = optimal_phi[i]
                dphi = optimal_dphi[i]
                reference = reference_function(dp_ref=dp_ref.T,
                                               p_ref=p_ref.T,
                                               dp_normed_ref=dp_normed_ref.T,
                                               phi_switch=np.expand_dims(phi_switch, 1),
                                               phi=phi,
                                               phi_prev=phi_prev,
                                               bp1=bp1.T,
                                               bp2=bp2.T,
                                               br1=br1.T,
                                               br2=br2.T,
                                               v1=v1.T,
                                               v2=v2.T,
                                               v3=v3.T,
                                               a4=a4,
                                               a3=a3,
                                               a2=a2,
                                               a1=a1,
                                               a0=a0,
                                               nr_segs=self.nr_segs)
                phi_prev = phi
                p_d = np.array(reference['p_d']).flatten()
                dp_d = np.array(reference['dp_d']).flatten()
                ddp_d = np.array(reference['ddp_d']).flatten()
                dp_normed_d = np.array(reference['dp_normed_d']).flatten()

                ref_data['p'].append(p_d)
                ref_data['dp'].append(dp_d)
                ref_data['ddp'].append(ddp_d)
                ref_data['dp_normed'].append(dp_normed_d)
                ref_data['r_par_bound'].append(
                    np.array(reference['r_par_bound']).flatten())
                ref_data['bound_lower'].append(
                    np.array(reference['bound_lower']).flatten())
                ref_data['bound_upper'].append(
                    np.array(reference['bound_upper']).flatten())
                ref_data['e_p_off'].append(
                    np.array(reference['e_p_off']).flatten())
                ref_data['e_r_off'].append(
                    np.array(reference['e_r_off']).flatten())
                ref_data['bp1'].append(np.array(reference['bp1_current']).squeeze())
                ref_data['bp2'].append(np.array(reference['bp2_current']).squeeze())
                ref_data['br1'].append(reference['br1_current'])
                ref_data['br2'].append(reference['br2_current'])
                ref_data['v1'].append(reference['v1_current'])
                ref_data['v2'].append(reference['v2_current'])
                ref_data['v3'].append(reference['v3_current'])

                # Compute errors
                errors = error_function(
                    p=optimal_i_omega[:, i],
                    v=optimal_vel[:, i],
                    p_ref=ref_data['p'][i],
                    dp_ref=ref_data['dp'][i],
                    dp_normed_ref=ref_data['dp_normed'][i],
                    dphi=dphi,
                    i_omega_0=i_omega_0,
                    i_omega_ref_0=iw_ref_copy,
                    dtau_init=self.dtau_init[:, 0],
                    dtau_init_par=self.dtau_init_par.T,
                    dtau_init_orth1=self.dtau_init_orth1.T,
                    dtau_init_orth2=self.dtau_init_orth2.T,
                    br1=ref_data['br1'][i].T,
                    br2=ref_data['br2'][i].T,
                    jac_dtau_l=jac_dtau_l,
                    jac_dtau_r=jac_dtau_r,
                    phi=phi,
                    phi_switch=phi_switch,
                    v1=ref_data['v1'][i].T,
                    v2=ref_data['v2'][i].T,
                    v3=ref_data['v3'][i].T,
                    nr_segs=self.nr_segs)
                e_p_par = np.array(errors['e_p_par']).flatten()
                e_p_orth = np.array(errors['e_p_orth']).flatten()
                de_p_par = np.array(errors['de_p_par']).flatten()
                de_p_orth = np.array(errors['de_p_orth']).flatten()
                e_p = np.array(errors['e_p']).flatten()
                de_p = np.array(errors['de_p']).flatten()
                e_r_par = np.array(errors['e_r_par']).flatten()
                e_r_orth1 = np.array(errors['e_r_orth1']).flatten()
                e_r_orth2 = np.array(errors['e_r_orth2']).flatten()
                e_r = np.array(errors['e_r']).flatten()
                de_r = np.array(errors['de_r']).flatten()

                err_data['e_p'].append(e_p)
                err_data['de_p'].append(de_p)
                err_data['e_p_par'].append(e_p_par)
                err_data['e_p_orth'].append(e_p_orth)
                err_data['de_p_par'].append(de_p_par)
                err_data['de_p_orth'].append(de_p_orth)
                err_data['e_r'].append(np.copy(e_r))
                err_data['de_r'].append(de_r)
                err_data['e_r_par'].append(e_r_par)
                err_data['e_r_orth1'].append(e_r_orth1)
                err_data['e_r_orth2'].append(e_r_orth2)

            # Update ref data to correct rotation reference
            ref_data['p'][0][3:] = np.copy(self.pr_ref)
            pr_ref = np.copy(self.pr_ref)
            for i in range(self.N - self.error_count - 1):
                ref_data['p'][i][3:] = np.copy(pr_ref)
                tauc = R.from_rotvec(optimal_traj[3:, i]).as_matrix()
                taud = R.from_rotvec(pr_ref).as_matrix()
                dtau_correct = R.from_matrix(tauc @ taud.T).as_rotvec()
                err_data['e_r'][i] = dtau_correct

                phi = optimal_phi[i]
                dphi = optimal_dphi[i]
                phi_next = optimal_phi[i+1]
                if phi_next > phi_switch[1] and phi < phi_switch[1]:
                    pr_ref = R.from_matrix(self.ref_path.r[self.ref_path.sector+1]).as_rotvec()
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 1], phi_switch[1],
                        phi_next)
                elif phi_next > phi_switch[2] and phi < phi_switch[2]:
                    pr_ref = R.from_matrix(self.ref_path.r[self.ref_path.sector+2]).as_rotvec()
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 2], phi_switch[2],
                        phi_next)
                elif phi_next > phi_switch[2]:
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 2], phi, phi_next)
                elif phi_next > phi_switch[1]:
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 1], phi, phi_next)
                else:
                    pr_ref = integrate_rotation_reference(
                        pr_ref, dp_ref[3:, 0], phi, phi_next)
            ref_data['p'][-1][3:] = np.copy(pr_ref)
            tauc = R.from_rotvec(optimal_traj[3:, -1]).as_matrix()
            taud = R.from_rotvec(pr_ref).as_matrix()
            dtau_correct = R.from_matrix(tauc @ taud.T).as_rotvec()
            err_data['e_r'][-1] = dtau_correct
        else:
            ref_data = None
            err_data = None

        traj_data = {}
        traj_data['p'] = optimal_traj
        traj_data['v'] = optimal_vel
        traj_data['a'] = optimal_acc
        traj_data['q'] = optimal_q
        traj_data['dq'] = optimal_dq
        traj_data['ddq'] = optimal_ddq
        traj_data['dddq'] = optimal_jerk
        traj_data['phi'] = optimal_phi
        traj_data['dphi'] = optimal_dphi
        traj_data['ddphi'] = optimal_ddphi
        traj_data['dddphi'] = optimal_jerk_phi

        return traj_data, ref_data, err_data
