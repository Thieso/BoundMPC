import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from bound_mpc.RobotModel import RobotModel
from bound_mpc.BoundMPC.mpc_utils_casadi import decompose_orthogonal_error
from bound_mpc.utils import project_position_bounds


class Plotter:
    """ Class for plotting the MPC data.
    """
    def __init__(self,
                 logger,
                 params,
                 t0,
                 p_via,
                 r_via,
                 path="",
                 tail="",
                 save_data=False,
                 x_values = 'time'):
        self.robot_model = RobotModel()
        # Create arrays for saving the values
        p_traj = np.array(logger.p_traj)
        v_traj = np.array(logger.v_traj)
        a_traj = np.array(logger.a_traj)
        q_traj = np.array(logger.q_traj)
        dq_traj = np.array(logger.dq_traj)
        ddq_traj = np.array(logger.ddq_traj)
        j_traj = np.array(logger.j_traj)
        j_cart_traj = np.empty((j_traj.shape[0], 6))
        for i in range(j_traj.shape[0]):
            jac = self.robot_model.jacobian_fk(q_traj[i, :])
            djac = self.robot_model.djacobian_fk(q_traj[i, :], dq_traj[i, :])
            ddjac = self.robot_model.ddjacobian_fk(q_traj[i, :], dq_traj[i, :], ddq_traj[i, :],)
            j_cart_traj[i, :] = jac @ j_traj[i, :] + 2 * djac @ ddq_traj[i, :] + ddjac @ dq_traj[i, :]
        phi_traj = np.array(logger.phi_traj)
        phi_horizon = np.array(logger.phi_horizon)
        dphi_traj = np.array(logger.dphi_traj)
        ddphi_traj = np.array(logger.ddphi_traj)
        dddphi_traj = np.array(logger.dddphi_traj)

        p_ref = np.array(logger.p_ref)
        dp_ref = np.array(logger.dp_ref)
        dp_ref_horizon = np.array(logger.dp_ref_horizon)
        p_l = np.array(logger.p_l)
        p_u = np.array(logger.p_u)
        r_par_bound = np.array(logger.r_par_bound)
        r_par_bound_horizon = np.array(logger.r_par_bound_horizon)
        p_l_horizon = np.array(logger.p_l_horizon)
        p_u_horizon = np.array(logger.p_u_horizon)
        e_p_off = np.array(logger.e_p_off)
        e_r_off = np.array(logger.e_r_off)
        bp1 = np.array(logger.bp1)
        bp2 = np.array(logger.bp2)
        br1 = np.array(logger.br1)
        br2 = np.array(logger.br2)
        bp1_horizon = np.array(logger.bp1_horizon)
        bp2_horizon = np.array(logger.bp2_horizon)
        br1_horizon = np.array(logger.br1_horizon)
        br2_horizon = np.array(logger.br2_horizon)

        e_par = np.array(logger.e_par)
        de_par = np.array(logger.de_par)
        de_r = np.array(logger.de_r)
        e_orth = np.array(logger.e_orth)
        e_orth_horizon = np.array(logger.e_orth_horizon)
        e_r_orth1_horizon = np.array(logger.e_r_orth1_horizon)
        e_r_orth2_horizon = np.array(logger.e_r_orth2_horizon)
        e_r = np.array(logger.e_r)
        e_r_horizon = np.array(logger.e_r_horizon)
        e_r_par = np.array(logger.e_r_par)
        e_r_par_horizon = np.array(logger.e_r_par_horizon)
        e_r_orth1 = np.array(logger.e_r_orth1)
        e_r_orth2 = np.array(logger.e_r_orth2)
        de_orth = np.array(logger.de_orth)
        t_traj = np.array(logger.t_traj)
        iterations = np.array(logger.iterations)
        t_comp = np.array(logger.t_comp)
        t_loops = np.array(logger.t_loops)
        t_overhead = np.array(logger.t_overhead)
        fails = np.array(logger.fails)
        t_switch = np.array(logger.t_switch)
        phi_switch = np.array(logger.phi_switch)

        limits = self.robot_model.get_robot_limits()
        q_lim_upper = limits[0]
        q_lim_lower = limits[1]
        dq_lim_upper = limits[2]
        dq_lim_lower = limits[3]
        tau_lim_upper = limits[4]
        tau_lim_lower = limits[5]
        u_max = limits[6]
        u_min = limits[7]

        # Compute true rotation errors
        e_r_par_true = np.empty_like(e_r_par)
        e_r_orths_true = np.empty((e_r_par.shape[0], 2))
        e_r_par_true_length = np.empty(e_r_par.shape[0])
        e_r_plane_horizon_true = np.empty((br1_horizon.shape[0], br1_horizon.shape[1], 3))
        for i in range(e_r_par.shape[0]):
            norm = np.linalg.norm(dp_ref[i, 3:])
            if norm > 1e-4:
                dp_normed = dp_ref[i, 3:] / norm
            else:
                dp_normed = np.array([0, 1.0, 0])
            r01 = np.zeros((3, 3))
            r01[:, 0] = br2[i, :]
            r01[:, 1] = dp_normed
            r01[:, 2] = br1[i, :]
            dtau_01 = r01.T @ R.from_rotvec(e_r[i, :]).as_matrix() @ r01
            eul = R.from_matrix(dtau_01).as_euler('zyx')

            e_r_orths_true[i, 1] = eul[2]
            e_r_orths_true[i, 0] = eul[0]
            e_r_par_true[i, :] = eul[1] * dp_normed
            e_r_par_true_length[i] = eul[1]

            for j in range(params.n):
                norm = np.linalg.norm(dp_ref_horizon[j, i, 3:])
                if norm > 1e-4:
                    dp_normed = dp_ref_horizon[j, i, 3:] / norm
                else:
                    dp_normed = np.array([0, 1.0, 0])
                r01 = np.zeros((3, 3))
                r01[:, 0] = br2_horizon[j, i, :]
                r01[:, 1] = dp_normed
                r01[:, 2] = br1_horizon[j, i, :]
                dtau_01 = r01.T @ R.from_rotvec(e_r_horizon[j, i, :]).as_matrix() @ r01
                eul = R.from_matrix(dtau_01).as_euler('zyx')

                e_r_plane_horizon_true[j, i, 0] = eul[0]
                e_r_plane_horizon_true[j, i, 1] = eul[2]
                e_r_plane_horizon_true[j, i, 2] = eul[1]
        e_r_orths_true *= 180 / np.pi

        # Compute errors
        e_par_norm = np.linalg.norm(e_par, axis=1)
        e_orth_norm = np.linalg.norm(e_orth, axis=1)
        de_par_norm = np.linalg.norm(de_par, axis=1)
        de_orth_norm = np.linalg.norm(de_orth, axis=1)
        de_orth_norm_num = np.linalg.norm(np.diff(e_orth, axis=0) / params.dt, axis=1)
        de_par_norm_num = np.linalg.norm(np.diff(e_par, axis=0) / params.dt, axis=1)
        e_r_par_norm = np.linalg.norm(e_r_par, axis=1) * 180 / np.pi
        de_r_norm_num = np.linalg.norm(np.diff(e_r, axis=0) / params.dt, axis=1)
        e_r_orth1_norm = np.linalg.norm(e_r_orth1, axis=1) * 180 / np.pi
        e_r_orth2_norm = np.linalg.norm(e_r_orth2, axis=1) * 180 / np.pi
        e_orth_dist = np.zeros((e_orth_norm.shape[0], 2))
        e_r_orth_dist = np.zeros((e_orth_dist.shape[0], 2))
        p_bound = np.zeros((e_orth_dist.shape[0], 2))
        r_bound = np.zeros((e_orth_dist.shape[0], 2))
        e_orth_plane = np.empty_like(e_p_off)
        e_r_orth_plane = np.empty_like(e_p_off)
        e_plane_horizon = np.empty((bp1_horizon.shape[0], bp1_horizon.shape[1], 2))
        e_r_plane_horizon = np.empty((br1_horizon.shape[0], br1_horizon.shape[1], 3))
        e_par_length = np.empty_like(e_par_norm)
        e_r_par_length = np.empty_like(e_r_par_norm)
        for i in range(e_orth_dist.shape[0]):
            dp_normed = dp_ref[i, :3] / np.linalg.norm(dp_ref[i, :3])
            b1 = bp1[i, :]
            b2 = bp2[i, :]
            e_orth_plane[i, :] = np.array(
                decompose_orthogonal_error(e_orth[i, :], b1, b2)).flatten()
            p_bound[i, :] = (p_u[i, :2] - p_l[i, :2]) / 2
            e_p_diff = e_orth_plane[i, :] - e_p_off[i, :]
            e_orth_dist[i, :] = (e_p_diff)**2
            e_par_length[i] = np.dot(dp_normed, e_par[i, :])

            norm = np.linalg.norm(dp_ref[i, 3:])
            if norm > 1e-4:
                dpr_normed = dp_ref[i, 3:] / norm
            else:
                dpr_normed = np.array([0, 1.0, 0])
            b1 = br1[i, :]
            b2 = br2[i, :]
            r_bound[i, :] = (p_u[i, 2:] - p_l[i, 2:]) / 2
            e_r_orth_plane[i, 0] = np.dot(b1, e_r_orth1[i, :])
            e_r_orth_plane[i, 1] = np.dot(b2, e_r_orth2[i, :])
            e_r_par_length[i] = np.dot(dpr_normed, e_r_par[i, :])

            for j in range(params.n):
                e_plane_horizon[j, i, :] = np.array(decompose_orthogonal_error(
                                                    e_orth_horizon[j, i, :],
                                                    bp1_horizon[j, i, :],
                                                    bp2_horizon[j, i, :])).flatten()
                e_r_plane_horizon[j, i, 0] = np.dot(br1_horizon[j, i, :],
                                                    e_r_orth1_horizon[j, i, :])
                e_r_plane_horizon[j, i, 1] = np.dot(br2_horizon[j, i, :],
                                                    e_r_orth2_horizon[j, i, :])
                norm = np.linalg.norm(dp_ref_horizon[j, i, 3:])
                if norm > 1e-4:
                    dpr_normed = dp_ref_horizon[j, i, 3:] / norm
                else:
                    dpr_normed = np.array([0, 1.0, 0])
                e_r_plane_horizon[j, i, 2] = np.dot(dpr_normed, e_r_par_horizon[j, i, :])
            e_r_orth_dist[i, 0] = e_r_orth1_norm[i]
            e_r_orth_dist[i, 1] = e_r_orth2_norm[i]

        # Compute trajectory in error plane
        p_ref_plane = np.empty((p_bound.shape[0], 4))
        p_plane = np.empty((p_bound.shape[0], 4))
        for i in range(p_ref_plane.shape[0]):
            b1 = bp1[i, :]
            b2 = bp2[i, :]
            p_ref_plane[i, :2] = np.array(
                decompose_orthogonal_error(p_ref[i, 3:], b1, b2)).flatten()
            p_plane[i, :2] = np.array(
                decompose_orthogonal_error(p_traj[i, 3:], b1, b2)).flatten()
            b1 = br1[i, :]
            b2 = br2[i, :]
            p_ref_plane[i, 2:] = np.array(
                decompose_orthogonal_error(p_ref[i, 3:], b1, b2)).flatten()
            p_plane[i, 2:] = np.array(
                decompose_orthogonal_error(p_traj[i, 3:], b1, b2)).flatten()

        # Compute correct orientation error
        e_r_correct = np.empty_like(e_r)
        for i in range(p_traj.shape[0]):
            r1 = R.from_rotvec(p_traj[i, 3:]).as_matrix()
            r2 = R.from_rotvec(p_ref[i, 3:]).as_matrix()
            dr = R.from_matrix(r1 @ r2.T)
            e_r_correct[i, :] = dr.as_rotvec()

        # Compute singularity data
        ellipsoid_volume = np.empty(q_traj.shape[0])
        condition_number = np.empty(q_traj.shape[0])
        manip_measure = np.empty(q_traj.shape[0])
        for i in range(q_traj.shape[0]):
            jac = self.robot_model.jacobian_fk(q_traj[i, :])
            a_manip = jac @ jac.T
            eig_a = np.linalg.eig(a_manip)[0]
            manip_measure[i] = self.robot_model.manipulability_measure(q_traj[i, :])
            ellipsoid_volume[i] = np.sqrt(np.linalg.det(a_manip))
            condition_number[i] = np.max(eig_a) / np.min(eig_a)

        # Set the x value to plot over time or path
        x_traj = phi_traj if x_values == 'path' else t_traj
        x_label = r"Path [$\theta$]" if x_values == 'path' else "Time [s]"

        # Orientation approximation errors
        approx_error = e_r_plane_horizon_true - e_r_plane_horizon
        approx_error_n_mean = np.mean(np.abs(approx_error), axis=1) * 180/np.pi
        approx_error_n_max = np.max(np.abs(approx_error), axis=1) * 180/np.pi
        approx_error_n_min = np.min(np.abs(approx_error), axis=1) * 180/np.pi
        approx_error_x_mean = np.mean(np.abs(approx_error), axis=0) * 180/np.pi
        approx_error_x_max = np.max(np.abs(approx_error), axis=0) * 180/np.pi
        approx_error_x_min = np.min(np.abs(approx_error), axis=0) * 180/np.pi

        plt.figure()
        plt.subplot(311)
        plt.plot(x_traj, bp1[:, 0])
        plt.plot(x_traj, bp1[:, 1])
        plt.plot(x_traj, bp1[:, 2])
        for j in range(e_r_plane_horizon.shape[1]):
            t_horizon = (j+1)/10 + np.arange(0, logger.N/10, 0.1)
            x_horizon = phi_horizon[:, j] if x_values == 'path' else t_horizon
            plt.plot(x_horizon,
                     bp1_horizon[:, j, 0],
                     'C0', linewidth=0.5, label="_Hidden")
            plt.plot(x_horizon,
                     bp1_horizon[:, j, 1],
                     'C1', linewidth=0.5, label="_Hidden")
            plt.plot(x_horizon,
                     bp1_horizon[:, j, 2],
                     'C2', linewidth=0.5, label="_Hidden")
        plt.title("B1")
        plt.subplot(312)
        plt.plot(x_traj, bp2[:, 0])
        plt.plot(x_traj, bp2[:, 1])
        plt.plot(x_traj, bp2[:, 2])
        for j in range(e_r_plane_horizon.shape[1]):
            t_horizon = (j+1)/10 + np.arange(0, logger.N/10, 0.1)
            x_horizon = phi_horizon[:, j] if x_values == 'path' else t_horizon
            plt.plot(x_horizon,
                     bp2_horizon[:, j, 0],
                     'C0', linewidth=0.5, label="_Hidden")
            plt.plot(x_horizon,
                     bp2_horizon[:, j, 1],
                     'C1', linewidth=0.5, label="_Hidden")
            plt.plot(x_horizon,
                     bp2_horizon[:, j, 2],
                     'C2', linewidth=0.5, label="_Hidden")
        plt.title("B2")
        plt.subplot(313)
        plt.plot(x_traj, np.linalg.norm(bp1, axis=1), label="B1")
        plt.plot(x_traj, np.linalg.norm(bp2, axis=1), label="B2")
        plt.plot(x_traj, np.diag(np.dot(bp1, dp_ref[:, :3].T)), label="B1 path")
        plt.plot(x_traj, np.diag(np.dot(bp2, dp_ref[:, :3].T)), label="B2 path")
        for j in range(e_r_plane_horizon.shape[1]):
            t_horizon = (j+1)/10 + np.arange(0, logger.N/10, 0.1)
            x_horizon = phi_horizon[:, j] if x_values == 'path' else t_horizon
            plt.plot(x_horizon,
                     np.linalg.norm(bp1_horizon[:, j, :], axis=1),
                     'C0', linewidth=0.5, label="_Hidden")
            plt.plot(x_horizon,
                     np.linalg.norm(bp2_horizon[:, j, :], axis=1),
                     'C1', linewidth=0.5, label="_Hidden")
        plt.title("Norms")
        plt.suptitle("Basis vectors position")

        # -----------------------------------------------------------------------
        # Plot approximation errors
        # -----------------------------------------------------------------------
        plt.figure()
        for i in range(3):
            if i == 0:
                label = r"$e^{\bot, 1}_\mathrm{o}$"
            elif i == 1:
                label = r"$e^{\bot, 2}_\mathrm{o}$"
            elif i == 2:
                label = r"$e^{||}_\mathrm{o}$"
            plt.subplot(2, 3, i+1)
            plt.bar(range(params.n), approx_error_n_min[:, i], width=0.8)
            plt.bar(range(params.n),
                    approx_error_n_mean[:, i] - approx_error_n_min[:, i],
                    width=0.8,
                    bottom=approx_error_n_min[:, i])
            plt.bar(range(params.n),
                    approx_error_n_max[:, i] - approx_error_n_mean[:, i],
                    width=0.8,
                    bottom=approx_error_n_mean[:, i])
            plt.title("Approximation error " + label)
            plt.xlabel("Horizon step")

            plt.subplot(2, 3, i+4)
            plt.bar(x_traj, approx_error_x_min[:, i], width=0.08)
            plt.bar(x_traj, approx_error_x_mean[:, i]-approx_error_x_min[:, i],
                    width=0.08,
                    bottom=approx_error_x_min[:, i])
            plt.bar(x_traj, approx_error_x_max[:, i]-approx_error_x_mean[:, i],
                    width=0.08,
                    bottom=approx_error_x_mean[:, i])
            plt.title("Approximation error " + label)
            plt.xlabel(x_label)
        plt.suptitle("Approximation error mean")

        # -----------------------------------------------------------------------
        # Plot orthogonal error decomposition in the basis vectors
        # -----------------------------------------------------------------------
        plt.figure()
        for i in range(2):
            plt.subplot(2, 2, i + 1)
            plt.plot(x_traj, e_p_off[:, i], 'k--', label="Error bound offset")
            plt.plot(x_traj, p_l[:, i], 'k', label="Error bound")
            plt.plot(x_traj, p_u[:, i], 'k', label="Error bound")
            plt.title("Position")
            plt.ylabel(f"Basis Direction {i}")
            # plt.xlabel("Time [s]")
            plt.xlabel(x_label)

            # Horizons
            for j in range(e_r_plane_horizon.shape[1]):
                t_horizon = (j+1)/10 + np.arange(0, logger.N/10, 0.1)
                x_horizon = phi_horizon[:, j] if x_values == 'path' else t_horizon
                plt.plot(x_horizon,
                         e_plane_horizon[:, j, i],
                         'C2', linewidth=0.5, label="_Hidden")
                plt.plot(x_horizon,
                         p_l_horizon[:, j, i],
                         'k',
                         linewidth=0.5, label="_Hidden")
                plt.plot(x_horizon,
                         p_u_horizon[:, j, i],
                         'k',
                         linewidth=0.5, label="_Hidden")
            plt.plot(x_traj, e_orth_plane[:, i], 'C0', label="Error")
            plt.legend()

            plt.subplot(2, 2, i + 3)
            plt.plot(x_traj,
                     e_r_off[:, i] * 180 / np.pi,
                     'k--',
                     label="Error bound offset")
            plt.plot(x_traj,
                     p_l[:, i + 2] * 180 / np.pi,
                     'k',
                     label="Error bound")
            plt.plot(x_traj,
                     p_u[:, i + 2] * 180 / np.pi,
                     'k',
                     label="Error bound")
            plt.title("Orientation")
            plt.ylabel(f"Basis Direction {i}")
            # plt.xlabel("Time [s]")
            plt.xlabel(x_label)

            # Horizons
            for j in range(e_r_plane_horizon.shape[1]):
                t_horizon = j/10 + np.arange(0, logger.N/10, 0.1)
                x_horizon = phi_horizon[:, j] if x_values == 'path' else t_horizon
                plt.plot(x_horizon,
                         e_r_plane_horizon[:, j, i] * 180 / np.pi,
                         'C2', linewidth=0.5, label="_Hidden")
                plt.plot(x_horizon,
                         p_l_horizon[:, j, i + 2] * 180 / np.pi,
                         'k',
                         linewidth=0.5, label="_Hidden")
                plt.plot(x_horizon,
                         p_u_horizon[:, j, i + 2] * 180 / np.pi,
                         'k',
                         linewidth=0.5, label="_Hidden")
            plt.plot(x_traj, e_r_orths_true[:, i], 'C1', label="True")
            plt.plot(x_traj,
                     e_r_orth_plane[:, i] * 180 / np.pi,
                     'C0',
                     label="Error")
            plt.legend()
        plt.legend()
        plt.suptitle("Orthogonal error plane")

        # -----------------------------------------------------------------------
        # Plot computation time
        # -----------------------------------------------------------------------
        plt.figure()
        t_comp = np.array(t_comp)
        print("Computation time stats:")
        print("--------------")
        print(f"Min: {np.min(t_comp):.4f}s")
        print(f"Max: {np.max(t_comp):.4f}s")
        print(f"Avg: {np.mean(t_comp):.4f}s")
        t_loops = np.array(t_loops)
        t_overhead = np.array(t_overhead)
        plt.subplot(2, 1, 1)
        plt.semilogy(x_traj,
                     t_comp,
                     label='MPC')
        plt.semilogy(x_traj,
                     t_loops,
                     label='Loop')
        plt.semilogy(x_traj,
                     t_overhead,
                     label='Overhead')
        plt.hlines(0.1, x_traj[0], x_traj[-1], linestyles='dashed')
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel("Computation Time [s]")
        plt.title("Computation Time")
        plt.subplot(2, 1, 2)
        plt.plot(x_traj,
                 iterations,
                 label='MPC')
        plt.ylim([0, 40])
        plt.xlabel(x_label)
        plt.ylabel("Iterations [-]")
        plt.title("Iterations")

        # -----------------------------------------------------------------------
        # Plot errors
        # -----------------------------------------------------------------------
        plt.figure()
        plt.subplot(231)
        plt.plot(x_traj, e_par_length, label="Position")
        plt.xlabel(x_label)
        plt.title("Position Parallel Error")
        plt.subplot(232)
        plt.plot(x_traj, de_orth_norm, label="Computed")
        plt.plot(x_traj[1:], de_orth_norm_num, label="Numerical")
        plt.legend()
        plt.xlabel(x_label)
        plt.title("Position Orthogonal Error Deriv")
        plt.subplot(233)
        plt.plot(x_traj, de_par_norm, label="Computed")
        plt.plot(x_traj[1:], de_par_norm_num, label="Numerical")
        plt.legend()
        plt.xlabel(x_label)
        plt.title("Position Parallel Error Deriv")
        plt.subplot(234)
        plt.plot(x_traj, r_par_bound * 180 / np.pi, 'k', label="Bound")
        plt.plot(x_traj, -r_par_bound * 180 / np.pi, 'k', label="Bound")
        # Horizons
        for j in range(e_r_plane_horizon.shape[1]):
            t_horizon = j/10 + np.arange(0, logger.N/10, 0.1)
            x_horizon = phi_horizon[:, j] if x_values == 'path' else t_horizon
            plt.plot(x_horizon,
                     e_r_plane_horizon[:, j, 2] * 180 / np.pi,
                     'C2', linewidth=0.5, label="_Hidden")
            plt.plot(x_horizon,
                     r_par_bound_horizon[:, j] * 180 / np.pi,
                     'k',
                     linewidth=0.5, label="_Hidden")
            plt.plot(x_horizon,
                     -r_par_bound_horizon[:, j] * 180 / np.pi,
                     'k',
                     linewidth=0.5, label="_Hidden")
        plt.plot(x_traj, e_r_par_true_length * 180 / np.pi, 'C1', label="True")
        plt.plot(x_traj, e_r_par_length * 180 / np.pi, 'C0', label="Approx")
        plt.legend()
        plt.xlabel(x_label)
        plt.title("Rotation Parallel Error")
        plt.subplot(236)
        plt.plot(x_traj, np.linalg.norm(de_r, axis=1), label="Computed")
        plt.plot(x_traj[1:], de_r_norm_num, label="Numerical")
        plt.xlabel(x_label)
        plt.title("Rotation Error Deriv")
        plt.legend()

        # -----------------------------------------------------------------------
        # Plot cartesian path
        # -----------------------------------------------------------------------
        plt.figure()
        plt.subplot(1, 1, 1, projection='3d')
        plt.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2], label="MPC")
        plt.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], label="Reference")
        plt.title("Cartesian Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        # -----------------------------------------------------------------------
        # Plot joint trajectories
        # -----------------------------------------------------------------------
        plt.figure()
        for i in range(7):
            plt.subplot(4, 7, i + 1)
            plt.plot(x_traj, q_traj[:, i] * 180 / np.pi, label="MPC")
            plt.hlines(q_lim_lower[i] * 180 / np.pi,
                       x_traj[0],
                       x_traj[-1],
                       linestyles='dashed')
            plt.hlines(q_lim_upper[i] * 180 / np.pi,
                       x_traj[0],
                       x_traj[-1],
                       linestyles='dashed')
            plt.xlabel(x_label)
            plt.title(f"Joint {i+1}")
            plt.ylabel(r"Angle [$\circ$]")
            plt.legend()
            plt.subplot(4, 7, i + 8)
            plt.plot(x_traj, dq_traj[:, i] * 180 / np.pi)
            plt.hlines(dq_lim_lower[i] * 180 / np.pi,
                       x_traj[0],
                       x_traj[-1],
                       linestyles='dashed')
            plt.hlines(dq_lim_upper[i] * 180 / np.pi,
                       x_traj[0],
                       x_traj[-1],
                       linestyles='dashed')
            plt.xlabel(x_label)
            plt.ylabel(r"Velocity [$\circ/s$]")
            plt.subplot(4, 7, i + 15)
            plt.plot(x_traj, ddq_traj[:, i] * 180 / np.pi)
            plt.ylabel(r"Acceleration [$\circ/s^2$]")
            plt.subplot(4, 7, i + 22)
            plt.xlabel(x_label)
            plt.plot(x_traj, j_traj[:, i] * 180 / np.pi, label="$u$")
            plt.hlines(u_max * 180 / np.pi,
                       x_traj[0],
                       x_traj[-1],
                       linestyles='dashed')
            plt.hlines(u_min * 180 / np.pi,
                       x_traj[0],
                       x_traj[-1],
                       linestyles='dashed')
            plt.xlabel(x_label)
            plt.ylabel(r"Jerk [$\circ/s^3$]")
        plt.suptitle(r"$q(t)$")

        # -----------------------------------------------------------------------
        # Plot position trajectories
        # -----------------------------------------------------------------------
        plt.figure()
        for i in range(3):
            plt.subplot(4, 3, i + 1)
            plt.plot(x_traj, p_traj[:, i], label="MPC")
            plt.plot(x_traj, p_ref[:, i], label="Reference")
            plt.xlabel(x_label)
            plt.title("Position")
            plt.legend()
            plt.subplot(4, 3, i + 4)
            plt.plot(x_traj, v_traj[:, i])
            plt.xlabel(x_label)
            plt.title("Velocity")
            plt.subplot(4, 3, i + 7)
            plt.plot(x_traj, a_traj[:, i])
            plt.xlabel(x_label)
            plt.title("Acceleration")
            plt.subplot(4, 3, i + 10)
            plt.plot(x_traj, j_cart_traj[:, i])
            plt.xlabel(x_label)
            plt.title("Jerk")
        plt.suptitle(r"$p(t)$")

        # -----------------------------------------------------------------------
        # Plot rotation trajectories
        # -----------------------------------------------------------------------
        angle_names = ["$\\alpha$", "$\\beta$", "$\\gamma$"]
        plt.figure()
        for i in range(3):
            plt.subplot(4, 3, i + 1)
            plt.plot(x_traj, p_traj[:, i + 3], label="MPC")
            plt.plot(x_traj, p_ref[:, i + 3], label="Reference")
            plt.xlabel(x_label)
            plt.title(angle_names[i])
            plt.legend()
            plt.subplot(4, 3, i + 4)
            plt.plot(x_traj, v_traj[:, i + 3])
            plt.title("Velocity")
            plt.subplot(4, 3, i + 7)
            plt.plot(x_traj, a_traj[:, i + 3])
            plt.xlabel(x_label)
            plt.title("Acceleration")
            plt.subplot(4, 3, i + 10)
            plt.plot(x_traj, j_cart_traj[:, i + 3])
            plt.xlabel(x_label)
            plt.title("Jerk")
        plt.suptitle(r"$r(t)$")

        # -----------------------------------------------------------------------
        # Plot rotation trajectories
        # -----------------------------------------------------------------------
        plt.figure()
        plt.subplot(1, 1, 1, projection='3d')
        rx = p_traj[3:, 3]
        ry = p_traj[3:, 4]
        rz = p_traj[3:, 5]
        rx_ref = p_ref[:, 3]
        ry_ref = p_ref[:, 4]
        rz_ref = p_ref[:, 5]
        plt.plot(rx, ry, rz, label="MPC")
        plt.plot(rx_ref, ry_ref, rz_ref, label="Reference")
        plt.xlabel(r"$r_x$")
        plt.ylabel(r"$r_y$")
        ax = plt.gca()
        ax.set_zlabel(r"$r_z$")
        plt.title("Rotation vector trajectory")

        # -----------------------------------------------------------------------
        # Plot rotation matrix
        # -----------------------------------------------------------------------
        # pr_real = np.empty((p_lie_traj_real.shape[0], pr_traj.shape[1]))
        # for i in range(p_lie_traj_real.shape[0]):
        # pr_real[i, :] = R.from_rotvec(p_lie_traj_real[i, 3:]).as_matrix().flatten()
        # plt.figure()
        # for i in range(9):
        # plt.subplot(3, 3, i+1)
        # plt.plot(t_traj, pr_traj[:, i], label="MPC")
        # plt.plot(t_traj, p_r_ref[:, i], label="Reference")
        # plt.plot(t_traj_real, pr_real[:, i], label="Real")
        # plt.xlabel("Time [s]")
        # plt.legend()
        # plt.suptitle("Rotation Matrix")

        # -----------------------------------------------------------------------
        # Plot path parameter trajectory
        # -----------------------------------------------------------------------
        plt.figure()
        plt.subplot(411)
        plt.plot(t_traj, phi_traj)
        # plt.hlines(phi_max, t_traj[0], t_traj[-1], linestyles='dashed')
        plt.title(r"$\theta$")
        plt.xlabel("Time [s]")
        plt.subplot(412)
        plt.plot(t_traj, dphi_traj)
        plt.title(r"$\dot{\theta}$")
        plt.xlabel("Time [s]")
        plt.subplot(413)
        plt.plot(t_traj, ddphi_traj)
        plt.title(r"$\ddot{\theta}$")
        plt.xlabel("Time [s]")
        plt.subplot(414)
        plt.plot(t_traj, dddphi_traj)
        plt.title(r"$\ddot{\theta}$")
        plt.xlabel("Time [s]")

        # -----------------------------------------------------------------------
        # Plot Jacobi Determinant
        # -----------------------------------------------------------------------
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.semilogy(x_traj, ellipsoid_volume, label="Ellipsoid Volume")
        plt.xlabel(x_label)
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.semilogy(x_traj, condition_number, label="Condition Number > 1")
        plt.xlabel(x_label)
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(x_traj, manip_measure, label="Manipulability measure")
        plt.xlabel(x_label)
        plt.legend()
        plt.suptitle("Manipulability measures")

        if save_data:
            # -----------------------------------------------------------------------
            # Saving data
            # -----------------------------------------------------------------------

            # Compute error bounds in global coordinates
            corners_3d = np.empty((bp1.shape[0], 12))
            # corners_3dr = np.empty((bp1.shape[0], 12))
            for i in range(p_l.shape[0]):
                # Compute the 4 corners of the bounding rectangle
                corners_3d[i, :3] = p_l[i, 0] * bp1[i, :] + p_l[i, 1] * bp2[i, :]
                corners_3d[i, 3:6] = p_l[i, 0] * bp1[i, :] + p_u[i, 1] * bp2[i, :]
                corners_3d[i, 6:9] = p_u[i, 0] * bp1[i, :] + p_l[i, 1] * bp2[i, :]
                corners_3d[i, 9:] = p_u[i, 0] * bp1[i, :] + p_u[i, 1] * bp2[i, :]

            # Project the errors onto the cartesian planes
            pidx = [0, 1]
            p_l_xy, p_u_xy = project_position_bounds(corners_3d, p_ref, dp_ref[:, :3], pidx, False)
            pidx = [0, 2]
            p_l_xz, p_u_xz = project_position_bounds(corners_3d, p_ref, dp_ref[:, :3], pidx, False)
            pidx = [1, 2]
            p_l_zy, p_u_zy = project_position_bounds(corners_3d, p_ref, dp_ref[:, :3], pidx, False)

            # Save data
            save_data = {}
            save_data["t"] = t_traj
            save_data["phi"] = phi_traj
            save_data["dphi"] = dphi_traj
            save_data["ddphi"] = ddphi_traj
            save_data["dddphi"] = dddphi_traj
            save_data["p"] = p_traj
            save_data["v"] = v_traj
            save_data["a"] = a_traj
            save_data["j"] = j_cart_traj
            save_data["q"] = q_traj
            save_data["dq"] = dq_traj
            save_data["ddq"] = ddq_traj
            save_data["dddq"] = j_traj

            save_data["p_ref"] = p_ref
            save_data["dp_ref"] = dp_ref
            save_data["p_via"] = p_via
            save_data["r_via"] = r_via

            save_data["bp1"] = bp1
            save_data["bp2"] = bp2
            save_data["br1"] = br1
            save_data["br2"] = br2
            save_data["bounds_lower"] = p_l
            save_data["bounds_upper"] = p_u
            save_data["bounds_r_par"] = r_par_bound

            save_data["e_par"] = e_par
            save_data["e_orth"] = e_orth
            save_data["e_orth_plane"] = e_orth_plane
            save_data["e_p_off"] = e_p_off

            save_data["e_r_par"] = e_r_par
            save_data["e_r_off"] = e_r_off
            save_data["e_r_orth_plane"] = e_r_orth_plane
            save_data["e_r_par_true"] = e_r_par_true
            save_data["e_r_orths_true"] = e_r_orths_true
            save_data["e_r_par_length"] = e_r_par_length
            save_data["e_r_par_true_length"] = e_r_par_true_length
            save_data["approx_error"] = approx_error

            save_data["bound_pl_proj_xy"] = p_l_xy
            save_data["bound_pu_proj_xy"] = p_u_xy
            save_data["bound_pl_proj_xz"] = p_l_xz
            save_data["bound_pu_proj_xz"] = p_u_xz
            save_data["bound_pl_proj_zy"] = p_l_zy
            save_data["bound_pu_proj_zy"] = p_u_zy

            np.savez(path + f'data_{tail}.npz', save_data)
