import casadi as ca
import numpy as np
from .jerk_trajectory_casadi import calcAngle, calcVelocity, calcAcceleration
from ..utils.lie_functions import rodrigues_matrix
from ..RobotModel import RobotModel
from .mpc_utils_casadi import (integrate_rot_error_diff,
                               compute_rot_error_velocity,
                               compute_position_error,
                               compute_fourth_order_error_bound_general,
                               decompose_orthogonal_error)


def get_current_segment(phi, phi_switch, array):
    """ Get the current entry of the array based on the value of phi if
    phi_switch gives the switching values between the entries.
    """
    result = array[-1, :]
    for i in reversed(range(array.shape[0] - 1)):
        result = ca.if_else(phi < phi_switch[i+1], array[i, :], result)
    return result


def get_current_and_next_segment_1d(phi, phi_switch, array):
    """ Get the current and the next entry of the array based on the value of
    phi if phi_switch gives the switching values between the entries. """
    result0 = array[-2]
    result1 = array[-1]
    for i in reversed(range(len(array) - 2)):
        result0 = ca.if_else(phi < phi_switch[i+1], array[i], result0)
        result1 = ca.if_else(phi < phi_switch[i+1], array[i+1], result1)
    return result0, result1


def get_current_and_next_segment(phi, phi_switch, array):
    """ Get the current and the next entry of the array based on the value of
    phi if phi_switch gives the switching values between the entries. """
    result = array[-2:, :]
    for i in reversed(range(array.shape[0] - 2)):
        result = ca.if_else(phi < phi_switch[i+1], array[i:i+2, :], result)
    return result[0, :], result[1, :]


def reference_function(dp_ref,
                       p_ref,
                       phi_switch,
                       phi,
                       phi_prev,
                       bp1,
                       bp2,
                       br1,
                       br2,
                       v1,
                       v2,
                       v3,
                       dp_normed_ref,
                       a4,
                       a3,
                       a2,
                       a1,
                       a0,
                       nr_segs,
                       length=0.05):
    if isinstance(dp_ref, np.ndarray):
        p_e_bound = np.zeros((4, 1))
        r_e_bound = np.zeros((4, 1))
        p_d = np.zeros((6, 1))
        r0 = np.zeros((2, 2))
        r1 = np.zeros((2, 2))
        ddp_lin = np.zeros((3, 1))
    else:
        p_e_bound = ca.SX.zeros(4)
        r_e_bound = ca.SX.zeros(4)
        p_d = ca.SX.zeros((6, 1))
        r0 = ca.SX.zeros((2, 2))
        r1 = ca.SX.zeros((2, 2))
        ddp_lin = ca.SX.zeros((3, 1))
    dp_d = get_current_segment(phi, phi_switch, dp_ref)
    phi_start, phi_end = get_current_and_next_segment(phi, phi_switch, phi_switch)
    p_ref_current = get_current_segment(phi, phi_switch, p_ref)
    p_d[:3] = dp_d[:3] * (phi - phi_start) + p_ref_current[:3]
    p_d[3:] = dp_d[3:] * (phi - phi_start) + p_ref_current[3:]

    phi0 = phi_start
    a4c = get_current_segment(phi, phi_switch, a4)
    a3c = get_current_segment(phi, phi_switch, a3)
    a2c = get_current_segment(phi, phi_switch, a2)
    a1c = get_current_segment(phi, phi_switch, a1)
    a0c = get_current_segment(phi, phi_switch, a0)
    for i in range(2):
        p_e_bound[i] = compute_fourth_order_error_bound_general(
            phi - phi0, a4c[i], a3c[i], a2c[i], a1c[i], a0c[i])
        r_e_bound[i] = compute_fourth_order_error_bound_general(
            phi - phi0, a4c[i + 4], a3c[i + 4], a2c[i + 4], a1c[i + 4],
            a0c[i + 4])
    for i in range(2):
        p_e_bound[i + 2] = compute_fourth_order_error_bound_general(
            phi - phi0, a4c[i + 2], a3c[i + 2], a2c[i + 2], a1c[i + 2],
            a0c[i + 2])
        r_e_bound[i + 2] = compute_fourth_order_error_bound_general(
            phi - phi0, a4c[i + 6], a3c[i + 6], a2c[i + 6], a1c[i + 6],
            a0c[i + 6])
    r_par_bound = compute_fourth_order_error_bound_general(
        phi - phi0, a4c[8], a3c[8], a2c[8], a1c[8], a0c[8])

    bound_lower = ca.vertcat(p_e_bound[2:], r_e_bound[2:])
    bound_upper = ca.vertcat(p_e_bound[:2], r_e_bound[:2])
    e_p_off = 0.5 * (p_e_bound[:2] + p_e_bound[2:])
    e_r_off = 0.5 * (r_e_bound[:2] + r_e_bound[2:])
    bp10, bp11 = get_current_and_next_segment(phi, phi_switch, bp1)
    bp20, bp21 = get_current_and_next_segment(phi, phi_switch, bp2)
    bp10 = bp10.T
    bp20 = bp20.T
    bp11 = bp11.T
    bp21 = bp21.T
    br10 = get_current_segment(phi, phi_switch, br1).T
    br20 = get_current_segment(phi, phi_switch, br2).T
    dp_normed_d = get_current_segment(phi, phi_switch, dp_normed_ref)
    v10 = get_current_segment(phi, phi_switch, v1).T
    v20 = get_current_segment(phi, phi_switch, v2).T
    v30 = get_current_segment(phi, phi_switch, v3).T

    outputs = [
        p_d,
        dp_d,
        0*dp_d,
        bp10,
        bp20,
        br10,
        br20,
        dp_normed_d,
        v10,
        v20,
        v30,
        bound_lower,
        bound_upper,
        r_par_bound,
        e_p_off,
        e_r_off,
    ]
    output_names = [
        'p_d', 'dp_d', 'ddp_d', 'bp1_current', 'bp2_current',
        'br1_current', 'br2_current', 'dp_normed_d', 'v1_current',
        'v2_current', 'v3_current', 'bound_lower',
        'bound_upper', 'r_par_bound', 'e_p_off', 'e_r_off'
    ]
    reference_data = {}
    for key, value in zip(output_names, outputs):
        reference_data[key] = value
    return reference_data


def error_function(p, v, p_ref, dp_ref, dp_normed_ref, dphi, i_omega_0,
                   i_omega_ref_0, dtau_init, dtau_init_par, dtau_init_orth1,
                   dtau_init_orth2, br1, br2, jac_dtau_l, jac_dtau_r, phi,
                   phi_switch, v1, v2, v3, nr_segs):
    # Compute position errors
    e_p_par, e_p_orth, de_p_par, de_p_orth, e_p, de_p = compute_position_error(
                                                    p[:3],
                                                    v[:3],
                                                    p_ref[:3],
                                                    dp_ref[:3],
                                                    0*dp_ref[:3].T,
                                                    dphi)

    # Compute orientation error and its derivative
    e_r = integrate_rot_error_diff(dtau_init, p[3:], i_omega_0, p_ref[3:],
                                   i_omega_ref_0, jac_dtau_l, jac_dtau_r)
    de_r = compute_rot_error_velocity(dp_ref[3:], v[3:],
                                      jac_dtau_l, jac_dtau_r, dphi)

    # Compute correct starting value based on the segment for the orientation
    # error
    e_par_init = get_current_segment(phi, phi_switch, dtau_init_par).T
    e_orth1_init = get_current_segment(phi, phi_switch, dtau_init_orth1).T
    e_orth2_init = get_current_segment(phi, phi_switch, dtau_init_orth2).T
    if isinstance(dtau_init_par, np.ndarray):
        e_par_init = e_par_init.T
        e_orth1_init = e_orth1_init.T
        e_orth2_init = e_orth2_init.T

    # Project error onto path
    scal_orth1 = ca.dot(e_r - dtau_init, v1)
    scal_par = ca.dot(e_r - dtau_init, v2)
    scal_orth2 = ca.dot(e_r - dtau_init, v3)

    # Final decomposed orientation errors
    e_r_orth1 = e_orth1_init + scal_orth1 * br1
    e_r_par = e_par_init + scal_par * dp_normed_ref
    e_r_orth2 = e_orth2_init + scal_orth2 * br2

    outputs = [
        e_p_par, e_p_orth, de_p_par, de_p_orth, e_p, de_p, e_r_par, e_r, de_r,
        e_r_orth1, e_r_orth2
    ]
    output_names = [
        'e_p_par', 'e_p_orth', 'de_p_par', 'de_p_orth', 'e_p', 'de_p',
        'e_r_par', 'e_r', 'de_r', 'e_r_orth1', 'e_r_orth2'
    ]
    error_data = {}
    for key, value in zip(output_names, outputs):
        error_data[key] = value
    return error_data


def objective_function(nr_joints, nr_u, x_phi, e_p_par, de_p, e_r_par, de_r,
                       x_phi_d, v, v_ref, a, a_ref, u, q, dq, ddq, qd, weights):
    """ Create the objective function for the MPC.
    """
    # Extract weights
    w_p = weights[0]
    w_r = weights[1]
    w_v = weights[2]
    # dw_r = weights[3]
    # w_speed = weights[4]
    w_a = weights[5]
    w_phi = weights[6]
    w_dphi = weights[7]
    w_ddphi = weights[8]
    w_dddphi = weights[9]
    w_q = weights[10]
    w_dq = weights[11]
    w_ddq = weights[12]
    w_jerk = weights[13]

    # Create objective term
    objective_term = 0
    objective_term += w_r * ca.sumsqr(e_r_par)
    objective_term += w_p * ca.sumsqr(e_p_par)

    # Cartesian velocity and acceleration
    objective_term += w_v * ca.sumsqr(v - v_ref)
    objective_term += w_a * ca.sumsqr(a - a_ref)

    # Joint state
    objective_term += w_q * ca.sumsqr(q - qd)
    objective_term += w_dq * ca.sumsqr(dq)
    objective_term += w_ddq * ca.sumsqr(ddq)
    objective_term += w_jerk * ca.sumsqr(u[:-1])

    # Path state
    objective_term += w_phi * ca.sumsqr(x_phi_d[0] - x_phi[0])
    objective_term += w_dphi * ca.sumsqr(x_phi_d[1] - x_phi[1])
    objective_term += w_ddphi * ca.sumsqr(x_phi_d[2] - x_phi[2])
    objective_term += w_dddphi * ca.sumsqr(u[-1])

    return objective_term


def integration_function(nr_joints, nr_u, dt, q, dq, ddq, p_rot, phi, dphi,
                         ddphi, u, u_prev):
    """
    Create the function to integrate the state of dx = f(x, u).
    """
    jerk_matrix = ca.vertcat(u_prev.T, u.T).T
    u_prevn = u

    # Integrate the state using triangle functions
    qn = calcAngle(jerk_matrix[:nr_joints, :], dt, q, dq, ddq, dt)
    dqn = calcVelocity(jerk_matrix[:nr_joints, :], dt, dq, ddq, dt)
    ddqn = calcAcceleration(jerk_matrix[:nr_joints, :], dt, ddq, dt)

    robot_model = RobotModel()
    pn_pos = robot_model.fk_pos(qn)
    v_cart = robot_model.velocity_ee(qn, dqn)
    v_rot = robot_model.omega_ee(qn, dqn)
    vn = ca.vertcat(v_cart, v_rot)

    # RK4 integration of omega
    # pn_rot = p_rot + dt * omega_ee(q, dq)
    # q_mid = calcAngle(jerk_matrix[:, :nr_joints], dt/2, q, dq, ddq, dt)
    # dq_mid = calcVelocity(jerk_matrix[:, :nr_joints], dt/2, dq, ddq, dt)
    # k1 = omega_ee(q, dq)
    # k2 = omega_ee(q_mid, dq_mid)
    # k4 = omega_ee(qn, dqn)
    # pn_rot = p_rot + 1/6 * dt * (k1 + 4*k2 + k4)

    # Trapezoidal integration of omega
    k1 = robot_model.omega_ee(q, dq)
    k2 = v_rot
    pn_rot = p_rot + 1/2 * dt * (k1 + k2)

    pn = ca.vertcat(pn_pos, pn_rot)
    phin = calcAngle(jerk_matrix[-1, :], dt, phi, dphi, ddphi, dt)
    dphin = calcVelocity(jerk_matrix[-1, :], dt, dphi, ddphi, dt)
    ddphin = calcAcceleration(jerk_matrix[-1, :], dt, ddphi, dt)

    outputs = [qn, dqn, ddqn, pn, vn, phin, dphin, ddphin, u_prevn]
    output_names = [
        'q_new', 'dq_new', 'ddq_new', 'p_new', 'v_new', 'phi_new', 'dphi_new',
        'ddphi_new', 'u_prev_new'
    ]
    int_data = {}
    for key, value in zip(output_names, outputs):
        int_data[key] = value
    return int_data


def decomp_function(e, e_off, b1, b2, p_lower, p_upper):
    e_plane = decompose_orthogonal_error(e, b1, b2)
    e_diff = e_plane - e_off
    bound = (p_upper - p_lower)/2
    constraint0 = ca.sumsqr(e_diff[0]) - ca.sumsqr(bound[0])
    constraint1 = ca.sumsqr(e_diff[1]) - ca.sumsqr(bound[1])

    outputs = [constraint0, constraint1]
    output_names = ['constraint0', 'constraint1']
    decomp_data = {}
    for key, value in zip(output_names, outputs):
        decomp_data[key] = value
    return decomp_data
